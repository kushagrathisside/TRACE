"""
Data ingestion pipeline: Semantic Scholar → LangChain Documents → ChromaDB.

Incremental sync
----------------
sync_status["per_person"][s2_id]["last_year"] records the most recent
publication year seen for each author.  On subsequent syncs we pass
`since_year = last_year` to the Semantic Scholar API so only new papers
are fetched.  New people (not in per_person) always get a full fetch.

Orphan cleanup is NOT done here — it is triggered explicitly when a person
is removed via the DELETE /api/people/{id} endpoint.

Status persistence
------------------
sync_status is written to data/sync_status.json after each sync so the
admin panel shows correct numbers after a server restart.
"""

import json
import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import config
import portalocker
from langchain_core.documents import Document
from pydantic import ValidationError
from rag.schemas import DocumentMetadata
from rag.vector_store import VectorStoreManager

from ingestion import people_registry, scholar_client

_STATUS_FILE = Path(config.SYNC_STATUS_PATH)
_LOCK_FILE = _STATUS_FILE.with_suffix(".lock")
logger = logging.getLogger(__name__)


# ── In-memory status (also persisted to disk) ─────────────────────────────────
def _default_status() -> dict:
    return {
        "status": "idle",
        "papers_indexed": 0,
        "last_sync": None,
        "errors": [],
        "current_person": None,
        "per_person": {},  # s2_id -> {"last_year": int, "count": int}
    }


def _load_status() -> dict:
    if _STATUS_FILE.exists():
        try:
            return json.loads(_STATUS_FILE.read_text())
        except Exception:
            pass
    return _default_status()


def _save_status(status: dict) -> None:
    _STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=_STATUS_FILE.parent, delete=False, suffix=".json"
    ) as tmp:
        json.dump(status, tmp, indent=2)
        tmp_path = Path(tmp.name)
    tmp_path.replace(_STATUS_FILE)


sync_status: dict = _load_status()


def _existing_institute_authors(vs, paper_ids: list[str]) -> dict[str, list[str]]:
    """
    institute_authors already stored for these papers, keyed by paper_id.

    Used to merge rather than overwrite on incremental syncs — see the call
    site.  Failure is non-fatal: a lookup error means we fall back to the
    previous overwrite behaviour rather than aborting the sync.
    """
    if not paper_ids:
        return {}
    try:
        result = vs._collection.get(ids=paper_ids, include=["metadatas"])
    except Exception as exc:
        logger.warning(f"Could not read existing metadata (non-fatal): {exc}")
        return {}

    out: dict[str, list[str]] = {}
    for meta in result.get("metadatas") or []:
        pid = (meta or {}).get("paper_id", "")
        names = [
            n.strip() for n in (meta or {}).get("institute_authors", "").split(",")
        ]
        if pid:
            out[pid] = [n for n in names if n]
    return out


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run_ingestion() -> None:
    try:
        with portalocker.Lock(str(_LOCK_FILE), timeout=1, flags=portalocker.LOCK_EX):
            _run_ingestion_internal()
    except portalocker.exceptions.LockException:
        logger.warning("Sync already running in another process.")


def _run_ingestion_internal() -> None:
    global sync_status

    sync_status.update(
        {
            "status": "running",
            "errors": [],
            "current_person": None,
        }
    )

    vs = VectorStoreManager.get_or_create()
    people = people_registry.get_all()

    s2_to_person: dict[str, dict] = {
        p["semantic_scholar_id"]: p for p in people if p.get("semantic_scholar_id")
    }

    # paper_id → {"paper": ..., "institute_authors": [...]}
    paper_map: dict[str, dict] = {}
    per_person_new: dict[str, dict] = {}

    for s2_id, person in s2_to_person.items():
        sync_status["current_person"] = person["name"]
        _save_status(sync_status)

        # Incremental: only fetch papers newer than last known year
        prev = sync_status.get("per_person", {}).get(s2_id, {})
        since_year: int | None = prev.get("last_year")

        try:
            papers = scholar_client.get_author_papers(s2_id, since_year=since_year)
            max_year = since_year or 0
            for paper in papers:
                pid = paper.get("paperId")
                if not pid:
                    continue
                if pid not in paper_map:
                    paper_map[pid] = {"paper": paper, "institute_authors": []}
                paper_map[pid]["institute_authors"].append(person)
                yr = paper.get("year")
                if yr and yr > max_year:
                    max_year = yr

            per_person_new[s2_id] = {
                "last_year": max_year,
                "count": len(papers),
            }
            time.sleep(1)  # stay within rate limit
        except Exception as exc:
            sync_status["errors"].append(f"{person['name']}: {exc}")

    # ── Merge with already-indexed institute authors ──────────────────────────
    # An incremental sync only refetches papers newer than each person's
    # last_year, so a paper co-authored by A and B may come back under A alone.
    # Upserting that would overwrite institute_authors with just "A" and drop B
    # from the paper permanently — degrading people_to_consult a little more on
    # every sync.  Existing names are merged back in before the upsert.
    existing_meta = _existing_institute_authors(vs, list(paper_map))
    name_to_person = {p["name"]: p for p in people}

    # ── Build and upsert Documents ────────────────────────────────────────────
    docs: list[Document] = []
    doc_ids: list[str] = []

    for pid, entry in paper_map.items():
        paper = entry["paper"]
        inst_authors = list(entry["institute_authors"])

        known = {p["name"] for p in inst_authors}
        for name in existing_meta.get(pid, []):
            # Only re-add people still in the registry: someone removed via
            # DELETE /api/people must not be resurrected by the next sync.
            if name not in known and name in name_to_person:
                inst_authors.append(name_to_person[name])
                known.add(name)

        title = (paper.get("title") or "").strip()
        abstract = (paper.get("abstract") or "").strip()
        if not title:
            continue

        content = (
            f"Title: {title}\n\nAbstract: {abstract}" if abstract else f"Title: {title}"
        )

        all_authors = ", ".join(a.get("name", "") for a in (paper.get("authors") or []))
        inst_names = ", ".join(p["name"] for p in inst_authors)
        inst_roles = ", ".join(sorted({p["role"] for p in inst_authors}))
        departments = ", ".join(
            sorted({p["department"] for p in inst_authors if p.get("department")})
        )

        url = ""
        if paper.get("openAccessPdf"):
            url = (paper["openAccessPdf"] or {}).get("url", "")
        if not url:
            doi = (paper.get("externalIds") or {}).get("DOI", "")
            url = f"https://doi.org/{doi}" if doi else ""

        try:
            meta = DocumentMetadata(
                paper_id=pid,
                paper_title=title,
                year=paper.get("year") or 0,
                venue=paper.get("venue") or "",
                authors=all_authors,
                institute_authors=inst_names,
                institute_roles=inst_roles,
                departments=departments,
                paper_url=url,
            )
        except ValidationError as ve:
            sync_status["errors"].append(f"Metadata error for {pid}: {ve}")
            continue

        docs.append(
            Document(
                page_content=content,
                metadata=meta.model_dump(),
            )
        )
        doc_ids.append(pid)

    if docs:
        vs.upsert_documents(docs, doc_ids)

    # Only update per_person history AFTER upsert succeeds
    # This ensures that if upsert fails/crashes, next sync will re-attempt these papers
    existing_pp = sync_status.get("per_person", {})
    existing_pp.update(per_person_new)

    # Check for degraded status
    error_count = len(sync_status["errors"])
    person_count = len(s2_to_person)
    final_status = "done"
    if person_count > 0 and error_count > (0.3 * person_count):
        final_status = "degraded"

    sync_status.update(
        {
            "status": final_status,
            "papers_indexed": vs.count(),  # total in DB (not just this batch)
            "last_sync": datetime.now(timezone.utc).isoformat(),
            "current_person": None,
            "per_person": existing_pp,
        }
    )
    _save_status(sync_status)
