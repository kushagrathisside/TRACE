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
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from langchain_core.documents import Document

from ingestion import scholar_client, people_registry
from rag.vector_store import VectorStoreManager
import config

_STATUS_FILE = Path(config.SYNC_STATUS_PATH)

# ── In-memory status (also persisted to disk) ─────────────────────────────────
def _default_status() -> dict:
    return {
        "status": "idle",
        "papers_indexed": 0,
        "last_sync": None,
        "errors": [],
        "current_person": None,
        "per_person": {},        # s2_id -> {"last_year": int, "count": int}
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
        mode='w',
        dir=_STATUS_FILE.parent,
        delete=False,
        suffix='.json'
    ) as tmp:
        json.dump(status, tmp, indent=2)
        tmp_path = Path(tmp.name)
    tmp_path.replace(_STATUS_FILE)

sync_status: dict = _load_status()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_ingestion() -> None:
    global sync_status

    sync_status.update({
        "status": "running",
        "errors": [],
        "current_person": None,
    })

    vs = VectorStoreManager.get_or_create()
    people = people_registry.get_all()

    s2_to_person: dict[str, dict] = {
        p["semantic_scholar_id"]: p
        for p in people
        if p.get("semantic_scholar_id")
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
            time.sleep(1)   # stay within rate limit
        except Exception as exc:
            sync_status["errors"].append(f"{person['name']}: {exc}")

    # ── Build and upsert Documents ────────────────────────────────────────────
    docs: list[Document] = []
    doc_ids: list[str] = []

    for pid, entry in paper_map.items():
        paper      = entry["paper"]
        inst_authors = entry["institute_authors"]

        title    = (paper.get("title") or "").strip()
        abstract = (paper.get("abstract") or "").strip()
        if not title:
            continue

        content = f"Title: {title}\n\nAbstract: {abstract}" if abstract else f"Title: {title}"

        all_authors = ", ".join(
            a.get("name", "") for a in (paper.get("authors") or [])
        )
        inst_names  = ", ".join(p["name"]       for p in inst_authors)
        inst_roles  = ", ".join(sorted({p["role"]       for p in inst_authors}))
        departments = ", ".join(sorted({p["department"] for p in inst_authors if p.get("department")}))

        url = ""
        if paper.get("openAccessPdf"):
            url = (paper["openAccessPdf"] or {}).get("url", "")
        if not url:
            doi = (paper.get("externalIds") or {}).get("DOI", "")
            url = f"https://doi.org/{doi}" if doi else ""

        docs.append(Document(
            page_content=content,
            metadata={
                "paper_title":       title,
                "paper_id":          pid,
                "paper_url":         url,
                "year":              paper.get("year") or 0,
                "venue":             paper.get("venue") or "",
                "authors":           all_authors,
                "institute_authors": inst_names,
                "institute_roles":   inst_roles,
                "departments":       departments,
            },
        ))
        doc_ids.append(pid)

    total_upserted = vs.upsert_documents(docs, doc_ids) if docs else 0

    # Only update per_person history AFTER upsert succeeds
    # This ensures that if upsert fails/crashes, next sync will re-attempt these papers
    existing_pp = sync_status.get("per_person", {})
    existing_pp.update(per_person_new)

    sync_status.update({
        "status":         "done",
        "papers_indexed": vs.count(),       # total in DB (not just this batch)
        "last_sync":      datetime.now(timezone.utc).isoformat(),
        "current_person": None,
        "per_person":     existing_pp,
    })
    _save_status(sync_status)
