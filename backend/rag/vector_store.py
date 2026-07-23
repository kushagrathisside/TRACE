"""
Persistent ChromaDB vector store with:
  - Tuned HNSW index (M=48, ef_construction=200, search_ef=150)
  - Cosine distance space (requires normalised embeddings — see LLMProvider)
  - Similarity search with scores for threshold filtering
  - Orphan cleanup when a person is removed
  - Singleton pattern so the embedding model is loaded exactly once

HNSW parameter rationale
-------------------------
  M=48    : bidirectional links per node (default 16).  Higher M → better
            recall, ~3× more index memory (~15 MB for 15 k papers).
  construction_ef=200 : candidate list during index build.  Larger → more
            accurate neighbourhood graph, slower build (one-time cost).
  search_ef=150 : candidate list during query.  Default is 10 (!).  Raising
            to 150 recovers ~5% more true nearest neighbours at <2 ms overhead
            per query at 15 k scale.

NOTE: HNSW settings are frozen at collection creation time.  If you change
them after the collection exists, delete data/chroma_db/ and re-sync.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import config
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from llm_provider import LLMProvider

logger = logging.getLogger(__name__)


def _check_embedding_model() -> None:
    """
    Guard against silent retrieval corruption caused by changing the embedding
    model after the vector DB has already been populated.

    The first time the DB is created we write the model name to
    data/chroma_meta.json.  On every subsequent start we compare the stored
    name to the current config value and raise immediately if they differ,
    instead of letting the mismatch silently degrade retrieval quality.

    Fix when triggered: delete data/chroma_db/ and data/chroma_meta.json,
    then run a sync.  This forces a full re-embed with the new model.
    """
    meta_path = Path(config.CHROMA_META_PATH)
    current = config.EMBEDDING_MODEL_NAME

    if meta_path.exists():
        try:
            stored = json.loads(meta_path.read_text()).get("embedding_model", "")
        except Exception:
            stored = ""
        if stored and stored != current:
            raise RuntimeError(
                f"\n\n{'=' * 60}\n"
                f"EMBEDDING MODEL MISMATCH — retrieval will be WRONG\n"
                f"{'=' * 60}\n"
                f"  DB was built with : '{stored}'\n"
                f"  Config now says   : '{current}'\n\n"
                f"  Fix:\n"
                f"    1. rm -rf backend/data/chroma_db backend/data/chroma_meta.json\n"
                f"    2. Restart the server and run a sync\n"
                f"{'=' * 60}\n"
            )
    else:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(
                {
                    "embedding_model": current,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        )


class VectorStoreManager:
    _instance: "VectorStoreManager | None" = None

    # ── Singleton ────────────────────────────────────────────────────────────

    @classmethod
    def get_or_create(cls) -> "VectorStoreManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    # ── Init ─────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        _check_embedding_model()
        self.embeddings = LLMProvider.get_embeddings()
        self._client = chromadb.PersistentClient(
            path=config.VECTOR_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="institute_research",
            metadata={
                "hnsw:space": config.HNSW_SPACE,
                "hnsw:M": config.HNSW_M,
                "hnsw:construction_ef": config.HNSW_CONSTRUCTION_EF,
                "hnsw:search_ef": config.HNSW_SEARCH_EF,
            },
        )
        # LangChain wrapper that re-uses the same client (no second connection)
        self._store = Chroma(
            client=self._client,
            collection_name="institute_research",
            embedding_function=self.embeddings,
        )
        logger.info(
            f"VectorStore ready — {self._collection.count()} documents, "
            f"HNSW M={config.HNSW_M} ef_construction={config.HNSW_CONSTRUCTION_EF} "
            f"search_ef={config.HNSW_SEARCH_EF}"
        )

    # ── Write ────────────────────────────────────────────────────────────────

    def upsert_documents(self, docs: list[Document], ids: list[str]) -> int:
        """
        Upsert by paper_id.  Existing documents are overwritten (not duplicated),
        making sync idempotent.

        Embedding happens in bounded batches rather than one call for the whole
        sync.  LangChain's Chroma wrapper hands every document to a single
        embed request, which for a large sync means a multi-megabyte payload:
        Ollama rejects it with "input length exceeds the context length" and the
        entire sync is lost.  Batching also keeps peak memory flat regardless of
        how many papers a sync pulls.

        A batch that still fails is retried document-by-document so one
        pathological abstract costs one paper, not the whole run.
        """
        indexed = 0
        for start in range(0, len(docs), config.EMBED_BATCH_SIZE):
            batch = docs[start : start + config.EMBED_BATCH_SIZE]
            batch_ids = ids[start : start + config.EMBED_BATCH_SIZE]
            try:
                self._store.add_documents(documents=batch, ids=batch_ids)
                indexed += len(batch)
            except Exception as exc:
                logger.warning(
                    f"Batch embed failed ({exc}); retrying {len(batch)} documents "
                    "individually"
                )
                for doc, doc_id in zip(batch, batch_ids):
                    try:
                        self._store.add_documents(documents=[doc], ids=[doc_id])
                        indexed += 1
                    except Exception as doc_exc:
                        logger.error(
                            f"Skipping paper {doc_id} "
                            f"({len(doc.page_content)} chars): {doc_exc}"
                        )
        return indexed

    # ── Read ─────────────────────────────────────────────────────────────────

    def similarity_search_with_score(
        self,
        query: str,
        k: int = config.RETRIEVAL_FETCH_K,
    ) -> list[tuple[Document, float]]:
        """
        Returns (Document, cosine_distance) pairs.
        Distance ∈ [0, 2]: 0 = identical, 1 = orthogonal, 2 = opposite.
        Threshold filtering happens in pipeline.py.
        """
        return self._store.similarity_search_with_score(query, k=k)

    def distances_for_ids(
        self,
        query_embedding: list[float],
        paper_ids: list[str],
    ) -> dict[str, float]:
        """
        Exact cosine distance between the query and a specific set of documents.

        Used to apply the similarity guard to BM25-only candidates.  Those
        documents were never in the dense top-k, so their distance is unknown —
        and without it the guard is trivially bypassed by the sparse arm, which
        is how irrelevant papers reached the LLM context.

        One extra Chroma query with an ID filter; negligible next to the
        cross-encoder, let alone generation.
        """
        ids = [pid for pid in paper_ids if pid]
        if not ids:
            return {}
        try:
            res = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=len(ids),
                where={"paper_id": {"$in": ids}},
                include=["distances", "metadatas"],
            )
        except Exception as exc:
            # Fail open: a guard that errors must not take retrieval down with it.
            logger.warning(f"distances_for_ids failed (non-fatal): {exc}")
            return {}

        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return {
            m.get("paper_id", ""): float(d)
            for m, d in zip(metas, dists)
            if m.get("paper_id")
        }

    def get_all_documents(self) -> list[Document]:
        """Load every stored document for BM25 index construction."""
        result = self._collection.get(include=["documents", "metadatas"])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        if len(docs) != len(metas):
            # Not an assert: `python -O` strips those, and this check exists
            # precisely to stop a corrupt index from being served silently.
            raise RuntimeError(
                f"ChromaDB corruption detected: {len(docs)} documents but "
                f"{len(metas)} metadata entries. "
                "Run a fresh sync after deleting backend/data/chroma_db/"
            )
        return [
            Document(page_content=content, metadata=meta)
            for content, meta in zip(docs, metas)
        ]

    # ── Delete / Cleanup ─────────────────────────────────────────────────────

    def remove_person_from_docs(self, person_name: str) -> tuple[int, int]:
        """
        When a person is removed from the registry, either:
          - Delete their papers (if they were the sole institute author), or
          - Update the metadata to remove their name (if paper is co-authored
            with other institute members, the paper itself stays).
        Returns (deleted_count, updated_count).
        """
        result = self._collection.get(include=["metadatas"])
        to_delete: list[str] = []
        to_update_ids: list[str] = []
        to_update_metas: list[dict] = []

        for doc_id, meta in zip(result["ids"], result["metadatas"]):
            inst_authors_raw: str = meta.get("institute_authors", "")
            if person_name not in inst_authors_raw:
                continue

            remaining = [
                a.strip()
                for a in inst_authors_raw.split(",")
                if a.strip() and a.strip() != person_name
            ]
            if not remaining:
                to_delete.append(doc_id)
            else:
                # Rebuild roles/departments strings from remaining authors' metadata
                # (we only have the stored strings here, so just strip the name)
                to_update_ids.append(doc_id)
                to_update_metas.append(
                    {
                        **meta,
                        "institute_authors": ", ".join(remaining),
                    }
                )

        if to_delete:
            self._collection.delete(ids=to_delete)
        if to_update_ids:
            # ChromaDB update without embeddings preserves the existing vectors
            self._collection.update(ids=to_update_ids, metadatas=to_update_metas)

        logger.info(
            f"remove_person_from_docs '{person_name}': "
            f"deleted={len(to_delete)}, updated={len(to_update_ids)}"
        )
        return len(to_delete), len(to_update_ids)

    # ── Stats ────────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._collection.count()

    def get_chroma_client(self) -> chromadb.Client:
        return self._client
