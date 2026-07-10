"""
Semantic query cache backed by a dedicated ChromaDB collection.

How it works
------------
When a student submits a query we embed it and check whether a semantically
identical query has been answered before (cosine distance < CACHE_HIT_DISTANCE).
On a hit we return the stored answer instantly — no retrieval, no LLM call.

Why cosine distance not exact string match
------------------------------------------
"What is NLP?" and "Can you explain natural language processing?" are the same
question.  Embedding-based similarity catches these rewrites where a dict/hash
cache would miss them.

Cache invalidation
------------------
After every sync the document set changes, so cached answers may reference
stale data.  `invalidate()` drops and recreates the collection.
"""

import json
import logging
import uuid
from datetime import datetime, timezone

import chromadb
import config

logger = logging.getLogger(__name__)


class SemanticCache:
    COLLECTION = "query_cache"

    def __init__(self, client: chromadb.Client):
        self._client = client
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def get(self, query_embedding: list[float]) -> dict | None:
        """Return cached result dict or None on miss."""
        if self._col.count() == 0:
            return None
        result = self._col.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["distances", "metadatas"],
        )
        distances = result["distances"][0]
        if not distances:
            return None
        dist = distances[0]
        if dist < config.CACHE_HIT_DISTANCE:
            meta = result["metadatas"][0][0]

            created_at = meta.get("created_at", 0)
            now = datetime.now(timezone.utc).timestamp()
            if now - created_at > config.CACHE_MAX_AGE_DAYS * 86400:
                logger.debug(f"semantic_cache EXPIRED dist={dist:.4f}")
                return None

            logger.info(f"semantic_cache HIT  dist={dist:.4f}")
            return {
                "answer": json.loads(meta["answer"]),
                "sources": json.loads(meta["sources"]),
            }
        logger.debug(f"semantic_cache MISS dist={dist:.4f}")
        return None

    def set(
        self,
        query_embedding: list[float],
        query: str,
        answer: dict,
        sources: list,
    ) -> None:
        self._col.upsert(
            embeddings=[query_embedding],
            documents=[query],
            metadatas=[
                {
                    "answer": json.dumps(answer),
                    "sources": json.dumps(sources),
                    "created_at": datetime.now(timezone.utc).timestamp(),
                }
            ],
            ids=[str(uuid.uuid4())],
        )

    def invalidate(self) -> None:
        """Drop all cached answers — called after every sync."""
        try:
            self._client.delete_collection(self.COLLECTION)
        except Exception:
            pass
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("semantic_cache invalidated")
