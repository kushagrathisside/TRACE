"""
Cross-encoder reranker.

Why reranking
-------------
The bi-encoder (all-MiniLM-L6-v2) used for retrieval produces fast approximate
similarity scores — it encodes query and document independently, then computes
dot-product.  This misses fine-grained interactions between query tokens and
document tokens.

A cross-encoder reads query + document TOGETHER and produces a single relevance
score.  It is ~100× slower per pair but far more accurate.  The strategy:

  1. Retrieve RETRIEVAL_FETCH_K=20 candidates cheaply with the bi-encoder.
  2. Rerank those 20 with the cross-encoder.
  3. Keep top RETRIEVAL_K=5.

This gives recall of a large retrieval set + precision of a strong reranker.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS-MARCO passage ranking (search relevance task).
  - 22 M params, runs on CPU in ~80 ms for 20 pairs.
  - Scores are raw logits (higher = more relevant).
"""

import logging
import numpy as np
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
import config

logger = logging.getLogger(__name__)


class Reranker:
    _instance: "Reranker | None" = None

    @classmethod
    def get(cls) -> "Reranker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        logger.info(f"Loading reranker: {config.RERANKER_MODEL_NAME}")
        self.model = CrossEncoder(config.RERANKER_MODEL_NAME, max_length=512)

    def rerank(
        self,
        query: str,
        docs: list[Document],
        top_k: int = config.RETRIEVAL_K,
    ) -> list[tuple[float, Document]]:
        """
        Return (score, doc) pairs sorted descending by relevance.
        Input is title + first 400 chars of abstract to stay within 512 tokens.
        """
        if not docs:
            return []
        pairs = [
            (
                query,
                f"{d.metadata.get('paper_title', '')} {d.page_content[:400]}",
            )
            for d in docs
        ]
        scores: np.ndarray = self.model.predict(pairs)
        ranked = sorted(
            zip(scores.tolist(), docs),
            key=lambda x: x[0],
            reverse=True,
        )
        return ranked[:top_k]
