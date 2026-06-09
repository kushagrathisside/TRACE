"""
BM25 keyword search + dense vector search fused via Reciprocal Rank Fusion (RRF).

Why hybrid search
-----------------
Dense retrieval excels at semantic similarity ("transformers for translation"
matches papers about "neural machine translation") but underperforms on exact
named entities.  If a student writes "Prof. Sharma's work on GNNs", the name
"Sharma" is a rare token with a low embedding signal.

BM25 (Okapi BM25) is a probabilistic keyword model — it scores documents by
term frequency normalised for document length.  It catches exact matches that
the dense model misses.

Reciprocal Rank Fusion (RRF)
-----------------------------
Given two ranked lists L1 (vector) and L2 (BM25), the fused score for a
document d is:

    RRF(d) = Σ  1 / (k + rank_i(d))
             i

where k=60 dampens the effect of high-rank outliers.  Documents that appear
in both lists are heavily boosted.  No score normalisation required.

Index lifecycle
---------------
The BM25 index is built over ALL documents in ChromaDB after each sync and on
server startup.  With 15 k papers × 200 words, build time ~0.5 s, memory ~25 MB.
"""

import logging
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class HybridSearcher:
    _instance: "HybridSearcher | None" = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @classmethod
    def build(cls, docs: list[Document]) -> "HybridSearcher":
        inst = object.__new__(cls)
        inst.docs = docs
        corpus = [
            (d.metadata.get("paper_title", "") + " " + d.page_content).lower().split()
            for d in docs
        ]
        inst.bm25 = BM25Okapi(corpus)
        cls._instance = inst
        logger.info(f"BM25 index built: {len(docs)} documents")
        return inst

    @classmethod
    def get(cls) -> "HybridSearcher | None":
        return cls._instance

    @classmethod
    def invalidate(cls) -> None:
        cls._instance = None

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 20) -> list[tuple[float, Document]]:
        """Return (bm25_score, doc) pairs for the top-k keyword matches."""
        tokens = query.lower().split()
        scores: np.ndarray = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (float(scores[i]), self.docs[i])
            for i in top_indices
            if scores[i] >= 0
        ]

    # ── Fusion ───────────────────────────────────────────────────────────────

    @staticmethod
    def rrf(
        vector_results: list[tuple[float, Document]],
        bm25_results: list[tuple[float, Document]],
        k: int = 60,
        top_n: int = 20,
    ) -> list[Document]:
        """
        Reciprocal Rank Fusion over two ranked lists, with a recency bonus.

        RRF score:  Σ  1 / (k + rank_i + 1)
        Recency bonus: RECENCY_WEIGHT * max(0, 1 - age_years / 10)
          - A paper from this year gets the full bonus.
          - A paper 10+ years old gets nothing.
          - Default RECENCY_WEIGHT=0.01 adds ≈ half a rank position for a
            current-year paper, so relevance still dominates.
        """
        import datetime
        current_year = datetime.date.today().year

        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, (_, doc) in enumerate(vector_results):
            pid = doc.metadata.get("paper_id", str(id(doc)))
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
            doc_map[pid] = doc

        for rank, (_, doc) in enumerate(bm25_results):
            pid = doc.metadata.get("paper_id", str(id(doc)))
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
            doc_map[pid] = doc

        # Recency discount
        import config as _cfg
        if _cfg.RECENCY_WEIGHT > 0:
            for pid, doc in doc_map.items():
                year = doc.metadata.get("year") or 0
                if year > 0:
                    age   = max(0, current_year - year)
                    bonus = _cfg.RECENCY_WEIGHT * max(0.0, 1.0 - age / 10.0)
                    scores[pid] = scores[pid] + bonus

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [doc_map[pid] for pid in sorted_ids[:top_n]]
