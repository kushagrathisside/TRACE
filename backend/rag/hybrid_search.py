"""
BM25 keyword search + dense vector search fused via Reciprocal Rank Fusion (RRF).

Why hybrid search
-----------------
Dense retrieval excels at semantic similarity ("transformers for translation"
matches papers about "neural machine translation") but underperforms on exact
named entities.  If a student writes "Kushagra Srivastava's work on GNNs", the surname
is a rare token with a low embedding signal.

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

What goes into the index
------------------------
Title + abstract + **author names**.  The author names matter: they live only
in Chroma metadata, so before they were added here the named-entity case above
could not work at all — the surname appeared in no indexed text, dense or sparse.
Adding them to the BM25 corpus needs no re-embedding, since this index is
rebuilt from Chroma at startup and after every sync.

Score floor
-----------
A query sharing no token with the corpus scores 0.0 against every document,
and rank_bm25 still returns a full top-k of those zeros.  Passing them to RRF
gives pure noise the same rank-based weight as genuine hits — a noise document
at BM25 rank 0 scores 1/61 = 0.0164 and outranks a real dense hit at rank 5
(1/66 = 0.0152).  Candidates at or below BM25_MIN_SCORE are therefore dropped.
"""

import logging
import re

import config
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Split on any non-alphanumeric run, so "Srivastava's" → ["srivastava", "s"] and
# "GNNs," → ["gnns"].  Plain .split() left punctuation attached to tokens,
# which silently broke exact-match retrieval for the entity queries BM25 is
# here to serve in the first place.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words carry no discriminative signal but do carry IDF weight in a
# small corpus, where they pull in unrelated documents on long natural-language
# queries ("I want to research ...").
_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have how i in is it its of on or that
    the to was were what when where which who will with using use used research
    study studies paper papers work works want like about into their there this
    these those we our can could would should my me
    """.split()
)

# Tokens shorter than this are dropped after stopword removal.
_MIN_TOKEN_LEN = 2


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords and 1-char tokens."""
    return [
        t
        for t in _TOKEN_RE.findall(text.lower())
        if len(t) >= _MIN_TOKEN_LEN and t not in _STOPWORDS
    ]


class HybridSearcher:
    _instance: "HybridSearcher | None" = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @classmethod
    def build(cls, docs: list[Document]) -> "HybridSearcher":
        inst = object.__new__(cls)
        inst.docs = docs
        corpus = [tokenize(cls._index_text(d)) for d in docs]
        inst.bm25 = BM25Okapi(corpus)
        cls._instance = inst
        logger.info(
            f"BM25 index built: {len(docs)} documents "
            f"(title + abstract + authors, stopwords removed)"
        )
        return inst

    @staticmethod
    def _index_text(doc: Document) -> str:
        """
        Text fed to BM25 for one document.

        page_content already begins with "Title: …", so the title is naturally
        weighted twice — which is the conventional field boost and is kept
        deliberately.  Author names come from metadata and appear nowhere in
        page_content, so without this they were unsearchable.
        """
        m = doc.metadata
        return " ".join(
            part
            for part in (
                m.get("paper_title", ""),
                doc.page_content,
                m.get("authors", ""),
                m.get("institute_authors", ""),
                m.get("venue", ""),
                m.get("departments", ""),
            )
            if part
        )

    @classmethod
    def get(cls) -> "HybridSearcher | None":
        return cls._instance

    @classmethod
    def invalidate(cls) -> None:
        cls._instance = None

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 20) -> list[tuple[float, Document]]:
        """
        Return (bm25_score, doc) pairs for the top-k keyword matches.

        Only documents scoring strictly above BM25_MIN_SCORE are returned, so a
        query with no lexical overlap yields an empty list rather than an
        arbitrary top-k of zero-score noise.  An empty list is the honest
        answer, and RRF handles it correctly.
        """
        tokens = tokenize(query)
        if not tokens:
            return []
        scores: np.ndarray = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (float(scores[i]), self.docs[i])
            for i in top_indices
            if scores[i] > config.BM25_MIN_SCORE
        ]

    # ── Fusion ───────────────────────────────────────────────────────────────

    @staticmethod
    def rrf(
        vector_results: list[tuple[float, Document]],
        bm25_results: list[tuple[float, Document]],
        k: int | None = None,
        top_n: int = 20,
    ) -> list[Document]:
        """Reciprocal Rank Fusion, returning documents only (see rrf_with_scores)."""
        return [
            doc
            for _, doc in HybridSearcher.rrf_with_scores(
                vector_results, bm25_results, k=k, top_n=top_n
            )
        ]

    @staticmethod
    def rrf_with_scores(
        vector_results: list[tuple[float, Document]],
        bm25_results: list[tuple[float, Document]],
        k: int | None = None,
        top_n: int = 20,
        weight_dense: float | None = None,
        weight_sparse: float | None = None,
    ) -> list[tuple[float, Document]]:
        """
        Weighted Reciprocal Rank Fusion over two ranked lists, plus a recency bonus.

        RRF score:  Σ  w_i / (k + rank_i + 1)
        Recency bonus: RECENCY_WEIGHT * max(0, 1 - age_years / 10)
          - A paper from this year gets the full bonus.
          - A paper 10+ years old gets nothing.
          - Default RECENCY_WEIGHT=0.01 adds ≈ half a rank position for a
            current-year paper, so relevance still dominates.

        Weights default to 1.0 / 1.0 (textbook RRF).  They are exposed because
        the right dense:sparse balance is corpus-specific and should be chosen
        from a sweep, not from the default in a paper.

        Returns (fused_score, doc) so the fusion stage can be traced — without
        the scores you cannot tell a document that ranked first in both lists
        from one that scraped in on a single weak list.
        """
        import datetime

        k = config.RRF_K if k is None else k
        w_dense = config.RRF_WEIGHT_DENSE if weight_dense is None else weight_dense
        w_sparse = config.RRF_WEIGHT_SPARSE if weight_sparse is None else weight_sparse
        current_year = datetime.date.today().year

        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for weight, results in ((w_dense, vector_results), (w_sparse, bm25_results)):
            if weight <= 0:
                continue
            for rank, (_, doc) in enumerate(results):
                pid = doc.metadata.get("paper_id", str(id(doc)))
                scores[pid] = scores.get(pid, 0.0) + weight / (k + rank + 1)
                doc_map[pid] = doc

        # Recency bonus
        if config.RECENCY_WEIGHT > 0:
            for pid, doc in doc_map.items():
                try:
                    year = int(doc.metadata.get("year") or 0)
                except (TypeError, ValueError):
                    continue
                if year > 0:
                    age = max(0, current_year - year)
                    bonus = config.RECENCY_WEIGHT * max(0.0, 1.0 - age / 10.0)
                    scores[pid] = scores[pid] + bonus

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [(scores[pid], doc_map[pid]) for pid in sorted_ids[:top_n]]
