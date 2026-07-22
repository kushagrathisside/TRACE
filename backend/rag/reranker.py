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

import config
import numpy as np
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    _instance: "Reranker | None" = None

    @classmethod
    def get(cls) -> "Reranker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self.model = None
        self.status = "ok"
        self.error: str | None = None

        if not config.RERANKER_MODEL_NAME:
            # Explicit opt-out — still reported, because a passthrough reranker
            # changes what every downstream metric means.
            self.status = "disabled"
            self.error = "RERANKER_MODEL_NAME is empty"
            logger.warning(
                "Reranker DISABLED (RERANKER_MODEL_NAME is empty) — ranking falls "
                "back to fusion order. nDCG@5 will be materially lower."
            )
            return

        logger.info(f"Loading reranker: {config.RERANKER_MODEL_NAME}")
        try:
            self.model = CrossEncoder(config.RERANKER_MODEL_NAME, max_length=512)
            logger.info(f"Reranker ready: {config.RERANKER_MODEL_NAME}")
        except Exception as exc:
            self.status = "degraded"
            self.error = str(exc)
            if config.RERANKER_REQUIRED:
                raise RuntimeError(
                    f"Reranker '{config.RERANKER_MODEL_NAME}' failed to load: {exc}\n"
                    "RERANKER_REQUIRED=true, refusing to serve a silently degraded "
                    "ranker. Set RERANKER_REQUIRED=false to run without it."
                ) from exc
            # Loud, not silent: this used to be a warning that scrolled past at
            # startup while the docs continued to advertise cross-encoder
            # reranking.  The mismatch is now visible in /health and in every
            # trace record and MLflow run.
            logger.error(
                f"RERANKER FAILED TO LOAD ({exc}) — falling back to passthrough. "
                "Ranking quality is degraded; /health reports 'degraded' and all "
                "traces are tagged reranker_active=false."
            )

    @property
    def active(self) -> bool:
        """True only when a real cross-encoder is scoring the candidates."""
        return self.model is not None

    @classmethod
    def health(cls) -> dict:
        """
        Reranker subsystem status for /health.

        Deliberately does NOT construct the instance: loading a cross-encoder
        downloads and initialises model weights, and a health probe that can
        block for the length of a model download is worse than no probe.
        The model is loaded during startup warm-up instead.
        """
        if cls._instance is None:
            return {
                "status": "not_loaded",
                "model": config.RERANKER_MODEL_NAME or None,
                "active": False,
            }
        inst = cls._instance
        return {
            "status": inst.status,
            "model": config.RERANKER_MODEL_NAME or None,
            "active": inst.active,
            **({"error": inst.error} if inst.error else {}),
        }

    @staticmethod
    def _document_text(doc: Document) -> str:
        """
        The document side of the cross-encoder pair.

        Author names are included deliberately.  They live only in metadata, so
        scoring title + abstract alone left the reranker structurally blind to
        the one field an author query is about: for "Yoshua Bengio research" it
        was scoring topical similarity against text containing no author names
        at all, and demoting the correct papers.  Measured effect on the author
        slice was nDCG@5 0.464 → 0.213 — a code defect, not a model limitation.

        Budget: ms-marco-MiniLM truncates at 512 tokens, so the abstract is
        capped to leave room for the metadata prefix.
        """
        m = doc.metadata
        head = " ".join(
            part
            for part in (
                m.get("paper_title", ""),
                m.get("institute_authors", "") or m.get("authors", ""),
                m.get("venue", ""),
            )
            if part
        )
        return f"{head} {doc.page_content[:350]}"

    def rerank(
        self,
        query: str,
        docs: list[Document],
        top_k: int = config.RETRIEVAL_K,
    ) -> list[tuple[float, Document]]:
        """
        Return (score, doc) pairs sorted descending by relevance.
        See _document_text for what the model actually reads.
        """
        if not docs:
            return []
        pairs = [(query, self._document_text(d)) for d in docs]

        if self.model is None:
            # Passthrough: preserve fusion order, and use NaN rather than a
            # fake descending score so no caller can mistake these for
            # cross-encoder relevance scores or threshold against them.
            logger.debug("Reranker inactive — preserving fusion order.")
            return [(float("nan"), doc) for doc in docs[:top_k]]

        scores: np.ndarray = self.model.predict(pairs)
        ranked = sorted(
            zip(scores.tolist(), docs),
            key=lambda x: x[0],
            reverse=True,
        )
        return ranked[:top_k]
