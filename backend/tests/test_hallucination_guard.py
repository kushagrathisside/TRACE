"""
Unit tests for the hallucination guard in rag/chain.py.

Tests the _is_grounded() matching logic and _hallucination_guard() filter.
These run without Ollama or ChromaDB.
"""

import os
import sys

os.environ.setdefault("ADMIN_PASSWORD", "test")
os.environ.setdefault("CORS_ORIGINS", "*")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document
from rag.chain import PaperReference, ResearchLandscape, _hallucination_guard


def _doc(title: str) -> Document:
    return Document(page_content="abstract", metadata={"paper_title": title})


def _paper(title: str) -> PaperReference:
    return PaperReference(title=title, year=2024)


# ── Exact / substring match ───────────────────────────────────────────────────


def test_exact_title_match_passes():
    docs = [_doc("Attention Is All You Need")]
    landscape = ResearchLandscape(
        landscape_summary="summary",
        related_papers=[_paper("Attention Is All You Need")],
    )
    result = _hallucination_guard(landscape, docs)
    assert len(result.related_papers) == 1


def test_substring_match_passes():
    docs = [
        _doc(
            "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
        )
    ]
    landscape = ResearchLandscape(
        landscape_summary="summary",
        related_papers=[
            _paper("BERT: Pre-training of Deep Bidirectional Transformers")
        ],
    )
    result = _hallucination_guard(landscape, docs)
    assert len(result.related_papers) == 1


def test_fabricated_title_is_dropped():
    docs = [_doc("Attention Is All You Need")]
    landscape = ResearchLandscape(
        landscape_summary="summary",
        related_papers=[_paper("A Novel Deep Learning Framework for Graph Embeddings")],
    )
    result = _hallucination_guard(landscape, docs)
    assert len(result.related_papers) == 0


# ── Jaccard fuzzy match (threshold 0.75) ─────────────────────────────────────


def test_high_word_overlap_passes():
    docs = [_doc("Graph Neural Networks for Molecular Property Prediction")]
    landscape = ResearchLandscape(
        landscape_summary="summary",
        related_papers=[
            _paper("Graph Neural Networks for Molecular Property Prediction Tasks")
        ],
    )
    result = _hallucination_guard(landscape, docs)
    assert len(result.related_papers) == 1


def test_low_word_overlap_below_threshold_is_dropped():
    # "Deep Learning" vs "Reinforcement Learning" — 1 word shared out of 3 unique
    # Jaccard = 1/3 ≈ 0.33, below 0.75 threshold
    docs = [_doc("Deep Learning")]
    landscape = ResearchLandscape(
        landscape_summary="summary",
        related_papers=[_paper("Reinforcement Learning")],
    )
    result = _hallucination_guard(landscape, docs)
    assert len(result.related_papers) == 0


# ── Mixed valid and invalid ───────────────────────────────────────────────────


def test_mixed_papers_only_valid_kept():
    docs = [
        _doc("Attention Is All You Need"),
        _doc("BERT: Pre-training of Deep Bidirectional Transformers"),
    ]
    landscape = ResearchLandscape(
        landscape_summary="summary",
        related_papers=[
            _paper("Attention Is All You Need"),
            _paper("A Fabricated Paper That Does Not Exist"),
        ],
    )
    result = _hallucination_guard(landscape, docs)
    assert len(result.related_papers) == 1
    assert result.related_papers[0].title == "Attention Is All You Need"


def test_empty_source_docs_drops_all():
    landscape = ResearchLandscape(
        landscape_summary="summary",
        related_papers=[_paper("Anything")],
    )
    result = _hallucination_guard(landscape, [])
    assert len(result.related_papers) == 0


def test_no_related_papers_passes_through():
    landscape = ResearchLandscape(
        landscape_summary="no results",
        related_papers=[],
        no_relevant_research=True,
    )
    result = _hallucination_guard(landscape, [])
    assert result.no_relevant_research is True
    assert result.related_papers == []
