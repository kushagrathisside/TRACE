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
from rag.chain import (
    PaperReference,
    PersonSuggestion,
    ResearchLandscape,
    _hallucination_guard,
)


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
    result, _ = _hallucination_guard(landscape, docs)
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
    result, _ = _hallucination_guard(landscape, docs)
    assert len(result.related_papers) == 1


def test_fabricated_title_is_dropped():
    docs = [_doc("Attention Is All You Need")]
    landscape = ResearchLandscape(
        landscape_summary="summary",
        related_papers=[_paper("A Novel Deep Learning Framework for Graph Embeddings")],
    )
    result, _ = _hallucination_guard(landscape, docs)
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
    result, _ = _hallucination_guard(landscape, docs)
    assert len(result.related_papers) == 1


def test_low_word_overlap_below_threshold_is_dropped():
    # "Deep Learning" vs "Reinforcement Learning" — 1 word shared out of 3 unique
    # Jaccard = 1/3 ≈ 0.33, below 0.75 threshold
    docs = [_doc("Deep Learning")]
    landscape = ResearchLandscape(
        landscape_summary="summary",
        related_papers=[_paper("Reinforcement Learning")],
    )
    result, _ = _hallucination_guard(landscape, docs)
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
    result, _ = _hallucination_guard(landscape, docs)
    assert len(result.related_papers) == 1
    assert result.related_papers[0].title == "Attention Is All You Need"


def test_empty_source_docs_drops_all():
    landscape = ResearchLandscape(
        landscape_summary="summary",
        related_papers=[_paper("Anything")],
    )
    result, _ = _hallucination_guard(landscape, [])
    assert len(result.related_papers) == 0


def test_no_related_papers_passes_through():
    landscape = ResearchLandscape(
        landscape_summary="no results",
        related_papers=[],
        no_relevant_research=True,
    )
    result, _ = _hallucination_guard(landscape, [])
    assert result.no_relevant_research is True
    assert result.related_papers == []


# ── People grounding ──────────────────────────────────────────────────────────
# Suggesting a person who does not appear on any retrieved paper is the most
# damaging thing this system can do — a student acts on it by emailing someone.
# Papers were checked from the start; people were not checked at all.


def _doc_with_authors(title: str, authors: str, institute: str = "") -> Document:
    return Document(
        page_content="abstract",
        metadata={
            "paper_title": title,
            "authors": authors,
            "institute_authors": institute or authors,
        },
    )


def test_person_on_retrieved_paper_is_kept():
    docs = [_doc_with_authors("Graph Nets", "Kushagra Srivastava, John Doe")]
    landscape = ResearchLandscape(
        landscape_summary="summary",
        people_to_consult=[
            PersonSuggestion(name="Kushagra Srivastava", role="faculty")
        ],
    )
    result, stats = _hallucination_guard(landscape, docs)
    assert len(result.people_to_consult) == 1
    assert stats["people_dropped"] == 0


def test_fabricated_person_is_dropped():
    docs = [_doc_with_authors("Graph Nets", "Kushagra Srivastava")]
    landscape = ResearchLandscape(
        landscape_summary="summary",
        people_to_consult=[PersonSuggestion(name="Dr. Imaginary Person")],
    )
    result, stats = _hallucination_guard(landscape, docs)
    assert result.people_to_consult == []
    assert stats["people_dropped"] == 1


def test_bare_surname_is_not_accepted():
    """ "Srivastava" alone identifies no one in particular."""
    docs = [_doc_with_authors("Graph Nets", "Kushagra Srivastava")]
    landscape = ResearchLandscape(
        landscape_summary="summary",
        people_to_consult=[PersonSuggestion(name="Srivastava")],
    )
    result, _ = _hallucination_guard(landscape, docs)
    assert result.people_to_consult == []


# ── Containment floor ─────────────────────────────────────────────────────────


def test_short_generic_title_does_not_match_by_containment():
    """
    A one-word fabricated title used to pass because it is a substring of
    almost every real title.
    """
    docs = [_doc("Deep Learning for Molecular Property Prediction")]
    landscape = ResearchLandscape(
        landscape_summary="summary",
        related_papers=[_paper("Learning")],
    )
    result, _ = _hallucination_guard(landscape, docs)
    assert result.related_papers == []


def test_guard_reports_counts():
    docs = [_doc("Attention Is All You Need")]
    landscape = ResearchLandscape(
        landscape_summary="summary",
        related_papers=[
            _paper("Attention Is All You Need"),
            _paper("Made Up Paper Title"),
        ],
    )
    _, stats = _hallucination_guard(landscape, docs)
    assert stats == {
        "papers_cited": 2,
        "papers_dropped": 1,
        "people_cited": 0,
        "people_dropped": 0,
    }
