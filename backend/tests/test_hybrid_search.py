"""
Unit tests for BM25 hybrid search and RRF fusion — no Ollama or ChromaDB needed.
"""

import os
import sys

import pytest

os.environ.setdefault("ADMIN_PASSWORD", "test")
os.environ.setdefault("CORS_ORIGINS", "*")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document
from rag.hybrid_search import HybridSearcher


def _doc(title: str, abstract: str = "") -> Document:
    return Document(
        page_content=f"Title: {title}\n\nAbstract: {abstract}",
        metadata={"paper_title": title, "paper_id": title.lower().replace(" ", "_")},
    )


DOCS = [
    _doc(
        "Graph Neural Networks for Molecular Property Prediction",
        "GNN molecular chemistry",
    ),
    _doc("Attention Is All You Need", "transformer self-attention NLP"),
    _doc("BERT Pre-training Transformers", "bidirectional encoder language model"),
    _doc("Federated Learning for Privacy", "distributed training privacy data"),
    _doc("Reinforcement Learning Robot Manipulation", "robot arm control reward"),
]


@pytest.fixture(autouse=True)
def build_index():
    HybridSearcher.build(DOCS)
    yield
    HybridSearcher.invalidate()


# ── Search ────────────────────────────────────────────────────────────────────


def test_search_returns_results():
    searcher = HybridSearcher.get()
    results = searcher.search("transformer attention NLP", top_k=3)
    assert len(results) > 0


def test_search_top_result_is_relevant():
    searcher = HybridSearcher.get()
    results = searcher.search("transformer attention", top_k=3)
    titles = [doc.metadata["paper_title"] for _, doc in results]
    assert any("Attention" in t or "BERT" in t or "Transformer" in t for t in titles)


def test_search_respects_top_k():
    searcher = HybridSearcher.get()
    results = searcher.search("learning", top_k=2)
    assert len(results) <= 2


def test_zero_score_docs_are_filtered_out():
    """
    A query with no lexical overlap must return NOTHING.

    This test previously asserted the opposite ("zero-score docs not filtered"),
    codifying the bug it should have caught: rank_bm25 returns a full top-k of
    0.0-scoring documents for a non-matching query, and passing those to RRF
    gives pure noise the same rank weight as genuine hits.
    """
    searcher = HybridSearcher.get()
    results = searcher.search("zzzzqqqq", top_k=5)
    assert results == []


def test_all_returned_scores_are_positive():
    searcher = HybridSearcher.get()
    results = searcher.search("transformer attention", top_k=5)
    assert results
    assert all(score > 0 for score, _ in results)


def test_stopword_only_query_returns_nothing():
    searcher = HybridSearcher.get()
    assert searcher.search("i want to research the", top_k=5) == []


def test_author_names_are_searchable():
    """Author names live in metadata; they must still reach the BM25 index."""
    doc = Document(
        page_content="Title: Some Paper\n\nAbstract: unrelated content here",
        metadata={
            "paper_title": "Some Paper",
            "paper_id": "p_author",
            "authors": "Kushagra Srivastava, John Doe",
            "institute_authors": "Kushagra Srivastava",
        },
    )
    HybridSearcher.build(DOCS + [doc])
    results = HybridSearcher.get().search("Srivastava", top_k=3)
    assert [d.metadata["paper_id"] for _, d in results] == ["p_author"]


def test_punctuation_is_stripped_from_query_tokens():
    """
    "Srivastava's" must match "Srivastava" — plain .split() left the apostrophe attached
    and silently broke the exact-entity lookups BM25 exists to serve.

    Needs a multi-document corpus: BM25 IDF is negative for a term present in
    every document, so a rare name is only distinguishing when most documents
    lack it.
    """
    doc = Document(
        page_content="Title: Graph Work\n\nAbstract: graphs",
        metadata={
            "paper_title": "Graph Work",
            "paper_id": "p_punct",
            "authors": "Kushagra Srivastava",
        },
    )
    HybridSearcher.build(DOCS + [doc])
    results = HybridSearcher.get().search("Srivastava's work", top_k=3)
    assert [d.metadata["paper_id"] for _, d in results] == ["p_punct"]


def test_invalidate_clears_index():
    HybridSearcher.invalidate()
    assert HybridSearcher.get() is None


# ── RRF Fusion ────────────────────────────────────────────────────────────────


def test_rrf_boosts_documents_in_both_lists():
    doc_a = _doc("Shared Document A")
    doc_b = _doc("Only in Vector B")
    doc_c = _doc("Only in BM25 C")

    # doc_a appears in both lists → should rank highly after fusion
    vector_list = [(0.3, doc_a), (0.2, doc_b)]
    bm25_list = [(5.0, doc_a), (3.0, doc_c)]

    fused = HybridSearcher.rrf(vector_list, bm25_list, top_n=3)
    titles = [d.metadata["paper_title"] for d in fused]
    assert titles[0] == "Shared Document A", "Document in both lists should rank first"


def test_rrf_returns_at_most_top_n():
    docs = [_doc(f"Doc {i}") for i in range(5)]
    vector_list = [(float(i), d) for i, d in enumerate(docs)]
    bm25_list = [(float(i), d) for i, d in enumerate(docs)]
    fused = HybridSearcher.rrf(vector_list, bm25_list, top_n=3)
    assert len(fused) <= 3


def test_rrf_handles_empty_bm25():
    doc = _doc("Only Vector")
    vector_list = [(0.5, doc)]
    fused = HybridSearcher.rrf(vector_list, [], top_n=5)
    assert len(fused) == 1
    assert fused[0].metadata["paper_title"] == "Only Vector"


def test_rrf_handles_empty_vector():
    doc = _doc("Only BM25")
    bm25_list = [(3.0, doc)]
    fused = HybridSearcher.rrf([], bm25_list, top_n=5)
    assert len(fused) == 1
