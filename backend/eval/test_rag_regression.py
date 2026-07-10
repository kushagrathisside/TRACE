"""
DeepEval regression test suite — run before every deployment.

Each test asserts that a known class of query produces an answer that meets
minimum quality criteria using the local Ollama model as the judge.

Install:   pip install deepeval
Run:       cd backend && python -m pytest eval/test_rag_regression.py -v --tb=short
CI usage:  Add to your CI step — a failing test blocks deployment.

Add new test cases whenever a regression is discovered or a new topic area
is added to the institute's research portfolio.
"""

import sys

import pytest

sys.path.insert(0, ".")

THRESHOLD = 0.6  # Minimum acceptable score (0–1).  Lower than RAGAS thresholds
# because DeepEval metrics use a different scale.

# ── Lazy imports (skip all tests gracefully if deepeval not installed) ────────
try:
    from deepeval import assert_test
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    from eval.deepeval_config import get_judge

    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not DEEPEVAL_AVAILABLE,
    reason="deepeval not installed — run: pip install deepeval",
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _run(query: str) -> tuple[str, list[str]]:
    """Run the full pipeline; return (answer_text, context_strings)."""
    from rag import pipeline

    result = pipeline.run(query)
    summary = result["answer"].get("landscape_summary", "")
    contexts = [
        s.get("title", "") + ". " + s.get("venue", "") for s in result["sources"]
    ]
    return summary, contexts


# ── Answer Relevancy tests ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "query",
    [
        "I want to research transformer architectures for low-resource NLP",
        "Federated learning for privacy-preserving medical data analysis",
        "Graph neural networks applied to molecular property prediction",
    ],
)
def test_answer_relevancy(query):
    judge = get_judge()
    answer, contexts = _run(query)
    case = LLMTestCase(input=query, actual_output=answer, retrieval_context=contexts)
    metric = AnswerRelevancyMetric(threshold=THRESHOLD, model=judge)
    assert_test(case, [metric])


# ── Faithfulness tests ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "query",
    [
        "Reinforcement learning for robotic manipulation",
        "Attention mechanisms in computer vision",
    ],
)
def test_faithfulness(query):
    judge = get_judge()
    answer, contexts = _run(query)
    case = LLMTestCase(input=query, actual_output=answer, retrieval_context=contexts)
    metric = FaithfulnessMetric(threshold=THRESHOLD, model=judge)
    assert_test(case, [metric])


# ── Edge-case / regression tests ──────────────────────────────────────────────


def test_out_of_domain_returns_no_results():
    """Queries completely unrelated to the institute should not hallucinate papers."""
    result = __import__("rag.pipeline", fromlist=["run"]).run(
        "19th century French Romantic poetry and troubadour music"
    )
    no_results = result["answer"].get("no_relevant_research") is True
    empty_sources = len(result["sources"]) == 0
    assert no_results or empty_sources, (
        "Expected no results for out-of-domain query but got sources: "
        + str(result["sources"][:2])
    )


def test_structured_output_has_required_fields():
    """Pipeline must always return a valid ResearchLandscape structure."""
    result = __import__("rag.pipeline", fromlist=["run"]).run(
        "deep learning for image classification"
    )
    answer = result["answer"]
    assert "landscape_summary" in answer, "Missing landscape_summary"
    assert "related_papers" in answer, "Missing related_papers"
    assert "people_to_consult" in answer, "Missing people_to_consult"
    assert "next_steps" in answer, "Missing next_steps"
    assert isinstance(answer["related_papers"], list)
    assert isinstance(answer["people_to_consult"], list)
    assert isinstance(answer["next_steps"], list)
