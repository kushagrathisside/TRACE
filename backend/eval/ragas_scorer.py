"""
Per-query RAGAS scoring — called from pipeline.run() after every answer.

Scores two reference-free metrics using the local Ollama LLM as the judge:

  Faithfulness (0–1):
    Fraction of claims in the answer that are supported by the retrieved
    context.  < 0.80 suggests the LLM is hallucinating or ignoring context.

  Answer Relevancy (0–1):
    How well the answer addresses the actual question.
    < 0.70 suggests the LLM went off-topic or gave a generic response.

Results are appended to feedback.jsonl alongside user thumbs-up/down data.
Non-fatal: if ragas is not installed, scoring is silently skipped.

Install: pip install ragas datasets
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import config
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def score_from_sources(query: str, answer_summary: str, sources: list[dict]) -> dict:
    """
    Score using the `sources` payload the pipeline returned.

    Offline runners hold serialised sources rather than Documents, and passing
    an empty context instead (as run_eval.py used to) makes faithfulness a
    meaningless number rather than an absent one — a model cannot be unfaithful
    to nothing.
    """
    docs = [
        Document(
            page_content=f"{s.get('title', '')}. {s.get('venue', '')}",
            metadata={"paper_id": s.get("paper_id", "")},
        )
        for s in sources
    ]
    return score_and_log(query, answer_summary, docs, log=False)


def score_and_log(
    query: str,
    answer_summary: str,
    context_docs: list[Document],
    query_id: str = "",
    log: bool = True,
) -> dict:
    """
    Compute RAGAS scores for one query-answer pair and append to feedback.jsonl.
    Returns {"faithfulness": float|None, "answer_relevancy": float|None}.

    `query_id` links the score to the query's trace record so judge scores,
    user ratings and the retrieved document set can be joined.

    Caveat worth keeping in mind when reading these numbers: the judge is the
    same small local model that generated the answer.  Treat the scores as a
    smoke alarm, not a gate, until you have measured agreement against
    hand-labelled examples.
    """
    scores: dict = {"faithfulness": None, "answer_relevancy": None}

    try:
        from datasets import Dataset
        from llm_provider import LLMProvider
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, faithfulness

        llm_wrapper = LangchainLLMWrapper(LLMProvider.get_llm(temperature=0.0))

        ds = Dataset.from_dict(
            {
                "question": [query],
                "answer": [answer_summary],
                "contexts": [[d.page_content for d in context_docs]]
                if context_docs
                else [[""]],
            }
        )
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy],
            llm=llm_wrapper,
            raise_exceptions=False,
        )
        scores = {
            "faithfulness": round(float(result["faithfulness"]), 3),
            "answer_relevancy": round(float(result["answer_relevancy"]), 3),
        }
    except ImportError:
        logger.debug("ragas not installed — skipping RAGAS scoring")
        return scores
    except Exception as exc:
        logger.warning(f"RAGAS scoring failed (non-fatal): {exc}")
        return scores

    if log:
        entry = {
            "event": "ragas_score",
            "query_id": query_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query[:200],
            **scores,
        }
        path = Path(config.FEEDBACK_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    return scores
