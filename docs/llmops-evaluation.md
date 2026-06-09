# TRACE — LLMOps Evaluation Guide

## What "LLMOps Evaluation" Means Here

LLMOps evaluation is not a single script you run once. It is a **continuous measurement system** that answers three questions at all times:

1. **Is the system correct right now?** (Online monitoring)
2. **Did my change make it better or worse?** (Offline regression testing)
3. **Where is it failing and why?** (Root-cause diagnostics)

For a RAG system, these three questions decompose into distinct measurement layers — retrieval quality, generation quality, and end-to-end system quality — each measured with different tools and cadences.

---

## Framework Landscape — Which One to Use

| Framework | Best for | Local-only? | Integration effort |
|---|---|---|---|
| **RAGAS** | RAG-specific metrics on live or batched queries | Yes (uses your LLM as judge) | Low — pip install, 10 lines |
| **TruLens** | Continuous monitoring with a live dashboard | Yes | Medium — wraps your app |
| **DeepEval** | Regression test suite in pytest (CI/CD) | Yes (local judge LLM) | Medium |
| **MLflow** | Tracking eval results across experiments/versions | Yes (fully local) | Low — logging API |
| **LangSmith** | Full trace observability (LangChain native) | No (cloud) | Low, but requires API key |
| **Promptfoo** | A/B testing prompts without code changes | Yes | Low |

**Recommended stack for this project:**
- **RAGAS** → live per-query scoring (faithfulness + answer relevancy on every request)
- **DeepEval** → offline regression test suite run before merging changes
- **MLflow** → experiment tracker that stores all eval results over time
- **TruLens** → optional dashboard if you want a visual monitoring UI

All four run fully locally. No cloud account required.

---

## 1. RAGAS — Live Per-Query Scoring

RAGAS measures two things without any ground truth: whether the answer is grounded in the retrieved context (Faithfulness) and whether it actually addresses the question (Answer Relevancy). Both use your own LLM as the judge — no external API needed.

### Install

```bash
pip install ragas
```

### Integration into `pipeline.py`

Add RAGAS scoring at the end of `pipeline.run()`, after the answer is generated but before writing to cache. Scores are appended to `feedback.jsonl` so they accumulate automatically alongside thumbs-up/down ratings.

```python
# backend/eval/ragas_scorer.py
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from langchain_core.documents import Document

import config
from llm_provider import LLMProvider

logger = logging.getLogger(__name__)


def score_and_log(
    query: str,
    answer_summary: str,
    context_docs: list[Document],
) -> dict:
    """
    Compute RAGAS Faithfulness and Answer Relevancy using the local Ollama LLM
    as the evaluator.  Scores are in [0, 1]; higher is better.

    Faithfulness:    fraction of claims in the answer that are supported by
                     the retrieved context.  < 0.8 suggests hallucination.
    Answer Relevancy: how well the answer addresses the actual question.
                     < 0.7 suggests the LLM went off-topic.
    """
    try:
        llm_wrapper = LangchainLLMWrapper(LLMProvider.get_llm(temperature=0.0))

        ds = Dataset.from_dict({
            "question": [query],
            "answer":   [answer_summary],
            "contexts": [[d.page_content for d in context_docs]],
        })

        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy],
            llm=llm_wrapper,
            raise_exceptions=False,
        )

        scores = {
            "faithfulness":     round(float(result["faithfulness"]),     3),
            "answer_relevancy": round(float(result["answer_relevancy"]), 3),
        }
    except Exception as exc:
        logger.warning(f"RAGAS scoring failed (non-fatal): {exc}")
        scores = {"faithfulness": None, "answer_relevancy": None}

    # Persist to feedback.jsonl
    entry = {
        "event":            "ragas_score",
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "query":            query[:200],
        **scores,
    }
    path = Path(config.FEEDBACK_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return scores
```

Then call it from `pipeline.py`, inside `run()`, right after generating the answer:

```python
# In pipeline.run(), after landscape = generate_answer(query, final_docs):
from eval.ragas_scorer import score_and_log
ragas_scores = score_and_log(query, landscape.landscape_summary, final_docs)
logger.info(json.dumps({"event": "ragas", **ragas_scores, "query": query[:80]}))
```

### What the Scores Mean in Practice

| Score | Meaning | Action |
|---|---|---|
| Faithfulness < 0.7 | LLM is citing things not in the context | Strengthen the hallucination guard, lower K, or try a larger model |
| Faithfulness 0.7–0.9 | Acceptable, some leakage | Monitor trend; tighten JSON prompt if trending down |
| Faithfulness > 0.9 | Good | No action needed |
| Answer Relevancy < 0.6 | Answer is off-topic or too generic | Improve the system prompt; check if query expansion is distorting queries |
| Answer Relevancy > 0.8 | Good | No action needed |

---

## 2. DeepEval — Offline Regression Test Suite

DeepEval turns evaluation into a pytest test suite. You write test cases with expected behaviours, run them before deploying any change, and get a pass/fail result. This is the CI/CD safety net.

### Install

```bash
pip install deepeval
```

### Configure a Local Judge

DeepEval uses an LLM to evaluate answers. Configure it to use your local Ollama model so no API key is needed:

```python
# backend/eval/deepeval_config.py
from deepeval.models.base_model import DeepEvalBaseLLM
from llm_provider import LLMProvider

class OllamaJudge(DeepEvalBaseLLM):
    """Wraps the local ChatOllama as DeepEval's evaluator LLM."""

    def load_model(self):
        return LLMProvider.get_llm(temperature=0.0)

    def generate(self, prompt: str) -> str:
        from langchain_core.messages import HumanMessage
        model = self.load_model()
        return model.invoke([HumanMessage(content=prompt)]).content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "ollama-local"
```

### Write the Test Suite

```python
# backend/eval/test_rag_regression.py
"""
Run with:  cd backend && python -m pytest eval/test_rag_regression.py -v

Each test asserts that a known query produces an answer that satisfies
specific quality criteria.  Add new test cases whenever a regression is found.
"""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    HallucinationMetric,
    ContextualRecallMetric,
)
import sys; sys.path.insert(0, ".")
from eval.deepeval_config import OllamaJudge
from rag import pipeline
from rag.vector_store import VectorStoreManager
from llm_provider import LLMProvider

judge = OllamaJudge()
THRESHOLD = 0.7   # minimum acceptable score


def _run_pipeline(query: str) -> tuple[str, list[str]]:
    """Helper: run the full pipeline and return (answer_text, retrieved_contexts)."""
    result = pipeline.run(query)
    answer = result["answer"]
    summary = answer.get("landscape_summary", "")
    contexts = [s["title"] + ". " + s.get("venue", "") for s in result["sources"]]
    return summary, contexts


# ── Test cases ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("query,expected_keywords", [
    (
        "I want to work on graph neural networks for molecular property prediction",
        ["graph", "neural", "molecule"],
    ),
    (
        "My thesis is on federated learning for privacy-preserving healthcare",
        ["federated", "privacy", "learning"],
    ),
    (
        "Low-resource NLP for Indian regional languages using transformer models",
        ["language", "transformer", "NLP"],
    ),
])
def test_answer_relevancy(query, expected_keywords):
    answer, contexts = _run_pipeline(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=contexts,
    )
    metric = AnswerRelevancyMetric(threshold=THRESHOLD, model=judge)
    assert_test(test_case, [metric])


@pytest.mark.parametrize("query", [
    "Graph neural networks for citation network analysis",
    "Reinforcement learning for robot manipulation tasks",
])
def test_faithfulness(query):
    answer, contexts = _run_pipeline(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=contexts,
    )
    metric = FaithfulnessMetric(threshold=THRESHOLD, model=judge)
    assert_test(test_case, [metric])


def test_no_results_for_out_of_domain_query():
    """A query completely unrelated to CS/engineering should return no_relevant_research=True."""
    result = pipeline.run("medieval French poetry and troubadour music")
    assert result["answer"].get("no_relevant_research") is True or \
           len(result["sources"]) == 0, \
           "Expected no results for out-of-domain query"
```

Run as part of CI before any deployment:
```bash
cd backend
python -m pytest eval/test_rag_regression.py -v --tb=short
```

---

## 3. MLflow — Experiment Tracking

MLflow tracks every eval run so you can compare metrics across prompt versions, model swaps, and K changes. Fully local, no cloud account.

### Install

```bash
pip install mlflow
```

### Eval Runner Script

```python
# backend/eval/run_eval.py
"""
Runs the full eval set and logs results to a local MLflow experiment.

Usage:
    cd backend
    python eval/run_eval.py --run-name "K=5-reranker-on"

View results:
    mlflow ui --backend-store-uri backend/data/mlruns
    # open http://localhost:5000
"""

import argparse
import json
import time
from pathlib import Path

import mlflow
import sys; sys.path.insert(0, ".")
from rag import pipeline
from eval.ragas_scorer import score_and_log
import config

EVAL_SET_PATH = Path("data/eval_set.json")
MLRUNS_PATH   = Path("data/mlruns")


def load_eval_set() -> list[dict]:
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(
            f"{EVAL_SET_PATH} not found. Create it with annotated queries — "
            "see developer-guide.md § Building Ground Truth."
        )
    return json.loads(EVAL_SET_PATH.read_text())


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    return float(any(pid in retrieved_ids[:k] for pid in relevant_ids))


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    import math
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, pid in enumerate(retrieved_ids[:k])
        if pid in relevant_ids
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / ideal if ideal > 0 else 0.0


def run(run_name: str):
    mlflow.set_tracking_uri(str(MLRUNS_PATH))
    mlflow.set_experiment("trace-eval")

    eval_set = load_eval_set()
    k = config.RETRIEVAL_K

    with mlflow.start_run(run_name=run_name):
        # Log current config as params
        mlflow.log_params({
            "retrieval_k":           k,
            "fetch_k":               config.RETRIEVAL_FETCH_K,
            "embedding_model":       config.EMBEDDING_MODEL_NAME,
            "reranker_model":        config.RERANKER_MODEL_NAME,
            "llm_model":             config.LLM_MODEL_NAME,
            "min_similarity_dist":   config.MIN_SIMILARITY_DISTANCE,
            "hnsw_M":                config.HNSW_M,
            "hnsw_search_ef":        config.HNSW_SEARCH_EF,
        })

        hit_rates, ndcgs, faithfulness_scores, relevancy_scores, latencies = [], [], [], [], []

        for item in eval_set:
            t0 = time.perf_counter()
            result = pipeline.run(item["query"])
            elapsed = (time.perf_counter() - t0) * 1000

            retrieved_ids = [s["title"] for s in result["sources"]]  # use title as proxy ID
            relevant_ids  = item.get("relevant_paper_ids", [])

            hit_rates.append(hit_rate_at_k(retrieved_ids, relevant_ids, k))
            ndcgs.append(ndcg_at_k(retrieved_ids, relevant_ids, k))
            latencies.append(elapsed)

            # RAGAS scores
            scores = score_and_log(
                item["query"],
                result["answer"].get("landscape_summary", ""),
                [],   # pass empty list if context_docs not available here
            )
            if scores["faithfulness"] is not None:
                faithfulness_scores.append(scores["faithfulness"])
            if scores["answer_relevancy"] is not None:
                relevancy_scores.append(scores["answer_relevancy"])

        def avg(lst): return sum(lst) / len(lst) if lst else 0.0

        metrics = {
            f"hit_rate_at_{k}":        avg(hit_rates),
            f"ndcg_at_{k}":            avg(ndcgs),
            "faithfulness_mean":       avg(faithfulness_scores),
            "answer_relevancy_mean":   avg(relevancy_scores),
            "latency_p50_ms":          sorted(latencies)[len(latencies)//2],
            "latency_p95_ms":          sorted(latencies)[int(len(latencies)*0.95)],
            "eval_set_size":           len(eval_set),
        }
        mlflow.log_metrics(metrics)

        print("\n── Eval Results ──")
        for k_, v in metrics.items():
            print(f"  {k_:<30} {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="baseline")
    args = parser.parse_args()
    run(args.run_name)
```

**Workflow for comparing two versions:**
```bash
# Baseline
python eval/run_eval.py --run-name "baseline-K5-llama32"

# After changing K or prompt:
python eval/run_eval.py --run-name "K7-expanded-prompt"

# View side-by-side in the MLflow UI:
mlflow ui --backend-store-uri data/mlruns
```

---

## 4. TruLens — Live Monitoring Dashboard (Optional)

TruLens wraps your pipeline and records every call with its RAG triad scores (Groundedness, Answer Relevance, Context Relevance). It ships a local web dashboard.

### Install

```bash
pip install trulens-eval
```

### Wrap the Query Endpoint

```python
# backend/eval/trulens_setup.py
from trulens_eval import Tru, TruCustomApp, Feedback, Select
from trulens_eval.feedback.provider import LiteLLM   # uses Ollama via LiteLLM
import litellm

# Point LiteLLM at local Ollama
litellm.api_base = "http://localhost:11434"

tru = Tru(database_url="sqlite:///data/trulens.db")

provider = LiteLLM(model_engine="ollama/llama3.2")

# Define the three feedback functions
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
)
f_answer_relevance = (
    Feedback(provider.relevance, name="Answer Relevance")
    .on_input()
    .on_output()
)
f_context_relevance = (
    Feedback(provider.context_relevance, name="Context Relevance")
    .on_input()
    .on(Select.RecordCalls.retrieve.rets.collect())
    .aggregate(lambda x: sum(x)/len(x) if x else 0)
)

FEEDBACKS = [f_groundedness, f_answer_relevance, f_context_relevance]
```

```python
# In main.py, wrap the pipeline call:
from eval.trulens_setup import tru, FEEDBACKS
from trulens_eval import TruCustomApp

recorder = TruCustomApp(pipeline, app_id="trace-v1", feedbacks=FEEDBACKS)

@app.post("/api/query")
async def query_endpoint(request: Request, req: QueryRequest):
    loop = asyncio.get_event_loop()
    with recorder as recording:
        result = await loop.run_in_executor(None, pipeline.run, req.idea)
    return result
```

**View the dashboard:**
```bash
python -c "from trulens_eval import Tru; Tru(database_url='sqlite:///data/trulens.db').run_dashboard()"
# open http://localhost:8501
```

The dashboard shows every query with its three RAG triad scores, a leaderboard across app versions, and time-series trends.

---

## 5. Self-Retrieval Test Script

Zero-cost sanity check. If a paper's own abstract does not retrieve that paper in top K, something is fundamentally broken with the embeddings or HNSW index.

```python
# backend/eval/self_retrieval.py
"""
Usage:  cd backend && python eval/self_retrieval.py

Expected: Hit Rate@5 > 0.90
If lower: check embedding model version, HNSW settings, or index corruption.
"""
import sys; sys.path.insert(0, ".")
from rag.vector_store import VectorStoreManager
import config


def run(k: int = 5, sample: int = 200):
    vs    = VectorStoreManager.get_or_create()
    docs  = vs.get_all_documents()

    if not docs:
        print("No documents in DB. Run a sync first.")
        return

    # Optionally sample to keep runtime short
    import random
    subset = random.sample(docs, min(sample, len(docs)))

    hits = 0
    for doc in subset:
        pid     = doc.metadata.get("paper_id", "")
        query   = doc.page_content[:300]   # use first 300 chars of abstract
        results = vs.similarity_search_with_score(query, k=k)
        retrieved_ids = [d.metadata.get("paper_id") for d, _ in results]
        if pid in retrieved_ids:
            hits += 1

    rate = hits / len(subset)
    status = "✓ PASS" if rate >= 0.90 else "✗ FAIL"
    print(f"{status}  Self-retrieval Hit Rate@{k}: {rate:.2%}  ({hits}/{len(subset)} sampled)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--k",      type=int, default=5)
    p.add_argument("--sample", type=int, default=200)
    run(**vars(p.parse_args()))
```

---

## 6. Feedback Analysis Script

Mines the existing `feedback.jsonl` to surface the weakest-performing queries.

```python
# backend/eval/analyse_feedback.py
"""
Usage:  cd backend && python eval/analyse_feedback.py

Outputs:
  - Overall thumbs-down rate
  - Top downvoted queries
  - RAGAS score trends (if scores have been logged)
  - Suggestions for what to fix
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import sys; sys.path.insert(0, ".")
import config


def run():
    path = Path(config.FEEDBACK_LOG_PATH)
    if not path.exists():
        print("No feedback data yet. Collect some queries first.")
        return

    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    # ── Thumbs up/down ────────────────────────────────────────────────────────
    ratings = [r for r in records if r.get("event") != "ragas_score" and "rating" in r]
    if ratings:
        total = len(ratings)
        downs = sum(1 for r in ratings if r["rating"] == "down")
        print(f"\n── User Ratings ──────────────────────────────")
        print(f"  Total rated queries:  {total}")
        print(f"  Thumbs-down rate:     {downs/total:.1%}  ({downs}/{total})")
        if downs:
            worst = Counter(r["query"][:80] for r in ratings if r["rating"] == "down")
            print(f"\n  Most downvoted queries:")
            for q, count in worst.most_common(5):
                print(f"    [{count}x] {q}")

    # ── RAGAS scores ──────────────────────────────────────────────────────────
    ragas = [r for r in records if r.get("event") == "ragas_score"]
    if ragas:
        faithfulness_scores = [r["faithfulness"]     for r in ragas if r.get("faithfulness")     is not None]
        relevancy_scores    = [r["answer_relevancy"] for r in ragas if r.get("answer_relevancy") is not None]
        print(f"\n── RAGAS Scores ({len(ragas)} queries scored) ──────")
        if faithfulness_scores:
            avg_f = sum(faithfulness_scores) / len(faithfulness_scores)
            low_f = sum(1 for s in faithfulness_scores if s < 0.7)
            print(f"  Faithfulness:     mean={avg_f:.3f}  low_count(<0.7)={low_f}")
        if relevancy_scores:
            avg_r = sum(relevancy_scores) / len(relevancy_scores)
            low_r = sum(1 for s in relevancy_scores if s < 0.7)
            print(f"  Answer Relevancy: mean={avg_r:.3f}  low_count(<0.7)={low_r}")

    # ── Actionable suggestions ────────────────────────────────────────────────
    print(f"\n── Suggestions ───────────────────────────────")
    if ratings and (downs / len(ratings)) > 0.3:
        print("  ! High thumbs-down rate. Run the self-retrieval test to check retrieval quality.")
    if ragas and faithfulness_scores and (sum(faithfulness_scores)/len(faithfulness_scores)) < 0.75:
        print("  ! Low faithfulness. Consider: tighten JSON prompt, reduce K, or upgrade LLM.")
    if ragas and relevancy_scores and (sum(relevancy_scores)/len(relevancy_scores)) < 0.65:
        print("  ! Low answer relevancy. Check if query expansion is distorting short queries.")
    print("  → Run: python eval/self_retrieval.py")
    print("  → Run: python eval/run_eval.py --run-name <description>")
    print()


if __name__ == "__main__":
    run()
```

---

## Eval File Structure

```
backend/
├── eval/
│   ├── ragas_scorer.py        Per-query RAGAS scoring (called from pipeline.py)
│   ├── deepeval_config.py     Local Ollama judge for DeepEval
│   ├── test_rag_regression.py pytest test suite (run in CI)
│   ├── run_eval.py            Full eval set runner → logs to MLflow
│   ├── self_retrieval.py      Zero-cost embedding sanity check
│   └── analyse_feedback.py    Mine feedback.jsonl for trends
└── data/
    ├── eval_set.json          Ground truth: (query, relevant_paper_ids) pairs
    ├── feedback.jsonl         Live ratings + RAGAS scores (append-only log)
    └── mlruns/                MLflow experiment store (auto-created)
```

---

## Recommended Eval Cadence

| When | Script | Purpose |
|---|---|---|
| After every sync | `self_retrieval.py` | Verify embeddings weren't corrupted |
| Weekly | `analyse_feedback.py` | Review quality trend from live traffic |
| Before any prompt/model change | `run_eval.py --run-name <description>` | Regression baseline |
| After any prompt/model change | `run_eval.py --run-name <new-description>` | Compare against baseline in MLflow |
| In CI/CD pipeline | `pytest eval/test_rag_regression.py` | Block bad deployments |
| Continuously (if TruLens enabled) | n/a — passive | Live RAG triad dashboard |

---

## Key Metrics to Track Over Time

Plot these in MLflow or a simple dashboard to watch system quality:

| Metric | Healthy range | Where it comes from |
|---|---|---|
| `hit_rate_at_5` | > 0.75 | `run_eval.py` against ground truth |
| `ndcg_at_5` | > 0.65 | `run_eval.py` against ground truth |
| `faithfulness_mean` | > 0.85 | RAGAS on live queries |
| `answer_relevancy_mean` | > 0.75 | RAGAS on live queries |
| Self-retrieval Hit Rate@5 | > 0.90 | `self_retrieval.py` |
| Thumbs-down rate | < 0.20 | `analyse_feedback.py` |
| Structured output parse rate | > 0.95 | Log in `chain.py` |
| Cache hit rate | Growing over time | Log in `pipeline.py` |
| LLM latency P95 | < 40 s | Logged in `pipeline.py` |

---

## Installing All Eval Dependencies

```bash
pip install ragas deepeval mlflow trulens-eval datasets litellm
```

Add to `requirements.txt` under a comment:

```
# ── Evaluation (optional, not needed to run the app) ─────────────────────────
ragas
deepeval
mlflow
trulens-eval
datasets
litellm
```
