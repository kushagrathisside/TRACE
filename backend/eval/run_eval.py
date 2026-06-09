"""
Full eval set runner — logs all metrics to a local MLflow experiment.

Reads backend/data/eval_set.json (ground-truth query/paper pairs) and computes:
  - Hit Rate@K    (did at least one relevant paper appear in top K?)
  - NDCG@K        (rank-aware quality)
  - RAGAS Faithfulness + Answer Relevancy (if ragas installed)
  - P50 / P95 latency

All results are written to data/mlruns so you can compare runs side-by-side:
    make mlflow-ui      # opens http://localhost:5050

Usage:
    cd backend && python eval/run_eval.py
    cd backend && python eval/run_eval.py --run-name "after-reranker-tuning"

Install: pip install mlflow ragas datasets
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import config

EVAL_SET  = Path("data/eval_set.json")
MLRUNS    = Path("data/mlruns")


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def hit_rate(retrieved: list[str], relevant: list[str], k: int) -> float:
    return float(any(pid in retrieved[:k] for pid in relevant))

def ndcg(retrieved: list[str], relevant: list[str], k: int) -> float:
    dcg   = sum(1.0 / math.log2(r + 2) for r, pid in enumerate(retrieved[:k]) if pid in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0

def avg(lst: list) -> float:
    return sum(lst) / len(lst) if lst else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def run(run_name: str) -> None:
    try:
        import mlflow
    except ImportError:
        print("MLflow not installed. Run: pip install mlflow")
        sys.exit(1)

    if not EVAL_SET.exists():
        print(
            f"ERROR: {EVAL_SET} not found.\n"
            "Create it by copying data/eval_set.json.example and filling in real entries.\n"
            "See docs/llmops-evaluation.md § Building Ground Truth."
        )
        sys.exit(1)

    items = json.loads(EVAL_SET.read_text()).get("queries", [])
    if not items:
        print("eval_set.json has no queries. Add annotated entries first.")
        sys.exit(1)

    print(f"Running eval on {len(items)} queries (run: '{run_name}')…\n")

    from rag import pipeline

    mlflow.set_tracking_uri(str(MLRUNS))
    mlflow.set_experiment("trace-eval")

    k = config.RETRIEVAL_K

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "retrieval_k":          k,
            "fetch_k":              config.RETRIEVAL_FETCH_K,
            "embedding_model":      config.EMBEDDING_MODEL_NAME,
            "reranker_model":       config.RERANKER_MODEL_NAME,
            "llm_model":            config.LLM_MODEL_NAME,
            "min_similarity_dist":  config.MIN_SIMILARITY_DISTANCE,
            "recency_weight":       config.RECENCY_WEIGHT,
            "hnsw_M":               config.HNSW_M,
            "hnsw_search_ef":       config.HNSW_SEARCH_EF,
        })

        hit_rates, ndcgs, latencies = [], [], []
        f_scores, r_scores = [], []

        for item in items:
            q = item["query"]
            print(f"  [{item.get('id','?')}] {q[:70]}…")

            t0     = time.perf_counter()
            result = pipeline.run(q)
            elapsed = (time.perf_counter() - t0) * 1000

            retrieved  = [s["title"] for s in result["sources"]]
            relevant   = item.get("relevant_paper_ids", [])

            hit_rates.append(hit_rate(retrieved, relevant, k))
            ndcgs.append(ndcg(retrieved, relevant, k))
            latencies.append(elapsed)

            # Optional RAGAS scoring
            try:
                from eval.ragas_scorer import score_and_log
                from rag.vector_store import VectorStoreManager
                scores = score_and_log(
                    q,
                    result["answer"].get("landscape_summary", ""),
                    [],
                )
                if scores["faithfulness"] is not None:
                    f_scores.append(scores["faithfulness"])
                if scores["answer_relevancy"] is not None:
                    r_scores.append(scores["answer_relevancy"])
            except Exception:
                pass

        lat_sorted = sorted(latencies)
        metrics = {
            f"hit_rate_at_{k}":      avg(hit_rates),
            f"ndcg_at_{k}":          avg(ndcgs),
            "faithfulness_mean":     avg(f_scores),
            "answer_relevancy_mean": avg(r_scores),
            "latency_p50_ms":        lat_sorted[len(lat_sorted) // 2],
            "latency_p95_ms":        lat_sorted[int(len(lat_sorted) * 0.95)],
            "eval_set_size":         len(items),
        }
        mlflow.log_metrics(metrics)

        print(f"\n── Results ───────────────────────────────────────")
        for name, val in metrics.items():
            bar = "█" * int(val * 20) if 0 <= val <= 1 else ""
            print(f"  {name:<30} {val:>7.4f}  {bar}")

        print(f"\n  View in MLflow:  make mlflow-ui\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=f"eval-{int(time.time())}")
    args = parser.parse_args()
    run(args.run_name)
