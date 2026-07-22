"""
Full eval-set runner — logs all metrics to a local MLflow experiment.

Reads backend/data/eval_set.json (ground-truth query/paper pairs) and computes:
  - Recall@fetch_k   (the ceiling: can the reranker even see a relevant doc?)
  - Recall@k, Precision@k, MRR@k, nDCG@k on the final result set
  - Hit Rate@k
  - Per-slice breakdown (query_type), with bootstrap CIs
  - P50 / P95 total and retrieval-only latency
  - RAGAS Faithfulness + Answer Relevancy against the REAL retrieved context

Two properties this runner guarantees, both of which were previously violated:

  1. Retrieval metrics are computed over paper_id on both sides.  Comparing
     retrieved titles against ground-truth IDs made every metric 0.0.
  2. The semantic cache is bypassed.  Otherwise the second run of an eval reads
     its own answers back out of the cache and reports cache latency as
     retrieval latency — and pollutes the production cache with eval queries.

Usage:
    cd backend && python eval/run_eval.py
    cd backend && python eval/run_eval.py --run-name "after-reranker-tuning"
    cd backend && python eval/run_eval.py --no-ragas      # retrieval only, fast

eval_set.json format:
    {"queries": [
        {"id": "q1",
         "query": "federated learning for medical imaging",
         "relevant_paper_ids": ["a1b2...", "c3d4..."],
         "query_type": "topic"}          # optional, enables slice breakdown
    ]}

Install: pip install mlflow ragas datasets
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
import config

from eval.metrics import (
    bootstrap_ci,
    hit_rate,
    mean,
    mrr,
    ndcg,
    precision_at_k,
    recall_at_k,
)

EVAL_SET = Path("data/eval_set.json")
MLRUNS = Path("data/mlruns")


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; safe for short lists."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100 * len(ordered))) - 1))
    return ordered[idx]


def evaluate_items(items: list[dict], use_ragas: bool) -> dict:
    """Run the pipeline over the eval set and return raw per-query results."""
    from rag import pipeline
    from rag.reranker import Reranker

    reranker_active = Reranker.get().active
    if not reranker_active:
        print(
            "\n  !! Reranker is NOT active — results below measure fusion order,\n"
            "     not cross-encoder ranking. Set RERANKER_MODEL_NAME to compare.\n"
        )

    rows: list[dict] = []
    for item in items:
        q = item["query"]
        relevant = item.get("relevant_paper_ids", [])
        print(f"  [{item.get('id', '?')}] {q[:70]}…")

        t0 = time.perf_counter()
        # bypass_cache=True: measure retrieval, not the cache.
        result = pipeline.run(q, bypass_cache=True)
        elapsed = (time.perf_counter() - t0) * 1000

        retrieved = [s.get("paper_id", "") for s in result["sources"]]
        answer = result.get("answer", {})

        row = {
            "id": item.get("id", ""),
            "query": q,
            "query_type": item.get("query_type", "unspecified"),
            "retrieved": retrieved,
            "relevant": relevant,
            "latency_ms": elapsed,
            "no_results": bool(answer.get("no_relevant_research")),
            "generation_failed": bool(answer.get("generation_failed")),
        }

        if use_ragas:
            try:
                from eval.ragas_scorer import score_from_sources

                # Score against the ACTUAL retrieved context.  Passing an empty
                # context (as this runner used to) makes faithfulness a
                # meaningless number rather than a missing one.
                row.update(
                    score_from_sources(
                        q, answer.get("landscape_summary", ""), result["sources"]
                    )
                )
            except Exception as exc:
                print(f"      ragas skipped: {exc}")

        rows.append(row)

    return {"rows": rows, "reranker_active": reranker_active}


def aggregate(rows: list[dict], k: int, fetch_k: int) -> dict:
    """Corpus-level metrics.  Missing inputs yield None, never 0.0."""
    scored = [r for r in rows if r["relevant"]]

    def collect(fn, depth):
        return [fn(r["retrieved"], r["relevant"], depth) for r in scored]

    ndcg_values = collect(ndcg, k)
    latencies = [r["latency_ms"] for r in rows]
    faith = [r["faithfulness"] for r in rows if r.get("faithfulness") is not None]
    relev = [
        r["answer_relevancy"] for r in rows if r.get("answer_relevancy") is not None
    ]

    metrics = {
        f"hit_rate_at_{k}": mean(collect(hit_rate, k)),
        f"recall_at_{k}": mean(collect(recall_at_k, k)),
        f"recall_at_{fetch_k}": mean(collect(recall_at_k, fetch_k)),
        f"precision_at_{k}": mean(collect(precision_at_k, k)),
        f"mrr_at_{k}": mean(collect(mrr, k)),
        f"ndcg_at_{k}": mean(ndcg_values),
        "faithfulness_mean": mean(faith),
        "answer_relevancy_mean": mean(relev),
        "no_results_rate": mean([float(r["no_results"]) for r in rows]),
        "generation_failure_rate": mean([float(r["generation_failed"]) for r in rows]),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "eval_set_size": float(len(rows)),
        "labelled_query_count": float(len(scored)),
    }

    ci = bootstrap_ci(ndcg_values)
    if ci:
        metrics[f"ndcg_at_{k}_ci_low"], metrics[f"ndcg_at_{k}_ci_high"] = ci

    return {key: val for key, val in metrics.items() if val is not None}


def slice_breakdown(rows: list[dict], k: int) -> dict[str, dict]:
    """
    Per-query-type metrics.

    A single average hides where search fails.  Author-name queries, exact-title
    lookups and broad topic queries fail for completely different reasons and
    are fixed by completely different changes — so they are measured apart.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["relevant"]:
            buckets[row["query_type"]].append(row)

    out: dict[str, dict] = {}
    for name, group in sorted(buckets.items()):
        values = [ndcg(r["retrieved"], r["relevant"], k) for r in group]
        recalls = [recall_at_k(r["retrieved"], r["relevant"], k) for r in group]
        ci = bootstrap_ci(values)
        out[name] = {
            "n": len(group),
            f"ndcg_at_{k}": mean(values),
            f"recall_at_{k}": mean(recalls),
            "ci": ci,
            "latency_p50_ms": percentile([r["latency_ms"] for r in group], 50),
        }
    return out


def run(run_name: str, use_ragas: bool = True) -> dict:
    try:
        import mlflow
    except ImportError:
        print("MLflow not installed. Run: pip install mlflow")
        sys.exit(1)

    if not EVAL_SET.exists():
        print(
            f"ERROR: {EVAL_SET} not found.\n"
            "Create it by copying data/eval_set.json.example and filling in real entries.\n"
            "Each entry needs relevant_paper_ids (Semantic Scholar paper IDs, NOT titles).\n"
            "See docs/llmops-evaluation.md § Building Ground Truth."
        )
        sys.exit(1)

    items = json.loads(EVAL_SET.read_text()).get("queries", [])
    if not items:
        print("eval_set.json has no queries. Add annotated entries first.")
        sys.exit(1)

    unlabelled = [i for i in items if not i.get("relevant_paper_ids")]
    if unlabelled:
        print(
            f"  note: {len(unlabelled)}/{len(items)} queries have no "
            "relevant_paper_ids and are excluded from ranking metrics."
        )

    print(f"Running eval on {len(items)} queries (run: '{run_name}')…\n")

    k = config.RETRIEVAL_K
    fetch_k = config.RETRIEVAL_FETCH_K

    mlflow.set_tracking_uri(str(MLRUNS))
    mlflow.set_experiment("trace-eval")

    with mlflow.start_run(run_name=run_name):
        outcome = evaluate_items(items, use_ragas)
        rows = outcome["rows"]

        mlflow.log_params(
            {**config.snapshot(), "reranker_active": outcome["reranker_active"]}
        )

        metrics = aggregate(rows, k, fetch_k)
        mlflow.log_metrics(metrics)

        slices = slice_breakdown(rows, k)
        for name, stats in slices.items():
            for metric_name, value in stats.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"slice.{name}.{metric_name}", value)

        _print_report(metrics, slices, k, outcome["reranker_active"])

        # Per-query results, so a regression can be traced to specific queries.
        detail = Path("data/eval_last_run.json")
        detail.write_text(json.dumps({"run": run_name, "rows": rows}, indent=2))
        mlflow.log_artifact(str(detail))
        print(f"  Per-query detail: {detail}")
        print("  View in MLflow:  make mlflow-ui\n")

    return metrics


def _print_report(metrics: dict, slices: dict, k: int, reranker_active: bool) -> None:
    print("\n── Results ───────────────────────────────────────")
    if not reranker_active:
        print(
            "  reranker: INACTIVE (passthrough — ranking metrics reflect fusion order)"
        )
    for name, val in metrics.items():
        bar = "█" * int(val * 20) if 0 <= val <= 1 else ""
        print(f"  {name:<30} {val:>9.4f}  {bar}")

    if slices:
        print("\n── By query type ─────────────────────────────────")
        print(f"  {'slice':<16}{'n':>4}{f'  nDCG@{k}':>12}{'  95% CI':>18}")
        for name, stats in slices.items():
            ci = stats.get("ci")
            ci_text = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "—"
            warn = "  ⚠ small n" if stats["n"] < 30 else ""
            print(
                f"  {name:<16}{stats['n']:>4}{stats[f'ndcg_at_{k}']:>12.4f}"
                f"{ci_text:>18}{warn}"
            )
        if any(s["n"] < 30 for s in slices.values()):
            print(
                "\n  ⚠  Slices under ~30 queries have intervals wide enough that\n"
                "     differences between them are usually sampling noise."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=f"eval-{int(time.time())}")
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Skip LLM-judge metrics (much faster; retrieval metrics unaffected)",
    )
    args = parser.parse_args()
    run(args.run_name, use_ragas=not args.no_ragas)
