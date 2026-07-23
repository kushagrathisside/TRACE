"""
Stage-wise ablation — measures the marginal contribution of each retrieval stage.

Produces the table that justifies the architecture:

    stage                 Recall@20   Recall@5   nDCG@5   MRR@5   retr. p50
    dense only               0.821      0.610    0.680    0.702      18 ms
    + BM25 (RRF)             0.904      0.641    0.712    0.735      24 ms
    + cross-encoder          0.904      0.783    0.871    0.884      71 ms

Read it column by column, not row by row:

  * **Recall@20 is flat across the last two rows by construction.**  The
    cross-encoder reorders the same 20 candidates it was given, so it cannot
    change what is in the top 20.  A table that only reports Recall@fetch_k
    will always show the reranker contributing nothing — which is a property of
    the metric, not a finding about the reranker.
  * The reranker's contribution lives entirely at the **truncation depth**:
    Recall@5, nDCG@5, MRR@5.  That is where reordering turns into what the user
    actually sees.
  * BM25 does the opposite — it raises the candidate ceiling (Recall@20) and
    barely moves ordering on its own.

So: BM25 buys recall, the cross-encoder converts recall into precision, and the
latency column prices each one.

Usage:
    cd backend && python eval/ablate.py                    # all configurations
    cd backend && python eval/ablate.py --only dense hybrid
    cd backend && python eval/ablate.py --sweep fetch_k --values 5,10,20,40,80
    cd backend && python eval/ablate.py --markdown          # paste-ready table
"""

import argparse
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
import config

from eval.metrics import mean, mrr, ndcg, recall_at_k
from eval.run_eval import percentile

EVAL_SET = Path("data/eval_set.json")


# ── Configuration variants ────────────────────────────────────────────────────
# Each entry is (label, config overrides).  Ordered so each row adds exactly one
# component to the row above it — the whole point is attributing a delta to a
# single change.
VARIANTS: dict[str, dict] = {
    "dense": {
        "RETRIEVAL_MODE": "dense",
        "RERANKER_MODEL_NAME": "",
    },
    "bm25": {
        "RETRIEVAL_MODE": "bm25",
        "RERANKER_MODEL_NAME": "",
    },
    "hybrid": {
        "RETRIEVAL_MODE": "hybrid",
        "RERANKER_MODEL_NAME": "",
    },
    # Reranker model is pinned explicitly in both variants so the comparison
    # survives a change to the config default.
    "hybrid+ce": {
        "RETRIEVAL_MODE": "hybrid",
        "RERANKER_MODEL_NAME": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    },
    # ms-marco is trained on short search-log queries. TRACE receives long
    # natural-language research ideas, which is the shape on which the
    # cross-encoder regressed against hand-written labels. bge-reranker-base is
    # trained on more varied and longer query/passage pairs.
    "hybrid+bge": {
        "RETRIEVAL_MODE": "hybrid",
        "RERANKER_MODEL_NAME": "BAAI/bge-reranker-base",
    },
}


def _apply(overrides: dict) -> dict:
    """
    Apply config overrides in-process and reset the affected singletons.

    Returns the previous values so the caller can restore them.  The reranker
    singleton caches a loaded model keyed to nothing, so it must be dropped
    whenever RERANKER_MODEL_NAME changes or the next variant silently reuses
    the previous model.
    """
    from rag.hybrid_search import HybridSearcher
    from rag.reranker import Reranker

    previous = {key: getattr(config, key) for key in overrides}
    for key, value in overrides.items():
        setattr(config, key, value)

    Reranker._instance = None
    # HybridSearcher holds only the BM25 index, which does not depend on any
    # overridden value; rebuilding it per variant would dominate runtime.
    _ = HybridSearcher
    return previous


def _score_variant(label: str, overrides: dict, items: list[dict]) -> dict:
    from rag import pipeline
    from rag.reranker import Reranker

    previous = _apply(overrides)
    try:
        active = Reranker.get().active
        print(f"\n▶ {label}  (reranker_active={active})")

        recalls_fetch, recalls_k, ndcgs, mrrs = [], [], [], []
        # Recall@k is only comparable across queries with similar label counts.
        # Author queries carry a median of 109 relevant papers, so their Recall@5
        # is structurally capped at 5/109 ≈ 0.05; averaging that together with
        # single-label queries (where Recall@5 is effectively 0 or 1) produces a
        # number that is neither. Single-label recall is tracked separately.
        recalls_k_single = []
        retrieval_latencies, total_latencies = [], []
        # Per-slice, because the aggregate hides the interesting part: lexical
        # slices (exact_title, author) and semantic ones (topic) reward
        # completely different retrievers, and a corpus average of the two is a
        # number no design decision can be made from.
        per_slice: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"ndcg": [], "recall_k": [], "recall_fetch": []}
        )

        for item in items:
            relevant = item.get("relevant_paper_ids", [])
            if not relevant:
                continue
            t0 = time.perf_counter()
            # retrieve_only: retrieval metrics depend only on which documents
            # come back, and generation costs 10-30 s per query without being
            # able to change any of them.
            result = pipeline.retrieve_only(item["query"])
            total_latencies.append((time.perf_counter() - t0) * 1000)

            retrieved = [s.get("paper_id", "") for s in result["sources"]]
            # Candidate-depth recall comes from the pre-rerank fusion list.
            recalls_fetch.append(
                recall_at_k(result["candidate_ids"], relevant, config.RETRIEVAL_FETCH_K)
            )
            recalls_k.append(recall_at_k(retrieved, relevant, config.RETRIEVAL_K))
            if len(relevant) == 1:
                recalls_k_single.append(recalls_k[-1])
            ndcgs.append(ndcg(retrieved, relevant, config.RETRIEVAL_K))
            mrrs.append(mrr(retrieved, relevant, config.RETRIEVAL_K))

            bucket = per_slice[item.get("query_type", "unspecified")]
            bucket["ndcg"].append(ndcgs[-1])
            bucket["recall_k"].append(recalls_k[-1])
            bucket["recall_fetch"].append(recalls_fetch[-1])

            retrieval_latencies.append(result["retrieval_ms"])

        return {
            "label": label,
            "reranker_active": active,
            "n": len(ndcgs),
            f"recall_at_{config.RETRIEVAL_FETCH_K}": mean(recalls_fetch),
            f"recall_at_{config.RETRIEVAL_K}": mean(recalls_k),
            f"recall_at_{config.RETRIEVAL_K}_single": mean(recalls_k_single),
            "n_single_label": len(recalls_k_single),
            f"ndcg_at_{config.RETRIEVAL_K}": mean(ndcgs),
            f"mrr_at_{config.RETRIEVAL_K}": mean(mrrs),
            "retrieval_p50_ms": percentile(retrieval_latencies, 50),
            "total_p50_ms": percentile(total_latencies, 50),
            "total_p95_ms": percentile(total_latencies, 95),
            "slices": {
                name: {
                    "n": len(vals["ndcg"]),
                    "ndcg": mean(vals["ndcg"]),
                    "recall_k": mean(vals["recall_k"]),
                    "recall_fetch": mean(vals["recall_fetch"]),
                }
                for name, vals in sorted(per_slice.items())
            },
        }
    finally:
        _apply(previous)


# ── Sweeps ────────────────────────────────────────────────────────────────────

SWEEPABLE = {
    "fetch_k": "RETRIEVAL_FETCH_K",
    "search_ef": "HNSW_SEARCH_EF",
    "cache_distance": "CACHE_HIT_DISTANCE",
    "min_similarity": "MIN_SIMILARITY_DISTANCE",
    "recency_weight": "RECENCY_WEIGHT",
    "rrf_weight_sparse": "RRF_WEIGHT_SPARSE",
    "rerank_blend": "RERANK_BLEND",
}


def _parse_values(raw: str, attr: str) -> list:
    caster = int if isinstance(getattr(config, attr), int) else float
    return [caster(v.strip()) for v in raw.split(",") if v.strip()]


# ── Reporting ─────────────────────────────────────────────────────────────────


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}" if value <= 1 else f"{value:.0f}"
    return str(value)


def print_table(results: list[dict], markdown: bool = False) -> None:
    fetch_k, k = config.RETRIEVAL_FETCH_K, config.RETRIEVAL_K
    columns = [
        ("variant", "label"),
        ("n", "n"),
        (f"Recall@{fetch_k}", f"recall_at_{fetch_k}"),
        (f"Recall@{k}", f"recall_at_{k}"),
        (f"R@{k} 1-lbl", f"recall_at_{k}_single"),
        (f"nDCG@{k}", f"ndcg_at_{k}"),
        (f"MRR@{k}", f"mrr_at_{k}"),
        ("retr p50", "retrieval_p50_ms"),
        ("+expand p50", "total_p50_ms"),
    ]

    if markdown:
        print("\n| " + " | ".join(c[0] for c in columns) + " |")
        print("|" + "|".join("---" for _ in columns) + "|")
        for row in results:
            print("| " + " | ".join(_fmt(row.get(key)) for _, key in columns) + " |")
    else:
        header = "".join(f"{name:>14}" for name, _ in columns)
        print("\n" + header)
        print("-" * len(header))
        for row in results:
            print("".join(f"{_fmt(row.get(key)):>14}" for _, key in columns))

    _print_slices(results, markdown=markdown)

    print(
        f"\n  Recall@{fetch_k} is identical across reranked and non-reranked rows by\n"
        f"  construction — the cross-encoder reorders the same {fetch_k} candidates.\n"
        f"  Its contribution appears at depth {k}: Recall@{k}, nDCG@{k}, MRR@{k}.\n"
        f"\n  Recall@{k} mixes queries with 1 relevant paper and queries with ~100;\n"
        f"  'R@{k} 1-lbl' restricts it to single-label queries and is the\n"
        f"  comparable column. nDCG and MRR are normalised and safe to compare.\n"
        f"\n  '+expand p50' is retrieval plus LLM query expansion — NOT end-to-end\n"
        f"  latency. Generation is not run here and adds seconds on top."
    )


def _print_slices(results: list[dict], markdown: bool = False) -> None:
    """nDCG@k per query slice, one column per variant."""
    slice_names: list[str] = []
    for row in results:
        for name in row.get("slices") or {}:
            if name not in slice_names:
                slice_names.append(name)
    if not slice_names:
        return

    k = config.RETRIEVAL_K
    print(f"\n  nDCG@{k} by query slice")
    header = ["slice", "n"] + [r["label"] for r in results]
    if markdown:
        print("\n| " + " | ".join(header) + " |")
        print("|" + "|".join("---" for _ in header) + "|")
    else:
        print("".join(f"{h:>14}" for h in header))

    for name in slice_names:
        first = next(
            (r["slices"][name] for r in results if name in (r.get("slices") or {})),
            None,
        )
        cells = [name, str(first["n"] if first else 0)]
        for row in results:
            entry = (row.get("slices") or {}).get(name)
            cells.append(_fmt(entry["ndcg"]) if entry else "—")
        if markdown:
            print("| " + " | ".join(cells) + " |")
        else:
            print("".join(f"{c:>14}" for c in cells))

    small = [
        n
        for n in slice_names
        if (
            next(
                (r["slices"][n]["n"] for r in results if n in (r.get("slices") or {})),
                0,
            )
        )
        < 30
    ]
    if small:
        print(
            f"\n  ⚠ slices under 30 queries ({', '.join(small)}): differences here\n"
            "    are usually sampling noise, not signal."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-wise retrieval ablation")
    parser.add_argument("--only", nargs="*", choices=list(VARIANTS), default=None)
    parser.add_argument("--sweep", choices=list(SWEEPABLE), default=None)
    parser.add_argument("--values", default="", help="Comma-separated sweep values")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--out", default="data/ablation_results.json")
    args = parser.parse_args()

    if not EVAL_SET.exists():
        print(f"ERROR: {EVAL_SET} not found — ablation needs labelled queries.")
        sys.exit(1)
    items = json.loads(EVAL_SET.read_text()).get("queries", [])
    labelled = [i for i in items if i.get("relevant_paper_ids")]
    if not labelled:
        print("ERROR: no queries in eval_set.json have relevant_paper_ids.")
        sys.exit(1)

    # Warm the BM25 index once; every variant reuses it.
    importlib.import_module("rag.pipeline").build_bm25_on_startup()

    results: list[dict] = []
    if args.sweep:
        attr = SWEEPABLE[args.sweep]
        if not args.values:
            print(f"--sweep {args.sweep} requires --values")
            sys.exit(1)
        for value in _parse_values(args.values, attr):
            results.append(
                _score_variant(f"{args.sweep}={value}", {attr: value}, labelled)
            )
    else:
        for label in args.only or list(VARIANTS):
            results.append(_score_variant(label, VARIANTS[label], labelled))

    print_table(results, markdown=args.markdown)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"queries": len(labelled), "config": config.snapshot(), "results": results},
            indent=2,
        )
    )
    print(f"\n  Raw results: {out}")


if __name__ == "__main__":
    main()
