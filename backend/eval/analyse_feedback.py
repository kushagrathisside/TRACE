"""
Mine feedback.jsonl and retrieval_traces.jsonl for quality trends.

Reads:
  - User ratings:  {"query_id": "...", "rating": "up"|"down", "query": "..."}
  - RAGAS scores:  {"event": "ragas_score", "query_id": "...", "faithfulness": ...}
  - Traces:        one record per query (see rag/trace.py)

Ratings are joined to traces on `query_id`, which is what makes them
actionable: a thumbs-down can then be attributed to a cache hit, an inactive
reranker, an empty result set, or a specific retrieved document set — rather
than only to the text of the query.

`analyse()` returns structured data; `format_report()` renders it.  The API
route calls both.  (It previously captured printed output by swapping
sys.stdout globally, which interleaves between concurrent requests.)

Usage:
    cd backend && python eval/analyse_feedback.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
import config


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a partially written trailing line
    return records


def analyse() -> dict:
    """Compute feedback and trace statistics.  Pure — no printing."""
    records = _read_jsonl(Path(config.FEEDBACK_LOG_PATH))
    traces = _read_jsonl(Path(config.TRACE_LOG_PATH))
    traces_by_id = {t["query_id"]: t for t in traces if t.get("query_id")}

    ratings = [r for r in records if "rating" in r and r.get("event") != "ragas_score"]
    ragas = [r for r in records if r.get("event") == "ragas_score"]

    report: dict = {
        "ratings": {"total": len(ratings)},
        "ragas": {"scored": len(ragas)},
        "traces": {"total": len(traces)},
        "suggestions": [],
    }

    # ── User ratings ──────────────────────────────────────────────────────────
    if ratings:
        downs = [r for r in ratings if r["rating"] == "down"]
        report["ratings"].update(
            {
                "up": len(ratings) - len(downs),
                "down": len(downs),
                "down_rate": len(downs) / len(ratings),
                "linked_to_trace": sum(
                    1 for r in ratings if r.get("query_id") in traces_by_id
                ),
                "top_downvoted": Counter(
                    r.get("query", "")[:100] for r in downs
                ).most_common(5),
            }
        )

        # Attribute negatives to pipeline state — the reason query_id exists.
        linked = [
            traces_by_id[r["query_id"]]
            for r in downs
            if r.get("query_id") in traces_by_id
        ]
        if linked:
            report["ratings"]["down_attribution"] = {
                "from_cache": sum(1 for t in linked if t.get("cached")),
                "no_results": sum(1 for t in linked if t.get("no_results")),
                "reranker_inactive": sum(
                    1 for t in linked if t.get("reranker_active") is False
                ),
                "generation_failed": sum(
                    1
                    for t in linked
                    if (t.get("grounding") or {}).get("papers_cited") == 0
                ),
            }

    # ── RAGAS ─────────────────────────────────────────────────────────────────
    f_scores = [r["faithfulness"] for r in ragas if r.get("faithfulness") is not None]
    r_scores = [
        r["answer_relevancy"] for r in ragas if r.get("answer_relevancy") is not None
    ]
    if f_scores:
        report["ragas"]["faithfulness_mean"] = sum(f_scores) / len(f_scores)
        report["ragas"]["faithfulness_low_count"] = sum(1 for s in f_scores if s < 0.7)
    if r_scores:
        report["ragas"]["answer_relevancy_mean"] = sum(r_scores) / len(r_scores)
        report["ragas"]["answer_relevancy_low_count"] = sum(
            1 for s in r_scores if s < 0.7
        )

    # ── Trace-derived operational metrics ─────────────────────────────────────
    if traces:
        latencies = sorted(t.get("latency_ms", 0) for t in traces)
        retrieval = sorted(
            t["retrieval_ms"] for t in traces if t.get("retrieval_ms") is not None
        )
        cached = [t for t in traces if t.get("cached")]
        grounded = [t.get("grounding") for t in traces if t.get("grounding")]
        cited = sum(g.get("papers_cited", 0) for g in grounded)
        dropped = sum(g.get("papers_dropped", 0) for g in grounded)
        people_cited = sum(g.get("people_cited", 0) for g in grounded)
        people_dropped = sum(g.get("people_dropped", 0) for g in grounded)

        report["traces"].update(
            {
                "cache_hit_rate": len(cached) / len(traces),
                "no_results_rate": sum(1 for t in traces if t.get("no_results"))
                / len(traces),
                "reranker_inactive_rate": sum(
                    1 for t in traces if t.get("reranker_active") is False
                )
                / len(traces),
                "latency_p50_ms": latencies[len(latencies) // 2],
                "latency_p95_ms": latencies[int(len(latencies) * 0.95)]
                if len(latencies) > 1
                else latencies[0],
                "retrieval_p50_ms": retrieval[len(retrieval) // 2]
                if retrieval
                else None,
                "paper_grounding_rate": (cited - dropped) / cited if cited else None,
                "person_grounding_rate": (people_cited - people_dropped) / people_cited
                if people_cited
                else None,
            }
        )

    report["suggestions"] = _suggest(report)
    return report


def _suggest(report: dict) -> list[str]:
    out: list[str] = []
    ratings, traces = report["ratings"], report["traces"]

    if traces.get("reranker_inactive_rate", 0) > 0:
        out.append(
            f"Reranker was inactive for {traces['reranker_inactive_rate']:.0%} of "
            "queries — ranking fell back to fusion order. Check RERANKER_MODEL_NAME "
            "and /health."
        )
    if ratings.get("total", 0) >= 5 and ratings.get("down_rate", 0) > 0.3:
        out.append(
            f"High thumbs-down rate ({ratings['down_rate']:.0%}). Check "
            "down_attribution below to see whether these were cache hits, empty "
            "result sets, or genuine ranking misses."
        )
    if ratings.get("total") and ratings.get("linked_to_trace", 0) < ratings["total"]:
        missing = ratings["total"] - ratings["linked_to_trace"]
        out.append(
            f"{missing} rating(s) have no matching trace — they came from an older "
            "client or a forged request and cannot be attributed."
        )
    if traces.get("no_results_rate", 0) > 0.2:
        out.append(
            f"No-results rate is {traces['no_results_rate']:.0%}. Either the index "
            "is stale or MIN_SIMILARITY_DISTANCE is too tight."
        )
    grounding = traces.get("paper_grounding_rate")
    if grounding is not None and grounding < 0.95:
        out.append(
            f"Paper grounding rate {grounding:.0%} — the model is citing papers "
            "that are not in the retrieved context."
        )
    person_grounding = traces.get("person_grounding_rate")
    if person_grounding is not None and person_grounding < 0.95:
        out.append(
            f"Person grounding rate {person_grounding:.0%} — suggested people are "
            "being dropped as ungrounded. Highest-severity failure mode; inspect "
            "traces directly."
        )
    faith = report["ragas"].get("faithfulness_mean")
    if faith is not None and faith < 0.75:
        out.append(
            "Low faithfulness. Note the judge is a small local model — confirm "
            "against hand-labelled examples before acting."
        )
    return out


def format_report(report: dict) -> str:
    """Render analyse() output as the plain-text report."""
    lines: list[str] = []
    add = lines.append

    add(f"\n── User Ratings {'─' * 40}")
    ratings = report["ratings"]
    if ratings.get("total"):
        add(f"  Total rated queries : {ratings['total']}")
        add(
            f"  Thumbs-up  ✓        : {ratings['up']}   ({1 - ratings['down_rate']:.1%})"
        )
        add(f"  Thumbs-down ✗       : {ratings['down']}  ({ratings['down_rate']:.1%})")
        add(f"  Linked to a trace   : {ratings['linked_to_trace']}/{ratings['total']}")
        attribution = ratings.get("down_attribution")
        if attribution:
            add("\n  Thumbs-down attribution:")
            for key, value in attribution.items():
                add(f"    {key:<20} {value}")
        if ratings.get("top_downvoted"):
            add("\n  Top downvoted queries:")
            for query, count in ratings["top_downvoted"]:
                add(f"    [{count}×] {query}")
    else:
        add("  No rating records found.")

    add(f"\n── Pipeline (from traces) {'─' * 31}")
    traces = report["traces"]
    if traces.get("total"):
        add(f"  Queries traced      : {traces['total']}")
        for key in (
            "cache_hit_rate",
            "no_results_rate",
            "reranker_inactive_rate",
            "paper_grounding_rate",
            "person_grounding_rate",
        ):
            if traces.get(key) is not None:
                add(f"  {key:<20}: {traces[key]:.1%}")
        add(
            f"  latency p50 / p95   : {traces['latency_p50_ms']:.0f} / {traces['latency_p95_ms']:.0f} ms"
        )
        if traces.get("retrieval_p50_ms") is not None:
            add(f"  retrieval p50       : {traces['retrieval_p50_ms']:.0f} ms")
    else:
        add("  No traces yet — run some queries first.")

    add(f"\n── RAGAS Scores {'─' * 42}")
    ragas = report["ragas"]
    if ragas.get("scored"):
        add(f"  Scored queries: {ragas['scored']}")
        if ragas.get("faithfulness_mean") is not None:
            add(
                f"  Faithfulness     : mean={ragas['faithfulness_mean']:.3f}  "
                f"low(<0.7)={ragas['faithfulness_low_count']}"
            )
        if ragas.get("answer_relevancy_mean") is not None:
            add(
                f"  Answer Relevancy : mean={ragas['answer_relevancy_mean']:.3f}  "
                f"low(<0.7)={ragas['answer_relevancy_low_count']}"
            )
    else:
        add("  No RAGAS scores logged yet (set ENABLE_RAGAS_SCORING=true).")

    add(f"\n── Suggestions {'─' * 43}")
    if report["suggestions"]:
        for suggestion in report["suggestions"]:
            add(f"  ⚠  {suggestion}")
    else:
        add("  ✓ All metrics within healthy ranges.")
    add("\n  Next steps:")
    add("    python eval/self_retrieval.py")
    add("    python eval/run_eval.py --run-name diagnosis")
    add("    python eval/ablate.py --markdown\n")
    return "\n".join(lines)


def run() -> None:
    report = analyse()
    if not report["ratings"]["total"] and not report["traces"]["total"]:
        print("No feedback or trace data yet. Run some queries first.")
        return
    print(format_report(report))


if __name__ == "__main__":
    run()
