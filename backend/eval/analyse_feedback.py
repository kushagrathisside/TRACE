"""
Mine feedback.jsonl for quality trends.

Reads two types of records from feedback.jsonl:
  - User ratings: {"rating": "up"|"down", "query": "...", "timestamp": "..."}
  - RAGAS scores: {"event": "ragas_score", "faithfulness": 0.9, ...}

Prints:
  - Thumbs-down rate + most downvoted queries
  - RAGAS score averages + low-score counts
  - Actionable suggestions based on thresholds

Usage:
    cd backend && python eval/analyse_feedback.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
import config


def run() -> None:
    path = Path(config.FEEDBACK_LOG_PATH)
    if not path.exists() or path.stat().st_size == 0:
        print("No feedback data yet. Collect queries with the feedback buttons first.")
        return

    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    ratings = [r for r in records if "rating" in r and r.get("event") != "ragas_score"]
    ragas   = [r for r in records if r.get("event") == "ragas_score"]

    print(f"\n── User Ratings {'─'*40}")
    if ratings:
        total = len(ratings)
        downs = sum(1 for r in ratings if r["rating"] == "down")
        ups   = total - downs
        print(f"  Total rated queries : {total}")
        print(f"  Thumbs-up  ✓        : {ups}   ({ups/total:.1%})")
        print(f"  Thumbs-down ✗       : {downs}  ({downs/total:.1%})")
        if downs:
            worst = Counter(r.get("query", "")[:100] for r in ratings if r["rating"] == "down")
            print(f"\n  Top downvoted queries:")
            for q, n in worst.most_common(5):
                print(f"    [{n}×] {q}")
    else:
        print("  No rating records found.")

    print(f"\n── RAGAS Scores {'─'*42}")
    if ragas:
        f_scores = [r["faithfulness"]     for r in ragas if r.get("faithfulness")     is not None]
        r_scores = [r["answer_relevancy"] for r in ragas if r.get("answer_relevancy") is not None]
        print(f"  Scored queries: {len(ragas)}")
        if f_scores:
            avg_f  = sum(f_scores) / len(f_scores)
            low_f  = sum(1 for s in f_scores if s < 0.7)
            trend  = "↓ declining" if len(f_scores) >= 10 and sum(f_scores[-5:])/5 < sum(f_scores[:5])/5 else "→ stable"
            print(f"  Faithfulness     : mean={avg_f:.3f}  low(<0.7)={low_f}  {trend}")
        if r_scores:
            avg_r = sum(r_scores) / len(r_scores)
            low_r = sum(1 for s in r_scores if s < 0.7)
            print(f"  Answer Relevancy : mean={avg_r:.3f}  low(<0.7)={low_r}")
    else:
        print("  No RAGAS scores logged yet.")
        print("  Tip: install ragas and call eval.ragas_scorer.score_and_log() from pipeline.py")

    print(f"\n── Suggestions {'─'*43}")
    issues = False
    if ratings and len(ratings) >= 5:
        down_rate = sum(1 for r in ratings if r["rating"] == "down") / len(ratings)
        if down_rate > 0.3:
            print(f"  ⚠  High thumbs-down rate ({down_rate:.0%}). Run self_retrieval.py to check retrieval quality.")
            issues = True
    if ragas:
        if f_scores and sum(f_scores)/len(f_scores) < 0.75:
            print("  ⚠  Low faithfulness. Try: tighten JSON prompt, reduce RETRIEVAL_K, or upgrade LLM.")
            issues = True
        if r_scores and sum(r_scores)/len(r_scores) < 0.65:
            print("  ⚠  Low answer relevancy. Check if query expansion is distorting short queries.")
            issues = True
    if not issues:
        print("  ✓ All metrics within healthy ranges.")
    print(f"\n  Next steps:")
    print(f"    python eval/self_retrieval.py")
    print(f"    python eval/run_eval.py --run-name diagnosis\n")


if __name__ == "__main__":
    run()
