"""
Unit tests for eval/metrics.py.

The central case here is `test_title_vs_id_mismatch_raises`: the previous eval
runner compared retrieved *titles* against ground-truth *IDs*, so every ranking
metric returned 0.0 on every run and looked like a real (terrible) measurement.
Metrics that cannot fail are worse than absent ones, so the mismatch is now an
error rather than a silent zero.
"""

import os
import sys

import pytest

os.environ.setdefault("ADMIN_PASSWORD", "test")
os.environ.setdefault("CORS_ORIGINS", "*")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.metrics import (  # noqa: E402
    bootstrap_ci,
    hit_rate,
    mean,
    mrr,
    ndcg,
    precision_at_k,
    recall_at_k,
)

RETRIEVED = ["p1", "p2", "p3", "p4", "p5"]


# ── The bug this module exists to prevent ─────────────────────────────────────


def test_title_vs_id_mismatch_raises():
    titles = ["Attention Is All You Need", "Deep Residual Learning For Images"]
    ids = ["649def34f8be52c8b66281af98ae884c09aef38b"]
    with pytest.raises(ValueError, match="identifier types"):
        ndcg(titles, ids, 5)
    with pytest.raises(ValueError, match="identifier types"):
        hit_rate(titles, ids, 5)


def test_matching_id_spaces_do_not_raise():
    assert ndcg(["a1", "b2"], ["a1"], 5) > 0


# ── Ranking metrics ───────────────────────────────────────────────────────────


def test_hit_rate_hit_and_miss():
    assert hit_rate(RETRIEVED, ["p3"], 5) == 1.0
    assert hit_rate(RETRIEVED, ["p3"], 2) == 0.0
    assert hit_rate(RETRIEVED, ["p9"], 5) == 0.0


def test_recall_counts_all_relevant():
    assert recall_at_k(RETRIEVED, ["p1", "p2"], 5) == 1.0
    assert recall_at_k(RETRIEVED, ["p1", "p9"], 5) == 0.5
    assert recall_at_k(RETRIEVED, ["p1", "p2"], 1) == 0.5


def test_recall_at_fetch_depth_is_the_ceiling():
    """Nothing downstream can retrieve what the candidate set never contained."""
    assert recall_at_k(RETRIEVED, ["p9"], 20) == 0.0


def test_precision_at_k():
    assert precision_at_k(RETRIEVED, ["p1", "p2"], 2) == 1.0
    assert precision_at_k(RETRIEVED, ["p1"], 5) == 0.2


def test_mrr_rewards_earlier_hits():
    assert mrr(RETRIEVED, ["p1"], 5) == 1.0
    assert mrr(RETRIEVED, ["p2"], 5) == 0.5
    assert mrr(RETRIEVED, ["p9"], 5) == 0.0


def test_ndcg_is_order_sensitive():
    """The property that makes nDCG the metric a reranker is judged on."""
    good = ndcg(["p1", "p9", "p8"], ["p1"], 3)
    bad = ndcg(["p9", "p8", "p1"], ["p1"], 3)
    assert good > bad
    assert good == 1.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg(["a", "b", "c"], ["a", "b"], 3) == 1.0


def test_ndcg_no_relevant_is_zero():
    assert ndcg(["a", "b"], ["z"], 2) == 0.0


# ── Aggregation hygiene ───────────────────────────────────────────────────────


def test_mean_of_empty_is_none_not_zero():
    """
    "No data" and "scored zero" must not render identically in a metrics chart —
    conflating them made a missing RAGAS install look like total failure.
    """
    assert mean([]) is None
    assert mean([1.0, 0.0]) == 0.5


def test_bootstrap_ci_brackets_the_mean():
    values = [0.8, 0.9, 0.7, 0.85, 0.75, 0.95, 0.6, 0.88]
    ci = bootstrap_ci(values, iterations=500)
    assert ci is not None
    low, high = ci
    assert low <= sum(values) / len(values) <= high


def test_bootstrap_ci_needs_at_least_two_points():
    assert bootstrap_ci([0.5]) is None
