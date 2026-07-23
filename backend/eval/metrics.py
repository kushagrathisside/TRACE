"""
Ranking metrics, kept separate from the runners so they are unit-testable.

Every function takes `retrieved` as a ranked list of **paper IDs** and
`relevant` as a set/list of ground-truth **paper IDs**.  This is deliberate and
load-bearing: the previous eval runner passed paper *titles* as `retrieved` and
compared them against ground-truth *IDs*, so no element could ever match and
Hit Rate / nDCG evaluated to 0.0 on every run regardless of retrieval quality.

A metric that cannot fail is worse than no metric — it looks like evidence.
So each function here rejects obviously mistyped input rather than silently
returning zero.
"""

import math

#: A paper ID that looks like prose is almost certainly a title.  Ground-truth
#: IDs from Semantic Scholar are 40-char hex SHAs; titles contain spaces.
_TITLE_LIKE_SPACES = 3


def _check_ids(retrieved: list[str], relevant: list[str]) -> None:
    """Guard against the title-vs-ID mixup that silently zeroed every metric."""

    def looks_like_title(values: list[str]) -> bool:
        sample = [v for v in values if v][:5]
        if not sample:
            return False
        return all(v.count(" ") >= _TITLE_LIKE_SPACES for v in sample)

    if looks_like_title(retrieved) != looks_like_title(relevant):
        raise ValueError(
            "retrieved and relevant appear to be different identifier types "
            "(one looks like titles, the other like IDs). Ranking metrics "
            "computed across mismatched key spaces are always 0.0 — pass "
            "paper_id for both."
        )


def hit_rate(retrieved: list[str], relevant: list[str], k: int) -> float:
    """1.0 if any relevant document appears in the top-k, else 0.0."""
    _check_ids(retrieved, relevant)
    rel = set(relevant)
    return float(any(pid in rel for pid in retrieved[:k]))


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Fraction of all relevant documents found in the top-k.

    Recall at the *fetch* depth is the ceiling on everything downstream — no
    reranker can promote a document that retrieval never returned.  Recall at
    the *final* depth is what the user actually sees.  Report both.
    """
    _check_ids(retrieved, relevant)
    rel = set(relevant)
    if not rel:
        return 0.0
    return len([pid for pid in retrieved[:k] if pid in rel]) / len(rel)


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    _check_ids(retrieved, relevant)
    rel = set(relevant)
    if k <= 0 or not retrieved:
        return 0.0
    return len([pid for pid in retrieved[:k] if pid in rel]) / min(k, len(retrieved))


def mrr(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Reciprocal rank of the first relevant hit — sensitive to ordering."""
    _check_ids(retrieved, relevant)
    rel = set(relevant)
    for i, pid in enumerate(retrieved[:k]):
        if pid in rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Binary-gain nDCG@k against an ideal ranking of all relevant documents."""
    _check_ids(retrieved, relevant)
    rel = set(relevant)
    dcg = sum(
        1.0 / math.log2(r + 2) for r, pid in enumerate(retrieved[:k]) if pid in rel
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal > 0 else 0.0


def mean(values: list[float]) -> float | None:
    """
    Mean, or None when there is nothing to average.

    Returning 0.0 for an empty list made "the judge was not installed"
    indistinguishable from "the system scored zero" in the MLflow charts.
    """
    return sum(values) / len(values) if values else None


def bootstrap_ci(
    values: list[float],
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float] | None:
    """
    Percentile bootstrap confidence interval for a mean.

    Slice-level tables are the point of this eval, and slices are small: with
    20-30 queries per slice a 0.05 nDCG gap between two slices is usually
    sampling noise.  Reporting the interval keeps that from being read as a
    finding.
    """
    if len(values) < 2:
        return None
    import random

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((1 - confidence) / 2 * iterations)]
    hi = means[int((1 + confidence) / 2 * iterations) - 1]
    return (lo, hi)
