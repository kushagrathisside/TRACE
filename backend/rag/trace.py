"""
Per-query retrieval tracing — the substrate every offline metric is built on.

Why this exists
---------------
Before this module the system logged one line per query containing total
latency and the first 80 characters of the query.  That is enough to know a
query was slow and nothing else.  In particular it was impossible to answer:

  - Which documents did each stage actually return?
  - Did the cross-encoder run, or did it silently fall back to a passthrough?
  - Was this answer served from the semantic cache?
  - Which stage consumed the latency?
  - Which retrieval config produced this result?

A thumbs-down is only actionable if you can reconstruct all of the above, so
each query writes one JSON record to data/retrieval_traces.jsonl keyed by a
`query_id` that is also returned to the browser and echoed back on feedback.

Format: one JSON object per line (JSONL), append-only.  Read it with
eval/analyse_feedback.py, or load it into a DataFrame for slicing.
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)

# Appends are serialised: writes come from the run_in_executor thread pool, and
# interleaved partial lines would corrupt the JSONL.
_write_lock = threading.Lock()


class StageTimer:
    """
    Accumulates per-stage wall-clock latency for a single query.

        timer = StageTimer()
        with timer("dense_search"):
            ...
        timer.stages  # {"dense_search": 18.3}

    Stage latency is what turns "p95 is 12 s" into "p95 is 12 s and 11.8 s of
    it is LLM generation", which is the difference between a number and a
    decision.
    """

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._t0 = time.perf_counter()

    def __call__(self, name: str) -> "_StageContext":
        return _StageContext(self, name)

    def record(self, name: str, ms: float) -> None:
        self.stages[name] = round(self.stages.get(name, 0.0) + ms, 2)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 2)

    @property
    def retrieval_ms(self) -> float:
        """
        Everything except LLM calls — the number a search team owns.

        Generation latency is dominated by the model and hardware; retrieval
        latency is dominated by your index parameters.  Reporting them together
        hides every tuning decision you make.
        """
        llm_stages = {"query_expansion", "generation", "ragas"}
        return round(
            sum(ms for name, ms in self.stages.items() if name not in llm_stages), 2
        )


class _StageContext:
    def __init__(self, timer: StageTimer, name: str) -> None:
        self._timer = timer
        self._name = name

    def __enter__(self) -> "_StageContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self._timer.record(self._name, (time.perf_counter() - self._start) * 1000)


def write(record: dict) -> None:
    """Append one trace record.  Never raises — tracing must not break a query."""
    try:
        record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        path = Path(config.TRACE_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str)
        with _write_lock:
            with open(path, "a") as f:
                f.write(line + "\n")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Trace write failed (non-fatal): {exc}")


def doc_ids(docs) -> list[str]:
    """Extract paper_ids from Documents or (score, Document) / (Document, score) pairs."""
    out: list[str] = []
    for item in docs:
        doc = item
        if isinstance(item, tuple):
            doc = next((x for x in item if hasattr(x, "metadata")), None)
        if doc is not None and hasattr(doc, "metadata"):
            out.append(doc.metadata.get("paper_id", ""))
    return out
