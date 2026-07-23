"""
Build data/eval_set.json — the labelled query set every offline metric reads.

Ground truth is only as honest as the way it was generated, so each slice below
states what it measures and where it is biased.

Slices
------
exact_title   Query = a paper's exact title.  Ground truth = that paper.
              Unambiguous, and a floor test: if a system cannot retrieve a
              document by its own title, nothing else it reports means much.

author        Query = "<author name> research".  Ground truth = every indexed
              paper that author appears on.  Unambiguous, and the slice that
              exposes whether author names are searchable at all (they were not
              until they were added to the BM25 corpus).

topic         Query = distinctive abstract terms that do NOT appear in the
              paper's title.  Ground truth = that paper.

              ⚠ Known bias, and a cautionary tale.  The first version reworded
              the title.  Measured afterwards, all 39 generated queries had
              100% of their tokens present verbatim in the source title — the
              slice was an exact-title test in disguise, BM25 dominated it for
              that reason alone, and the resulting "topic retrieval" numbers
              described nothing of the sort.  Excluding title tokens forces the
              query onto abstract vocabulary.  Residual leakage remains (the
              abstract is indexed), so these are still silver labels: treat
              cross-slice comparisons as directional and use hand-written
              queries for anything load-bearing.

manual        Hand-written queries, empty labels.  Fill these in yourself —
              they are the only slice with no generation bias, and the
              tiebreaker when silver and hand labels disagree.

Usage:
    cd backend && python eval/build_eval_set.py
    cd backend && python eval/build_eval_set.py --per-slice 40 --no-llm
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
import config

OUT = Path("data/eval_set.json")
#: Hand-judged labels, committed to the repo. Re-applied on every regeneration
#: so that rebuilding the eval set does not silently discard human work — the
#: one input here that no script can reproduce.
LABELS = Path("data/eval_labels.json")

#: Research ideas of the kind students actually bring to an advisor.  These
#: carry no automatic labels — they exercise the live system and populate the
#: trace log for latency and no-result-rate measurement.
MANUAL_QUERIES = [
    "I want to use transformers for low-resource language translation",
    "federated learning for privacy-preserving medical imaging",
    "graph neural networks for molecular property prediction",
    "reinforcement learning for robotic manipulation from demonstrations",
    "interpretability methods for large language model decisions",
    "self-supervised pretraining for video understanding",
    "detecting and mitigating social bias in language models",
    "efficient fine-tuning of large models on a single GPU",
    "using diffusion models for scientific data generation",
    "evaluating factual grounding in retrieval-augmented generation",
    "curriculum learning for sample-efficient reinforcement learning",
    "multimodal models that combine vision and language for robotics",
]


def _abstract_to_idea(doc, max_terms: int = 8) -> str | None:
    """
    Build a topical query from abstract terms that do NOT appear in the title.

    The previous version reworded the title. Measured against the corpus, all
    39 generated queries had **100% of their tokens present verbatim in the
    source title** — so the slice was a title-match test wearing a research-idea
    prefix, and reported nothing about topical retrieval. BM25 dominated it for
    exactly that reason.

    Excluding title tokens forces the query to rely on abstract vocabulary, so
    a hit requires matching body content rather than replaying the headline.
    Leakage is reduced, not eliminated: the abstract is indexed too, so this is
    still a silver label. Hand-written `manual` queries remain the only
    bias-free slice.

    Returns None when the abstract yields too few distinctive terms to make a
    query worth scoring.
    """
    from rag.hybrid_search import tokenize

    title_tokens = set(tokenize(doc.metadata.get("paper_title", "")))
    abstract = doc.page_content.split("Abstract:", 1)[-1]

    seen: set[str] = set()
    terms: list[str] = []
    for token in tokenize(abstract):
        if token in title_tokens or token in seen or len(token) < 4:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= max_terms:
            break

    if len(terms) < 5:
        return None
    return "research on " + " ".join(terms)


def build(per_slice: int, seed: int = 13) -> dict:
    from rag.vector_store import VectorStoreManager

    docs = VectorStoreManager.get_or_create().get_all_documents()
    if not docs:
        print("No documents indexed. Run: python eval/seed_corpus.py --ingest")
        sys.exit(1)

    rng = random.Random(seed)
    queries: list[dict] = []

    # Only papers with a real abstract are usable: a title-only document has
    # nothing for dense retrieval to work with beyond the query itself.
    substantive = [d for d in docs if len(d.page_content) > 300]
    print(f"  {len(docs)} documents indexed, {len(substantive)} with real abstracts")

    # ── exact_title ───────────────────────────────────────────────────────────
    for i, doc in enumerate(rng.sample(substantive, min(per_slice, len(substantive)))):
        queries.append(
            {
                "id": f"title-{i:03d}",
                "query": doc.metadata.get("paper_title", ""),
                "relevant_paper_ids": [doc.metadata.get("paper_id", "")],
                "query_type": "exact_title",
            }
        )

    # ── author ────────────────────────────────────────────────────────────────
    by_author: dict[str, list[str]] = defaultdict(list)
    for doc in docs:
        for name in doc.metadata.get("institute_authors", "").split(","):
            name = name.strip()
            if name:
                by_author[name].append(doc.metadata.get("paper_id", ""))

    for i, (name, paper_ids) in enumerate(sorted(by_author.items())):
        if len(paper_ids) < 3:
            continue
        queries.append(
            {
                "id": f"author-{i:03d}",
                "query": f"{name} research",
                "relevant_paper_ids": paper_ids,
                "query_type": "author",
            }
        )

    # ── topic (silver) ────────────────────────────────────────────────────────
    # Sample generously: _abstract_to_idea rejects abstracts that yield too few
    # distinctive non-title terms.
    pool = rng.sample(substantive, min(per_slice * 3, len(substantive)))
    kept = 0
    for i, doc in enumerate(pool):
        if kept >= per_slice:
            break
        query = _abstract_to_idea(doc)
        if query is None:
            continue
        queries.append(
            {
                "id": f"topic-{i:03d}",
                "query": query,
                "relevant_paper_ids": [doc.metadata.get("paper_id", "")],
                "query_type": "topic",
                "note": (
                    "silver label: abstract terms absent from the title. "
                    "Abstract is indexed, so residual leakage remains."
                ),
            }
        )
        kept += 1

    # ── manual ────────────────────────────────────────────────────────────────
    saved = {}
    label_source = "human"
    if LABELS.exists():
        store = json.loads(LABELS.read_text())
        saved = store.get("labels", {})
        label_source = store.get("label_source", "human")

    for i, text in enumerate(MANUAL_QUERIES):
        # Only keep labels whose papers are still in this corpus build; a
        # re-seeded index can legitimately drop a paper.
        known = {d.metadata.get("paper_id", "") for d in docs}
        relevant = [pid for pid in saved.get(text, []) if pid in known]
        queries.append(
            {
                "id": f"manual-{i:03d}",
                "query": text,
                "relevant_paper_ids": relevant,
                "query_type": "manual",
                **({"label_source": label_source} if relevant else {}),
                "note": (
                    f"hand-written; {label_source}-labelled"
                    if relevant
                    else "hand-written; run `make label-pool` to label this query"
                ),
            }
        )

    return {
        "generated_by": "eval/build_eval_set.py",
        "institute": config.INSTITUTE_NAME,
        "corpus_documents": len(docs),
        "slice_caveats": {
            "topic": (
                "Queries derived from paper titles favour dense retrieval "
                "(near-paraphrase matching). Cross-slice comparisons are "
                "directional only."
            ),
            "manual": "Unlabelled until you add relevant_paper_ids.",
        },
        "queries": queries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the labelled eval set")
    parser.add_argument("--per-slice", type=int, default=40)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    data = build(args.per_slice, seed=args.seed)

    counts: dict[str, int] = defaultdict(int)
    for q in data["queries"]:
        counts[q["query_type"]] += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2))

    print(f"\n  Wrote {len(data['queries'])} queries to {OUT}")
    for name, count in sorted(counts.items()):
        flag = "  ⚠ under 30 — CIs will be wide" if count < 30 else ""
        print(f"    {name:<14} {count:>4}{flag}")
    print(
        "\n  Next: python eval/run_eval.py --no-ragas"
        "\n        python eval/ablate.py --markdown\n"
    )


if __name__ == "__main__":
    main()
