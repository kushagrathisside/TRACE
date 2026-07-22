"""
Seed TRACE-Institute with real, well-known AI/ML researchers and their papers.

Why real authors instead of synthetic data
------------------------------------------
Retrieval quality is a property of a corpus, not of code.  Synthetic abstracts
are lexically clean and topically separable, so every retriever scores well on
them and the ablation shows no differences.  Real publication records have the
messy properties that make hybrid retrieval worth having: near-duplicate titles,
shared jargon across subfields, author-name queries, and abstracts written in
several different styles.

The people below are public figures in AI/ML, registered here as stand-in
"institute members" purely so the pipeline has a realistic corpus to retrieve
over.  Semantic Scholar author IDs are resolved by name at runtime rather than
hardcoded, since IDs change as profiles are merged.

Memory
------
Ingestion is bounded by config.MAX_PAPERS_PER_PERSON (default 200) and papers
are embedded in batches, so peak memory stays flat regardless of how prolific
the authors are.  With the default roster this produces a corpus in the low
thousands of documents — a few hundred MB of Chroma index, comfortable on an
8 GB machine.

Usage:
    cd backend && python eval/seed_corpus.py                 # resolve + register
    cd backend && python eval/seed_corpus.py --ingest        # …and sync papers
    cd backend && python eval/seed_corpus.py --cap 100 --ingest
"""

import argparse
import sys
import time

sys.path.insert(0, ".")
import config
from ingestion import people_registry, scholar_client

# (display name, department, role) — the department/role strings are what the
# answer surfaces as "people to consult", so they are filled in plausibly.
ROSTER: list[tuple[str, str, str]] = [
    ("Yoshua Bengio", "Deep Learning", "faculty"),
    ("Geoffrey Hinton", "Deep Learning", "faculty"),
    ("Yann LeCun", "Computer Vision", "faculty"),
    ("Fei-Fei Li", "Computer Vision", "faculty"),
    ("Christopher Manning", "Natural Language Processing", "faculty"),
    ("Percy Liang", "Natural Language Processing", "faculty"),
    ("Pieter Abbeel", "Robotics and RL", "faculty"),
    ("Sergey Levine", "Robotics and RL", "faculty"),
    ("Jure Leskovec", "Graph Learning", "faculty"),
    ("Been Kim", "Interpretability", "faculty"),
    ("Ian Goodfellow", "Generative Models", "faculty"),
    ("Emily M. Bender", "Computational Linguistics", "faculty"),
]


def _name_key(name: str) -> tuple[str, str]:
    """(surname, first initial), lowercased and stripped of punctuation."""
    parts = [p.strip(".,").lower() for p in name.split() if p.strip(".,")]
    if not parts:
        return ("", "")
    return (parts[-1], parts[0][:1])


def resolve(name: str) -> dict | None:
    """
    Find the Semantic Scholar author record for a name.

    Matching on the exact display string picks the wrong profile: Semantic
    Scholar's canonical record for "Geoffrey Hinton" is filed under
    "Geoffrey E. Hinton", so an exact-string filter finds only the sparse
    duplicate profiles (5 papers instead of ~700) and the resulting corpus is
    almost empty.

    Match on (surname, first initial) instead, then take the profile with the
    most papers — duplicates are always the thin ones.
    """
    try:
        results = scholar_client.search_author(name)
    except Exception as exc:
        print(f"    ! lookup failed for {name}: {exc}")
        return None

    target = _name_key(name)
    compatible = [r for r in results if _name_key(r.get("name", "")) == target]
    candidates = compatible or results
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.get("paperCount") or 0)


def seed(cap: int | None = None) -> list[dict]:
    if cap is not None:
        config.MAX_PAPERS_PER_PERSON = cap

    existing = {p["name"] for p in people_registry.get_all()}
    registered: list[dict] = []

    print(f"Registering {len(ROSTER)} researchers with {config.INSTITUTE_NAME}…\n")
    for name, department, role in ROSTER:
        if name in existing:
            print(f"  = {name} (already registered)")
            continue

        author = resolve(name)
        if not author:
            print(f"  ✗ {name}: no Semantic Scholar match")
            continue

        person = people_registry.add_person(
            name=name,
            role=role,
            department=department,
            email="",
            semantic_scholar_id=str(author["authorId"]),
        )
        registered.append(person)
        print(
            f"  ✓ {name:<24} id={author['authorId']:<10} "
            f"papers={author.get('paperCount', '?')}"
        )
        time.sleep(1.2)  # unauthenticated rate limit is ~1 req/s

    return registered


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed a realistic evaluation corpus")
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Max papers per person (default: config.MAX_PAPERS_PER_PERSON)",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run the ingestion pipeline after registering",
    )
    args = parser.parse_args()

    seed(cap=args.cap)

    people = people_registry.get_all()
    print(f"\nRegistry now holds {len(people)} people.")

    if not args.ingest:
        print("Run with --ingest to fetch and index their papers.")
        return

    print(
        f"\nIngesting (cap {config.MAX_PAPERS_PER_PERSON} papers/person)… "
        "this calls Semantic Scholar and embeds every abstract.\n"
    )
    from ingestion import ingestor
    from rag import pipeline
    from rag.vector_store import VectorStoreManager

    t0 = time.perf_counter()
    ingestor.run_ingestion()
    elapsed = time.perf_counter() - t0

    pipeline.post_sync_rebuild()

    count = VectorStoreManager.get_or_create().count()
    print(f"\n  Indexed {count} documents in {elapsed:.1f}s")
    if ingestor.sync_status.get("errors"):
        print(f"  Errors: {len(ingestor.sync_status['errors'])}")
        for err in ingestor.sync_status["errors"][:5]:
            print(f"    - {err}")


if __name__ == "__main__":
    main()
