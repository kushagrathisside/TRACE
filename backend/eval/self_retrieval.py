"""
Self-retrieval sanity check.

Uses each stored paper's own abstract as a query and verifies the paper
retrieves itself in the top-K results.  A healthy embedding + HNSW index
should score > 90% Hit Rate@5.

If the score is low:
  - Check that EMBEDDING_MODEL_NAME in config matches how the DB was built
    (the mismatch guard in vector_store.py will raise on startup if they differ)
  - Try deleting data/chroma_db/ and re-syncing with the same model
  - Verify HNSW settings: lower search_ef degrades recall

Usage:
    cd backend && python eval/self_retrieval.py
    cd backend && python eval/self_retrieval.py --k 5 --sample 300
"""

import argparse
import random
import sys

sys.path.insert(0, ".")
from rag.vector_store import VectorStoreManager
import config


def run(k: int = 5, sample: int = 200) -> float:
    vs   = VectorStoreManager.get_or_create()
    docs = vs.get_all_documents()

    if not docs:
        print("No documents in DB. Run a sync first.")
        return 0.0

    subset = random.sample(docs, min(sample, len(docs)))
    hits = 0

    for doc in subset:
        pid     = doc.metadata.get("paper_id", "")
        query   = doc.page_content[:400]
        results = vs.similarity_search_with_score(query, k=k)
        ids     = [d.metadata.get("paper_id") for d, _ in results]
        if pid in ids:
            hits += 1

    rate   = hits / len(subset)
    status = "✓ PASS" if rate >= 0.90 else "✗ FAIL (expected > 0.90)"
    print(f"\n{status}")
    print(f"  Self-retrieval Hit Rate@{k}: {rate:.2%}  ({hits}/{len(subset)} sampled from {len(docs)} total)\n")
    return rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-retrieval sanity check")
    parser.add_argument("--k",      type=int, default=5,   help="Top-K to check (default 5)")
    parser.add_argument("--sample", type=int, default=200, help="Number of docs to sample (default 200)")
    args = parser.parse_args()
    rate = run(k=args.k, sample=args.sample)
    sys.exit(0 if rate >= 0.90 else 1)
