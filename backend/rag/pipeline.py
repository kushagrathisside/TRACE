"""
Full RAG pipeline — the single entry-point called by the API layer.

Query flow (6 stages)
---------------------
  1. Embed query  →  check semantic cache  (cache hit → return instantly)
  2. Query expansion via LLM  (adds related keywords to improve recall)
  3. Dense vector search with similarity scores  (bi-encoder, HNSW, k=20)
  4. Similarity threshold guard  (drop candidates too far from query)
  5. BM25 hybrid search + Reciprocal Rank Fusion  (catches named-entity misses)
  6. Cross-encoder reranking  (expensive but accurate, top-5 out of 20)
  7. LLM structured generation  (JSON mode + Pydantic validation)
  8. Write to semantic cache

Post-sync hook
--------------
`post_sync_rebuild()` is called by the ingestor after every sync:
  - Invalidates the semantic cache (stale answers)
  - Rebuilds the BM25 index over new documents
  - Resets the VectorStoreManager singleton (clears LangChain wrapper cache)

Stage latencies (approximate, CPU, 15 k papers)
------------------------------------------------
  Embed query         ~  5 ms
  Semantic cache      ~  2 ms
  Query expansion     ~ 800 ms  (LLM call, skipped on cache hit)
  HNSW vector search  ~  8 ms   (search_ef=150, 15 k docs)
  BM25 search         ~  3 ms
  RRF fusion          <  1 ms
  Cross-encoder (×20) ~ 80 ms
  LLM generation      ~ 8–30 s  (llama3.2 3B on CPU)
"""

import json
import logging
import time

import config
from langchain_core.documents import Document
from llm_provider import LLMProvider

from rag.chain import ResearchLandscape, generate_answer
from rag.hybrid_search import HybridSearcher
from rag.reranker import Reranker
from rag.semantic_cache import SemanticCache
from rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────
_cache: SemanticCache | None = None

NO_RESULTS = ResearchLandscape(
    landscape_summary=(
        "No closely related institute research was found for this topic. "
        "This may be a genuinely novel direction at the institute, or the relevant "
        "work has not yet been indexed. Try again after the next data sync, or "
        "consult a faculty advisor directly."
    ),
    no_relevant_research=True,
)


def _get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        vs = VectorStoreManager.get_or_create()
        _cache = SemanticCache(vs.get_chroma_client())
    return _cache


def _expand_query(query: str) -> str:
    """
    Ask the LLM (temperature=0) for related technical keywords.
    Appending them to the query improves recall for vague or short ideas without
    distorting the semantics — the original query text still dominates.
    """
    llm = LLMProvider.get_llm(temperature=0.0)
    resp = llm.invoke(
        f'Research idea: "{query}"\n'
        "List 5 related technical keywords, methods, and sub-fields relevant to "
        "finding academic papers on this topic. "
        "Return ONLY a comma-separated list, nothing else."
    )
    keywords = resp.content.strip().rstrip(".")
    return f"{query} {keywords}"


# ── Public API ────────────────────────────────────────────────────────────────


def run(query: str, bypass_cache: bool = False) -> dict:
    """
    Execute the full pipeline synchronously.
    Called from the FastAPI endpoint via run_in_executor so it doesn't block
    the async event loop during the long LLM generation step.
    """
    t0 = time.perf_counter()
    embeddings = LLMProvider.get_embeddings()

    # ── Stage 1: Semantic cache check ────────────────────────────────────────
    query_vec = embeddings.embed_query(query)
    cache = _get_cache()

    if not bypass_cache:
        hit = cache.get(query_vec)
        if hit:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                json.dumps(
                    {
                        "event": "query",
                        "cached": True,
                        "latency_ms": round(elapsed, 1),
                        "query_preview": query[:80],
                    }
                )
            )
            return {**hit, "cached": True}

    # ── Stage 2: Query expansion ──────────────────────────────────────────────
    expanded = _expand_query(query)

    # ── Stage 3: Dense vector search ─────────────────────────────────────────
    vs = VectorStoreManager.get_or_create()
    raw_results: list[tuple[Document, float]] = vs.similarity_search_with_score(
        expanded, k=config.RETRIEVAL_FETCH_K
    )

    # ── Stage 4: Similarity threshold guard ──────────────────────────────────
    # Keep only candidates whose cosine distance ≤ MIN_SIMILARITY_DISTANCE.
    # This prevents the LLM from hallucinating connections when nothing truly
    # relevant was retrieved.
    close: list[tuple[Document, float]] = [
        (doc, dist)
        for doc, dist in raw_results
        if dist <= config.MIN_SIMILARITY_DISTANCE
    ]
    if not close:
        cache.set(query_vec, query, NO_RESULTS.model_dump(), [])
        return {
            "answer": NO_RESULTS.model_dump(),
            "sources": [],
            "cached": False,
        }

    # ── Stage 5: BM25 + RRF hybrid search ────────────────────────────────────
    hybrid = HybridSearcher.get()
    if hybrid:
        bm25_hits = hybrid.search(query, top_k=config.RETRIEVAL_FETCH_K)
        fused: list[Document] = HybridSearcher.rrf(
            [(dist, doc) for doc, dist in close],
            bm25_hits,
            top_n=config.RETRIEVAL_FETCH_K,
        )
    else:
        fused = [doc for doc, _ in close]

    # ── Stage 6: Cross-encoder reranking ─────────────────────────────────────
    reranker = Reranker.get()
    reranked_pairs: list[tuple[float, Document]] = reranker.rerank(
        query, fused, top_k=config.RETRIEVAL_K
    )
    final_docs = [doc for _, doc in reranked_pairs]

    # ── Stage 7: LLM structured generation ───────────────────────────────────
    landscape = generate_answer(query, final_docs)

    # ── Optional: RAGAS evaluation ───────────────────────────────────────────
    if config.ENABLE_RAGAS_SCORING:
        try:
            from eval.ragas_scorer import score_and_log

            score_and_log(query, landscape.landscape_summary, final_docs)
        except Exception as exc:
            logger.warning(f"RAGAS scoring failed (non-fatal): {exc}")

    # ── Build sources list ────────────────────────────────────────────────────
    sources = [
        {
            "title": d.metadata.get("paper_title", ""),
            "year": d.metadata.get("year", ""),
            "venue": d.metadata.get("venue", ""),
            "authors": d.metadata.get("authors", ""),
            "institute_authors": d.metadata.get("institute_authors", ""),
            "institute_roles": d.metadata.get("institute_roles", ""),
            "departments": d.metadata.get("departments", ""),
            "url": d.metadata.get("paper_url", ""),
        }
        for d in final_docs
    ]

    result = {
        "answer": landscape.model_dump(),
        "sources": sources,
        "cached": False,
    }

    # ── Stage 8: Write to semantic cache ──────────────────────────────────────
    cache.set(query_vec, query, landscape.model_dump(), sources)

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        json.dumps(
            {
                "event": "query",
                "cached": False,
                "retrieved": len(final_docs),
                "top_rerank_score": round(reranked_pairs[0][0], 3)
                if reranked_pairs
                else None,
                "latency_ms": round(elapsed, 1),
                "query_preview": query[:80],
            }
        )
    )

    return result


def post_sync_rebuild() -> None:
    """
    Called by ingestor after every successful sync.
    Resets all state that depends on the document set.
    """
    global _cache

    # Invalidate semantic cache (answers may reference stale documents)
    if _cache:
        _cache.invalidate()
    _cache = None

    # Reset vector store singleton (forces reconnect to updated chroma_db)
    VectorStoreManager.reset()

    # Rebuild BM25 index over new document set
    vs = VectorStoreManager.get_or_create()
    docs = vs.get_all_documents()
    if docs:
        HybridSearcher.build(docs)
    else:
        HybridSearcher.invalidate()

    # Run self-retrieval test to detect embedding corruption
    try:
        from eval.self_retrieval import run as run_self_retrieval

        logger.info("Running self-retrieval test...")
        run_self_retrieval(k=5, sample=min(100, len(docs)))
    except Exception as exc:
        logger.warning(f"Self-retrieval test failed (non-fatal): {exc}")

    logger.info(f"post_sync_rebuild complete — BM25 index: {len(docs)} docs")


def build_bm25_on_startup() -> None:
    """Warm up the BM25 index from the existing ChromaDB on server start."""
    try:
        vs = VectorStoreManager.get_or_create()
        docs = vs.get_all_documents()
        if docs:
            HybridSearcher.build(docs)
            logger.info(f"Startup BM25 index built: {len(docs)} docs")
        else:
            logger.info("Startup BM25: no documents in DB yet")
    except Exception as exc:
        logger.warning(f"Startup BM25 build failed (non-fatal): {exc}")
