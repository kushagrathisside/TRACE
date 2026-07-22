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
import math
import uuid
from dataclasses import dataclass

import config
from langchain_core.documents import Document
from llm_provider import LLMProvider

from rag import trace
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


#: Expansion output longer than this is discarded — a chatty model returning a
#: paragraph would swamp the original query in the embedded text.
_MAX_EXPANSION_CHARS = 300


def _expand_query(query: str) -> str:
    """
    Ask the LLM (temperature=0) for related technical keywords.

    Appending them to the query improves recall for vague or short ideas without
    distorting the semantics — the original query text still dominates.  Whether
    that is true for *your* corpus is measurable: set ENABLE_QUERY_EXPANSION=false
    and compare nDCG@5 with eval/ablate.py before assuming it helps.

    Degrades to the raw query on any failure.  Expansion is a recall optimisation,
    never a reason to fail a request.
    """
    try:
        llm = LLMProvider.get_llm(temperature=0.0)
        resp = llm.invoke(
            f'Research idea: "{query}"\n'
            "List 5 related technical keywords, methods, and sub-fields relevant to "
            "finding academic papers on this topic. "
            "Return ONLY a comma-separated list, nothing else."
        )
        keywords = (resp.content or "").strip().rstrip(".")
    except Exception as exc:
        logger.warning(f"Query expansion failed, using raw query: {exc}")
        return query

    # Reject anything that does not look like a keyword list.
    if not keywords or len(keywords) > _MAX_EXPANSION_CHARS or "\n" in keywords:
        logger.warning(
            f"Discarding malformed query expansion ({len(keywords)} chars); "
            "using raw query"
        )
        return query
    return f"{query} {keywords}"


def _apply_guard_post_fusion(
    fused: list[tuple[float, Document]],
    dense_distances: dict[str, float],
    query_vec: list[float],
) -> tuple[list[tuple[float, Document]], int]:
    """
    Enforce MIN_SIMILARITY_DISTANCE on candidates that only BM25 proposed.

    The dense arm is filtered before fusion, but BM25 candidates never were —
    so a keyword coincidence was enough to put an unrelated paper in front of
    the LLM, which is exactly what the guard exists to prevent.  Distances for
    those documents are fetched in one extra Chroma query.

    Returns (kept, n_dropped).
    """
    unknown = [
        doc.metadata.get("paper_id", "")
        for _, doc in fused
        if doc.metadata.get("paper_id", "") not in dense_distances
    ]
    if unknown:
        vs = VectorStoreManager.get_or_create()
        dense_distances = {
            **dense_distances,
            **vs.distances_for_ids(query_vec, unknown),
        }

    kept: list[tuple[float, Document]] = []
    dropped = 0
    for score, doc in fused:
        pid = doc.metadata.get("paper_id", "")
        dist = dense_distances.get(pid)
        # Unknown distance (lookup failed) → keep, so a guard failure degrades
        # to today's behaviour instead of emptying the result set.
        if dist is None or dist <= config.MIN_SIMILARITY_DISTANCE:
            kept.append((score, doc))
        else:
            dropped += 1
    return kept, dropped


# ── Public API ────────────────────────────────────────────────────────────────


def _blend_with_fusion(
    scored: list[tuple[float, Document]],
    fused: list[tuple[float, Document]],
    reranker_active: bool,
) -> list[tuple[float, Document]]:
    """
    Combine cross-encoder order with fusion order per RERANK_BLEND.

    Both signals are converted to a normalised rank in [0, 1] (1 = best) rather
    than blending raw values: cross-encoder logits and RRF scores live on
    incomparable scales, and logit ranges differ between reranker models, so a
    weighted sum of raw scores would silently change meaning when the model is
    swapped. Rank is the only common currency.

    Returns pairs carrying the ORIGINAL cross-encoder score, so `rerank_score`
    in the API and the trace stays interpretable.
    """
    if not reranker_active or config.RERANK_BLEND >= 1.0 or len(scored) < 2:
        return scored
    if config.RERANK_BLEND <= 0.0:
        order = {id(doc): i for i, (_, doc) in enumerate(fused)}
        return sorted(scored, key=lambda pair: order.get(id(pair[1]), len(fused)))

    n = len(scored)
    ce_rank = {id(doc): i for i, (_, doc) in enumerate(scored)}
    fusion_rank = {id(doc): i for i, (_, doc) in enumerate(fused)}

    def combined(pair: tuple[float, Document]) -> float:
        doc = pair[1]
        ce = 1.0 - ce_rank.get(id(doc), n - 1) / (n - 1)
        fu = 1.0 - fusion_rank.get(id(doc), n - 1) / (n - 1)
        return config.RERANK_BLEND * ce + (1.0 - config.RERANK_BLEND) * fu

    return sorted(scored, key=combined, reverse=True)


def _build_sources(docs: list[Document], scores: list[float]) -> list[dict]:
    """
    Serialise retrieved documents for the API response.

    `paper_id` is included deliberately: without a stable identifier the eval
    harness had nothing to compare against ground-truth labels, so every
    retrieval metric silently evaluated to zero.
    """
    out = []
    for doc, score in zip(docs, scores):
        m = doc.metadata
        out.append(
            {
                "paper_id": m.get("paper_id", ""),
                "title": m.get("paper_title", ""),
                "year": m.get("year", ""),
                "venue": m.get("venue", ""),
                "authors": m.get("authors", ""),
                "institute_authors": m.get("institute_authors", ""),
                "institute_roles": m.get("institute_roles", ""),
                "departments": m.get("departments", ""),
                "url": m.get("paper_url", ""),
                "rerank_score": None if math.isnan(score) else round(score, 4),
            }
        )
    return out


@dataclass
class RetrievalResult:
    """Everything retrieval produced for one query, before any generation."""

    docs: list[Document]
    scores: list[float]
    query_vec: list[float]
    record: dict
    timer: trace.StageTimer
    no_results: bool = False


def retrieve(
    query: str,
    query_id: str | None = None,
    timer: "trace.StageTimer | None" = None,
    record: dict | None = None,
    query_vec: list[float] | None = None,
) -> RetrievalResult:
    """
    Stages 2-6: expansion, dense search, guard, BM25, fusion, reranking.

    Split out from run() because retrieval metrics (Recall, nDCG, MRR) depend
    only on which documents come back — generation is irrelevant to them and
    costs 10-30 s per query.  Running it anyway made the ablation take hours
    for numbers it could not change.

    Callers that also need an answer use run(); eval/ablate.py and
    run_eval.py --retrieval-only use this directly.
    """
    query_id = query_id or str(uuid.uuid4())
    timer = timer or trace.StageTimer()
    record = (
        record
        if record is not None
        else {
            "query_id": query_id,
            "query": query,
            "config": config.snapshot(),
        }
    )
    if query_vec is None:
        with timer("embed"):
            query_vec = LLMProvider.get_embeddings().embed_query(query)

    mode = config.RETRIEVAL_MODE

    # ── Stage 2: Query expansion ──────────────────────────────────────────────
    if config.ENABLE_QUERY_EXPANSION:
        with timer("query_expansion"):
            expanded = _expand_query(query)
    else:
        expanded = query
    record["expanded_query"] = expanded

    # ── Stage 3: Dense vector search ─────────────────────────────────────────
    close: list[tuple[Document, float]] = []
    dense_distances: dict[str, float] = {}
    if mode in ("dense", "hybrid"):
        vs = VectorStoreManager.get_or_create()
        with timer("dense_search"):
            raw_results: list[tuple[Document, float]] = vs.similarity_search_with_score(
                expanded, k=config.RETRIEVAL_FETCH_K
            )
        record["dense"] = [
            {"id": d.metadata.get("paper_id", ""), "distance": round(dist, 4)}
            for d, dist in raw_results
        ]
        dense_distances = {
            d.metadata.get("paper_id", ""): dist for d, dist in raw_results
        }

        # ── Stage 4: Similarity threshold guard ──────────────────────────────
        # Keep only candidates whose cosine distance ≤ MIN_SIMILARITY_DISTANCE.
        # This prevents the LLM from hallucinating connections when nothing
        # truly relevant was retrieved.
        close = [
            (doc, dist)
            for doc, dist in raw_results
            if dist <= config.MIN_SIMILARITY_DISTANCE
        ]
        record["dense_kept"] = len(close)

    # ── Stage 5: BM25 + weighted RRF ─────────────────────────────────────────
    hybrid = HybridSearcher.get()
    bm25_hits: list[tuple[float, Document]] = []
    if mode in ("bm25", "hybrid") and hybrid:
        with timer("bm25_search"):
            bm25_hits = hybrid.search(query, top_k=config.RETRIEVAL_FETCH_K)
        record["bm25"] = [
            {"id": d.metadata.get("paper_id", ""), "score": round(s, 4)}
            for s, d in bm25_hits
        ]

    if mode == "dense":
        fused = [(0.0, doc) for doc, _ in close]
    elif mode == "bm25":
        fused = list(bm25_hits)
    elif hybrid:
        with timer("fusion"):
            fused = HybridSearcher.rrf_with_scores(
                [(dist, doc) for doc, dist in close],
                bm25_hits,
                top_n=config.RETRIEVAL_FETCH_K,
            )
    else:
        fused = [(0.0, doc) for doc, _ in close]

    # BM25-only candidates never faced the distance guard — apply it now, or it
    # is bypassed entirely by the sparse arm.
    if config.ENFORCE_GUARD_POST_FUSION and bm25_hits and mode != "dense":
        with timer("guard"):
            fused, n_dropped = _apply_guard_post_fusion(
                fused, dense_distances, query_vec
            )
        record["guard_dropped"] = n_dropped

    record["fused_ids"] = [d.metadata.get("paper_id", "") for _, d in fused]

    if not fused:
        record["no_results"] = True
        return RetrievalResult([], [], query_vec, record, timer, no_results=True)

    # ── Stage 6: Cross-encoder reranking ─────────────────────────────────────
    reranker = Reranker.get()
    with timer("rerank"):
        # Score the whole candidate set, not just the top-k, so fusion order and
        # cross-encoder order can be blended before truncation.
        scored_all: list[tuple[float, Document]] = reranker.rerank(
            query, [doc for _, doc in fused], top_k=len(fused)
        )
        reranked_pairs = _blend_with_fusion(scored_all, fused, reranker.active)[
            : config.RETRIEVAL_K
        ]
    record["reranker_active"] = reranker.active
    record["reranked"] = [
        {
            "id": d.metadata.get("paper_id", ""),
            "score": None if math.isnan(s) else round(s, 4),
        }
        for s, d in reranked_pairs
    ]

    # Relevance floor: without it every query returns RETRIEVAL_K documents no
    # matter how weak the match, and the LLM narrates a connection anyway.
    # Only meaningful when a real cross-encoder produced the scores.
    if config.MIN_RERANK_SCORE is not None and reranker.active:
        before = len(reranked_pairs)
        reranked_pairs = [
            (s, d) for s, d in reranked_pairs if s >= config.MIN_RERANK_SCORE
        ]
        record["rerank_floor_dropped"] = before - len(reranked_pairs)
        if not reranked_pairs:
            record["no_results"] = True
            return RetrievalResult([], [], query_vec, record, timer, no_results=True)

    final_docs = [doc for _, doc in reranked_pairs]
    final_scores = [score for score, _ in reranked_pairs]
    record["final_ids"] = [d.metadata.get("paper_id", "") for d in final_docs]
    record["top_rerank_score"] = (
        None
        if not final_scores or math.isnan(final_scores[0])
        else round(final_scores[0], 4)
    )
    return RetrievalResult(final_docs, final_scores, query_vec, record, timer)


def retrieve_only(query: str, query_id: str | None = None) -> dict:
    """
    Retrieval without generation — what offline retrieval metrics need.

    Skips the semantic cache (an eval must measure retrieval, not the cache)
    and skips the LLM entirely. Writes a trace record tagged
    `retrieval_only: true` so these runs are distinguishable from real traffic
    when the trace log is analysed.
    """
    result = retrieve(query, query_id=query_id)
    result.record.update(
        {
            "retrieval_only": True,
            "cached": False,
            "stages": result.timer.stages,
            "latency_ms": result.timer.total_ms,
            "retrieval_ms": result.timer.retrieval_ms,
        }
    )
    result.record.setdefault("final_ids", [])
    trace.write(result.record)
    return {
        "sources": _build_sources(result.docs, result.scores),
        # The candidate set the reranker was handed, in fusion order.  Recall at
        # the FETCH depth must be computed from this, not from the final top-k:
        # the final list is truncated to RETRIEVAL_K, so Recall@20 measured over
        # it silently collapses to Recall@5 and the candidate ceiling — the one
        # number the reranker cannot change — becomes invisible.
        "candidate_ids": result.record.get("fused_ids", []),
        "no_results": result.no_results,
        "retrieval_ms": result.timer.retrieval_ms,
        "query_id": result.record["query_id"],
    }


def run(query: str, bypass_cache: bool = False, query_id: str | None = None) -> dict:
    """
    Execute the full pipeline synchronously.

    Called from the FastAPI endpoint via run_in_executor so it doesn't block
    the async event loop during the long LLM generation step.

    Every call emits one trace record (see rag/trace.py) carrying the candidate
    set at each stage, per-stage latency, and the config snapshot.  The returned
    `query_id` links that record to the user's later thumbs-up/down.
    """
    query_id = query_id or str(uuid.uuid4())
    timer = trace.StageTimer()
    record: dict = {
        "query_id": query_id,
        "query": query,
        "config": config.snapshot(),
    }

    # ── Stage 1: Semantic cache check ────────────────────────────────────────
    with timer("embed"):
        query_vec = LLMProvider.get_embeddings().embed_query(query)
    cache = _get_cache()

    if not bypass_cache:
        with timer("cache_lookup"):
            hit = cache.get(query_vec)
        if hit:
            record.update(
                {
                    "cached": True,
                    "stages": timer.stages,
                    "latency_ms": timer.total_ms,
                    "retrieval_ms": timer.retrieval_ms,
                    "final_ids": [
                        s.get("paper_id", "") for s in hit.get("sources", [])
                    ],
                }
            )
            trace.write(record)
            logger.info(
                json.dumps(
                    {
                        "event": "query",
                        "query_id": query_id,
                        "cached": True,
                        "latency_ms": timer.total_ms,
                        "query_preview": query[:80],
                    }
                )
            )
            return {**hit, "cached": True, "query_id": query_id}

    record["cached"] = False

    # ── Stages 2-6: retrieval ────────────────────────────────────────────────
    retrieved = retrieve(
        query, query_id=query_id, timer=timer, record=record, query_vec=query_vec
    )

    def _finish(payload: dict) -> dict:
        record.update(
            {
                "stages": timer.stages,
                "latency_ms": timer.total_ms,
                "retrieval_ms": timer.retrieval_ms,
            }
        )
        record.setdefault("final_ids", [])
        trace.write(record)
        return payload

    if retrieved.no_results:
        cache.set(query_vec, query, NO_RESULTS.model_dump(), [])
        return _finish(
            {
                "answer": NO_RESULTS.model_dump(),
                "sources": [],
                "cached": False,
                "query_id": query_id,
            }
        )

    final_docs, final_scores = retrieved.docs, retrieved.scores

    # ── Stage 7: LLM structured generation ───────────────────────────────────
    with timer("generation"):
        landscape, guard_stats = generate_answer(query, final_docs)
    record["grounding"] = guard_stats

    # ── Optional: RAGAS evaluation ───────────────────────────────────────────
    if config.ENABLE_RAGAS_SCORING:
        with timer("ragas"):
            try:
                from eval.ragas_scorer import score_and_log

                score_and_log(
                    query, landscape.landscape_summary, final_docs, query_id=query_id
                )
            except Exception as exc:
                logger.warning(f"RAGAS scoring failed (non-fatal): {exc}")

    sources = _build_sources(final_docs, final_scores)
    result = {
        "answer": landscape.model_dump(),
        "sources": sources,
        "cached": False,
        "query_id": query_id,
    }

    # ── Stage 8: Write to semantic cache ──────────────────────────────────────
    with timer("cache_write"):
        cache.set(query_vec, query, landscape.model_dump(), sources)

    _finish(result)
    logger.info(
        json.dumps(
            {
                "event": "query",
                "query_id": query_id,
                "cached": False,
                "mode": config.RETRIEVAL_MODE,
                "reranker_active": record.get("reranker_active"),
                "retrieved": len(final_docs),
                "top_rerank_score": record.get("top_rerank_score"),
                "retrieval_ms": timer.retrieval_ms,
                "latency_ms": timer.total_ms,
                "query_preview": query[:80],
            }
        )
    )
    return result


def post_sync_rebuild() -> float | None:
    """
    Called by ingestor after every successful sync.
    Resets all state that depends on the document set.

    Returns the self-retrieval hit rate (or None if the check could not run) so
    the caller can mark the sync degraded — previously the check ran here *and*
    again in the API layer, doubling the cost for one number.
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
    rate: float | None = None
    try:
        from eval.self_retrieval import run as run_self_retrieval

        logger.info("Running self-retrieval test...")
        rate = run_self_retrieval(k=5, sample=min(200, len(docs)))
    except Exception as exc:
        logger.warning(f"Self-retrieval test failed (non-fatal): {exc}")

    logger.info(f"post_sync_rebuild complete — BM25 index: {len(docs)} docs")
    return rate


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
