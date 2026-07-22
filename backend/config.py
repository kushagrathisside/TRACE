import os

# Optional HuggingFace mirror, for networks where huggingface.co is slow or
# blocked.  This is NOT defaulted to a third-party mirror: when the mirror is
# unreachable the cross-encoder silently fails to download and the pipeline
# degrades to passthrough ranking — a large quality regression caused by a
# default nobody set deliberately.  Set HF_ENDPOINT explicitly if you need it.
if os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = os.environ["HF_ENDPOINT"]

# ── Institute ────────────────────────────────────────────────────────────────
INSTITUTE_NAME = os.getenv("INSTITUTE_NAME", "TRACE-Institute")

# ── Models ───────────────────────────────────────────────────────────────────
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.2:3b")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-minilm")
# Empty string is a deliberate "run without reranking" choice; any other value
# is loaded eagerly and a load failure is reported loudly (see RERANKER_REQUIRED).
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ── Evaluation (optional) ──────────────────────────────────────────────────────
ENABLE_RAGAS_SCORING = os.getenv("ENABLE_RAGAS_SCORING", "false").lower() == "true"

# ── Ollama / KV-cache ────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# keep_alive=-1 keeps the model resident in VRAM/RAM indefinitely so that the
# KV-cache of the static system-prompt prefix is reused across every request.
OLLAMA_KEEP_ALIVE = int(os.getenv("OLLAMA_KEEP_ALIVE", "-1"))
# Context window.  8 k covers 7 retrieved chunks × ~800 tokens each + prompt.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

# ── HNSW index (ChromaDB) ────────────────────────────────────────────────────
# These are applied only when the collection is first created.
# Delete data/chroma_db/ to force recreation with updated settings.
HNSW_SPACE = "cosine"
HNSW_M = int(os.getenv("HNSW_M", "48"))  # 16→64; higher = better recall, more RAM
HNSW_CONSTRUCTION_EF = int(
    os.getenv("HNSW_CONSTRUCTION_EF", "200")
)  # larger = better index quality
HNSW_SEARCH_EF = int(os.getenv("HNSW_SEARCH_EF", "150"))  # larger = better query recall

# ── Retrieval ────────────────────────────────────────────────────────────────
RETRIEVAL_FETCH_K = int(
    os.getenv("RETRIEVAL_FETCH_K", "20")
)  # candidates for reranking
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))  # final top-k after rerank

# cosine distance threshold: 0 = identical, 2 = opposite.
# Queries with no retrieved doc closer than this are reported as "no results".
MIN_SIMILARITY_DISTANCE = float(os.getenv("MIN_SIMILARITY_DISTANCE", "0.85"))

# ── Retrieval ablation axes ───────────────────────────────────────────────────
# These exist so eval/ablate.py can measure the marginal contribution of each
# retrieval stage.  Changing them changes what the production pipeline does, so
# every value is echoed into the per-query trace and into MLflow run params.
#
#   dense   — bi-encoder / HNSW only
#   bm25    — BM25 keyword search only
#   hybrid  — both, fused with weighted RRF (default)
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").lower()

# LLM query expansion before dense retrieval.  Costs ~800 ms; whether it helps
# is an empirical question — measure it with ablate.py before assuming it does.
ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"

# BM25 candidates scoring at or below this are dropped.  MUST stay > 0: a query
# sharing no token with the corpus scores 0 against EVERY document, and rank_bm25
# still returns the full top-k.  Letting those through hands RRF a list of pure
# noise which then outranks genuine dense hits (a doc at BM25 rank 0 scores
# 1/61 = 0.0164, beating a real dense hit at rank 5 with 1/66 = 0.0152).
BM25_MIN_SCORE = float(os.getenv("BM25_MIN_SCORE", "0.0"))

# Reciprocal Rank Fusion.  k dampens high-rank outliers; the weights let you
# bias the fusion toward one retriever.  Equal weights = textbook RRF.
RRF_K = int(os.getenv("RRF_K", "60"))
RRF_WEIGHT_DENSE = float(os.getenv("RRF_WEIGHT_DENSE", "1.0"))
# Kept at parity deliberately.  A sweep appeared to show 2.0 winning
# (nDCG@5 0.830 → 0.885), but the entire gain came from 13 author-name queries
# — the exact_title and topic slices scored bit-identically at every weight —
# and those 13 were depressed by a reranker defect (author names were absent
# from the cross-encoder's input), not by the fusion weights.  Tuning a global
# default on a 13-query artifact of a separate bug is how you bake a workaround
# into a constant.  Re-sweep on a corpus-representative eval set before moving
# it:  python eval/ablate.py --sweep rrf_weight_sparse --values 1,2,3,5
RRF_WEIGHT_SPARSE = float(os.getenv("RRF_WEIGHT_SPARSE", "1.0"))

# Documents pulled in by BM25 alone never passed the dense distance guard.
# When true, a BM25-only candidate must still be within MIN_SIMILARITY_DISTANCE
# of the query to survive fusion — otherwise the guard is trivially bypassed by
# the sparse arm and irrelevant papers reach the LLM context.
ENFORCE_GUARD_POST_FUSION = (
    os.getenv("ENFORCE_GUARD_POST_FUSION", "true").lower() == "true"
)

# Minimum cross-encoder score for a document to reach the LLM.  Empty = no
# floor (every query returns RETRIEVAL_K docs no matter how weak the match).
# Scale is model-specific — ms-marco-MiniLM logits run roughly -11..+11, so
# something around -5 is a reasonable starting floor.  Tune with ablate.py.
_min_rr = os.getenv("MIN_RERANK_SCORE", "").strip()
MIN_RERANK_SCORE = float(_min_rr) if _min_rr else None

# How much the cross-encoder is allowed to override fusion order.
#   1.0 = pure cross-encoder ranking (previous behaviour)
#   0.0 = fusion order, reranker ignored
# Blending exists because the two disagree systematically, and each is right
# about a different query class. Measured against hand-written queries the
# cross-encoder REGRESSED nDCG@5 from 0.728 to 0.519 (ms-marco) / 0.502
# (bge-reranker-base) — two independently trained models, so it is not a
# property of one model's training data.
#
# The mechanism is a task mismatch. Cross-encoders are trained for "does this
# passage answer this query"; TRACE needs "is this the most useful thing the
# institute has written", over a corpus where the honest answer is often a
# near-miss. On "evaluating factual grounding in retrieval-augmented
# generation" the reranker scores the corpus's only RAG paper at -2.9 and
# demotes it — correct by its objective, wrong for the product.
#
# On derived slices (exact title, author name) the reranker is strongly
# positive, so neither extreme wins everywhere.  Measured nDCG@5 by blend:
#
#   slice        n    0.0    0.25   0.5    0.75   1.0
#   manual      10   0.728  0.749  0.764  0.655  0.519   ← peaks at 0.5
#   exact_title 40   0.789  0.830  0.935  0.966  0.975
#   author      13   0.464  0.550  0.594  0.639  0.649
#   topic       40   0.593  0.642  0.745  0.844  0.866
#
# 0.5 is chosen because `manual` is the only slice whose queries were written
# by a human rather than derived from the documents they must retrieve, and it
# is the shape production traffic actually takes — students describing a
# research idea.  The trade is explicit: -0.040 on exact_title against +0.245
# on manual versus pure reranking.  `topic` pushes hardest toward 1.0 and is
# the least trustworthy slice (its queries are verbatim bags of abstract
# tokens), so it is discounted here.
#
# PROVISIONAL: manual is n=10 and the curve is not monotone. Grow that slice
# past 30 queries before treating 0.5 as settled.  Re-run with:
#   python eval/ablate.py --sweep rerank_blend --values 0,0.25,0.5,0.75,1
RERANK_BLEND = float(os.getenv("RERANK_BLEND", "0.5"))

# When the configured reranker cannot be loaded the pipeline falls back to a
# no-op passthrough.  That is a silent ~15-point nDCG@5 regression, so the
# fallback is reported in /health, in every trace record, and in MLflow params.
# Set true to make an unloadable reranker a hard startup failure instead.
RERANKER_REQUIRED = os.getenv("RERANKER_REQUIRED", "false").lower() == "true"

# Semantic-cache hit: distance < this means "same question, use cached answer"
CACHE_HIT_DISTANCE = float(os.getenv("CACHE_HIT_DISTANCE", "0.08"))
CACHE_MAX_AGE_DAYS = int(os.getenv("CACHE_MAX_AGE_DAYS", "15"))

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(__file__)
VECTOR_DB_PATH = os.path.join(_HERE, "data", "chroma_db")
PEOPLE_REGISTRY_PATH = os.path.join(_HERE, "data", "people.json")
SYNC_STATUS_PATH = os.path.join(_HERE, "data", "sync_status.json")
FEEDBACK_LOG_PATH = os.path.join(_HERE, "data", "feedback.jsonl")
# One JSON record per query: candidate IDs and scores at every retrieval stage,
# per-stage latency, and the config that produced them.  This is the substrate
# every offline metric is computed from — without it a thumbs-down tells you a
# query was bad but not what was retrieved or which stages ran.
TRACE_LOG_PATH = os.path.join(_HERE, "data", "retrieval_traces.jsonl")
FRONTEND_DIR = os.path.join(_HERE, "..", "frontend")

# ── Semantic Scholar ──────────────────────────────────────────────────────────
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
PAPER_FIELDS = (
    "title,abstract,year,authors,venue,externalIds,openAccessPdf,publicationTypes"
)
# Upper bound on papers fetched per registered person, newest first.  Bounds
# index size, embedding time and peak memory during a sync — a handful of
# prolific authors can otherwise pull tens of thousands of records.
# 0 = unlimited (only sensible with an API key and plenty of RAM).
MAX_PAPERS_PER_PERSON = int(os.getenv("MAX_PAPERS_PER_PERSON", "200"))

# Documents embedded per request during a sync.  The vector-store wrapper would
# otherwise send the whole sync in one call, which Ollama rejects outright once
# the payload is large enough — losing the entire run.  Batching also keeps peak
# memory flat no matter how many papers are pulled.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))

# ── Recency scoring ──────────────────────────────────────────────────────────
# Bonus added to the RRF score per document based on publication recency.
# A paper from the current year gets +RECENCY_WEIGHT; a 10-year-old paper
# gets 0.  The typical single-list RRF contribution is ~0.016, so 0.01 adds
# roughly half a rank's worth of signal without drowning out relevance.
RECENCY_WEIGHT = float(os.getenv("RECENCY_WEIGHT", "0.01"))

# ── Context window management ─────────────────────────────────────────────────
# Max characters of abstract text per retrieved chunk fed into the LLM prompt.
# Computed from the context window, leaving room for system prompt (~800 tokens),
# query (~200 tokens), and output (~1500 tokens).  Rough rule: 4 chars ≈ 1 token.
_CONTEXT_BUDGET = max(0, OLLAMA_NUM_CTX - 800 - 200 - 1500)
MAX_ABSTRACT_CHARS = int(
    os.getenv(
        "MAX_ABSTRACT_CHARS",
        str(max(300, int(_CONTEXT_BUDGET / max(1, RETRIEVAL_K) * 4))),
    )
)

# ── Sidecar meta file (embedding-model mismatch guard) ────────────────────────
CHROMA_META_PATH = os.path.join(_HERE, "data", "chroma_meta.json")

# ── Admin / Rate-limiting ─────────────────────────────────────────────────────
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
if not ADMIN_PASSWORD:
    raise ValueError(
        "ADMIN_PASSWORD environment variable must be set. "
        "Set it to a strong password before deployment."
    )
RATE_LIMIT_QUERIES = os.getenv("RATE_LIMIT_QUERIES", "10/minute")  # per IP
RATE_LIMIT_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "60/minute")
RATE_LIMIT_ADMIN = os.getenv("RATE_LIMIT_ADMIN", "60/minute")


# ── Configuration validation ───────────────────────────────────────────────────
def _validate_config() -> None:
    """Validate all numeric and logical config constraints."""
    if RETRIEVAL_K > RETRIEVAL_FETCH_K:
        raise ValueError(
            f"RETRIEVAL_K ({RETRIEVAL_K}) must be <= RETRIEVAL_FETCH_K ({RETRIEVAL_FETCH_K})"
        )
    if HNSW_M < 1:
        raise ValueError(f"HNSW_M must be >= 1, got {HNSW_M}")
    if HNSW_CONSTRUCTION_EF < 1:
        raise ValueError(
            f"HNSW_CONSTRUCTION_EF must be >= 1, got {HNSW_CONSTRUCTION_EF}"
        )
    if HNSW_SEARCH_EF < 1:
        raise ValueError(f"HNSW_SEARCH_EF must be >= 1, got {HNSW_SEARCH_EF}")
    if MIN_SIMILARITY_DISTANCE < 0 or MIN_SIMILARITY_DISTANCE > 2:
        raise ValueError(
            f"MIN_SIMILARITY_DISTANCE must be in [0, 2], got {MIN_SIMILARITY_DISTANCE}"
        )
    if CACHE_HIT_DISTANCE < 0 or CACHE_HIT_DISTANCE > 2:
        raise ValueError(
            f"CACHE_HIT_DISTANCE must be in [0, 2], got {CACHE_HIT_DISTANCE}"
        )
    if RETRIEVAL_MODE not in ("dense", "bm25", "hybrid"):
        raise ValueError(
            f"RETRIEVAL_MODE must be one of dense|bm25|hybrid, got '{RETRIEVAL_MODE}'"
        )
    if BM25_MIN_SCORE < 0:
        raise ValueError(f"BM25_MIN_SCORE must be >= 0, got {BM25_MIN_SCORE}")
    if RRF_K < 1:
        raise ValueError(f"RRF_K must be >= 1, got {RRF_K}")
    if RRF_WEIGHT_DENSE < 0 or RRF_WEIGHT_SPARSE < 0:
        raise ValueError("RRF weights must be >= 0")
    if RRF_WEIGHT_DENSE == 0 and RRF_WEIGHT_SPARSE == 0:
        raise ValueError("At least one RRF weight must be > 0")


def snapshot() -> dict:
    """
    The retrieval-relevant configuration, for trace records and MLflow params.

    Any offline number is only interpretable next to the config that produced
    it — this is what lets you tell "nDCG went up because of the reranker" from
    "nDCG went up because fetch_k changed at the same time".
    """
    return {
        "retrieval_mode": RETRIEVAL_MODE,
        "retrieval_k": RETRIEVAL_K,
        "fetch_k": RETRIEVAL_FETCH_K,
        "min_similarity_distance": MIN_SIMILARITY_DISTANCE,
        "min_rerank_score": MIN_RERANK_SCORE,
        "guard_post_fusion": ENFORCE_GUARD_POST_FUSION,
        "query_expansion": ENABLE_QUERY_EXPANSION,
        "bm25_min_score": BM25_MIN_SCORE,
        "rrf_k": RRF_K,
        "rrf_weight_dense": RRF_WEIGHT_DENSE,
        "rrf_weight_sparse": RRF_WEIGHT_SPARSE,
        "recency_weight": RECENCY_WEIGHT,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": RERANKER_MODEL_NAME,
        "llm_model": LLM_MODEL_NAME,
        "hnsw_m": HNSW_M,
        "hnsw_search_ef": HNSW_SEARCH_EF,
        "cache_hit_distance": CACHE_HIT_DISTANCE,
    }


_validate_config()
