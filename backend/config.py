import os

# ── Institute ────────────────────────────────────────────────────────────────
INSTITUTE_NAME = os.getenv("INSTITUTE_NAME", "Our Institute")

# ── Models ───────────────────────────────────────────────────────────────────
LLM_MODEL_NAME       = os.getenv("LLM_MODEL_NAME", "llama3.2")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
RERANKER_MODEL_NAME  = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Evaluation (optional) ──────────────────────────────────────────────────────
ENABLE_RAGAS_SCORING = os.getenv("ENABLE_RAGAS_SCORING", "false").lower() == "true"

# ── Ollama / KV-cache ────────────────────────────────────────────────────────
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# keep_alive=-1 keeps the model resident in VRAM/RAM indefinitely so that the
# KV-cache of the static system-prompt prefix is reused across every request.
OLLAMA_KEEP_ALIVE = int(os.getenv("OLLAMA_KEEP_ALIVE", "-1"))
# Context window.  8 k covers 7 retrieved chunks × ~800 tokens each + prompt.
OLLAMA_NUM_CTX    = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

# ── HNSW index (ChromaDB) ────────────────────────────────────────────────────
# These are applied only when the collection is first created.
# Delete data/chroma_db/ to force recreation with updated settings.
HNSW_SPACE           = "cosine"
HNSW_M               = int(os.getenv("HNSW_M", "48"))          # 16→64; higher = better recall, more RAM
HNSW_CONSTRUCTION_EF = int(os.getenv("HNSW_CONSTRUCTION_EF", "200"))  # larger = better index quality
HNSW_SEARCH_EF       = int(os.getenv("HNSW_SEARCH_EF", "150"))         # larger = better query recall

# ── Retrieval ────────────────────────────────────────────────────────────────
RETRIEVAL_FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", "20"))  # candidates for reranking
RETRIEVAL_K       = int(os.getenv("RETRIEVAL_K", "5"))         # final top-k after rerank

# cosine distance threshold: 0 = identical, 2 = opposite.
# Queries with no retrieved doc closer than this are reported as "no results".
MIN_SIMILARITY_DISTANCE = float(os.getenv("MIN_SIMILARITY_DISTANCE", "0.85"))

# Semantic-cache hit: distance < this means "same question, use cached answer"
CACHE_HIT_DISTANCE = float(os.getenv("CACHE_HIT_DISTANCE", "0.08"))

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(__file__)
VECTOR_DB_PATH      = os.path.join(_HERE, "data", "chroma_db")
PEOPLE_REGISTRY_PATH = os.path.join(_HERE, "data", "people.json")
SYNC_STATUS_PATH    = os.path.join(_HERE, "data", "sync_status.json")
FEEDBACK_LOG_PATH   = os.path.join(_HERE, "data", "feedback.jsonl")
FRONTEND_DIR        = os.path.join(_HERE, "..", "frontend")

# ── Semantic Scholar ──────────────────────────────────────────────────────────
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_API_KEY  = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
PAPER_FIELDS = "title,abstract,year,authors,venue,externalIds,openAccessPdf,publicationTypes"

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
_CONTEXT_BUDGET  = max(0, OLLAMA_NUM_CTX - 800 - 200 - 1500)
MAX_ABSTRACT_CHARS = int(os.getenv(
    "MAX_ABSTRACT_CHARS",
    str(max(300, int(_CONTEXT_BUDGET / max(1, RETRIEVAL_K) * 4))),
))

# ── Sidecar meta file (embedding-model mismatch guard) ────────────────────────
CHROMA_META_PATH = os.path.join(_HERE, "data", "chroma_meta.json")

# ── Admin / Rate-limiting ─────────────────────────────────────────────────────
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
if not ADMIN_PASSWORD:
    raise ValueError(
        "ADMIN_PASSWORD environment variable must be set. "
        "Set it to a strong password before deployment."
    )
RATE_LIMIT_QUERIES   = os.getenv("RATE_LIMIT_QUERIES", "10/minute")   # per IP
RATE_LIMIT_DEFAULT   = os.getenv("RATE_LIMIT_DEFAULT", "60/minute")


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
        raise ValueError(f"HNSW_CONSTRUCTION_EF must be >= 1, got {HNSW_CONSTRUCTION_EF}")
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


_validate_config()
