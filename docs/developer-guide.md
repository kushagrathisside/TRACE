# TRACE — Developer Guide

> **T**rustworthy **R**etrieval with **A**utomated **C**ontinuous **E**valuation

## What This Application Does

A web application where students describe a thesis/project idea and receive AI-generated guidance on related research done within the institute. Responses are fully attributed: paper titles, authors, years, venues, and specific faculty/students to consult. Data is sourced from Semantic Scholar using stored author IDs for all institute members.

---

## Repository Layout

```
trace/
├── backend/
│   ├── main.py                   FastAPI app — routes, middleware, startup
│   ├── config.py                 All settings, env-var overridable (validated on startup)
│   ├── llm_provider.py           Singleton factory: LLM, JSON-LLM, embeddings
│   ├── ingestion/
│   │   ├── scholar_client.py     Semantic Scholar API (incremental sync, retry on 429)
│   │   ├── people_registry.py    CRUD over data/people.json (file-locked)
│   │   └── ingestor.py           Sync pipeline + atomic status persistence
│   ├── rag/
│   │   ├── semantic_cache.py     Semantic query cache (ChromaDB-backed)
│   │   ├── reranker.py           Cross-encoder reranker (sentence-transformers)
│   │   ├── hybrid_search.py      BM25 index + Reciprocal Rank Fusion
│   │   ├── vector_store.py       Persistent ChromaDB, HNSW-tuned, corruption detection
│   │   ├── chain.py              Structured JSON generation + hallucination guard
│   │   └── pipeline.py           Full pipeline orchestration (8 stages)
│   ├── eval/
│   │   ├── ragas_scorer.py       Per-query RAGAS scoring (called from pipeline.py)
│   │   ├── deepeval_config.py    Local Ollama judge for DeepEval
│   │   ├── test_rag_regression.py pytest test suite (run in CI)
│   │   ├── run_eval.py           Full eval set runner → logs to MLflow
│   │   ├── self_retrieval.py     Zero-cost embedding sanity check (auto-run post-sync)
│   │   └── analyse_feedback.py   Mine feedback.jsonl for trends (via /api/feedback/analysis)
│   └── data/
│       ├── people.json           Faculty/student registry
│       ├── sync_status.json      Persisted sync state (atomic writes, survives restarts)
│       ├── feedback.jsonl        Thumbs-up/down log + optional RAGAS scores
│       ├── eval_set.json         Ground truth: (query, relevant_paper_ids) pairs
│       └── chroma_db/            Persistent vector index (created on first sync)
├── frontend/
│   ├── index.html                Student query page
│   ├── admin.html                Admin panel
│   └── static/style.css
├── requirements.txt
└── docs/
    ├── developer-guide.md        This file — architecture, design decisions, API reference
    ├── llmops-evaluation.md      Evaluation strategy: RAGAS, DeepEval, MLflow, TruLens
    └── feature-upgrades.md       Remaining feature upgrades and enhancement backlog
```

---

## Full Query Pipeline

```
Student types idea
      │
      ▼  (1) Embed query  →  cosine lookup in semantic cache
      │        Cache HIT → return cached answer instantly (~7 ms total)
      │        Cache MISS → continue
      ▼
      (2) Query expansion via LLM (temperature=0)
          "transformers for low-resource NLP"
          → "transformers, low-resource NLP, data augmentation,
              cross-lingual transfer, multilingual BERT, …"
      ▼
      (3) HNSW vector search  [k=20 candidates]
          all-MiniLM-L6-v2 bi-encoder, cosine distance
          ChromaDB: M=48, ef_construction=200, search_ef=150
      ▼
      (4) Similarity threshold guard
          Drop candidates with distance > MIN_SIMILARITY_DISTANCE (0.85)
          → caches NO_RESULTS and returns early if nothing close enough
      ▼
      (5) BM25 keyword search  [k=20 candidates]  +  RRF fusion
          rank_bm25 on title+abstract corpus
          Reciprocal Rank Fusion combines both ranked lists
      ▼
      (6) Cross-encoder reranking  [20 → top 5]
          cross-encoder/ms-marco-MiniLM-L-6-v2
          Reads query+doc jointly → accurate relevance scores
      ▼
      (7) LLM structured generation
          ChatOllama format="json", temperature=0
          System prompt with JSON schema example
          Pydantic ResearchLandscape model
          Hallucination guard: drops fabricated citations (substring + Jaccard ≥ 0.75)
      ▼
      (7b) Optional: RAGAS evaluation (if ENABLE_RAGAS_SCORING=true)
           Faithfulness + Answer Relevancy scores logged to feedback.jsonl
      ▼
      (8) Write to semantic cache  →  return to student
```

---

## Technology Choices and Rationale

### FastAPI
Async-first HTTP framework. `BackgroundTasks` handles the long sync job without blocking API responses. `run_in_executor` wraps the synchronous LLM call (blocking, ~10-30s) so the event loop stays free for other students' requests.

### Semantic Scholar API
Free official API from the Allen Institute for AI. No scraping, no proxies, good coverage of CS/engineering. Uses its own author IDs. Trade-off: author IDs differ from Google Scholar IDs, requiring a one-time admin lookup per person. Rate limits: 100 req/5 min without key, 10 req/s with key. The client retries up to 4 times on 429 with exponential backoff and surfaces a clear error if all retries are exhausted.

### ChromaDB with HNSW Tuning
Local persistent vector database. Default HNSW settings (M=16, ef=10) sacrifice recall for speed at small scale. We raise M=48 and search_ef=150 to recover ~5% more true nearest neighbours with <2ms overhead per query. Settings are frozen at collection creation — delete `data/chroma_db/` and re-sync to apply changes. A startup guard (`chroma_meta.json`) detects if the embedding model has been swapped without clearing the DB, raising a clear error rather than silently returning wrong results.

### HuggingFace `all-MiniLM-L6-v2` Embeddings
22M params, 384 dimensions, runs on CPU, ~5ms per query embed. `normalize_embeddings=True` ensures unit-norm vectors — required for ChromaDB's cosine distance formula (`1 - cosine_similarity`) to be exact. Trade-off vs. OpenAI embeddings: slightly lower recall on subtle semantic connections.

### Semantic Query Cache
A second ChromaDB collection (`query_cache`) stores query embeddings → answers. On a new query we embed it and check if any cached query has cosine distance < 0.08 (meaning >92% similar). Cache hits skip the entire 7-stage pipeline, returning in ~7ms vs. 30s. Cache is invalidated after every sync. "No results" responses are also cached — repeated out-of-domain queries skip the full pipeline instead of re-running it. This is the highest-ROI latency optimisation at low engineering cost.

### BM25 + Reciprocal Rank Fusion (Hybrid Search)
Dense retrieval misses exact named-entity matches ("Prof. Sharma", "AlexNet"). BM25 scores documents by term frequency. RRF combines the two ranked lists: `score(d) = Σ 1/(k + rank_i(d))` with k=60. Documents appearing in both lists are strongly boosted. BM25 index is built in-memory (~25 MB) from all ChromaDB documents on startup and after each sync.

### Cross-Encoder Reranking
Bi-encoder retrieval is fast but approximate — it encodes query and document independently. A cross-encoder reads them together and captures fine-grained token interactions. We retrieve 20 candidates cheaply, then rerank with `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, ~80ms for 20 pairs on CPU), keeping the top 5.

### Structured JSON Output + Hallucination Guard
`ChatOllama(format="json")` forces the model to emit valid JSON. A concrete schema example in the system prompt constrains the output shape. Pydantic validates and coerces the result. The exception handler now distinguishes `json.JSONDecodeError`, `pydantic.ValidationError`, and unexpected runtime errors separately, so failures are logged at the correct severity level.

The post-parse hallucination guard cross-checks every cited paper against the retrieved source documents. It uses a strict ID-based approach first:
1. **Exact `paper_id` Match**: The model is prompted to output the Semantic Scholar `paper_id` alongside the title. If the ID matches a retrieved source, it is 100% grounded.
2. **Fuzzy Fallback**: If the ID is malformed, it falls back to a title substring match or Jaccard word overlap (≥ 0.75).

Fabricated references are silently dropped. This handles the main failure mode of small models (llama3.2 3B) in RAG: citing plausible but non-existent papers.

### KV Cache (Ollama)
`keep_alive=-1` keeps the model resident in memory indefinitely. The critical benefit: Ollama caches the KV state for the system prompt prefix. Since the same prompt prefix is sent on every query, the ~800 system-prompt tokens are only encoded once and reused from KV cache on subsequent calls. `keep_alive=0` (default) would evict the model after 5 minutes, forcing a cold-start KV re-encode. The cost: persistent RAM/VRAM usage (~2GB for llama3.2 3B).

### Rate Limiting (slowapi)
`@limiter.limit("10/minute")` on `/api/query` uses `slowapi` (a FastAPI-compatible port of `flask-limiter`). Key function is remote IP address. Returns HTTP 429 with a proper `Retry-After` header. Admin endpoints (`/api/sync`, `/api/people`, etc.) are similarly rate-limited to 10 requests per minute to prevent brute-forcing the shared password.

### File Locking (portalocker)
`people.json` and `sync_status.json` operations utilize `portalocker` for cross-process file locking via sidecar `.lock` files. This prevents JSON corruption when two admins add/remove people simultaneously, and prevents multi-worker concurrency clashes if two admin requests attempt to trigger the sync pipeline at the same time. Reads are unlocked.

### Incremental Sync
`sync_status.json` persists `per_person[s2_id]["last_year"]` — the most recent publication year seen for each author. On the next sync, the Semantic Scholar API is called with `year=<last_year>-`, fetching only papers from that year onward. A new person always gets a full historical fetch. This turns a 3-hour full re-sync into a ~30-second delta sync.

The `per_person` map is only updated **after** all papers have been successfully upserted to ChromaDB. This ensures that if a sync crashes mid-way, the next sync will re-fetch the failed author's papers rather than skipping them. Status is persisted with an atomic rename so the file is never left in a corrupted half-written state.

### Orphan Cleanup
When a person is removed from the registry, `VectorStoreManager.remove_person_from_docs()` iterates ChromaDB metadata in Python (no $contains filter available) to find their papers, then either deletes (sole institute author) or updates metadata (co-authored with remaining members). The semantic cache and BM25 index are then rebuilt. A post-rebuild self-retrieval test runs automatically to verify the index is intact.

### CORS Middleware
`CORS_ORIGINS` is read from the environment variable of the same name. It must be set explicitly — the server refuses to start without it. Set to `"*"` only for development behind a private network; in production, restrict to `"https://institute.edu"`.

### Admin Authentication
The `X-Admin-Password` header is compared using `hmac.compare_digest()` rather than plain `==`, preventing timing attacks where an attacker can measure response time to guess the password character by character. The password itself must be set via the `ADMIN_PASSWORD` environment variable — there is no default; the server will refuse to start if the variable is absent.

---

## Configuration Reference

| Variable | Default | Required | What it controls |
|---|---|---|---|
| `ADMIN_PASSWORD` | — | **Yes** | Shared admin secret; no default, must be set |
| `CORS_ORIGINS` | — | **Yes** | Comma-separated list of allowed origins; `*` for dev |
| `INSTITUTE_NAME` | `"Our Institute"` | No | Appears in LLM prompt and page headers |
| `LLM_MODEL_NAME` | `"llama3.2"` | No | Ollama model for generation |
| `EMBEDDING_MODEL_NAME` | `"all-MiniLM-L6-v2"` | No | HuggingFace bi-encoder |
| `RERANKER_MODEL_NAME` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | No | Cross-encoder for reranking |
| `OLLAMA_BASE_URL` | `"http://localhost:11434"` | No | Ollama API base URL |
| `OLLAMA_KEEP_ALIVE` | `-1` | No | KV cache persistence; -1 = never evict |
| `OLLAMA_NUM_CTX` | `8192` | No | Context window (tokens) |
| `HNSW_M` | `48` | No | HNSW graph connectivity (recall vs. RAM) |
| `HNSW_CONSTRUCTION_EF` | `200` | No | Index build quality (one-time cost) |
| `HNSW_SEARCH_EF` | `150` | No | Query recall (vs. latency) |
| `RETRIEVAL_FETCH_K` | `20` | No | Candidates before reranking |
| `RETRIEVAL_K` | `5` | No | Final results after reranking |
| `MIN_SIMILARITY_DISTANCE` | `0.85` | No | Threshold to filter irrelevant docs |
| `CACHE_HIT_DISTANCE` | `0.08` | No | Semantic cache similarity threshold |
| `CACHE_MAX_AGE_DAYS` | `15` | No | Number of days before a cache entry expires |
| `RECENCY_WEIGHT` | `0.01` | No | RRF score bonus per year of recency |
| `MAX_ABSTRACT_CHARS` | computed | No | Characters of abstract per chunk in LLM prompt |
| `RATE_LIMIT_QUERIES` | `"10/minute"` | No | Per-IP rate limit on `/api/query` |
| `RATE_LIMIT_DEFAULT` | `"60/minute"` | No | Global default rate limit |
| `RATE_LIMIT_ADMIN` | `"10/minute"` | No | Global rate limit on admin routes |
| `SEMANTIC_SCHOLAR_API_KEY` | `""` | No | Optional; raises rate limit to 10 req/s |
| `ENABLE_RAGAS_SCORING` | `"false"` | No | Set to `"true"` to score every live query with RAGAS |

All optional variables are overridden via environment: `INSTITUTE_NAME="IIT Bombay" ADMIN_PASSWORD="secret" CORS_ORIGINS="*" uvicorn main:app`.

Config constraints are validated on startup. Mismatches (e.g. `RETRIEVAL_K > RETRIEVAL_FETCH_K`) cause an immediate `ValueError` with a descriptive message rather than silently producing bad results.

---

## API Reference

### Student (no auth required)

| Method | Path | Body / Response |
|---|---|---|
| `GET` | `/` | Student page HTML |
| `POST` | `/api/query` | `{idea}` → `{answer: ResearchLandscape, sources: [...], cached: bool}` |
| `POST` | `/api/feedback` | `{query, rating: "up"\|"down", comment}` → `{ok}` |

### Admin (all require `X-Admin-Password` header)

| Method | Path | Description |
|---|---|---|
| `GET` | `/admin` | Admin panel HTML |
| `GET` | `/health` | Subsystem health (chromadb, ollama) |
| `GET` | `/api/people` | List all registered people (paginated: `?page=1&page_size=100`) |
| `POST` | `/api/people` | Add person |
| `DELETE` | `/api/people/{id}` | Remove person + orphan cleanup |
| `POST` | `/api/sync` | Start incremental sync (background task) |
| `GET` | `/api/sync/status` | Live sync progress + per-person history |
| `GET` | `/api/stats` | Paper count, people count, last sync |
| `GET` | `/api/feedback/analysis` | Analyse feedback.jsonl — thumbs-down rate, RAGAS trends |
| `POST` | `/api/author/search` | Proxy to Semantic Scholar author search |

### `ResearchLandscape` response schema

```json
{
  "answer": {
    "landscape_summary": "2-3 sentence analysis",
    "related_papers": [
      {"title": "...", "year": 2023, "authors": "...", "venue": "...", "relevance": "..."}
    ],
    "people_to_consult": [
      {"name": "...", "role": "faculty", "department": "...", "relevant_work": "..."}
    ],
    "next_steps": ["step 1", "step 2"],
    "no_relevant_research": false
  },
  "sources": [{"title":"...", "year":"...", "venue":"...", "authors":"...", "institute_authors":"...", "url":"..."}],
  "cached": false
}
```

---

## Running the Application

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) running locally with the model pulled: `ollama pull llama3.2`

### Required Environment Variables

The server **will not start** without these two variables:

```bash
export ADMIN_PASSWORD="your-strong-password-here"
export CORS_ORIGINS="http://localhost:8000"   # or "*" for dev, "https://institute.edu" for prod
```

### Setup
```bash
cd trace
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Start
```bash
cd backend
ADMIN_PASSWORD="secret" CORS_ORIGINS="*" uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Student page: `http://localhost:8000/`
- Admin panel: `http://localhost:8000/admin` (use the password you set in `ADMIN_PASSWORD`)
- Health check: `http://localhost:8000/health`
- Auto-generated API docs: `http://localhost:8000/docs`

### First-time setup
1. Open `/admin` → add faculty/students using the Semantic Scholar search widget.
2. Click **Sync Now** → wait for completion (watch the status badge).
3. After sync, a self-retrieval test runs automatically and logs results — check logs for `PASS/FAIL`.
4. Go to `/` and submit a test idea.

---

## Common Extension Points

### Swap the LLM to a cloud model
In `llm_provider.py`, replace `ChatOllama` with `ChatOpenAI` from `langchain_openai`. No chain or pipeline changes needed.

### Add full-text indexing (not just abstracts)
In `ingestor.py`, for papers with `openAccessPdf` URL: download the PDF with `httpx`, parse with `PyMuPDFLoader`, chunk with `RecursiveCharacterTextSplitter`, store chunks as `{paper_id}_{chunk_index}` IDs.

### Replace the embedding model
Change `EMBEDDING_MODEL_NAME` in config. Then delete `data/chroma_db/` and `data/chroma_meta.json` (old vectors are incompatible) and re-sync. The startup guard will raise a clear error if you forget.

### Add automatic scheduled sync
Use `cron` or a systemd timer to `POST /api/sync` with the admin password header. Or add a `lifespan` startup task in `main.py` (already has the hook).

### Improve auth for public deployment
Replace the `X-Admin-Password` header check with OAuth2 or institute SSO (SAML/OIDC) using FastAPI's `OAuth2PasswordBearer` or a middleware that validates JWT tokens from the institute IdP.

### Enable live RAGAS evaluation
Set `ENABLE_RAGAS_SCORING=true`. Every query will be scored for faithfulness and answer relevancy using the local LLM as judge. Scores are appended to `feedback.jsonl` alongside thumbs ratings. View the trend via `GET /api/feedback/analysis`.

### Add RAGAS evaluation to the query path (manual)
The `eval/ragas_scorer.py` module already exists and is called automatically when `ENABLE_RAGAS_SCORING=true`. To call it manually:

```python
from eval.ragas_scorer import score_and_log

score_and_log(query, landscape.landscape_summary, final_docs)
```

---

## Evaluation Strategy

> **Full LLMOps evaluation guide** (RAGAS integration, DeepEval regression tests, MLflow experiment tracking, TruLens dashboard, ready-to-run scripts): see [llmops-evaluation.md](llmops-evaluation.md).

Evaluation has four independent layers. Each layer has its own ground truth requirements and metrics. The key insight: **retrieval quality is the ceiling for generation quality** — a perfect LLM cannot compensate for bad retrieval, so retrieval metrics should be instrumented first.

---

### Layer 1 — Retrieval Evaluation

Measures whether the right papers surface for a query, independent of the LLM.

#### Metrics

| Metric | What it measures | Notes |
|---|---|---|
| **Hit Rate@K** | Did at least one relevant paper appear in top K? | Easiest to compute, use as primary signal |
| **Precision@K** | Of K retrieved, what fraction are relevant? | Penalises irrelevant results |
| **Recall@K** | Of all relevant papers, how many were in top K? | Penalises missing relevant results |
| **NDCG@K** | Rank-aware quality — penalises rank-5 more than rank-1 | Best single metric if you have relevance grades |
| **MRR** | Mean Reciprocal Rank — where did the first relevant result land? | Good for "first hit" use cases |

#### Building Ground Truth (the hard part)

You need labelled `(query, relevant_paper_ids)` pairs. Three practical sources:

**1. Known supervisor–student pairs** (zero labelling effort)
If Prof. Sharma supervised a thesis on GNNs, her papers *must* appear when that thesis topic is queried. Every historical thesis title + advisor pairing is automatically correct ground truth.

```python
# Example ground truth entry
{
  "query": "Graph neural networks for social network analysis",
  "relevant_paper_ids": ["abc123", "def456"],  # Prof. Sharma's GNN papers
  "source": "supervisor_pair"
}
```

**2. Self-retrieval test** (fast, zero cost, imperfect)
For each paper in the DB, use its abstract as the query and check whether the paper retrieves itself in top K. A system with working embeddings should score >90% Hit Rate@1 on this test. This now runs **automatically after every sync** — watch the logs for the `PASS/FAIL` line. Run it manually with:

```bash
cd backend && python eval/self_retrieval.py
```

**3. Human labelling** (most accurate, one-time cost)
Ask 3–4 faculty to annotate 30–40 seed queries — mark which paper IDs are relevant for each. One afternoon of work, reusable forever as a regression test set.

Store in `backend/data/eval_set.json`:
```json
[
  {
    "id": "q001",
    "query": "federated learning for healthcare data privacy",
    "relevant_paper_ids": ["s2:abc123", "s2:def456"],
    "annotated_by": "Dr. Mehta",
    "date": "2026-06-06"
  }
]
```

#### Component Ablation Tests

Run the same eval set under different pipeline configurations to quantify what each component contributes:

| Experiment | Change | Metric to watch |
|---|---|---|
| Reranker ablation | Disable cross-encoder, keep top-20 by cosine | NDCG@5 |
| Hybrid vs. dense-only | Disable BM25, use pure vector search | Hit Rate@5 on queries with proper nouns |
| K sensitivity | Test K=3, 5, 7, 10 | NDCG@K vs. generation faithfulness |
| Threshold sensitivity | Vary `MIN_SIMILARITY_DISTANCE` 0.6→1.0 | Recall@5 vs. no-results rate |
| Query expansion on/off | Skip `_expand_query` step | Recall@5 on short/vague queries |

---

### Layer 2 — Generation Evaluation

Measures whether the LLM answer is accurate, grounded, and useful, independent of whether the right papers were retrieved.

#### Reference-Free Metrics (no ground truth needed)

**Faithfulness rate**
The hallucination guard in `chain.py` already tracks which cited papers pass or fail the grounding check. Log `valid / total_cited` per query. A rate below 0.9 means the model is frequently fabricating citations despite the JSON prompt.

**Context utilisation**
Of the 5 retrieved papers, how many are mentioned in `related_papers` in the answer? Near-zero means the LLM is ignoring the context; near-1.0 means it's using all of it (may indicate padding rather than selectivity).

```python
utilisation = len(landscape.related_papers) / len(final_docs)
```

**Structured output parse success rate**
Log whether `json.loads(response.content)` succeeded or fell back to raw text. A low parse rate (< 95%) means the model is not reliably following the JSON schema — try a larger model or a clearer schema prompt.

#### LLM-as-Judge

Use a stronger model (GPT-4o-mini, or locally `llama3.1:8b`) to score answers on three dimensions. This is the most useful single quality signal you can add without human raters.

```python
JUDGE_PROMPT = """
You are evaluating a RAG system answer for an institute research navigator.

Student's idea: {query}
System answer: {answer}

Rate from 1-5 (5=best) on:
1. Relevance: does the answer address the student's specific idea?
2. Attribution accuracy: do the cited papers sound plausible for this topic?
3. Actionability: would a student know what to do next after reading this?

Return JSON: {{"relevance": N, "attribution": N, "actionability": N, "notes": "..."}}
"""
```

Log scores to `feedback.jsonl` alongside the existing thumbs-up/down signal.

---

### Layer 3 — RAG-Specific Framework (RAGAS)

RAGAS is the standard library for end-to-end RAG evaluation. Install with `pip install ragas`.

| Metric | What it measures | Ground truth needed? |
|---|---|---|
| **Faithfulness** | Claims in answer are supported by retrieved context | No |
| **Answer Relevancy** | Answer actually addresses the question | No |
| **Context Precision** | Retrieved chunks are genuinely relevant | Yes |
| **Context Recall** | All relevant information was retrieved | Yes |

Enable automatic per-query RAGAS scoring with `ENABLE_RAGAS_SCORING=true`. When enabled, `eval/ragas_scorer.py` is called after every answer generation and scores are appended to `feedback.jsonl`. View aggregated trends via `GET /api/feedback/analysis`.

---

### Layer 4 — System-Level Analysis

#### Analyse feedback via the API

```bash
curl -H "X-Admin-Password: yourpassword" http://localhost:8000/api/feedback/analysis
```

Or run the script directly:
```bash
cd backend && python eval/analyse_feedback.py
```

Outputs thumbs-down rate, worst-performing queries, RAGAS score trends, and actionable recommendations.

**Topic cluster analysis**: embed all queries, cluster them (K-means, k=10), compute thumbs-down rate per cluster. Clusters with high down-rates reveal which topic domains the system is weakest on — guides where to add more faculty/papers.

#### Cache Effectiveness

Log and track over time:
- **Cache hit rate**: should grow as repeated/similar queries accumulate
- **False positive rate** (manual spot-check): did the cache return an answer to a genuinely different question?

```python
# In pipeline.run(), after the cache check:
logger.info(json.dumps({"event": "cache_check", "hit": hit is not None, "query": query[:80]}))
```

#### Per-Stage Latency Profiling

`pipeline.py` already logs `latency_ms` for the full pipeline. Extend it to log per-stage timing:

```python
t_embed    = time.perf_counter()
query_vec  = embeddings.embed_query(query)
t_cache    = time.perf_counter()
hit        = cache.get(query_vec)
t_expand   = time.perf_counter()
expanded   = _expand_query(query)
# ... etc.

logger.info(json.dumps({
    "event": "latency_breakdown",
    "embed_ms":    round((t_cache   - t_embed)   * 1000, 1),
    "cache_ms":    round((t_expand  - t_cache)   * 1000, 1),
    "expand_ms":   round((t_vector  - t_expand)  * 1000, 1),
    "vector_ms":   round((t_bm25    - t_vector)  * 1000, 1),
    "rerank_ms":   round((t_llm     - t_bm25)    * 1000, 1),
    "llm_ms":      round((t_end     - t_llm)     * 1000, 1),
}))
```

Plot P50/P95 per stage over real traffic. The stage with the highest P95 is the optimisation target.

---

### Practical Evaluation Roadmap

| When | What to build | Effort | Value |
|---|---|---|---|
| **Immediately** | Self-retrieval test (auto-runs post-sync) | — | Sanity-check embeddings work |
| **Immediately** | Feedback analysis: `GET /api/feedback/analysis` | — | Mine existing signal |
| **This week** | 25-query ground truth set (supervisor pairs) | 1 day with faculty | Permanent regression harness |
| **This week** | Hit Rate@5 + NDCG@5 eval runner | 2 hours | Quantify retrieval quality |
| **Next sprint** | Set `ENABLE_RAGAS_SCORING=true` | 5 minutes | Continuous quality monitoring |
| **Next sprint** | Per-stage latency logging in `pipeline.py` | 2 hours | Find bottleneck |
| **Later** | LLM-as-judge (3 dimensions) logged to feedback.jsonl | 1 day | Richer quality signal |
| **Later** | Reranker + hybrid search ablation study | 1 day | Justify component cost |
| **If scaling** | MLflow or W&B experiment tracking | 1–2 days | Compare prompt/model versions |

---

## Known Limitations and Open Problems

Understanding what the system *cannot* do well is as important as knowing what it does. These are honest limitations worth being aware of.

### Data Quality
- **Abstracts only.** The vector index contains titles and abstracts, not full paper text. A paper whose relevance lies in its methods section (not abstracted) may never be retrieved.
- **Semantic Scholar coverage.** Papers published in non-indexed conferences, technical reports, and most theses are invisible. Coverage is strongest for published CS/engineering work.
- **No thesis/project indexing.** Student theses that were never published are not on Semantic Scholar and therefore not in the DB. A separate manual ingestion pipeline would be needed.

### Retrieval
- **Query expansion can backfire.** For very specific technical queries, expanding with related keywords may dilute the embedding and pull in tangentially related papers. Consider making expansion conditional on query length (skip for queries > 20 words).
- **No recency bias.** A 2024 paper and a 2010 paper with identical abstracts score identically. The `RECENCY_WEIGHT` config adds a small RRF bonus for newer papers, but the effect is subtle. A more aggressive recency discount would help for "what's recent in X" queries.

### Generation
- **llama3.2 3B is a small model.** It reliably follows the JSON schema on well-formed queries but struggles with highly technical jargon, multi-hop reasoning, and very long context. Upgrading to llama3.1:8b or a cloud model is the fastest quality improvement available.
- **Structured output degrades with long context.** At ~6000 context tokens (5 papers × 1200 tokens each), llama3.2 3B occasionally truncates its JSON or omits fields. Reduce `RETRIEVAL_K` or truncate abstracts more aggressively if this occurs.

### Operations
- **Single admin password.** No audit trail of who made what change. For multi-admin deployments, replace with per-user API keys stored hashed.

> **Remaining feature upgrades and detailed enhancement backlog:** see [feature-upgrades.md](feature-upgrades.md).
