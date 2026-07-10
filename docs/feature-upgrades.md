# TRACE — Feature Upgrades Backlog

This document tracks all remaining feature upgrades, enhancements, and open engineering tasks. Items are grouped by area and ordered by impact within each group.

Last reviewed: 2026-06-09

---

## Priority Legend

| Symbol | Meaning |
|---|---|
| 🔴 HIGH | Significant impact on reliability, security, or quality |
| 🟡 MEDIUM | Clear improvement, reasonable effort |
| 🟢 LOW | Nice-to-have, low urgency |

---

## 1. Security & Authentication

### 🔴 Per-User Admin API Keys
**Problem:** A single shared password means no audit trail of which admin added or removed a person, or triggered a sync. One leaked password gives full access.

**Implementation:**
- Replace `ADMIN_PASSWORD` with a `data/api_keys.json` file: `{"key_hash": "sha256...", "name": "Dr. Mehta", "created": "..."}`
- Hash keys at rest with `hashlib.sha256`; compare with `hmac.compare_digest`
- Log `admin_name` on every admin action to a separate `audit.jsonl`
- Add `POST /api/keys` (bootstrap-only, gated on a one-time setup token) and `DELETE /api/keys/{name}`

**Effort:** 1 day

---


## 2. Data Quality & Ingestion


### 🟡 Full-Text Indexing (PDF Abstracts)
**Problem:** Many papers have relevant content in their body that is not captured in the abstract. The system can only see what Semantic Scholar abstracts expose.

**Implementation:**
- In `ingestor.py`, for papers with `openAccessPdf.url`: download with `httpx` (async, with timeout)
- Parse with `PyMuPDFLoader` (add `pymupdf` to requirements)
- Chunk with `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)`
- Store chunks as IDs `{paper_id}_{chunk_index}`
- Upsert alongside abstract chunk

**Effort:** 1 day | **Dependencies:** `pymupdf`, async download queue

---

### 🟡 No Thesis/Project Indexing
**Problem:** Student theses never published to Semantic Scholar are completely invisible to the system.

**Implementation:**
- Add a `POST /api/papers/manual` endpoint (admin-only) accepting `{title, abstract, authors, year, venue, url}`
- Assign a synthetic `paper_id` prefixed `"manual:"` to avoid collision with S2 IDs
- These papers are never touched by incremental sync (no S2 ID to query against)
- Admin panel gets a manual paper upload form

**Effort:** 1 day

---

### 🟢 Email Validation in People Registry
**Problem:** `add_person()` accepts any string as email with no format validation.

**Implementation:**
- Add `pydantic.EmailStr` type to the `PersonRequest` model in `main.py`
- Install `email-validator` (already a Pydantic optional dep)
- Invalid emails return HTTP 422 with a clear message

**Effort:** 30 minutes

---

## 3. Retrieval Quality

### 🟡 Conditional Query Expansion
**Problem:** For very specific technical queries (>20 words), query expansion may dilute the embedding by adding loosely related keywords, pulling in tangentially relevant papers.

**Implementation:**
- In `pipeline._expand_query()`, check `len(query.split()) > 20`
- If over the threshold, skip expansion and return the original query unchanged
- Add `QUERY_EXPANSION_MAX_WORDS` to `config.py` (default 20)

**Effort:** 1 hour

---


### 🟡 Recency Bias Tuning
**Problem:** `RECENCY_WEIGHT=0.01` adds a subtle bonus to newer papers but is likely too small to meaningfully influence rankings. "What's recent in X" queries still surface old papers.

**Implementation:**
- Log per-query `year_distribution` of final_docs (min, max, mean) to see the actual effect
- Run ablation: compare NDCG@5 with `RECENCY_WEIGHT` in [0, 0.01, 0.05, 0.1]
- Consider a sigmoid decay: `recency_score = 1 / (1 + exp(-0.5 * (year - mean_year)))`
- Expose as `RECENCY_DECAY_FUNCTION` config option

**Effort:** 4 hours

---

### 🟢 Per-Stage Latency Logging
**Problem:** `pipeline.py` logs total latency but not per-stage. The P95 bottleneck is unknown without profiling.

**Implementation:**
- Add `time.perf_counter()` checkpoints around each stage
- Log a `latency_breakdown` JSON event: `{"embed_ms":5, "cache_ms":2, "expand_ms":800, ...}`
- Use these to identify whether to optimise embedding, BM25, or reranking first

**Effort:** 1 hour

---

## 4. Generation Quality

### 🟡 Larger LLM for Production
**Problem:** `llama3.2 3B` frequently struggles with highly technical jargon, multi-hop reasoning, and JSON schema compliance at long contexts.

**Implementation options (in order of ease):**
1. Switch to `llama3.1:8b` — same Ollama setup, noticeably better JSON compliance
2. Switch to a cloud model: replace `ChatOllama` with `ChatAnthropic` (`claude-haiku-4-5`) or `ChatOpenAI` in `llm_provider.py`
3. No chain or pipeline changes required — the factory pattern in `LLMProvider` isolates the swap

**Effort:** 30 minutes (config change) to 2 hours (cloud model key setup)

---

### 🟡 LLM-as-Judge Scoring
**Problem:** Thumbs-up/down is too coarse. Two queries may both get a thumbs-up but one has much better attribution than the other.

**Implementation:**
Add `eval/llm_judge.py`:
```python
JUDGE_PROMPT = """
Rate from 1-5 on:
1. Relevance: does the answer address the student's idea?
2. Attribution accuracy: do cited papers sound plausible?
3. Actionability: would a student know what to do next?
Return JSON: {"relevance": N, "attribution": N, "actionability": N}
"""
```
- Call after every answer generation if `ENABLE_LLM_JUDGE=true`
- Use a separate, stronger judge LLM (configurable via `JUDGE_LLM_MODEL`)
- Append scores to `feedback.jsonl`

**Effort:** 1 day

---

### 🟢 Structured Output Fallback Retry
**Problem:** When `json.JSONDecodeError` or `ValidationError` is raised, the system immediately falls back to returning the raw LLM string. A single retry with a simplified prompt often recovers valid JSON.

**Implementation:**
- On parse failure, retry once with a shorter prompt: `"Fix the JSON you returned. Return only valid JSON matching this schema: {schema}. Your broken output was: {raw}"`
- If retry also fails, use the current raw-text fallback

**Effort:** 2 hours

---

## 5. Operations & Observability

### 🟡 MLflow Experiment Tracking Integration
**Problem:** There is no way to compare two configurations (e.g., K=5 vs K=7, or llama3.2 vs llama3.1:8b) side-by-side with quantitative metrics.

**Implementation:**
- `eval/run_eval.py` already exists with the full MLflow integration
- Missing step: run it as part of a pre-deployment checklist
- Add `make eval` target or `scripts/run_eval.sh` that executes: `python eval/run_eval.py --run-name "$(git rev-parse --short HEAD)"`
- View results: `mlflow ui --backend-store-uri data/mlruns`

**Effort:** 1 hour (integration) + eval set creation (1 day with faculty)

---

### 🟡 DeepEval CI Regression Tests
**Problem:** `eval/test_rag_regression.py` exists but is not wired to CI. Changes to prompts, models, or retrieval parameters can silently degrade quality.

**Implementation:**
- Add `.github/workflows/eval.yml` that runs `pytest eval/test_rag_regression.py -v` on PRs
- Gate merges on test pass
- Start with 3–5 test cases; grow the suite as regressions are found

**Effort:** 2 hours (CI config) + test case creation

---

### 🟡 Health Check Improvements
**Problem:** `/health` checks ChromaDB and Ollama but not: whether the BM25 index is built, whether the embedding model is loaded, or whether any papers exist in the DB.

**Implementation:**
```python
checks["bm25"]      = {"status": "ok" if HybridSearcher.get() else "not_built"}
checks["papers"]    = {"status": "ok" if vs.count() > 0 else "empty", "count": vs.count()}
checks["embedding"] = {"status": "ok"}  # fails on exception during warmup
```

**Effort:** 1 hour

---

### 🟢 Structured Logging Format
**Problem:** Log output is a mix of JSON-structured events and freeform strings, making it hard to query in log aggregators (Grafana Loki, Datadog, CloudWatch).

**Implementation:**
- Add `python-json-logger` to requirements
- Replace `basicConfig` in `main.py` with a `JsonFormatter`
- All `logger.info/warning/error` calls already use `json.dumps(...)` for structured events — standardise the remaining freeform strings

**Effort:** 2 hours

---

## 6. Frontend

### 🟡 Admin Panel: Show Sync Error Details
**Problem:** When a sync partially fails, the admin panel shows the status badge but not which people failed or why.

**Implementation:**
- `GET /api/sync/status` already returns `sync_status["errors"]`
- Add an expandable "Errors" section in `admin.html` that lists per-person errors when `status == "partial"`
- Highlight the status badge in amber for `"partial"`, red for `"error"`

**Effort:** 2 hours (frontend JS)

---

### 🟡 Query History for Students
**Problem:** Students cannot see their previous queries or revisit past answers in the same session.

**Implementation:**
- Store the last N queries in `sessionStorage` (browser, no server changes)
- Render a collapsible "Recent Queries" sidebar in `index.html`
- Click a past query to re-run it (or show cached answer instantly)

**Effort:** 3 hours (frontend JS, no backend changes)

---

### 🟡 Feedback Analysis Dashboard (Admin Panel)
**Problem:** `GET /api/feedback/analysis` returns plain text. Admins must read raw output.

**Implementation:**
- Parse the JSON analysis response in `admin.html`
- Render: thumbs-down rate as a gauge, worst queries as a table, RAGAS score trend as a sparkline
- Add a "Run Analysis" button in the admin panel that calls the endpoint

**Effort:** 4 hours (frontend JS + chart library, e.g. Chart.js via CDN)

---

### 🟢 Dark Mode
**Problem:** `static/style.css` has no dark mode support.

**Implementation:**
- Add `@media (prefers-color-scheme: dark)` CSS block
- Toggle via a button stored in `localStorage`

**Effort:** 2 hours

---

## 7. Infrastructure & Deployment

### 🟡 Docker Compose Setup
**Problem:** Setting up the full stack (Python venv + Ollama) requires manual steps. A `docker-compose.yml` makes it reproducible.

**Implementation:**
```yaml
# docker-compose.yml
services:
  app:
    build: .
    ports: ["8000:8000"]
    environment:
      - ADMIN_PASSWORD=${ADMIN_PASSWORD}
      - CORS_ORIGINS=${CORS_ORIGINS}
      - OLLAMA_BASE_URL=http://ollama:11434
    volumes:
      - ./backend/data:/app/data
  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama
```

**Effort:** 4 hours

---

### 🟡 Automatic Scheduled Sync
**Problem:** Admins must manually trigger syncs. New papers from the last week are invisible until someone remembers to sync.

**Implementation options:**
1. `cron` job: `0 2 * * * curl -X POST http://localhost:8000/api/sync -H "X-Admin-Password: $ADMIN_PASSWORD"`
2. Add a `SYNC_INTERVAL_HOURS` config and a background asyncio task in `lifespan()` in `main.py`:
   ```python
   async def _auto_sync_loop():
       while True:
           await asyncio.sleep(config.SYNC_INTERVAL_HOURS * 3600)
           ingestor.run_ingestion()
           pipeline.post_sync_rebuild()
   asyncio.ensure_future(_auto_sync_loop())
   ```

**Effort:** 1 hour

---

### 🟢 Database Migration Strategy
**Problem:** If the ChromaDB schema or document metadata fields change, there is no migration path. The only option is to delete and re-sync.

**Implementation:**
- Add a `schema_version` field to `chroma_meta.json`
- On startup, compare stored version to current version
- If mismatch: log a clear error with migration instructions rather than silently degrading
- Long-term: write migration scripts for common schema changes (adding a field with a default value)

**Effort:** 4 hours

---

## Summary Table

| Area | Item | Priority | Effort |
|------|------|----------|--------|
| Security | Per-user API keys | 🔴 | 1 day |

| Data | Full-text PDF indexing | 🟡 | 1 day |
| Data | Manual thesis ingestion | 🟡 | 1 day |
| Data | Email validation | 🟢 | 30 min |
| Retrieval | Conditional query expansion | 🟡 | 1 hr |

| Retrieval | Recency bias tuning | 🟡 | 4 hrs |
| Retrieval | Per-stage latency logging | 🟢 | 1 hr |
| Generation | Upgrade to llama3.1:8b | 🟡 | 30 min |
| Generation | LLM-as-judge scoring | 🟡 | 1 day |
| Generation | Structured output retry | 🟢 | 2 hrs |
| Ops | MLflow experiment tracking | 🟡 | 1 hr + 1 day |
| Ops | DeepEval CI tests | 🟡 | 2 hrs |
| Ops | Health check improvements | 🟡 | 1 hr |
| Ops | Structured JSON logging | 🟢 | 2 hrs |
| Frontend | Sync error details UI | 🟡 | 2 hrs |
| Frontend | Student query history | 🟡 | 3 hrs |
| Frontend | Feedback dashboard | 🟡 | 4 hrs |
| Frontend | Dark mode | 🟢 | 2 hrs |
| Infra | Docker Compose | 🟡 | 4 hrs |
| Infra | Auto scheduled sync | 🟡 | 1 hr |
| Infra | DB migration strategy | 🟢 | 4 hrs |
