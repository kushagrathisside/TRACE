<div align="center">

# TRACE

**Trustworthy Retrieval with Automated Continuous Evaluation**

*A self-hosted research discovery engine for academic institutes —
with the measurement stack to prove it works.*

[![CI](https://github.com/kushagrathisside/TRACE/actions/workflows/ci.yml/badge.svg)](https://github.com/kushagrathisside/TRACE/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Runs offline](https://img.shields.io/badge/LLM-local%20via%20Ollama-success.svg)](https://ollama.ai)

[Quick Start](#quick-start) ·
[Architecture](#architecture-overview) ·
[Evaluation](docs/retrieval-experiments.md) ·
[API](docs/api-reference.md) ·
[Contributing](CONTRIBUTING.md)

</div>

---

Students describe a thesis or project idea in plain language. TRACE returns
relevant papers, the faculty to talk to, and concrete next steps — grounded
exclusively in real publications by members of their own institute, pulled from
Semantic Scholar. No cloud LLM, no data leaving the building.

The harder problem is not generating an answer. It is knowing whether the answer
is any good. So TRACE ships with the measurement layer most RAG systems skip:

| | |
|---|---|
| **Every query is traced** | Candidate set at each retrieval stage, per-stage latency, grounding counts, and the exact config that produced them — written to `retrieval_traces.jsonl` and keyed by a `query_id` returned to the browser |
| **Feedback is attributable** | A thumbs-down joins back to what was actually retrieved, so it can be attributed to a cache hit, an empty result set or a genuine ranking miss — not just to the query text |
| **Stages are priced** | `make ablate` reports what dense, BM25, fusion and the cross-encoder each contribute, in both quality and milliseconds |
| **Nothing is fabricated** | Cited papers *and* suggested people are verified against the retrieved context before display; the drop rate is a tracked metric |

### Why the evaluation is the interesting part

Building the retrieval pipeline took less effort than establishing that its
numbers meant anything. Along the way the harness was found comparing paper
titles against paper IDs — so every ranking metric read `0.0` regardless of
quality — and two successive ground-truth generators were found to leak, each
producing a confident result that reversed once the leak was removed.

[**docs/retrieval-experiments.md**](docs/retrieval-experiments.md) is the full
technical report: seven experiments, every configuration tried including the two
that failed, eleven measurement defects, and the evidence behind each adopted
setting.

---

## Quick Start

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.13 tested |
| [Ollama](https://ollama.ai) | Latest | Must be running before server start |
| Ollama model | `llama3.2:3b` | `ollama pull llama3.2:3b` |
| Ollama embeddings | `all-minilm` | `ollama pull all-minilm` |
| RAM | ~4 GB free | 3B model + embeddings + index |

### 1. Clone and Install

```bash
git clone https://github.com/kushagrathisside/TRACE.git
cd TRACE
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Required Environment Variables

The server refuses to start without these two variables:

```bash
export ADMIN_PASSWORD="your-strong-password"
export CORS_ORIGINS="http://localhost:8000"
```

For development, `CORS_ORIGINS="*"` is acceptable. In production, set it to your institute's domain.

### 3. Start

```bash
make dev          # reload enabled, reads .env
```

Or without make:

```bash
cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

First start downloads the cross-encoder (~90 MB) and warms both models, so give
it a minute before the first query.

### 4. Load some data

Pick whichever fits. Both take a few minutes, most of it Semantic Scholar rate
limits.

**A — Your own institute** (what the system is for):

1. Open `http://localhost:8000/admin`
2. Search for faculty and students by name and add them
3. Click **Sync Now** — wait for the badge to read `done`
4. Go to `http://localhost:8000/` and submit a test query

**B — A demo corpus** (fastest way to see it work, and what the published
evaluation numbers are measured on):

```bash
make seed        # registers 13 well-known AI/ML researchers, indexes ~1,100 papers
```

This queries the Semantic Scholar API directly; nothing is downloaded from us.
It exists because the evaluation needs a real corpus — synthetic abstracts are
too lexically clean, so every retriever scores well on them and comparisons
between configurations show no difference.

### 5. Check it came up correctly

```bash
curl -s localhost:8000/health | python -m json.tool
```

`checks.reranker.active` must be `true`. If it reads `degraded`, ranking has
fallen back to fusion order — the system still answers, but materially worse,
and every trace is tagged `reranker_active=false`. That state used to be
reported only by a single startup log line, which is how it went unnoticed in a
running deployment.

Behind a corporate proxy, add `--noproxy '*'` to any manual `curl` against
Ollama — otherwise you get the proxy's HTML error page back with exit code 0,
which looks exactly like success.

---

## Running with Docker

An alternative to installing Python and Ollama on the host. Everything runs in
containers; the only prerequisite is Docker.

```bash
cp .env.example .env        # set ADMIN_PASSWORD at minimum
docker compose up -d        # CPU
docker compose --profile gpu up -d   # NVIDIA GPU (needs the Container Toolkit)
```

Then open `http://localhost:8000`.

| Service | Role |
|---|---|
| `ollama` | Model server. Weights live in the `ollama-models` volume |
| `ollama-init` | One-shot: pulls `llama3.2:3b` and `all-minilm`, then exits |
| `trace` | The application. Waits for `ollama-init` to finish before starting |

First start downloads ~2.3 GB of model weights into a volume. That happens once.
`backend/data/` is bind-mounted, so your index, registry, traces and labels
survive rebuilds; the HuggingFace cache is a volume, so the cross-encoder is not
re-downloaded on restart.

**Behind a proxy**, pass it at build time — the Dockerfile takes it as a build
argument rather than baking it into the image, so an image built on a proxied
network still works elsewhere:

```bash
docker compose build \
  --build-arg HTTP_PROXY=$http_proxy \
  --build-arg HTTPS_PROXY=$https_proxy
```

> **Note:** the Docker path is provided for convenience and has not been
> end-to-end verified on a clean host — it was authored on a proxied network
> where the image build could not be completed. The compose file validates and
> the design is conventional, but treat the first run as untested. `make dev`
> on the host is the exercised path.

---

## Interface

### Student interface (`/`)

The student page is a single-input form. Students type a research idea in natural language and receive:

- **Landscape summary** — 2–3 sentences situating the idea within existing institute work
- **Related papers** — titles, authors, years, venues, and a relevance note
- **People to consult** — faculty and students whose work is most relevant, with department and role
- **Next steps** — concrete suggestions for moving the idea forward
- **Thumbs feedback** — up/down rating and optional comment

### Admin interface (`/admin`)

The admin panel is a password-protected dashboard for managing data:

- **People registry** — add/remove faculty and students by searching Semantic Scholar
- **Sync status** — live progress indicator for the ingestion pipeline; shows per-person paper counts
- **Stats** — total papers indexed, people registered, last sync time
- **Health** — ChromaDB, reranker and Ollama subsystem status

---

## Architecture Overview

```
                     ┌──────────────────────────────────────────┐
                     │             FastAPI Backend               │
  Student ──────────▶│  /api/query  (rate-limited: 10/min/IP)   │
  Admin ────────────▶│  /api/*      (X-Admin-Password header)    │
                     └──────────┬───────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────────┐
              │                 │                     │
              ▼                 ▼                     ▼
   ┌──────────────────┐  ┌────────────┐  ┌──────────────────────┐
   │  ChromaDB        │  │  Ollama    │  │  Semantic Scholar     │
   │  (HNSW index)    │  │  llama3.2  │  │  Graph API           │
   │  + query cache   │  │  + embeds  │  │  (ingestion only)    │
   └──────────────────┘  └────────────┘  └──────────────────────┘
```

### Query Pipeline

```
Embed query → Semantic cache check
                    ↓ MISS
Query expansion via LLM            (optional: ENABLE_QUERY_EXPANSION)
                    ↓
HNSW vector search (k=20)          ─┐
                    ↓               │ RETRIEVAL_MODE selects
Similarity threshold guard          │ dense | bm25 | hybrid
                    ↓               │
BM25 search + weighted RRF         ─┘
                    ↓
Similarity guard re-applied to BM25-only candidates
                    ↓
Cross-encoder reranking (top 5 of 20)  + optional MIN_RERANK_SCORE floor
                    ↓
LLM structured JSON generation
                    ↓
Hallucination guard (papers AND people)
                    ↓
Write to semantic cache → return {answer, sources, query_id}
```

Every query writes one record to `data/retrieval_traces.jsonl` containing the
candidate set at each stage, per-stage latency, the grounding counts and the
config that produced them. The `query_id` in the response links that record to
any feedback the student later submits — which is what makes a thumbs-down
attributable to a cache hit, an empty result set or a genuine ranking miss
rather than just to the text of the query.

Two guards are worth calling out because they are easy to get wrong:

* **The similarity guard runs twice.** Dense candidates are filtered before
  fusion; BM25-only candidates never faced that filter, so it is re-applied
  after fusion (one extra Chroma lookup). Without this, a keyword coincidence
  is enough to put an unrelated paper in front of the LLM.
* **BM25 candidates must score strictly above `BM25_MIN_SCORE`.** A query
  sharing no token with the corpus scores 0.0 against *every* document, and
  `rank_bm25` still returns a full top-k of those zeros — which RRF then ranks
  as highly as genuine hits.

Cache hits return in ~7 ms. Full pipeline runs in ~10–30 s (dominated by LLM
generation); `retrieval_ms` in each trace isolates the part that retrieval
tuning actually controls.

---

## Repository Structure

```
trace/
├── backend/
│   ├── main.py                   FastAPI application — routes, auth, middleware
│   ├── config.py                 All env-var settings + startup validation
│   ├── llm_provider.py           Singleton factory for LLM, embeddings, JSON-LLM
│   │
│   ├── ingestion/
│   │   ├── ingestor.py           Sync pipeline: Scholar API → ChromaDB
│   │   ├── scholar_client.py     Semantic Scholar client (retry, rate-limit backoff)
│   │   └── people_registry.py    CRUD over data/people.json with file locking
│   │
│   ├── rag/
│   │   ├── pipeline.py           Full query orchestration + tracing
│   │   ├── chain.py              LLM call, JSON parsing, hallucination guard
│   │   ├── vector_store.py       ChromaDB wrapper (HNSW-tuned, corruption detection)
│   │   ├── hybrid_search.py      BM25 index (title+abstract+authors) + weighted RRF
│   │   ├── reranker.py           Cross-encoder reranking (loud on fallback)
│   │   ├── trace.py              Per-query trace records + stage timers
│   │   └── semantic_cache.py     Query-level cache (ChromaDB collection)
│   │
│   ├── eval/
│   │   ├── metrics.py            Recall / nDCG / MRR / Precision + bootstrap CIs
│   │   ├── run_eval.py           Full eval set runner → MLflow, with slice breakdown
│   │   ├── ablate.py             Stage-wise ablation + parameter sweeps
│   │   ├── build_eval_set.py     Generates the labelled query set
│   │   ├── seed_corpus.py        Builds a realistic corpus from real AI/ML authors
│   │   ├── ragas_scorer.py       Live RAGAS scoring (opt-in via ENABLE_RAGAS_SCORING)
│   │   ├── analyse_feedback.py   Joins user ratings to traces
│   │   ├── self_retrieval.py     Embedding sanity check (auto-runs post-sync)
│   │   ├── deepeval_config.py    Local judge for DeepEval regression tests
│   │   └── test_rag_regression.py  pytest regression suite
│   │
│   └── data/                     Runtime data (git-ignored)
│       ├── people.json           Faculty/student registry
│       ├── sync_status.json      Incremental sync state (atomic writes)
│       ├── feedback.jsonl        User ratings + optional RAGAS scores
│       ├── retrieval_traces.jsonl  One record per query (all offline metrics read this)
│       ├── eval_set.json         Labelled ground truth (build_eval_set.py)
│       └── chroma_db/            Persistent vector index
│
├── frontend/
│   ├── index.html                Student query page (vanilla JS, no build step)
│   ├── admin.html                Admin dashboard (vanilla JS, no build step)
│   └── static/
│       └── style.css             Shared stylesheet
│
├── docs/
│   ├── developer-guide.md        Architecture, design decisions, full API reference
│   ├── llmops-evaluation.md      Metrics, ground truth, ablation and their biases
│   ├── retrieval-experiments.md  Formal report: every experiment, defect and result
│   └── feature-upgrades.md       Remaining enhancements and engineering backlog
│
├── requirements.txt
└── README.md                     This file
```

---

## Configuration

The server validates all config on startup. Mismatches raise a `ValueError` with a descriptive message.

### Required (no defaults — server will not start without these)

| Variable | Example | Purpose |
|---|---|---|
| `ADMIN_PASSWORD` | `"letmein-strong-42"` | Admin panel and all `/api/*` admin routes |
| `CORS_ORIGINS` | `"https://institute.edu"` or `"*"` | Comma-separated list of allowed browser origins |

### Optional (have defaults)

| Variable | Default | Purpose |
|---|---|---|
| `INSTITUTE_NAME` | `"TRACE-Institute"` | Appears in LLM prompt and page title |
| `LLM_MODEL_NAME` | `"llama3.2:3b"` | Ollama model used for generation and query expansion |
| `EMBEDDING_MODEL_NAME` | `"all-minilm"` | **Ollama** embedding model (`ollama pull all-minilm`) |
| `RERANKER_MODEL_NAME` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | Cross-encoder (sentence-transformers). Empty = run without reranking |
| `RERANKER_REQUIRED` | `"false"` | `"true"` makes an unloadable reranker a hard startup failure |
| `OLLAMA_BASE_URL` | `"http://localhost:11434"` | Ollama API base URL |
| `OLLAMA_KEEP_ALIVE` | `-1` | `-1` = keep model in VRAM forever (KV-cache benefit) |
| `OLLAMA_NUM_CTX` | `8192` | Context window in tokens |
| `RETRIEVAL_FETCH_K` | `20` | Candidates retrieved before reranking |
| `RETRIEVAL_K` | `5` | Final results after reranking |
| `MIN_SIMILARITY_DISTANCE` | `0.85` | Cosine distance above which results are discarded |
| `CACHE_HIT_DISTANCE` | `0.08` | Below this distance, a cached answer is returned |
| `RECENCY_WEIGHT` | `0.01` | RRF score bonus per year of recency |
| `MAX_PAPERS_PER_PERSON` | `200` | Cap on papers fetched per person (bounds index size and sync memory; `0` = unlimited) |
| `RATE_LIMIT_QUERIES` | `"10/minute"` | Per-IP rate limit on `/api/query` and `/api/feedback` |
| `SEMANTIC_SCHOLAR_API_KEY` | `""` | Optional S2 key: raises limit from 100/5min to 10/s |
| `ENABLE_RAGAS_SCORING` | `"false"` | Set `"true"` to score every query with RAGAS |

> **Note on model names.** Embeddings run through Ollama, not HuggingFace, so
> `EMBEDDING_MODEL_NAME` must be an Ollama model tag (`all-minilm`), not a
> HuggingFace repo path. The reranker is the exception — it loads through
> `sentence-transformers` and does take a HuggingFace path.

### Retrieval ablation axes

These change what the pipeline does. They exist so `eval/ablate.py` can measure
the marginal contribution of each stage; every value is recorded in the query
trace and in MLflow run params.

| Variable | Default | Purpose |
|---|---|---|
| `RETRIEVAL_MODE` | `"hybrid"` | `dense` \| `bm25` \| `hybrid` |
| `ENABLE_QUERY_EXPANSION` | `"true"` | LLM keyword expansion before dense search (~800 ms) |
| `BM25_MIN_SCORE` | `0.0` | BM25 candidates must score strictly above this |
| `RRF_K` | `60` | RRF rank damping constant |
| `RRF_WEIGHT_DENSE` / `RRF_WEIGHT_SPARSE` | `1.0` / `1.0` | Weighted fusion; equal weights = textbook RRF |
| `ENFORCE_GUARD_POST_FUSION` | `"true"` | Re-apply the distance guard to BM25-only candidates |
| `MIN_RERANK_SCORE` | *(unset)* | Cross-encoder score floor; unset = always return `RETRIEVAL_K` |

---

## API Reference

### Student Endpoints (no auth)

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Student query page |
| `POST` | `/api/query` | `{"idea": "..."}` → `{answer, sources, cached, query_id}` |
| `POST` | `/api/feedback` | `{"query", "rating": "up"\|"down", "comment", "query_id"}` → `{ok}` |

### Admin Endpoints (`X-Admin-Password: <password>` required)

| Method | Path | Description |
|---|---|---|
| `GET` | `/admin` | Admin panel |
| `GET` | `/health` | Subsystem health: chromadb, reranker, ollama |
| `GET` | `/api/people` | List registered people (`?page=1&page_size=100`) |
| `POST` | `/api/people` | Add a person |
| `DELETE` | `/api/people/{id}` | Remove a person and clean up their papers |
| `POST` | `/api/sync` | Trigger incremental sync (background task) |
| `GET` | `/api/sync/status` | Live sync progress |
| `GET` | `/api/stats` | Papers indexed, people, last sync timestamp |
| `GET` | `/api/feedback/analysis` | Feedback trends: thumbs-down rate, RAGAS scores |
| `POST` | `/api/author/search` | `{"name": "..."}` → Semantic Scholar author search |
| `GET` | `/docs` | Auto-generated OpenAPI interactive docs |

---

## Component Status

### Backend

| Component | File | Status |
|---|---|---|
| RAG pipeline (traced, ablatable) | `rag/pipeline.py` | Stable |
| HNSW vector store | `rag/vector_store.py` | Stable, corruption detection added |
| BM25 + RRF hybrid search | `rag/hybrid_search.py` | Stable |
| Cross-encoder reranker | `rag/reranker.py` | Stable |
| Semantic query cache | `rag/semantic_cache.py` | Stable, no-results now cached |
| LLM chain + hallucination guard | `rag/chain.py` | Stable, guard threshold raised to 0.75 |
| Incremental sync | `ingestion/ingestor.py` | Stable, atomic writes, year=0 fix |
| Semantic Scholar client | `ingestion/scholar_client.py` | Stable, retry with context-aware 429 errors |
| People registry | `ingestion/people_registry.py` | Stable, file-locked |
| Config validation | `config.py` | Validates on startup |
| Admin auth | `main.py` | `hmac.compare_digest`, required env var |
| CORS | `main.py` | Env-var configured, no unsafe defaults |
| Rate limiting | `main.py` | slowapi, per-IP |
| Health check | `main.py` | ChromaDB + Ollama, timeout-aware |
| RAGAS integration | `eval/ragas_scorer.py` | Opt-in via `ENABLE_RAGAS_SCORING=true` |
| Self-retrieval test | `eval/self_retrieval.py` | Auto-runs post-sync |
| Feedback analysis API | `main.py` + `eval/analyse_feedback.py` | `GET /api/feedback/analysis` |
| DeepEval regression tests | `eval/test_rag_regression.py` | Run with `pytest eval/` |
| MLflow eval runner | `eval/run_eval.py` | Requires `data/eval_set.json` |

### Known Issues / In-Progress

| Issue | Location | Severity |
|---|---|---|
| No per-user API keys — single shared password | `config.py`, `main.py` | High |
| Sync degraded status not surfaced in UI | `ingestor.py`, `admin.html` | Medium |
| No rate limit on admin endpoints | `main.py` | Medium |
| Metadata not schema-validated (bare dicts) | `ingestor.py`, `rag/vector_store.py` | Medium |
| Cache entries never expire by age | `rag/semantic_cache.py` | Low |
| No concurrent sync lock (multi-worker) | `ingestor.py` | Low |

See [docs/feature-upgrades.md](docs/feature-upgrades.md) for the full backlog.

---

### Frontend

### `frontend/index.html` — Student Query Page

**Status: Stable, vanilla JS, no build step**

Features:
- Single-input query form
- Displays landscape summary, papers, people, next steps
- Shows `cached: true` indicator when a cache hit is returned
- Thumbs-up/down feedback widget with optional comment
- Error states: empty input, 429 rate-limit, 503 Ollama unavailable, 500 server error

Known gaps:
- No query history within session
- No dark mode
- Feedback submission does not animate

### `frontend/admin.html` — Admin Dashboard

**Status: Stable, vanilla JS, no build step**

Features:
- Login gate (password stored in `sessionStorage` for the session only)
- People registry: add by name search, remove with confirmation
- Semantic Scholar author search widget
- Sync trigger + live status polling
- Stats panel (total papers, people, last sync)
- Health check display

Known gaps:
- Sync errors not shown (API returns them, UI does not render them)
- No feedback analysis dashboard (available via API, not wired in UI)
- No dark mode

### `frontend/static/style.css`

**Status: Stable, single shared stylesheet**

- Responsive layout (mobile-friendly)
- No build step required — no SASS, no PostCSS
- No dark mode support

---

## Development Notes

### Running the Eval Suite

```bash
cd backend

# ── One-time: build a realistic corpus ────────────────────────────────────
# Registers well-known AI/ML researchers as TRACE-Institute members and indexes
# their real publication records from the Semantic Scholar API. Synthetic
# abstracts are too lexically clean — every retriever scores well on them and
# the ablation shows no differences between configurations.
#
# The corpus is rebuilt, never downloaded: shipping ~1,100 publisher abstracts
# would be redistributing third-party text. The hand-judged relevance labels
# ARE committed (data/eval_labels.json) and are re-applied automatically, so a
# clean clone reproduces the published numbers with no manual labelling.
python eval/seed_corpus.py --ingest          # bounded by MAX_PAPERS_PER_PERSON
python eval/build_eval_set.py                # exact_title / author / topic / manual

# ── Offline metrics ───────────────────────────────────────────────────────
python eval/run_eval.py --run-name baseline --no-ragas   # retrieval only, fast
python eval/run_eval.py --run-name baseline              # + LLM-judge metrics
mlflow ui --backend-store-uri data/mlruns

# ── Stage-wise ablation: what does each retrieval stage actually buy? ─────
python eval/ablate.py --markdown
python eval/ablate.py --sweep fetch_k --values 5,10,20,40,80

# ── Index integrity + online health ───────────────────────────────────────
python eval/self_retrieval.py
python eval/analyse_feedback.py
curl -H "X-Admin-Password: yourpassword" http://localhost:8000/api/feedback/analysis

# ── Regression test suite ─────────────────────────────────────────────────
python -m pytest eval/test_rag_regression.py -v
```

**Measure the baseline before you fix anything.** When changing retrieval, run
`run_eval.py` *first*, against the unfixed system. That broken baseline is the
"before" column of every comparison worth showing; fix everything first and you
have one number and no way to attribute it.

**Reading the ablation table**: `Recall@20` cannot change between the `hybrid`
and `hybrid+ce` rows — the cross-encoder reorders the same 20 candidates, so it
is mathematically pinned. Its contribution shows up only at the truncation
depth (`Recall@5`, `nDCG@5`, `MRR@5`). A table reporting only `Recall@fetch_k`
will always claim the reranker does nothing.

### Changing the Embedding Model

```bash
# 1. Update config
export EMBEDDING_MODEL_NAME="BAAI/bge-small-en-v1.5"

# 2. Delete old index (vectors are incompatible between models)
rm -rf backend/data/chroma_db backend/data/chroma_meta.json

# 3. Restart and re-sync
uvicorn main:app ...
# POST /api/sync via admin panel
```

If you skip step 2, the startup guard reads `chroma_meta.json` and raises a clear `RuntimeError` describing the mismatch.

### Swapping the LLM

```bash
# Pull the new model
ollama pull llama3.1:8b

# Set env var and restart
LLM_MODEL_NAME=llama3.1:8b ADMIN_PASSWORD=secret CORS_ORIGINS=* uvicorn main:app --reload
```

No code changes required — the LLM swap is fully config-driven.

### Debugging a Poor Query Result

1. Check `/health` — ensure both `chromadb` and `ollama` are `"ok"`
2. Check logs for `"event": "query"` — look at `retrieved` count and `top_rerank_score`
3. If `retrieved: 0`, lower `MIN_SIMILARITY_DISTANCE` (try 0.95)
4. If top score is low (<0.3), try rephrasing the query or expanding the BM25 index
5. If the answer cites wrong papers, check the hallucination guard log for `Hallucination dropped:`
6. Run `python eval/self_retrieval.py` to verify the embedding index is intact

---

## Deployment Checklist

Before going to production:

- [ ] Set `ADMIN_PASSWORD` to a strong, unique value
- [ ] Set `CORS_ORIGINS` to your institute's actual domain (not `*`)
- [ ] Set `INSTITUTE_NAME` to your institute's name
- [ ] Run `ollama pull llama3.2` on the production server
- [ ] Run at least one sync and verify paper count > 0
- [ ] Confirm `/health` returns `"status": "ok"` for all subsystems
- [ ] Run `python eval/self_retrieval.py` and verify `PASS`
- [ ] Test a student query end-to-end
- [ ] Set up a weekly cron job for sync: `curl -X POST .../api/sync -H "X-Admin-Password: ..."`
- [ ] Consider `ENABLE_RAGAS_SCORING=true` for continuous quality monitoring
- [ ] Restrict Ollama to `localhost` only if running on a shared server

---

## Documentation

| Document | What it covers |
|---|---|
| [**retrieval-experiments.md**](docs/retrieval-experiments.md) | **Technical report** — every experiment run, defect found and configuration adopted, with the evidence |
| [llmops-evaluation.md](docs/llmops-evaluation.md) | Metrics, ground-truth construction, ablation method, and the biases that make RAG evaluation lie |
| [developer-guide.md](docs/developer-guide.md) | Architecture reference, design decisions, technology rationale |
| [architecture.md](docs/architecture.md) | System diagram and component responsibilities |
| [api-reference.md](docs/api-reference.md) | Every endpoint, request and response shape |
| [data-ingestion.md](docs/data-ingestion.md) | Sync lifecycle, incremental fetching, orphan cleanup |
| [setup-guide.md](docs/setup-guide.md) | Local setup, including WSL and corporate-proxy notes |
| [deployment-guide.md](docs/deployment-guide.md) | Production configuration, reverse proxy, worker constraints |
| [scalability-requirements.md](docs/scalability-requirements.md) | Resource requirements and known scaling bottlenecks |
| [feature-upgrades.md](docs/feature-upgrades.md) | Engineering backlog with effort estimates |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Branch naming, test conventions, PR expectations |

---

## Contributing

Issues and pull requests are welcome. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) first — in particular, a pull request that
changes retrieval behaviour should include `make ablate` output from before and
after the change. A retrieval change without a before/after table is not
reviewable.

---

## License

Licensed under the [Apache License 2.0](LICENSE).

Paper metadata is retrieved from the [Semantic Scholar Academic Graph API](https://www.semanticscholar.org/product/api)
at runtime and is subject to their terms. This repository redistributes no
paper text: the evaluation corpus is rebuilt from the API by `make seed`.
