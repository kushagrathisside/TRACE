# TRACE Setup Guide

This guide covers how to set up and run the TRACE server locally using WSL on Windows, optimized for the lowest possible resource usage.

## 1. Prerequisites

1. **WSL (Windows Subsystem for Linux)**: Ensure you have Ubuntu installed in WSL.
2. **Python 3.10+**: TRACE requires Python 3.10 or newer (tested with 3.13).
3. **Ollama**: Download and install [Ollama for Windows](https://ollama.ai/download/windows).

## 2. Pulling the AI Models

Open a PowerShell or WSL terminal and pull both required models:
```bash
ollama pull llama3.2:3b     # generation + query expansion
ollama pull all-minilm      # embeddings
```
Embeddings are served by **Ollama**, not HuggingFace, so `EMBEDDING_MODEL_NAME`
must be an Ollama tag (`all-minilm`) — a HuggingFace repo path such as
`all-MiniLM-L6-v2` will not resolve. The cross-encoder reranker is the one
exception: it loads through `sentence-transformers` and does take a HuggingFace
path, downloading ~90 MB on first use.

## 3. Configuration

1. Copy `.env.example` to `.env` in the root of the TRACE directory.
2. Open `.env` and configure your settings.
3. **Crucially**, you must set the `ADMIN_PASSWORD` variable. The server will refuse to start without it.

```env
# ── Models ──
LLM_MODEL_NAME=llama3.2:3b
EMBEDDING_MODEL_NAME=all-minilm
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2

# ── Admin ──
ADMIN_PASSWORD=your_secure_password
```

### About the reranker

The cross-encoder is ~22 M parameters (~90 MB) and adds roughly 70 ms per query
on CPU. It is the single largest contributor to ranking quality in this
pipeline, so leaving it enabled is strongly recommended even on a laptop.

If you must run without it, set `RERANKER_MODEL_NAME=` (empty). That is treated
as a deliberate choice rather than a failure — but ranking then falls back to
fusion order, `/health` reports `reranker.active: false`, and every trace record
is tagged `reranker_active=false` so no evaluation result can be mistakenly
attributed to a reranker that was not running.

### Behind a corporate proxy

Ollama runs on localhost and must not be proxied. TRACE appends
`localhost,127.0.0.1,::1` to `no_proxy` at import time. It deliberately does
*not* delete your proxy variables: doing so previously broke outbound HTTPS for
Semantic Scholar ingestion, which surfaced as SSL handshake timeouts pointing
nowhere near the cause.

## 4. First-Time Python Setup

Open your WSL terminal and navigate to the TRACE directory:
```bash
cd /home/yourusername/TRACE
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 5. Starting the Server (The Easy Way)

```bash
make dev          # reload enabled, reads .env
```

Ensure Ollama is running first — on Windows that means the app is live in the
system tray; WSL reaches it on `localhost:11434`. The server starts at
`http://0.0.0.0:8000`.

If you would rather not install Python and Ollama on the host at all, see
[Running with Docker](../README.md#running-with-docker) — one command, no
host dependencies beyond Docker itself.

## 6. Initial Data Ingestion

1. Navigate to **http://localhost:8000/admin** in your browser.
2. Enter your `ADMIN_PASSWORD`.
3. Add researchers (faculty/students) using their Semantic Scholar IDs or names.
4. Click **Sync Now** to pull their papers and build the semantic index.
5. Once the sync says `done`, navigate back to the main student page at `http://localhost:8000/` and try your first search query!

### Keeping the sync bounded

`MAX_PAPERS_PER_PERSON` (default 200, newest first) caps how many papers are
pulled per researcher. A handful of prolific authors can otherwise pull tens of
thousands of records, and embedding them all is what will actually exhaust
memory on a laptop. Embedding runs in batches of `EMBED_BATCH_SIZE` (default 64)
so peak memory stays flat regardless of corpus size.

Without a `SEMANTIC_SCHOLAR_API_KEY` the quota is 100 requests per 5 minutes, so
a first sync of a dozen researchers takes several minutes — this is expected,
and the client backs off and retries rather than failing.

### Building an evaluation corpus quickly

To populate a realistic corpus without adding people by hand:

```bash
cd backend
python eval/seed_corpus.py --ingest     # registers well-known AI/ML researchers
python eval/build_eval_set.py           # generates the labelled query set
```
