# Scalability & Resource Requirements

## Overview
TRACE is currently configured for a single-server deployment optimal for institute-level scale. As the system scales to handle more students or larger document bases, the following bottlenecks and requirements should be addressed.

## Resource Requirements (Current Single-Server Baseline)
* **RAM:** Minimum 8GB (ChromaDB runs in-memory; BM25 index built in-memory).
* **VRAM / Compute:** 
  * If running `llama3.2 3B` locally on Ollama: ~2.5GB VRAM (or RAM if CPU-only).
  * The Bi-encoder and Cross-encoder run efficiently on CPU but benefit from GPU.
* **Disk:** ~500MB per 10,000 indexed abstracts for ChromaDB persistence.
* **Sync memory:** bounded by `MAX_PAPERS_PER_PERSON` (fetch cap) and `EMBED_BATCH_SIZE` (embedding batch). Peak memory during a sync is flat in corpus size; it was previously proportional to it, because the whole document set went to a single embed call — which also failed outright once the payload grew large enough.

## Known Scaling Bottlenecks

### 1. Vector Database (ChromaDB)
* **Limit:** ChromaDB currently runs locally inside the FastAPI process. This prevents horizontal scaling (running multiple `uvicorn` workers) because they cannot safely share the same on-disk DB concurrently without corruption.
* **Solution:** Migrate to a client-server VectorDB deployment (e.g., ChromaDB server mode, Qdrant, or Milvus). Update `VectorStoreManager` to connect via HTTP/gRPC. Effort: Low-Medium.

### 2. BM25 Memory Usage
* **Limit:** The BM25 index is built entirely in-memory at startup and post-sync. While this is fast for <50,000 papers, it scales linearly in memory and startup time.
* **Solution:** If the document count exceeds 100k, replace the local `rank_bm25` implementation with a dedicated sparse search engine like Elasticsearch or OpenSearch. Effort: Medium-High.

### 3. Local LLM Generation (Ollama)
* **Limit:** Generating answers with a local LLM takes 10-30 seconds per query. While `run_in_executor` keeps the server responsive, a single Ollama instance can only generate a few tokens at a time and will queue concurrent requests, blowing up P95 latency under load.
* **Solution:** 
  * Switch to a cloud LLM provider (OpenAI, Anthropic) for instant scalability.
  * If self-hosting is required, deploy a dedicated vLLM cluster with continuous batching and scale GPU nodes horizontally. Effort: Low (Cloud API) to High (Local Cluster).

### 4. Trace Log Growth
* **Limit:** `data/retrieval_traces.jsonl` grows by one record (~2-4 KB) per query, unbounded. At 1,000 queries/day that is roughly 1 GB/year.
* **Solution:** rotate daily and compress, or ship records to DuckDB/Parquet for querying. The analysis scripts read the file linearly, so this becomes the slow path long before it becomes a disk problem. Effort: Low.

### 5. Semantic Cache Growth
* **Limit:** the `query_cache` collection grows without eviction; entries are only cleared wholesale on sync. Lookup cost rises with cache size.
* **Solution:** add TTL-based eviction (records already carry `created_at`) or an LRU cap. Effort: Low.

### 6. Admin API Rate Limiting
* **Limit:** Admin endpoints are sensitive to timing and brute-force attacks.
* **Solution:** We have applied a 10 request/minute rate limit to mitigate brute-forcing of the `X-Admin-Password`. At enterprise scale, replace this shared password with SSO/OAuth2. Effort: Medium.
