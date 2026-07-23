# Deployment & Production Guide

This guide details how to move TRACE from a local development environment into a production deployment.

## 1. Environment Configuration (`.env`)
Ensure your `.env` file is heavily guarded and strictly configured:
```ini
# Security
ADMIN_PASSWORD=your_secure_random_string
CORS_ORIGINS=https://your-institute-domain.edu

# Rate Limiting
RATE_LIMIT_DEFAULT=60/minute
RATE_LIMIT_QUERIES=10/minute
RATE_LIMIT_ADMIN=10/minute

# Cache
CACHE_MAX_AGE_DAYS=15

# Retrieval — verify the reranker is loadable on the production host.
# An empty value is treated as a deliberate "no reranking" choice; a value that
# fails to load falls back to fusion order, which is a material ranking
# regression. Set RERANKER_REQUIRED=true in production so the server refuses to
# start rather than serving a silently degraded ranker.
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_REQUIRED=true

# Ingestion bounds — cap fetch and embedding batch size so a sync cannot
# exhaust memory on a shared host.
MAX_PAPERS_PER_PERSON=200
EMBED_BATCH_SIZE=64
SEMANTIC_SCHOLAR_API_KEY=your_key_here
```

### Proxies
If the host sits behind an HTTP proxy, leave the proxy variables in place.
TRACE appends `localhost,127.0.0.1,::1` to `no_proxy` so Ollama is reached
directly; it does not clear your proxy configuration, because doing so breaks
outbound HTTPS to Semantic Scholar.

## 2. Model Hosting (Ollama)
For production, running Ollama on the same machine as the web server is not recommended unless it's a dedicated GPU node.
- **Dedicated GPU Server**: Install Ollama and set `OLLAMA_HOST=0.0.0.0` so it accepts external connections. Set `OLLAMA_BASE_URL` in the FastAPI `.env` to point to the dedicated GPU box.
- **Cloud LLM**: If latency is an issue, swap out `LLMProvider` in `backend/llm_provider.py` to use a cloud provider like OpenAI or Anthropic.

## 3. Web Server (Uvicorn / Gunicorn)
**CRITICAL WARNING**: Because ChromaDB currently runs locally inside the application process in single-tenant mode, **you must only run a single Uvicorn worker process.** 

Starting multiple workers (e.g. `gunicorn -w 4`) will cause database locking conflicts and corruption for both ChromaDB and the `people.json` file.

To run the application safely:
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
```

Two further pieces of state assume a single process: `ingestor.sync_status` is a
module-level dict (the `/api/sync/status` endpoint reads the copy belonging to
whichever worker serves the request), and the `Reranker` / `VectorStoreManager`
singletons are per-process. Both would need externalising before scaling out.

### Post-deploy verification

```bash
curl -s localhost:8000/health | jq
```

Confirm `checks.reranker.active == true`. If it reports `degraded`, ranking has
fallen back to fusion order — the deployment is serving materially worse results
than the one you evaluated, and every trace will be tagged
`reranker_active=false`.

## 4. Reverse Proxy (Nginx)
Expose the FastAPI app through an Nginx reverse proxy to handle SSL termination and static file serving.

```nginx
server {
    listen 443 ssl;
    server_name trace.institute.edu;

    ssl_certificate /etc/letsencrypt/live/trace/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/trace/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## 5. Background Jobs (Cron)
While the sync can be triggered manually via the Admin Dashboard, it's recommended to automate this process. Set up a nightly cron job to ping the sync endpoint:

```bash
# Run every night at 2:00 AM
0 2 * * * curl -X POST https://trace.institute.edu/api/sync -H "X-Admin-Password: your_secure_random_string"
```
