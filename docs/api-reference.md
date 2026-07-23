# API Reference

All backend communication happens over REST via the FastAPI server.

## Global Headers
- **`X-Admin-Password`**: Required for all endpoints under the Admin routes.

## Rate Limiting
- **Student Queries (`/api/query`)**: 10 requests per minute per IP.
- **Feedback (`/api/feedback`)**: same budget as queries. An unthrottled endpoint that feeds a quality metric is a quality metric anyone can forge.
- **Admin Routes**: 60 requests per minute.
- **Default Fallback**: 60 requests per minute.

---

## Student Routes

### `POST /api/query`
Executes the RAG pipeline.

**Request Body**:
```json
{
  "idea": "Research topic about NLP transformers",
  "bypass_cache": false
}
```
`idea` must be non-empty and at most 2000 characters.

**Response**:
```json
{
  "answer": {
    "landscape_summary": "Summary text...",
    "related_papers": [
      {"paper_id": "...", "title": "...", "year": 2023,
       "authors": "...", "venue": "...", "relevance": "..."}
    ],
    "people_to_consult": [],
    "next_steps": [],
    "no_relevant_research": false,
    "generation_failed": false
  },
  "sources": [
    {"paper_id": "...", "title": "...", "year": 2023, "venue": "...",
     "authors": "...", "institute_authors": "...", "institute_roles": "...",
     "departments": "...", "url": "...", "rerank_score": 6.71}
  ],
  "cached": false,
  "query_id": "3f2a1c9e-..."
}
```

**`query_id`** identifies the trace record for this query in
`data/retrieval_traces.jsonl`. Send it back with any feedback so a rating can be
attributed to what was actually retrieved rather than only to the query text.

**`sources[].paper_id`** is what offline evaluation joins against ground-truth
labels. `rerank_score` is the cross-encoder logit, or `null` when the reranker
is inactive — never a stand-in value, so it cannot be mistaken for a real score.

**Errors**: `400` empty or oversized idea · `503` Ollama unreachable ·
`500` pipeline failure (body carries the `query_id` as a reference; internal
details are logged, not returned).

### `POST /api/feedback`
Records user feedback for evaluation.

**Request Body**:
```json
{
  "query": "Research topic...",
  "rating": "up",
  "comment": "Optional context",
  "query_id": "3f2a1c9e-..."
}
```
Omitting `query_id` still records the rating, but it cannot be joined to a trace
and `analyse_feedback.py` will report it as unattributable.

---

## Admin Routes
*Requires `X-Admin-Password` header.*

### `GET /health`
Subsystem status. Always returns HTTP 200 so uptime monitors can read the body;
alert on `status != "ok"`.

```json
{
  "status": "degraded",
  "checks": {
    "chromadb": {"status": "ok", "documents": 1043},
    "reranker": {"status": "degraded", "model": "cross-encoder/...",
                 "active": false, "error": "..."},
    "ollama":   {"status": "ok", "model": "llama3.2:3b", "model_available": true}
  }
}
```

`reranker.active: false` means ranking has fallen back to fusion order — a
material quality regression. `status: "not_loaded"` simply means no query has
been served yet and warm-up has not finished; the probe deliberately does not
construct the model, since a health check that can block for a model download is
worse than no health check.

### `GET /api/people`
Returns a paginated list of registered authors.

### `POST /api/people`
Registers a new author to track.
**Request Body**:
```json
{
  "name": "Jane Doe",
  "role": "Faculty",
  "department": "Computer Science",
  "email": "jane@institute.edu",
  "semantic_scholar_id": "1234567"
}
```

### `DELETE /api/people/{person_id}`
Removes a person and cascades deletion to their exclusive papers in ChromaDB.

### `POST /api/sync`
Triggers the background ingestion pipeline to fetch new papers from Semantic Scholar.

### `GET /api/sync/status`
Returns the current status of the ingestion pipeline (e.g., `idle`, `running`, `degraded`).

### `GET /api/stats`
Returns system statistics (total papers, people, last sync time).

### `GET /api/feedback/analysis`
Generates an LLM-driven analysis report on accumulated user feedback.
