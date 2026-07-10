# API Reference

All backend communication happens over REST via the FastAPI server.

## Global Headers
- **`X-Admin-Password`**: Required for all endpoints under the Admin routes.

## Rate Limiting
- **Student Queries (`/api/query`)**: Limited to 10 requests per minute per IP.
- **Admin Routes**: Limited to 10 requests per minute to prevent brute force timing attacks.
- **Default Fallback**: 60 requests per minute.

---

## Student Routes

### `POST /api/query`
Executes the RAG pipeline.
**Request Body**:
```json
{
  "idea": "Research topic about NLP transformers"
}
```
**Response**:
```json
{
  "landscape_summary": "Summary text...",
  "related_papers": [
    {
      "paper_id": "...",
      "title": "...",
      "year": 2023,
      "authors": "...",
      "venue": "...",
      "relevance": "..."
    }
  ],
  "people_to_consult": [],
  "next_steps": []
}
```

### `POST /api/feedback`
Records user feedback for evaluation.
**Request Body**:
```json
{
  "query": "Research topic...",
  "rating": "up",
  "comment": "Optional context"
}
```

---

## Admin Routes
*Requires `X-Admin-Password` header.*

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
