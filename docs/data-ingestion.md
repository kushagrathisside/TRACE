# Data & Ingestion Pipeline

The Ingestion Engine is the mechanism TRACE uses to populate and maintain the underlying knowledge base from external sources (Semantic Scholar).

## The Pipeline Lifecycle

### 1. Triggering the Sync
A sync is triggered via the `/api/sync` admin endpoint.
To prevent database corruption if multiple admins attempt to trigger a sync concurrently, `backend/ingestion/ingestor.py` utilizes `portalocker` to claim an exclusive OS-level file lock on `data/sync_status.lock`. 

### 2. Incremental Fetching
For each person in the `people.json` registry, the ingestor checks their specific `last_year` timestamp stored in `data/sync_status.json`. 
When querying the Semantic Scholar Graph API, it passes `since_year=last_year`, meaning it **only fetches new papers** rather than re-downloading a researcher's entire historical catalog.

### 3. Metadata Parsing and Validation
As papers are retrieved, their metadata is aggregated (e.g. coalescing multiple authors from the same institute onto the same paper record).
Crucially, before a paper is embedded, its metadata is rigorously validated against the `DocumentMetadata` Pydantic model (`backend/rag/schemas.py`). 
This uses `extra="forbid"` to ensure no unexpected, corrupted, or unsupported fields are ever pushed into the Vector database.

### 4. Vector Embedding & Upsertion
Validated papers are passed to Langchain's Chroma integration. The `HuggingFaceBgeEmbeddings` model generates dense vector embeddings of the abstracts, and the records are written to disk.

### 5. Status Management
- **Success**: The `last_year` value is updated in the status file, and the total indexed paper count is refreshed.
- **Degraded**: If the Semantic Scholar API rate limits or errors out for more than 30% of the authors in the registry, the sync status is explicitly marked as `degraded`.
- **Evaluation**: Post-sync, a `self_retrieval` script runs automatically. It attempts to search for a random sample of 200 abstracts. If the retrieval `Hit Rate@5` falls below 90%, the sync status is updated to `degraded` to warn admins of an embedding anomaly.

## Orphan Cleanup
Paper deletion is **not** handled during routine syncing. 
If an author is removed via `DELETE /api/people/{id}`, a synchronous cleanup is performed:
1. TRACE finds all papers where the deleted person was an author.
2. If they were the *only* institute author on the paper, the paper is deleted.
3. If they co-authored with other active institute members, the paper is kept, but their name is stripped from the `institute_authors` metadata.
