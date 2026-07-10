# Architecture & System Design

TRACE (Trustworthy Retrieval with Automated Continuous Evaluation) is a robust Retrieval-Augmented Generation (RAG) system built around FastAPI, ChromaDB, and local LLMs.

## High-Level Architecture

```mermaid
flowchart TD
    subgraph Frontend
        UI[Web Interface]
        AdminUI[Admin Dashboard]
    end

    subgraph FastAPI Backend
        API[API Router]
        Pipeline[RAG Pipeline]
        Ingest[Ingestion Engine]
    end

    subgraph Data Stores
        Chroma[(ChromaDB Vector Store)]
        BM25[(In-Memory BM25)]
        Cache[(Semantic Cache)]
        People[(people.json)]
    end

    subgraph External
        S2[Semantic Scholar API]
        Ollama[Ollama LLM Server]
    end

    UI -->|/api/query| API
    AdminUI -->|/api/sync| API
    
    API --> Pipeline
    API --> Ingest
    
    Ingest -->|Fetch Papers| S2
    Ingest -->|Upsert| Chroma
    
    Pipeline -->|1. Check Cache| Cache
    Pipeline -->|2. Search| Chroma
    Pipeline -->|3. Search| BM25
    Pipeline -->|4. Generate| Ollama
```

## System Components

### 1. The RAG Pipeline (`backend/rag/pipeline.py`)
The query pipeline operates in a non-blocking thread executor to keep the async event loop free:
1. **Semantic Cache Check**: If the query's vector embedding distance is `< 0.08` to a recent query (within the 15-day TTL), the cached JSON is returned immediately.
2. **Hybrid Retrieval**: Combines sparse BM25 retrieval (keyword matching) with dense ChromaDB retrieval (semantic matching).
3. **Cross-Encoder Reranking**: Re-scores the retrieved documents to surface the most relevant papers to the top.
4. **LLM Generation**: Uses local models via Ollama. It enforces a strict JSON schema and employs a hallucination guard that cross-references `paper_id`s to ensure the model doesn't invent fake citations.

### 2. Ingestion Engine (`backend/ingestion/ingestor.py`)
Fetches and maintains a local database of papers authored by registered institute members. It performs incremental fetching from the Semantic Scholar Graph API and strictly validates metadata using Pydantic before upserting embeddings into ChromaDB.

### 3. State & Persistence
- **ChromaDB**: Persists embeddings and metadata to disk (`data/chroma_db`).
- **People Registry**: Uses `portalocker` for thread-safe file access to `data/people.json`.
- **Sync Status**: Maintained in `data/sync_status.json` and locked during operations to prevent multi-worker concurrency corruption.
