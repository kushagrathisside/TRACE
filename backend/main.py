import asyncio
import hmac
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# Make backend/ the importable root
sys.path.insert(0, os.path.dirname(__file__))

import config
import httpx
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ingestion import ingestor, people_registry
from ingestion.scholar_client import search_author
from pydantic import BaseModel
from rag import pipeline
from rag.vector_store import VectorStoreManager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[config.RATE_LIMIT_DEFAULT],
)


# ── Startup / shutdown ────────────────────────────────────────────────────────
def _warmup() -> None:
    """
    Pre-load the embedding model and LLM into memory during startup so the
    first real student query doesn't hit a 30-60s cold-start delay.

    Embedding model: triggers the ~90 MB HuggingFace download on first run,
    then loads the model weights into RAM.

    LLM warm-up: sends a trivial prompt to Ollama so the model is resident
    in RAM/VRAM with its KV-cache primed.  Non-fatal: if Ollama is not yet
    running the server still starts normally and the LLM loads on first query.
    """
    from langchain_core.messages import HumanMessage
    from llm_provider import LLMProvider

    try:
        emb = LLMProvider.get_embeddings()
        emb.embed_query("warm-up")
        logger.info("Embedding model pre-loaded")
    except Exception as exc:
        logger.warning(f"Embedding warm-up failed: {exc}")

    try:
        llm = LLMProvider.get_llm()
        llm.invoke([HumanMessage(content="Hi")])
        logger.info("LLM pre-loaded and KV-cache primed")
    except Exception as exc:
        logger.warning(f"LLM warm-up failed (Ollama may not be running yet): {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    # Build BM25 index from existing ChromaDB (non-blocking)
    await loop.run_in_executor(None, pipeline.build_bm25_on_startup)
    # Pre-warm embedding model + LLM (non-blocking, non-fatal)
    asyncio.ensure_future(loop.run_in_executor(None, _warmup))
    yield


app = FastAPI(
    title="TRACE — Trustworthy Retrieval with Automated Continuous Evaluation",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

cors_origins = (
    os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else []
)
if not cors_origins:
    raise ValueError(
        "CORS_ORIGINS environment variable not set. "
        "Set to comma-separated list of allowed origins (e.g., 'http://localhost:3000,https://institute.edu') "
        "or '*' if this is development and behind a private network."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# ── Static files ──────────────────────────────────────────────────────────────
FRONTEND = Path(config.FRONTEND_DIR).resolve()
app.mount("/static", StaticFiles(directory=str(FRONTEND / "static")), name="static")


# ── Auth helper ───────────────────────────────────────────────────────────────
def require_admin(x_admin_password: str = Header(default="")) -> None:
    if not hmac.compare_digest(x_admin_password, config.ADMIN_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid admin password")


# ═══════════════════════════════════════════════════════════════════════════════
# Frontend routes
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/", include_in_schema=False)
def student_page():
    return FileResponse(str(FRONTEND / "index.html"))


@app.get("/admin", include_in_schema=False)
def admin_page():
    return FileResponse(str(FRONTEND / "admin.html"))


# ═══════════════════════════════════════════════════════════════════════════════
# Health check
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/health")
async def health():
    """
    Returns status of each subsystem.  Used by load balancers and monitoring.
    A degraded response (non-2xx for any subsystem) still returns HTTP 200 so
    uptime monitors can inspect the body — alert on status != "ok".
    """
    checks: dict = {}

    # ChromaDB
    try:
        vs = VectorStoreManager.get_or_create()
        checks["chromadb"] = {"status": "ok", "documents": vs.count()}
    except Exception as exc:
        checks["chromadb"] = {"status": "error", "error": str(exc)}

    # Ollama
    try:
        resp = httpx.get(
            f"{config.OLLAMA_BASE_URL}/api/tags",
            timeout=2,
        )
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        model_available = any(config.LLM_MODEL_NAME in m for m in models)
        checks["ollama"] = {
            "status": "ok" if model_available else "degraded",
            "model": config.LLM_MODEL_NAME,
            "model_available": model_available,
        }
    except httpx.TimeoutException:
        checks["ollama"] = {
            "status": "unknown",
            "error": "Ollama health check timed out (>2s)",
        }
    except Exception as exc:
        checks["ollama"] = {"status": "error", "error": str(exc)}

    overall = "ok" if all(v["status"] == "ok" for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}


# ═══════════════════════════════════════════════════════════════════════════════
# Student API
# ═══════════════════════════════════════════════════════════════════════════════


class QueryRequest(BaseModel):
    idea: str
    bypass_cache: bool = False


@app.post("/api/query")
@limiter.limit(config.RATE_LIMIT_QUERIES)
async def query_endpoint(request: Request, req: QueryRequest):
    """
    Run the full RAG pipeline (cache → expand → hybrid → rerank → LLM).
    Executes in a thread pool via run_in_executor so the async event loop
    stays free to serve other requests during the ~10-30s LLM generation.
    """
    if not req.idea.strip():
        raise HTTPException(status_code=400, detail="Idea cannot be empty")

    try:
        resp = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=2)
        resp.raise_for_status()
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="LLM service unavailable. Check that Ollama is running.",
        )

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, pipeline.run, req.idea, req.bypass_cache
        )
        return result
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Feedback ──────────────────────────────────────────────────────────────────


class FeedbackRequest(BaseModel):
    query: str
    rating: Literal["up", "down"]
    comment: str = ""


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    """
    Write a thumbs-up / thumbs-down + optional comment to feedback.jsonl.
    This file can be mined later to evaluate prompt quality or retrieval accuracy.
    """
    entry = {
        "query": req.query,
        "rating": req.rating,
        "comment": req.comment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(config.FEEDBACK_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"ok": True}


@app.get("/api/feedback/analysis")
@limiter.limit(config.RATE_LIMIT_ADMIN)
def feedback_analysis(request: Request, x_admin_password: str = Header(default="")):
    """
    Analyse accumulated feedback to surface system quality trends.
    Returns summary stats: thumbs-down rate, RAGAS scores if enabled, recommendations.
    """
    require_admin(x_admin_password)
    try:
        import io
        import sys

        from eval.analyse_feedback import run as analyse_feedback

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            analyse_feedback()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        return {"analysis": output}
    except Exception as exc:
        logger.error(f"Feedback analysis failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Admin: People
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/api/people")
@limiter.limit(config.RATE_LIMIT_ADMIN)
def list_people(
    request: Request,
    x_admin_password: str = Header(default=""),
    page: int = 1,
    page_size: int = 100,
):
    """
    Returns a paginated list of registered people.
    Default page_size=100 covers most institute deployments in a single call.
    Use ?page=2&page_size=50 for larger registries.
    """
    require_admin(x_admin_password)
    if page < 1:
        raise HTTPException(status_code=400, detail="page must be >= 1")
    if page_size < 1 or page_size > 1000:
        raise HTTPException(status_code=400, detail="page_size must be in [1, 1000]")

    all_people = people_registry.get_all()
    total = len(all_people)
    start = (page - 1) * page_size
    pages = -(-total // page_size)  # ceil division, now safe

    return {
        "people": all_people[start : start + page_size],
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": pages,
    }


class PersonRequest(BaseModel):
    name: str
    role: str
    department: str
    email: str
    semantic_scholar_id: str


@app.post("/api/people", status_code=201)
@limiter.limit(config.RATE_LIMIT_ADMIN)
def add_person(
    request: Request, req: PersonRequest, x_admin_password: str = Header(default="")
):
    require_admin(x_admin_password)
    return people_registry.add_person(
        req.name, req.role, req.department, req.email, req.semantic_scholar_id
    )


@app.delete("/api/people/{person_id}")
@limiter.limit(config.RATE_LIMIT_ADMIN)
def delete_person(
    request: Request, person_id: str, x_admin_password: str = Header(default="")
):
    """
    Remove person from registry AND clean up their papers in ChromaDB:
      - Papers where they were the sole institute author are deleted.
      - Papers co-authored with remaining members are updated (name removed).
    The semantic cache and BM25 index are then rebuilt.
    """
    require_admin(x_admin_password)
    person = people_registry.remove_person(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")

    vs = VectorStoreManager.get_or_create()
    deleted, updated = vs.remove_person_from_docs(person["name"])

    # Rebuild BM25 and invalidate semantic cache
    pipeline.post_sync_rebuild()

    return {
        "ok": True,
        "papers_deleted": deleted,
        "papers_updated": updated,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Admin: Sync
# ═══════════════════════════════════════════════════════════════════════════════


@app.post("/api/sync")
@limiter.limit(config.RATE_LIMIT_ADMIN)
def trigger_sync(
    request: Request,
    background_tasks: BackgroundTasks,
    x_admin_password: str = Header(default=""),
):
    require_admin(x_admin_password)
    if ingestor.sync_status["status"] == "running":
        raise HTTPException(status_code=409, detail="Sync already running")

    def _run():
        ingestor.run_ingestion()
        # Post-sync: rebuild BM25, invalidate semantic cache, reset VS singleton
        pipeline.post_sync_rebuild()

        try:
            from eval.self_retrieval import run as run_self_retrieval

            rate = run_self_retrieval(k=5, sample=200)
            if rate < 0.90:
                ingestor.sync_status["status"] = "degraded"
                ingestor._save_status(ingestor.sync_status)
        except Exception as e:
            logger.error(f"Self-retrieval eval failed: {e}")

    background_tasks.add_task(_run)
    return {"ok": True, "message": "Sync started"}


@app.get("/api/sync/status")
@limiter.limit(config.RATE_LIMIT_ADMIN)
def sync_status_route(request: Request, x_admin_password: str = Header(default="")):
    require_admin(x_admin_password)
    return ingestor.sync_status


# ═══════════════════════════════════════════════════════════════════════════════
# Admin: Stats
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/api/stats")
@limiter.limit(config.RATE_LIMIT_ADMIN)
def stats(request: Request, x_admin_password: str = Header(default="")):
    require_admin(x_admin_password)
    vs = VectorStoreManager.get_or_create()
    return {
        "total_papers": vs.count(),
        "total_people": len(people_registry.get_all()),
        "last_sync": ingestor.sync_status.get("last_sync"),
        "sync_status": ingestor.sync_status.get("status"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Admin: Author search proxy
# ═══════════════════════════════════════════════════════════════════════════════


class AuthorSearchRequest(BaseModel):
    name: str


@app.post("/api/author/search")
@limiter.limit(config.RATE_LIMIT_ADMIN)
def author_search(
    request: Request,
    req: AuthorSearchRequest,
    x_admin_password: str = Header(default=""),
):
    require_admin(x_admin_password)
    try:
        return {"results": search_author(req.name)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
