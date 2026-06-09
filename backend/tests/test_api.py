"""
Integration tests for FastAPI routes using TestClient (no real Ollama needed).

Routes that call the pipeline are tested for correct HTTP behaviour only —
pipeline.run is mocked so Ollama / ChromaDB are not required.
"""

import os
import sys

import pytest

os.environ["ADMIN_PASSWORD"] = "test-secret"
os.environ["CORS_ORIGINS"] = "*"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def client():
    from unittest.mock import patch, MagicMock

    # Mock pipeline.run so no real LLM is needed
    mock_result = {
        "answer": {
            "landscape_summary": "Test summary",
            "related_papers": [],
            "people_to_consult": [],
            "next_steps": [],
            "no_relevant_research": False,
        },
        "sources": [],
        "cached": False,
    }

    # Mock Ollama health check so /api/query doesn't need Ollama running
    import httpx

    def mock_get(url, **kwargs):
        r = MagicMock()
        r.status_code = 200
        r.json.return_value = {"models": [{"name": "llama3.2"}]}
        r.raise_for_status = lambda: None
        return r

    with patch("rag.pipeline.run", return_value=mock_result), \
         patch("httpx.get", side_effect=mock_get):
        from fastapi.testclient import TestClient
        import main
        yield TestClient(main.app, raise_server_exceptions=False)


ADMIN_HEADERS = {"X-Admin-Password": "test-secret"}
WRONG_HEADERS = {"X-Admin-Password": "wrong"}


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "checks" in body


# ── Student: query ────────────────────────────────────────────────────────────

def test_query_empty_idea_returns_400(client):
    resp = client.post("/api/query", json={"idea": "   "})
    assert resp.status_code == 400


def test_query_valid_idea_returns_200(client):
    resp = client.post("/api/query", json={"idea": "graph neural networks"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert "sources" in body
    assert "cached" in body


def test_query_response_has_landscape_keys(client):
    resp = client.post("/api/query", json={"idea": "federated learning"})
    answer = resp.json()["answer"]
    assert "landscape_summary" in answer
    assert "related_papers" in answer
    assert "people_to_consult" in answer
    assert "next_steps" in answer


# ── Student: feedback ─────────────────────────────────────────────────────────

def test_feedback_thumbs_up(client):
    resp = client.post("/api/feedback", json={
        "query": "test query",
        "rating": "up",
        "comment": "",
    })
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_feedback_thumbs_down(client):
    resp = client.post("/api/feedback", json={
        "query": "bad result query",
        "rating": "down",
        "comment": "results were irrelevant",
    })
    assert resp.status_code == 200


def test_feedback_invalid_rating_returns_422(client):
    resp = client.post("/api/feedback", json={
        "query": "test",
        "rating": "maybe",
    })
    assert resp.status_code == 422


# ── Admin: auth ───────────────────────────────────────────────────────────────

def test_admin_endpoints_reject_wrong_password(client):
    for path in ["/api/people", "/api/stats", "/api/sync/status"]:
        resp = client.get(path, headers=WRONG_HEADERS)
        assert resp.status_code == 401, f"{path} should return 401"


def test_admin_endpoints_reject_missing_password(client):
    resp = client.get("/api/people")
    assert resp.status_code == 401


# ── Admin: people ─────────────────────────────────────────────────────────────

def test_list_people_returns_paginated(client):
    resp = client.get("/api/people", headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert "people" in body
    assert "total" in body
    assert "page" in body
    assert "pages" in body


def test_list_people_invalid_page_size(client):
    resp = client.get("/api/people?page_size=0", headers=ADMIN_HEADERS)
    assert resp.status_code == 400


def test_list_people_invalid_page(client):
    resp = client.get("/api/people?page=0", headers=ADMIN_HEADERS)
    assert resp.status_code == 400


# ── Admin: stats ──────────────────────────────────────────────────────────────

def test_stats_returns_expected_keys(client):
    resp = client.get("/api/stats", headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert "total_papers" in body
    assert "total_people" in body
    assert "sync_status" in body
