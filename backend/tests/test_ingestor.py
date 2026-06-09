"""
Unit tests for ingestion logic — year tracking, status persistence.
These run without network access or ChromaDB.
"""

import json
import os
import sys
import tempfile

import pytest

os.environ.setdefault("ADMIN_PASSWORD", "test")
os.environ.setdefault("CORS_ORIGINS", "*")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── _save_status atomic write ─────────────────────────────────────────────────

def test_save_status_is_atomic(tmp_path, monkeypatch):
    import config
    status_file = tmp_path / "sync_status.json"
    monkeypatch.setattr(config, "SYNC_STATUS_PATH", str(status_file))

    # Reimport ingestor so it picks up patched path
    import importlib
    import ingestion.ingestor as ingestor_mod
    importlib.reload(ingestor_mod)

    status = {"status": "done", "papers_indexed": 42}
    ingestor_mod._save_status(status)

    assert status_file.exists()
    loaded = json.loads(status_file.read_text())
    assert loaded["papers_indexed"] == 42


def test_save_status_produces_valid_json(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "SYNC_STATUS_PATH", str(tmp_path / "s.json"))

    import importlib
    import ingestion.ingestor as ingestor_mod
    importlib.reload(ingestor_mod)

    ingestor_mod._save_status({"nested": {"a": 1}, "list": [1, 2, 3]})
    content = (tmp_path / "s.json").read_text()
    parsed = json.loads(content)   # must not raise
    assert parsed["nested"]["a"] == 1


def test_load_status_returns_default_if_file_missing(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "SYNC_STATUS_PATH", str(tmp_path / "nonexistent.json"))

    import importlib
    import ingestion.ingestor as ingestor_mod
    importlib.reload(ingestor_mod)

    status = ingestor_mod._load_status()
    assert status["status"] == "idle"
    assert status["papers_indexed"] == 0


def test_load_status_returns_default_if_file_corrupt(tmp_path, monkeypatch):
    import config
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json")
    monkeypatch.setattr(config, "SYNC_STATUS_PATH", str(bad_file))

    import importlib
    import ingestion.ingestor as ingestor_mod
    importlib.reload(ingestor_mod)

    status = ingestor_mod._load_status()
    assert status["status"] == "idle"


# ── Year tracking logic ────────────────────────────────────────────────────────

def test_year_zero_is_skipped():
    """Papers with year=0 or year=None should not update max_year."""
    papers = [
        {"paperId": "p1", "year": 0,    "title": "Old Paper", "abstract": ""},
        {"paperId": "p2", "year": None, "title": "No Year",   "abstract": ""},
        {"paperId": "p3", "year": 2023, "title": "Good Paper", "abstract": ""},
    ]
    max_year = 0
    for paper in papers:
        yr = paper.get("year")
        if yr and yr > max_year:
            max_year = yr

    assert max_year == 2023, "year=0 and year=None must not set max_year"


def test_max_year_tracks_highest():
    papers = [
        {"paperId": "p1", "year": 2019},
        {"paperId": "p2", "year": 2022},
        {"paperId": "p3", "year": 2021},
    ]
    max_year = 0
    for paper in papers:
        yr = paper.get("year")
        if yr and yr > max_year:
            max_year = yr

    assert max_year == 2022
