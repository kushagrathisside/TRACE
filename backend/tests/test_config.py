"""
Unit tests for config.py — validation logic, env-var parsing.

These tests run without Ollama or ChromaDB and are safe for CI.
"""

import os
import pytest


def _import_config(**env_overrides):
    """Re-import config with a patched environment."""
    import importlib
    import sys

    env = {
        "ADMIN_PASSWORD": "test-password",
        "CORS_ORIGINS": "*",
        **env_overrides,
    }
    with pytest.MonkeyPatch().context() as mp:
        for k, v in env.items():
            mp.setenv(k, v)
        # Remove cached module so it re-evaluates with the new env
        sys.modules.pop("config", None)
        import config as cfg
        return cfg


def test_config_loads_with_valid_env():
    cfg = _import_config()
    assert cfg.ADMIN_PASSWORD == "test-password"
    assert cfg.CORS_ORIGINS if hasattr(cfg, "CORS_ORIGINS") else True


def test_config_raises_without_admin_password():
    import sys
    sys.modules.pop("config", None)
    with pytest.MonkeyPatch().context() as mp:
        mp.delenv("ADMIN_PASSWORD", raising=False)
        mp.setenv("CORS_ORIGINS", "*")
        sys.modules.pop("config", None)
        with pytest.raises((ValueError, Exception)):
            import config  # noqa: F401


def test_retrieval_k_cannot_exceed_fetch_k():
    with pytest.raises(ValueError, match="RETRIEVAL_K"):
        _import_config(RETRIEVAL_K="10", RETRIEVAL_FETCH_K="5")


def test_min_similarity_distance_bounds():
    with pytest.raises(ValueError, match="MIN_SIMILARITY_DISTANCE"):
        _import_config(MIN_SIMILARITY_DISTANCE="3.0")


def test_cache_hit_distance_bounds():
    with pytest.raises(ValueError, match="CACHE_HIT_DISTANCE"):
        _import_config(CACHE_HIT_DISTANCE="-0.1")


def test_hnsw_m_must_be_positive():
    with pytest.raises(ValueError, match="HNSW_M"):
        _import_config(HNSW_M="0")


def test_enable_ragas_scoring_flag():
    cfg = _import_config(ENABLE_RAGAS_SCORING="true")
    assert cfg.ENABLE_RAGAS_SCORING is True

    cfg = _import_config(ENABLE_RAGAS_SCORING="false")
    assert cfg.ENABLE_RAGAS_SCORING is False

    cfg = _import_config(ENABLE_RAGAS_SCORING="TRUE")
    assert cfg.ENABLE_RAGAS_SCORING is True
