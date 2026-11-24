"""Health says when pages are read but cannot be indexed for search.

Live regression (2026-11-24): with no chat key, every page's Steps said "Not
indexed for search: the embedding service is unavailable", while the quality
badge said the text "is used for search". The default RAG_EMBEDDING_MODEL is a
provider model reached with the chat key. Health now says so, and how to index
with the local embedder instead; the model choice stays the operator's.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings

NOT_INDEXED = (
    "Pages are read but not indexed for search: the embedding model needs the chat key. Set the key, "
    "or set RAG_EMBEDDING_MODEL= to index with the local embedder (see docs/deployment.md)."
)


def _no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("chat_ai_api_key", "saia_api_key", "archai_chat_ai_api_key", "archai_saia_api_key"):
        monkeypatch.setattr(settings, name, "")
    for name in ("CHAT_AI_API_KEY", "SAIA_API_KEY", "ARCHAI_CHAT_AI_API_KEY", "ARCHAI_SAIA_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _search_index() -> dict:
    return TestClient(main.app).get("/api/health").json()["checks"]["search_index"]


def test_a_provider_embedding_model_without_a_key_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_key(monkeypatch)
    monkeypatch.setattr(settings, "rag_embedding_model", "multilingual-e5-large-instruct")
    check = _search_index()
    assert check == {"ok": False, "detail": NOT_INDEXED}
    assert TestClient(main.app).get("/api/health").json()["status"] in ("degraded", "unavailable")


def test_the_local_embedder_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_key(monkeypatch)
    monkeypatch.setattr(settings, "rag_embedding_model", "")
    assert _search_index() == {"ok": True, "detail": "Pages are indexed with the local embedder."}


def test_a_provider_model_with_a_key_is_not_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "chat_ai_api_key", "k" * 30)
    monkeypatch.setattr(settings, "rag_embedding_model", "multilingual-e5-large-instruct")
    check = _search_index()
    assert check["ok"] is True
    assert "multilingual" not in check["detail"]  # the endpoint is open: no model or host names


def test_indexing_switched_off_is_not_a_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_key(monkeypatch)
    monkeypatch.setattr(settings, "rag_auto_index", False)
    assert _search_index() == {"ok": True, "detail": "Pages are not indexed automatically (RAG_AUTO_INDEX is off)."}
