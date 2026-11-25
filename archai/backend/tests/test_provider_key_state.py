"""What the chat provider last said about this server's key, and who is told.

Found on a user's installation (2026-11-25): the provider refused the
configured key (401 "Unknown API key"), and nothing said so until a question
failed. GET /api/chat/models caught the refusal and answered 200 with the
configured and fallback model names, /api/health said "A chat API key is
configured; it is not validated here", and search_index said ok because a key
existed although every page went unindexed.

The provider's answers to the calls the app makes anyway (the model list,
chat, translation, embeddings, SAIA OCR) are now remembered: refused (401/403),
accepted (2xx), or not yet used. Health and the model list report it; health
itself sends nothing to the provider.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.services import chat_ai, provider_key, rag_store

KEY = "k" * 32
REFUSED_SENTENCE = "The chat provider refused this server's key: chat and translation are unavailable until it is replaced."


class _Provider(BaseHTTPRequestHandler):
    """A stand-in OpenAI-compatible provider on loopback; `status` says how it answers."""

    status = 401

    def _answer(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        if length:
            self.rfile.read(length)
        if self.status == 401:
            body = {"error": {"message": "Unknown API key", "type": "invalid_request_error"}}
        elif self.path.endswith("/models"):
            body = {"object": "list", "data": [{"id": "model-a", "object": "model"}, {"id": "model-b", "object": "model"}]}
        elif self.path.endswith("/embeddings"):
            body = {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}], "model": "e5", "usage": {"prompt_tokens": 1, "total_tokens": 1}}
        else:
            body = {"id": "x", "object": "chat.completion", "created": 0, "model": "model-a", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
        data = json.dumps(body).encode()
        self.send_response(self.status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = _answer
    do_POST = _answer

    def log_message(self, *args: object) -> None:  # quiet
        pass


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_Provider]]:
    handler = type("Provider", (_Provider,), {"status": 401})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    for name in ("chat_ai_api_key",):
        monkeypatch.setattr(settings, name, KEY)
    for name in ("chat_ai_base_url", "archai_chat_ai_base_url"):
        monkeypatch.setattr(settings, name, base)
    monkeypatch.setattr(rag_store, "_EMBED_CLIENTS", {})
    provider_key.reset()
    yield handler
    server.shutdown()
    provider_key.reset()


def _health() -> dict:
    return TestClient(main.app).get("/api/health").json()


def test_nothing_is_known_before_the_provider_is_asked(provider: type[_Provider]) -> None:
    assert provider_key.state() == "untried"
    check = _health()["checks"]["chat_api_key"]
    assert check == {"ok": True, "detail": "A chat API key is configured; the provider has not been asked yet."}


def test_a_refused_key_is_remembered_from_the_model_list_and_reported(provider: type[_Provider]) -> None:
    listed = chat_ai.list_available_models()
    assert listed["provider"] == "refused"
    assert listed["models"]  # the configured names, so the workspace still renders
    assert provider_key.state() == "refused"

    body = _health()
    assert body["status"] == "degraded"
    assert body["checks"]["chat_api_key"] == {"ok": False, "detail": REFUSED_SENTENCE}
    text = json.dumps(body)
    assert KEY not in text and "127.0.0.1" not in text and "Unknown API key" not in text


def test_the_search_index_is_not_ok_on_a_refused_key(provider: type[_Provider], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "rag_embedding_model", "multilingual-e5-large-instruct")
    monkeypatch.setattr(settings, "rag_auto_index", True)
    with pytest.raises(rag_store.EmbeddingUnavailable):
        rag_store._embed(["a passage"])
    assert provider_key.state() == "refused"
    check = _health()["checks"]["search_index"]
    assert check["ok"] is False
    assert "refused this server's key" in check["detail"]
    # The local embedder needs no key: a refusal does not concern it.
    monkeypatch.setattr(settings, "rag_embedding_model", "")
    assert _health()["checks"]["search_index"]["ok"] is True


def test_a_later_success_clears_the_refusal(provider: type[_Provider]) -> None:
    chat_ai.list_available_models()
    assert provider_key.state() == "refused"
    provider.status = 200
    listed = chat_ai.list_available_models()
    assert listed["provider"] == "ok"
    assert listed["models"][:2] != []
    assert provider_key.state() == "accepted"
    assert _health()["checks"]["chat_api_key"] == {"ok": True, "detail": "A chat API key is configured and the provider accepted it."}


def test_chat_and_embeddings_are_heard_too(provider: type[_Provider], monkeypatch: pytest.MonkeyPatch) -> None:
    provider.status = 200
    monkeypatch.setattr(settings, "rag_embedding_model", "multilingual-e5-large-instruct")
    assert rag_store._embed(["a passage"]) == [[0.1, 0.2]]
    assert provider_key.state() == "accepted"
    provider.status = 401
    with pytest.raises(Exception):
        chat_ai.create_chat_completion(messages=[{"role": "user", "content": "hi"}], model="model-a")
    assert provider_key.state() == "refused"


def test_an_unreachable_provider_is_not_a_refused_key(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_key.reset()
    monkeypatch.setattr(settings, "chat_ai_api_key", KEY)
    monkeypatch.setattr(settings, "chat_ai_base_url", "http://127.0.0.1:9/v1")  # nothing listens there
    monkeypatch.setattr(settings, "chat_ai_timeout_seconds", 2)
    listed = chat_ai.list_available_models()
    assert listed["provider"] == "unreachable"
    assert provider_key.state() == "untried"


def test_the_models_route_marks_the_answer(provider: type[_Provider]) -> None:
    body = TestClient(main.app).get("/api/chat/models").json()
    assert body["provider"] == "refused"
    assert body["models"]


def test_health_sends_nothing_to_the_provider(provider: type[_Provider]) -> None:
    requests_before = provider_key.state()
    _health()
    _health()
    assert provider_key.state() == requests_before == "untried"
