"""Provider embeddings are batched, time-limited and reuse one client (audit RAG-02).

_embed built a new OpenAI client per call with no timeout - a hung provider
held the calling thread for good - and sent a run's chunks in one request,
which a provider payload limit failed as a whole.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from app.services import rag_store


class _FakeOpenAI:
    instances: list["_FakeOpenAI"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.batches: list[list[str]] = []
        _FakeOpenAI.instances.append(self)
        self.embeddings = types.SimpleNamespace(create=self._create)

    def _create(self, *, model: str, input: list[str]) -> Any:
        self.batches.append(list(input))
        return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[float(len(t))]) for t in input])


@pytest.fixture
def fake_openai(monkeypatch):
    _FakeOpenAI.instances = []
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setattr(rag_store, "_EMBED_CLIENTS", {})
    monkeypatch.setattr(rag_store.settings, "rag_embedding_model", "test-embedder")
    import app.services.chat_ai as chat_ai

    monkeypatch.setattr(chat_ai, "_require_api_key", lambda: "key")
    monkeypatch.setattr(chat_ai, "_base_url", lambda: "https://embed.example/v1")
    return _FakeOpenAI


def test_texts_are_sent_in_batches_and_come_back_in_order(fake_openai):
    texts = ["x" * n for n in range(1, 151)]
    vectors = rag_store._embed(texts)
    [client] = fake_openai.instances
    assert [len(batch) for batch in client.batches] == [64, 64, 22]
    assert vectors == [[float(n)] for n in range(1, 151)]


def test_the_client_has_a_timeout_and_is_reused(fake_openai):
    rag_store._embed(["a"])
    rag_store._embed(["b"])
    [client] = fake_openai.instances
    assert client.kwargs["timeout"] == rag_store._EMBED_TIMEOUT_SECONDS
