"""A question embeds its query once (eval/chat/README.md, "What a question costs the provider").

The chunk and the entity collection each embedded the same query text: two
provider requests a question where one does, against a cap of 800 requests a
day per key. Counted on a stand-in for the provider's embedding call.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.config import settings
from app.services import rag_store
from app.services.rag_store import EmbeddingUnavailable


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """A provider model is configured; every embedding request is recorded and answered with a fixed vector."""
    requests: list[list[str]] = []

    def embed(texts: list[str]) -> list[list[float]]:
        requests.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]

    monkeypatch.setattr(settings, "rag_embedding_model", "multilingual-e5-large-instruct")
    monkeypatch.setattr(rag_store, "_embed", embed)
    return requests


def test_the_same_query_is_embedded_once(provider: list[list[str]]) -> None:
    first = rag_store._embed_query("Who is the king?")
    second = rag_store._embed_query("Who is the king?")
    assert first == second and len(provider) == 1
    rag_store._embed_query("Who is the queen?")
    assert len(provider) == 2


def test_another_model_embeds_again(provider: list[list[str]], monkeypatch: pytest.MonkeyPatch) -> None:
    rag_store._embed_query("Who is the king?")
    monkeypatch.setattr(settings, "rag_embedding_model", "e5-mistral-7b-instruct")
    rag_store._embed_query("Who is the king?")
    assert len(provider) == 2


def test_a_failure_is_not_kept(monkeypatch: pytest.MonkeyPatch, provider: list[list[str]]) -> None:
    def down(texts: list[str]) -> Any:
        raise EmbeddingUnavailable("provider down")

    good = rag_store._embed
    monkeypatch.setattr(rag_store, "_embed", down)
    with pytest.raises(EmbeddingUnavailable):
        rag_store._embed_query("Who is the king?")
    monkeypatch.setattr(rag_store, "_embed", good)
    rag_store._embed_query("Who is the king?")
    assert len(provider) == 1  # asked again once the provider answers


def test_the_store_is_bounded(provider: list[list[str]], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag_store, "_QUERY_VECTORS_MAX", 3)
    for n in range(5):
        rag_store._embed_query(f"question {n}")
    assert len(rag_store._QUERY_VECTORS) == 3
    rag_store._embed_query("question 0")  # evicted, so embedded again
    assert len(provider) == 6


def test_the_local_space_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "rag_embedding_model", "")
    monkeypatch.setattr(rag_store, "_embed", lambda texts: None)
    assert rag_store._embed_query("Who is the king?") is None and not rag_store._QUERY_VECTORS


def test_a_question_searching_chunks_and_entities_makes_one_request(provider: list[list[str]], monkeypatch: pytest.MonkeyPatch) -> None:
    """Both collections of one question go through _embed_query; the second is served without a request."""
    seen: list[str] = []
    real = rag_store._embed_query

    def recording(query: str) -> Any:
        seen.append(query)
        return real(query)

    monkeypatch.setattr(rag_store, "_embed_query", recording)
    monkeypatch.setattr(rag_store, "_search_scope", lambda run_ids, asset_ref: [{"run_id": "r", "asset_ref": "a"}])
    monkeypatch.setattr(rag_store, "_ensure_chunk_runs_indexed", lambda scope: None)
    monkeypatch.setattr(rag_store, "_ensure_entity_runs_indexed", lambda scope: None)
    monkeypatch.setattr(settings, "rag_lexical_fusion", False)

    class Empty:
        def query(self, **kwargs: Any) -> dict[str, Any]:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        def count(self) -> int:
            return 0

    monkeypatch.setattr(rag_store, "_collection", lambda *a, **k: Empty())
    monkeypatch.setattr(rag_store, "_entity_collection", lambda *a, **k: Empty(), raising=False)
    rag_store.retrieve_chunks("Who is the king?", run_ids=["r"])
    rag_store.retrieve_entities("Who is the king?", run_ids=["r"])
    assert seen == ["Who is the king?", "Who is the king?"]
    assert len(provider) == 1
