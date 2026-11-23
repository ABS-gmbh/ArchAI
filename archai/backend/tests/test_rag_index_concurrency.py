"""Concurrent retrievals for one unindexed run embed its chunks once (RAG-03).

Auto-indexing checks whether a run's vectors are current and then indexes it.
Unlocked, two chat requests for a freshly finished run both saw it unindexed and
both embedded and upserted every chunk, paying the provider twice on the request
path.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from app.config import settings
from app.db import pipeline_db
from app.services import rag_store
from tests.test_retrieval_scope import _embed, _run


@pytest.fixture
def slow_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []

    def slow_embed(texts: list[str]) -> list[list[float]]:
        calls.append(len(texts))
        time.sleep(0.3)  # long enough for a second request to reach the check
        return _embed(texts)

    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(rag_store, "_CLIENT", None)
    monkeypatch.setattr(settings, "rag_embedding_model", "test-embedder")
    monkeypatch.setattr(rag_store, "_embed", slow_embed)
    monkeypatch.setattr(settings, "rag_auto_index", True)
    return calls


def _concurrently(target, n: int = 3) -> None:
    errors: list[BaseException] = []

    def wrapped() -> None:
        try:
            target()
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=wrapped) for _ in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not errors, errors


def test_concurrent_auto_index_of_one_run_embeds_its_chunks_once(slow_store: list[int]) -> None:
    run_id = _run("page", sha="sha-p", allowed=True)
    _concurrently(lambda: rag_store._ensure_chunk_runs_indexed([run_id]))
    assert slow_store == [3]
    assert rag_store._collection(backend_key=rag_store._space_key()).count() == 3


def test_concurrent_auto_index_of_one_run_embeds_its_entities_once(
    slow_store: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = _run("page", sha="sha-p", allowed=True)
    payload = (["e1", "e2"], ["Caldus li roys", "Sage"], [{"run_id": run_id}, {"run_id": run_id}], 0)
    monkeypatch.setattr(rag_store, "_build_entity_index_payload", lambda _run_id, _run: payload)
    _concurrently(lambda: rag_store._ensure_entity_runs_indexed([run_id]))
    assert slow_store == [2]


def test_different_runs_still_index_in_parallel(slow_store: list[int], monkeypatch: pytest.MonkeyPatch) -> None:
    first = _run("page-a", sha="sha-a", allowed=True)
    second = _run("page-b", sha="sha-b", allowed=True)
    runs = iter([first, second])
    lock = threading.Lock()
    # Both embeddings must be in flight at once to pass the barrier. Behind one
    # shared lock the first would wait here alone until the barrier breaks. No
    # wall-clock limit: a slow CI runner made the old 0.55 s bound fail.
    together = threading.Barrier(2, timeout=10)

    def embed_together(texts: list[str]) -> list[list[float]]:
        slow_store.append(len(texts))
        together.wait()
        return _embed(texts)

    monkeypatch.setattr(rag_store, "_embed", embed_together)

    def next_run() -> None:
        with lock:
            run_id = next(runs)
        rag_store._ensure_chunk_runs_indexed([run_id])

    _concurrently(next_run, n=2)
    assert slow_store == [3, 3]
    assert not together.broken  # not serialised behind one lock


def test_delete_waits_for_an_index_of_the_same_run_in_flight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DELETE /index/run/{id} during indexing used to leave the vectors the index upserted after it."""
    embedding = threading.Event()

    def slow_embed(texts: list[str]) -> list[list[float]]:
        embedding.set()
        time.sleep(0.3)
        return _embed(texts)

    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(rag_store, "_CLIENT", None)
    monkeypatch.setattr(settings, "rag_embedding_model", "test-embedder")
    monkeypatch.setattr(rag_store, "_embed", slow_embed)
    monkeypatch.setattr(rag_store, "_build_entity_index_payload", lambda _run_id, _run: ([], [], [], 0))
    run_id = _run("page", sha="sha-p", allowed=True)

    indexer = threading.Thread(target=rag_store.index_run, args=(run_id,))
    indexer.start()
    assert embedding.wait(timeout=5)
    deleted = rag_store.delete_run(run_id)  # arrives while the chunks are being embedded
    indexer.join(timeout=10)

    assert deleted == 3
    chunks = rag_store._collection(backend_key=rag_store._space_key())
    assert chunks.get(where={"run_id": run_id}, include=[])["ids"] == []
