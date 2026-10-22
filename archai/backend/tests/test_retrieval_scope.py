"""What the vector store may search: gate-approved runs, one per page, in one space.

These run against a real ChromaDB store in a temporary directory, with a
deterministic stand-in for the embedding provider, so the ``where`` filters,
deletions and collection names are ChromaDB's own.
"""

from __future__ import annotations

import math
import zlib
from pathlib import Path
from typing import Any

import pytest

from app.config import settings
from app.db import pipeline_db
from app.services import rag_store

PAGE_TEXT = [
    "Sage fu enfez et endiz",
    "Caldus li roys qui molt fu ber",
    "La fist requerre et demander",
]


def _embed(texts: list[str]) -> list[list[float]]:
    """Hashed character trigrams: similar text, similar vector; no model needed."""
    vectors = []
    for text in texts:
        vector = [0.0] * 64
        folded = f"  {text.lower()}  "
        for i in range(len(folded) - 2):
            vector[zlib.crc32(folded[i : i + 3].encode()) % 64] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vectors.append([v / norm for v in vector])
    return vectors


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(rag_store, "_CLIENT", None)
    monkeypatch.setattr(settings, "rag_embedding_model", "test-embedder")
    monkeypatch.setattr(rag_store, "_embed", _embed)
    monkeypatch.setattr(settings, "rag_auto_index", True)
    monkeypatch.setattr(settings, "rag_lexical_fusion", True)


def _run(asset_ref: str, *, sha: str | None, allowed: bool | None, lines: list[str] = PAGE_TEXT) -> str:
    run_id = pipeline_db.create_run(asset_ref=asset_ref, asset_sha256=sha)
    rows, offset = [], 0
    for idx, text in enumerate(lines):
        rows.append({"chunk_id": f"{run_id}:{idx}", "idx": idx, "start_offset": offset, "end_offset": offset + len(text), "text": text})
        offset += len(text) + 1
    pipeline_db.insert_chunks(run_id, rows)
    if allowed is not None:
        pipeline_db.update_run_fields(run_id, search_allowed=1 if allowed else 0)
    return run_id


def _report(run_id: str, *, pass_idx: int, allowed: bool) -> None:
    pipeline_db.insert_ocr_quality_report(
        run_id,
        {"pass_idx": pass_idx, "quality_label": "OK" if allowed else "RISKY", "token_search_allowed": allowed},
    )


def _chunks() -> Any:
    return rag_store._collection(backend_key=rag_store._space_key())


def _indexed_runs() -> set[str]:
    return {meta["run_id"] for meta in _chunks().get(include=["metadatas"])["metadatas"]}


# ── the quality gate ────────────────────────────────────────────────────


def test_a_query_never_indexes_a_run_the_gate_refused(store: None) -> None:
    """Reproduction from the accuracy roadmap: one retrieval made 15 refused chunks citable."""
    refused = _run("garbled-page", sha="sha-g", allowed=False)
    hits = rag_store.retrieve_chunks("what does this page say?", run_ids=[refused])
    assert (list(hits), hits.mode, hits.notes) == ([], "none", ["no searchable run in scope"])
    assert _chunks().count() == 0


def test_index_run_refuses_the_run_and_removes_what_a_bypass_left(store: None) -> None:
    refused = _run("garbled-page", sha="sha-g", allowed=False)
    _chunks().upsert(
        ids=[f"{refused}:0"], documents=[PAGE_TEXT[0]], metadatas=[{"run_id": refused}], embeddings=_embed([PAGE_TEXT[0]])
    )
    result = rag_store.index_run(refused)
    assert result["status"] == "blocked_by_quality_gate"
    assert result["chunks_indexed"] == 0
    assert _chunks().count() == 0


def test_entity_retrieval_respects_the_gate(store: None) -> None:
    refused = _run("garbled-page", sha="sha-g", allowed=False)
    hits = rag_store.retrieve_entities("Caldus", run_ids=[refused])
    assert (list(hits), hits.mode) == ([], "none")


def test_older_runs_fall_back_to_their_latest_quality_report(store: None) -> None:
    proofread_passed = _run("a", sha="sha-a", allowed=None)
    _report(proofread_passed, pass_idx=0, allowed=False)
    _report(proofread_passed, pass_idx=10, allowed=True)
    proofread_failed = _run("b", sha="sha-b", allowed=None)
    _report(proofread_failed, pass_idx=0, allowed=True)
    _report(proofread_failed, pass_idx=10, allowed=False)
    ungraded = _run("c", sha="sha-c", allowed=None)
    searchable = {row["run_id"] for row in pipeline_db.searchable_runs()}
    assert searchable == {proofread_passed}
    assert ungraded not in searchable


def test_a_recorded_decision_outranks_the_reports(store: None) -> None:
    """The tiled path writes a report per attempt and keeps the best, not the last."""
    run_id = _run("tiled", sha="sha-t", allowed=True)
    _report(run_id, pass_idx=0, allowed=True)
    _report(run_id, pass_idx=1, allowed=False)
    assert [row["run_id"] for row in pipeline_db.searchable_runs()] == [run_id]


# ── one run per page ────────────────────────────────────────────────────


def test_a_rerun_of_a_page_retires_the_earlier_vectors(store: None) -> None:
    first = _run("page-1-upload-a", sha="sha-page", allowed=True)
    rag_store.index_run(first)
    second = _run("page-1-upload-b", sha="sha-page", allowed=True)
    result = rag_store.index_run(second)
    assert result["superseded_runs_retired"] == [first]
    assert _indexed_runs() == {second}


def test_global_search_returns_one_copy_per_page(store: None) -> None:
    """Even with an older run's vectors still in the store, as older code left them."""
    first = _run("e-codices_fmb-cb-0001_001r_max.jpg", sha="sha-page", allowed=True)
    second = _run("thesis-page-001r", sha="sha-page", allowed=True)
    rag_store.index_run(second)
    _chunks().upsert(
        ids=[f"{first}:{i}" for i in range(len(PAGE_TEXT))],
        documents=PAGE_TEXT,
        metadatas=[{"run_id": first, "asset_ref": "e-codices_fmb-cb-0001_001r_max.jpg", "chunk_idx": i} for i in range(len(PAGE_TEXT))],
        embeddings=_embed(PAGE_TEXT),
    )
    hits = rag_store.retrieve_chunks("Caldus li roys", top_k=5)
    assert {hit["run_id"] for hit in hits} == {second}
    assert len({hit["text"] for hit in hits}) == len(hits)


def test_pages_without_an_image_hash_are_told_apart_by_asset_ref(store: None) -> None:
    left = _run("folio-1r", sha=None, allowed=True, lines=["Incipit liber primus"])
    right = _run("folio-1v", sha=None, allowed=True, lines=["Explicit liber primus"])
    assert {row["run_id"] for row in pipeline_db.searchable_runs()} == {left, right}


def test_an_explicit_run_is_searched_even_when_superseded(store: None) -> None:
    """Chat scopes to the transcription the user is looking at."""
    first = _run("page-a", sha="sha-page", allowed=True)
    _run("page-b", sha="sha-page", allowed=True)
    hits = rag_store.retrieve_chunks("Caldus li roys", top_k=3, run_ids=[first])
    assert hits and {hit["run_id"] for hit in hits} == {first}
    assert first in _indexed_runs()


def test_vectors_of_chunks_gone_from_the_database_are_not_returned(store: None) -> None:
    run_id = _run("page", sha="sha-page", allowed=True)
    rag_store.index_run(run_id)
    _chunks().upsert(
        ids=["orphan"], documents=["Caldus li roys qui molt fu ber"], metadatas=[{"run_id": run_id}],
        embeddings=_embed(["Caldus li roys qui molt fu ber"]),
    )
    hits = rag_store.retrieve_chunks("Caldus li roys qui molt fu ber", top_k=5, run_ids=[run_id])
    assert "orphan" not in {hit["chunk_id"] for hit in hits}
    assert hits[0]["chunk_id"] == f"{run_id}:1"


# ── one embedding space ─────────────────────────────────────────────────


def test_an_embedding_failure_writes_nothing_to_any_space(store: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def down(_texts: list[str]) -> list[list[float]]:
        raise rag_store.EmbeddingUnavailable("test-embedder: 401 Unauthorized")

    monkeypatch.setattr(rag_store, "_embed", down)
    run_id = _run("page", sha="sha-page", allowed=True)
    result = rag_store.index_run(run_id)
    assert result["status"] == "embedding_unavailable"
    names = {collection.name for collection in rag_store._chroma_client().list_collections()}
    assert settings.rag_collection_name not in names  # the local space was never touched
    assert _chunks().count() == 0


def test_the_space_follows_configuration_not_provider_health(store: None, monkeypatch: pytest.MonkeyPatch) -> None:
    assert rag_store._space_key() == "test_embedder"
    monkeypatch.setattr(settings, "rag_embedding_model", "")
    assert rag_store._space_key() == rag_store._LOCAL_EMBED_BACKEND


def test_retrieve_debug_reports_how_the_hits_were_found(store: None, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = _run("page", sha="sha-page", allowed=True)
    rag_store.index_run(run_id)

    def down(_texts: list[str]) -> list[list[float]]:
        raise rag_store.EmbeddingUnavailable("provider down")

    monkeypatch.setattr(rag_store, "_embed", down)
    payload = rag_store.retrieve_debug("Caldus li roys", k=2, run_id=run_id)
    assert payload["mode"] == "lexical"
    assert payload["results"][0]["chunk_id"] == f"{run_id}:1"
    assert any("provider down" in note for note in payload["notes"])
