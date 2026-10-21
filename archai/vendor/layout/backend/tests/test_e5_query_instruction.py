"""e5-instruct queries carry their instruction; documents and other models do not."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.config import settings
from app.db import pipeline_db
from app.services import rag_store
from app.services.rag_store import E5_RETRIEVAL_TASK, embedding_input

E5 = "multilingual-e5-large-instruct"


def test_an_e5_instruct_query_states_the_task() -> None:
    assert embedding_input("Caldus li roys", query=True, model=E5) == f"Instruct: {E5_RETRIEVAL_TASK}\nQuery: Caldus li roys"
    assert embedding_input("q", query=True, model=f"intfloat/{E5}").startswith("Instruct: ")


def test_documents_are_embedded_as_they_are() -> None:
    """Only the query side changes, so the stored embeddings stay valid."""
    assert embedding_input("Caldus li roys", query=False, model=E5) == "Caldus li roys"


@pytest.mark.parametrize("model", ["multilingual-e5-large", "text-embedding-3-small", ""])
def test_other_models_get_the_text_unchanged(model: str) -> None:
    assert embedding_input("Caldus li roys", query=True, model=model) == "Caldus li roys"


def test_the_configured_model_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "rag_embedding_model", E5)
    assert embedding_input("q", query=True).startswith("Instruct: ")
    monkeypatch.setattr(settings, "rag_embedding_model", "")
    assert embedding_input("q", query=True) == "q"


@pytest.fixture
def searchable_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, list[list[str]]]:
    """A gate-approved run, and an embedder that records what it was given and then fails."""
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    monkeypatch.setattr(settings, "rag_embedding_model", E5)
    run_id = pipeline_db.create_run(asset_ref="page-1", asset_sha256="sha-1")
    pipeline_db.insert_chunks(run_id, [{"chunk_id": "c0", "idx": 0, "start_offset": 0, "end_offset": 14, "text": "Caldus li roys"}])
    pipeline_db.update_run_fields(run_id, search_allowed=1)
    seen: list[list[str]] = []

    def embed(texts: list[str]) -> Any:
        seen.append(list(texts))
        raise rag_store.EmbeddingUnavailable("recorded")

    monkeypatch.setattr(rag_store, "_embed", embed)
    return run_id, seen


def test_chunk_retrieval_embeds_the_query_with_its_instruction(searchable_run: tuple[str, list[list[str]]]) -> None:
    run_id, seen = searchable_run
    rag_store.retrieve_chunks("Caldus li roys", top_k=1, run_ids=[run_id])
    assert seen == [[f"Instruct: {E5_RETRIEVAL_TASK}\nQuery: Caldus li roys"]]


def test_entity_retrieval_embeds_the_query_with_its_instruction(searchable_run: tuple[str, list[list[str]]]) -> None:
    run_id, seen = searchable_run
    rag_store.retrieve_entities("Caldus li roys", top_k=1, run_ids=[run_id])
    assert seen == [[f"Instruct: {E5_RETRIEVAL_TASK}\nQuery: Caldus li roys"]]
