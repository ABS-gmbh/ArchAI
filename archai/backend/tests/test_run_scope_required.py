"""RAG_REQUIRE_RUN_SCOPE: on a server several groups of readers use, a search names its run.

Without a run, retrieval covers the newest reading of every page on the
server. A chat question sent with no run in its context, or
/api/index/search without run_id, was therefore answered from every reader's
pages, with their run ids in the sources (found before a deployment was opened
to the public, 2026-11-25). With the setting on, such a search finds nothing
and says why; a search that names a run is unchanged. Off by default: a
single reader's installation searches across its own pages.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings, settings
from app.routers import index as index_router
from app.services import chat_ai, rag_store


@pytest.fixture
def scope_calls(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Records every lookup of the runs in scope, and says none are searchable."""
    calls: list[Any] = []

    def scope(run_ids: Any, asset_ref: Any) -> list[str]:
        calls.append((list(run_ids) if run_ids else None, asset_ref))
        return []

    monkeypatch.setattr(rag_store, "_search_scope", scope)
    return calls


def test_it_is_off_unless_set() -> None:
    assert Settings.model_fields["rag_require_run_scope"].default is False


def test_off_a_search_without_a_run_covers_the_server(scope_calls: list[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "rag_require_run_scope", False)
    rag_store.retrieve_chunks("who is the king")
    rag_store.retrieve_entities("who is the king")
    assert scope_calls == [(None, None), (None, None)]


@pytest.mark.parametrize("run_ids", [None, [], [""], ["  "]])
def test_on_a_search_without_a_run_finds_nothing_and_says_why(
    run_ids: Any, scope_calls: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "rag_require_run_scope", True)
    for retrieve in (rag_store.retrieve_chunks, rag_store.retrieve_entities):
        result = retrieve("who is the king", run_ids=run_ids, asset_ref="page-1.png")
        assert list(result) == []
        assert result.mode == "none"
        assert result.notes == [rag_store.RUN_SCOPE_REQUIRED]
    assert scope_calls == []  # no run was looked up, so nothing of another reader's was read


def test_on_a_search_that_names_a_run_is_unchanged(scope_calls: list[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "rag_require_run_scope", True)
    rag_store.retrieve_chunks("who is the king", run_ids=["run-1"])
    rag_store.retrieve_entities("who is the king", run_ids=["run-1"])
    assert scope_calls == [(["run-1"], None), (["run-1"], None)]


def test_on_a_chat_question_without_a_run_gets_no_evidence(scope_calls: list[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "rag_require_run_scope", True)
    evidence = chat_ai._retrieve_evidence([{"role": "user", "content": "Who is the king?"}], {})
    assert str(evidence) == ""
    assert scope_calls == []
    assert rag_store.RUN_SCOPE_REQUIRED in " ".join(str(note) for note in evidence.retrieval.get("notes", []))


def test_on_a_chat_question_with_its_run_searches_that_run(scope_calls: list[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "rag_require_run_scope", True)
    chat_ai._retrieve_evidence([{"role": "user", "content": "Who is the king?"}], {"ocr_run_id": "run-1"})
    assert scope_calls and all(run_ids == ["run-1"] for run_ids, _ in scope_calls)


def test_on_the_search_route_refuses_a_search_without_run_id(scope_calls: list[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "rag_require_run_scope", True)
    app = FastAPI()
    app.include_router(index_router.router)
    client = TestClient(app)
    refused = client.post("/index/search?query=arthur")
    assert refused.status_code == 422
    assert "run_id" in refused.json()["detail"]
    assert scope_calls == []
    assert client.post("/index/search?query=arthur&run_id=run-1").status_code == 200
    assert scope_calls and scope_calls[0][0] == ["run-1"]
