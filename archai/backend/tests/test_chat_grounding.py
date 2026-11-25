"""Chat answers stay grounded in the page, and say when retrieval was degraded.

Before: the RAG instruction told the model to "answer from general knowledge"
when the evidence was irrelevant; a page-scoped question that retrieved
nothing got no instruction at all; and a lexical-only or failed retrieval was
invisible - retrieve_chunks' mode and notes were dropped, and an exception was
swallowed into "no hits".
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.config import settings
from app.services import chat_ai
from app.services.rag_store import RetrievalResult
from tests.test_chat_api import _FakeClient, _FakeStreamClient

PAGE = {"ocr_run_id": "run-1"}
PAGE_WITH_TRANSCRIPT = {"ocr_run_id": "run-1", "transcript": "Caldus li roys qui molt fu ber"}
HIT = {
    "chunk_id": "run-1:0",
    "run_id": "run-1",
    "asset_ref": "page-1",
    "chunk_idx": 0,
    "start_offset": 0,
    "end_offset": 14,
    "text": "Caldus li roys",
    "distance": None,
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> _FakeClient:
    fake = _FakeClient(["meta-llama-3.1-8b-instruct"], completion_text="Draft answer.")
    monkeypatch.setattr(chat_ai, "_create_client", lambda: fake)
    monkeypatch.setattr(settings, "chat_rag_model", "meta-llama-3.1-8b-instruct")
    monkeypatch.setattr(settings, "paleography_verification_enabled", False)
    monkeypatch.setattr("app.db.pipeline_db.list_mention_links_for_run", lambda run_id: [])
    monkeypatch.setattr("app.services.rag_store.retrieve_entities", lambda *a, **k: RetrievalResult(mode="dense"))
    return fake


def _chunks(monkeypatch: pytest.MonkeyPatch, result: Any) -> None:
    def retrieve(*args: Any, **kwargs: Any) -> Any:
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("app.services.rag_store.retrieve_chunks", retrieve)


def _ask(context: dict[str, Any] | None = PAGE) -> dict[str, Any]:
    return chat_ai.create_chat_completion(
        messages=[{"role": "user", "content": "Who is the king?"}], model=None, context=context
    )


def _system(fake: _FakeClient) -> str:
    return fake.calls[0]["messages"][0]["content"]


def test_irrelevant_evidence_no_longer_licenses_general_knowledge(client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _chunks(monkeypatch, RetrievalResult([HIT], mode="hybrid"))
    result = _ask()
    system = _system(client)
    assert "answer from general knowledge" not in system
    assert "sources for this page do not contain the answer" in system
    assert "Caldus li roys" in system
    assert result["text"] == "Draft answer."
    assert result["retrieval"] == {
        "status": "ok",
        "chunk_mode": "hybrid",
        "entity_mode": "dense",
        "chunk_hits": 1,
        "entity_hits": 0,
        "run_scoped": True,
        "notes": [],
        "verifier": "disabled",
        "transcript_in_context": False,
    }


def test_a_page_question_that_retrieves_nothing_is_told_to_say_so(client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _chunks(monkeypatch, RetrievalResult([], mode="hybrid"))
    result = _ask()
    assert "No passage of this page's transcription was retrieved" in _system(client)
    assert result["retrieval"]["status"] == "ok"
    assert result["text"] == "Draft answer."  # nothing degraded, so no notice


def test_lexical_only_retrieval_is_visible(client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _chunks(monkeypatch, RetrievalResult([HIT], mode="lexical", notes=["dense ranking unavailable: provider 401"]))
    result = _ask()
    assert result["retrieval"]["status"] == "degraded"
    assert result["retrieval"]["chunk_mode"] == "lexical"
    assert result["retrieval"]["notes"] == ["dense ranking unavailable"]  # the reason, not the error text
    assert result["text"].endswith("[Retrieval]\nSemantic search was unavailable; evidence was found by keyword match only.")


def test_a_retrieval_exception_is_reported_not_swallowed(client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _chunks(monkeypatch, RuntimeError("chroma is gone at /srv/secret/path"))
    result = _ask()
    assert result["retrieval"]["status"] == "unavailable"
    assert result["retrieval"]["chunk_mode"] == "error"
    assert result["retrieval"]["notes"] == ["chunk retrieval failed (RuntimeError)"]
    assert "/srv/secret/path" not in json.dumps(result)
    assert "not grounded in the page" in result["text"]
    assert "No passage of this page's transcription was retrieved" in _system(client)


def test_a_page_the_gate_refused_is_reported(client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _chunks(monkeypatch, RetrievalResult(mode="none", notes=["no searchable run in scope"]))
    result = _ask()
    assert result["retrieval"]["status"] == "no_searchable_run"
    assert "OCR quality gate did not allow search" in result["text"]


def test_a_general_chat_gets_no_page_notice(client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _chunks(monkeypatch, RetrievalResult(mode="none", notes=["no searchable run in scope"]))
    result = _ask(context=None)
    assert result["retrieval"]["run_scoped"] is False
    assert result["text"] == "Draft answer."
    assert "No passage of this page" not in _system(client)


def test_an_unavailable_verifier_is_reported(client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _chunks(monkeypatch, RetrievalResult([HIT], mode="hybrid"))
    monkeypatch.setattr(settings, "paleography_verification_enabled", True)
    monkeypatch.setattr(
        "app.agents.paleography_verification_agent.PaleographyVerificationAgent.run",
        lambda self, payload: (_ for _ in ()).throw(RuntimeError("401")),
    )
    result = _ask()
    assert result["retrieval"]["verifier"] == "unavailable"
    assert "[Verification]" not in result["text"]  # a state, not a block of text


def test_a_skipped_verifier_is_reported(client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _chunks(monkeypatch, RetrievalResult([], mode="hybrid"))
    monkeypatch.setattr(settings, "paleography_verification_enabled", True)
    assert _ask()["retrieval"]["verifier"] == "skipped"  # no evidence to check against


def test_the_stream_carries_the_status_and_the_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeStreamClient(["meta-llama-3.1-8b-instruct"], deltas=["Draft ", "answer."])
    monkeypatch.setattr(chat_ai, "_create_client", lambda: fake)
    monkeypatch.setattr(settings, "chat_rag_model", "meta-llama-3.1-8b-instruct")
    monkeypatch.setattr(settings, "paleography_verification_enabled", False)
    monkeypatch.setattr("app.db.pipeline_db.list_mention_links_for_run", lambda run_id: [])
    monkeypatch.setattr("app.services.rag_store.retrieve_entities", lambda *a, **k: RetrievalResult(mode="none"))
    _chunks(monkeypatch, RetrievalResult([HIT], mode="lexical", notes=["dense ranking unavailable: timeout"]))
    events = list(
        chat_ai.stream_chat_completion(messages=[{"role": "user", "content": "Who is the king?"}], context=PAGE_WITH_TRANSCRIPT)
    )
    done = events[-1]
    assert done["type"] == "done"
    assert done["retrieval"]["status"] == "degraded"
    assert done["text"].startswith("Draft answer.\n\n[Retrieval]")
    assert done["inspection"]["final_output"] == "Draft answer."
    assert done["retrieval"]["transcript_in_context"] is True


# ── the client sent the page transcript ────────────────────────────────


def test_a_gate_refused_page_with_a_transcript_may_be_answered_from_the_snippet(
    client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The snippet is in the prompt: "say the sources hold nothing" would contradict it."""
    _chunks(monkeypatch, RetrievalResult(mode="none", notes=["no searchable run in scope"]))
    result = _ask(context=PAGE_WITH_TRANSCRIPT)
    system = _system(client)
    assert "transcript_snippet:\nCaldus li roys qui molt fu ber" in system
    assert "No passage of this page's transcription was retrieved" not in system
    assert "Answer from that snippet alone if it holds the answer" in system
    assert "OCR quality gate did not allow search on this transcription" in system
    assert "general knowledge" in system
    assert result["retrieval"]["transcript_in_context"] is True
    assert "not grounded in the page" not in result["text"]
    assert result["text"].endswith("the answer can only draw on the transcript sent with the question, which may be unreliable.")


def test_a_failed_search_with_a_transcript_says_the_snippet_is_unsearched(
    client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _chunks(monkeypatch, RuntimeError("boom"))
    result = _ask(context=PAGE_WITH_TRANSCRIPT)
    system = _system(client)
    assert "Answer from that snippet alone" in system
    assert "OCR quality gate" not in system  # the gate did not refuse it; search failed
    assert result["text"].endswith("the answer can only draw on the transcript sent with the question, unsearched.")


def test_an_empty_search_with_a_transcript_points_at_the_snippet(client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _chunks(monkeypatch, RetrievalResult([], mode="hybrid"))
    result = _ask(context=PAGE_WITH_TRANSCRIPT)
    assert "Search retrieved no passage of this page" in _system(client)
    assert result["text"] == "Draft answer."  # search ran normally: no notice


def test_evidence_with_a_transcript_keeps_the_evidence_instruction(client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _chunks(monkeypatch, RetrievalResult([HIT], mode="hybrid"))
    _ask(context=PAGE_WITH_TRANSCRIPT)
    system = _system(client)
    assert "sources for this page do not contain the answer" in system
    assert "Search retrieved no passage" not in system
