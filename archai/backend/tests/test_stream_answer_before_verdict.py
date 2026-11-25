"""The streamed answer is complete before the verifier runs; the verdict follows on the same stream.

The stream used to send the answer's deltas, then run the verifier, then send
"done", and the workspace kept the composer busy until "done": a slow verifier
made the reader wait for an answer already on screen (the user, 2026-11-25:
"make the pipeline fast. we dont want to wait forever for answers"). Now an
"answer" event closes the answer (text, model, sources and retrieval), the
verifier runs, and a "verification" event carries the verdict. "done" stays
last and whole, for clients that read only it.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.services import chat_ai


class _Chunk:
    def __init__(self, text: str) -> None:
        self.choices = [type("C", (), {"delta": type("D", (), {"content": text})(), "message": None})()]


@pytest.fixture
def stream(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(chat_ai, "_create_client", lambda: object())
    monkeypatch.setattr(chat_ai, "_retrieve_evidence_text", lambda *a, **k: "[OCR_CHUNK_EVIDENCE] Caldus de Lausanne", raising=False)

    def run(**kwargs: Any) -> tuple[Any, str]:
        return iter([_Chunk("The page "), _Chunk("names Lausanne.")]), "chat-model"

    monkeypatch.setattr(chat_ai, "_run_chat_call_with_retries", run)

    def verify(**kwargs: Any) -> dict:
        calls.append("verified")
        return {"assessment": "supported", "corrected_answer": kwargs["draft_text"], "notes": [], "citations_checked": [],
                "model_used": "checker", "self_check": False}

    monkeypatch.setattr(chat_ai, "_run_paleography_verification", verify)
    monkeypatch.setattr(chat_ai, "_should_run_verification", lambda context, evidence: True)
    return calls


MESSAGES = [{"role": "user", "content": "Who is named?"}]
CONTEXT = {"ocr_run_id": "run-1", "transcript": "Caldus de Lausanne"}


def test_the_answer_event_comes_before_the_verifier_runs(stream: list[str]) -> None:
    events = chat_ai.stream_chat_completion(messages=MESSAGES, context=CONTEXT)
    seen = []
    for event in events:
        seen.append(event["type"])
        if event["type"] == "answer":
            assert stream == []  # the verifier has not run yet
            assert event["text"] == "The page names Lausanne."
            assert event["model"] == "chat-model"
            assert event["checking"] is True
            assert "retrieval" in event
    assert seen.index("answer") < seen.index("verification") < seen.index("done")
    assert seen[-1] == "done"


def test_the_verdict_arrives_as_its_own_event_and_done_is_whole(stream: list[str]) -> None:
    events = list(chat_ai.stream_chat_completion(messages=MESSAGES, context=CONTEXT))
    verdict = next(e for e in events if e["type"] == "verification")
    assert verdict["verification"]["assessment"] == "supported"
    assert verdict["retrieval"]["verifier"] == "ran"
    done = events[-1]
    assert done["type"] == "done" and done["verification"]["assessment"] == "supported"
    assert done["text"].startswith("The page names Lausanne.")


def test_no_check_means_no_waiting_and_no_verdict_event(stream: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_ai, "_should_run_verification", lambda context, evidence: False)
    monkeypatch.setattr(chat_ai, "_run_paleography_verification", lambda **k: None)
    events = list(chat_ai.stream_chat_completion(messages=MESSAGES, context=CONTEXT))
    answer = next(e for e in events if e["type"] == "answer")
    assert answer["checking"] is False
    assert not any(e["type"] == "verification" for e in events)
