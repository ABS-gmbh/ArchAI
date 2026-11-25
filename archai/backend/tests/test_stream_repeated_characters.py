"""A streamed answer must be what the model wrote, character for character.

Found on 2026-11-25 with a live provider: the stream "1", "3", "0", "0" reached
the reader as "130", "1100" as "10", "aabb" as "abb" and a chunk id lost a
digit. The stream helper took any chunk that repeated the end of the text so
far for a re-sent one and dropped it. In a tool for dating manuscripts that
turns 1300 into 130 without a trace.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.config import settings
from app.services import chat_ai
from app.services.chat_ai import _normalize_stream_delta, stream_chat_completion


class _StreamClient:
    def __init__(self, deltas: list[str]) -> None:
        self._deltas = deltas
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="chat-model")]))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        return (SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=delta))]) for delta in self._deltas)


def _streamed(monkeypatch: pytest.MonkeyPatch, deltas: list[str], context: dict[str, Any] | None = None) -> tuple[str, str]:
    monkeypatch.setattr(chat_ai, "_create_client", lambda: _StreamClient(deltas))
    monkeypatch.setattr(settings, "chat_rag_model", "chat-model")
    monkeypatch.setattr(settings, "translation_model", "chat-model")
    events = list(stream_chat_completion(messages=[{"role": "user", "content": "Say it."}], model=None, context=context))
    assert events[-1]["type"] == "done"
    return "".join(event["delta"] for event in events if event["type"] == "delta"), events[-1]["text"]


# What the live provider sent, token by token, for "1300 1100 2000 aabbccdd 0000 success committee 9d18c1175685".
LIVE_DELTAS = ["1", "3", "0", "0", " ", "1", "1", "0", "0", " ", "2", "0", "0", "0", " a", "a", "bb", "cc", "dd", " ", "0", "0", "0", "0",
               " success", " committee", " ", "9", "d", "1", "8", "c", "1", "1", "7", "5", "6", "8", "5"]
LIVE_TEXT = "1300 1100 2000 aabbccdd 0000 success committee 9d18c1175685"


def test_a_chunk_that_repeats_the_end_of_the_text_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    shown, final = _streamed(monkeypatch, LIVE_DELTAS)
    assert shown == LIVE_TEXT
    assert final == LIVE_TEXT


def test_repeated_words_and_overlapping_chunks_are_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    deltas = ["had", " had", " had", " no", " no", "non", "sense", ".", ".", "."]
    shown, final = _streamed(monkeypatch, deltas)
    assert shown == final == "had had had no nononsense..."


def test_a_translation_keeps_its_years(monkeypatch: pytest.MonkeyPatch) -> None:
    deltas = ["In the year ", "1", "3", "0", "0", " the bishop of Lausanne", " gave", " 2", "0", "0", " pounds."]
    shown, final = _streamed(monkeypatch, deltas, context={"chat_stage": "translation", "transcript": "Anno domini .m.ccc."})
    assert "1300" in final and "200 pounds" in final
    assert shown == final


def test_every_chunk_is_appended_as_it_came() -> None:
    text = ""
    for chunk in ["1", "1", "13", "3", "0", "0"]:
        delta, text = _normalize_stream_delta(chunk, text)
        assert delta == chunk
    assert text == "1113300"
    assert _normalize_stream_delta("", "abc") == ("", "abc")


def test_a_provider_that_resends_the_whole_text_can_be_declared(monkeypatch: pytest.MonkeyPatch) -> None:
    # Not the OpenAI-compatible behaviour, so it is a setting and off by default.
    monkeypatch.setattr(settings, "chat_stream_cumulative", True)
    shown, final = _streamed(monkeypatch, ["T", "Tr", "Tra", "Trans", "Transl", "Translate"])
    assert shown == final == "Translate"
