"""Without a chat key, or with one the provider refuses, a reader is told what they can act on.

The workspace showed end users "Chat API key not configured. Set
CHAT_AI_API_KEY (or SAIA_API_KEY) in backend/.env or environment." (live
regression, 2026-11-24). The sentence is now for the reader; the setting names
go to the server log once per failure streak, and are already in the health
banner. Questions, translation (a chat completion), the stream, the model list
and label analysis all answer the same way.
"""

from __future__ import annotations

import json
import logging

import httpx
import openai
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import chat
from app.routers.errors import NOT_SET_UP, REFUSED_KEY, configuration_message
from app.services import chat_ai, setup_notices
from app.services.chat_ai import ChatConfigError
from app.services.saia_client import SaiaConfigError

KEY_MESSAGE = "Chat API key not configured. Set CHAT_AI_API_KEY (or SAIA_API_KEY) in backend/.env or environment."
MESSAGES = {"messages": [{"role": "user", "content": "hi"}]}


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(chat.router)
    return TestClient(app)


def _raise(exc: BaseException):
    def raiser(*args, **kwargs):
        raise exc

    return raiser


@pytest.fixture(autouse=True)
def fresh_streak() -> None:
    setup_notices.reset()


def test_the_sentences_name_no_setting() -> None:
    for sentence in (NOT_SET_UP, REFUSED_KEY):
        assert "API_KEY" not in sentence and ".env" not in sentence
    assert NOT_SET_UP == "Chat is not set up on this server. Tell whoever runs it."
    assert REFUSED_KEY == "The chat provider refused this server's key. Tell whoever runs it."


@pytest.mark.parametrize(
    ("call", "target"),
    [
        (lambda c: c.get("/chat/models"), "list_available_models"),
        (lambda c: c.post("/chat/completions", json={**MESSAGES, "stream": False}), "create_chat_completion"),
    ],
)
def test_without_a_key_the_reader_is_told_chat_is_not_set_up(monkeypatch: pytest.MonkeyPatch, call, target: str) -> None:
    monkeypatch.setattr(chat, target, _raise(ChatConfigError(KEY_MESSAGE)))
    response = call(_client())
    assert response.json()["detail"] == NOT_SET_UP
    assert "CHAT_AI_API_KEY" not in response.text


def test_the_stream_says_the_same(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_key(**kwargs):
        raise ChatConfigError(KEY_MESSAGE)
        yield  # a generator, like the real one

    monkeypatch.setattr(chat, "stream_chat_completion", no_key)
    response = _client().post("/chat/completions", json={**MESSAGES, "stream": True})
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert events == [{"type": "error", "error": NOT_SET_UP}]


def test_label_analysis_says_the_same(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Agent:
        def run(self, payload):
            raise SaiaConfigError(KEY_MESSAGE)

    monkeypatch.setattr(chat, "_get_label_analysis_agent", lambda: _Agent())
    response = _client().post("/chat/label-analysis", json={"question": "q", "label_name": "MainZone", "image_b64": "AAAA"})
    assert response.json()["detail"] == NOT_SET_UP


def test_a_refused_key_is_said_for_the_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    request = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
    refused = openai.AuthenticationError("no", response=httpx.Response(401, request=request), body=None)
    monkeypatch.setattr(chat, "create_chat_completion", _raise(refused))
    response = _client().post("/chat/completions", json={**MESSAGES, "stream": False})
    assert (response.status_code, response.json()["detail"]) == (502, REFUSED_KEY)


def test_the_settings_are_logged_once_per_failure_streak(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(chat, "create_chat_completion", _raise(ChatConfigError(KEY_MESSAGE)))
    client = _client()
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            client.post("/chat/completions", json={**MESSAGES, "stream": False})
    assert sum("CHAT_AI_API_KEY" in record.getMessage() for record in caplog.records) == 1

    # A key is found again (the operator set it): the next failure starts a new streak.
    monkeypatch.setattr(chat_ai.settings, "chat_ai_api_key", "k" * 30)
    chat_ai._require_api_key()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        client.post("/chat/completions", json={**MESSAGES, "stream": False})
    assert sum("CHAT_AI_API_KEY" in record.getMessage() for record in caplog.records) == 1


def test_other_providers_get_their_own_words() -> None:
    assert configuration_message(SaiaConfigError(KEY_MESSAGE), provider="OCR provider") == (
        "Reading with the OCR provider is not set up on this server. Tell whoever runs it."
    )
