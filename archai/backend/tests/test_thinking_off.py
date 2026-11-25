"""Provider-side thinking is turned off for the models that think by default.

Measured by worker 2 against the provider (2026-11-25): qwen3.5-397b, qwen3.6-35b,
qwen3.8-27b and glm-5.3 think by default, and with thinking on they returned
empty text at max_tokens 200 and took 10 to 48 s; with
extra_body={"chat_template_kwargs": {"enable_thinking": False}} qwen3.6-35b
answered an image question in 0.5 s. The app sent nothing of the kind. Now
every chat completion it asks for (chat, streamed or not, translation, the
verifier, and through saia_client label analysis and the provider OCR) carries
that kwarg while CHAT_DISABLE_THINKING is on, for every model but those named
in CHAT_DISABLE_THINKING_EXCEPT (mistral-large answers it with 400 "chat_template
is not supported"). A model that answers it so is asked again without it,
once, and not sent it again by this process.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from app.config import settings
from app.services import chat_ai, provider_thinking
from app.services.saia_client import SaiaClient

KEY = "k" * 32
THINKING_OFF = {"enable_thinking": False}


class _Provider(BaseHTTPRequestHandler):
    """A stand-in that records each body; `rejects` lists models that answer the kwarg with 400."""

    protocol_version = "HTTP/1.1"
    bodies: list[dict[str, Any]] = []
    rejects: set[str] = set()

    def _json(self, status: int, body: object) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._json(200, {"object": "list", "data": []})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
        type(self).bodies.append(body)
        model = str(body.get("model"))
        if model in self.rejects and "chat_template_kwargs" in body:
            return self._json(400, {"error": {"message": "chat_template is not supported"}})
        if body.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            for line in (
                json.dumps({"id": "c", "object": "chat.completion.chunk", "created": 0, "model": model,
                            "choices": [{"index": 0, "delta": {"content": "Lausanne."}, "finish_reason": None}]}),
                "[DONE]",
            ):
                data = f"data: {line}\n\n".encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            return None
        return self._json(200, {"id": "a", "object": "chat.completion", "created": 0, "model": model,
                                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Lausanne."}, "finish_reason": "stop"}]})

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_Provider]]:
    handler = type("Provider", (_Provider,), {"bodies": [], "rejects": set()})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    monkeypatch.setattr(settings, "chat_ai_api_key", KEY)
    for name in ("chat_ai_base_url", "archai_chat_ai_base_url"):
        monkeypatch.setattr(settings, name, base)
    monkeypatch.setattr(settings, "chat_disable_thinking", True)
    monkeypatch.setattr(settings, "chat_disable_thinking_except", "mistral-large")
    monkeypatch.setattr(SaiaClient, "_shared_models_cache", None)
    provider_thinking.reset()
    yield handler
    server.shutdown()
    provider_thinking.reset()


MESSAGES = [{"role": "user", "content": "Who is named?"}]


def _kwargs(body: dict[str, Any]) -> object:
    return body.get("chat_template_kwargs")


def test_every_model_is_asked_with_thinking_off_but_an_excepted_one(provider: type[_Provider]) -> None:
    chat_ai.create_chat_completion(messages=MESSAGES, model="qwen3.6-35b-a3b")
    chat_ai.create_chat_completion(messages=MESSAGES, model="a-model-added-tomorrow")
    chat_ai.create_chat_completion(messages=MESSAGES, model="mistral-large-4-1t-a52b")
    assert _kwargs(provider.bodies[0]) == THINKING_OFF
    assert _kwargs(provider.bodies[1]) == THINKING_OFF
    assert "chat_template_kwargs" not in provider.bodies[2]
    assert len(provider.bodies) == 3


def test_the_streamed_answer_too(provider: type[_Provider]) -> None:
    events = list(chat_ai.stream_chat_completion(messages=MESSAGES, model="qwen3.5-397b-a17b"))
    assert any(e.get("type") == "done" for e in events)
    assert _kwargs(provider.bodies[0]) == THINKING_OFF


def test_label_analysis_and_the_provider_ocr_path_too(provider: type[_Provider]) -> None:
    client = SaiaClient()
    client.chat_completion(model="qwen3.5-397b-a17b", messages=MESSAGES, deterministic=True)
    body = provider.bodies[0]
    assert _kwargs(body) == THINKING_OFF
    assert body.get("do_sample") is False  # the deterministic flag's own extra_body is kept


def test_the_verifier_too(provider: type[_Provider], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "paleography_verification_model", "qwen3.8-27b")
    chat_ai._run_paleography_verification(
        question="Who is named?",
        draft_text="Lausanne.",
        context={"ocr_run_id": "run-1"},
        evidence_text="[OCR_CHUNK_EVIDENCE] Caldus de Lausanne",
        answer_model="qwen3-30b-a3b-instruct-2507",
    )
    verifier_bodies = [b for b in provider.bodies if b.get("model") == "qwen3.8-27b"]
    assert verifier_bodies and _kwargs(verifier_bodies[0]) == THINKING_OFF


def test_switched_off_nothing_is_sent(provider: type[_Provider], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "chat_disable_thinking", False)
    chat_ai.create_chat_completion(messages=MESSAGES, model="qwen3.6-35b-a3b")
    assert "chat_template_kwargs" not in provider.bodies[0]


def test_a_model_that_rejects_the_kwarg_is_asked_again_without_it_and_remembered(provider: type[_Provider]) -> None:
    provider.rejects = {"another-model"}
    first = chat_ai.create_chat_completion(messages=MESSAGES, model="another-model")
    assert first["text"].startswith("Lausanne.")
    assert [("chat_template_kwargs" in b) for b in provider.bodies] == [True, False]
    chat_ai.create_chat_completion(messages=MESSAGES, model="another-model")
    assert "chat_template_kwargs" not in provider.bodies[2]  # not sent again by this process
    assert provider_thinking.rejected() == {"another-model"}


def test_the_streamed_answer_is_asked_again_too(provider: type[_Provider]) -> None:
    provider.rejects = {"another-model"}
    events = list(chat_ai.stream_chat_completion(messages=MESSAGES, model="another-model"))
    assert any(e.get("type") == "done" for e in events)
    assert [("chat_template_kwargs" in b) for b in provider.bodies] == [True, False]


def test_a_400_that_is_not_about_the_kwarg_is_not_blamed_on_it(provider: type[_Provider]) -> None:
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        import httpx
        import openai

        request = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
        raise openai.BadRequestError("context too long", response=httpx.Response(400, request=request), body=None)

    with pytest.raises(Exception, match="context too long"):
        provider_thinking.create(create, model="qwen3.6-35b-a3b", messages=MESSAGES)
    assert len(calls) == 1  # not asked again: the 400 does not name the chat template
    assert provider_thinking.rejected() == set()
