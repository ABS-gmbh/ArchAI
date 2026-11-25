"""A model that spends its answer on thinking is said to, not taken as an empty answer.

Measured against the provider (2026-11-25): thinking arrives in
`reasoning_content`, not `content`, and with a max_tokens budget a thinking
model can return an empty `content` because the budget went on thinking
(glm-5.3 keeps thinking although asked not to; mistral-large cannot be asked).
The app took that as an empty answer. Now the reader is told the model spent
its answer on thinking (chat, streamed or not, and translation), label
analysis falls back or says so, the verifier's verdict says the answer was not
checked and why, and the proofreader leaves the text as read. Thinking text is
never shown as answer text.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.agents.label_analysis_agent import LabelAnalysisAgent
from app.agents.ocr_proofreader_agent import OcrProofreaderAgent
from app.config import settings
from app.services import chat_ai, provider_thinking
from app.services.saia_client import SaiaClient

KEY = "k" * 32
THINKING = "Let me consider the evidence line by line. The name might be"
THINKING_ONLY_SENTENCE = (
    "The model glm-5.3 spent its whole answer on thinking and gave no answer. Ask again, or choose another model."
)


class _Provider(BaseHTTPRequestHandler):
    """A stand-in whose answers carry thinking; `answer` is the content after it ("" for none)."""

    protocol_version = "HTTP/1.1"
    answer = ""

    def _json(self, body: object) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._json({"object": "list", "data": []})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
        model = str(body.get("model"))
        if not body.get("stream"):
            message = {"role": "assistant", "content": self.answer, "reasoning_content": THINKING}
            finish = "stop" if self.answer else "length"
            return self._json({"id": "a", "object": "chat.completion", "created": 0, "model": model,
                               "choices": [{"index": 0, "message": message, "finish_reason": finish}]})
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        deltas: list[dict[str, Any]] = [{"reasoning_content": part} for part in THINKING.split(" ")]
        if self.answer:
            deltas += [{"content": "Caldus "}, {"content": "de Lausanne."}]
        for delta in deltas:
            chunk = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": model,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            data = f"data: {json.dumps(chunk)}\n\n".encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        data = b"data: [DONE]\n\n"
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n0\r\n\r\n")
        return None

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_Provider]]:
    handler = type("Provider", (_Provider,), {"answer": ""})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    monkeypatch.setattr(settings, "chat_ai_api_key", KEY)
    for name in ("chat_ai_base_url", "archai_chat_ai_base_url"):
        monkeypatch.setattr(settings, name, base)
    monkeypatch.setattr(SaiaClient, "_shared_models_cache", None)
    yield handler
    server.shutdown()


MESSAGES = [{"role": "user", "content": "Who is named?"}]


def _post(stream: bool) -> Any:
    return TestClient(main.app).post(
        "/api/chat/completions", json={"messages": MESSAGES, "model": "glm-5.3", "stream": stream}
    )


def _events(response: Any) -> list[dict[str, Any]]:
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_a_thinking_only_answer_is_said_to_be_one(provider: type[_Provider]) -> None:
    response = _post(stream=False)
    assert response.status_code == 502
    assert response.json()["detail"] == THINKING_ONLY_SENTENCE


def test_the_streamed_answer_never_shows_the_thinking(provider: type[_Provider]) -> None:
    provider.answer = "yes"
    events = _events(_post(stream=True))
    deltas = "".join(e["delta"] for e in events if e.get("type") == "delta")
    answer = next(e for e in events if e.get("type") == "answer")
    assert answer["text"] == "Caldus de Lausanne."
    assert deltas.startswith("Caldus de Lausanne.")
    assert "consider" not in json.dumps(events)


def test_a_streamed_thinking_only_answer_is_said_to_be_one(provider: type[_Provider]) -> None:
    events = _events(_post(stream=True))
    assert events == [{"type": "error", "error": THINKING_ONLY_SENTENCE}]


def test_a_streamed_translation_too(provider: type[_Provider]) -> None:
    with pytest.raises(provider_thinking.ThinkingOnlyAnswer):
        list(chat_ai.stream_chat_completion(messages=MESSAGES, model="glm-5.3", context={"stage": "translation"}))


def test_an_answer_after_thinking_is_the_answer(provider: type[_Provider]) -> None:
    provider.answer = "Caldus de Lausanne."
    result = chat_ai.create_chat_completion(messages=MESSAGES, model="glm-5.3")
    assert result["text"].startswith("Caldus de Lausanne.")
    assert "consider" not in result["text"]


def test_saia_calls_raise_it_for_label_analysis_and_the_provider_ocr(provider: type[_Provider]) -> None:
    with pytest.raises(provider_thinking.ThinkingOnlyAnswer) as caught:
        SaiaClient().chat_completion(model="glm-5.3", messages=MESSAGES, max_tokens=32)
    assert caught.value.model == "glm-5.3"


def test_the_verifier_says_the_answer_was_not_checked_and_why(provider: type[_Provider], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "paleography_verification_model", "glm-5.3")
    verdict = chat_ai._run_paleography_verification(
        question="Who is named?",
        draft_text="Caldus de Lausanne.",
        context={"ocr_run_id": "run-1"},
        evidence_text="[OCR_CHUNK_EVIDENCE] Caldus de Lausanne",
        answer_model="qwen3-30b-a3b-instruct-2507",
    )
    assert verdict is not None
    assert verdict["assessment"] == "unavailable"
    assert verdict["notes"] == ["The checking model spent its whole answer on thinking; the answer was not checked."]
    assert verdict["model_used"] == "glm-5.3"


def test_the_proofreader_leaves_the_text_as_read(provider: type[_Provider]) -> None:
    agent = OcrProofreaderAgent(model_override="glm-5.3")
    assert agent.proofread("Caldus de Lausane", script_hint="latin") == "Caldus de Lausane"


class _ThinkingFirstClient:
    """Label analysis's client: the first model spends its 32 tokens thinking, the fallback answers."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.models: list[str] = []

    def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        model = kwargs["model"]
        self.models.append(model)
        if not self.answers.get(model):
            raise provider_thinking.ThinkingOnlyAnswer(model)
        return {"text": self.answers[model]}

    def list_models(self) -> list[str]:
        return list(self.answers)


def _initial_payload() -> SimpleNamespace:
    import base64
    import io

    from PIL import Image

    image = Image.new("RGB", (120, 90), (255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return SimpleNamespace(
        question="Which letter is this embellished initial?",
        label_name="Embellished",
        image_b64=base64.b64encode(buffer.getvalue()).decode(),
        filename="folio_001.png",
        regions=[SimpleNamespace(region_id="ann-1", bbox_xyxy=[20, 15, 50, 45], polygons=[])],
    )


def test_label_analysis_falls_back_when_the_first_model_only_thinks(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _ThinkingFirstClient({"thinking-vl": "", "answering-vl": "Letter: R"})
    agent = LabelAnalysisAgent(client=client)
    monkeypatch.setattr(agent, "_choose_retry_model", lambda requested_model, prefer_vision: "answering-vl")
    monkeypatch.setattr(settings, "label_visual_model", "thinking-vl")
    monkeypatch.setattr(settings, "saia_label_analysis_model", "thinking-vl")
    result = agent.run(_initial_payload())
    assert client.models[-1] == "answering-vl"
    assert result.model_used == "answering-vl"
    assert any("LABEL_THINKING_FALLBACK:answering-vl" in warning for warning in result.warnings)


def test_label_analysis_says_so_when_no_model_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.agents.label_analysis_agent import LabelAnalysisAgentError

    client = _ThinkingFirstClient({"thinking-vl": ""})
    agent = LabelAnalysisAgent(client=client)
    monkeypatch.setattr(agent, "_choose_retry_model", lambda requested_model, prefer_vision: None)
    monkeypatch.setattr(settings, "label_visual_model", "thinking-vl")
    monkeypatch.setattr(settings, "saia_label_analysis_model", "thinking-vl")
    with pytest.raises(LabelAnalysisAgentError) as caught:
        agent.run(_initial_payload())
    assert str(caught.value) == (
        "The model thinking-vl spent its whole answer on thinking and returned no answer text. "
        "Ask again, or choose another model."
    )
