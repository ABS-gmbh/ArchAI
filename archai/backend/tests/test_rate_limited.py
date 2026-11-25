"""A provider's 429 says how long to wait, when the provider says so.

The provider limits each user to 10 requests in 10 s and 400 an hour. Asked
through a stand-in that answers 429 with Retry-After: 7 (2026-11-25), the
chat route answered "The chat provider is limiting requests right now. Try
again in a minute." whatever the provider said, with no Retry-After header;
label analysis answered "The provider call failed (RateLimitError)", and the
verifier's verdict said its response "could not be parsed cleanly". Now each
gives one sentence with the provider's wait when it gave one.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.routers.errors import provider_failure
from app.services import chat_ai

KEY = "k" * 32


def _limited(retry_after: str | None) -> openai.RateLimitError:
    request = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return openai.RateLimitError("Rate limit exceeded: 10 per 10 second", response=httpx.Response(429, request=request, headers=headers), body=None)


@pytest.mark.parametrize(
    ("retry_after", "words", "header"),
    [
        ("7", "Try again in 7 seconds.", "7"),
        ("1", "Try again in 1 second.", "1"),
        ("2.5", "Try again in 3 seconds.", "3"),
        ("120", "Try again in 2 minutes.", "120"),
        ("3600", "Try again in 60 minutes.", "3600"),
        ("7200", "Try again in 2 hours.", "7200"),
        (None, "Try again in a minute.", None),
        ("soon", "Try again in a minute.", None),
        ("0", "Try again in a minute.", None),
    ],
)
def test_the_sentence_carries_the_providers_wait(retry_after: str | None, words: str, header: str | None) -> None:
    response = provider_failure(_limited(retry_after), provider="chat provider")
    assert response is not None and response.status_code == 429
    assert response.detail == f"The chat provider is limiting requests right now. {words}"
    assert (response.headers or {}).get("Retry-After") == header


def test_an_http_date_is_read_too() -> None:
    when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=40), usegmt=True)
    response = provider_failure(_limited(when), provider="chat provider")
    assert response is not None
    wait = int((response.headers or {})["Retry-After"])
    assert 35 <= wait <= 41
    assert response.detail.endswith(f"Try again in {wait} seconds.")


class _Provider(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("content-length") or 0))
        data = json.dumps({"error": {"message": "Rate limit exceeded: 10 per 10 second"}}).encode()
        self.send_response(429)
        self.send_header("content-type", "application/json")
        self.send_header("retry-after", "1")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    monkeypatch.setattr(settings, "chat_ai_api_key", KEY)
    for name in ("chat_ai_base_url", "archai_chat_ai_base_url"):
        monkeypatch.setattr(settings, name, base)
    yield
    server.shutdown()


SENTENCE = "The chat provider is limiting requests right now. Try again in 1 second."


def test_the_chat_answer_says_it_streamed_or_not(provider: None) -> None:
    client = TestClient(main.app)
    body = {"messages": [{"role": "user", "content": "Who?"}], "model": "qwen3-30b-a3b-instruct-2507"}
    answered = client.post("/api/chat/completions", json={**body, "stream": False})
    assert answered.status_code == 429
    assert answered.json() == {"detail": SENTENCE}
    assert answered.headers["retry-after"] == "1"
    streamed = client.post("/api/chat/completions", json={**body, "stream": True})
    events = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith("data: ")]
    assert events == [{"type": "error", "error": SENTENCE}]


def test_the_verifier_says_why_the_answer_was_not_checked(provider: None, monkeypatch: pytest.MonkeyPatch) -> None:
    verdict = chat_ai._run_paleography_verification(
        question="Who is named?",
        draft_text="Lausanne.",
        context={"ocr_run_id": "run-1"},
        evidence_text="[OCR_CHUNK_EVIDENCE] Caldus de Lausanne",
        answer_model="qwen3-30b-a3b-instruct-2507",
    )
    assert verdict is not None and verdict["assessment"] == "unavailable"
    assert verdict["notes"] == [f"Not checked: {SENTENCE}"]


class _LimitedClient:
    def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        raise _limited("7")

    def list_models(self) -> list[str]:
        return []


def test_label_analysis_says_it() -> None:
    import base64
    import io

    from PIL import Image

    from app.agents.label_analysis_agent import LabelAnalysisAgent, LabelAnalysisAgentError

    buffer = io.BytesIO()
    Image.new("RGB", (120, 90), "white").save(buffer, format="PNG")
    payload = SimpleNamespace(
        question="What is this embellished initial?",
        label_name="Embellished",
        image_b64=base64.b64encode(buffer.getvalue()).decode(),
        filename="folio_001.png",
        regions=[SimpleNamespace(region_id="ann-1", bbox_xyxy=[20, 15, 50, 45], polygons=[])],
    )
    with pytest.raises(LabelAnalysisAgentError) as caught:
        LabelAnalysisAgent(client=_LimitedClient()).run(payload)
    assert str(caught.value) == "SAIA label analysis failed: The chat provider is limiting requests right now. Try again in 7 seconds."
