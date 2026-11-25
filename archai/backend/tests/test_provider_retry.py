"""Who tries a provider call again, and how often: provider_retry, once, for every client.

Measured against a stand-in on 2026-11-25. Before, the OpenAI SDK retried
every 429 and 5xx twice by itself. That made 3 calls per failure, and 12 s
before a 429 with Retry-After: 6 reached the reader; a 502 cost 18 chat calls
(SDK x chat_ai's own retries x a second model) and 9 in label analysis. Now
the SDK retries nothing (max_retries=0) and provider_retry.call, around every
call to the provider (chat, saia, embeddings, the model lists), makes it
again at most once:

* a 429 whose Retry-After is at most PROVIDER_RETRY_MAX_WAIT_SECONDS (5),
  after that wait;
* a 5xx, or a refused connection, after a short pause;
* nothing else: a longer wait, or a 429 with no Retry-After (the hourly cap),
  is answered at once.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from app.config import settings
from app.services import chat_ai, provider_retry, rag_store
from app.services.saia_client import SaiaClient

KEY = "k" * 32
MESSAGES = [{"role": "user", "content": "Who is named?"}]


class _Provider(BaseHTTPRequestHandler):
    """Answers each POST from `script` in turn (its last entry from then on): a status and a Retry-After or None."""

    protocol_version = "HTTP/1.1"
    script: list[tuple[int, str | None]] = [(200, None)]
    posts: list[str] = []

    def _send(self, status: int, body: object, retry_after: str | None = None) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        if retry_after is not None:
            self.send_header("retry-after", retry_after)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._send(200, {"object": "list", "data": [{"id": "qwen3.5-397b-a17b", "object": "model"}]})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
        type(self).posts.append(self.path)
        status, retry_after = self.script[min(len(self.posts) - 1, len(self.script) - 1)]
        if status != 200:
            return self._send(status, {"error": {"message": "limited" if status == 429 else "upstream"}}, retry_after)
        if self.path.endswith("/embeddings"):
            return self._send(200, {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                                    "model": "e5", "usage": {"prompt_tokens": 1, "total_tokens": 1}})
        if body.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            for line in (json.dumps({"id": "c", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
                                     "choices": [{"index": 0, "delta": {"content": "Lausanne."}, "finish_reason": None}]}), "[DONE]"):
                data = f"data: {line}\n\n".encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            return None
        return self._send(200, {"id": "a", "object": "chat.completion", "created": 0, "model": body["model"],
                                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Lausanne."}, "finish_reason": "stop"}]})

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[type[_Provider], list[float]]]:
    handler = type("Provider", (_Provider,), {"script": [(200, None)], "posts": []})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    monkeypatch.setattr(settings, "chat_ai_api_key", KEY)
    for name in ("chat_ai_base_url", "archai_chat_ai_base_url"):
        monkeypatch.setattr(settings, name, base)
    monkeypatch.setattr(settings, "rag_embedding_model", "multilingual-e5-large-instruct")
    monkeypatch.setattr(settings, "provider_retry_max_wait_seconds", 5.0)
    monkeypatch.setattr(rag_store, "_EMBED_CLIENTS", {})
    waits: list[float] = []
    monkeypatch.setattr(provider_retry.time, "sleep", waits.append)
    yield handler, waits
    server.shutdown()


def _chat() -> object:
    return chat_ai.create_chat_completion(messages=MESSAGES, model="qwen3.5-397b-a17b")["text"]


def _chat_stream() -> object:
    return [e for e in chat_ai.stream_chat_completion(messages=MESSAGES, model="qwen3.5-397b-a17b") if e["type"] == "answer"][0]["text"]


def _saia() -> object:
    return SaiaClient().chat_completion(model="qwen3.5-397b-a17b", messages=MESSAGES)["text"]


def _embeddings() -> object:
    return rag_store._embed(["a passage"])


PATHS = {"chat": _chat, "chat stream": _chat_stream, "saia": _saia, "embeddings": _embeddings}

# (scenario, the provider's answers, calls made, waits slept, whether it succeeds)
SCENARIOS = [
    ("success", [(200, None)], 1, [], True),
    ("429 with a short Retry-After, then success", [(429, "2"), (200, None)], 2, [2.0], True),
    ("429 with a short Retry-After, twice", [(429, "2")], 2, [2.0], False),
    ("429 with Retry-After: 0", [(429, "0"), (200, None)], 2, [0.0], True),
    ("429 with a long Retry-After", [(429, "30")], 1, [], False),
    ("429 with no Retry-After", [(429, None)], 1, [], False),
    ("500, then success", [(500, None), (200, None)], 2, [0.5], True),
    ("500 twice", [(500, None)], 2, [0.5], False),
    ("400", [(400, None)], 1, [], False),
]


@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize(("scenario", "script", "calls", "waits", "succeeds"), SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_calls_per_request(
    provider: tuple[type[_Provider], list[float]], path: str, scenario: str, script: list[tuple[int, str | None]],
    calls: int, waits: list[float], succeeds: bool,
) -> None:
    handler, slept = provider
    handler.script = script
    if succeeds:
        assert PATHS[path]()
    else:
        with pytest.raises(Exception):
            PATHS[path]()
    assert len(handler.posts) == calls, handler.posts
    assert slept == waits


def test_the_longest_wait_retried_is_a_setting(provider: tuple[type[_Provider], list[float]], monkeypatch: pytest.MonkeyPatch) -> None:
    handler, slept = provider
    handler.script = [(429, "8"), (200, None)]
    monkeypatch.setattr(settings, "provider_retry_max_wait_seconds", 10.0)
    assert _saia() == "Lausanne."
    assert (len(handler.posts), slept) == (2, [8.0])
    monkeypatch.setattr(settings, "provider_retry_max_wait_seconds", 0.0)
    handler.posts.clear()
    slept.clear()
    with pytest.raises(Exception):
        _saia()
    assert (len(handler.posts), slept) == (1, [])


def test_a_refused_connection_is_tried_once_more(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "chat_ai_api_key", KEY)
    monkeypatch.setattr(settings, "chat_ai_base_url", "http://127.0.0.1:9/v1")  # nothing listens there
    waits: list[float] = []
    monkeypatch.setattr(provider_retry.time, "sleep", waits.append)
    with pytest.raises(Exception):
        _saia()
    assert waits == [0.5]


def test_no_client_retries_by_itself() -> None:
    import openai

    clients: list[Any] = []
    real = openai.OpenAI

    class Recording(real):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            clients.append(kwargs)
            super().__init__(*args, **kwargs)

    settings_before = settings.chat_ai_api_key
    try:
        settings.chat_ai_api_key = KEY
        openai.OpenAI = Recording  # type: ignore[misc]
        chat_ai._create_client()
        SaiaClient()
        rag_store._embedding_client(Recording, KEY, "http://127.0.0.1:9/v1")
    finally:
        openai.OpenAI = real  # type: ignore[misc]
        settings.chat_ai_api_key = settings_before
    assert len(clients) == 3
    assert all(kwargs.get("max_retries") == 0 for kwargs in clients)


def test_the_thinking_switch_retry_is_outside_the_transient_one(provider: tuple[type[_Provider], list[float]]) -> None:
    """Worst case for one model: a 5xx, then the 400 that refuses the switch, then a 5xx twice without it: 4 calls."""
    handler, slept = provider
    original = _Provider.do_POST

    answers = iter([(500, None), (400, "chat_template"), (500, None), (500, None)])

    def scripted(self: _Provider) -> None:
        status, note = next(answers, (200, None))
        if status == 400:
            self.rfile.read(int(self.headers.get("content-length") or 0))
            type(self).posts.append(self.path)
            return self._send(400, {"error": {"message": "chat_template is not supported"}})
        type(self).script = [(status, note)]
        return original(self)

    handler.do_POST = scripted  # type: ignore[method-assign]
    with pytest.raises(Exception):
        _saia()
    assert len(handler.posts) == 4
    assert slept == [0.5, 0.5]


def test_no_provider_client_is_built_to_retry_or_with_a_transport() -> None:
    """Every OpenAI client the backend builds says max_retries=0, and none gets a transport (it would
    switch off HTTP_PROXY handling): checked in the source, so a client added later is checked too."""
    import ast
    from pathlib import Path

    app_dir = Path(__file__).resolve().parents[1] / "app"
    constructors: list[str] = []
    for path in app_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            keywords = {k.arg: k.value for k in node.keywords}
            assert "transport" not in keywords, f"{path.name}:{node.lineno} passes a transport"
            if name in {"OpenAI", "AsyncOpenAI", "openai_class"}:
                retries = keywords.get("max_retries")
                assert isinstance(retries, ast.Constant) and retries.value == 0, f"{path.name}:{node.lineno} {name}() without max_retries=0"
                constructors.append(f"{path.name}:{name}")
    assert sorted(constructors) == ["chat_ai.py:OpenAI", "rag_store.py:openai_class", "saia_client.py:OpenAI"]
