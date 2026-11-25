"""A server that reaches the provider only through HTTP_PROXY / HTTPS_PROXY still reaches it.

The provider clients are built by the OpenAI SDK on httpx (or httpx2), which
reads the proxy from the environment only while no transport is passed to it
(`allow_env_proxies = trust_env and transport is None`). A retrying transport
passed to every client (2026-11-25, not merged) would have cut off a server on
a network that reaches the provider only through a proxy, with no setting to
get it back. Here a stand-in proxy on loopback receives the requests for a
provider at a name that resolves nowhere.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.config import settings
from app.services import chat_ai, rag_store
from app.services.saia_client import SaiaClient

KEY = "k" * 32
PROVIDER = "http://provider.invalid/v1"  # .invalid never resolves: only the proxy can answer for it


class _Proxy(BaseHTTPRequestHandler):
    """A forward proxy that answers for the provider itself, and notes each request line's target."""

    protocol_version = "HTTP/1.1"
    seen: list[str] = []

    def _answer(self) -> None:
        type(self).seen.append(self.path)
        length = int(self.headers.get("content-length") or 0)
        if length:
            self.rfile.read(length)
        if self.path.endswith("/models"):
            body: object = {"object": "list", "data": [{"id": "qwen3.5-397b-a17b", "object": "model"}]}
        elif self.path.endswith("/embeddings"):
            body = {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}], "model": "e5",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1}}
        else:
            body = {"id": "a", "object": "chat.completion", "created": 0, "model": "qwen3.5-397b-a17b",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Lausanne."}, "finish_reason": "stop"}]}
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = _answer
    do_POST = _answer

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def proxy(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_Proxy]]:
    handler = type("Proxy", (_Proxy,), {"seen": []})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    address = f"http://127.0.0.1:{server.server_address[1]}"
    for name in ("NO_PROXY", "no_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HTTP_PROXY", "http_proxy"):
        monkeypatch.setenv(name, address)
    monkeypatch.setattr(settings, "chat_ai_api_key", KEY)
    for name in ("chat_ai_base_url", "archai_chat_ai_base_url"):
        monkeypatch.setattr(settings, name, PROVIDER)
    monkeypatch.setattr(settings, "rag_embedding_model", "multilingual-e5-large-instruct")
    monkeypatch.setattr(rag_store, "_EMBED_CLIENTS", {})
    monkeypatch.setattr(SaiaClient, "_shared_models_cache", None)
    yield handler
    server.shutdown()


MESSAGES = [{"role": "user", "content": "Who is named?"}]


def test_chat_goes_through_the_proxy(proxy: type[_Proxy]) -> None:
    assert chat_ai.create_chat_completion(messages=MESSAGES, model="qwen3.5-397b-a17b")["text"].startswith("Lausanne.")
    assert proxy.seen == [f"{PROVIDER}/chat/completions"]


def test_the_model_list_goes_through_the_proxy(proxy: type[_Proxy]) -> None:
    assert chat_ai.list_available_models()["provider"] == "ok"
    assert proxy.seen == [f"{PROVIDER}/models"]


def test_label_analysis_and_ocr_calls_go_through_the_proxy(proxy: type[_Proxy]) -> None:
    assert SaiaClient().chat_completion(model="qwen3.5-397b-a17b", messages=MESSAGES)["text"] == "Lausanne."
    assert proxy.seen == [f"{PROVIDER}/chat/completions"]


def test_embeddings_go_through_the_proxy(proxy: type[_Proxy]) -> None:
    assert rag_store._embed(["a passage"]) == [[0.1]]
    assert proxy.seen == [f"{PROVIDER}/embeddings"]
