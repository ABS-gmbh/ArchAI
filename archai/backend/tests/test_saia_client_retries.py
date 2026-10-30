"""The SAIA client retries transient gateway errors itself (audit AG-14).

The audit read the OCR agents' own error handling and concluded that one 429
or 502 from the SAIA gateway fails a page. The client underneath is the
OpenAI SDK, which by default retries 408, 409, 429 and 5xx responses (and
connection errors) twice with backoff, honouring Retry-After. These tests run
the real SaiaClient against a local server to pin that behaviour, so the
retries are not lost by someone setting max_retries=0.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.services import saia_client
from app.services.saia_client import SaiaClient

REPLY = {
    "id": "x", "object": "chat.completion", "created": 0, "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
}


def _server(statuses: list[int]) -> tuple[ThreadingHTTPServer, list[int]]:
    seen: list[int] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            status = statuses[min(len(seen), len(statuses) - 1)]
            seen.append(status)
            body = json.dumps(REPLY if status == 200 else {"error": {"message": "gateway"}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def _client(monkeypatch, server: ThreadingHTTPServer) -> SaiaClient:
    monkeypatch.setattr(saia_client.settings, "chat_ai_api_key", "test-key")
    monkeypatch.setattr(saia_client.settings, "chat_ai_base_url", f"http://127.0.0.1:{server.server_port}/v1")
    return SaiaClient()


@pytest.mark.parametrize("status", [429, 502, 503])
def test_a_transient_error_is_retried(monkeypatch, status):
    server, seen = _server([status, 200])
    try:
        result = _client(monkeypatch, server).chat_completion(model="m", messages=[{"role": "user", "content": "hi"}])
    finally:
        server.shutdown()
    assert seen == [status, 200]
    assert result["text"] == "ok"


def test_retries_stop_after_two(monkeypatch):
    server, seen = _server([502])
    try:
        with pytest.raises(Exception):
            _client(monkeypatch, server).chat_completion(model="m", messages=[{"role": "user", "content": "hi"}])
    finally:
        server.shutdown()
    assert seen == [502, 502, 502]


def test_a_client_error_is_not_retried(monkeypatch):
    server, seen = _server([400])
    try:
        with pytest.raises(Exception):
            _client(monkeypatch, server).chat_completion(model="m", messages=[{"role": "user", "content": "hi"}])
    finally:
        server.shutdown()
    assert seen == [400]
