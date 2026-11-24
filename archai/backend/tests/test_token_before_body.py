"""With API_TOKEN set, a request without the token is refused before its body is read.

The token used to be a route dependency, which FastAPI resolves after it has
received and parsed the body: a 200 MB JSON body with no token was read whole
(the process grew from 194 to 793 MB) and answered 422, and a 200 MB multipart
upload was spooled to disk, then answered 401. Any client that could reach the
port could make the server read up to the upload limit (about 1.3 GB) per
request without knowing the token.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import app.main as main
from app.body_limit import max_request_bytes
from app.config import settings

TOKEN = "token-for-tests-only-0123456789"
MB = 1 << 20


def _drive(path: str, content_type: bytes, total: int, headers: list[tuple[bytes, bytes]] | None = None) -> tuple[int | None, int, bytes]:
    """POST *total* bytes to the app, a megabyte at a time; return (status, bytes the app read, body)."""
    consumed = 0
    status: int | None = None
    body = b""

    async def receive() -> dict[str, Any]:
        nonlocal consumed
        if consumed >= total:
            return {"type": "http.disconnect"}
        n = min(MB, total - consumed)
        consumed += n
        return {"type": "http.request", "body": b"a" * n, "more_body": consumed < total}

    async def send(message: dict[str, Any]) -> None:
        nonlocal status, body
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            body += message.get("body", b"")

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
        "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "", "scheme": "http",
        "server": ("testserver", 80), "client": ("203.0.113.9", 4000),
        "headers": [(b"content-type", content_type), (b"transfer-encoding", b"chunked"), *(headers or [])],
    }
    asyncio.run(main.app(scope, receive, send))
    return status, consumed, body


@pytest.fixture(autouse=True)
def _token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_token", TOKEN)


def test_a_json_body_without_the_token_is_not_read() -> None:
    status, consumed, body = _drive("/api/ocr/extract", b"application/json", 16 * MB)
    assert status == 401, body[:200]
    assert consumed == 0


def test_a_multipart_upload_without_the_token_is_not_read() -> None:
    status, consumed, _ = _drive("/api/predict/single", b"multipart/form-data; boundary=XBOUND", 16 * MB)
    assert status == 401
    assert consumed == 0


def test_a_wrong_token_is_refused_the_same_way() -> None:
    status, consumed, _ = _drive(
        "/api/ocr/extract", b"application/json", 4 * MB, headers=[(b"x-api-token", b"wrong-token-wrong-token-wrong")]
    )
    assert (status, consumed) == (401, 0)


def test_the_right_token_still_reaches_the_route() -> None:
    """A body with the token is read and parsed as before (here: not JSON, so 422)."""
    status, consumed, _ = _drive("/api/ocr/extract", b"application/json", 1 * MB, headers=[(b"x-api-token", TOKEN.encode())])
    assert status == 422
    assert consumed == 1 * MB


def test_health_and_cors_preflights_need_no_token() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(main.app)
    assert client.get("/api/health").status_code in (200, 503)
    preflight = client.options(
        "/api/ocr/extract",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"},
    )
    assert preflight.status_code == 200


def test_the_401_carries_the_challenge_and_cors_headers() -> None:
    from fastapi.testclient import TestClient

    response = TestClient(main.app).post("/api/ocr/extract", json={}, headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")
    assert response.json()["detail"].startswith("Authentication required")
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_json_bodies_have_a_much_smaller_limit_than_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.body_limit import max_json_bytes

    monkeypatch.setattr(settings, "max_json_megabytes", 2)
    assert max_json_bytes() == 2 * MB
    assert max_json_bytes() < max_request_bytes() // 10
    headers = [(b"x-api-token", TOKEN.encode())]
    status, consumed, body = _drive("/api/ocr/extract", b"application/json", 3 * MB, headers=headers)
    assert status == 413, body[:200]
    assert consumed <= 3 * MB
    assert b"larger than the server accepts (2 MB)" in body
