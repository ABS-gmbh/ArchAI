"""A trailing slash is a 404, not a redirect that names the backend's address.

Second security review (2026-11-24): `GET /api/classes/` answered 307 with
`location: http://<backend host>:8010/api/classes`, built from the Host the
frontend proxy sends, and the proxy passed it to the browser. No route is
declared with a trailing slash and nothing calls one, so redirect_slashes is off.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings

TOKEN = "token-for-tests-only-0123456789"


@pytest.mark.parametrize(
    ("method", "path"),
    [("GET", "/api/classes/"), ("POST", "/api/ocr/runs/"), ("GET", "/api/health/"), ("GET", "/api/ocr/trace/abc/")],
)
def test_a_trailing_slash_is_not_redirected(monkeypatch: pytest.MonkeyPatch, method: str, path: str) -> None:
    monkeypatch.setattr(settings, "api_token", TOKEN)
    client = TestClient(main.app, base_url="http://backend.internal:8010", follow_redirects=False)
    response = client.request(method, path, headers={"X-API-Token": TOKEN})
    assert response.status_code == 404
    assert "location" not in response.headers
    assert "backend.internal" not in response.text


def test_the_routes_without_the_slash_still_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_token", TOKEN)
    client = TestClient(main.app)
    assert client.get("/api/classes", headers={"X-API-Token": TOKEN}).status_code == 200
    assert client.get("/api/health").status_code in (200, 503)
