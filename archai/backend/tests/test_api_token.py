"""Optional API token (API_TOKEN): off by default, and when set, required by every route but health."""

from __future__ import annotations

import logging
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app.main as main
from app.auth import require_api_token
from app.config import Settings, settings
from app.db import pipeline_db
from app.logging_setup import RedactTokenFilter

TOKEN = "token-for-tests-only-0123456789"
# Open on purpose: monitors need health; the docs routes have their own switch.
OPEN_PATHS = {"/api/health", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}


def _calls(dependant: Any) -> Iterator[Any]:
    yield dependant.call
    for sub in dependant.dependencies:
        yield from _calls(sub)


def _routes(routes: list[Any]) -> Iterator[tuple[str, Any]]:
    """(path, dependant) for every endpoint, through FastAPI's included-router wrappers."""
    for route in routes:
        if hasattr(route, "effective_route_contexts"):
            for context in route.effective_route_contexts():
                yield context.path_format, context.dependant
        elif hasattr(route, "dependant"):
            yield route.path, route.dependant
        elif hasattr(route, "path"):
            yield route.path, None


def test_every_route_but_health_and_docs_requires_the_token() -> None:
    unprotected = [
        path
        for path, dependant in _routes(main.app.routes)
        if path not in OPEN_PATHS
        and (dependant is None or require_api_token not in set(_calls(dependant)))
    ]
    seen = [path for path, _ in _routes(main.app.routes)]
    assert len(seen) > 30
    assert unprotected == []


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(settings, "api_token", TOKEN)
    return TestClient(main.app, raise_server_exceptions=False)


def test_without_a_token_setting_nothing_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_token", "")
    assert TestClient(main.app).get("/api/classes").status_code == 200


def test_a_missing_or_wrong_token_is_a_401(client: TestClient) -> None:
    for headers in ({}, {"Authorization": "Bearer wrong-token-wrong-token-wrong"}, {"Authorization": TOKEN}):
        response = client.get("/api/classes", headers=headers)
        assert response.status_code == 401
        assert response.headers["www-authenticate"].startswith("Bearer")
        assert response.json()["detail"].startswith("Authentication required")


@pytest.mark.parametrize("headers", [{"Authorization": f"Bearer {TOKEN}"}, {"X-API-Token": TOKEN}])
def test_the_right_token_passes(client: TestClient, headers: dict[str, str]) -> None:
    assert client.get("/api/classes", headers=headers).status_code == 200


def test_health_stays_open(client: TestClient) -> None:
    assert client.get("/api/health").status_code in (200, 503)


def test_the_query_token_works_only_on_the_two_streams(client: TestClient) -> None:
    run_id = pipeline_db.create_run(asset_ref="page")
    pipeline_db.update_run_fields(run_id, status="COMPLETED", current_stage="DONE")
    assert client.get(f"/api/ocr/trace/stream/{run_id}", params={"token": TOKEN}).status_code == 200
    assert client.get(f"/api/ocr/trace/stream/{run_id}").status_code == 401
    assert client.get("/api/predict/batch/abcdef123456/progress", params={"token": TOKEN}).status_code == 404
    assert client.get("/api/predict/batch/abcdef123456/progress").status_code == 401
    assert client.get(f"/api/ocr/trace/{run_id}", params={"token": TOKEN}).status_code == 401
    assert client.get("/api/classes", params={"token": TOKEN}).status_code == 401


def test_analytics_keeps_its_own_login(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both use Authorization: Bearer, so analytics clients send the API token as X-API-Token."""
    monkeypatch.setattr(settings, "analytics_username", "operator")
    monkeypatch.setattr(settings, "analytics_password", "a-password")
    monkeypatch.setattr(settings, "jwt_secret", "j" * 40)
    api = {"X-API-Token": TOKEN}
    assert client.post("/api/analytics/login", json={"username": "operator", "password": "a-password"}).status_code == 401
    login = client.post("/api/analytics/login", json={"username": "operator", "password": "a-password"}, headers=api)
    assert login.status_code == 200
    session = {**api, "Authorization": f"Bearer {login.json()['token']}"}
    assert client.get("/api/analytics/data", headers=api).status_code in (401, 403)  # no session
    assert client.get("/api/analytics/data", headers=session).status_code == 200


@pytest.mark.parametrize("value", ["short", "x" * 23])
def test_a_short_token_is_refused_by_name(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("API_TOKEN", value)
    with pytest.raises(ValidationError, match="API_TOKEN must be at least 24 characters"):
        Settings(_env_file=None)


def test_the_query_token_never_reaches_the_access_log() -> None:
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", "GET", f"/api/ocr/trace/stream/r1?token={TOKEN}", "1.1", 200), None,
    )
    RedactTokenFilter().filter(record)
    assert TOKEN not in record.getMessage() and "token=[redacted]" in record.getMessage()


def test_startup_warns_when_the_api_is_open(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import asyncio

    monkeypatch.setattr(settings, "api_token", "")
    monkeypatch.setattr(main, "init_model_pool", lambda: None)
    monkeypatch.setattr(main, "shutdown_model_pool", lambda: None)
    monkeypatch.setattr(main, "cleanup_expired_tasks", lambda: {})
    monkeypatch.setattr(main, "configure_logging", lambda *args, **kwargs: None)

    async def start_and_stop() -> None:
        async with main.lifespan(main.app):
            pass

    with caplog.at_level(logging.WARNING, logger="app.main"):
        asyncio.run(start_and_stop())
    assert "API_TOKEN is not set: the API is unauthenticated" in caplog.text
