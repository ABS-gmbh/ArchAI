"""/api/health reports the dependencies, not just that the process is up.

It used to return {"status": "ok"} unconditionally, so a backend whose database
was unusable or whose Ollama host was down still looked healthy.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.routers import health

SECRET = "chat-key-for-tests-only"


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    for name in ("chat_ai_api_key", "saia_api_key", "archai_chat_ai_api_key", "archai_saia_api_key"):
        monkeypatch.setattr(settings, name, "")
    for name in ("CHAT_AI_API_KEY", "SAIA_API_KEY", "ARCHAI_CHAT_AI_API_KEY", "ARCHAI_SAIA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(settings, "glmocr_ollama_host", "http://127.0.0.1:11434")
    monkeypatch.setattr(health.requests, "get", lambda url, timeout: _Response({"version": "0.6.0"}))
    app = FastAPI()
    app.include_router(health.router, prefix="/api")
    return TestClient(app)


def test_all_dependencies_up_is_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "chat_ai_api_key", SECRET)
    response = client.get("/api/health")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert {name: check["ok"] for name, check in body["checks"].items()} == {
        "database": True,
        "ollama": True,
        "chat_api_key": True,
    }
    assert body["checks"]["ollama"]["version"] == "0.6.0"
    assert "models_loaded" in body


def test_the_key_itself_is_never_returned(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "chat_ai_api_key", SECRET)
    assert SECRET not in client.get("/api/health").text


def test_an_unreachable_ollama_degrades_but_serves(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(url: str, timeout: float) -> Any:
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(health.requests, "get", refuse)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["checks"]["ollama"]["ok"] is False


def test_a_missing_key_degrades(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["checks"]["chat_api_key"]["ok"] is False


def test_an_unusable_database_is_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> Any:
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(health.pipeline_db, "_connect", broken)
    response = client.get("/api/health")
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert "unable to open database file" in response.json()["checks"]["database"]["detail"]
