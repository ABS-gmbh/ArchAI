"""ENABLE_API_DOCS switches off /docs, /redoc and /openapi.json; on by default while there is no API_TOKEN."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import Settings, settings


def test_docs_are_on_by_default() -> None:
    # Unset: on without a token (the tests run without one), off with one (test_docs_with_token.py).
    assert Settings.model_fields["enable_api_docs"].default is None
    client = TestClient(main.app)
    assert [client.get(path).status_code for path in ("/docs", "/redoc", "/openapi.json")] == [200, 200, 200]


def test_switched_off_they_are_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "enable_api_docs", False)
    try:
        importlib.reload(main)
        client = TestClient(main.app)
        assert [client.get(path).status_code for path in ("/docs", "/redoc", "/openapi.json")] == [404, 404, 404]
        assert client.get("/api/health").status_code in (200, 503)
    finally:
        monkeypatch.setattr(settings, "enable_api_docs", None)
        importlib.reload(main)


def test_it_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_API_DOCS", "false")
    assert Settings(_env_file=None).enable_api_docs is False
