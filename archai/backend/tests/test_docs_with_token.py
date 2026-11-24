"""With API_TOKEN set, nothing but /api/health answers without it unless an operator asks.

Found by a security review: with API_TOKEN set and no token, /docs, /redoc,
/openapi.json, /docs/oauth2-redirect and /static/trace.html answered 200. The
docs describe every route to anyone; trace.html was a developer page nothing
linked to.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator

import app.main as main
import pytest
from app.config import Settings, settings
from fastapi.testclient import TestClient

TOKEN = "token-for-tests-only-0123456789"
DOCS = ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect")


@pytest.fixture
def rebuilt_app(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The docs routes are fixed when the app is built: rebuild it, and restore it afterwards."""
    yield
    monkeypatch.undo()
    importlib.reload(main)


@pytest.mark.usefixtures("rebuilt_app")
def test_with_a_token_the_docs_are_off_unless_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_token", TOKEN)
    monkeypatch.setattr(settings, "enable_api_docs", None)
    importlib.reload(main)
    client = TestClient(main.app)
    assert [client.get(path).status_code for path in DOCS] == [401, 401, 401, 401]
    with_token = [client.get(path, headers={"X-API-Token": TOKEN}).status_code for path in DOCS]
    assert with_token == [404, 404, 404, 404]


@pytest.mark.usefixtures("rebuilt_app")
def test_an_operator_can_still_switch_them_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_token", TOKEN)
    monkeypatch.setattr(settings, "enable_api_docs", True)
    importlib.reload(main)
    client = TestClient(main.app)
    assert [client.get(path).status_code for path in ("/docs", "/redoc", "/openapi.json")] == [200, 200, 200]


def test_without_a_token_they_stay_on_by_default() -> None:
    from app.config import api_docs_enabled

    assert api_docs_enabled(Settings(_env_file=None, api_token="", enable_api_docs=None)) is True
    assert api_docs_enabled(Settings(_env_file=None, api_token=TOKEN, enable_api_docs=None)) is False
    assert api_docs_enabled(Settings(_env_file=None, api_token=TOKEN, enable_api_docs=True)) is True
    assert api_docs_enabled(Settings(_env_file=None, api_token="", enable_api_docs=False)) is False


def test_the_unlinked_trace_page_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_token", TOKEN)
    client = TestClient(main.app)
    assert client.get("/static/trace.html").status_code == 401
    assert client.get("/static/trace.html", headers={"X-API-Token": TOKEN}).status_code == 404
