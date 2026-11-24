"""An exception no route handles is answered in JSON, through the middlewares.

Second security review (2026-11-24): it was answered by Starlette's outermost
ServerErrorMiddleware, outside every middleware of the app, as plain text
"Internal Server Error" without X-Content-Type-Options: nosniff or the CORS
headers a browser needs to read it.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.error_handlers import UNEXPECTED_ERROR_DETAIL
from app.routers import classes


class _Exploding:
    def items(self):
        raise RuntimeError("secret /path/in/exception")


@pytest.fixture
def exploding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(classes, "coco_class_mapping", _Exploding())


def test_an_unhandled_exception_is_a_json_500_with_the_app_headers(exploding: None, caplog: pytest.LogCaptureFixture) -> None:
    client = TestClient(main.app, raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR):
        response = client.get("/api/classes", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 500
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"detail": UNEXPECTED_ERROR_DETAIL}
    assert "secret" not in response.text
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    # The details go to the log, with the traceback.
    assert any(record.exc_info and "secret /path/in/exception" in str(record.exc_info[1]) for record in caplog.records)


def test_the_exception_does_not_escape_to_the_server(exploding: None) -> None:
    client = TestClient(main.app, raise_server_exceptions=True)
    assert client.get("/api/classes").status_code == 500
