"""/api/health says whether the reading pipeline's models are installed.

A backend without layout weights or a line recogniser answered status "ok"
(or "degraded" only for Ollama or the chat key). Its "models_loaded" field
reported whether a worker process pool existed: true on Linux with no weights
at all, false on macOS always. The first page read then failed, or fell back
to GLM-OCR, and the workspace said nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.routers import health
from tests.test_health import SECRET, _Response

LAYOUT = ("best_emanuskript_segmentation.pt", "best_catmus.pt", "best_zone_detection.pt")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(settings, "chat_ai_api_key", SECRET)
    monkeypatch.setattr(settings, "glmocr_ollama_host", "http://127.0.0.1:11434")
    monkeypatch.setattr(health, "_last_ok", {})
    monkeypatch.setattr(health.requests, "get", lambda url, timeout: _Response({"version": "0.6.0"}))
    app = FastAPI()
    app.include_router(health.router, prefix="/api")
    return TestClient(app)


def _install(monkeypatch: pytest.MonkeyPatch, root: Path, *, layout: bool, recogniser: bool, packages: bool = True) -> None:
    from app.services import model_files

    models = root / "models"
    models.mkdir(exist_ok=True)
    if layout:
        for name in LAYOUT:
            (models / name).write_bytes(b"weights")
    recogniser_path = root / "catmus_medieval.mlmodel"
    if recogniser:
        recogniser_path.write_bytes(b"weights")
    monkeypatch.setattr(settings, "model_dir", str(models))
    monkeypatch.setattr(settings, "kraken_catmus_model_path", str(recogniser_path))
    monkeypatch.setattr(model_files, "_installed", lambda package: packages)


def _body(client: TestClient) -> dict[str, Any]:
    response = client.get("/api/health")
    assert response.status_code == 200
    return response.json()


def test_missing_models_make_the_backend_degraded_and_name_what_is_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch, tmp_path, layout=False, recogniser=False)
    body = _body(client)
    assert body["status"] == "degraded"
    assert body["models_loaded"] is False
    layout, recogniser = body["checks"]["layout_models"], body["checks"]["line_recognizer"]
    assert layout["ok"] is False and recogniser["ok"] is False
    assert "Pages cannot be read with the segmented engine" in layout["detail"]
    for name in LAYOUT:
        assert name in layout["detail"]
    assert "catmus_medieval.mlmodel" in recogniser["detail"]
    for check in (layout, recogniser):
        assert "docs/model_setup.md" in check["detail"]
        # Unauthenticated: file names and settings, never where the server keeps them.
        assert str(tmp_path) not in check["detail"]


def test_missing_packages_are_named(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install(monkeypatch, tmp_path, layout=True, recogniser=True, packages=False)
    body = _body(client)
    assert "ultralytics" in body["checks"]["layout_models"]["detail"]
    assert "kraken" in body["checks"]["line_recognizer"]["detail"]
    assert body["models_loaded"] is False


def test_everything_installed_is_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install(monkeypatch, tmp_path, layout=True, recogniser=True)
    body = _body(client)
    assert body["status"] == "ok"
    assert body["checks"]["layout_models"]["ok"] is True
    assert body["checks"]["line_recognizer"]["ok"] is True
    assert body["models_loaded"] is True


def test_a_missing_recogniser_alone_is_reported(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install(monkeypatch, tmp_path, layout=True, recogniser=False)
    body = _body(client)
    assert body["status"] == "degraded"
    assert body["checks"]["layout_models"]["ok"] is True
    assert body["checks"]["line_recognizer"]["ok"] is False
    assert body["models_loaded"] is False
