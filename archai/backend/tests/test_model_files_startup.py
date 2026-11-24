"""Startup says which model files and packages are missing, naming the setting for each.

A backend installed and configured as the READMEs say starts without a word
about models. The first page then fails deep in a request: /api/predict/single
answers 500 "Layout segmentation failed on the server", and only the server
log's traceback names models/best_emanuskript_segmentation.pt. A page read
falls back to GLM-OCR without saying why.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

import app.main as main
from app.config import settings

LOGGER = "app.services.model_files"


def _quiet_lifespan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "init_model_pool", lambda: None)
    monkeypatch.setattr(main, "shutdown_model_pool", lambda: None)
    monkeypatch.setattr(main, "cleanup_expired_tasks", lambda: {})
    monkeypatch.setattr(main, "configure_logging", lambda *args, **kwargs: None)


def _start(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> list[str]:
    _quiet_lifespan(monkeypatch)

    async def run() -> None:
        async with main.lifespan(main.app):
            pass

    with caplog.at_level(logging.INFO, logger=LOGGER):
        asyncio.run(run())
    return [record.getMessage() for record in caplog.records if record.name == LOGGER and record.levelno >= logging.WARNING]


def test_missing_layout_and_recognition_models_are_named(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(settings, "model_dir", str(tmp_path / "models"))
    monkeypatch.setattr(settings, "kraken_catmus_model_path", str(tmp_path / "weights" / "catmus_medieval.mlmodel"))
    warnings = _start(monkeypatch, caplog)
    assert len(warnings) == 1
    message = warnings[0]
    for name in ("best_emanuskript_segmentation.pt", "best_catmus.pt", "best_zone_detection.pt", "MODEL_DIR"):
        assert name in message
    assert "catmus_medieval.mlmodel" in message and "KRAKEN_CATMUS_MODEL_PATH" in message
    assert "docs/model_setup.md" in message


def test_nothing_is_said_when_everything_is_there(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from app.services import model_files

    models = tmp_path / "models"
    models.mkdir()
    for name in ("best_emanuskript_segmentation.pt", "best_catmus.pt", "best_zone_detection.pt"):
        (models / name).write_bytes(b"weights")
    recogniser = tmp_path / "catmus_medieval.mlmodel"
    recogniser.write_bytes(b"weights")
    monkeypatch.setattr(settings, "model_dir", str(models))
    monkeypatch.setattr(settings, "kraken_catmus_model_path", str(recogniser))
    monkeypatch.setattr(model_files, "_installed", lambda package: True)
    assert _start(monkeypatch, caplog) == []


def test_missing_packages_are_named(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import model_files

    monkeypatch.setattr(model_files, "_installed", lambda package: package != "kraken")
    assert any("kraken" in item and "package" in item for item in model_files.missing_models())
