"""The backend configures logging, so its INFO records reach the output.

Nothing configured logging before: under uvicorn the app's loggers had no
handler, Python's last-resort handler printed WARNING and above as bare text,
and INFO records were dropped.
"""

from __future__ import annotations

import io
import json
import logging

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.logging_setup import configure_logging


@pytest.fixture
def restore_root():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def test_info_records_from_app_loggers_are_written(restore_root: None) -> None:
    stream = io.StringIO()
    configure_logging("INFO", "text", stream=stream)
    logging.getLogger("app.services.rag_store").info("Indexed run %s", "r1")
    assert "INFO app.services.rag_store: Indexed run r1" in stream.getvalue()


def test_json_lines_keep_extras(restore_root: None) -> None:
    stream = io.StringIO()
    configure_logging("INFO", "json", stream=stream)
    logging.getLogger("archai.ocr").warning("stage failed", extra={"run_id": "r1"})
    record = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert (record["level"], record["logger"], record["message"], record["run_id"]) == (
        "WARNING",
        "archai.ocr",
        "stage failed",
        "r1",
    )


def test_configuring_twice_installs_one_handler(restore_root: None) -> None:
    first = configure_logging("INFO", "text", stream=io.StringIO())
    second = configure_logging("DEBUG", "json", stream=io.StringIO())
    root = logging.getLogger()
    assert first not in root.handlers and second in root.handlers
    assert root.level == logging.DEBUG


@pytest.mark.parametrize(("name", "value"), [("LOG_LEVEL", "LOUD"), ("LOG_FORMAT", "xml")])
def test_bad_settings_are_refused_by_name(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError, match=f"{name} must be"):
        Settings(_env_file=None)


def test_settings_are_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("LOG_FORMAT", "JSON")
    settings = Settings(_env_file=None)
    assert (settings.log_level, settings.log_format) == ("DEBUG", "json")


def test_the_model_pool_logs_instead_of_printing() -> None:
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app" / "services" / "inference.py").read_text()
    assert "print(" not in source
