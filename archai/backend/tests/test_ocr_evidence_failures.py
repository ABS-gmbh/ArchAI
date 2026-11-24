"""A failed OCR evidence write is logged once per failure streak, and the OCR call still succeeds.

The OCR agent wrapped the evidence writer in ``except Exception: pass``: with
the log unwritable (a read-only volume, a full disk), every OCR call went on
unrecorded and nothing said so, although docs/deployment.md lists the log as
the record of every OCR call, to back up.
"""

from __future__ import annotations

import logging
import os
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.agents import ocr_agent
from app.config import settings
from app.services import ocr_evidence
from tests.test_kraken_line_crops import _DummyClient, _EchoBackend, _request

pytestmark = pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0, reason="needs POSIX permissions and a non-root user"
)


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The OCR agent with a stub recogniser, and the real evidence writer."""
    backend = _EchoBackend()
    monkeypatch.setattr(ocr_agent, "build_backend_runtime", lambda *_args, **_kwargs: {"kraken_catmus": backend})
    monkeypatch.setattr(ocr_evidence, "_failing", False, raising=False)


@pytest.fixture
def locked(tmp_path: Path) -> Iterator[Path]:
    path = tmp_path / "read-only"
    path.mkdir()
    path.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        yield path
    finally:
        path.chmod(stat.S_IRWXU)


def _read() -> str:
    return ocr_agent.run_ocr_extraction(_request([40, 40]), client=_DummyClient(), upscale_factor=2).text


def _evidence_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == "app.services.ocr_evidence"]


def test_a_failure_streak_is_logged_once_and_ocr_succeeds(
    agent: None, locked: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(settings, "ocr_evidence_path", str(locked / "evidence" / "ocr.jsonl"))
    with caplog.at_level(logging.INFO, logger="app.services.ocr_evidence"):
        assert _read() == "line l0\nline l1"
        assert _read() == "line l0\nline l1"
    failures = _evidence_records(caplog)
    assert [record.levelname for record in failures] == ["ERROR"]
    assert str(locked / "evidence" / "ocr.jsonl") in failures[0].getMessage()

    # The log becomes writable again: the next call is recorded, and the recovery said once.
    caplog.clear()
    writable = tmp_path / "evidence.jsonl"
    monkeypatch.setattr(settings, "ocr_evidence_path", str(writable))
    with caplog.at_level(logging.INFO, logger="app.services.ocr_evidence"):
        _read()
        _read()
    assert [record.levelname for record in _evidence_records(caplog)] == ["WARNING"]
    assert len(writable.read_text(encoding="utf-8").splitlines()) == 2

    # A new streak is reported again.
    caplog.clear()
    monkeypatch.setattr(settings, "ocr_evidence_path", str(locked / "again.jsonl"))
    with caplog.at_level(logging.INFO, logger="app.services.ocr_evidence"):
        _read()
    assert [record.levelname for record in _evidence_records(caplog)] == ["ERROR"]


def test_a_record_that_cannot_be_serialised_does_not_fail_the_call(
    agent: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(settings, "ocr_evidence_path", str(tmp_path / "evidence.jsonl"))
    def unserialisable(_record: object) -> dict:
        raise TypeError("not JSON")

    monkeypatch.setattr(ocr_evidence, "asdict", unserialisable)
    with caplog.at_level(logging.INFO, logger="app.services.ocr_evidence"):
        assert _read() == "line l0\nline l1"
    assert [record.levelname for record in _evidence_records(caplog)] == ["ERROR"]
    assert not (tmp_path / "evidence.jsonl").exists()
