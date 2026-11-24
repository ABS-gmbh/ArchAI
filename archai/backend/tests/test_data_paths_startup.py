"""Startup refuses a data path it cannot write, naming the setting, and logs the paths in use.

Only TASK_BASE_DIR was checked. With ARCHAI_DB_PATH in a read-only directory
the server stopped on a bare PermissionError traceback that named neither the
setting nor what to do. With CHROMA_PERSIST_DIR, ARCHAI_WIKIDATA_CACHE or
OCR_EVIDENCE_PATH there it started and answered /api/health with 200; indexing
and linking then failed on first use, and the OCR evidence log, which the
backup table calls the record of every OCR call, silently stopped being
written (its writer swallows the error). Nothing in the startup log said which
paths the process was using.
"""

from __future__ import annotations

import asyncio
import logging
import os
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest

import app.main as main
from app.config import settings

pytestmark = pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0, reason="needs POSIX permissions and a non-root user"
)


@pytest.fixture
def locked(tmp_path: Path) -> Iterator[Path]:
    """A directory this process can read but not write."""
    path = tmp_path / "read-only"
    path.mkdir()
    path.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        yield path
    finally:
        path.chmod(stat.S_IRWXU)


def _quiet_lifespan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "init_model_pool", lambda: None)
    monkeypatch.setattr(main, "shutdown_model_pool", lambda: None)
    monkeypatch.setattr(main, "cleanup_expired_tasks", lambda: {})
    monkeypatch.setattr(main, "configure_logging", lambda *args, **kwargs: None)


def _start_and_stop() -> None:
    async def run() -> None:
        async with main.lifespan(main.app):
            pass

    asyncio.run(run())


def _point(monkeypatch: pytest.MonkeyPatch, variable: str, value: Path) -> None:
    if variable in ("ARCHAI_DB_PATH", "ARCHAI_WIKIDATA_CACHE"):
        monkeypatch.setenv(variable, str(value))
    else:
        monkeypatch.setattr(settings, variable.lower(), str(value))


@pytest.mark.parametrize(
    ("variable", "name"),
    [
        ("ARCHAI_DB_PATH", "archai.sqlite"),
        ("CHROMA_PERSIST_DIR", "chroma"),
        ("ARCHAI_WIKIDATA_CACHE", "wikidata_cache.sqlite"),
        ("OCR_EVIDENCE_PATH", "ocr_evidence.jsonl"),
    ],
)
def test_startup_refuses_an_unwritable_path_by_name(
    monkeypatch: pytest.MonkeyPatch, locked: Path, variable: str, name: str
) -> None:
    _quiet_lifespan(monkeypatch)
    _point(monkeypatch, variable, locked / name)
    with pytest.raises(RuntimeError, match=f"{variable}=.*cannot write"):
        _start_and_stop()


@pytest.mark.parametrize("variable", ["ARCHAI_DB_PATH", "ARCHAI_WIKIDATA_CACHE", "OCR_EVIDENCE_PATH"])
def test_an_existing_read_only_file_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, variable: str) -> None:
    target = tmp_path / "data" / "file"
    target.parent.mkdir()
    target.write_text("")
    target.chmod(stat.S_IRUSR)
    _point(monkeypatch, variable, target)
    try:
        from app.services import data_paths

        with pytest.raises(RuntimeError, match=f"{variable}=.*cannot write"):
            data_paths.check_data_paths()
    finally:
        target.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_missing_paths_under_a_writable_directory_pass_and_nothing_is_created(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for variable, name in (
        ("ARCHAI_DB_PATH", "db/archai.sqlite"),
        ("CHROMA_PERSIST_DIR", "chroma/store"),
        ("ARCHAI_WIKIDATA_CACHE", "cache/wikidata.sqlite"),
        ("OCR_EVIDENCE_PATH", "evidence/ocr.jsonl"),
    ):
        _point(monkeypatch, variable, tmp_path / name)
    from app.services import data_paths

    data_paths.check_data_paths()
    assert list(tmp_path.iterdir()) == []


def test_startup_logs_the_paths_in_use(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog) -> None:
    _quiet_lifespan(monkeypatch)
    _point(monkeypatch, "CHROMA_PERSIST_DIR", tmp_path / "chroma")
    _point(monkeypatch, "OCR_EVIDENCE_PATH", tmp_path / "evidence.jsonl")
    with caplog.at_level(logging.INFO, logger="app.services.data_paths"):
        _start_and_stop()
    lines = [record.getMessage() for record in caplog.records if record.name == "app.services.data_paths"]
    assert len(lines) == 1
    for variable in ("ARCHAI_DB_PATH", "CHROMA_PERSIST_DIR", "ARCHAI_WIKIDATA_CACHE", "OCR_EVIDENCE_PATH", "TASK_BASE_DIR"):
        assert f"{variable}=" in lines[0]
    assert str(tmp_path / "chroma") in lines[0]
