"""A second backend process on the same data directory is reported, not refused.

One worker process is supported (docs/deployment.md). The lifespan takes an
advisory lock beside the pipeline database; a second holder logs an error.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from app import process_guard
from app.process_guard import ProcessGuard, guard_for_database

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_a_second_guard_on_the_same_directory_reports_the_conflict(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    first = guard_for_database(tmp_path / "archai.sqlite")
    second = guard_for_database(tmp_path / "archai.sqlite")
    assert first.acquire()
    with caplog.at_level(logging.ERROR, logger="app.process_guard"):
        assert second.acquire() is False
    message = caplog.text
    assert str(tmp_path) in message
    assert "Only one worker process is supported" in message and "docs/deployment.md" in message
    first.release()


def test_releasing_the_first_lets_a_third_succeed(tmp_path: Path) -> None:
    first = ProcessGuard(tmp_path / "archai.sqlite.lock")
    assert first.acquire()
    assert ProcessGuard(tmp_path / "archai.sqlite.lock").acquire() is False
    first.release()
    third = ProcessGuard(tmp_path / "archai.sqlite.lock")
    assert third.acquire()
    third.release()


def test_another_database_has_its_own_lock(tmp_path: Path) -> None:
    first = guard_for_database(tmp_path / "a" / "archai.sqlite")
    other = guard_for_database(tmp_path / "b" / "archai.sqlite")
    assert first.acquire() and other.acquire()
    first.release()
    other.release()


def test_without_fcntl_the_guard_does_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_guard, "fcntl", None)
    assert ProcessGuard(tmp_path / "x.lock").acquire()
    assert not (tmp_path / "x.lock").exists()


def test_the_lifespan_holds_the_lock_and_releases_it(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    import app.main as main
    from app.db import pipeline_db

    monkeypatch.setattr(main, "init_model_pool", lambda: None)
    monkeypatch.setattr(main, "shutdown_model_pool", lambda: None)
    monkeypatch.setattr(main, "cleanup_expired_tasks", lambda: {})
    monkeypatch.setattr(main, "configure_logging", lambda *args, **kwargs: None)
    database = pipeline_db._db_config().path
    held: list[bool] = []

    async def start_and_stop() -> None:
        async with main.lifespan(main.app):
            probe = guard_for_database(database)
            held.append(not probe.acquire())

    asyncio.run(start_and_stop())
    after = guard_for_database(database)
    assert held == [True]
    assert after.acquire()
    after.release()


def test_the_lock_file_is_ignored_by_git() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "archai/backend/app/archai.sqlite.lock"], cwd=REPO_ROOT, check=False
    )
    assert result.returncode == 0
