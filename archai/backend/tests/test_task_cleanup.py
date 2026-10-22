"""Tests for reclaiming expired task working directories.

cleanup_expired_tasks existed but had no callers, so .tasks/ grew without bound.
Measured on a real deployment: 462 directories totalling 3.0 GB, 460 of them past
a 60-minute TTL, the oldest 218 days old. That disk pressure is what caused macOS
to evict this repository's git pack to iCloud.

The function also could not have fixed it if called. It walked _task_store, a
plain dict on the worker process, so every restart emptied the dict while the
directories stayed on disk - leaving them permanently invisible to the sweep.
"""

from __future__ import annotations

import os
import time

import pytest

import app.services.file_manager as file_manager


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point the task store at a temp dir; never touch a real .tasks."""
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path))
    file_manager._task_store.clear()
    yield tmp_path
    file_manager._task_store.clear()


def make_dir(base, name: str, age_days: float) -> str:
    path = os.path.join(str(base), name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "blob.bin"), "wb") as handle:
        handle.write(b"x" * 1024)
    stamp = time.time() - age_days * 86400
    os.utime(path, (stamp, stamp))
    return path


def test_orphaned_directories_are_reclaimed(sandbox) -> None:
    """The population that actually accumulates: dirs with no in-memory entry."""
    for i in range(5):
        make_dir(sandbox, f"orphan-{i}", 30)
    result = file_manager.cleanup_expired_tasks()
    assert result["orphaned"] == 5
    assert os.listdir(sandbox) == []


def test_fresh_orphans_are_kept(sandbox) -> None:
    """A directory inside the TTL may belong to another live worker."""
    make_dir(sandbox, "recent", 0.0)
    file_manager.cleanup_expired_tasks()
    assert "recent" in os.listdir(sandbox)


def test_a_live_task_survives(sandbox) -> None:
    task = file_manager.create_task()
    file_manager.cleanup_expired_tasks()
    assert task.task_id in os.listdir(sandbox)
    assert task.task_id in file_manager._task_store


def test_tracked_expired_tasks_are_removed(sandbox) -> None:
    task = file_manager.create_task()
    task.created_at = time.time() - 10 * 86400
    result = file_manager.cleanup_expired_tasks()
    assert result["tracked"] == 1
    assert task.task_id not in file_manager._task_store
    assert task.task_id not in os.listdir(sandbox)


def test_tracked_and_orphaned_are_counted_separately(sandbox) -> None:
    task = file_manager.create_task()
    task.created_at = time.time() - 10 * 86400
    for i in range(3):
        make_dir(sandbox, f"orphan-{i}", 30)
    result = file_manager.cleanup_expired_tasks()
    assert result == {"tracked": 1, "orphaned": 3}


def test_a_missing_base_directory_is_not_an_error(sandbox, monkeypatch) -> None:
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(sandbox / "absent"))
    assert file_manager.cleanup_expired_tasks() == {"tracked": 0, "orphaned": 0}


def test_stray_files_are_left_alone(sandbox) -> None:
    """Only directories are task workspaces."""
    stray = os.path.join(str(sandbox), "notes.txt")
    with open(stray, "w", encoding="utf-8") as handle:
        handle.write("x")
    os.utime(stray, (time.time() - 30 * 86400,) * 2)
    file_manager.cleanup_expired_tasks()
    assert os.path.exists(stray)


def test_cleanup_is_idempotent(sandbox) -> None:
    for i in range(3):
        make_dir(sandbox, f"orphan-{i}", 30)
    assert file_manager.cleanup_expired_tasks()["orphaned"] == 3
    assert file_manager.cleanup_expired_tasks() == {"tracked": 0, "orphaned": 0}


def test_ttl_is_honoured(sandbox, monkeypatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "task_ttl_minutes", 60 * 24 * 40)  # 40 days
    make_dir(sandbox, "thirty-days", 30)
    file_manager.cleanup_expired_tasks()
    assert "thirty-days" in os.listdir(sandbox), "inside a widened TTL it must survive"


def test_the_cleanup_is_actually_wired_into_startup() -> None:
    """A function nobody calls fixes nothing - that was the original defect."""
    import inspect

    import app.main as main

    source = inspect.getsource(main)
    assert "cleanup_expired_tasks()" in source
    assert "_periodic_task_cleanup" in source
