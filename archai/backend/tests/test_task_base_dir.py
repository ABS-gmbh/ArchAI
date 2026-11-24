"""The layout task directory is a setting, TASK_BASE_DIR, checked at startup.

It was fixed to archai/backend/.tasks, which a server could not move to its
data volume.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from app.config import Settings
from app.services import file_manager


def test_it_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TASK_BASE_DIR", str(tmp_path / "tasks"))
    assert Settings(_env_file=None).task_base_dir == str(tmp_path / "tasks")


def test_the_default_is_unchanged() -> None:
    default = Settings.model_fields["task_base_dir"].default
    assert Path(default) == Path(__file__).resolve().parents[1] / ".tasks"


def test_tasks_go_where_it_points(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path / "elsewhere"))
    task = file_manager.create_task()
    assert (tmp_path / "elsewhere" / task.task_id).is_dir()


def test_a_missing_directory_is_created(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path / "new" / "tasks"))
    assert file_manager.check_task_base_dir() == str(tmp_path / "new" / "tasks")
    assert (tmp_path / "new" / "tasks").is_dir()


def test_a_file_is_refused_by_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "tasks").write_text("not a directory")
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path / "tasks"))
    with pytest.raises(RuntimeError, match="TASK_BASE_DIR="):
        file_manager.check_task_base_dir()


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs POSIX permissions and a non-root user")
def test_a_read_only_directory_is_refused_by_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    locked = tmp_path / "tasks"
    locked.mkdir()
    locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(locked))
    try:
        with pytest.raises(RuntimeError, match="TASK_BASE_DIR=.*not writable"):
            file_manager.check_task_base_dir()
    finally:
        locked.chmod(stat.S_IRWXU)
