"""Temporary file and task state management."""

import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from threading import Lock
from typing import Any, Dict

from app.config import settings

_LOGGER = logging.getLogger(__name__)

TASK_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".tasks")


@dataclass
class TaskState:
    task_id: str
    status: str = "pending"  # pending | processing | completed | error
    progress: int = 0
    total: int = 0
    current_image: str = ""
    message: str = ""
    created_at: float = field(default_factory=time.time)
    # Result data
    coco_json: dict | None = None
    stats: dict | None = None
    annotated_image_path: str | None = None
    # Batch-specific
    batch_coco_list: list = field(default_factory=list)
    gallery: list = field(default_factory=list)  # list of {filename, annotated_path}
    stats_per_image: list = field(default_factory=list)
    stats_summary: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


_task_store: Dict[str, TaskState] = {}
_lock = Lock()


def create_task() -> TaskState:
    """Create a new task with a unique ID and working directory."""
    task_id = uuid.uuid4().hex[:12]
    task_dir = os.path.join(TASK_BASE_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    task = TaskState(task_id=task_id)
    with _lock:
        _task_store[task_id] = task
    return task


_TASK_ID_RE = re.compile(r"[0-9a-f]{12}")
_STATE_FILE = "state.json"


def get_task(task_id: str) -> TaskState | None:
    """The task in memory, or the state saved with its directory.

    The store is a dict in the worker process: a restart emptied it while the
    task directories stayed, and every download of a finished task answered
    404 though its files were on disk (audit TASK-02).
    """
    with _lock:
        task = _task_store.get(task_id)
        if task is not None:
            return task
    if not _TASK_ID_RE.fullmatch(str(task_id)):
        return None
    path = os.path.join(TASK_BASE_DIR, task_id, _STATE_FILE)
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        known = {f.name for f in fields(TaskState)}
        task = TaskState(**{key: value for key, value in data.items() if key in known})
    except (OSError, ValueError, TypeError):
        return None
    with _lock:
        return _task_store.setdefault(task_id, task)


def save_task_state(task: TaskState) -> None:
    """Write the task's state beside its files, so it outlives the process."""
    directory = get_task_dir(task.task_id)
    temporary = os.path.join(directory, f".{_STATE_FILE}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(asdict(task), handle, ensure_ascii=False)
        os.replace(temporary, os.path.join(directory, _STATE_FILE))
    except (OSError, TypeError, ValueError):
        _LOGGER.exception("Could not save the state of task %s", task.task_id)


def get_task_dir(task_id: str) -> str:
    path = os.path.join(TASK_BASE_DIR, task_id)
    os.makedirs(path, exist_ok=True)
    return path


def cleanup_expired_tasks() -> dict[str, int]:
    """Remove task state and working directories older than the TTL.

    Sweeps two populations, because they diverge:

    * entries in the in-process ``_task_store``
    * directories under TASK_BASE_DIR with no live entry at all

    The second is what actually accumulates. ``_task_store`` is a plain dict on
    the worker process, so every restart empties it while the directories stay on
    disk - and a sweep that only walked the dict could never see them again.
    Measured on a real deployment before this change: 462 directories totalling
    3.0 GB, 460 of them past a 60-minute TTL, the oldest 218 days old.

    Returns a count of what was removed, so callers can log it.
    """
    cutoff = time.time() - max(1, int(settings.task_ttl_minutes)) * 60
    removed_tracked = 0
    removed_orphans = 0

    with _lock:
        expired = [tid for tid, t in _task_store.items() if t.created_at < cutoff]
        for tid in expired:
            del _task_store[tid]
            shutil.rmtree(os.path.join(TASK_BASE_DIR, tid), ignore_errors=True)
            removed_tracked += 1
        live_ids = set(_task_store)

    # Orphan sweep, outside the lock: directory mtime is the only evidence left
    # once the in-memory entry is gone.
    try:
        entries = os.listdir(TASK_BASE_DIR)
    except OSError:
        entries = []

    for name in entries:
        if name in live_ids:
            continue
        path = os.path.join(TASK_BASE_DIR, name)
        if not os.path.isdir(path):
            continue
        try:
            if os.path.getmtime(path) >= cutoff:
                continue
        except OSError:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed_orphans += 1

    if removed_tracked or removed_orphans:
        _LOGGER.info(
            "Task cleanup removed %d tracked and %d orphaned task directories.",
            removed_tracked,
            removed_orphans,
        )
    return {"tracked": removed_tracked, "orphaned": removed_orphans}
