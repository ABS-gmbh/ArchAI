"""Temporary file and task state management."""

import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
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


def get_task(task_id: str) -> TaskState | None:
    with _lock:
        return _task_store.get(task_id)


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
