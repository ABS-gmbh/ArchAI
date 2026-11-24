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

# From TASK_BASE_DIR (settings.task_base_dir); default archai/backend/.tasks.
# A module attribute, read at each use, so tests can point it elsewhere.
TASK_BASE_DIR = settings.task_base_dir


def check_task_base_dir() -> str:
    """Create the task directory if missing and make sure it is writable; called at startup.

    Raises RuntimeError naming TASK_BASE_DIR, so a wrong path stops the server at
    startup instead of failing the first upload.
    """
    path = TASK_BASE_DIR
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"TASK_BASE_DIR={path!r} cannot be created: {exc}") from exc
    if not os.path.isdir(path):
        raise RuntimeError(f"TASK_BASE_DIR={path!r} is not a directory")
    if not os.access(path, os.W_OK | os.X_OK):
        raise RuntimeError(f"TASK_BASE_DIR={path!r} is not writable by this process")
    return path


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
# Tasks with work queued or running on them (a batch from submission until it
# ends). The TTL sweep leaves them alone: it deleted a batch that had waited
# 61 minutes in the queue, upload and all, and the batch then failed with
# FileNotFoundError (second security review, D).
_held: set[str] = set()


def hold_task(task_id: str, *, limit: int | None = None) -> bool:
    """Mark *task_id* as having work queued or running; False, and nothing held, when *limit* tasks already are."""
    with _lock:
        if limit is not None and len(_held) >= limit:
            return False
        _held.add(task_id)
        return True


def release_task(task_id: str) -> None:
    """The work on *task_id* has ended (or was never started)."""
    with _lock:
        _held.discard(task_id)


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
        _rebase_paths(task)
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    with _lock:
        return _task_store.setdefault(task_id, task)


def _rebased(task_id: str, path: str | None) -> str | None:
    """A saved file path, moved under this TASK_BASE_DIR if it named the task's directory elsewhere.

    state.json stores absolute paths. When TASK_BASE_DIR changes and the task
    directories are moved there, the saved paths still name the old place, and
    the downloads, which serve only files inside the task's own directory,
    answered 404. A path without the task's directory in it is left as it is
    (and the downloads still refuse it).
    """
    if not path:
        return path
    marker = f"{os.sep}{task_id}{os.sep}"
    if marker not in path:
        return path
    return os.path.join(TASK_BASE_DIR, task_id, path.rsplit(marker, 1)[1])


def _rebase_paths(task: TaskState) -> None:
    task.annotated_image_path = _rebased(task.task_id, task.annotated_image_path)
    for entry in task.gallery:
        if isinstance(entry, dict) and isinstance(entry.get("annotated_path"), str):
            entry["annotated_path"] = _rebased(task.task_id, entry["annotated_path"])


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


def discard_task(task_id: str) -> None:
    """Forget a task and remove its directory, for a request refused before it produced anything.

    A refused upload (empty, not an image, over the limit) left its task in the
    store and its directory, with the uploaded bytes, on disk until the TTL
    sweep.
    """
    with _lock:
        _task_store.pop(task_id, None)
    if _TASK_ID_RE.fullmatch(str(task_id)):
        shutil.rmtree(os.path.join(TASK_BASE_DIR, task_id), ignore_errors=True)


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

    A task whose batch is still queued or running (:func:`hold_task`) is kept
    however old it is; it becomes sweepable when the batch ends.

    Returns a count of what was removed, so callers can log it.
    """
    cutoff = time.time() - max(1, int(settings.task_ttl_minutes)) * 60
    removed_tracked = 0
    removed_orphans = 0

    with _lock:
        expired = [tid for tid, t in _task_store.items() if t.created_at < cutoff and tid not in _held]
        for tid in expired:
            del _task_store[tid]
            shutil.rmtree(os.path.join(TASK_BASE_DIR, tid), ignore_errors=True)
            removed_tracked += 1
        live_ids = set(_task_store) | _held

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
