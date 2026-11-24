"""Warn when a second backend process uses the same data directory.

The backend supports one worker process (docs/deployment.md): layout tasks,
the embedded Chroma client and the per-run locks live in process memory.
At startup the lifespan takes an exclusive, non-blocking advisory lock
(fcntl.flock) on a file next to the pipeline database and holds it until
shutdown. A second process that finds it held logs one ERROR and carries on.
It warns by design rather than refusing to start, so a setup that runs today
keeps running.

Only the process that runs the lifespan takes the lock, so uvicorn --reload's
supervisor does not. Model-pool processes are spawned, not forked, and do not
inherit it. Where fcntl does not exist (Windows), nothing happens.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import IO

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

_LOGGER = logging.getLogger(__name__)


class ProcessGuard:
    def __init__(self, lock_path: Path) -> None:
        self.lock_path = Path(lock_path)
        self._handle: IO[str] | None = None

    def acquire(self) -> bool:
        """Take the lock; False, after logging the conflict, when another process holds it."""
        if fcntl is None:
            return True
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.lock_path, "a+", encoding="utf-8")  # noqa: SIM115 - held for the process's life
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.seek(0)
            holder = handle.read().strip() or "unknown"
            handle.close()
            _LOGGER.error(
                "A second backend process is using the data directory %s (lock %s, held by pid %s). "
                "Only one worker process is supported: tasks, the vector store and the per-run locks "
                "live in process memory. See docs/deployment.md.",
                self.lock_path.parent,
                self.lock_path.name,
                holder,
            )
            return False
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        self._handle = handle
        return True

    def release(self) -> None:
        if self._handle is None or fcntl is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def guard_for_database(database: Path) -> ProcessGuard:
    """The guard for a pipeline database: <name>.lock beside it."""
    database = Path(database)
    return ProcessGuard(database.with_name(database.name + ".lock"))
