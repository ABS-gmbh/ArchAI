"""The data paths a server writes to: checked and logged at startup.

TASK_BASE_DIR has its own check (file_manager.check_task_base_dir), which
creates the directory. The paths here are only checked, never created: each
store creates its own file or directory on first use, so a missing one is
fine as long as it could be created.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from app.config import settings

_LOGGER = logging.getLogger(__name__)


def _paths() -> list[tuple[str, Path, bool]]:
    """(setting, path, is a directory) for each store, resolved as the code that writes it resolves it."""
    from app.db import pipeline_db
    from app.services import wikidata_client

    return [
        ("ARCHAI_DB_PATH", pipeline_db._db_config().path, False),
        ("CHROMA_PERSIST_DIR", Path(settings.chroma_persist_dir).expanduser(), True),
        ("ARCHAI_WIKIDATA_CACHE", wikidata_client._cache_path(), False),
        ("OCR_EVIDENCE_PATH", Path(settings.ocr_evidence_path).expanduser(), False),
    ]


def _writable_dir(path: Path) -> bool:
    return path.is_dir() and os.access(path, os.W_OK | os.X_OK)


def _problem(path: Path, is_dir: bool) -> str | None:
    """Why this process could not write *path*, or None if it could."""
    if is_dir and path.exists():
        return None if _writable_dir(path) else "is not a writable directory"
    if not is_dir and path.exists():
        if path.is_dir():
            return "is a directory, not a file"
        return None if os.access(path, os.W_OK) else "is a file this process cannot write"
    # Missing: the nearest existing ancestor must let it be created.
    ancestor = path.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if _writable_dir(ancestor):
        return None
    return f"does not exist, and {str(ancestor)!r} does not let this process create it"


def check_data_paths() -> None:
    """Raise RuntimeError naming the first setting whose path this process cannot write."""
    for variable, path, is_dir in _paths():
        problem = _problem(path, is_dir)
        if problem:
            raise RuntimeError(
                f"{variable}={str(path)!r}: the server cannot write there ({problem}). "
                f"Point {variable} at a writable location (docs/deployment.md)."
            )


def log_data_paths(task_base_dir: str) -> None:
    """One INFO line with every path the server keeps data in, so a deployment can confirm them.

    The analytics database is named too, as read only: the server never writes
    it, so it is not checked, but it is backed up with the rest.
    """
    entries = [f"{variable}={str(path)!r}" for variable, path, _ in _paths()]
    entries.append(f"TASK_BASE_DIR={task_base_dir!r}")
    entries.append(f"ANALYTICS_DB_PATH={str(Path(settings.analytics_db_path).expanduser())!r} (read only)")
    _LOGGER.info("Data paths: %s", ", ".join(entries))
