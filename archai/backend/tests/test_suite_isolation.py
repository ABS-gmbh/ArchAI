"""The suite never touches the live stores in the checkout it runs from.

The pipeline database, the Chroma store and the task directory default to
paths inside the backend directory, which are the live data on a developer's
machine. conftest.py moves all three to a session temp directory.
"""

from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.db import pipeline_db
from app.services import file_manager

BACKEND = Path(__file__).resolve().parents[1]


def _inside_backend(path: str | Path) -> bool:
    return Path(path).resolve().is_relative_to(BACKEND)


def test_the_pipeline_database_is_not_the_checkouts() -> None:
    assert not _inside_backend(pipeline_db._db_config().path)


def test_the_vector_store_is_not_the_checkouts() -> None:
    assert not _inside_backend(settings.chroma_persist_dir)


def test_the_task_directory_is_not_the_checkouts() -> None:
    assert not _inside_backend(file_manager.TASK_BASE_DIR)
