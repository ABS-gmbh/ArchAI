"""Pipeline DB connections close when their ``with`` block ends.

sqlite3.Connection's own context manager commits and leaves the connection
open; they were freed only by the cyclic garbage collector, so a burst of
reads held hundreds of connections and twice as many file descriptors.
"""

from __future__ import annotations

import gc
import sqlite3
from pathlib import Path

import pytest

from app.db import pipeline_db


def _open_connections() -> int:
    return sum(
        1
        for obj in gc.get_objects()
        if isinstance(obj, sqlite3.Connection) and _is_open(obj)
    )


def _is_open(conn: sqlite3.Connection) -> bool:
    try:
        conn.total_changes  # noqa: B018 - raises ProgrammingError once closed
    except sqlite3.ProgrammingError:
        return False
    return True


def test_reads_do_not_accumulate_open_connections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    run_id = pipeline_db.create_run(asset_ref="page")
    gc.collect()
    gc.disable()
    try:
        before = _open_connections()
        for _ in range(200):
            pipeline_db.get_run(run_id)
            pipeline_db.list_chunks(run_id)
        after = _open_connections()
    finally:
        gc.enable()
    assert after - before <= 1


def test_a_write_is_still_committed_and_an_error_rolled_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    run_id = pipeline_db.create_run(asset_ref="page")
    with pytest.raises(RuntimeError):
        with pipeline_db._connect() as conn:
            conn.execute("UPDATE pipeline_runs SET asset_ref='changed' WHERE run_id=?", (run_id,))
            raise RuntimeError("abort")
    assert pipeline_db.get_run(run_id)["asset_ref"] == "page"
    with pipeline_db._connect() as conn:
        conn.execute("UPDATE pipeline_runs SET asset_ref='kept' WHERE run_id=?", (run_id,))
    assert pipeline_db.get_run(run_id)["asset_ref"] == "kept"
