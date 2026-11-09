"""Opening an older pipeline DB keeps its v1 entity decisions (audit DB-06).

The first touch of a database migrated entity_decisions to v2 by dropping the
v1 table and entity_attempts, lazily, inside any ordinary call: pointing the
app at an older database destroyed its decision history with no log line.
They are now renamed aside.
"""

from __future__ import annotations

import logging
import sqlite3

from app.db import pipeline_db


def _v1_database(path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE entity_decisions (decision_id TEXT PRIMARY KEY, run_id TEXT, surface TEXT, status TEXT);
            CREATE TABLE entity_attempts (attempt_id TEXT PRIMARY KEY, decision_id TEXT
                REFERENCES entity_decisions(decision_id));
            CREATE INDEX idx_entity_decisions_run_id ON entity_decisions(run_id);
            CREATE INDEX idx_entity_attempts_decision_id ON entity_attempts(decision_id);
            INSERT INTO entity_decisions VALUES ('d1', 'r1', 'Arthur', 'linked'), ('d2', 'r1', 'Camelot', 'unresolved');
            INSERT INTO entity_attempts VALUES ('a1', 'd1');
            """
        )


def _open(monkeypatch, path) -> None:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(path))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    pipeline_db.get_run("no-such-run")


def _tables(path) -> dict[str, list[str]]:
    with sqlite3.connect(path) as conn:
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {name: [r[1] for r in conn.execute(f'PRAGMA table_info("{name}")')] for name in names}


def test_v1_decisions_are_set_aside_not_dropped(monkeypatch, tmp_path, caplog):
    path = tmp_path / "old.sqlite"
    _v1_database(path)

    with caplog.at_level(logging.WARNING, logger="archai.pipeline_db"):
        _open(monkeypatch, path)

    tables = _tables(path)
    [old_decisions] = [n for n in tables if n.startswith("entity_decisions_v1_")]
    [old_attempts] = [n for n in tables if n.startswith("entity_attempts_v1_")]
    with sqlite3.connect(path) as conn:
        assert conn.execute(f'SELECT surface FROM "{old_decisions}" ORDER BY decision_id').fetchall() == [
            ("Arthur",), ("Camelot",)
        ]
        assert conn.execute(f'SELECT count(*) FROM "{old_attempts}"').fetchone()[0] == 1
    assert "span_key" in tables["entity_decisions"]
    assert "entity_attempts" in tables
    assert "Set aside v1 table entity_decisions" in caplog.text


def test_the_v2_tables_get_their_indexes(monkeypatch, tmp_path):
    path = tmp_path / "old.sqlite"
    _v1_database(path)
    _open(monkeypatch, path)
    with sqlite3.connect(path) as conn:
        indexes = dict(conn.execute("SELECT name, tbl_name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"))
    assert indexes["idx_entity_decisions_run_id"] == "entity_decisions"
    assert indexes["idx_entity_attempts_decision_id"] == "entity_attempts"


def test_a_v2_database_is_left_alone(monkeypatch, tmp_path):
    path = tmp_path / "new.sqlite"
    _open(monkeypatch, path)
    before = _tables(path)
    _open(monkeypatch, path)
    assert _tables(path) == before
    assert not any("_v1_" in name for name in before)


# ── DB-05: a failed column migration is reported at startup ─────────────


class _FailingAlter:
    """A connection whose ALTER TABLE on ocr_attempts fails with `message`."""

    def __init__(self, conn: sqlite3.Connection, message: str) -> None:
        self._conn = conn
        self._message = message

    def execute(self, sql: str, *args):
        if sql.startswith("ALTER TABLE ocr_attempts"):
            raise sqlite3.OperationalError(self._message)
        return self._conn.execute(sql, *args)


def _v1_ocr_attempts(path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE pipeline_runs (run_id TEXT PRIMARY KEY);"
        "CREATE TABLE ocr_attempts (attempt_id INTEGER PRIMARY KEY, run_id TEXT);"
    )
    return conn


def test_a_failed_column_migration_raises(tmp_path):
    """Audit DB-05: it was swallowed, and the run failed later on the missing column."""
    import pytest

    conn = _v1_ocr_attempts(tmp_path / "old.sqlite")
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        pipeline_db._ensure_run_columns(_FailingAlter(conn, "database is locked"))


def test_a_column_that_already_exists_is_passed_over(tmp_path):
    conn = _v1_ocr_attempts(tmp_path / "old.sqlite")
    pipeline_db._ensure_run_columns(_FailingAlter(conn, "duplicate column name: tile_grid"))


def test_missing_columns_are_added(tmp_path):
    conn = _v1_ocr_attempts(tmp_path / "old.sqlite")
    pipeline_db._ensure_run_columns(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(ocr_attempts)")}
    assert {"tile_grid", "text_sha256", "noop_detected"} <= columns
