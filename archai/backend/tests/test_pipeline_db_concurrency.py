"""The pipeline DB lets readers through while a run writes (audit DB-01).

It was opened in SQLite's default rollback-journal mode with Python's 5 s
busy timeout. OCR runs write from worker threads while trace streams poll
every 0.4 s; a writer's lock shut readers out, and waiting past five seconds
raised "database is locked" mid-run. It now uses WAL and a 10 s busy timeout.
"""

from __future__ import annotations

import sqlite3
import time

from app.db import pipeline_db


def _connect(monkeypatch, tmp_path) -> sqlite3.Connection:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "pipeline.sqlite"))
    return pipeline_db._connect()


def test_connections_use_wal_and_a_busy_timeout(monkeypatch, tmp_path):
    conn = _connect(monkeypatch, tmp_path)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 10000
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        conn.close()


def test_a_reader_is_not_blocked_by_a_writer(monkeypatch, tmp_path):
    writer = _connect(monkeypatch, tmp_path)
    writer.execute("CREATE TABLE t (x INTEGER)")
    writer.execute("INSERT INTO t VALUES (1)")
    writer.commit()
    reader = pipeline_db._connect()
    reader.execute("PRAGMA busy_timeout=200")
    try:
        writer.execute("BEGIN EXCLUSIVE")
        writer.execute("INSERT INTO t VALUES (2)")
        started = time.monotonic()
        rows = reader.execute("SELECT x FROM t").fetchall()
        assert [row[0] for row in rows] == [1]
        assert time.monotonic() - started < 0.1
        writer.commit()
        assert [row[0] for row in reader.execute("SELECT x FROM t ORDER BY x")] == [1, 2]
    finally:
        writer.close()
        reader.close()
