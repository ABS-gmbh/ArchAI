import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Optional

from archai_backend.config.settings import DB_PATH


@contextmanager
def get_conn() -> Iterable[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pages (
                page_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                page_id TEXT NOT NULL,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress INTEGER NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spans (
                span_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                page_id TEXT NOT NULL,
                text TEXT NOT NULL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def _now() -> str:
    return datetime.utcnow().isoformat()


def create_document(doc_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO documents (doc_id, created_at) VALUES (?, ?)",
            (doc_id, _now()),
        )


def create_page(page_id: str, doc_id: str, image_path: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO pages (page_id, doc_id, image_path, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (page_id, doc_id, image_path, _now()),
        )


def create_job(job_id: str, doc_id: str, page_id: str) -> None:
    now = _now()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO jobs (job_id, doc_id, page_id, status, stage, progress, message, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, doc_id, page_id, "running", "uploaded", 0, "Queued", now, now),
        )


def update_job(job_id: str, status: str, stage: str, progress: int, message: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?, stage = ?, progress = ?, message = ?, updated_at = ?
            WHERE job_id = ?
            """,
            (status, stage, progress, message, _now(), job_id),
        )


def get_job(job_id: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        return cur.fetchone()


def insert_span(
    span_id: str,
    doc_id: str,
    page_id: str,
    text: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO spans (span_id, doc_id, page_id, text, x1, y1, x2, y2, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (span_id, doc_id, page_id, text, x1, y1, x2, y2, _now()),
        )


def get_spans(doc_id: str, page_id: Optional[str], limit: int) -> list[sqlite3.Row]:
    with get_conn() as conn:
        if page_id:
            cur = conn.execute(
                """
                SELECT * FROM spans
                WHERE doc_id = ? AND page_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (doc_id, page_id, limit),
            )
        else:
            cur = conn.execute(
                """
                SELECT * FROM spans
                WHERE doc_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (doc_id, limit),
            )
        return cur.fetchall()


def get_span(span_id: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM spans WHERE span_id = ?", (span_id,))
        return cur.fetchone()


def get_page_image_path(doc_id: str, page_id: str) -> Optional[str]:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT image_path FROM pages WHERE doc_id = ? AND page_id = ?",
            (doc_id, page_id),
        )
        row = cur.fetchone()
        if row:
            return row["image_path"]
        return None
