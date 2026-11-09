"""SQLite analytics queries."""

import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

from app.config import settings

_LOGGER = logging.getLogger(__name__)

_EMPTY = {"summary": {}, "by_country": [], "by_day": [], "recent_visits": []}


def get_analytics_data(days_filter: int = 30) -> dict:
    """Fetch analytics data from the SQLite database.

    Returns a dict with keys: summary, by_country, by_day, recent_visits, and
    "error" when there is nothing to read. The database is opened read-only: a
    missing file used to be created empty, and the dashboard showed zeros with
    no sign that it had no data (audit AN-02).
    """
    if not os.path.exists(settings.analytics_db_path):
        return {**_EMPTY, "error": f"Analytics database not found: {settings.analytics_db_path}"}
    # The cutoff and today's date are bound parameters, not spliced into the
    # SQL, and the connection is closed on every path - it leaked whenever a
    # query failed (audit AN-01). With no day filter the cutoff is NULL, which
    # admits every row, as the old query without a WHERE clause did.
    cutoff = (datetime.now() - timedelta(days=days_filter)).strftime("%Y-%m-%d") if days_filter > 0 else None
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        uri = f"{Path(settings.analytics_db_path).resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            by_country = [dict(r) for r in conn.execute(
                """
                SELECT
                    COALESCE(NULLIF(country, ''), 'Unknown') as country,
                    COUNT(*) as visits,
                    COUNT(DISTINCT visitor_id) as unique_visitors
                FROM visits
                WHERE (? IS NULL OR DATE(timestamp) >= ?)
                GROUP BY country
                ORDER BY visits DESC
                """,
                (cutoff, cutoff),
            )]
            by_day = [dict(r) for r in conn.execute(
                """
                SELECT
                    DATE(timestamp) as date,
                    COUNT(*) as visits,
                    COUNT(DISTINCT visitor_id) as unique_visitors
                FROM visits
                WHERE (? IS NULL OR DATE(timestamp) >= ?)
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                LIMIT 30
                """,
                (cutoff, cutoff),
            )]
            recent_visits = [dict(r) for r in conn.execute(
                """
                SELECT
                    timestamp,
                    COALESCE(NULLIF(country, ''), 'Unknown') as country,
                    COALESCE(NULLIF(city, ''), 'Unknown') as city,
                    browser, os, device, page, action
                FROM visits
                WHERE (? IS NULL OR DATE(timestamp) >= ?)
                ORDER BY timestamp DESC
                LIMIT 100
                """,
                (cutoff, cutoff),
            )]
            summary_row = conn.execute(
                """
                SELECT
                    COUNT(*) as total_visits,
                    COUNT(DISTINCT visitor_id) as unique_visitors,
                    COUNT(DISTINCT COALESCE(NULLIF(country, ''), 'Unknown')) as countries,
                    SUM(CASE WHEN DATE(timestamp) = ? THEN 1 ELSE 0 END) as today_visits,
                    SUM(CASE WHEN is_bot = 1 THEN 1 ELSE 0 END) as bot_visits
                FROM visits
                WHERE (? IS NULL OR DATE(timestamp) >= ?)
                """,
                (today, cutoff, cutoff),
            ).fetchone()
    except Exception as e:
        _LOGGER.exception("Analytics DB error")
        return {**_EMPTY, "error": f"Analytics database could not be read: {e}"}
    return {
        "summary": dict(summary_row) if summary_row else {},
        "by_country": by_country,
        "by_day": by_day,
        "recent_visits": recent_visits,
    }
