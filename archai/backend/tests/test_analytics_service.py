"""The analytics dashboard says when it has no database instead of showing zeros (audit AN-02).

analytics_db_path points at a file that does not exist in this deployment.
sqlite3.connect created it empty, the first query failed with "no such table:
visits", and the blanket except turned that into an all-zero dashboard,
reported only by a print().
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from app.services import analytics_service
from app.services.analytics_service import get_analytics_data


def test_a_missing_database_is_reported_and_not_created(monkeypatch, tmp_path):
    path = tmp_path / "analytics.db"
    monkeypatch.setattr(analytics_service.settings, "analytics_db_path", str(path))

    data = get_analytics_data()

    assert "not found" in data["error"]
    assert data["summary"] == {} and data["recent_visits"] == []
    assert not path.exists()


def test_a_database_without_visits_is_reported(monkeypatch, tmp_path):
    path = tmp_path / "analytics.db"
    sqlite3.connect(path).close()
    monkeypatch.setattr(analytics_service.settings, "analytics_db_path", str(path))

    assert "no such table: visits" in get_analytics_data()["error"]


def test_visits_are_counted(monkeypatch, tmp_path):
    path = tmp_path / "analytics.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE visits (timestamp TEXT, visitor_id TEXT, country TEXT, city TEXT, browser TEXT,"
            " os TEXT, device TEXT, page TEXT, action TEXT, is_bot INTEGER)"
        )
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.executemany(
            "INSERT INTO visits VALUES (?, ?, ?, '', '', '', '', '/', 'view', 0)",
            [(now, "v1", "DE"), (now, "v2", "FR"), (now, "v1", "DE")],
        )
    monkeypatch.setattr(analytics_service.settings, "analytics_db_path", str(path))

    data = get_analytics_data()

    assert "error" not in data
    assert (data["summary"]["total_visits"], data["summary"]["unique_visitors"]) == (3, 2)
