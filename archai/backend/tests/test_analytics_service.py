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


# ── AN-01: bound parameters, and the connection is always closed ────────


def _visits_db(path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE visits (timestamp TEXT, visitor_id TEXT, country TEXT, city TEXT, browser TEXT,"
            " os TEXT, device TEXT, page TEXT, action TEXT, is_bot INTEGER)"
        )
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.executemany(
            "INSERT INTO visits VALUES (?, ?, 'DE', '', '', '', '', '/', 'view', 0)",
            [(now, "v1"), ("2001-01-01 10:00:00", "v2"), (None, "v3")],
        )


def test_the_day_filter_limits_and_no_filter_admits_every_row(monkeypatch, tmp_path):
    path = tmp_path / "analytics.db"
    _visits_db(path)
    monkeypatch.setattr(analytics_service.settings, "analytics_db_path", str(path))
    assert get_analytics_data(30)["summary"]["total_visits"] == 1
    assert get_analytics_data(0)["summary"]["total_visits"] == 3


def test_the_sql_takes_the_dates_as_parameters():
    import inspect

    source = inspect.getsource(analytics_service.get_analytics_data)
    assert "{date_filter}" not in source and "'{today}'" not in source


def test_the_connection_is_closed_when_a_query_fails(monkeypatch, tmp_path):
    path = tmp_path / "analytics.db"
    sqlite3.connect(path).close()  # no visits table: the first query fails
    monkeypatch.setattr(analytics_service.settings, "analytics_db_path", str(path))
    closed: list[bool] = []
    real_connect = sqlite3.connect

    class Tracked:
        def __init__(self, *args, **kwargs):
            self._conn = real_connect(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def close(self):
            closed.append(True)
            self._conn.close()

    monkeypatch.setattr(analytics_service.sqlite3, "connect", Tracked)
    assert "error" in get_analytics_data()
    assert closed == [True]
