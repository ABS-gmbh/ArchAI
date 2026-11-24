"""Analytics errors tell the client what happened in a fixed sentence; paths and exceptions go to the log.

Found by a security review: /api/analytics/data answered "Analytics database
not found: <absolute path>", and "Analytics database could not be read: <the
exception>", which named files and SQL on the server to whoever held a
session.
"""

from __future__ import annotations

import logging
import sqlite3

import pytest

from app.services import analytics_service
from app.services.analytics_service import get_analytics_data


def test_a_missing_database_names_no_path(monkeypatch: pytest.MonkeyPatch, tmp_path, caplog) -> None:
    path = tmp_path / "secret-dir" / "analytics.db"
    monkeypatch.setattr(analytics_service.settings, "analytics_db_path", str(path))
    with caplog.at_level(logging.WARNING, logger=analytics_service.__name__):
        data = get_analytics_data()
    assert data["error"] == "The analytics database was not found. The server log names the file it looked for."
    assert str(tmp_path) not in data["error"]
    assert str(path) in caplog.text


def test_an_unreadable_database_gives_no_exception_text(monkeypatch: pytest.MonkeyPatch, tmp_path, caplog) -> None:
    path = tmp_path / "analytics.db"
    sqlite3.connect(path).close()  # no visits table
    monkeypatch.setattr(analytics_service.settings, "analytics_db_path", str(path))
    with caplog.at_level(logging.ERROR, logger=analytics_service.__name__):
        data = get_analytics_data()
    assert data["error"] == "The analytics database could not be read. The server log has the details."
    assert "no such table" in caplog.text and str(path) in caplog.text
