"""Startup says loudly when analytics is configured without JWT_SECRET.

The behaviour is unchanged: a random signing secret per process. The
consequence, every analytics session ending at restart, used to show only as a
WARNING at the first login.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

import app.main as main
from app.config import settings

MESSAGE = "JWT_SECRET is not set while analytics login is configured"


def _start(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, message: str = MESSAGE
) -> list[logging.LogRecord]:
    monkeypatch.setattr(main, "init_model_pool", lambda: None)
    monkeypatch.setattr(main, "shutdown_model_pool", lambda: None)
    monkeypatch.setattr(main, "cleanup_expired_tasks", lambda: {})
    monkeypatch.setattr(main, "configure_logging", lambda *args, **kwargs: None)

    async def start_and_stop() -> None:
        async with main.lifespan(main.app):
            pass

    with caplog.at_level(logging.WARNING, logger="app.main"):
        asyncio.run(start_and_stop())
    return [record for record in caplog.records if message in record.getMessage()]


def test_configured_analytics_without_a_secret_is_an_error(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    monkeypatch.setattr(settings, "analytics_username", "operator")
    monkeypatch.setattr(settings, "analytics_password", "a-password")
    monkeypatch.setattr(settings, "jwt_secret", "")
    records = _start(monkeypatch, caplog)
    assert [record.levelname for record in records] == ["ERROR"]
    assert "every login ends when the process restarts" in records[0].getMessage()


@pytest.mark.parametrize(
    ("username", "secret"), [("", ""), ("operator", "j" * 40)], ids=["analytics-off", "secret-set"]
)
def test_nothing_is_said_otherwise(monkeypatch: pytest.MonkeyPatch, caplog, username: str, secret: str) -> None:
    monkeypatch.setattr(settings, "analytics_username", username)
    monkeypatch.setattr(settings, "analytics_password", "a-password")
    monkeypatch.setattr(settings, "jwt_secret", secret)
    assert _start(monkeypatch, caplog) == []


SHORT = "JWT_SECRET is shorter than 32 characters"


@pytest.mark.parametrize("analytics_user", ["operator", ""])
def test_a_short_secret_is_an_error_that_does_not_print_it(
    monkeypatch: pytest.MonkeyPatch, caplog, analytics_user: str
) -> None:
    monkeypatch.setattr(settings, "analytics_username", analytics_user)
    monkeypatch.setattr(settings, "analytics_password", "a-password")
    monkeypatch.setattr(settings, "jwt_secret", "short-secret-value")
    records = _start(monkeypatch, caplog, SHORT)
    assert [record.levelname for record in records] == ["ERROR"]
    text = records[0].getMessage()
    assert "at least 32 characters" in text
    assert "short-secret-value" not in text and str(len("short-secret-value")) not in text


def test_a_secret_of_32_characters_is_not_mentioned(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    monkeypatch.setattr(settings, "analytics_username", "operator")
    monkeypatch.setattr(settings, "analytics_password", "a-password")
    monkeypatch.setattr(settings, "jwt_secret", "j" * 32)
    assert _start(monkeypatch, caplog, SHORT) == []
