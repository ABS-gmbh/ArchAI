"""Failed analytics logins are limited per client address and user name (app/services/login_limiter.py).

There was no lockout: the analytics password could be guessed as fast as
requests could be sent.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings, settings
from app.routers import analytics
from app.services.login_limiter import FailedLoginLimiter

LIMITS = {"max_failures": 3, "window_seconds": 60.0}


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _fail(limiter: FailedLoginLimiter, address: str, times: int) -> None:
    for _ in range(times):
        assert limiter.attempt({address: LIMITS["max_failures"]}, window_seconds=LIMITS["window_seconds"]) is None


def _wait(limiter: FailedLoginLimiter, address: str) -> float | None:
    """Seconds to wait, or None; an attempt that may go ahead is taken back at once."""
    wait = limiter.attempt({address: LIMITS["max_failures"]}, window_seconds=LIMITS["window_seconds"])
    if wait is None:
        limiter._failures[address].pop()
    return wait


def test_an_address_is_blocked_after_too_many_failures_until_the_window_passes() -> None:
    clock = Clock()
    limiter = FailedLoginLimiter(clock)
    _fail(limiter, "10.0.0.1", 2)
    assert _wait(limiter, "10.0.0.1") is None
    _fail(limiter, "10.0.0.1", 1)
    assert _wait(limiter, "10.0.0.1") == pytest.approx(60.0)
    clock.now += 45
    assert _wait(limiter, "10.0.0.1") == pytest.approx(15.0)
    clock.now += 15
    assert _wait(limiter, "10.0.0.1") is None


def test_addresses_are_counted_separately_and_a_success_clears_one() -> None:
    limiter = FailedLoginLimiter(Clock())
    _fail(limiter, "10.0.0.1", 3)
    assert _wait(limiter, "10.0.0.2") is None
    limiter.succeeded(["10.0.0.1"])
    assert _wait(limiter, "10.0.0.1") is None


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Clock]:
    clock = Clock()
    monkeypatch.setattr(analytics, "_login_limiter", FailedLoginLimiter(clock))
    monkeypatch.setattr(settings, "analytics_username", "operator")
    monkeypatch.setattr(settings, "analytics_password", "the-right-password")
    monkeypatch.setattr(settings, "analytics_login_max_failures", 5)
    monkeypatch.setattr(settings, "analytics_login_window_seconds", 900)
    app = FastAPI()
    app.include_router(analytics.router)
    return TestClient(app), clock


def _login(client: TestClient, password: str):
    return client.post("/analytics/login", json={"username": "operator", "password": password})


def test_the_sixth_failure_is_a_429_with_retry_after(client: tuple[TestClient, Clock]) -> None:
    http, clock = client
    for _ in range(5):
        assert _login(http, "wrong").status_code == 401
    blocked = _login(http, "the-right-password")
    assert blocked.status_code == 429
    assert blocked.headers["retry-after"] == "900"
    assert blocked.json()["detail"] == "Too many failed logins. Try again in 900 seconds."
    clock.now += 901
    assert _login(http, "the-right-password").status_code == 200


def test_a_success_resets_the_count(client: tuple[TestClient, Clock]) -> None:
    http, _ = client
    for _ in range(4):
        _login(http, "wrong")
    assert _login(http, "the-right-password").status_code == 200
    for _ in range(4):
        assert _login(http, "wrong").status_code == 401


@pytest.mark.parametrize(
    "name", ["ANALYTICS_LOGIN_MAX_FAILURES", "ANALYTICS_LOGIN_MAX_FAILURES_PER_USERNAME", "ANALYTICS_LOGIN_WINDOW_SECONDS"]
)
def test_limits_below_one_are_refused_by_name(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(name, "0")
    with pytest.raises(ValidationError, match=f"{name} must be at least 1"):
        Settings(_env_file=None)
