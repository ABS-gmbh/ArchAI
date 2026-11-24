"""The analytics login limit cannot be escaped by changing address, racing, or flooding its table.

Found by a security review of the queue tip: the limit was keyed only on the
client address, which a client chose itself (the frontend proxy forwarded its
X-Forwarded-For, and uvicorn trusts that header from 127.0.0.1): 40 wrong
passwords with 40 made-up addresses were all answered 401, never 429. The check
and the recording of a failure were two steps, so about 40 parallel guesses all
passed the check before the first failure was recorded. And past 10,000
addresses every failure rescanned the whole table under the lock (20,000
addresses took 4.2 s to record, 40,000 took 21 s).
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from app.config import Settings, settings
from app.routers import analytics
from app.services.login_limiter import FailedLoginLimiter


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setattr(analytics, "_login_limiter", FailedLoginLimiter())
    monkeypatch.setattr(settings, "analytics_username", "operator")
    monkeypatch.setattr(settings, "analytics_password", "the-right-password")
    monkeypatch.setattr(settings, "analytics_login_max_failures", 5)
    monkeypatch.setattr(settings, "analytics_login_window_seconds", 900)
    application = FastAPI()
    application.include_router(analytics.router)
    return application


def test_a_new_address_for_every_guess_still_meets_a_limit(app: FastAPI) -> None:
    # As deployed: uvicorn trusts X-Forwarded-For from 127.0.0.1, where the frontend proxy runs.
    client = TestClient(ProxyHeadersMiddleware(app, trusted_hosts="127.0.0.1"), client=("127.0.0.1", 50000))
    codes = [
        client.post(
            "/analytics/login",
            headers={"X-Forwarded-For": f"10.0.{i // 250}.{i % 250}"},
            json={"username": "operator", "password": f"guess{i}"},
        ).status_code
        for i in range(40)
    ]
    assert codes.count(401) == settings.analytics_login_max_failures_per_username
    assert set(codes[settings.analytics_login_max_failures_per_username:]) == {429}


def test_the_user_name_limit_is_case_and_space_blind(app: FastAPI) -> None:
    client = TestClient(ProxyHeadersMiddleware(app, trusted_hosts="127.0.0.1"), client=("127.0.0.1", 50000))
    names = ["operator", "Operator", " operator", "OPERATOR "]
    codes = [
        client.post(
            "/analytics/login",
            headers={"X-Forwarded-For": f"10.1.0.{i}"},
            json={"username": names[i % len(names)], "password": "wrong"},
        ).status_code
        for i in range(30)
    ]
    assert 429 in codes


def test_parallel_guesses_from_one_address_are_counted_before_they_are_checked(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_compare = analytics.secrets.compare_digest

    def slow_compare(a: bytes, b: bytes) -> bool:
        time.sleep(0.05)  # every guess is checked while the others are in flight
        return real_compare(a, b)

    monkeypatch.setattr(analytics.secrets, "compare_digest", slow_compare)
    client = TestClient(app)
    barrier = threading.Barrier(40)
    codes: list[int] = []

    def guess(i: int) -> None:
        barrier.wait()
        codes.append(client.post("/analytics/login", json={"username": "operator", "password": f"g{i}"}).status_code)

    threads = [threading.Thread(target=guess, args=(i,)) for i in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert codes.count(401) == settings.analytics_login_max_failures
    assert codes.count(429) == 40 - settings.analytics_login_max_failures


def test_recording_a_failure_does_not_rescan_a_large_table() -> None:
    limiter = FailedLoginLimiter()
    started = time.perf_counter()
    for i in range(30_000):
        assert limiter.attempt({f"a{i}": 5}, window_seconds=900) is None
    assert time.perf_counter() - started < 3.0  # was about 10 s


def test_the_table_has_a_bound() -> None:
    limiter = FailedLoginLimiter(max_entries=1000)
    for i in range(5000):
        limiter.attempt({f"a{i}": 5}, window_seconds=900)
    assert len(limiter) <= 1000


def test_the_user_name_limit_is_a_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings.model_fields["analytics_login_max_failures_per_username"].default == 10
    monkeypatch.setenv("ANALYTICS_LOGIN_MAX_FAILURES_PER_USERNAME", "0")
    with pytest.raises(ValidationError, match="ANALYTICS_LOGIN_MAX_FAILURES_PER_USERNAME must be at least 1"):
        Settings(_env_file=None)


# Second security review (2026-11-24): with the operator's name locked, 5,000
# attempts with new addresses and names pushed `user:operator` out of the
# bounded table, and ten more guesses at the operator's password were accepted.
def test_a_flood_of_new_keys_does_not_evict_the_operator_lockout(monkeypatch: pytest.MonkeyPatch, app: FastAPI) -> None:
    monkeypatch.setattr(analytics, "_login_limiter", FailedLoginLimiter(max_entries=50))
    client = TestClient(ProxyHeadersMiddleware(app, trusted_hosts="127.0.0.1"), client=("127.0.0.1", 50000))

    def guess(address: str, username: str) -> int:
        return client.post(
            "/analytics/login",
            headers={"X-Forwarded-For": address},
            json={"username": username, "password": "wrong"},
        ).status_code

    per_name = settings.analytics_login_max_failures_per_username
    assert [guess(f"10.1.0.{i}", "Operator") for i in range(per_name)] == [401] * per_name
    assert guess("10.1.1.1", "operator") == 429
    flood = [guess(f"10.2.{i // 250}.{i % 250}", f"someone-{i}") for i in range(200)]
    assert set(flood) == {401}
    assert len(analytics._login_limiter) <= 50
    assert [guess(f"10.3.0.{i}", "operator") for i in range(10)] == [429] * 10


def test_the_table_stays_bounded_when_only_pinned_keys_are_left() -> None:
    limiter = FailedLoginLimiter(max_entries=2)
    for key in ("user:a", "user:b", "user:c"):
        assert limiter.attempt({key: 5}, window_seconds=60, pinned={"user:a", "user:b", "user:c"}) is None
    assert len(limiter) == 3  # pinned keys are never evicted; there are as few as configured users
    assert limiter.attempt({"addr:x": 5}, window_seconds=60, pinned={"user:a", "user:b", "user:c"}) is None
    assert len(limiter) == 3  # the unpinned key went, the pinned ones stayed
