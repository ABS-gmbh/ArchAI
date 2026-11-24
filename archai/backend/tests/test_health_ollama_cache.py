"""/api/health asks Ollama once per few seconds, not once per call (security review C5).

The check is tokenless and made a blocking request (2 s timeout) per call in
the threadpool. With an Ollama host that accepts but never answers, 200
health calls made an authenticated GET /api/classes take 9.9 s.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from app.config import settings
from app.routers import health


class _Version:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"version": "0.6.0"}


@pytest.fixture
def slow_ollama(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    calls: list[float] = []

    def get(url: str, timeout: float) -> _Version:
        calls.append(time.monotonic())
        time.sleep(0.3)  # a slow host; a silent one takes the full timeout
        return _Version()

    monkeypatch.setattr(settings, "glmocr_ollama_host", "http://127.0.0.1:11434")
    monkeypatch.setattr(health.requests, "get", get)
    monkeypatch.setattr(health, "_last_ok", {})
    monkeypatch.setattr(health, "_ollama_cache", None, raising=False)
    return calls


def test_concurrent_health_calls_make_one_ollama_request(slow_ollama: list[float]) -> None:
    results: list[dict[str, Any]] = []
    threads = [threading.Thread(target=lambda: results.append(health._check_ollama())) for _ in range(20)]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(slow_ollama) == 1
    assert time.monotonic() - started < 1.0  # no call waits behind another's request
    assert sum(1 for r in results if r["ok"]) >= 1


def test_the_answer_is_reused_for_a_few_seconds_then_asked_again(slow_ollama: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1000.0]
    monkeypatch.setattr(health.time, "monotonic", lambda: now[0])
    assert health._check_ollama()["ok"] is True
    now[0] += health._OLLAMA_CACHE_SECONDS - 0.1
    assert health._check_ollama()["version"] == "0.6.0"
    assert len(slow_ollama) == 1
    now[0] += 0.2
    health._check_ollama()
    assert len(slow_ollama) == 2


def test_a_changed_host_is_asked_at_once(slow_ollama: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    health._check_ollama()
    monkeypatch.setattr(settings, "glmocr_ollama_host", "http://127.0.0.1:11435")
    health._check_ollama()
    assert len(slow_ollama) == 2
