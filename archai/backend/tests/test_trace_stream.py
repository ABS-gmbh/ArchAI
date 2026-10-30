"""The OCR trace stream ends for every finished run, and reads off the event loop.

FAILED_QUALITY, the status the quality gates set, was not among the statuses
that end the stream, so it polled such a run every 0.4 s for as long as the
client stayed (audit OCR-010); a run left RUNNING by a restart did the same.
Each poll ran its SQLite queries on the event loop thread (audit DB-03).
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from app.routers import ocr as ocr_router


def _snapshot(status: str) -> dict[str, Any]:
    return {"run": {"run_id": "run-1", "status": status, "current_stage": "X", "updated_at": "t"}, "events": []}


def _stream(monkeypatch: Any, snapshot) -> list[str]:
    monkeypatch.setattr(ocr_router, "get_run", lambda run_id: {"run_id": run_id})
    monkeypatch.setattr(ocr_router, "_build_trace_snapshot", snapshot)

    async def collect() -> list[str]:
        response = await ocr_router.ocr_trace_stream("run-1")
        return [chunk async for chunk in response.body_iterator]

    return asyncio.run(asyncio.wait_for(collect(), timeout=5))


@pytest.mark.parametrize("status", ["COMPLETED", "FAILED", "FAILED_QUALITY"])
def test_the_stream_ends_when_the_run_has_finished(monkeypatch: Any, status: str) -> None:
    chunks = _stream(monkeypatch, lambda run_id: _snapshot(status))
    assert len(chunks) == 2
    assert f'"status": "{status}"' in chunks[0]
    assert chunks[1].startswith("event: end")


def test_the_snapshot_is_read_off_the_event_loop(monkeypatch: Any) -> None:
    threads: list[str] = []

    def snapshot(run_id: str) -> dict[str, Any]:
        threads.append(threading.current_thread().name)
        return _snapshot("COMPLETED")

    _stream(monkeypatch, snapshot)
    assert threads and threads[0] != threading.main_thread().name


def test_a_run_that_never_finishes_does_not_hold_the_stream(monkeypatch: Any) -> None:
    readings = iter([0.0, 1.0, 10.0 ** 6])  # start, first check, after the first wait
    monkeypatch.setattr(ocr_router, "time", SimpleNamespace(monotonic=lambda: next(readings)))
    chunks = _stream(monkeypatch, lambda run_id: _snapshot("RUNNING"))
    assert '"status": "RUNNING"' in chunks[0]
    assert chunks[-1].startswith("event: end")
