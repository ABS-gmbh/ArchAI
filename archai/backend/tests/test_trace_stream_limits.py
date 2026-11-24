"""Trace streams are bounded: a cap on open streams, a slower poll when nothing changes (security review S1).

A stream on a reserved run stayed open up to an hour, rebuilding a snapshot
every 0.4 s, with no cap on how many were open.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

from app.config import settings
from app.routers import ocr as ocr_router


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from app.db import pipeline_db

    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    monkeypatch.setattr(ocr_router, "_open_trace_streams", 0, raising=False)
    return pipeline_db


async def _drain(response: Any) -> list[str]:
    return [chunk async for chunk in response.body_iterator]


def test_streams_beyond_the_cap_are_refused_and_a_closed_one_frees_its_place(db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ocr_trace_stream_max_open", 1, raising=False)
    run_id = db.create_run(asset_ref="page", asset_sha256=None)
    db.update_run_fields(run_id, status="COMPLETED")

    async def scenario() -> None:
        first = await ocr_router.ocr_trace_stream(run_id)
        with pytest.raises(HTTPException) as refused:
            await ocr_router.ocr_trace_stream(run_id)
        assert refused.value.status_code == 429
        await _drain(first)  # the run is finished: the stream ends
        third = await ocr_router.ocr_trace_stream(run_id)
        await _drain(third)

    asyncio.run(scenario())


def test_the_poll_slows_while_nothing_changes_and_quickens_on_a_change(db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = db.create_run(asset_ref="page", asset_sha256=None)
    snapshots = iter(
        [{"run": {"status": "RUNNING", "updated_at": "a"}, "events": []}] * 6
        + [{"run": {"status": "RUNNING", "updated_at": "b"}, "events": [{}]}]
        + [{"run": {"status": "RUNNING", "updated_at": "b"}, "events": [{}]}]
        + [{"run": {"status": "COMPLETED", "updated_at": "c"}, "events": [{}]}]
    )
    monkeypatch.setattr(ocr_router, "_build_trace_snapshot", lambda _run_id: next(snapshots))
    slept: list[float] = []

    async def no_sleep(seconds: float) -> None:
        slept.append(round(seconds, 2))

    monkeypatch.setattr(ocr_router.asyncio, "sleep", no_sleep)

    async def scenario() -> None:
        await _drain(await ocr_router.ocr_trace_stream(run_id))

    asyncio.run(scenario())
    assert slept[0] == 0.4
    assert slept[1:6] == sorted(slept[1:6]) and max(slept) <= 1.0 and slept[5] == 1.0  # slower, never above 1 s
    assert slept[6] == 0.4  # a change: back to the fast poll


def test_the_stream_of_an_unclaimed_expired_reservation_ends(db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = db.reserve_run()
    created = (datetime.now(timezone.utc) - timedelta(seconds=settings.ocr_run_reservation_ttl_seconds + 60)).isoformat()
    with db._connect() as conn:
        conn.execute("UPDATE pipeline_runs SET created_at=?, updated_at=? WHERE run_id=?", (created, created, run_id))
        conn.commit()

    async def no_sleep(_seconds: float) -> None:
        raise AssertionError("an expired reservation must end its stream at once")

    monkeypatch.setattr(ocr_router.asyncio, "sleep", no_sleep)

    async def scenario() -> list[str]:
        return await _drain(await ocr_router.ocr_trace_stream(run_id))

    chunks = asyncio.run(scenario())
    assert chunks[-1].startswith("event: end")
