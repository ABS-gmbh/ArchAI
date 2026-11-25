"""A trace-stream tick reads its snapshot on one connection (audit DB-03).

Each 400 ms tick of /ocr/trace/stream built its snapshot from six reads, each
on a connection of its own (get_run, list_events, list_chunks,
list_entity_mentions, count_chunks, count_entity_mentions). Measured on a
scratch DB with 400 events (2026-11-25): 6 connections and 3.35 ms per tick
before, 1 connection and 0.90 ms after. The snapshot a client receives is
the same.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.db import pipeline_db
from app.routers import ocr as ocr_router


def _run_with_events(events: int) -> str:
    run_id = pipeline_db.create_run("asset-db03")
    for i in range(events):
        pipeline_db.log_event(run_id, f"STAGE_{i % 4}", "END" if i % 2 else "START", f"event {i}")
    pipeline_db.insert_chunks(run_id, [{"idx": i, "start_offset": i * 10, "end_offset": i * 10 + 9, "text": f"chunk {i}"} for i in range(25)])
    return run_id


def _snapshot_from_the_public_reads(run_id: str) -> dict[str, Any]:
    """What the snapshot was built from before: the six public reads."""
    run = pipeline_db.get_run(run_id)
    assert run is not None
    events = pipeline_db.list_events(run_id)
    return {
        "run_id": run_id,
        "run": run,
        "events": events,
        "latest_completed_stage": ocr_router._latest_completed_stage(events),
        "chunks_count": int(run.get("chunks_count") or pipeline_db.count_chunks(run_id)),
        "mentions_count": int(run.get("mentions_count") or pipeline_db.count_entity_mentions(run_id)),
        "chunks_preview": pipeline_db.list_chunks(run_id, limit=20),
        "top_mentions": pipeline_db.list_entity_mentions(run_id, limit=20),
    }


def test_a_tick_opens_one_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = _run_with_events(30)
    opened: list[int] = []
    real = pipeline_db._connect

    def counting() -> Any:
        opened.append(1)
        return real()

    monkeypatch.setattr(pipeline_db, "_connect", counting)
    snapshot = ocr_router._build_trace_snapshot(run_id)
    assert snapshot is not None and len(snapshot["events"]) == 30
    assert len(opened) == 1


def test_the_snapshot_is_what_the_six_reads_gave() -> None:
    run_id = _run_with_events(12)
    assert ocr_router._build_trace_snapshot(run_id) == _snapshot_from_the_public_reads(run_id)
    snapshot = ocr_router._build_trace_snapshot(run_id)
    assert snapshot is not None
    assert snapshot["chunks_count"] == 25 and len(snapshot["chunks_preview"]) == 20


def test_an_unknown_run_has_no_snapshot() -> None:
    assert ocr_router._build_trace_snapshot("no-such-run") is None
