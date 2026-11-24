"""Every server-sent-event stream tells proxies not to buffer or cache it.

Chat set X-Accel-Buffering: no; the OCR trace stream and the batch progress
stream did not, so behind nginx their events arrived together at the end.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.db import pipeline_db
from app.routers import ocr, predict
from app.services import file_manager


def _assert_unbuffered(response) -> None:
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers.get("x-accel-buffering") == "no"
    assert response.headers.get("cache-control") == "no-cache"


def test_the_trace_stream_is_not_buffered() -> None:
    run_id = pipeline_db.create_run(asset_ref="page")
    pipeline_db.update_run_fields(run_id, status="COMPLETED", current_stage="DONE")
    app = FastAPI()
    app.include_router(ocr.router)
    _assert_unbuffered(TestClient(app).get(f"/ocr/trace/stream/{run_id}"))


def test_the_batch_progress_stream_is_not_buffered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path / "tasks"))
    task = file_manager.create_task()
    task.status = "completed"
    app = FastAPI()
    app.include_router(predict.router)
    _assert_unbuffered(TestClient(app).get(f"/predict/batch/{task.task_id}/progress"))
