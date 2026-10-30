"""A failed batch ends as "error" and its progress stream ends (audit PRED-001).

_process_batch had no exception handling and set "processing" only after
unpacking, so a corrupt ZIP raised out of the background task with the status
still "pending". The progress stream stops only on completed or error, so it
polled every 0.5 s for as long as the client stayed connected.
"""

from __future__ import annotations

import asyncio
import json
import zipfile
from types import SimpleNamespace

from app.routers import predict
from app.services import file_manager
from app.services.file_manager import TaskState


def _task(monkeypatch, tmp_path) -> TaskState:
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path / "tasks"))
    return TaskState(task_id="t1")


def _events(task: TaskState, monkeypatch) -> list[dict]:
    monkeypatch.setattr(predict, "get_task", lambda task_id: task)

    async def collect() -> list[dict]:
        response = await predict.batch_progress("t1")
        return [json.loads(chunk.removeprefix("data: ")) async for chunk in response.body_iterator]

    return asyncio.run(asyncio.wait_for(collect(), timeout=5))


def test_a_corrupt_zip_ends_the_task_as_an_error(monkeypatch, tmp_path):
    task = _task(monkeypatch, tmp_path)
    bad = tmp_path / "upload.zip"
    bad.write_bytes(b"PK\x03\x04 this is not a zip")

    predict._process_batch(task, str(bad), 0.25, 0.3, [])

    assert task.status == "error"
    assert task.message.startswith("Batch processing failed:")


def test_the_progress_stream_reports_the_error_and_ends(monkeypatch, tmp_path):
    task = _task(monkeypatch, tmp_path)
    task.status, task.message = "error", "Batch processing failed: File is not a zip file"

    events = _events(task, monkeypatch)

    assert [event["status"] for event in events] == ["error"]
    assert events[0]["message"] == task.message


def test_a_zip_without_images_completes_empty(monkeypatch, tmp_path):
    task = _task(monkeypatch, tmp_path)
    upload = tmp_path / "upload.zip"
    with zipfile.ZipFile(upload, "w") as archive:
        archive.writestr("notes.txt", "no images here")

    predict._process_batch(task, str(upload), 0.25, 0.3, [])

    assert (task.status, task.total, task.progress) == ("completed", 0, 0)


def test_a_stuck_task_does_not_hold_the_stream_forever(monkeypatch, tmp_path):
    task = _task(monkeypatch, tmp_path)
    readings = iter([0.0, 1.0, 7200.0])  # start, first check, after the first wait
    monkeypatch.setattr(predict, "time", SimpleNamespace(monotonic=lambda: next(readings)))
    monkeypatch.setattr(predict.settings, "task_ttl_minutes", 60)

    events = _events(task, monkeypatch)

    assert [event["status"] for event in events] == ["pending"]
