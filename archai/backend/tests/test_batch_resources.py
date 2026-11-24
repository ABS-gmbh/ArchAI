"""Batches run outside FastAPI's threadpool, extract under their task directory and drop their upload (security review S3)."""

from __future__ import annotations

import io
import threading
import zipfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import predict
from app.services import file_manager
from app.services.file_manager import TaskState


def test_a_batch_runs_in_its_own_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path / "tasks"))
    ran = threading.Event()
    seen: list[str] = []

    def record(*_args: object) -> None:
        seen.append(threading.current_thread().name)
        ran.set()

    monkeypatch.setattr(predict, "_process_batch", record)
    app = FastAPI()
    app.include_router(predict.router)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("p.png", b"x")
    response = TestClient(app).post("/predict/batch", files={"zip_file": ("b.zip", archive.getvalue(), "application/zip")})
    assert response.status_code == 202
    assert ran.wait(5)
    assert seen[0].startswith("archai-batch")  # not FastAPI's threadpool (AnyIO worker threads)


def test_extraction_happens_under_the_task_directory_and_the_upload_is_removed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path / "tasks"))
    monkeypatch.setattr(predict, "save_task_state", lambda _task: None)
    task = TaskState(task_id="t1")
    task_dir = Path(predict.get_task_dir("t1"))
    upload = task_dir / "upload.zip"
    with zipfile.ZipFile(upload, "w") as zf:
        zf.writestr("notes.txt", b"no images here")
    seen: list[str] = []
    real = predict._safe_extract_zip
    monkeypatch.setattr(predict, "_safe_extract_zip", lambda zip_path, extract_dir: (seen.append(extract_dir), real(zip_path, extract_dir)))
    predict._process_batch(task, str(upload), 0.25, 0.3, [])
    assert seen and Path(seen[0]).parent == task_dir
    assert not Path(seen[0]).exists()
    assert not upload.exists()


def _post_batch(client: TestClient) -> object:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("p.png", b"x")
    return client.post("/predict/batch", files={"zip_file": ("b.zip", archive.getvalue(), "application/zip")})


def test_batches_past_the_active_limit_are_refused_and_leave_nothing_on_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Second review, D: twelve batches from one client all queued, uploads on
    # disk, and another user's batch waited behind them without bound.
    tasks = tmp_path / "tasks"
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tasks))
    monkeypatch.setattr(predict.settings, "max_active_batches", 2)
    monkeypatch.setattr(predict, "save_task_state", lambda _task: None)
    release = threading.Event()
    started = threading.Semaphore(0)

    def slow(*_args: object) -> None:
        started.release()
        release.wait(10)

    monkeypatch.setattr(predict, "_run_batch", slow)
    app = FastAPI()
    app.include_router(predict.router)
    client = TestClient(app)
    try:
        accepted = [_post_batch(client) for _ in range(2)]
        assert [r.status_code for r in accepted] == [202, 202]
        before = sorted(p.name for p in tasks.iterdir())
        refused = _post_batch(client)
        assert refused.status_code == 429
        assert "MAX_ACTIVE_BATCHES" in refused.json()["detail"]
        assert refused.headers.get("retry-after") == "60"
        assert sorted(p.name for p in tasks.iterdir()) == before  # no task directory, no upload kept
    finally:
        release.set()
    for _ in range(2):
        assert started.acquire(timeout=5)
    for _ in range(100):  # the two batches end and release their places
        if not file_manager._held:
            break
        threading.Event().wait(0.05)
    assert _post_batch(client).status_code == 202


def test_the_sweep_keeps_a_queued_batch_however_long_it_waits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Second review, D: after 61 minutes in the queue the TTL sweep deleted the
    # waiting batch's task and upload; it then failed with FileNotFoundError.
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path / "tasks"))
    monkeypatch.setattr(file_manager.settings, "task_ttl_minutes", 60)
    waiting = file_manager.create_task()
    finished = file_manager.create_task()
    upload = Path(file_manager.get_task_dir(waiting.task_id)) / "upload.zip"
    upload.write_bytes(b"PK")
    assert file_manager.hold_task(waiting.task_id)
    for task in (waiting, finished):
        task.created_at -= 61 * 60
    file_manager.cleanup_expired_tasks()
    assert upload.exists() and file_manager.get_task(waiting.task_id) is waiting
    assert file_manager.get_task(finished.task_id) is None
    file_manager.release_task(waiting.task_id)
    file_manager.cleanup_expired_tasks()
    assert not upload.exists()
