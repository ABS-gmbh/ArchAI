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
