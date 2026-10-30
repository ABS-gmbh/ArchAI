"""The batch results ZIP is built on disk off the event loop (audit DL-001).

download_results_zip built the whole archive in a BytesIO inside the async
handler: a large batch held hundreds of megabytes per download and compressed
on the event loop thread, stalling every other request.
"""

from __future__ import annotations

import io
import threading
import zipfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.routers import download
from app.services import file_manager
from app.services.file_manager import TaskState


def _client_with_task(monkeypatch, tmp_path: Path) -> tuple[TestClient, Path]:
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path / "tasks"))
    task_dir = Path(file_manager.get_task_dir("t1"))
    gallery = []
    for index, name in enumerate(["f1.jpg", "f2.jpg"]):
        path = task_dir / f"annotated_{index}.jpg"
        Image.new("RGB", (40, 30), "white").save(path)
        gallery.append({"filename": name, "annotated_path": str(path)})
    task = TaskState(task_id="t1", status="completed", gallery=gallery, coco_json={"images": [], "annotations": []})
    monkeypatch.setitem(file_manager._task_store, "t1", task)
    app = FastAPI()
    app.include_router(download.router)
    return TestClient(app), task_dir


def test_the_zip_holds_the_images_and_annotations_and_is_removed_after(monkeypatch, tmp_path):
    client, task_dir = _client_with_task(monkeypatch, tmp_path)

    response = client.get("/download/t1/results_zip")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert 'filename="results_t1.zip"' in response.headers["content-disposition"]
    names = zipfile.ZipFile(io.BytesIO(response.content)).namelist()
    assert sorted(names) == ["annotations.json", "images/f1.jpg", "images/f2.jpg"]
    assert not list(task_dir.glob("results_*.zip"))


def test_the_zip_is_written_off_the_event_loop(monkeypatch, tmp_path):
    client, _ = _client_with_task(monkeypatch, tmp_path)
    threads: list[str] = []
    original = download._write_results_zip

    def recording(*args, **kwargs):
        threads.append(threading.current_thread().name)
        return original(*args, **kwargs)

    monkeypatch.setattr(download, "_write_results_zip", recording)
    loop_threads: list[str] = []

    @client.app.middleware("http")
    async def record_loop_thread(request, call_next):
        loop_threads.append(threading.current_thread().name)
        return await call_next(request)

    assert client.get("/download/t1/results_zip").status_code == 200
    assert threads and threads[0] != loop_threads[0]
