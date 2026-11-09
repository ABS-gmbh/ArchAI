"""A finished task is still reachable after a restart (audit TASK-02).

Task state lived only in a dict in the worker process. A restart emptied it
while the task directories stayed on disk, so every download of a finished
task answered 404 "Task not found" though its files were there.
"""

from __future__ import annotations

import io
import os

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.routers import download, predict
from app.services import file_manager
from app.services.file_manager import TaskState, get_task, save_task_state


def _restart(monkeypatch) -> None:
    monkeypatch.setattr(file_manager, "_task_store", {})


def test_a_saved_task_is_read_back_after_a_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path / "tasks"))
    task = TaskState(task_id="0123456789ab", status="completed", total=2, progress=2,
                     coco_json={"images": [], "annotations": []}, gallery=[{"filename": "a.jpg", "annotated_path": "/x"}])
    save_task_state(task)
    _restart(monkeypatch)

    again = get_task("0123456789ab")

    assert again is not None
    assert (again.status, again.total, again.coco_json, again.gallery) == (
        "completed", 2, {"images": [], "annotations": []}, [{"filename": "a.jpg", "annotated_path": "/x"}]
    )
    assert not os.path.exists(tmp_path / "tasks" / "0123456789ab" / ".state.json.tmp")


def test_an_unknown_or_unsafe_id_is_not_read(monkeypatch, tmp_path):
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path / "tasks"))
    assert get_task("ffffffffffff") is None
    assert get_task("../../etc") is None


def test_a_single_page_download_works_after_a_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path / "tasks"))

    def segment(path, *, annotated_output_path, **kwargs):
        Image.new("RGB", (8, 8), "white").save(annotated_output_path)
        return {"images": [], "annotations": [], "categories": []}, {}

    monkeypatch.setattr(predict, "run_single_segmentation", segment)
    app = FastAPI()
    app.include_router(predict.router)
    app.include_router(download.router)
    client = TestClient(app)
    page = io.BytesIO()
    Image.new("RGB", (60, 40), "white").save(page, format="PNG")
    task_id = client.post("/predict/single", files={"image": ("p.png", page.getvalue(), "image/png")}).json()["task_id"]
    _restart(monkeypatch)

    assert client.get(f"/download/{task_id}/annotated_image").status_code == 200
    assert client.get(f"/download/{task_id}/coco_json").status_code == 200
