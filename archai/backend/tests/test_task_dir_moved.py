"""A finished layout task's images are still served after its directory moves to a new TASK_BASE_DIR.

Found by an upgrade drill: a task's state.json stores the absolute paths of
its annotated images. TASK_BASE_DIR is new, and an operator who sets it and
moves .tasks/ there finds every download of a finished task's images
answering 404 "Annotated image file not found.", because the stored path
still names the old directory, and a path outside the task's directory is
refused. (The old code served it from the old location, as long as the file
was still there.)
"""

from __future__ import annotations

import io
import shutil
import zipfile

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.routers import download
from app.services import file_manager
from app.services.file_manager import TaskState, get_task, save_task_state

TASK_ID = "0123456789ab"


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def _task_in(base, monkeypatch) -> None:
    """A finished single-page and batch task, saved under *base*."""
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(base))
    directory = base / TASK_ID
    (directory / "annotated").mkdir(parents=True)
    (directory / "annotated.jpg").write_bytes(_jpeg())
    (directory / "annotated" / "p1.jpg").write_bytes(_jpeg())
    save_task_state(TaskState(
        task_id=TASK_ID, status="completed", coco_json={"images": [], "annotations": []},
        annotated_image_path=str(directory / "annotated.jpg"),
        gallery=[{"filename": "p1.png", "annotated_path": str(directory / "annotated" / "p1.jpg")}],
        batch_coco_list=[{"images": [], "annotations": []}],
    ))


def _moved(tmp_path, monkeypatch) -> TestClient:
    _task_in(tmp_path / "old-tasks", monkeypatch)
    shutil.move(str(tmp_path / "old-tasks"), str(tmp_path / "new-tasks"))
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path / "new-tasks"))
    monkeypatch.setattr(file_manager, "_task_store", {})  # a restart
    app = FastAPI()
    app.include_router(download.router)
    return TestClient(app)


def test_the_single_page_image_is_served_from_the_new_directory(monkeypatch, tmp_path):
    client = _moved(tmp_path, monkeypatch)
    response = client.get(f"/download/{TASK_ID}/annotated_image")
    assert response.status_code == 200
    assert response.content == (tmp_path / "new-tasks" / TASK_ID / "annotated.jpg").read_bytes()


def test_the_batch_images_and_zip_are_served_from_the_new_directory(monkeypatch, tmp_path):
    client = _moved(tmp_path, monkeypatch)
    assert client.get(f"/download/{TASK_ID}/annotated/0").status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(client.get(f"/download/{TASK_ID}/results_zip").content))
    assert "images/p1.png" in archive.namelist()


def test_the_paths_are_read_back_under_the_new_directory(monkeypatch, tmp_path):
    _moved(tmp_path, monkeypatch)
    task = get_task(TASK_ID)
    assert task is not None
    assert task.annotated_image_path == str(tmp_path / "new-tasks" / TASK_ID / "annotated.jpg")
    assert task.gallery[0]["annotated_path"] == str(tmp_path / "new-tasks" / TASK_ID / "annotated" / "p1.jpg")


def test_a_path_that_names_no_task_directory_is_left_alone(monkeypatch, tmp_path):
    """And is still refused by the downloads: only a file inside the task's directory is served."""
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path / "tasks"))
    save_task_state(TaskState(task_id=TASK_ID, status="completed", annotated_image_path="/etc/hosts"))
    monkeypatch.setattr(file_manager, "_task_store", {})
    assert get_task(TASK_ID).annotated_image_path == "/etc/hosts"
    app = FastAPI()
    app.include_router(download.router)
    assert TestClient(app).get(f"/download/{TASK_ID}/annotated_image").status_code == 404
