"""A batch ZIP cannot overwrite the batch task's own files.

The single-page upload could replace its task's state.json, which the
downloads trust after a restart. The batch route extracts into a fresh
temporary directory, outside the task directory, reads only image files from
it, and names its outputs by index; this pins that down.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from app.routers import predict
from app.services import file_manager
from app.services.file_manager import TaskState, save_task_state

FORGED = json.dumps({"task_id": "x", "status": "completed", "annotated_image_path": "/etc/hosts"}).encode()


def _png(colour: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (30, 20), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def test_zip_members_never_land_in_the_task_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path / "tasks"))
    monkeypatch.setattr(predict, "run_models_parallel", lambda path, out, **kwargs: {})
    monkeypatch.setattr(
        predict, "combine_and_filter_predictions", lambda path, folders: {"images": [], "annotations": [], "categories": []}
    )

    def draw(path, coco, classes, output_path):
        Path(output_path).write_bytes(b"server output")
        return output_path

    monkeypatch.setattr(predict, "draw_coco_on_image", draw)

    task = TaskState(task_id="abcdef123456")
    task_dir = Path(file_manager.get_task_dir(task.task_id))
    zip_path = task_dir / "upload.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("state.json", FORGED)
        archive.writestr("annotations.json", b"forged")
        archive.writestr("../state.json", FORGED)
        archive.writestr(f"../{task.task_id}/state.json", FORGED)
        archive.writestr("annotated_0.jpg", _png("black"))
        archive.writestr("page.png", _png("white"))

    predict._run_batch(task, str(zip_path), 0.25, 0.3, predict.FINAL_CLASSES)
    save_task_state(task)

    assert sorted(os.listdir(task_dir)) == [
        "annotated_0.jpg",
        "annotated_1.jpg",
        "annotations.json",
        "state.json",
        "upload.zip",
    ]
    assert json.loads((task_dir / "annotations.json").read_text())["images"] == []  # the server's merge, not b"forged"
    state = json.loads((task_dir / "state.json").read_text())
    assert state["task_id"] == task.task_id and state["status"] == "completed"
    assert all(entry["annotated_path"].startswith(str(task_dir)) for entry in state["gallery"])
    assert (task_dir / "annotated_0.jpg").read_bytes() == b"server output"
    assert not (tmp_path / "tasks" / "state.json").exists()
