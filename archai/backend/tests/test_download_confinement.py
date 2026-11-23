"""Downloads serve only files inside their own task directory.

A task's paths come from its state, read back from state.json after a
restart. When uploads kept the client's file name, an upload called
state.json could replace that file, and the annotated-image downloads served
whatever path it named: any file the server could read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.routers import download
from app.services import file_manager

TASK_ID = "abcdef123456"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path / "tasks"))
    monkeypatch.setattr(file_manager, "_task_store", {})
    app = FastAPI()
    app.include_router(download.router)
    return TestClient(app)


def _forge_state(tmp_path: Path, **fields) -> Path:
    """A state.json as a malicious upload could have written it, read back after a restart."""
    secret = tmp_path / "secret.txt"
    secret.write_text("server-side secret", encoding="utf-8")
    task_dir = tmp_path / "tasks" / TASK_ID
    task_dir.mkdir(parents=True)
    state = {"task_id": TASK_ID, "status": "completed", **{k: v.format(secret=secret) if isinstance(v, str) else v for k, v in fields.items()}}
    (task_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return secret


def test_a_forged_annotated_image_path_is_refused(client: TestClient, tmp_path: Path) -> None:
    _forge_state(tmp_path, annotated_image_path="{secret}")
    response = client.get(f"/download/{TASK_ID}/annotated_image")
    assert response.status_code == 404
    assert "server-side secret" not in response.text


def test_a_forged_gallery_path_is_refused(client: TestClient, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    _forge_state(tmp_path, gallery=[{"filename": "x.jpg", "annotated_path": str(secret)}])
    assert client.get(f"/download/{TASK_ID}/annotated/0").status_code == 404
    response = client.get(f"/download/{TASK_ID}/results_zip")
    assert b"server-side secret" not in response.content


def test_a_real_annotated_image_is_still_served(client: TestClient, tmp_path: Path) -> None:
    task_dir = tmp_path / "tasks" / TASK_ID
    task_dir.mkdir(parents=True)
    image = task_dir / "annotated.jpg"
    Image.new("RGB", (20, 10), "white").save(image)
    state = {"task_id": TASK_ID, "status": "completed", "annotated_image_path": str(image)}
    (task_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    response = client.get(f"/download/{TASK_ID}/annotated_image")
    assert response.status_code == 200 and response.content[:2] == b"\xff\xd8"
