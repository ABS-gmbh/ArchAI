"""Every backend response carries X-Content-Type-Options: nosniff.

None did. A browser may then sniff a response's type instead of trusting its
Content-Type: a downloaded COCO file, an annotated image or an error body
could be rendered as something else. Streaming answers keep streaming: the
header is added to the start of the response, not by buffering it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
from app.config import settings
from app.services import file_manager


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(settings, "api_token", "")
    return TestClient(main.app)


def test_json_answers_carry_nosniff(client: TestClient) -> None:
    for path, status in (("/api/health", None), ("/api/classes", 200), ("/api/nope", 404)):
        response = client.get(path)
        if status is not None:
            assert response.status_code == status
        assert response.headers.get("x-content-type-options") == "nosniff", path


def test_a_refused_body_carries_nosniff(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_upload_megabytes", 1)
    response = client.post("/api/ocr/extract_full_page", content=b"x" * (3 << 20), headers={"content-type": "application/json"})
    assert response.status_code == 413
    assert response.headers.get("x-content-type-options") == "nosniff"


def test_downloads_carry_nosniff(client: TestClient) -> None:
    task = file_manager.create_task()
    image = Path(file_manager.TASK_BASE_DIR) / task.task_id / "annotated.jpg"
    Image.new("RGB", (8, 8), "white").save(image, format="JPEG")
    task.annotated_image_path = str(image)
    task.coco_json = {"images": [], "annotations": []}

    picture = client.get(f"/api/download/{task.task_id}/annotated_image")
    assert picture.status_code == 200
    assert picture.headers["content-type"] == "image/jpeg"
    assert picture.headers.get("x-content-type-options") == "nosniff"

    coco = client.get(f"/api/download/{task.task_id}/coco_json")
    assert coco.status_code == 200
    assert json.loads(coco.content) == {"images": [], "annotations": []}
    assert coco.headers.get("x-content-type-options") == "nosniff"


def test_a_streamed_answer_carries_nosniff_and_still_streams(client: TestClient) -> None:
    batch = client.post("/api/predict/batch", files={"zip_file": ("b.zip", b"not a zip", "application/zip")})
    assert batch.status_code == 202
    with client.stream("GET", f"/api/predict/batch/{batch.json()['task_id']}/progress") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers.get("x-content-type-options") == "nosniff"
        assert any(line.startswith("data:") for line in response.iter_lines())
