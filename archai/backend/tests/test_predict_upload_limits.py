"""/predict uploads are capped and batch ZIPs cannot inflate without bound (audit PRED-002).

Both routes read the whole upload into memory before any check, and
_safe_extract_zip inflated each member with one src.read(): a small ZIP
whose member expands to gigabytes was enough to exhaust memory.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import predict
from app.services import file_manager
from app.services.file_manager import TaskState

MB = 1 << 20


@pytest.fixture
def client(monkeypatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path / "tasks"))
    monkeypatch.setattr(predict.settings, "max_upload_megabytes", 1)
    monkeypatch.setattr(predict, "_process_batch", lambda *args, **kwargs: None)
    app = FastAPI()
    app.include_router(predict.router)
    return TestClient(app)


def _uploads(tmp_path: Path) -> list[Path]:
    return [p for p in (tmp_path / "tasks").rglob("*") if p.is_file()]


def test_an_upload_over_the_limit_is_refused_and_not_kept(client, tmp_path):
    big = io.BytesIO(b"\0" * (2 * MB))
    response = client.post("/predict/batch", files={"zip_file": ("pages.zip", big, "application/zip")})
    assert response.status_code == 413
    assert _uploads(tmp_path) == []


def test_an_empty_upload_is_still_refused(client, tmp_path):
    response = client.post("/predict/batch", files={"zip_file": ("pages.zip", io.BytesIO(b""), "application/zip")})
    assert response.status_code == 422


def test_an_upload_within_the_limit_is_saved_whole(client, tmp_path):
    data = b"PK" + b"x" * (MB // 2)
    response = client.post("/predict/batch", files={"zip_file": ("pages.zip", io.BytesIO(data), "application/zip")})
    assert response.status_code == 202
    [saved] = _uploads(tmp_path)
    assert saved.read_bytes() == data


def _bomb(path: Path, megabytes: int) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("page.png", b"\0" * (megabytes * MB))


def test_extraction_stops_once_past_the_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(predict.settings, "max_extracted_megabytes", 1)
    _bomb(tmp_path / "bomb.zip", 3)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="expands past 1 MB"):
        predict._safe_extract_zip(str(tmp_path / "bomb.zip"), str(out))
    assert sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) <= MB


def test_a_zip_bomb_ends_its_batch_as_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(predict.settings, "max_extracted_megabytes", 1)
    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path / "tasks"))
    _bomb(tmp_path / "bomb.zip", 3)
    task = TaskState(task_id="t1")
    predict._process_batch(task, str(tmp_path / "bomb.zip"), 0.25, 0.3, [])
    assert task.status == "error"
    assert "expands past" in task.message


# ── the single-page route (audit PRED-003) ──────────────────────────────


def test_a_single_page_is_prepared_off_the_event_loop(client, monkeypatch):
    """Audit PRED-003: decoding and resizing a large page ran on the event loop."""
    import threading

    from PIL import Image

    threads: dict[str, str] = {}
    original = predict._prepare_image_for_segmentation

    def recording(path: str):
        threads["prepare"] = threading.current_thread().name
        return original(path)

    @client.app.middleware("http")
    async def record_loop_thread(request, call_next):
        threads["loop"] = threading.current_thread().name
        return await call_next(request)

    monkeypatch.setattr(predict, "_prepare_image_for_segmentation", recording)
    monkeypatch.setattr(predict, "run_single_segmentation",
                        lambda *a, **k: ({"images": [], "annotations": [], "categories": []}, {}))
    page = io.BytesIO()
    Image.new("RGB", (60, 40), "white").save(page, format="PNG")
    page.seek(0)

    response = client.post("/predict/single", files={"image": ("page.png", page, "image/png")})

    assert response.status_code == 200
    assert threads["prepare"] != threads["loop"]
