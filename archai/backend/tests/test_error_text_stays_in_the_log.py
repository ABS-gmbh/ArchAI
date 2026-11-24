"""Exception text (absolute paths, module names, hosts) stays in the log, not the response (security review C8)."""

from __future__ import annotations

import base64
import io
import logging
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from app.config import settings

SECRET_DIR = "/srv/archai-private/models/calamari"


def _page_b64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_a_backend_that_is_not_installed_does_not_name_its_path(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from app.agents import ocr_agent
    from app.schemas.agents_ocr import OCRExtractRequest

    for name in ("calamari_default_model_dir", "calamari_gt4histocr_model_dir", "calamari_models_dir"):
        monkeypatch.setattr(settings, name, SECRET_DIR)
    monkeypatch.setattr(ocr_agent, "write_ocr_evidence_jsonl", lambda _record: None)
    payload = OCRExtractRequest(
        image_b64=_page_b64(), region={"bbox_xyxy": [0, 0, 60, 60]}, options={"backend": "calamari", "apply_proofread": False}
    )
    with caplog.at_level(logging.WARNING):
        try:
            result = ocr_agent.run_ocr_extraction(payload, client=object())  # the provider is never asked
            text = result.model_dump_json()
        except Exception as exc:  # noqa: BLE001 - the error, if any, is what the client would see
            text = str(exc)
    assert SECRET_DIR not in text
    assert "calamari is not installed or not configured" in text
    assert SECRET_DIR in caplog.text  # the path is in the log


def test_a_run_event_names_the_failure_not_its_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from app.db import pipeline_db
    from app.services import rag_store, trace_pipeline

    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    run_id = pipeline_db.create_run(asset_ref="eval:x", asset_sha256="x")

    def fail(_run_id: str) -> dict:
        raise ModuleNotFoundError(f"No module named 'chromadb' (searched {SECRET_DIR})")

    monkeypatch.setattr(rag_store, "index_run", fail)
    trace_pipeline._auto_index_run(run_id)
    messages = [event["message"] for event in pipeline_db.list_events(run_id)]
    assert any("Auto-indexing failed (ModuleNotFoundError)" in m for m in messages)
    assert not any(SECRET_DIR in m or "chromadb" in m for m in messages)


def test_a_failed_batch_names_the_failure_not_its_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from app.routers import predict
    from app.services.file_manager import TaskState

    def explode(*_args: object, **_kwargs: object) -> None:
        raise OSError(f"[Errno 13] Permission denied: '{SECRET_DIR}/upload.zip'")

    monkeypatch.setattr(predict, "_run_batch", explode)
    monkeypatch.setattr(predict, "save_task_state", lambda _task: None)
    task = TaskState(task_id="t1")
    predict._process_batch(task, "unused.zip", 0.25, 0.45, [])
    assert task.status == "error" and SECRET_DIR not in task.message and "(OSError)" in task.message
    # A limit the archive broke is still said as it is.
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for i in range(3):
            zf.writestr(f"{i}.png", b"")
    monkeypatch.setattr(predict.settings, "max_zip_members", 2)
    monkeypatch.setattr(predict, "_run_batch", lambda *a, **k: predict._safe_extract_zip(str(archive), str(tmp_path)))
    predict._process_batch(task, str(archive), 0.25, 0.45, [])
    assert "at most 2 are accepted" in task.message


def test_the_chat_model_list_does_not_carry_the_provider_address(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import chat_ai

    monkeypatch.setattr(chat_ai, "_create_client", lambda: object())
    monkeypatch.setattr(chat_ai, "_raw_server_models", lambda _client: ["qwen3-30b-a3b-instruct-2507"])
    assert "base_url" not in chat_ai.list_available_models()
