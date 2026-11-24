"""The trace tables and a page's evidence spans are read with a limit (security review S3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from app.db import pipeline_db

    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    return pipeline_db


def test_trace_tables_return_at_most_limit_rows_each(db: Any) -> None:
    from app.routers import ocr as ocr_router

    run_id = db.create_run(asset_ref="page", asset_sha256=None)
    for i in range(12):
        db.log_event(run_id, "STAGE", "INFO", f"event {i}")
    app = FastAPI()
    app.include_router(ocr_router.router)
    client = TestClient(app)
    tables = {t["table"]: t for t in client.get(f"/ocr/trace/{run_id}/tables", params={"limit": 5}).json()["tables"]}
    assert len(tables["pipeline_events"]["rows"]) == 5
    assert [row[tables["pipeline_events"]["columns"].index("message")] for row in tables["pipeline_events"]["rows"]][0] == "event 0"
    default = {t["table"]: t for t in client.get(f"/ocr/trace/{run_id}/tables").json()["tables"]}
    assert len(default["pipeline_events"]["rows"]) == 12  # under the default limit
    assert client.get(f"/ocr/trace/{run_id}/tables", params={"limit": 0}).status_code == 422


def test_evidence_spans_return_at_most_limit_rows(db: Any) -> None:
    from app.routers import evidence

    run_id = db.create_run(asset_ref="page", asset_sha256=None)
    app = FastAPI()
    app.include_router(evidence.router)
    client = TestClient(app)
    for i in range(7):
        assert client.post("/evidence/spans", json={"run_id": run_id, "page_id": "p1", "text": f"span {i}"}).status_code == 200
    assert len(client.get("/evidence/spans/p1", params={"limit": 3}).json()) == 3
    assert len(client.get("/evidence/spans/p1").json()) == 7
