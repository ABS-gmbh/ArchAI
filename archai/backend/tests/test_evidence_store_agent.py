from __future__ import annotations

from pathlib import Path


from app.agents import evidence_store_agent  # type: ignore[import-untyped]
from app.db import pipeline_db  # type: ignore[import-untyped]
from app.schemas.agents_ocr import EvidenceSpanCreateRequest  # type: ignore[import-untyped]


def test_evidence_store_agent_persists_spans_in_sqlite(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    run_id = pipeline_db.create_run(asset_ref="page-1", asset_sha256="sha")
    pipeline_db.insert_chunks(
        run_id,
        [
            {
                "chunk_id": "chunk-1",
                "idx": 0,
                "start_offset": 0,
                "end_offset": 6,
                "text": "Arthur",
            }
        ],
    )
    pipeline_db.insert_entity_mentions(
        run_id,
        [
            {
                "mention_id": "mention-1",
                "chunk_id": "chunk-1",
                "start_offset": 0,
                "end_offset": 6,
                "surface": "Arthur",
                "norm": "arthur",
                "label": "PERSON",
                "ent_type": "person",
                "confidence": 0.8,
                "method": "test",
            }
        ],
    )

    created = evidence_store_agent.create_span(
        EvidenceSpanCreateRequest(
            run_id=run_id,
            asset_ref="page-1",
            page_id="page-1",
            region_id="region-1",
            mention_id="mention-1",
            text="Arthur",
            raw_text="Arthur",
            normalized_text="arthur",
            start_offset=0,
            end_offset=6,
            bbox_xyxy=[10, 20, 30, 40],
            model_used="glm-ocr:latest",
        )
    )

    rows = evidence_store_agent.list_spans("page-1")

    assert created.page_id == "page-1"
    assert created.region_id == "region-1"
    assert len(rows) == 1
    assert rows[0].span_id == created.span_id
    assert rows[0].raw_text == "Arthur"
    assert rows[0].bbox_xyxy == [10.0, 20.0, 30.0, 40.0]


def test_a_span_for_an_unknown_run_is_a_422_not_a_500() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.routers import evidence

    app = FastAPI()
    app.include_router(evidence.router)
    client = TestClient(app, raise_server_exceptions=False)
    assert client.post("/evidence/spans", json={}).status_code == 422
    response = client.post("/evidence/spans", json={"run_id": "no-such-run", "text": "Arthur"})
    assert response.status_code == 422
    assert "run_id" in response.json()["detail"]
