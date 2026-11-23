"""The post-OCR stages run once, in trace_pipeline.run_downstream_stages (audit OCR-011).

The sequence (analysis, mention recall, linking, consolidated report, search
decision, index, DONE) existed three times and had drifted: only the
full-page copy returned authority_report. These pin the shared function's
contract, and pin the trace route's stage events, which must not change.
"""

from __future__ import annotations

import asyncio
import types
from pathlib import Path
from typing import Any

import pytest

from app.db import pipeline_db
from app.services import trace_pipeline

KEYS = {
    "chunks_count",
    "mentions_count",
    "top_mentions",
    "mention_report",
    "mention_recall",
    "linking_result",
    "authority_report",
    "consolidated_report",
}
MENTION = {"mention_id": "m1", "start_offset": 0, "end_offset": 6, "surface": "Arthur"}


@pytest.fixture
def stubs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    calls: dict[str, Any] = {"high_recall": 0, "linked": 0, "indexed": 0, "search": None}
    monkeypatch.setattr(trace_pipeline, "_run_trace_analysis", lambda run_id, text: ([{"idx": 0}], [dict(MENTION)], [], {}))
    monkeypatch.setattr(
        trace_pipeline,
        "check_mention_recall",
        lambda text, count, label: {"mention_recall_ok": False, "reason": "recall low", "trigger_high_recall": True},
    )

    def high_recall(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        calls["high_recall"] += 1
        return []

    def link(run_id: str) -> dict[str, Any]:
        calls["linked"] += 1
        return {"linked_total": 1}

    def index(run_id: str) -> None:
        calls["indexed"] += 1

    def search(run_id: str, allowed: bool) -> None:
        calls["search"] = allowed

    monkeypatch.setattr(trace_pipeline, "extract_high_recall_mentions", high_recall)
    monkeypatch.setattr(trace_pipeline, "_run_authority_linking_stage", link)
    monkeypatch.setattr(trace_pipeline, "_auto_index_run", index)
    monkeypatch.setattr(trace_pipeline, "_record_search_decision", search)
    monkeypatch.setattr(trace_pipeline, "_build_mention_extraction_report", lambda *args, **kwargs: "MENTIONS")
    monkeypatch.setattr(trace_pipeline, "_build_consolidated_report", lambda *args, **kwargs: "CONSOLIDATED")
    monkeypatch.setattr(trace_pipeline, "_persist_deferred_authority_links", lambda run_id, reason: ({"deferred": True}, "DEFERRED REPORT"))
    import app.services.authority_linking as authority_linking

    monkeypatch.setattr(authority_linking, "build_linking_report_from_db", lambda run_id: "LINKING REPORT")
    return calls


def _run(gates: dict[str, Any], **options: Any) -> tuple[str, dict[str, Any]]:
    run_id = pipeline_db.create_run(asset_ref="page-1")
    result = trace_pipeline.run_downstream_stages(
        run_id,
        "page-1",
        base_text="Arthur li rois",
        base_text_source="proofread",
        recall_text="Arthur li rois",
        stored_proofread_text="Arthur li rois",
        quality_label="OK",
        gate_decisions=gates,
        **options,
    )
    return run_id, result


def test_every_path_returns_one_key_set_with_the_authority_report(stubs: dict[str, Any]) -> None:
    _, linked = _run({"ner_allowed": True, "token_search_allowed": True}, high_recall=True)
    _, deferred = _run({"ner_allowed": True, "token_search_allowed": False})
    _, degraded = _run({"ner_allowed": False, "token_search_allowed": False})
    assert set(linked) == set(deferred) == set(degraded) == KEYS
    assert linked["authority_report"] == "LINKING REPORT"
    assert deferred["authority_report"] == "DEFERRED REPORT"
    assert stubs["linked"] == 1 and stubs["indexed"] == 1


def test_high_recall_and_recall_warnings_only_where_asked(stubs: dict[str, Any]) -> None:
    payload = {"warnings": ["earlier"]}
    _run({"ner_allowed": True, "token_search_allowed": True}, high_recall=True, ocr_payload=payload)
    assert stubs["high_recall"] == 1
    assert payload["warnings"] == ["earlier", "recall low"]
    _run({"ner_allowed": True, "token_search_allowed": True})
    assert stubs["high_recall"] == 1


def test_a_curated_fixture_skips_linking_and_search(stubs: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        trace_pipeline, "_run_curated_fixture_analysis", lambda run_id, text, mentions: ([{"idx": 0}], [dict(MENTION)], [], {})
    )
    fixture = types.SimpleNamespace(semantic_mentions=("x",), wait_seconds=0)
    run_id, result = _run({"ner_allowed": True, "token_search_allowed": True}, fixture=fixture)
    assert stubs["linked"] == 0 and stubs["indexed"] == 0 and stubs["search"] is False
    assert "status: skipped" in result["authority_report"]
    stages = [(e["stage"], e["event"]) for e in pipeline_db.list_events(run_id)]
    assert ("LINKING_SKIPPED", "INFO") in stages and ("INDEX_SKIPPED", "INFO") in stages


def test_the_stage_events_of_a_run(stubs: dict[str, Any]) -> None:
    """The events the trace UI shows, unchanged by the move into one function."""
    run_id, _ = _run({"ner_allowed": True, "token_search_allowed": True}, high_recall=True)
    events = [(e["stage"], e["event"], e["message"]) for e in pipeline_db.list_events(run_id)]
    assert events == [
        ("ANALYZE_RUNNING", "START", "Running chunk/entity analysis."),
        ("QUALITY_GATES", "INFO", "recall low"),
        ("ANALYZE_DONE", "END", "chunks=1, mentions=1"),
        ("STORED", "END", "Run results persisted in SQLite."),
        ("DONE", "END", "Pipeline completed successfully."),
    ]


def test_the_segmented_trace_route_returns_the_authority_report(stubs: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Additive: only the full-page route returned it before."""
    from app.routers import ocr as ocr_router
    from app.schemas.agents_ocr import SaiaFullPageExtractRequest, SaiaOCRLocationSuggestion
    from tests.test_ocr_extract_full_page_api import SAMPLE_B64, _FakeStructuredOcrAgent

    monkeypatch.setattr(trace_pipeline, "_get_ocr_agent", lambda: _FakeStructuredOcrAgent())
    payload = SaiaFullPageExtractRequest(
        document_id="doc-1",
        page_id="page-1",
        image_b64=SAMPLE_B64,
        apply_proofread=True,
        location_suggestions=[
            SaiaOCRLocationSuggestion(region_id="r1", category="Main script black", bbox_xywh=[10.0, 20.0, 100.0, 80.0])
        ],
    )
    result = asyncio.run(ocr_router.ocr_page_with_trace(payload))
    assert result["status"] == "COMPLETED"
    assert "authority_report" in result
