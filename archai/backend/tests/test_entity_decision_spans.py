"""One entity decision per span, so a page's analysis is not lost to a FOREIGN KEY error."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_backend = Path(__file__).resolve().parents[1]
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from app.db import pipeline_db  # type: ignore[import-untyped]  # noqa: E402
from app.db.pipeline_db import merge_span_decisions  # type: ignore[import-untyped]  # noqa: E402
from app.routers import ocr as ocr_router  # type: ignore[import-untyped]  # noqa: E402
from app.schemas.agents_ocr import SaiaFullPageExtractRequest, SaiaFullPageExtractResponse  # type: ignore[import-untyped]  # noqa: E402

READINGS = _backend / "eval" / "quality_gate" / "readings.json"
ANTITUS = next(p for p in json.loads(READINGS.read_text(encoding="utf-8"))["pages"] if p["page_id"] == "middle-french-antitus")


def _decision(decision_id: str, status: str, reason: str) -> dict[str, Any]:
    return {"decision_id": decision_id, "start_offset": 161, "end_offset": 166, "surface": "grace", "status": status, "method": "m", "reason": reason}


def test_a_span_decided_twice_keeps_one_decision_and_its_attempt() -> None:
    decisions, attempts = merge_span_decisions(
        [_decision("near-miss", "REJECT_LINKABLE", "nd=0.400>0.25"), _decision("gate", "FILTERED_OUT", "place_likeness")],
        [{"decision_id": "near-miss", "attempt_idx": 0, "candidate": "work"}],
    )
    assert [d["decision_id"] for d in decisions] == ["near-miss"]
    assert decisions[0]["meta_json"] == {"merged": [{"status": "FILTERED_OUT", "method": "m", "reason": "place_likeness"}]}
    assert attempts == [{"decision_id": "near-miss", "attempt_idx": 0, "candidate": "work"}]


def test_the_stronger_decision_wins_and_takes_the_other_attempts() -> None:
    decisions, attempts = merge_span_decisions(
        [_decision("gate", "FILTERED_OUT", "gate"), _decision("accepted", "ACCEPT_LINKABLE", "accepted")],
        [{"decision_id": "accepted", "attempt_idx": 0}, {"decision_id": "gate", "attempt_idx": 0}],
    )
    assert [d["status"] for d in decisions] == ["ACCEPT_LINKABLE"]
    assert [(a["decision_id"], a["attempt_idx"]) for a in attempts] == [("accepted", 0), ("accepted", 1)]


def test_distinct_spans_are_left_alone() -> None:
    rows = [_decision("a", "FILTERED_OUT", "x"), {**_decision("b", "FILTERED_OUT", "y"), "start_offset": 170, "end_offset": 175}]
    decisions, attempts = merge_span_decisions(rows, [])
    assert [d["decision_id"] for d in decisions] == ["a", "b"]
    assert all(d.get("meta_json") is None for d in decisions)
    assert attempts == []


@pytest.mark.parametrize("reading", ["reference", "catmus"])
def test_the_antitus_page_is_analysed(tmp_path: Path, monkeypatch: Any, reading: str) -> None:
    """It raised "FOREIGN KEY constraint failed" on both readings."""
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    text = ANTITUS["reference"] if reading == "reference" else ANTITUS["readings"]["kraken_catmus"]
    run_id = pipeline_db.create_run(asset_ref="antitus", asset_sha256=None)

    ocr_router._run_trace_analysis(run_id, text)

    decisions = pipeline_db.list_entity_decisions(run_id)
    spans = [d["span_key"] for d in decisions]
    assert len(spans) == len(set(spans))
    attempts = pipeline_db.list_entity_attempts_for_run(run_id)
    assert {a["decision_id"] for a in attempts} <= {d["decision_id"] for d in decisions}


def test_the_antitus_run_completes(tmp_path: Path, monkeypatch: Any) -> None:
    """It was left RUNNING at ANALYZE_RUNNING, ungraded and unindexed."""
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    monkeypatch.setattr(ocr_router, "_run_authority_linking_stage", lambda run_id: None)
    monkeypatch.setattr(ocr_router, "_auto_index_run", lambda run_id: None)
    text = ANTITUS["reference"]
    response = SaiaFullPageExtractResponse(
        status="FULL",
        model_used="kraken_catmus",
        detected_language="middle_french",
        script_hint="latin",
        confidence=0.9,
        warnings=[],
        lines=text.splitlines(),
        text=text,
        fallbacks=[],
        comparison_runs=[],
    )
    payload = SaiaFullPageExtractRequest(document_id="doc-1", page_id="antitus", image_b64="aGk=", apply_proofread=False)

    result = ocr_router._run_full_page_post_ocr_pipeline(payload, response, b"image-bytes", ocr_backend="segmented")

    run = pipeline_db.get_run(str(result["run_id"]))
    assert run["status"] == "COMPLETED"
    assert run["search_allowed"] is not None
    assert result["mentions_count"] > 0
