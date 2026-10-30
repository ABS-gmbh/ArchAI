"""The OCR retry loop in ocr_page_with_trace, driven with a fake OCR agent.

Storage, quality reports and gates are replaced by stubs, so each test sets
the outcome of every attempt and reads which OCR calls the loop made and
what the run recorded when OCR was done. The run is stopped there with a
BaseException: the stages after it are not what these tests are about.
"""

from __future__ import annotations

import asyncio
import base64
import io
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from app.routers import ocr as ocr_router
from app.schemas.agents_ocr import SaiaFullPageExtractRequest, SaiaOCRLocationSuggestion, SaiaOCRResponse
from app.services.ocr_quality import OCRQualityReport

def _page_b64(size: tuple[int, int] = (800, 1200)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


PAGE_SIZE = (800, 1200)
SUGGESTION = SaiaOCRLocationSuggestion(region_id="r1", category="text", bbox_xywh=[80, 120, 640, 960])


class _Stop(BaseException):
    """Ends the run after the calls under test; the endpoint catches only Exception."""


def _gates(passed: bool) -> dict[str, Any]:
    return {
        "gates": {"quality": {"passed": passed}},
        "blocked_stages": [] if passed else ["ner"],
        "downstream_mode": "full" if passed else "degraded",
        "seam_retry_required": False,
    }


class _Harness:
    def __init__(self, monkeypatch: Any, attempts: list[tuple[Any, ...]]) -> None:
        """attempts: (text, quality label, gates passed[, response fields]) per OCR call, in order;
        the last repeats."""
        self.attempts = attempts
        self.requests: list[Any] = []
        self.reported: list[str] = []
        self.ocr_done: dict[str, Any] = {}
        harness = self

        class _Agent:
            def extract(self, request: Any) -> SaiaOCRResponse:
                harness.requests.append(request)
                entry = harness.attempts[min(len(harness.requests), len(harness.attempts)) - 1]
                fields = {"detected_language": "latin", "script_hint": "latin", "confidence": 0.8,
                          **(entry[3] if len(entry) > 3 else {})}
                return SaiaOCRResponse(status="FULL", model_used="fake", text=entry[0],
                                       lines=entry[0].splitlines(), **fields)

        def report(text: str, **kwargs: Any) -> OCRQualityReport:
            label = next((a[1] for a in harness.attempts if a[0] == text), "OK")
            harness.reported.append(text)
            return OCRQualityReport(quality_label=label, gibberish_score=0.1)

        def gates(report: OCRQualityReport, **kwargs: Any) -> dict[str, Any]:
            text = harness.reported[-1]
            return _gates(next((a[2] for a in harness.attempts if a[0] == text), True))

        def update_run_fields(run_id: str, **fields: Any) -> None:
            if fields.get("current_stage") == "OCR_DONE":
                harness.ocr_done = fields
                raise _Stop()

        noop = lambda *args, **kwargs: None  # noqa: E731
        for name in ("log_event", "insert_ocr_quality_report", "insert_ocr_attempt", "insert_ocr_benchmark_reference"):
            monkeypatch.setattr(ocr_router, name, noop)
        monkeypatch.setattr(ocr_router, "update_run_fields", update_run_fields)
        monkeypatch.setattr(ocr_router, "create_run", lambda **kwargs: "run-1")
        monkeypatch.setattr(ocr_router, "get_test_ocr_fixture", lambda image_bytes: None)
        monkeypatch.setattr(ocr_router, "get_test_ocr_override", lambda image_bytes: None)
        monkeypatch.setattr(ocr_router, "_regions_from_location_suggestions", lambda suggestions: [])
        monkeypatch.setattr(ocr_router, "_get_saia_ocr_agent", lambda: _Agent())
        monkeypatch.setattr(ocr_router, "compute_quality_report", report)
        monkeypatch.setattr(ocr_router, "enforce_quality_gates", gates)
        monkeypatch.setattr(ocr_router, "build_effective_quality",
                            lambda *a, **k: SimpleNamespace(to_dict=lambda: {}))
        monkeypatch.setattr(ocr_router, "_lexical_plausibility", lambda *a, **k: None)
        monkeypatch.setattr(ocr_router, "format_gate_report", lambda *a, **k: "")

    def run(self) -> None:
        payload = SaiaFullPageExtractRequest(
            page_id="page-1", image_b64=_page_b64(PAGE_SIZE), apply_proofread=False,
            location_suggestions=[SUGGESTION],
        )
        with pytest.raises(_Stop):
            asyncio.run(ocr_router.ocr_page_with_trace(payload))


# ── OCR-008: the cross-pass stability pass ──────────────────────────────


def test_a_first_attempt_that_passes_as_ok_gets_a_stability_pass(monkeypatch: Any) -> None:
    """Audit OCR-008: it waited for prev_tiling_plan, set only when an attempt fails."""
    harness = _Harness(monkeypatch, [("In nomine domini amen", "OK", True)])
    harness.run()
    assert len(harness.requests) == 2  # the attempt, then the stability pass


def test_a_high_first_attempt_needs_no_stability_pass(monkeypatch: Any) -> None:
    harness = _Harness(monkeypatch, [("In nomine domini amen", "HIGH", True)])
    harness.run()
    assert len(harness.requests) == 1


def test_the_stability_pass_perturbs_the_kept_attempts_tiles(monkeypatch: Any) -> None:
    """It perturbed the last attempt's plan; the kept one can be an earlier attempt."""
    harness = _Harness(monkeypatch, [("Arthur rex", "OK", False), ("Arthur rex fuit", "RISKY", False),
                                     ("Arthur", "UNRELIABLE", False)])
    harness.run()
    from app.services.seam_strategies import expand_overlap

    attempts = ocr_router._MAX_OCR_ATTEMPTS
    assert attempts >= 2
    assert len(harness.requests) == attempts + 1
    assert harness.ocr_done["ocr_text"] == "Arthur rex"  # attempt 0 is kept

    w, h = PAGE_SIZE
    kept_plan = ocr_router.default_plan_from_suggestions(
        [{"region_id": "r1", "category": "text", "bbox_xywh": list(SUGGESTION.bbox_xywh)}], w, h
    )
    expected = ocr_router.plan_to_suggestions(
        expand_overlap(kept_plan.tile_boxes, w, h, expand_pct=0.10, attempt_idx=99), w, h
    )
    sent = [list(s.bbox_xywh) for s in harness.requests[-1].location_suggestions]
    assert sent == [list(d["bbox_xywh"]) for d in expected]

