"""The search and NER flags agree with the hard gates (audit SVC-05)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_backend = Path(__file__).resolve().parents[1]
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from app.services.ocr_quality import OCRQualityReport  # type: ignore[import-untyped]  # noqa: E402
from app.services.pipeline_hardening import enforce_quality_gates  # type: ignore[import-untyped]  # noqa: E402


def _report(**fields: object) -> OCRQualityReport:
    """A report the label allows: OK, nothing wrong but what a test sets."""
    report = OCRQualityReport(quality_label="OK", token_count=200, tiled=False)
    report.token_search_allowed = True
    report.ner_allowed = True
    for name, value in fields.items():
        setattr(report, name, value)
    return report


def test_a_clean_report_is_searched() -> None:
    gates = enforce_quality_gates(_report(), lexical_plausibility=0.8)
    assert gates["blocked_stages"] == []
    assert gates["token_search_allowed"] and gates["ner_allowed"]


@pytest.mark.parametrize(
    ("fields", "lexical", "gate"),
    [
        ({}, 0.05, "LEXICAL_PLAUSIBILITY"),
        ({"cross_pass_stability": 0.10}, 0.8, "CROSS_PASS_STABILITY"),
        ({"gibberish_score": 0.95}, 0.8, "GIBBERISH"),
        ({"non_wordlike_frac": 0.95}, 0.8, "NON_WORDLIKE"),
    ],
)
def test_a_failed_gate_blocks_search_and_ner_whatever_the_label(fields: dict, lexical: float, gate: str) -> None:
    gates = enforce_quality_gates(_report(**fields), lexical_plausibility=lexical)
    assert not gates["gates"][gate]["passed"]
    assert {"token_search", "token_ner"} <= set(gates["blocked_stages"])
    assert not gates["token_search_allowed"]
    assert not gates["ner_allowed"]


def test_a_label_that_forbids_search_still_does() -> None:
    report = _report()
    report.token_search_allowed = False
    report.ner_allowed = False
    gates = enforce_quality_gates(report, lexical_plausibility=0.8)
    assert not gates["token_search_allowed"] and not gates["ner_allowed"]
