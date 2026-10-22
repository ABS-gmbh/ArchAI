"""Tests for the proofreader hallucination guard.

proofreading_quality_guard only ever rejected output whose quality DEGRADED.
The worst failure mode of an LLM proofreader on a recognisable passage is the
opposite: it recognises the text and writes it out from memory. The fabricated
result is fluent and grades HIGH while the true garbled OCR grades RISKY or
UNRELIABLE, so a guard that only looks for degradation actively PREFERS the
invention. check_proofread_delta, which measures how much the text changed
rather than how good it looks, was reachable from only one of four call sites.
"""

from __future__ import annotations

from app.services.ocr_quality import compute_quality_report
from app.services.pipeline_hardening import proofreading_quality_guard

GARBLED = "\n".join([
    "Pat noſter qui eſ in celis",
    "ſanctifi etur nomen tuu",
    "aduen at regnu tuu",
    "fi at uolutas tua",
])

# What a model produces when it recognises the Pater Noster and recites it.
HALLUCINATED = "\n".join([
    "Pater noster qui es in caelis sanctificetur nomen tuum",
    "adveniat regnum tuum fiat voluntas tua",
    "sicut in caelo et in terra panem nostrum quotidianum da nobis hodie",
    "et dimitte nobis debita nostra sicut et nos dimittimus debitoribus nostris",
])

# A conservative fix: same lines, same structure, spelling repaired.
CORRECTED = "\n".join([
    "Pat noster qui es in celis",
    "sanctificetur nomen tuum",
    "adveniat regnum tuum",
    "fiat uolutas tua",
])


def report_for(text: str):
    return compute_quality_report(text, language="latin")


def test_invented_text_is_rejected() -> None:
    text, accepted, reason = proofreading_quality_guard(GARBLED, HALLUCINATED, report_for(GARBLED))
    assert accepted is False
    assert text == GARBLED, "the original transcription must be kept"
    assert reason


def test_a_genuine_conservative_correction_is_accepted() -> None:
    """The guard must not cost us real corrections."""
    text, accepted, _ = proofreading_quality_guard(GARBLED, CORRECTED, report_for(GARBLED))
    assert accepted is True
    assert text == CORRECTED


def test_token_churn_alone_does_not_condemn_a_correction() -> None:
    """Correcting spelling inherently creates new tokens.

    The genuine correction above measures ~50% token churn at only ~11%
    character edit; rejecting on churn alone would discard it.
    """
    from app.agents.ocr_proofreader_agent import check_proofread_delta

    delta = check_proofread_delta(GARBLED, CORRECTED)
    assert delta.token_churn > 0.35, "precondition: this correction does churn tokens"
    _, accepted, _ = proofreading_quality_guard(GARBLED, CORRECTED, report_for(GARBLED))
    assert accepted is True


def test_wholesale_rewrite_is_rejected_even_if_it_reads_better() -> None:
    rewritten = "The quick brown fox jumps over the lazy dog. " * 4
    text, accepted, _ = proofreading_quality_guard(GARBLED, rewritten, report_for(GARBLED))
    assert accepted is False
    assert text == GARBLED


def test_empty_proofread_output_is_rejected() -> None:
    text, accepted, reason = proofreading_quality_guard(GARBLED, "   ", report_for(GARBLED))
    assert accepted is False
    assert text == GARBLED
    assert "empty" in reason


def test_identical_text_is_accepted_unchanged() -> None:
    text, accepted, _ = proofreading_quality_guard(GARBLED, GARBLED, report_for(GARBLED))
    assert accepted is True
    assert text == GARBLED


def test_the_delta_check_is_reachable_from_the_router_path() -> None:
    """Regression: this check ran at only one of four proofreader call sites."""
    _, accepted, reason = proofreading_quality_guard(GARBLED, HALLUCINATED, report_for(GARBLED))
    assert accepted is False
    assert "char_edit_ratio" in reason or "invention" in reason
