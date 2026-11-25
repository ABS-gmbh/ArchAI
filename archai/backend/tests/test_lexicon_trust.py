"""Regression tests for lexical trust scoring."""
from __future__ import annotations


from app.services.lexicon_trust import (  # type: ignore[import-untyped]
    lexical_plausibility,
    lexical_trust_adjustment,
)


# ── lexical_plausibility ──────────────────────────────────────────────


def test_plausible_latin_text() -> None:
    text = "dominus noster et patris sancti"
    score = lexical_plausibility(text, "latin")
    assert score >= 0.50, f"Expected >= 0.50 for plausible Latin, got {score}"


def test_plausible_old_french_text() -> None:
    text = "furent les noces et le festin grant joie"
    score = lexical_plausibility(text, "old_french")
    assert score >= 0.50, f"Expected >= 0.50 for plausible Old French, got {score}"


def test_garbage_text_low_score() -> None:
    text = "qjxvvbbx cccnnn zzppp xkwbfq"
    score = lexical_plausibility(text, "latin")
    assert score < 0.35, f"Expected < 0.35 for garbage text, got {score}"


def test_unknown_language_neutral() -> None:
    text = "any text here"
    score = lexical_plausibility(text, "unknown")
    assert score == 0.50


def test_empty_text_neutral() -> None:
    score = lexical_plausibility("", "latin")
    assert score == 0.50


def test_alias_anglo_norman() -> None:
    text = "furent les noces et le festin"
    score_of = lexical_plausibility(text, "old_french")
    score_an = lexical_plausibility(text, "anglo_norman")
    assert score_of == score_an, "anglo_norman should alias to old_french"


# ── lexical_trust_adjustment ──────────────────────────────────────────


def test_trust_adjustment_drops_for_garbage() -> None:
    adj_conf, warnings = lexical_trust_adjustment(0.80, "qjxvvbbx cccnnn", "latin")
    assert adj_conf < 0.80
    assert any("LEXICAL" in w for w in warnings)


def test_trust_adjustment_stable_for_good_text() -> None:
    adj_conf, warnings = lexical_trust_adjustment(0.80, "dominus noster et patris", "latin")
    assert adj_conf >= 0.75  # should not drop much


def test_trust_adjustment_clamped() -> None:
    adj_conf, _ = lexical_trust_adjustment(0.01, "qjxvvbbx cccnnn", "latin")
    assert adj_conf >= 0.05  # clamped minimum
