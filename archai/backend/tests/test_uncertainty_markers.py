"""apply_uncertainty_markers marks an uncertain token as a whole (audit SVC-15).

For a non-wordlike token with unlikely bigrams it replaced each letter whose
non_wordlike_score passed 0.3 - and a single letter always scores 0.5, so
every letter became "?", a made-up token of a definite length.
"""

from __future__ import annotations

from app.services.ocr_quality import apply_uncertainty_markers


def _mark(text: str, other_pass: str) -> tuple[str, int]:
    return apply_uncertainty_markers(text, cross_pass_text=other_pass, cross_pass_stability=0.1)


def test_a_garbled_unstable_token_becomes_one_span_marker():
    text, count = _mark("in nomine xqzvbt amen", "in nomine domini amen")
    assert text == "in nomine […] amen"
    assert count == 1
    assert "?" not in text


def test_stable_text_is_left_alone():
    assert _mark("in nomine domini amen", "in nomine domini amen") == ("in nomine domini amen", 0)
