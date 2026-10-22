"""Tests for the score that chooses between OCR backends.

_region_quality_value is the only signal used to decide which backend's text
becomes the transcription, and it could not tell good text from bad. Its one
content-aware term was lexical_plausibility(text, hint), and that returns a flat
0.50 for "unknown" - the value language detection frequently yields - so with no
usable hint the score collapsed to a near-constant. Measured on the old code:

    clean Latin          0.8137
    random letters       0.8250   <- outscored clean text
    repetition loop x20  0.8236   <- the failure mode the OCR prompt warns about

A wrong hint was actively harmful: clean Old French scored 0.8481 under
"old_french" but 0.4626 under "latin".

The pipeline already holds a sample-based election - _reorder_attempt_backends_from_samples
scores up to 6 regions per backend and reorders the plan by mean score. That
election was blind, not absent.
"""

from __future__ import annotations

import random

import pytest

from app.agents.ocr_agent import _region_quality_value, best_fit_language

CLEAN_LATIN = (
    "Igitur in nomine domini nostri Iesu Christi incipit liber de vita "
    "et moribus sanctorum patrum"
)
CLEAN_OLD_FRENCH = (
    "Et quant li rois ot ce oi il fu mout dolenz et corrociez si comme cil qui mout amoit"
)
REVERSED_LATIN = " ".join(w[::-1] for w in CLEAN_LATIN.split())
REPETITION_LOOP = "\n".join([CLEAN_LATIN] * 20)


def random_letters(seed: int = 3, words: int = 20) -> str:
    rng = random.Random(seed)
    return " ".join("".join(rng.choice("abcdefghilmnopqrstuv") for _ in range(6)) for _ in range(words))


# ─────────────────────────────────────────────── best-fit language ──


@pytest.mark.parametrize(
    ("text", "expected"), [(CLEAN_LATIN, "latin"), (CLEAN_OLD_FRENCH, "old_french")]
)
def test_best_fit_identifies_the_language_without_a_hint(text: str, expected: str) -> None:
    assert best_fit_language(text)[0] == expected


def test_best_fit_scores_corrupt_text_far_below_clean_text() -> None:
    assert best_fit_language(CLEAN_LATIN)[1] > best_fit_language(REVERSED_LATIN)[1] + 0.3
    assert best_fit_language(CLEAN_LATIN)[1] > best_fit_language(random_letters())[1] + 0.3


def test_best_fit_handles_empty_input() -> None:
    assert best_fit_language("") == ("unknown", 0.0)
    assert best_fit_language("   ")[1] == 0.0


# ──────────────────────────────────────── the selector discriminates ──


def test_clean_text_outscores_noise_without_any_hint() -> None:
    """Regression: random letters used to score HIGHER than clean Latin."""
    assert _region_quality_value(CLEAN_LATIN, 0.9, None) > _region_quality_value(
        random_letters(), 0.9, None
    )


def test_clean_text_outscores_transposed_text_without_a_hint() -> None:
    assert _region_quality_value(CLEAN_LATIN, 0.9, None) > _region_quality_value(
        REVERSED_LATIN, 0.9, None
    )


def test_a_decoding_loop_is_penalised() -> None:
    """Regression: the repetition loop used to score near the top."""
    assert _region_quality_value(REPETITION_LOOP, 0.9, None) < _region_quality_value(
        CLEAN_LATIN, 0.9, None
    )


@pytest.mark.parametrize("hint", [None, "unknown", "latin", "old_french", "spanish", "klingon"])
def test_a_wrong_or_missing_hint_never_penalises_clean_text(hint: str | None) -> None:
    """Clean Old French scored 0.8481 under the right hint and 0.4626 under a wrong one."""
    assert _region_quality_value(CLEAN_OLD_FRENCH, 0.9, hint) >= 0.75


def test_the_score_is_stable_across_hint_conditions() -> None:
    scores = {_region_quality_value(CLEAN_LATIN, 0.9, h) for h in (None, "unknown", "latin", "spanish")}
    assert max(scores) - min(scores) < 0.01


def test_confidence_still_moves_the_score() -> None:
    assert _region_quality_value(CLEAN_LATIN, 0.95, None) > _region_quality_value(
        CLEAN_LATIN, 0.10, None
    )


def test_empty_text_scores_zero() -> None:
    assert _region_quality_value("", 0.9, None) == 0.0
    assert _region_quality_value("   ", 0.9, None) == 0.0


def test_the_score_stays_in_range() -> None:
    for text in (CLEAN_LATIN, REVERSED_LATIN, random_letters(), REPETITION_LOOP, "x"):
        for conf in (None, 0.0, 0.5, 1.0):
            assert 0.0 <= _region_quality_value(text, conf, None) <= 1.0


# ──────────────────────────────────────────── the election it feeds ──


def test_the_election_picks_the_correct_backend() -> None:
    """The scores are averaged per backend and the plan reordered by that mean.

    With the old score the good and garbled backends tied exactly, and the
    backend stuck in a decoding loop won.
    """
    rng = random.Random(11)
    good = [
        "Igitur in nomine domini nostri Iesu Christi incipit liber",
        "de vita et moribus sanctorum patrum qui in heremo",
        "Et quant li rois ot ce oi il fu mout dolenz et corrociez",
        "si comme cil qui mout amoit le chevalier del Lac",
    ]
    garbled = [" ".join("".join(rng.choice("abcdefghilmnopqrstuv") for _ in w) for w in s.split()) for s in good]
    looping = ["\n".join([good[0]] * 12)] * 4

    def mean(texts: list[str]) -> float:
        return sum(_region_quality_value(t, 0.9, "unknown") for t in texts) / len(texts)

    scores = {"catmus": mean(good), "mccatmus": mean(garbled), "cremma": mean(looping)}
    assert max(scores, key=lambda k: scores[k]) == "catmus"
    assert scores["catmus"] > scores["mccatmus"] + 0.15
    assert scores["catmus"] > scores["cremma"] + 0.15


def test_backends_producing_different_text_no_longer_tie() -> None:
    """A tie made the election meaningless: the hardcoded order simply survived."""
    good = _region_quality_value(CLEAN_LATIN, 0.9, "unknown")
    bad = _region_quality_value(random_letters(), 0.9, "unknown")
    assert good != bad
