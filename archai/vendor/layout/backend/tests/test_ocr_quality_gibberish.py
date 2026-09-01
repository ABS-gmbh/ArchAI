"""Regression tests for the OCR quality gate.

Before these signals existed, gibberish_score was computed from character-level
statistics only. Those statistics are invariant to letter transposition and to
repetition, so clean text and thoroughly corrupt text produced identical scores
of 0.0 and both were graded HIGH. The gate that decides whether a transcription
may feed entity linking and RAG indexing could therefore never reject anything.
"""

from __future__ import annotations

import random

import pytest

from app.services.ocr_quality import (
    compute_quality_report,
    gibberish_score,
    lexical_implausibility,
    repetition_score,
)
from app.services.ocr_quality_config import (
    GIBBERISH_HARD_LIMIT,
    LEXICON_CLEAN_FLOOR,
    REPETITION_HARD_LIMIT,
    REPETITION_SOFT_LIMIT,
)

CLEAN_OLD_FRENCH = (
    "Et quant li rois ot ce oi il fu mout dolenz et corrociez "
    "si comme cil qui mout amoit le chevalier"
)
CLEAN_LATIN = (
    "Igitur in nomine domini nostri Iesu Christi incipit liber "
    "de vita et moribus sanctorum patrum"
)


def reverse_words(text: str) -> str:
    """Letter-reverse every word: destroys meaning, preserves every character statistic."""
    return " ".join(word[::-1] for word in text.split())


def random_letters(seed: int = 7, words: int = 40) -> str:
    rng = random.Random(seed)
    return " ".join("".join(rng.choice("abcdefghilmnopqrstuv") for _ in range(6)) for _ in range(words))


# ─────────────────────────────────────────────────────── repetition ──


def test_repetition_score_is_zero_for_normal_prose() -> None:
    assert repetition_score(CLEAN_OLD_FRENCH) == 0.0


def test_repetition_score_detects_a_repeated_line() -> None:
    """The degenerate decoding loop a VLM falls into."""
    assert repetition_score("\n".join([CLEAN_OLD_FRENCH] * 40)) >= REPETITION_HARD_LIMIT


def test_repetition_score_detects_a_loop_that_is_not_line_aligned() -> None:
    looped = ("alpha beta gamma delta epsilon " * 20).strip()
    assert repetition_score(looped) >= REPETITION_HARD_LIMIT


def test_repetition_score_ignores_short_input() -> None:
    assert repetition_score("a b") == 0.0
    assert repetition_score("") == 0.0


def test_a_few_duplicate_lines_do_not_trip_the_hard_limit() -> None:
    """Manuscript pages legitimately repeat short refrains."""
    lines = [f"line number {i} of the folio" for i in range(20)] + [CLEAN_OLD_FRENCH] * 2
    assert repetition_score("\n".join(lines)) < REPETITION_SOFT_LIMIT


# ──────────────────────────────────────────────── lexical signal ──


@pytest.mark.parametrize(("text", "language"), [(CLEAN_LATIN, "latin"), (CLEAN_OLD_FRENCH, "old_french")])
def test_clean_text_sits_at_or_below_the_clean_floor(text: str, language: str) -> None:
    value = lexical_implausibility(text, language)
    assert value is not None
    assert value <= LEXICON_CLEAN_FLOOR + 0.01


@pytest.mark.parametrize(("text", "language"), [(CLEAN_LATIN, "latin"), (CLEAN_OLD_FRENCH, "old_french")])
def test_reversed_text_is_far_more_implausible_than_clean(text: str, language: str) -> None:
    clean = lexical_implausibility(text, language)
    corrupt = lexical_implausibility(reverse_words(text), language)
    assert clean is not None and corrupt is not None
    assert corrupt > clean + 0.25


def test_lexical_implausibility_is_none_without_a_language() -> None:
    assert lexical_implausibility(CLEAN_LATIN, None) is None


def test_lexical_implausibility_is_none_for_a_language_with_no_profile() -> None:
    assert lexical_implausibility(CLEAN_LATIN, "klingon") is None
    assert lexical_implausibility(CLEAN_LATIN, "unknown") is None


# ─────────────────────────────────────────── gibberish score ──


def test_clean_text_scores_near_zero() -> None:
    assert gibberish_score(CLEAN_LATIN, "latin", "latin") < 0.10
    assert gibberish_score(CLEAN_OLD_FRENCH, "latin", "old_french") < 0.10


@pytest.mark.parametrize(("text", "language"), [(CLEAN_LATIN, "latin"), (CLEAN_OLD_FRENCH, "old_french")])
def test_reversed_text_no_longer_scores_the_same_as_clean(text: str, language: str) -> None:
    """The original defect: these two were bit-identical at 0.0."""
    clean = gibberish_score(text, "latin", language)
    corrupt = gibberish_score(reverse_words(text), "latin", language)
    assert corrupt > clean
    assert corrupt >= GIBBERISH_HARD_LIMIT * 0.7


def test_random_letters_score_higher_than_reversed_real_words() -> None:
    noise = gibberish_score(random_letters(), "latin", "old_french")
    reversed_real = gibberish_score(reverse_words(CLEAN_OLD_FRENCH), "latin", "old_french")
    assert noise > reversed_real


def test_gibberish_score_is_stable_for_empty_input() -> None:
    assert gibberish_score("", "latin", "latin") == 0.0
    assert gibberish_score("   ", "latin", "latin") == 0.0


def test_gibberish_score_signature_stays_backward_compatible() -> None:
    """Existing callers pass (text) or (text, script) only."""
    assert gibberish_score(CLEAN_LATIN) >= 0.0
    assert gibberish_score(CLEAN_LATIN, "latin") >= 0.0


# ───────────────────────────────────────── end-to-end labels ──


@pytest.mark.parametrize(
    ("name", "text", "language"),
    [
        ("reversed old french", reverse_words(CLEAN_OLD_FRENCH), "old_french"),
        ("reversed latin", reverse_words(CLEAN_LATIN), "latin"),
        ("repetition loop", "\n".join([CLEAN_OLD_FRENCH] * 40), "old_french"),
        ("random letters", random_letters(), "old_french"),
    ],
)
def test_corrupt_text_is_not_graded_high(name: str, text: str, language: str) -> None:
    report = compute_quality_report(text, language=language)
    assert report.quality_label in ("RISKY", "UNRELIABLE"), f"{name} graded {report.quality_label}"


@pytest.mark.parametrize(
    ("text", "language"), [(CLEAN_LATIN, "latin"), (CLEAN_OLD_FRENCH, "old_french")]
)
def test_clean_text_is_still_graded_high(text: str, language: str) -> None:
    """The fix must not cost us true positives on good transcriptions."""
    assert compute_quality_report(text, language=language).quality_label == "HIGH"


def test_corrupt_text_loses_downstream_permissions() -> None:
    """The point of the gate: garbage must not reach entity linking or RAG."""
    report = compute_quality_report(reverse_words(CLEAN_LATIN), language="latin")
    assert report.quality_label == "UNRELIABLE"
    assert report.ner_allowed is False
    assert report.token_search_allowed is False


def test_repetition_is_caught_even_without_a_language_profile() -> None:
    report = compute_quality_report("\n".join([CLEAN_OLD_FRENCH] * 40))
    assert report.quality_label in ("RISKY", "UNRELIABLE")


def test_report_records_the_new_signals() -> None:
    report = compute_quality_report(CLEAN_LATIN, language="latin")
    payload = report.to_dict()
    assert "repetition_score" in payload
    assert "lexical_implausibility" in payload
    assert payload["detected_language"] == "latin"


def test_unmeasured_lexical_signal_is_negative_not_neutral() -> None:
    """-1.0 must mean 'not measured', distinct from 'measured as plausible'."""
    report = compute_quality_report(CLEAN_LATIN)
    assert report.lexical_implausibility == -1.0
