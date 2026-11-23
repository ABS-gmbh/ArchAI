"""Language detection gives one answer per text (langdetect is seeded)."""

from __future__ import annotations

import random

from app.services import language_detection


def _corrupt(text: str, rate: float, seed: int) -> str:
    rng = random.Random(seed)
    letters = "abcdefghijklmnopqrstuvwxyz"
    return "".join(rng.choice(letters) if ch.isalpha() and rng.random() < rate else ch for ch in text)


def test_the_same_corrupted_text_always_gets_the_same_language() -> None:
    page = (
        "In diebus illis uenit angelus et locutus est mecum dicens ueni ostendam tibi sponsam uxorem agni "
        "et sustulit me in spiritu in montem magnum et altum et ostendit mihi ciuitatem sanctam hyerusalem "
        "descendentem de caelo habentem claritatem dei lumen eius simile lapidi pretioso"
    )
    noisy = _corrupt(page, 0.5, seed=7)
    # Language and probability: unseeded, even the probability drifts between calls.
    answers = {language_detection._detect_language_metadata(noisy) for _ in range(12)}
    assert len(answers) == 1, answers
