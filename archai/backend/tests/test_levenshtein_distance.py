"""Whole-page edit distance is exact and fast (audit SVC-19, AG-10).

The cross-pass stability check and the proofreader guard each ran a two-row
Levenshtein DP in interpreted Python over whole pages: measured 2.1 s for a
4,000-character page, on the event loop in the trace pipeline. Both now use
a bit-parallel algorithm with the same result.
"""

from __future__ import annotations

import random
import time

import pytest

from app.agents.ocr_proofreader_agent import _normalised_levenshtein
from app.services.ocr_text_metrics import levenshtein_distance, normalized_levenshtein_similarity


def _reference(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        curr = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = curr
    return prev[-1]


@pytest.mark.parametrize("a, b, expected", [
    ("", "", 0), ("", "abc", 3), ("abc", "", 3), ("kitten", "sitting", 3),
    ("flaw", "lawn", 2), ("dñs", "dominus", 5), ("ſanctus", "sanctus", 1),
])
def test_known_distances(a, b, expected):
    assert levenshtein_distance(a, b) == expected == levenshtein_distance(b, a)


def test_it_agrees_with_the_dynamic_programme():
    rng = random.Random(7)
    for _ in range(1500):
        a = "".join(rng.choice("abcñ ſ\n") for _ in range(rng.randint(0, 60)))
        b = "".join(rng.choice("abcñ ſ\n") for _ in range(rng.randint(0, 60)))
        assert levenshtein_distance(a, b) == _reference(a, b), (a, b)


def test_a_whole_page_takes_milliseconds():
    rng = random.Random(3)
    page = "".join(rng.choice("abcdefghij klmnop\n") for _ in range(4000))
    other = list(page)
    for _ in range(300):
        other[rng.randrange(len(other))] = rng.choice("xyz")
    started = time.perf_counter()
    levenshtein_distance(page, "".join(other))
    assert time.perf_counter() - started < 0.5


def test_the_two_normalised_forms_are_unchanged():
    assert normalized_levenshtein_similarity("kitten", "sitting") == pytest.approx(1 - 3 / 7)
    assert _normalised_levenshtein("kitten", "sitting") == pytest.approx(3 / 7)
    assert normalized_levenshtein_similarity("", "a") == 0.0 and _normalised_levenshtein("", "a") == 1.0
