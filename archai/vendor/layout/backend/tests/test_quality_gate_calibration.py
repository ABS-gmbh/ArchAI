"""The OCR quality gate on transcriptions of known quality.

The gate decides whether a page may feed entity extraction, authority linking,
the search index and translation. It was calibrated on two short, normalised
sentences, and on real medieval transcriptions it was close to inverted:

* it blocked three of the five reference transcriptions - correct by
  construction - and the CATMuS readings of four of the five pages;
* langdetect has no Latin model, so the Latin pages were scored as Catalan or
  French, and their trigrams against the wrong profile;
* the trigram filter kept only a-z and Latin-1 letters, deleting the ẽ, ũ and ꝑ
  of a diplomatic transcription from the middle of their words;
* a line-start rule meant for words cut at tile seams counted every separated
  verse initial, although the full-page route never cuts a page into tiles;
* on 72 transcriptions from the pipeline database it blocked 71.

``eval/quality_gate/readings.json`` holds the five references and every
recogniser's reading of the same pages; scripts/calibrate_quality_gate.py
prints the full table.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import pytest

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.routers import ocr as ocr_router  # type: ignore[import-untyped]  # noqa: E402
from app.services.lexicon_trust import best_fit_plausibility  # type: ignore[import-untyped]  # noqa: E402
from app.services.ocr_quality import compute_quality_report  # type: ignore[import-untyped]  # noqa: E402
from app.services.pipeline_hardening import enforce_quality_gates  # type: ignore[import-untyped]  # noqa: E402

READINGS = Path(__file__).resolve().parents[1] / "eval" / "quality_gate" / "readings.json"
PAGES: list[dict[str, Any]] = json.loads(READINGS.read_text(encoding="utf-8"))["pages"]
PAGE_IDS = [page["page_id"] for page in PAGES]


def gate(text: str) -> tuple[str, str, bool]:
    """(detected language, label, NER and search allowed), as the full-page pipeline decides."""
    detected, _confidence = ocr_router._detect_language_metadata(text)
    language = ocr_router._normalize_detected_language(detected)
    report = compute_quality_report(text, language=ocr_router._quality_language(language), tiled=False)
    plausibility = best_fit_plausibility(text, language) if language != "unknown" else None
    decisions = enforce_quality_gates(report, lexical_plausibility=plausibility)
    blocked = {"token_ner", "token_search"} & set(decisions["blocked_stages"])
    return language, report.quality_label, bool(report.ner_allowed and report.token_search_allowed and not blocked)


def page(page_id: str) -> dict[str, Any]:
    return next(item for item in PAGES if item["page_id"] == page_id)


def reverse_words(text: str) -> str:
    return "\n".join(" ".join(word[::-1] for word in line.split()) for line in text.splitlines())


def random_letters(text: str, seed: int = 3) -> str:
    rng = random.Random(seed)
    return "\n".join(
        " ".join("".join(rng.choice("abcdefghilmnopqrstuvxyz") for _ in word) for word in line.split())
        for line in text.splitlines()
    )


def substitute(text: str, rate: float, seed: int = 5) -> str:
    rng = random.Random(seed)
    return "".join(
        rng.choice("abcdefghilmnopqrstuvxyz") if ch.isalpha() and rng.random() < rate else ch for ch in text
    )


# ------------------------------------------------- faithful text passes --


@pytest.mark.parametrize("page_id", PAGE_IDS)
def test_a_reference_transcription_reaches_entities_and_search(page_id: str) -> None:
    language, label, allowed = gate(page(page_id)["reference"])
    assert allowed, f"{page_id} reference graded {label} as {language}"
    assert label in {"HIGH", "OK"}


@pytest.mark.parametrize("page_id", PAGE_IDS)
def test_the_catmus_reading_of_every_page_passes(page_id: str) -> None:
    """4-18% CER: the readings of the segmented full-page route (ABS-gmbh/ArchAI#15)."""
    _language, label, allowed = gate(page(page_id)["readings"]["kraken_catmus"])
    assert allowed, f"{page_id} CATMuS reading graded {label}"


@pytest.mark.parametrize("page_id", PAGE_IDS)
def test_light_noise_does_not_block_a_page(page_id: str) -> None:
    _language, label, allowed = gate(substitute(page(page_id)["reference"], 0.10))
    assert allowed, f"{page_id} with 10% of its letters substituted graded {label}"


@pytest.mark.parametrize("page_id", ["latin-abaton", "latin-apocalypse"])
def test_latin_is_detected_as_latin(page_id: str) -> None:
    for text in (page(page_id)["reference"], page(page_id)["readings"]["kraken_catmus"]):
        assert gate(text)[0] == "latin"


@pytest.mark.parametrize("page_id", ["old-french", "middle-french-antitus", "old-french-verse"])
def test_french_stays_french(page_id: str) -> None:
    assert gate(page(page_id)["reference"])[0] in {"french", "old_french", "middle_french"}


# ---------------------------------------------------- garbage is blocked --


@pytest.mark.parametrize("page_id", PAGE_IDS)
@pytest.mark.parametrize("corruption", ["reversed words", "random letters", "70% substituted"])
def test_corrupt_text_is_blocked(page_id: str, corruption: str) -> None:
    reference = page(page_id)["reference"]
    text = {
        "reversed words": reverse_words(reference),
        "random letters": random_letters(reference),
        "70% substituted": substitute(reference, 0.70),
    }[corruption]
    _language, label, allowed = gate(text)
    assert not allowed, f"{page_id} {corruption} graded {label}"
    assert label in {"RISKY", "UNRELIABLE"}


@pytest.mark.parametrize("page_id", ["old-french", "latin-abaton", "middle-french-antitus"])
def test_a_decoding_loop_is_unreliable(page_id: str) -> None:
    """GLM-OCR's readings of these pages repeat a block of lines - or its own
    instructions - until the token limit, 3 to 15 times the page's length."""
    _language, label, allowed = gate(page(page_id)["readings"]["glmocr"])
    assert (label, allowed) == ("UNRELIABLE", False)
