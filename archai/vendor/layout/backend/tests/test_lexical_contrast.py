"""Unit tests for the lexical signals behind the OCR quality gate."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.services import lexicon_trust  # type: ignore[import-untyped]  # noqa: E402
from app.services.lexicon_trust import (  # type: ignore[import-untyped]  # noqa: E402
    CONTRAST_MIN_TRIGRAMS,
    best_fit_plausibility,
    lexical_contrast,
    lexical_plausibility,
    reads_as_latin,
)
from app.services.ocr_quality import (  # type: ignore[import-untyped]  # noqa: E402
    compute_quality_report,
    implausibility_from_contrast,
    leading_fragment_ratio,
    repetition_score,
    seam_fragment_ratio,
)
from app.services.ocr_quality_config import (  # type: ignore[import-untyped]  # noqa: E402
    LEXICAL_CONTRAST_RISKY,
    LEXICAL_CONTRAST_UNRELIABLE,
    LEXICAL_PLAUSIBILITY_FLOOR,
    REPETITION_HARD_LIMIT,
)
from app.services.pipeline_hardening import enforce_quality_gates  # type: ignore[import-untyped]  # noqa: E402

READINGS = Path(__file__).resolve().parents[1] / "eval" / "quality_gate" / "readings.json"
PAGES = {page["page_id"]: page for page in json.loads(READINGS.read_text(encoding="utf-8"))["pages"]}
# Every page but the seven lines of middle-french-antitus is long enough to measure.
MEASURED = [
    page_id
    for page_id, page in PAGES.items()
    if len(lexicon_trust._extract_trigrams(page["reference"])) >= CONTRAST_MIN_TRIGRAMS
]
# A diplomatic Latin page: potẽcia, adũsariũ, ⁊.
LATIN_PAGE = PAGES["latin-abaton"]["reference"]
FRENCH_PAGE = PAGES["old-french"]["reference"]
SHORT_PAGE = PAGES["middle-french-antitus"]["reference"]

# Lines of the reference transcriptions, written as the manuscripts write them.
DIPLOMATIC_FRENCH = (
    "il iure contre sa cõciẽce. ⁊ est a antendre qnͣt lẽ se ꝑiure apenseement "
    "⁊ a delib̾acion. Mes cil qͥ iure uoir a son escient ⁊ toutes uoies por noiant"
)
LATIN = (
    "Igitur in nomine domini nostri Iesu Christi incipit liber "
    "de vita et moribus sanctorum patrum qui in eremo habitauerunt"
)


def reverse_words(text: str) -> str:
    return "\n".join(" ".join(word[::-1] for word in line.split()) for line in text.splitlines())


def random_letters(text: str, seed: int = 11) -> str:
    rng = random.Random(seed)
    return "\n".join(
        " ".join("".join(rng.choice("abcdefghilmnopqrstuvxyz") for _ in word) for word in line.split())
        for line in text.splitlines()
    )


# --------------------------------------------------------------- trigrams --


def test_abbreviated_letters_stay_in_their_words() -> None:
    """Regression: "cõciẽce" lost its õ and ẽ and was scored as "ccice"."""
    assert lexicon_trust._extract_trigrams("cõciẽce") == lexicon_trust._extract_trigrams("concience")


def test_abbreviation_signs_are_expanded() -> None:
    assert lexicon_trust._extract_trigrams("ꝑfitable") == lexicon_trust._extract_trigrams("perfitable")


def test_profiles_are_folded_like_the_text() -> None:
    """The text's v is folded to u, so the profile's "vil" must be found as "uil"."""
    assert "uil" in lexicon_trust._TRIGRAM_PROFILES["old_french"]
    assert lexical_plausibility("vilain", "old_french") == lexical_plausibility("uilain", "old_french")


def test_words_in_other_scripts_are_not_scored() -> None:
    assert lexicon_trust._extract_trigrams("λόγος καὶ ἀλήθεια") == []


def test_a_diplomatic_line_is_as_plausible_as_its_expansion() -> None:
    expanded = DIPLOMATIC_FRENCH.replace("cõciẽce", "conscience").replace("⁊", "et").replace("ꝑiure", "pariure")
    assert lexical_plausibility(DIPLOMATIC_FRENCH, "old_french") >= lexical_plausibility(expanded, "old_french") - 0.10


# --------------------------------------------------------------- contrast --


@pytest.mark.parametrize("page_id", MEASURED)
def test_a_faithful_page_is_not_risky_by_its_contrast(page_id: str) -> None:
    for text in (PAGES[page_id]["reference"], PAGES[page_id]["readings"]["kraken_catmus"]):
        contrast = lexical_contrast(text)
        assert contrast is not None
        assert contrast.ratio >= LEXICAL_CONTRAST_RISKY, (page_id, contrast)


@pytest.mark.parametrize("page_id", MEASURED)
def test_the_page_reversed_or_in_random_letters_is_unreliable_by_it(page_id: str) -> None:
    reference = PAGES[page_id]["reference"]
    for text in (reverse_words(reference), random_letters(reference)):
        contrast = lexical_contrast(text)
        assert contrast is not None
        assert contrast.ratio < LEXICAL_CONTRAST_UNRELIABLE, (page_id, contrast)


def test_the_best_fitting_language_is_reported() -> None:
    assert lexical_contrast(LATIN_PAGE).language == "latin"  # type: ignore[union-attr]
    assert lexical_contrast(FRENCH_PAGE).language in {"old_french", "middle_french", "french"}  # type: ignore[union-attr]


def test_a_mistaken_hint_cannot_lower_the_contrast() -> None:
    assert lexical_contrast(LATIN_PAGE, "catalan").ratio == lexical_contrast(LATIN_PAGE).ratio  # type: ignore[union-attr]


def test_a_text_too_short_to_measure_is_unmeasured() -> None:
    assert len(lexicon_trust._extract_trigrams(SHORT_PAGE)) < CONTRAST_MIN_TRIGRAMS
    assert lexical_contrast(SHORT_PAGE) is None


def test_the_contrast_is_a_function_of_the_text() -> None:
    assert lexical_contrast(LATIN_PAGE) == lexical_contrast(LATIN_PAGE)


def test_latin_is_told_from_french() -> None:
    for page_id in ("latin-abaton", "latin-apocalypse"):
        assert reads_as_latin(PAGES[page_id]["reference"]), page_id
        assert reads_as_latin(PAGES[page_id]["readings"]["kraken_catmus"]), page_id
    for page_id in ("old-french", "old-french-verse"):
        assert not reads_as_latin(PAGES[page_id]["reference"]), page_id


def test_best_fit_plausibility_never_falls_below_the_hint() -> None:
    assert best_fit_plausibility(LATIN, "catalan") >= lexical_plausibility(LATIN, "latin")
    assert best_fit_plausibility(LATIN, "catalan") >= lexical_plausibility(LATIN, "catalan")


def test_best_fit_plausibility_stays_neutral_without_a_profile() -> None:
    """No profile can judge German text; it must not be scored against Latin."""
    assert best_fit_plausibility("Uns ist in alten maeren wunders vil geseit", "german") == 0.50


# ------------------------------------------------------- report and gate --


def test_the_report_records_the_contrast_and_its_language() -> None:
    report = compute_quality_report(LATIN_PAGE, language="catalan")
    contrast = lexical_contrast(LATIN_PAGE)
    assert contrast is not None
    assert report.lexical_contrast == contrast.ratio
    assert report.lexical_language == "latin"
    assert report.lexical_implausibility == implausibility_from_contrast(contrast.ratio)
    assert report.lexical_plausibility == round(best_fit_plausibility(LATIN_PAGE, "catalan"), 4)
    assert report.to_dict()["lexical_contrast"] == report.lexical_contrast


def test_a_faithful_diplomatic_latin_page_is_not_blocked() -> None:
    """Regression: detected as Catalan and scored against the French profile, it was UNRELIABLE."""
    report = compute_quality_report(LATIN_PAGE, language="catalan", tiled=False)
    assert report.quality_label in {"HIGH", "OK"}
    assert report.ner_allowed and report.token_search_allowed


def test_reversed_latin_is_still_unreliable() -> None:
    report = compute_quality_report(reverse_words(LATIN_PAGE), language="latin", tiled=False)
    assert report.quality_label == "UNRELIABLE"
    assert not report.ner_allowed and not report.token_search_allowed


def test_a_page_too_short_for_the_contrast_is_judged_by_its_hit_rate() -> None:
    """Without the floor, the seven lines reversed were graded HIGH."""
    faithful = compute_quality_report(SHORT_PAGE, language="middle_french", tiled=False)
    reversed_page = compute_quality_report(reverse_words(SHORT_PAGE), language="middle_french", tiled=False)
    assert faithful.lexical_contrast == reversed_page.lexical_contrast == -1.0
    assert faithful.lexical_plausibility >= LEXICAL_PLAUSIBILITY_FLOOR
    assert faithful.quality_label in {"HIGH", "OK"}
    assert reversed_page.lexical_plausibility < LEXICAL_PLAUSIBILITY_FLOOR
    assert reversed_page.quality_label in {"RISKY", "UNRELIABLE"}


def test_the_plausibility_is_unmeasured_without_a_profile() -> None:
    assert compute_quality_report(LATIN_PAGE, language="german").lexical_plausibility == -1.0
    assert compute_quality_report(LATIN_PAGE).lexical_plausibility == -1.0


# -------------------------------------------------------------- fragments --

VERSE = [
    "C aldꝰ li roys qui ml̾t fu ber",
    "L a fist requerre ⁊ demander",
    "P or .i. suen filz ⁊ issi vint",
    "M arcadigas anõ a voit",
    "F urent ꝯ affiert a tel gent",
]


def test_a_separated_initial_is_not_a_fragment() -> None:
    """Regression: these marked 47% of a faithful verse page as cut words."""
    assert leading_fragment_ratio(VERSE) == 0.0
    assert seam_fragment_ratio(VERSE) == 0.0


def test_a_cut_word_still_is() -> None:
    assert leading_fragment_ratio(["s tibi dico", "ns ille tuus", "c sicut erat"]) > 0.0


# Every line opens with the tail of a word cut off at a tile seam.
CUT_LINES = "\n".join(
    [
        "tr in principio erat uerbum",
        "ns et uerbum erat apud deum",
        "rt et deus erat uerbum hoc",
        "x erat in principio apud deum",
        "q omnia per ipsum facta sunt",
    ]
)


def test_fragments_count_only_where_the_page_was_tiled() -> None:
    tiled = compute_quality_report(CUT_LINES, language="latin")
    whole = compute_quality_report(CUT_LINES, language="latin", tiled=False)
    assert tiled.seam_retry_required and tiled.quality_label == "RISKY"
    assert not whole.seam_retry_required and whole.quality_label in {"HIGH", "OK"}


def test_the_fragment_gate_passes_untiled_text() -> None:
    decisions = enforce_quality_gates(compute_quality_report(CUT_LINES, language="latin", tiled=False))
    assert decisions["gates"]["LEADING_FRAGMENT"]["passed"]
    assert "seam_not_resolved" not in decisions["blocked_stages"]


def test_the_fragment_gate_still_holds_tiled_text() -> None:
    decisions = enforce_quality_gates(compute_quality_report(CUT_LINES, language="latin"))
    assert not decisions["gates"]["LEADING_FRAGMENT"]["passed"]
    assert "seam_not_resolved" in decisions["blocked_stages"]


# -------------------------------------------------------------- repetition --


def test_a_block_loop_is_repetition() -> None:
    """Regression: an 8-line page emitted 16 times scored 0.14, each line being 1/8 of the lines."""
    page = "\n".join(f"line {index} of the dedication to the bishop" for index in range(8))
    assert repetition_score("\n".join([page] * 16)) >= REPETITION_HARD_LIMIT


def test_a_refrain_is_not() -> None:
    lines = [f"versus {index} psalmi" for index in range(30)] + ["Gloria patri et filio"] * 3
    assert repetition_score("\n".join(lines)) < 0.10
