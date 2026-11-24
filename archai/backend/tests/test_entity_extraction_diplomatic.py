"""Mention extraction on diplomatic transcriptions, as CATMuS writes them."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_backend = Path(__file__).resolve().parents[1]
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from app.services import mention_extraction  # type: ignore[import-untyped]  # noqa: E402
from app.services.text_normalization import is_blacklisted_token  # type: ignore[import-untyped]  # noqa: E402

READINGS = _backend / "eval" / "quality_gate" / "readings.json"
VERSE = next(p for p in json.loads(READINGS.read_text(encoding="utf-8"))["pages"] if p["page_id"] == "old-french-verse")["reference"]


def _mentions(text: str, ent_type: str | None = None) -> list[dict[str, Any]]:
    mentions, _candidates, _debug = mention_extraction._extract_mentions_from_text(text)
    return [m for m in mentions if ent_type is None or m["ent_type"] == ent_type]


def _surfaces(text: str, ent_type: str) -> list[str]:
    return [m["surface"] for m in _mentions(text, ent_type)]


# ── words with abbreviation marks stay whole ────────────────────────────


def test_a_name_with_a_superscript_letter_is_one_word() -> None:
    """Regression: "mͣrcadigas" was read as "m" and "rcadigas"."""
    assert [m.group(0) for m in mention_extraction._SALVAGE_TOKEN_RE.finditer("Roys mͣrcadigas ⁊ amerent")] == ["Roys", "mͣrcadigas", "amerent"]
    assert [m.group(0) for m in mention_extraction._MENTION_TOKEN_RE.finditer("Caldꝰ li roys")] == ["Caldꝰ", "li", "roys"]


def test_lexicons_are_compared_by_search_key() -> None:
    assert mention_extraction._word_key("Caldꝰ") == "caldus"
    assert is_blacklisted_token("uoie")  # "voie" is listed


# ── titles and names ────────────────────────────────────────────────────


def test_a_name_after_the_nominative_title() -> None:
    assert any("Marcadigas" in s for s in _surfaces("Roys Marcadigas ⁊ amerent", "person"))


def test_an_unlisted_name_needs_a_capital_or_a_second_occurrence() -> None:
    once = "Roys mͣrcadigas ⁊ amerent"
    assert _surfaces(once, "person") == []
    twice = once + "\nPor cen marcadigas haoit"
    assert len(_surfaces(twice, "person")) == 2


def test_a_name_before_li_rois() -> None:
    assert any(s.startswith("Caldꝰ") for s in _surfaces("Sage fu enfez\nCaldꝰ li roys qui ml̾t fu ber", "person"))


def test_a_latin_title_introduces_a_name() -> None:
    assert any("iohannis" in s for s in _surfaces("reuelatio. beati apostoli\n& euangelistæ iohannis:\nIn diebus illis.", "person"))


def test_a_conjunction_or_a_line_end_ends_a_name() -> None:
    """Regression: "Roys mͣrcadigas ⁊ amerent / Quer" made "Quer" a name."""
    for surface in _surfaces("Roys mͣrcadigas ⁊ amerent\nQuer molt fu sagez", "person"):
        assert "amerent" not in surface and "Quer" not in surface


def test_an_unlisted_lowercase_word_once_after_a_title_is_no_name() -> None:
    """Held-out OCR readings: "roys enguit honeur monta" made a king of "enguit"."""
    assert _surfaces("Cis roys enguit honeur monta.\nPor cen q lez gentilz ama", "person") == []


def test_a_found_name_is_found_everywhere_on_the_page() -> None:
    persons = _surfaces(VERSE, "person")
    assert sum("marcadigas" in mention_extraction._word_key(s) for s in persons) == 4


# ── places ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", ["Gent estraite de uilanie", "Qui de lignage erent gentil", "Honte qnͣt de vilain lui vient"])
def test_a_lowercase_common_word_after_de_is_no_place(text: str) -> None:
    assert _surfaces(text, "place") == []


def test_a_known_place_after_de_still_is() -> None:
    assert _surfaces("euesque et prince de lausanne.", "place") == ["lausanne"]


# ── the benchmark ───────────────────────────────────────────────────────


def test_the_reference_pages_score_as_measured() -> None:
    spec = importlib.util.spec_from_file_location("benchmark_entities", _backend / "scripts" / "benchmark_entities.py")
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    try:
        spec.loader.exec_module(bench)
        result = bench.run()
    finally:
        sys.modules.pop(spec.name, None)
    assert (result["all"]["tp"], result["all"]["fp"], result["all"]["fn"]) == (9, 0, 5)
    assert result["pages"]["old-french"]["fp"] == result["pages"]["latin-abaton"]["fp"] == 0


def test_the_high_recall_pass_adds_only_false_positives_on_the_reference_pages() -> None:
    """What the pipeline stores: it runs only where the standard pass found too little (SVC-24)."""
    spec = importlib.util.spec_from_file_location("benchmark_entities", _backend / "scripts" / "benchmark_entities.py")
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    try:
        spec.loader.exec_module(bench)
        result = bench.run(with_high_recall=True)
    finally:
        sys.modules.pop(spec.name, None)
    assert (result["all"]["tp"], result["all"]["fp"], result["all"]["fn"]) == (9, 22, 5)
    # It triggers on the two pages with no entity, and nowhere else.
    assert {pid for pid, page in result["pages"].items() if page["fp"]} == {"old-french", "latin-abaton"}


# ── lexicon gaps found on the held-out B-dev pages ─────────────────────


def test_the_middle_french_sainct_introduces_a_name() -> None:
    """"sainct Pol", "sainct Frãcois" (B-dev, Paris BnF Rés. Ye 1325): the list had saint and sainz only."""
    assert "sainct Pol" in _surfaces("¶ Sainct Pierre print le glaiue de sainct Pol", "person")
    # The span runs on to the next word, as after the other titles; the
    # authority query is trimmed separately.
    assert any(s.startswith("sainct Frãcois") for s in _surfaces("Cuydant trouuer sainct Frãcois lasche ⁊ mol", "person"))


def test_de_la_inside_a_capitalised_name_does_not_cut_it() -> None:
    """"Rommant de la Rose" (B-dev, Paris BnF fr. 604): "la" after "de" broke the run of capitals."""
    surfaces = [m["surface"] for m in _mentions("Item les epiltres sur les Rommant de la Rose")]
    assert "Rommant de la Rose" in surfaces
    assert [m["surface"] for m in _mentions("Iehan de le Mote et Pierre de les Pres")] == ["Iehan de le Mote", "Pierre de les Pres"]


def test_la_without_de_still_ends_a_run() -> None:
    assert "Rommant la Rose" not in [m["surface"] for m in _mentions("Item Rommant la Rose")]


# ── sentence punctuation ends a run of capitals ────────────────────────


def test_a_sentence_point_ends_a_name() -> None:
    """"Dominos Franciscum. Iacobum" was one mention across the point, and "Sancti Marci. In" took the next sentence's first word."""
    surfaces = [m["surface"] for m in _mentions("deuenire in Dominos Franciscum. Iacobum. et Philippũ.")]
    assert "Dominos Franciscum" in surfaces
    assert not any("." in s for s in surfaces)  # no mention runs across a point
    assert not any(s.endswith(" In") for s in [m["surface"] for m in _mentions("Procuratorum Sancti Marci. In quo")])
    assert [m["surface"] for m in _mentions("Ducentesimo. Sexagesimo")] == []


def test_a_point_after_an_abbreviated_word_does_not_end_the_name() -> None:
    """"Theob̃. Walt̃" (B-dev, Bodleian Rawl. B 499): the point abbreviates Theobaldus."""
    assert "Theob̃. Walt̃" in [m["surface"] for m in _mentions("Theob̃. Walt̃ dedit Will̃mo de burc")]
