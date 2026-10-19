"""Mention extraction on diplomatic transcriptions, as CATMuS writes them."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_backend = Path(__file__).resolve().parents[1]
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

with contextlib.redirect_stdout(io.StringIO()):
    from app.routers import ocr as ocr_router  # type: ignore[import-untyped]  # noqa: E402
from app.services.text_normalization import is_blacklisted_token  # type: ignore[import-untyped]  # noqa: E402

READINGS = _backend / "eval" / "quality_gate" / "readings.json"
VERSE = next(p for p in json.loads(READINGS.read_text(encoding="utf-8"))["pages"] if p["page_id"] == "old-french-verse")["reference"]


def _mentions(text: str, ent_type: str | None = None) -> list[dict[str, Any]]:
    mentions, _candidates, _debug = ocr_router._extract_mentions_from_text(text)
    return [m for m in mentions if ent_type is None or m["ent_type"] == ent_type]


def _surfaces(text: str, ent_type: str) -> list[str]:
    return [m["surface"] for m in _mentions(text, ent_type)]


# ── words with abbreviation marks stay whole ────────────────────────────


def test_a_name_with_a_superscript_letter_is_one_word() -> None:
    """Regression: "mͣrcadigas" was read as "m" and "rcadigas"."""
    assert [m.group(0) for m in ocr_router._SALVAGE_TOKEN_RE.finditer("Roys mͣrcadigas ⁊ amerent")] == ["Roys", "mͣrcadigas", "amerent"]
    assert [m.group(0) for m in ocr_router._MENTION_TOKEN_RE.finditer("Caldꝰ li roys")] == ["Caldꝰ", "li", "roys"]


def test_lexicons_are_compared_by_search_key() -> None:
    assert ocr_router._word_key("Caldꝰ") == "caldus"
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
    assert sum("marcadigas" in ocr_router._word_key(s) for s in persons) == 4


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
