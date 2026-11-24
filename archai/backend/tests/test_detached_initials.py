"""Versal initials Kraken writes apart from their word are joined, on geometry alone.

Records and cuts are stubbed: no image, no model.
"""

from __future__ import annotations

import unicodedata
from types import SimpleNamespace

import pytest

from app.services import ocr_backends
from app.services.detached_initials import join_first_glyph, leading_glyph, lines_to_join

RULE = {"min_column_lines": 5, "min_share": 0.5, "min_gap_ratio": 1.0}


def _cuts(text: str, first_gap: float, word_gap: float = 10.0, glyph: float = 8.0) -> list[tuple[float, float, float, float]]:
    """One bbox cut per character: glyphs *glyph* wide, word gaps *word_gap*, the first space *first_gap*."""
    cuts, x = [], 0.0
    first_space = True
    for char in text:
        if char == " ":
            width = first_gap if first_space else word_gap
            first_space = False
            cuts.append((x, 0.0, x + width, 10.0))
            x += width
        elif unicodedata.combining(char):
            cuts.append((x - glyph, 0.0, x, 10.0))  # a mark sits over its letter
        else:
            cuts.append((x, 0.0, x + glyph, 10.0))
            x += glyph
    return cuts


def test_the_gap_after_a_detached_glyph_is_measured_against_the_word_gaps() -> None:
    measured = leading_glyph("C aldꝰ li roys", _cuts("C aldꝰ li roys", first_gap=25.0))
    assert measured == {"detached": True, "gap_ratio": 2.5}
    qui = unicodedata.normalize("NFD", "Qͥ gnͣt conseil uilain")
    assert leading_glyph(qui, _cuts(qui, first_gap=6.0))["gap_ratio"] == 0.6  # a word at word spacing
    assert leading_glyph("C aldꝰ", _cuts("C aldꝰ", first_gap=25.0)) == {"detached": True, "gap_ratio": None}
    assert leading_glyph("Caldꝰ li roys", _cuts("Caldꝰ li roys", first_gap=10.0)) == {"detached": False, "gap_ratio": None}
    assert leading_glyph("C ald", [(0, 0, 1, 1)]) is None  # cuts that do not match the prediction


def test_the_kraken_backend_records_it_for_a_one_line_crop() -> None:
    record = SimpleNamespace(prediction=" M arcadigas anõ", cuts=[(0, 0, 0, 0)] + _cuts("M arcadigas anõ", first_gap=20.0))
    assert ocr_backends._leading_glyph_of([record]) == {"detached": True, "gap_ratio": 2.0}
    assert ocr_backends._leading_glyph_of([record, record]) is None  # several records: not judged


def _line(text: str, *, y: float, x0: float = 100.0, ratio: float | None = 2.0) -> dict:
    detached = text[1:2] == " " or (unicodedata.combining(text[1:2] or "a") and " " in text[:4])
    return {"text": text, "bbox_xyxy": [x0, y, x0 + 600.0, y + 40.0], "leading_glyph": {"detached": bool(detached), "gap_ratio": ratio}}


def _column(texts: list[str], x0: float = 100.0, ratio: float | None = 2.0) -> list[dict]:
    return [_line(t, y=100.0 + 50 * i, x0=x0, ratio=ratio) for i, t in enumerate(texts)]


def test_a_column_that_sets_its_initials_apart_is_joined() -> None:
    lines = _column(["C aldꝰ li roys", "L a fist requerre", "P or .i. sueu filz", "M arcadigas anõ", "Sages ⁊ pͦuz", "y nabele ⁊ marcadigas"])
    chosen = lines_to_join(lines, **RULE)
    assert chosen == {0, 1, 2, 3, 5}
    assert [join_first_glyph(lines[i]["text"]) for i in sorted(chosen)][:2] == ["Caldꝰ li roys", "La fist requerre"]


def test_a_column_where_only_two_of_eighteen_lines_start_detached_is_left_alone() -> None:
    texts = ["R Rian geliste iohannisi", "R habe bat murumenag"] + [f"line {i} of the lection" for i in range(16)]
    assert lines_to_join(_column(texts), **RULE) == set()


def test_a_word_sign_and_a_word_at_word_spacing_stay_apart() -> None:
    lines = _column(["C aldꝰ li roys", "L a fist requerre", "P or .i. sueu filz", "⁊ larges ⁊ cortoiz", "M arcadigas anõ"])
    lines.append(_line("Q̾ r ia uilain ne", y=400.0, ratio=0.99))
    chosen = lines_to_join(lines, **RULE)
    assert 3 not in chosen  # ⁊ is a word sign, not an initial, even with a versal-sized gap
    assert 5 not in chosen  # Q̾ at word spacing
    assert chosen == {0, 1, 2, 4}


def test_a_line_with_no_other_gap_counts_as_passing_and_a_short_column_is_not_judged() -> None:
    lines = _column(["C ald", "L a fist", "P or", "M arcadigas", "N e senfͣmerent"], ratio=None)
    assert lines_to_join(lines, **RULE) == {0, 1, 2, 3, 4}
    assert lines_to_join(lines[:4], **RULE) == set()  # four lines: below the column minimum


def test_columns_are_judged_apart() -> None:
    verse = _column(["C aldꝰ li roys", "L a fist", "P or cen", "M arcadigas", "N e senfͣmerent"], x0=100.0)
    # The facing column starts one line in five with a one-letter word: under half, so it is not judged.
    prose = _column(["a son esciẽt il se", "fet contre ce", "peche mortieument", "est a antendre", "delibacion mes"], x0=900.0)
    assert lines_to_join(verse + prose, **RULE) == {0, 1, 2, 3, 4}


@pytest.mark.parametrize("text", ["Caldꝰ li roys", "C", "C  ald", " C ald"])
def test_text_that_does_not_start_detached_is_returned_as_is(text: str) -> None:
    assert join_first_glyph(text) == text
