"""A looping GLM-OCR reading is cut to its first copy; a manuscript's own repeats are kept.

On the user's Mac (Ollama 0.40.6, 2026-11-25) a clean two-line test image
came back as the two lines repeated 177 times, with a markdown fence around
most copies: 531 "lines", 7,432 characters, all shown in the transcription
pane. Worker 2's saved readings of the reference pages show the same
signature: one line 52 times, blocks of 7, 21 and 28 lines 3 to 16 times,
always back to back to the end of the text, the last copy often cut short.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.services import glm_ollama_ocr
from app.services.glm_ollama_ocr import clean_model_output, cut_repetition_loop

PAIR = "Caldus de Lausanne\nanno domini MCCXIV"


def _the_531_line_reading() -> str:
    """The shape seen: the pair, then fenced copies of it, until the tokens run out mid-line."""
    copies = [PAIR, "```markdown\n" + PAIR + "\n```"] + ["```\n" + PAIR + "\n```"] * 175
    return "\n".join(copies) + "\nCaldus de Lausanne\nanno dom"


def test_the_loop_is_cut_to_its_first_copy() -> None:
    text, copies = cut_repetition_loop(clean_model_output(_the_531_line_reading()))
    assert text == PAIR
    assert copies == 178


def _lines(prefix: str, count: int, width: int) -> list[str]:
    """*count* different lines about *width* characters long."""
    return [f"{prefix} {i} ".ljust(width, "x") for i in range(count)]


# The five loops measured on 2026-11-25, rebuilt with their shape: lines before
# the loop, the repeated block's length and line width, the copies, and how
# many lines of a last, cut-short copy. Worker 2's readings themselves stay out
# of the repository; their counts are in cut_repetition_loop's docstring.
def _old_french() -> str:  # 105 lines: one 56-character line 52 times after 53 others
    return "\n".join(_lines("verse", 53, 56) + ["Source gallica.bnf.fr / Bibliotheque Mazarine. Ms 870-1"] * 52)


def _latin_abaton() -> str:  # 126 lines: a 28-line block 3 times, then 2 lines of a 4th, after 40 others
    block = _lines("abaton", 28, 35)
    return "\n".join(_lines("prologue", 40, 35) + block * 3 + [block[0], block[1][:20]])


def _latin_apocalypse() -> str:  # 161 lines: a 21-line block 7 times and 14 lines of an 8th
    block = _lines("apoc", 21, 18)
    return "\n".join(block * 7 + block[:13] + [block[13][:2]])


def _middle_french_antitus() -> str:  # 110 lines: a 7-line block 15 times and 5 lines of a 16th
    block = _lines("antitus", 7, 32)
    return "\n".join(block * 15 + block[:4] + [block[4][:25]])


@pytest.mark.parametrize(
    ("reading", "kept_lines"),
    [(_old_french, 54), (_latin_abaton, 68), (_latin_apocalypse, 21), (_middle_french_antitus, 7),
     (lambda: clean_model_output(_the_531_line_reading()), 2)],
    ids=["old-french", "latin-abaton", "latin-apocalypse", "middle-french-antitus", "two-line-test"],
)
def test_the_measured_loops_are_cut_without_ollamas_stop_reason(reading: Any, kept_lines: int) -> None:
    text, copies = cut_repetition_loop(reading())
    assert len(text.split("\n")) == kept_lines
    assert copies >= 3


@pytest.mark.parametrize(
    "kept",
    [
        # A page that ends on its own repeats: the loop's periodic tail, but nowhere near a loop's size.
        "Pater noster qui es in caelis\nsanctificetur nomen tuum\nAmen\nAmen\nAmen",
        "Kyrie eleison\nChriste eleison\nKyrie eleison\nKyrie eleison\nKyrie eleison",
        "Ecce ancilla domini\nOra pro nobis\nRefrain one\nRefrain two\nRefrain one\nRefrain two\nRefrain one\nRefrain two",
        # A refrain sung twice at the end of the page.
        "Ecce ancilla domini\nOra pro nobis\nSancta Dei genitrix\nOra pro nobis\nRefrain one\nRefrain two\nRefrain one\nRefrain two",
        # A block three times in a row, but the page goes on after it: not to the end.
        "Kyrie eleison\nKyrie eleison\nKyrie eleison\nChriste eleison",
        # The same line many times, but never back to back.
        "Amen\nfirst prayer\nAmen\nsecond prayer\nAmen\nthird prayer\nAmen",
        # Nothing repeated.
        PAIR,
    ],
)
def test_a_manuscripts_own_repeats_are_kept(kept: str) -> None:
    assert cut_repetition_loop(kept) == (kept, 0)


def test_a_reading_that_stopped_by_itself_is_never_cut() -> None:
    looping = clean_model_output(_the_531_line_reading())
    assert cut_repetition_loop(looping, ran_out=False) == (looping, 0)


def test_a_reading_that_ran_out_of_tokens_is_cut_at_its_periodic_tail_whatever_its_size() -> None:
    assert cut_repetition_loop("Pater noster\nAmen\nAmen\nAmen", ran_out=True) == ("Pater noster\nAmen", 3)
    # Ran out, but no periodic tail: nothing to cut.
    assert cut_repetition_loop(PAIR, ran_out=True) == (PAIR, 0)


def test_ollamas_stop_reason_is_read() -> None:
    assert glm_ollama_ocr.OllamaText("x", "length").ran_out is True
    assert glm_ollama_ocr.OllamaText("x", "stop").ran_out is False
    assert glm_ollama_ocr.OllamaText("x").ran_out is None
    assert glm_ollama_ocr.OllamaText("x", "length") == "x"


def test_a_wrapping_fence_and_fence_lines_go() -> None:
    assert clean_model_output("```\n" + PAIR + "\n```") == PAIR
    assert clean_model_output("```markdown\n" + PAIR + "\n```") == PAIR
    assert clean_model_output(PAIR + "\n```\nlast line") == PAIR + "\nlast line"
    # A fence inside a line of text is not a fence line.
    assert clean_model_output("a ``` b") == "a ``` b"


def test_the_reading_says_it_was_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    variant = glm_ollama_ocr.PreparedOcrImage(
        name="rgb_jpeg_1536", image_bytes=b"v", width=900, height=300, size_bytes=15361,
        resized=False, compressed=False, applied_autocontrast=False, jpeg_quality=90,
    )
    monkeypatch.setattr(glm_ollama_ocr, "preprocess_variants", lambda image_bytes, *, max_payload_bytes: ((900, 300), [variant]))

    def looping(**_kwargs: Any) -> str:
        return _the_531_line_reading()

    monkeypatch.setattr(glm_ollama_ocr, "ollama_extract", looping)
    result = glm_ollama_ocr.run_glm_ollama_ocr(
        b"page", image_ref="page-1", model="glm-ocr:latest", host="http://localhost:11434",
        timeout=30, temperature=0.0, retries=1, max_payload_bytes=2_000_000,
    )
    assert result.text == PAIR
    assert result.lines == PAIR.split("\n")
    assert "OCR_REPETITION_CUT:178" in result.warnings


def test_a_reading_ollama_says_stopped_by_itself_keeps_its_repeats(monkeypatch: pytest.MonkeyPatch) -> None:
    variant = glm_ollama_ocr.PreparedOcrImage(
        name="rgb_jpeg_1536", image_bytes=b"v", width=900, height=300, size_bytes=15361,
        resized=False, compressed=False, applied_autocontrast=False, jpeg_quality=90,
    )
    monkeypatch.setattr(glm_ollama_ocr, "preprocess_variants", lambda image_bytes, *, max_payload_bytes: ((900, 300), [variant]))
    page = "Pater noster qui es in caelis\nAmen\nAmen\nAmen"
    monkeypatch.setattr(glm_ollama_ocr, "ollama_extract", lambda **_kwargs: glm_ollama_ocr.OllamaText(page, "stop"))
    result = glm_ollama_ocr.run_glm_ollama_ocr(
        b"page", image_ref="page-1", model="glm-ocr:latest", host="http://localhost:11434",
        timeout=30, temperature=0.0, retries=1, max_payload_bytes=2_000_000,
    )
    assert result.text == page
    assert not [warning for warning in result.warnings if warning.startswith("OCR_REPETITION_CUT")]
