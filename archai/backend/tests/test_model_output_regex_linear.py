"""Parsing model output is linear: long runs of whitespace cannot stall the process.

Each of these patterns let `\\s*` or `\\s+` start at every position of a
whitespace run (or every line start under (?m)), which is quadratic: 8,000
characters took 0.1 to 1.9 s, and a model that loops on blank lines writes more.
(The translation-note patterns were fixed the same way, but their input has its
whitespace collapsed first, so they are not timed here.)
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from app.agents import label_analysis_agent, ocr_proofreader_agent, ocr_text_parsing, saia_ocr_agent
from app.services import chat_presentation, glm_ollama_ocr

RUNS = ["\n" * 20000, "\t" * 20000, " \n" * 10000]

CASES: list[tuple[str, Callable[[str], object]]] = [
    ("verification value", lambda run: chat_presentation._clean_verification_value("x" + run + "x")),
    ("ocr fences", lambda run: ocr_text_parsing.sanitize_ocr_text("```\nli rois" + run + "x")),
    ("json fences", lambda run: ocr_text_parsing._extract_json_candidate("```\nli rois" + run + "x")),
    ("proofreader fences", lambda run: ocr_proofreader_agent._clean_plain_text("```\nli rois" + run + "x")),
    ("saia fences", lambda run: saia_ocr_agent._strip_fences("```\nli rois" + run + "x")),
    ("glm note", lambda run: glm_ollama_ocr.clean_model_output("x" + run + "x")),
    ("initial letter", lambda run: label_analysis_agent._INITIAL_LETTER_VALUE_RE.search("ab" + run + "ab")),
    ("initial confidence", lambda run: label_analysis_agent._INITIAL_LETTER_CONFIDENCE_RE.search("ab" + run + "ab")),
]


@pytest.mark.parametrize(("name", "call"), CASES, ids=[name for name, _ in CASES])
def test_whitespace_runs_cost_linear_time(name: str, call: Callable[[str], object]) -> None:
    for run in RUNS:
        started = time.perf_counter()
        call(run)
        assert time.perf_counter() - started < 0.3, (name, repr(run[:4]))


def test_the_rewritten_patterns_match_as_before() -> None:
    assert chat_presentation._normalize_translation_output("The king came.  Note: literal") == "The king came."
    assert ocr_text_parsing.sanitize_ocr_text("```text\nli rois\n  ```").strip() == "li rois"
    assert saia_ocr_agent._strip_fences("```json\n{}\n ```") == "{}"
    assert glm_ollama_ocr.clean_model_output("li rois\n\n  Note: unsure").strip() == "li rois"
    assert label_analysis_agent._INITIAL_LETTER_VALUE_RE.search("  Letter: Q\n").group(1) == "Q"
    assert "thinking" not in chat_presentation._clean_verification_value("Reasoning: thinking\nsupported")
