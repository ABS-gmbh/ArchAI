"""extract_high_recall_mentions keeps its capital requirement, and its docstring says so (audit SVC-24).

The audit read the docstring ("capitalization-independent: scan ALL tokens
>= 4 chars") as the intent and the capital check as the bug. Measured on the
gold pages, the docstring's version adds 246 false positives where the code
adds 22, with no entity found by either: the code was right, the docstring
wrong.
"""

from __future__ import annotations

from app.services.pipeline_hardening import extract_high_recall_mentions


def _capitalised(text: str) -> list[str]:
    return [m["surface"] for m in extract_high_recall_mentions(text) if m["method"] == "high_recall:capitalized"]


def test_capitalised_long_tokens_are_candidates_and_lowercase_ones_are_not() -> None:
    assert _capitalised("puis vint Marcadigas en la cite") == ["Marcadigas"]
    assert _capitalised("puis vint marcadigas en la cite") == []
    assert _capitalised("Caldus") == ["Caldus"]
    assert _capitalised("Ader") == []  # under five characters


def test_the_docstring_no_longer_promises_capitalisation_independence() -> None:
    doc = extract_high_recall_mentions.__doc__ or ""
    assert "Capitalization-independent" not in doc
    assert "Capitalised tokens of 5 or more characters" in doc
