"""The verifier's output parsing is linear: model output full of blank lines cannot stall the process."""

from __future__ import annotations

import time

from app.agents import paleography_verification_agent as verifier


def test_reasoning_lines_are_still_stripped_and_labels_still_read() -> None:
    raw = "Reasoning: I compared the lines.\n  Assessment: supported\nAnswer:  Caldus is the king\nNotes: none"
    assert "compared" not in verifier._strip_reasoning_leaks(raw)
    fields = verifier._salvage_structured_fields(raw)
    assert fields.get("assessment") == "supported"
    assert fields.get("verified_answer") == "Caldus is the king"


def test_blank_lines_cost_linear_time() -> None:
    # `^\\s*` under (?m): \\s matches newlines, so every line start scanned all the
    # newlines after it. 8,000 newlines took 1.9 s in _REASONING_LINE_RE and
    # 0.4 s in the "answer:" pattern; 20,000 took tens of seconds.
    raw = "x" + "\n" * 20000 + "x"
    started = time.perf_counter()
    verifier._strip_reasoning_leaks(raw)
    verifier._salvage_structured_fields(raw)
    assert time.perf_counter() - started < 0.5
