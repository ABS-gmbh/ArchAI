"""The OCR retry loop keeps the attempt that passed the quality gates (audit OCR-002).

The loop in ocr_page_with_trace stops at the first attempt that passes every
gate. It used to keep that attempt only if its quality label or gibberish score
beat the attempts before it, so the scenario from the audit went wrong:

  attempt 0: RISKY, gibberish 0.10, gates fail (NER blocked)
  attempt 1: RISKY, gibberish 0.20, gates pass  -> loop breaks

attempt 1 was not "better", the run went on with attempt 0's text and its
failing gate decisions, and the hard stop was skipped because the gates had
passed once. Gate outcome is now the first key of the ranking.
"""

from __future__ import annotations

import ast
from pathlib import Path

from app.routers.ocr import _attempt_rank

OCR_ROUTER = Path(__file__).resolve().parents[1] / "app" / "routers" / "ocr.py"


def _best(attempts):
    best = None
    for index, attempt in enumerate(attempts):
        key = _attempt_rank(*attempt)
        if best is None or key < best[0]:
            best = (key, index)
    return best[1]


def test_audit_scenario_keeps_the_passing_attempt():
    assert _best([(False, "RISKY", 0.10), (True, "RISKY", 0.20)]) == 1


def test_passing_beats_a_better_label_that_failed():
    assert _best([(False, "HIGH", 0.0), (True, "UNRELIABLE", 0.9)]) == 1


def test_label_then_gibberish_break_ties_within_a_gate_outcome():
    assert _best([(False, "RISKY", 0.1), (False, "OK", 0.5)]) == 1
    assert _best([(False, "OK", 0.3), (False, "OK", 0.2)]) == 1
    assert _best([(False, "OK", 0.2), (False, "OK", 0.2)]) == 0
    assert _best([(True, "OK", 0.2), (True, "HIGH", 0.4)]) == 1


def test_unknown_label_ranks_last():
    assert _best([(False, "???", 0.0), (False, "UNRELIABLE", 0.9)]) == 1


def test_retry_loop_ranks_attempts_with_the_gate_outcome():
    tree = ast.parse(OCR_ROUTER.read_text(encoding="utf-8"))
    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "ocr_page_with_trace"
    )
    calls = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_attempt_rank"
    ]
    assert len(calls) == 1
    assert ast.unparse(calls[0].args[0]) == "all_gates_passed"
