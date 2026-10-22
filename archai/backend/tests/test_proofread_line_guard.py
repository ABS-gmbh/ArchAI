"""The line-level proofreader guard: corrections are kept, inventions put back as read."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_backend = Path(__file__).resolve().parents[1]
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from app.agents.ocr_proofreader_agent import OcrProofreaderAgent  # type: ignore[import-untyped]  # noqa: E402
from app.services.proofread_guard import guard_proofread, judge_line  # type: ignore[import-untyped]  # noqa: E402

PAGE = "\n".join(
    [
        "Sage fu enfez ⁊ endiz",
        "Caldꝰ li roys qui ml̾t fu ber",
        "La fist reqerre ⁊ demander",
        "Por .i. suen filz ⁊ issi vint",
        "Que li mariages auint",
    ]
)


# ── single lines ────────────────────────────────────────────────────────


def test_a_letter_level_correction_is_kept() -> None:
    assert judge_line("La fist reqerre ⁊ demander", "La fist requerre ⁊ demander") is None
    assert judge_line("dorninus noster", "dominus noster") is None


def test_a_word_split_or_joined_is_a_correction() -> None:
    assert judge_line("ſanctifi etur nomen tuu", "ſanctifietur nomen tuu") is None


def test_an_invented_word_is_reverted() -> None:
    assert "word invented" in str(judge_line("La fist reqerre ⁊ demander", "La fist chevalier ⁊ demander"))


def test_a_line_from_elsewhere_is_reverted() -> None:
    assert "rewritten" in str(judge_line("Que li mariages auint", "In principio erat uerbum et uerbum"))


def test_a_filled_gap_is_reverted() -> None:
    assert "uncertain" in str(judge_line("Caldꝰ li […] qui ml̾t fu ber", "Caldꝰ li roys qui ml̾t fu ber"))


# ── pages ───────────────────────────────────────────────────────────────


def test_only_the_invented_line_of_a_page_goes_back() -> None:
    proposed = PAGE.replace("reqerre", "requerre").replace("Que li mariages auint", "Que li rois Artus vint")
    guarded = guard_proofread(PAGE, proposed)
    lines = guarded.text.splitlines()
    assert lines[2] == "La fist requerre ⁊ demander"  # the correction stays
    assert lines[4] == "Que li mariages auint"  # the invention goes
    assert (guarded.lines_kept, guarded.lines_reverted) == (4, 1)


def test_added_lines_are_dropped_and_deleted_lines_restored() -> None:
    lines = PAGE.splitlines()
    added = guard_proofread(PAGE, "\n".join([*lines, "Explicit liber primus de natura rerum"]))
    assert added.text == PAGE and added.lines_dropped == 1
    deleted = guard_proofread(PAGE, "\n".join(lines[:2] + lines[3:]))
    assert deleted.text == PAGE and deleted.lines_restored == 1


def test_a_set_apart_initial_joined_to_its_line_is_kept() -> None:
    guarded = guard_proofread("R\neuerendi patris\net domini", "Reuerendi patris\net domini")
    assert guarded.text == "Reuerendi patris\net domini"
    assert not guarded.changed


def test_a_recited_prayer_does_not_replace_the_reading() -> None:
    garbled = "Pat noſter qui eſ in celis\nſanctifi etur nomen tuu\naduen at regnu tuu\nfi at uolutas tua"
    recital = "\n".join(
        [
            "Pater noster qui es in caelis sanctificetur nomen tuum",
            "adveniat regnum tuum fiat voluntas tua",
            "sicut in caelo et in terra panem nostrum quotidianum da nobis hodie",
            "et dimitte nobis debita nostra sicut et nos dimittimus debitoribus nostris",
        ]
    )
    assert "panem" not in guard_proofread(garbled, recital).text


# ── the agent ───────────────────────────────────────────────────────────


class _Client:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def list_models(self, force_refresh: bool = False) -> list[str]:
        return ["model"]

    def chat_completion(self, **_kwargs: Any) -> dict[str, Any]:
        return {"text": self.answer}


def test_the_agent_returns_the_guarded_text() -> None:
    """Every call site gets the guard: two used to take the model's text unchecked."""
    invented = PAGE.replace("Que li mariages auint", "Que li rois Artus vint")
    agent = OcrProofreaderAgent(client=_Client(invented), model_override="model")
    assert agent.proofread(PAGE, "latin") == PAGE


# ── the calibration ─────────────────────────────────────────────────────


def test_no_invented_line_survives_on_the_reference_pages() -> None:
    spec = importlib.util.spec_from_file_location("calibrate_proofread_guard", _backend / "scripts" / "calibrate_proofread_guard.py")
    calibration = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = calibration
    try:
        spec.loader.exec_module(calibration)
        pages = calibration.json.loads(calibration.READINGS.read_text(encoding="utf-8"))["pages"][:2]
        cases = [case for case in calibration.page_cases(pages) if case["invented"]]
    finally:
        sys.modules.pop(spec.name, None)
    for case in cases:
        text = guard_proofread(case["raw"], case["proof"]).text.splitlines()
        assert not any(line in text for line in case["invented"]), case["page"]
