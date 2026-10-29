"""The OCR agent's parsing and tiling helpers give what they gave before they were moved.

The fixture was recorded from ``app/agents/ocr_agent.py`` before its text
parsing and tile building moved out; see
``tests/characterise_ocr_agent_helpers.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("characterise_ocr_agent_helpers", _TESTS / "characterise_ocr_agent_helpers.py")
characterise = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = characterise
_spec.loader.exec_module(characterise)

EXPECTED = json.loads(characterise.FIXTURE.read_text(encoding="utf-8"))
CASES = [(s, n) for s in sorted(EXPECTED) if isinstance(EXPECTED[s], dict) for n in sorted(EXPECTED[s])]


@pytest.fixture(scope="module")
def recorded() -> dict:
    return json.loads(json.dumps(characterise.record(), ensure_ascii=False, sort_keys=True))


def test_every_input_is_recorded(recorded: dict) -> None:
    assert sorted(recorded) == sorted(EXPECTED)
    for section, value in EXPECTED.items():
        if isinstance(value, dict):
            assert sorted(recorded[section]) == sorted(value)


def test_model_choice_matches_the_recording(recorded: dict) -> None:
    assert recorded["models"] == EXPECTED["models"]


@pytest.mark.parametrize(("section", "name"), CASES)
def test_helpers_match_the_recording(recorded: dict, section: str, name: str) -> None:
    got, want = recorded[section][name], EXPECTED[section][name]
    for key in sorted(set(got) | set(want)):
        assert got.get(key) == want.get(key), f"{section}/{name}: {key} changed"
