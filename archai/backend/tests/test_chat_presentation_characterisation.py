"""Chat's presentation helpers give what they gave before they were moved.

The fixture was recorded from ``app/services/chat_ai.py`` before its payload
and markdown helpers moved to ``app/services/chat_presentation.py``; see
``tests/characterise_chat_presentation.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("characterise_chat_presentation", _TESTS / "characterise_chat_presentation.py")
characterise = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = characterise
_spec.loader.exec_module(characterise)

EXPECTED = json.loads(characterise.FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def recorded() -> dict:
    return json.loads(json.dumps(characterise.record(), ensure_ascii=False, sort_keys=True))


def test_every_payload_is_recorded(recorded: dict) -> None:
    assert sorted(recorded) == sorted(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_presentation_matches_the_recording(recorded: dict, name: str) -> None:
    got, want = recorded[name], EXPECTED[name]
    for key in sorted(set(got) | set(want)):
        assert got.get(key) == want.get(key), f"{name}: {key} changed"
