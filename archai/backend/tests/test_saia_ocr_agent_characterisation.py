"""The SAIA OCR agent's helpers give what they gave before they were moved.

The fixture was recorded from ``app/agents/saia_ocr_agent.py`` before its
image preparation, tile geometry and text sanity checks moved into
``app/services``; see ``tests/characterise_saia_ocr_agent.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("characterise_saia_ocr_agent", _TESTS / "characterise_saia_ocr_agent.py")
characterise = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = characterise
_spec.loader.exec_module(characterise)

EXPECTED = json.loads(characterise.FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def recorded() -> dict:
    return json.loads(json.dumps(characterise.record(), ensure_ascii=False, sort_keys=True))


@pytest.mark.parametrize("section", sorted(EXPECTED))
def test_every_input_is_recorded(recorded: dict, section: str) -> None:
    assert sorted(recorded[section]) == sorted(EXPECTED[section])


@pytest.mark.parametrize(("section", "name"), [(s, n) for s in sorted(EXPECTED) for n in sorted(EXPECTED[s])])
def test_helpers_match_the_recording(recorded: dict, section: str, name: str) -> None:
    got, want = recorded[section][name], EXPECTED[section][name]
    for key in sorted(set(got) | set(want)):
        assert got.get(key) == want.get(key), f"{section}/{name}: {key} changed"
