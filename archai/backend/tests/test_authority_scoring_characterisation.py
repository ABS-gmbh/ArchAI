"""Authority linking's scoring gives what it gave before it was moved.

The fixture was recorded from ``app/services/authority_linking.py`` before its
scoring and candidate filtering moved to ``app/services/authority_scoring.py``;
see ``tests/characterise_authority_scoring.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("characterise_authority_scoring", _TESTS / "characterise_authority_scoring.py")
characterise = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = characterise
_spec.loader.exec_module(characterise)

EXPECTED = json.loads(characterise.FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def recorded() -> dict:
    return json.loads(json.dumps(characterise.record(), ensure_ascii=False, sort_keys=True))


def test_every_surface_is_recorded(recorded: dict) -> None:
    assert sorted(recorded) == sorted(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_scoring_matches_the_recording(recorded: dict, name: str) -> None:
    got, want = recorded[name], EXPECTED[name]
    for key in sorted(set(got) | set(want)):
        assert got.get(key) == want.get(key), f"{name}: {key} changed"
