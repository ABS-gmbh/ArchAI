"""The mypy error count matches the baseline README.md records.

README.md recorded 89 errors on 2026-11-23; four commits later mypy reported 95,
and nothing noticed. This runs mypy (where it is installed; CI's backend job
does not install it) and compares the count with the README's figure, both
ways: a new error fails, and so does a fixed one until the README says so.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mypy")

BACKEND = Path(__file__).resolve().parents[1]
_RECORDED = re.compile(r"Baseline on [\d-]+ \(mypy [^)]*\): (\d+) errors in (\d+) files")
_REPORTED = re.compile(r"Found (\d+) errors? in (\d+) files?")


def _recorded() -> tuple[int, int]:
    match = _RECORDED.search((BACKEND / "README.md").read_text(encoding="utf-8"))
    assert match, "README.md no longer states the mypy baseline in the form this test reads"
    return int(match.group(1)), int(match.group(2))


def test_the_error_count_is_the_recorded_baseline() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mypy"], cwd=BACKEND, capture_output=True, text=True, timeout=600
    )
    summary = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else result.stderr
    found = _REPORTED.search(summary)
    count = (int(found.group(1)), int(found.group(2))) if found else (0, 0)
    assert "Success: no issues found" in summary or found, summary
    assert count == _recorded(), (
        f"mypy reports {count[0]} errors in {count[1]} files; README.md records {_recorded()[0]} in "
        f"{_recorded()[1]}. Fix a new error, or update the baseline paragraph (\"Type checking\")."
    )
