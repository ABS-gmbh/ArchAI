"""uncertainty_density is a fraction of characters: a "[…]" span counts 3, not 4 (audit SVC-17)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from app.services.ocr_text_metrics import uncertainty_density

BACKEND = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("text", "density"),
    [("[…]", 1.0), ("ab[…]", 3 / 5), ("a?b", 1 / 3), ("a…b", 1 / 3), ("[…] … ?", (3 + 1 + 1) / 7), ("abc", 0.0), ("", 0.0)],
)
def test_density_counts_each_marker_character_once(text: str, density: float) -> None:
    assert uncertainty_density(text) == pytest.approx(density)


def test_density_never_exceeds_one() -> None:
    assert uncertainty_density("[…][…][…]") == 1.0  # was 4/3


def test_the_marked_set_script_runs_on_a_small_readings_file(tmp_path: Path) -> None:
    import json

    spec = importlib.util.spec_from_file_location("calibrate_uncertainty_markers", BACKEND / "scripts" / "calibrate_uncertainty_markers.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert module.density_before_svc17("[…]") == pytest.approx(4 / 3)
        rows = module.run(BACKEND / "eval" / "quality_gate" / "readings.json")
    finally:
        sys.modules.pop(spec.name, None)
    assert len(rows) == 29
    assert all(row["density"] < row["density_before_svc17"] for row in rows)
    verse = next(r for r in rows if r["page"] == "old-french-verse" and r["reading"] == "kraken_catmus")
    assert (verse["markers"], verse["density"], verse["density_before_svc17"]) == (14, 0.0212, 0.0283)
    json.dumps(rows)
