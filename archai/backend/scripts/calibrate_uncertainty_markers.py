"""Uncertainty markers on real reading pairs: density and verdict, as a synthetic set.

Usage::

    python scripts/calibrate_uncertainty_markers.py
    python scripts/calibrate_uncertainty_markers.py --json

None of the readings in ``eval/quality_gate/readings.json`` contains an
uncertainty marker (``?``, ``…``, ``[…]``), so the calibration set cannot show
how the gate treats marked text. This builds a marked set from it: for every
ordered pair of different recogniser readings of the same page, reading A is
passed through the pipeline's own ``apply_uncertainty_markers`` with reading B
as its cross-pass text, the cross-pass stability and fragment gate computed as
the OCR router computes them. The result is synthetic: two recognisers stand in
for two OCR passes. For each pair where markers were inserted it reports the
CER of A, the markers inserted, the gate's verdict on A unmarked and marked, and
the marked text's uncertainty density, both as computed now and with the
double count audit SVC-17 removed (each "[…]" counting 4).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

READINGS = ROOT / "eval" / "quality_gate" / "readings.json"


def density_before_svc17(text: str) -> float:
    """uncertainty_density as it was: each "[…]" counted 4 for its 3 characters."""
    if not text:
        return 0.0
    return (text.count("?") + text.count("…") + text.count("[…]") * 3) / max(len(text), 1)


def run(path: Path = READINGS) -> list[dict[str, Any]]:
    import calibrate_quality_gate as calibrate

    from app.services.ocr_quality import apply_uncertainty_markers, compute_quality_report
    from app.services.ocr_quality_config import frag_gate_value
    from app.services.ocr_text_metrics import compute_cross_pass_stability, uncertainty_density

    rows: list[dict[str, Any]] = []
    for page in json.loads(path.read_text(encoding="utf-8"))["pages"]:
        readings = {name: text for name, text in page["readings"].items() if (text or "").strip()}
        for a, b in itertools.permutations(readings, 2):
            text_a, text_b = readings[a], readings[b]
            report = compute_quality_report(text_a, run_id="markers", pass_idx=0, tiled=False)
            marked, count = apply_uncertainty_markers(
                text_a,
                cross_pass_text=text_b,
                cross_pass_stability=compute_cross_pass_stability(text_a, text_b),
                frag_gate_val=frag_gate_value(report.leading_fragment_ratio, report.seam_fragment_ratio),
                uncertainty_dens=report.uncertainty_density,
            )
            if not count:
                continue
            unmarked, verdict = calibrate.verdict(text_a), calibrate.verdict(marked)
            rows.append(
                {
                    "page": page["page_id"],
                    "reading": a,
                    "cross_pass": b,
                    "cer": round(calibrate.character_error_rate(page["reference"], text_a), 4),
                    "markers": count,
                    "unmarked": [unmarked.label, unmarked.allowed],
                    "marked": [verdict.label, verdict.allowed],
                    "density": round(uncertainty_density(marked), 4),
                    "density_before_svc17": round(density_before_svc17(marked), 4),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--readings", type=Path, default=READINGS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    rows = run(args.readings)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{len(rows)} reading pairs with markers (synthetic: a second recogniser stands in for a second pass)")
    print(f"{'page':22s} {'reading':24s} {'cross-pass':24s} {'CER':>6s} {'marks':>5s} {'unmarked':16s} {'marked':16s} {'density':>8s} {'before':>7s}")
    for row in rows:
        print(
            f"{row['page']:22s} {row['reading']:24s} {row['cross_pass']:24s} {row['cer']:6.1%} {row['markers']:5d} "
            f"{'/'.join(map(str, row['unmarked'])):16s} {'/'.join(map(str, row['marked'])):16s} "
            f"{row['density']:8.4f} {row['density_before_svc17']:7.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
