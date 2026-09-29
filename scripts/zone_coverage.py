"""How much of a page's text lies inside the zones the pipeline transcribes?

Usage::

    python scripts/zone_coverage.py pages/*.jpg --config config.example.yaml

Kraken's page segmenter is run on each whole page as an independent census of
its text lines. A line counts as covered when the midpoint of its baseline lies
inside a transcribed zone. The script reports coverage by main-text zones alone
and by main and secondary zones together - the measure ACCURACY_ROADMAP.md used
for the finding that only MainZone was ever transcribed.

The census is not ground truth: full-page segmentation also reports decoration
as lines, and misses some lines per-zone segmentation finds, so read the numbers
as relative. Needs the inference stack (ultralytics, kraken) and the weights.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from archai_ocr.config import load_config  # noqa: E402
from archai_ocr.pipeline.layout_yolo import detect_page_layout  # noqa: E402

Box = tuple[int, int, int, int]


def _midpoint(baseline: list[list[int]]) -> tuple[float, float]:
    n = len(baseline)
    if n % 2:
        x, y = baseline[n // 2]
        return float(x), float(y)
    (x1, y1), (x2, y2) = baseline[n // 2 - 1], baseline[n // 2]
    return (x1 + x2) / 2, (y1 + y2) / 2


def _inside(point: tuple[float, float], boxes: list[Box]) -> bool:
    return any(x1 <= point[0] <= x2 and y1 <= point[1] <= y2 for x1, y1, x2, y2 in boxes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--config", default="config.example.yaml")
    args = parser.parse_args(argv)

    from kraken import blla
    from kraken.lib.vgsl import TorchVGSLModel
    from PIL import Image

    config = load_config(args.config)
    segmenter = TorchVGSLModel.load_model(str(config.weights.kraken_segmentation))
    logger = logging.getLogger("zone_coverage")

    totals = [0, 0, 0]
    print(f"{'page':<40}{'lines':>7}{'in main':>10}{'in main+secondary':>20}")
    with tempfile.TemporaryDirectory() as scratch:
        for image_path in args.images:
            layout = detect_page_layout(image_path, Path(scratch), config, logger)
            with Image.open(image_path) as image:
                lines = blla.segment(image.convert("L"), model=segmenter).lines
            points = [_midpoint(line.baseline) for line in lines if line.baseline]
            main = [region.bbox for region in layout.main]
            both = main + [region.bbox for region in layout.secondary]
            row = [len(points), sum(_inside(p, main) for p in points), sum(_inside(p, both) for p in points)]
            totals = [total + value for total, value in zip(totals, row, strict=True)]
            print(f"{image_path.name[:39]:<40}{row[0]:>7}{row[1]:>10}{row[2]:>20}")

    lines_total, main_total, both_total = totals
    if lines_total:
        print(
            f"\n{lines_total} lines: {main_total / lines_total:.1%} inside main-text zones, "
            f"{both_total / lines_total:.1%} inside main or secondary zones"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
