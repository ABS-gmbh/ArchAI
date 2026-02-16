from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image

from archai_ocr.pipeline.layout_yolo import RegionDetection


def crop_regions(
    image_path: Path,
    regions: Sequence[RegionDetection],
    crops_dir: Path,
    padding: int = 5,
) -> list[Path]:
    crops_dir.mkdir(parents=True, exist_ok=True)
    crop_paths: list[Path] = []

    with Image.open(image_path) as image:
        width, height = image.size
        for idx, region in enumerate(regions):
            x1, y1, x2, y2 = region.bbox
            left = max(0, x1 - padding)
            top = max(0, y1 - padding)
            right = min(width, x2 + padding)
            bottom = min(height, y2 + padding)

            crop = image.crop((left, top, right, bottom))
            crop_path = crops_dir / f"region_{idx:03d}.png"
            crop.save(crop_path, format="PNG")
            crop_paths.append(crop_path)

    return crop_paths
