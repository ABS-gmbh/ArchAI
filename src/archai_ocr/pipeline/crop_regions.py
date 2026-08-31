from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from archai_ocr.pipeline.layout_yolo import RegionDetection


def crop_regions(
    image_path: Path,
    regions: Sequence[RegionDetection],
    crops_dir: Path,
    padding: int = 5,
    min_size: int = 8,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """Write one PNG per detected region.

    Regions are clamped to the page and skipped when the clamped box is smaller
    than min_size in either dimension. Saving a zero-width crop previously
    produced a file that Kraken either rejected or silently recognized as noise.
    """
    crops_dir.mkdir(parents=True, exist_ok=True)
    crop_paths: list[Path] = []
    skipped = 0

    with Image.open(image_path) as opened:
        # Manuscript scans commonly carry an EXIF orientation tag; without this the
        # crop coordinates (computed on the detector's view) and the pixels we cut
        # can disagree by a 90 degree rotation.
        image: Image.Image = _apply_exif_orientation(opened)
        width, height = image.size

        for idx, region in enumerate(regions):
            x1, y1, x2, y2 = region.bbox
            left = max(0, min(x1, x2) - padding)
            top = max(0, min(y1, y2) - padding)
            right = min(width, max(x1, x2) + padding)
            bottom = min(height, max(y1, y2) + padding)

            if right - left < min_size or bottom - top < min_size:
                skipped += 1
                if logger is not None:
                    logger.warning(
                        "crop.region_too_small",
                        extra={
                            "stage": "crop",
                            "count": idx,
                            "output": f"clamped box {(left, top, right, bottom)} below min_size={min_size}",
                        },
                    )
                continue

            crop = image.crop((left, top, right, bottom))
            crop_path = crops_dir / f"region_{idx:03d}.png"
            crop.save(crop_path, format="PNG")
            crop_paths.append(crop_path)

    if skipped and logger is not None:
        logger.warning("crop.regions_skipped", extra={"stage": "crop", "count": skipped})
    return crop_paths


def _apply_exif_orientation(image: Image.Image) -> Image.Image:
    try:
        from PIL import ImageOps

        return ImageOps.exif_transpose(image) or image
    except Exception:
        return image
