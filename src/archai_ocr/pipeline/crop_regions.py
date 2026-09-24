from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from archai_ocr.pipeline.layout_yolo import RegionDetection


@dataclass(frozen=True)
class RegionCrop:
    """A written crop and where it came from.

    ``box`` is the crop rectangle in page pixels, padding and clamping applied:
    the offset that maps coordinates inside the crop back onto the page.
    """

    index: int
    path: Path
    box: tuple[int, int, int, int]


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
    crops = crop_region_images(
        image_path, regions, crops_dir, padding=padding, min_size=min_size, logger=logger
    )
    return [crop.path for crop in crops]


def crop_region_images(
    image_path: Path,
    regions: Sequence[RegionDetection],
    crops_dir: Path,
    *,
    padding: int = 5,
    min_size: int = 8,
    logger: logging.Logger | None = None,
    name_prefix: str = "region",
) -> list[RegionCrop]:
    """Like :func:`crop_regions`, also returning each crop's page rectangle."""
    crops_dir.mkdir(parents=True, exist_ok=True)
    crops: list[RegionCrop] = []
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
            crop_path = crops_dir / f"{name_prefix}_{idx:03d}.png"
            crop.save(crop_path, format="PNG")
            crops.append(RegionCrop(index=idx, path=crop_path, box=(left, top, right, bottom)))

    if skipped and logger is not None:
        logger.warning("crop.regions_skipped", extra={"stage": "crop", "count": skipped})
    return crops


_EXIF_ORIENTATION = 0x0112
# EXIF orientations 5-8 include a transposition, so width and height swap.
_TRANSPOSING_ORIENTATIONS = frozenset({5, 6, 7, 8})


def oriented_image_size(image_path: Path) -> tuple[int, int]:
    """Page size after EXIF orientation: the space crops and coordinates live in.

    Read from the header rather than by transposing the pixels, which for a
    6000 x 8000 folio would decode the whole image just to learn its size.
    """
    with Image.open(image_path) as opened:
        width, height = opened.size
        orientation = opened.getexif().get(_EXIF_ORIENTATION, 1)
    if orientation in _TRANSPOSING_ORIENTATIONS:
        width, height = height, width
    return int(width), int(height)


def _apply_exif_orientation(image: Image.Image) -> Image.Image:
    try:
        from PIL import ImageOps

        return ImageOps.exif_transpose(image) or image
    except Exception:
        return image
