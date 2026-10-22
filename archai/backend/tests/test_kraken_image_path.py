"""Tests for the image handed to the recogniser.

Two defects compounded on this path:

* crop_agent pads a line-like region by 45% of its height on EACH side, making
  the crop about 1.9x taller than the line, then returned only the PNG. The
  recognition side could not know about the padding, so _local_boundary_from_metadata
  derived scale from crop_size / bbox_size - assuming the crop WAS the bounding
  box - and with no polygon emitted the entire padded crop as the line boundary.
  Kraken was handed a box containing the neighbouring lines.

* The crop arrives already upscaled (ocr_crop_upscale, default 2), which widens
  strokes, while the adaptive-threshold window was hardcoded at 31px. Measured
  against Otsu as the ink reference on a real e-codices line, that erased 65% of
  the stroke ink at upscale=2.
"""

from __future__ import annotations

import pytest
from PIL import Image

from app.agents.crop_agent import _expand_bbox, crop_region
from app.schemas.agents_ocr import OCRRegionInput
from app.services.ocr_backends import (
    OCRRecognitionMetadata,
    _local_boundary_from_metadata,
    _preprocess_kraken_crop_with_metadata,
)

PAGE_W, PAGE_H = 6132, 8176
LINE = (400, 1000, 3400, 1120)  # 3000 x 120, a realistic text line
UPSCALE = 2


def bounding_box(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def metadata(**kwargs) -> OCRRecognitionMetadata:
    base = dict(page_id=None, image_id=None, region_id="r", bbox_xyxy=list(LINE), polygon=None)
    base.update(kwargs)
    return OCRRecognitionMetadata(**base)


# ─────────────────────────────────────────────── the padding is real ──


def test_line_like_regions_are_padded_far_beyond_the_line() -> None:
    """Precondition for the bug: the crop is much taller than the region."""
    crop = _expand_bbox(LINE, width=PAGE_W, height=PAGE_H, line_like=True)
    ratio = (crop[3] - crop[1]) / (LINE[3] - LINE[1])
    assert ratio == pytest.approx(1.9, abs=0.05)


def test_crop_region_reports_the_rectangle_it_used() -> None:
    """Regression: only the PNG was returned, so the padding was invisible."""
    image = Image.new("RGB", (800, 600), "white")
    import base64, io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    result = crop_region(b64, OCRRegionInput(region_id="r", bbox_xyxy=[100, 200, 400, 240]), upscale_factor=2)
    assert result.crop_box is not None
    assert result.upscale_factor == 2
    x1, y1, x2, y2 = result.crop_box
    # The recorded box must be the padded one, and must contain the region.
    assert x1 <= 100 and y1 <= 200 and x2 >= 400 and y2 >= 240


# ──────────────────────────────────────────── the boundary handed out ──


def test_boundary_matches_the_true_line_height_when_the_crop_box_is_known() -> None:
    crop = _expand_bbox(LINE, width=PAGE_W, height=PAGE_H, line_like=True)
    cw = (crop[2] - crop[0]) * UPSCALE
    ch = (crop[3] - crop[1]) * UPSCALE
    box = bounding_box(
        _local_boundary_from_metadata(metadata(crop_box=crop, crop_upscale=UPSCALE), cw, ch)
    )
    expected_height = (LINE[3] - LINE[1]) * UPSCALE
    assert box[3] - box[1] == pytest.approx(expected_height, abs=2)


def test_the_old_derivation_spanned_the_whole_padded_crop() -> None:
    """Documents the defect: without a crop box the boundary is the entire crop."""
    crop = _expand_bbox(LINE, width=PAGE_W, height=PAGE_H, line_like=True)
    cw = (crop[2] - crop[0]) * UPSCALE
    ch = (crop[3] - crop[1]) * UPSCALE
    box = bounding_box(_local_boundary_from_metadata(metadata(), cw, ch))
    assert box[3] - box[1] >= ch - 2, "legacy path still spans the crop"


def test_boundary_is_clamped_inside_the_crop() -> None:
    crop = _expand_bbox(LINE, width=PAGE_W, height=PAGE_H, line_like=True)
    cw = (crop[2] - crop[0]) * UPSCALE
    ch = (crop[3] - crop[1]) * UPSCALE
    for x, y in _local_boundary_from_metadata(metadata(crop_box=crop, crop_upscale=UPSCALE), cw, ch):
        assert 0 <= x <= cw - 1
        assert 0 <= y <= ch - 1


def test_a_polygon_is_mapped_through_the_same_offset() -> None:
    crop = _expand_bbox(LINE, width=PAGE_W, height=PAGE_H, line_like=True)
    cw = (crop[2] - crop[0]) * UPSCALE
    ch = (crop[3] - crop[1]) * UPSCALE
    polygon = [[LINE[0], LINE[1]], [LINE[2], LINE[1]], [LINE[2], LINE[3]], [LINE[0], LINE[3]]]
    box = bounding_box(
        _local_boundary_from_metadata(
            metadata(polygon=polygon, crop_box=crop, crop_upscale=UPSCALE), cw, ch
        )
    )
    assert box[3] - box[1] == pytest.approx((LINE[3] - LINE[1]) * UPSCALE, abs=2)


def test_missing_crop_box_still_works() -> None:
    """Backward compatibility: older callers pass no crop geometry."""
    assert _local_boundary_from_metadata(metadata(), 100, 50)


# ──────────────────────────────────────────────────── preprocessing ──


def test_kraken_receives_grayscale_not_binary() -> None:
    """Kraken's models are trained on grayscale and normalise internally."""
    import numpy as np

    rng = np.random.default_rng(0)
    noisy = Image.fromarray(rng.integers(0, 255, (120, 900), dtype=np.uint8), mode="L").convert("RGB")
    processed, _ = _preprocess_kraken_crop_with_metadata(noisy)
    assert len(set(np.array(processed).ravel().tolist())) > 2


def test_calamari_receives_binary() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    noisy = Image.fromarray(rng.integers(0, 255, (120, 900), dtype=np.uint8), mode="L").convert("RGB")
    processed, _ = _preprocess_kraken_crop_with_metadata(noisy, binarise=True)
    assert set(np.array(processed).ravel().tolist()) <= {0, 255}


def test_block_size_grows_with_stroke_width() -> None:
    """A fixed window hollows thick strokes; it must scale with the ink."""
    import numpy as np

    from app.services.ocr_backends import _adaptive_block_size

    def strokes(thickness: int) -> "np.ndarray":
        arr = np.full((200, 900), 255, dtype=np.uint8)
        for x in range(40, 880, 40):
            arr[60:140, x : x + thickness] = 0
        return arr

    thin = _adaptive_block_size(strokes(3))
    thick = _adaptive_block_size(strokes(12))
    assert thick > thin
    assert thin % 2 == 1 and thick % 2 == 1


def test_block_size_falls_back_safely_on_blank_input() -> None:
    import numpy as np

    from app.services.ocr_backends import _adaptive_block_size

    block = _adaptive_block_size(np.full((50, 50), 255, dtype=np.uint8))
    assert block % 2 == 1 and block >= 15
