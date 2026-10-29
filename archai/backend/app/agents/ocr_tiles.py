"""Tiles for the OCR agent: the text block of a page, its tile boxes, the
crops and their resizing, and the tiles a request's regions give. Moved out
of ``app/agents/ocr_agent.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import os

from PIL import Image

from app.agents.crop_agent import decode_image_bytes, encode_png_base64
from app.config import settings
from app.schemas.agents_ocr import OCRExtractRequest
from app.agents.ocr_agent_errors import (
    OCRAgentError,
)


OCR_MAX_PIXELS_PER_TILE = max(
    1,
    int(settings.ocr_max_pixels_per_tile or os.getenv("OCR_MAX_PIXELS_PER_TILE", "160000000") or "160000000"),
)


OCR_MAX_LONG_EDGE = max(
    512,
    int(settings.ocr_max_long_edge or os.getenv("OCR_MAX_LONG_EDGE", "12000") or "12000"),
)


OCR_IMAGE_RETRY_SHRINK = float(
    settings.ocr_image_retry_shrink or os.getenv("OCR_IMAGE_RETRY_SHRINK", "0.82") or "0.82"
)


if OCR_IMAGE_RETRY_SHRINK <= 0.0 or OCR_IMAGE_RETRY_SHRINK >= 1.0:
    OCR_IMAGE_RETRY_SHRINK = 0.82


@dataclass(frozen=True)
class OCRTile:
    tile_id: str
    bbox_xyxy: list[float]
    image_b64: str
    width: int
    height: int
    resized_for_limits: bool = False


def _clip_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    cx1 = max(0, min(x1, width - 1))
    cy1 = max(0, min(y1, height - 1))
    cx2 = max(cx1 + 1, min(x2, width))
    cy2 = max(cy1 + 1, min(y2, height))
    return cx1, cy1, cx2, cy2


def _detect_text_block_bbox(source: Image.Image) -> tuple[int, int, int, int]:
    width, height = source.size
    gray = source.convert("L")
    ink = gray.point(lambda p: 255 if p < 225 else 0, mode="L")
    bbox = ink.getbbox()
    if bbox is None:
        return 0, 0, width, height

    x1, y1, x2, y2 = bbox
    ink_area = max(1, (x2 - x1) * (y2 - y1))
    full_area = max(1, width * height)

    if ink_area < full_area * 0.03:
        return 0, 0, width, height

    xpad = max(12, int(width * 0.04))
    ypad = max(10, int(height * 0.02))
    return _clip_box(x1 - xpad, y1 - ypad, x2 + xpad, y2 + ypad, width, height)


def _choose_band_count(block_height: int) -> int:
    if block_height < 280:
        return 1
    bands = int(round(block_height / 260.0))
    bands = max(5, min(8, bands))
    if block_height // max(1, bands) < 90:
        bands = max(1, block_height // 90)
    return max(1, bands)


def _build_tile_boxes(source: Image.Image) -> list[tuple[int, int, int, int]]:
    width, height = source.size
    block_x1, block_y1, block_x2, block_y2 = _detect_text_block_bbox(source)
    block_w = block_x2 - block_x1
    block_h = block_y2 - block_y1

    boxes: list[tuple[int, int, int, int]] = [(block_x1, block_y1, block_x2, block_y2)]

    if min(block_w, block_h) < 120:
        return boxes

    bands = _choose_band_count(block_h)
    if bands <= 1:
        return boxes

    band_height = max(1, int(round(block_h / bands)))
    overlap_y = max(8, int(round(band_height * 0.12)))
    split_columns = block_w >= 1100
    overlap_x = max(18, int(round(block_w * 0.05)))

    for band_idx in range(bands):
        base_top = block_y1 + band_idx * band_height
        base_bottom = block_y1 + (band_idx + 1) * band_height
        band_top = block_y1 if band_idx == 0 else max(block_y1, base_top - overlap_y)
        band_bottom = block_y2 if band_idx == bands - 1 else min(block_y2, base_bottom + overlap_y)

        if not split_columns:
            boxes.append((block_x1, band_top, block_x2, band_bottom))
            continue

        mid_x = block_x1 + block_w // 2
        left_box = _clip_box(block_x1, band_top, mid_x + overlap_x, band_bottom, width, height)
        right_box = _clip_box(mid_x - overlap_x, band_top, block_x2, band_bottom, width, height)
        boxes.append(left_box)
        boxes.append(right_box)

    return boxes


def _resize_crop_for_ocr(crop: Image.Image, *, requested_scale: float) -> tuple[Image.Image, bool]:
    width, height = crop.size
    if width <= 0 or height <= 0:
        return crop, False

    target_scale = max(0.05, float(requested_scale))
    base_pixels = float(width * height)
    if base_pixels * (target_scale**2) > float(OCR_MAX_PIXELS_PER_TILE):
        target_scale = min(target_scale, (float(OCR_MAX_PIXELS_PER_TILE) / max(base_pixels, 1.0)) ** 0.5)

    long_edge = float(max(width, height))
    if long_edge * target_scale > float(OCR_MAX_LONG_EDGE):
        target_scale = min(target_scale, float(OCR_MAX_LONG_EDGE) / max(long_edge, 1.0))

    target_scale = max(0.05, target_scale)
    new_w = max(1, int(round(width * target_scale)))
    new_h = max(1, int(round(height * target_scale)))
    if new_w == width and new_h == height:
        return crop, False

    resample = Image.Resampling.LANCZOS if target_scale >= 1.0 else Image.Resampling.BICUBIC
    return crop.resize((new_w, new_h), resample), True


def _downscale_tile_for_retry(tile: OCRTile) -> OCRTile | None:
    try:
        raw = decode_image_bytes(tile.image_b64)
        with Image.open(io.BytesIO(raw)) as decoded:
            image = decoded.convert("RGB")
    except Exception:
        return None

    target_w = max(1, int(round(image.width * OCR_IMAGE_RETRY_SHRINK)))
    target_h = max(1, int(round(image.height * OCR_IMAGE_RETRY_SHRINK)))
    if target_w >= image.width and target_h >= image.height:
        return None

    resized = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
    resized, _ = _resize_crop_for_ocr(resized, requested_scale=1.0)
    if resized.width >= image.width and resized.height >= image.height:
        return None

    return OCRTile(
        tile_id=tile.tile_id,
        bbox_xyxy=list(tile.bbox_xyxy),
        image_b64=encode_png_base64(resized),
        width=resized.width,
        height=resized.height,
        resized_for_limits=True,
    )


def _boxes_to_tiles(source: Image.Image, boxes: list[tuple[int, int, int, int]], *, upscale_factor: int) -> list[OCRTile]:
    unique: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for box in boxes:
        if box in seen:
            continue
        seen.add(box)
        unique.append(box)

    tiles: list[OCRTile] = []
    requested_scale = float(max(1, int(upscale_factor)))
    for idx, (x1, y1, x2, y2) in enumerate(unique):
        crop = source.crop((x1, y1, x2, y2))
        crop, resized_for_limits = _resize_crop_for_ocr(crop, requested_scale=requested_scale)
        tile_id = "tile_full" if idx == 0 else f"tile_{idx:03d}"
        tiles.append(
            OCRTile(
                tile_id=tile_id,
                bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
                image_b64=encode_png_base64(crop),
                width=crop.width,
                height=crop.height,
                resized_for_limits=resized_for_limits,
            )
        )
    return tiles


def _make_tiles_from_payload(payload: OCRExtractRequest, *, upscale_factor: int) -> tuple[list[OCRTile], str]:
    source_b64 = str(payload.cropped_image_b64 or payload.image_b64 or "").strip()
    if not source_b64:
        raise OCRAgentError("image_b64 or cropped_image_b64 is required.")

    image_bytes = decode_image_bytes(source_b64)
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            source = image.convert("RGB")
    except Exception as exc:  # pragma: no cover - decode helper already validated bytes
        raise OCRAgentError("Could not decode source image for tiled OCR.") from exc

    boxes = _build_tile_boxes(source)
    tiles = _boxes_to_tiles(source, boxes, upscale_factor=upscale_factor)
    if not tiles:
        raise OCRAgentError("No OCR tiles were generated from input image.")

    return tiles, source_b64
