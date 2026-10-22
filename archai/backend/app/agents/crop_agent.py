from __future__ import annotations

import base64
import io
from typing import NamedTuple

from PIL import Image, ImageDraw

from app.schemas.agents_ocr import OCRRegionInput


class CropAgentError(RuntimeError):
    """Raised when a region crop cannot be generated."""


def decode_image_bytes(image_b64: str) -> bytes:
    payload = image_b64.strip()
    if payload.startswith("data:image/") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=False)
    except Exception as exc:
        raise CropAgentError("Invalid image_b64 payload.") from exc


def encode_png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _bbox_from_polygon(polygon: list[list[float]]) -> tuple[int, int, int, int]:
    xs = [int(round(point[0])) for point in polygon]
    ys = [int(round(point[1])) for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _looks_line_like(label: str | None) -> bool:
    value = str(label or "").strip().lower()
    return any(token in value for token in ("line", "main script", "variant script", "defaultlines", "gloss"))


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    line_like: bool,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    region_w = max(1, x2 - x1)
    region_h = max(1, y2 - y1)
    if line_like:
        pad_x = max(10, int(round(region_w * 0.08)))
        pad_y = max(10, int(round(region_h * 0.45)))
    else:
        pad_x = max(8, int(round(region_w * 0.04)))
        pad_y = max(8, int(round(region_h * 0.12)))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


class CropResult(NamedTuple):
    """A cropped region plus the page rectangle it was actually taken from.

    crop_box matters because _expand_bbox pads the region before cropping - by
    45% of its height per side for line-like labels, making the crop 1.9x taller
    than the line. Returning only the PNG left the recognition path unable to
    know that, so it treated the padded crop as if it were the region and handed
    Kraken a line boundary spanning the neighbours above and below.
    """

    region_id: str
    crop_b64: str
    crop_box: tuple[int, int, int, int]
    upscale_factor: int


def decode_page(image_b64: str) -> Image.Image:
    """The page as RGB, decoded once for every crop taken from it."""
    image_bytes = decode_image_bytes(image_b64)
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise CropAgentError("Could not decode source image.") from exc


def crop_region(
    image_b64: str, region: OCRRegionInput, upscale_factor: int = 2
) -> CropResult:
    return crop_page_region(decode_page(image_b64), region, upscale_factor)


def crop_page_region(
    source: Image.Image, region: OCRRegionInput, upscale_factor: int = 2
) -> CropResult:
    """Crop *region* from a decoded page.

    Pass the same decoded page for every region of it. Decoding per region, as
    ``crop_region`` must, costs a full-page JPEG decode each time - on a
    6132x8176 e-codices scan, most of the time a 70-line page spent in
    recognition.
    """
    width, height = source.size

    line_like = _looks_line_like(getattr(region, "label", None))

    if region.polygon is not None:
        x1, y1, x2, y2 = _bbox_from_polygon(region.polygon)
    elif region.bbox_xyxy is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in region.bbox_xyxy]
    else:
        raise CropAgentError("Region must include bbox_xyxy or polygon.")

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    x1, y1, x2, y2 = _expand_bbox((x1, y1, x2, y2), width=width, height=height, line_like=line_like)

    crop = source.crop((x1, y1, x2, y2))
    if region.polygon is not None:
        # Whiten everything outside the polygon. Masking only the crop window
        # yields the same pixels as masking the whole page first, without a
        # page-sized mask and canvas for every line.
        mask = Image.new("L", crop.size, 0)
        points = [(int(round(px)) - x1, int(round(py)) - y1) for px, py in region.polygon]
        ImageDraw.Draw(mask).polygon(points, fill=255)
        masked = Image.new("RGB", crop.size, (255, 255, 255))
        masked.paste(crop, mask=mask)
        crop = masked
    if upscale_factor > 1:
        crop = crop.resize((crop.width * upscale_factor, crop.height * upscale_factor), Image.Resampling.LANCZOS)
    region_id = region.region_id or f"region-{x1}-{y1}-{x2}-{y2}"
    return CropResult(region_id, encode_png_base64(crop), (x1, y1, x2, y2), max(1, upscale_factor))
