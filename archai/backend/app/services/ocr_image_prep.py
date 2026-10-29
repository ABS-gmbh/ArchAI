"""Page images prepared for a vision-language OCR model.

Decoding, contrast enhancement for faded manuscripts, resizing within the
provider's pixel limits, downscaling for a retry, and tile crops. Moved out of
``app/agents/saia_ocr_agent.py``.
"""

from __future__ import annotations

import base64
import io
import os

import numpy as np
from PIL import Image, ImageFilter

from app.config import settings
from app.agents.saia_ocr_errors import (
    SaiaOCRAgentError,
)


SAIA_OCR_MAX_PIXELS = max(
    1,
    int(settings.ocr_max_pixels_per_tile or os.getenv("OCR_MAX_PIXELS_PER_TILE", "160000000") or "160000000"),
)


SAIA_OCR_MAX_LONG_EDGE = max(
    1024,
    int(settings.ocr_max_long_edge or os.getenv("OCR_MAX_LONG_EDGE", "12000") or "12000"),
)


SAIA_OCR_IMAGE_RETRY_SHRINK = float(
    settings.ocr_image_retry_shrink or os.getenv("OCR_IMAGE_RETRY_SHRINK", "0.82") or "0.82"
)
if SAIA_OCR_IMAGE_RETRY_SHRINK <= 0.0 or SAIA_OCR_IMAGE_RETRY_SHRINK >= 1.0:
    SAIA_OCR_IMAGE_RETRY_SHRINK = 0.82


def _open_rgb_image_from_b64(image_b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(image_b64, validate=False)
    except Exception as exc:
        raise SaiaOCRAgentError("image_b64 is not valid base64.") from exc

    previous_max_pixels = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(io.BytesIO(raw)) as decoded:
            image = decoded.convert("RGB")
    except Exception as exc:
        raise SaiaOCRAgentError("Unable to decode image payload for OCR.") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_max_pixels
    return image


def _encode_png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


_ENHANCE_CLAHE_CLIP = 2.0       # clip limit for CLAHE (moderate — don't blow out minims)


_ENHANCE_CLAHE_GRID = 8         # tile grid for adaptive histogram


_ENHANCE_BILATERAL_D = 5        # bilateral filter diameter (lighter touch)


_ENHANCE_BILATERAL_SC = 40.0    # sigma-colour  (preserve edges, lighter)


_ENHANCE_BILATERAL_SS = 40.0    # sigma-space   (spatial smoothing, lighter)


_ENHANCE_SHARPEN_RADIUS = 1     # unsharp-mask radius in px (gentle — don't warp minims)


_ENHANCE_SHARPEN_PERCENT = 80   # unsharp-mask strength (%) (reduced from 120)


_ENHANCE_SHARPEN_THRESH = 4     # unsharp-mask threshold (raised to avoid sharpening noise)


_ENHANCE_ENABLED = True         # master switch (can be overridden via env)


def _enhance_manuscript_image(image: Image.Image) -> Image.Image:
    """Enhance a manuscript image for better vision-model OCR.

    Steps:
      1. Convert RGB → L*a*b*, apply CLAHE to L* channel, convert back.
      2. Bilateral-filter the result (denoise while keeping stroke edges).
      3. Unsharp-mask sharpen to crisp up character strokes.

    Returns an enhanced PIL RGB image of the same dimensions.
    """
    if not _ENHANCE_ENABLED:
        return image

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        # Fallback: PIL-only sharpening if OpenCV is unavailable
        return image.filter(
            ImageFilter.UnsharpMask(
                radius=_ENHANCE_SHARPEN_RADIUS,
                percent=_ENHANCE_SHARPEN_PERCENT,
                threshold=_ENHANCE_SHARPEN_THRESH,
            )
        )

    arr = np.array(image.convert("RGB"), dtype=np.uint8)

    # --- 1. CLAHE on the L* channel of CIE-LAB ---
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=_ENHANCE_CLAHE_CLIP,
        tileGridSize=(_ENHANCE_CLAHE_GRID, _ENHANCE_CLAHE_GRID),
    )
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # --- 2. Bilateral filter (edge-preserving denoise) ---
    arr = cv2.bilateralFilter(
        arr,
        d=_ENHANCE_BILATERAL_D,
        sigmaColor=_ENHANCE_BILATERAL_SC,
        sigmaSpace=_ENHANCE_BILATERAL_SS,
    )

    enhanced = Image.fromarray(arr, mode="RGB")

    # --- 3. Unsharp-mask sharpening (PIL, works on RGB directly) ---
    enhanced = enhanced.filter(
        ImageFilter.UnsharpMask(
            radius=_ENHANCE_SHARPEN_RADIUS,
            percent=_ENHANCE_SHARPEN_PERCENT,
            threshold=_ENHANCE_SHARPEN_THRESH,
        )
    )

    return enhanced


def _resize_for_limits(image: Image.Image) -> tuple[Image.Image, bool]:
    width, height = image.size
    if width <= 0 or height <= 0:
        return image, False

    pixels = float(width * height)
    long_edge = float(max(width, height))
    scale = 1.0

    if pixels > float(SAIA_OCR_MAX_PIXELS):
        scale = min(scale, (float(SAIA_OCR_MAX_PIXELS) / max(pixels, 1.0)) ** 0.5)
    if long_edge > float(SAIA_OCR_MAX_LONG_EDGE):
        scale = min(scale, float(SAIA_OCR_MAX_LONG_EDGE) / max(long_edge, 1.0))

    if scale >= 0.999:
        return image, False

    target_w = max(1, int(round(width * scale)))
    target_h = max(1, int(round(height * scale)))
    resized = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
    return resized, True


def _prepare_image_for_ocr(image_b64: str) -> tuple[str, list[str]]:
    image = _open_rgb_image_from_b64(image_b64)
    original_w, original_h = image.size
    warnings: list[str] = []

    # --- Enhancement: improve contrast / sharpness for the vision model ---
    image = _enhance_manuscript_image(image)

    resized, changed = _resize_for_limits(image)
    if changed:
        warnings.append(
            "AUTO_RESIZED_FOR_LIMIT:"
            f"{original_w}x{original_h}->{resized.width}x{resized.height}"
        )
        image = resized

    return _encode_png_base64(image), warnings


def _downscale_image_b64_for_retry(image_b64: str) -> str | None:
    image = _open_rgb_image_from_b64(image_b64)
    target_w = max(1, int(round(image.width * SAIA_OCR_IMAGE_RETRY_SHRINK)))
    target_h = max(1, int(round(image.height * SAIA_OCR_IMAGE_RETRY_SHRINK)))
    if target_w >= image.width and target_h >= image.height:
        return None

    resized = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
    resized, _ = _resize_for_limits(resized)
    if resized.width >= image.width and resized.height >= image.height:
        return None
    return _encode_png_base64(resized)


_TILE_UPSCALE_TARGET = 512  # upscale tiny crops so shortest side >= this


def _crop_and_maybe_upscale(
    image: Image.Image,
    rect: tuple[int, int, int, int],
) -> Image.Image:
    """Crop tile from image, upscale if too small, and enhance for OCR."""
    x1, y1, x2, y2 = rect
    crop = image.crop((x1, y1, x2, y2))
    cw, ch = crop.size
    if cw < _TILE_UPSCALE_TARGET or ch < _TILE_UPSCALE_TARGET:
        short_side = min(cw, ch)
        if short_side > 0:
            scale = max(1.0, float(_TILE_UPSCALE_TARGET) / float(short_side))
            if scale > 1.0:
                new_w = max(1, int(round(cw * scale)))
                new_h = max(1, int(round(ch * scale)))
                crop = crop.resize((new_w, new_h), Image.Resampling.BICUBIC)
    # Enhance the tile crop for better OCR
    crop = _enhance_manuscript_image(crop)
    return crop
