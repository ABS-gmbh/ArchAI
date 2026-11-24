"""One limit on the pixels of an image the server decodes: OCR_MAX_DECODE_PIXELS.

PIL reads the width and height from the header when an image is opened; the
pixels are decoded only by load(), convert() or a resize. A 12 KB PNG can
declare 100 million pixels, and decoding it took more than 500 MB (second
security review, I). Every place that decodes an upload checks the header
against this limit first: the full-page OCR route, the layout segmentation
of /predict, the images of a batch, the page of region OCR, the page of a
label analysis and the provider path (ocr_image_prep, which reads the same
setting).
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.services.public_errors import RequestRefused


class ImageTooLarge(RequestRefused):
    """An image over OCR_MAX_DECODE_PIXELS; the message is for the client (a route's 422)."""


def max_decode_pixels() -> int:
    return max(1, int(settings.ocr_max_decode_pixels))


def too_many_pixels(image: Any) -> str | None:
    """A sentence for the client when the opened (not yet decoded) *image* is over the limit, else None."""
    width, height = image.size
    limit = max_decode_pixels()
    if int(width) * int(height) <= limit:
        return None
    return f"The image has {int(width) * int(height)} pixels; at most {limit} are read (OCR_MAX_DECODE_PIXELS)."
