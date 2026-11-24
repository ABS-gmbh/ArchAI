"""Decoding a page for the provider path never touches PIL's global pixel limit (security review C3).

It set Image.MAX_IMAGE_PIXELS = None around Image.open and restored it without
a lock: after one round of concurrent calls the global stayed None for the
life of the process, and a decompression bomb opened anywhere in it.
"""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from app.agents.saia_ocr_errors import SaiaOCRAgentError
from app.services import ocr_image_prep


def _png_b64(size: tuple[int, int], mode: str = "L") -> str:
    buffer = io.BytesIO()
    Image.new(mode, size, 1).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_the_global_limit_is_left_alone_while_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []
    real_open = Image.open

    def recording_open(*args, **kwargs):
        seen.append(Image.MAX_IMAGE_PIXELS)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 12_345_678)
    monkeypatch.setattr(ocr_image_prep.Image, "open", recording_open)
    image = ocr_image_prep._open_rgb_image_from_b64(_png_b64((40, 30)))
    assert image.mode == "RGB" and image.size == (40, 30)
    assert seen == [12_345_678]  # the old code opened with the limit switched off (None)
    assert Image.MAX_IMAGE_PIXELS == 12_345_678


def test_an_image_over_the_configured_limit_is_refused_before_it_is_converted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_image_prep, "SAIA_OCR_MAX_DECODE_PIXELS", 1_000_000)
    with pytest.raises(SaiaOCRAgentError, match="pixels"):
        ocr_image_prep._open_rgb_image_from_b64(_png_b64((2000, 1000), mode="1"))  # 2 MP in a few KB
    assert ocr_image_prep._open_rgb_image_from_b64(_png_b64((1000, 1000))).size == (1000, 1000)
