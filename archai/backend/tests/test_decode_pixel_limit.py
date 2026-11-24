"""Every route that decodes an upload refuses one over OCR_MAX_DECODE_PIXELS from its header (second security review, I).

A 12 KB PNG declaring 100 million pixels decoded with +500 MB RSS: anything up
to 178.9 MP was accepted, and /predict called im.load() before any size check.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from fastapi import HTTPException
from PIL import Image, ImageFile

from app.config import Settings, settings

LIMIT = 1_000_000


def _png(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("1", size, 0).save(buffer, format="PNG")  # a few KB however many pixels
    return buffer.getvalue()


@pytest.fixture
def no_decoding(monkeypatch: pytest.MonkeyPatch):
    """The limit, and a load() that is recorded and refused: nothing over the limit may be decoded."""
    monkeypatch.setattr(settings, "ocr_max_decode_pixels", LIMIT)
    loads: list[int] = []

    def refuse(self: ImageFile.ImageFile) -> None:
        loads.append(self.size[0] * self.size[1])
        raise AssertionError(f"decoded an image of {self.size[0] * self.size[1]} pixels")

    monkeypatch.setattr(ImageFile.ImageFile, "load", refuse)
    yield
    assert loads == []  # callers that swallow the AssertionError are caught here


OVER = (2000, 1000)  # 2 MP


def test_the_full_page_route_refuses_it_with_413(no_decoding: None) -> None:
    from app.routers import ocr

    with pytest.raises(HTTPException) as refused:
        ocr._require_readable_image(_png(OVER))
    assert refused.value.status_code == 413 and "OCR_MAX_DECODE_PIXELS" in str(refused.value.detail)


def test_predict_refuses_it_before_load(no_decoding: None, tmp_path: Path) -> None:
    from app.routers import predict

    path = tmp_path / "page.png"
    path.write_bytes(_png(OVER))
    with pytest.raises(HTTPException) as refused:
        predict._prepare_image_for_segmentation(str(path))
    assert refused.value.status_code == 413 and "2000000 pixels" in str(refused.value.detail)


def test_a_batch_skips_it_undecoded(no_decoding: None, tmp_path: Path) -> None:
    from app.routers import predict

    path = tmp_path / "page.png"
    path.write_bytes(_png(OVER))
    assert predict._is_valid_image_file(str(path)) is False


def test_region_ocr_and_label_analysis_refuse_it(no_decoding: None) -> None:
    from app.agents.crop_agent import decode_page
    from app.agents.label_analysis_agent import LabelAnalysisAgent, LabelAnalysisAgentError
    from app.services.image_limits import ImageTooLarge

    b64 = base64.b64encode(_png(OVER)).decode("ascii")
    with pytest.raises(ImageTooLarge, match="pixels"):
        decode_page(b64)
    with pytest.raises(LabelAnalysisAgentError, match="pixels"):
        LabelAnalysisAgent._decode_page(b64)


def test_an_image_within_the_limit_is_still_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from app.agents.crop_agent import decode_page
    from app.routers import ocr, predict

    monkeypatch.setattr(settings, "ocr_max_decode_pixels", LIMIT)
    data = _png((1000, 1000))
    ocr._require_readable_image(data)
    path = tmp_path / "page.png"
    path.write_bytes(data)
    assert predict._is_valid_image_file(str(path))
    assert predict._prepare_image_for_segmentation(str(path)).path
    assert decode_page(base64.b64encode(data).decode("ascii")).size == (1000, 1000)


def test_the_default_leaves_room_for_the_largest_real_pages_and_refuses_the_reviewed_bomb() -> None:
    default = Settings.model_fields["ocr_max_decode_pixels"].default
    assert default >= 6130 * 8175 * 1.5  # e-codices full-size downloads, with headroom
    assert default < 100_000_000  # the 12 KB PNG of the review decoded with +500 MB
    assert default == Image.MAX_IMAGE_PIXELS  # PIL's own warning point
