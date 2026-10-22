"""Tests for the line crops the OCR agent hands to Kraken.

Three things cost accuracy or time on the way from a page to a recognised line:

* Each crop was contrast-stretched, denoised and deskewed before recognition.
  The deskew fed numpy's (row, column) pairs to cv2.minAreaRect as if they were
  (x, y), and folded the angle for OpenCV's pre-4.5 convention, so it rotated
  lines by angles it had not measured. Kraken's models are trained on plain
  grayscale lines; on the Latin reference page the preprocessing took CER from
  6.6% to 11.7%.
* Every region decoded the whole page again - base64, JPEG, RGB conversion -
  and, when it had a polygon, built a page-sized mask and canvas to whiten the
  outside of it. On a 6132x8176 scan that was most of the time spent reading
  the page.
* Every crop was upscaled twofold, although Kraken resizes each line to 120 px
  high before reading it: a 218 px line gained nothing but encoding time.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageOps

from app.agents import crop_agent, ocr_agent
from app.agents.crop_agent import crop_page_region, crop_region, decode_page
from app.schemas.agents_ocr import OCRExtractOptions, OCRExtractRequest, OCRRegionInput
from app.services.ocr_backends import OCRBackendResult, _preprocess_kraken_crop_with_metadata


def _b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _pixels(crop_b64: str) -> np.ndarray:
    with Image.open(io.BytesIO(base64.b64decode(crop_b64))) as image:
        return np.array(image.convert("RGB"))


def _tilted_line(angle: float = 3.0) -> Image.Image:
    """A dark band of 'ink' across a light crop, tilted by *angle* degrees."""
    image = Image.new("L", (900, 160), 235)
    ImageDraw.Draw(image).rectangle((60, 65, 840, 95), fill=30)
    return image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=235).convert("RGB")


# ----------------------------------------------------------- preprocessing --


def test_kraken_gets_the_crop_in_plain_grayscale() -> None:
    crop = _tilted_line()
    processed, transform = _preprocess_kraken_crop_with_metadata(crop)
    assert np.array_equal(np.array(processed), np.array(ImageOps.grayscale(crop)))
    assert transform["deskew_angle"] == 0.0
    assert transform["scale_x"] == transform["scale_y"] == 1.0


def test_a_tilted_line_is_left_as_it_is() -> None:
    """Regression: the deskew rotated this crop by an angle it had not measured."""
    for angle in (-3.0, 1.5, 4.0):
        processed, transform = _preprocess_kraken_crop_with_metadata(_tilted_line(angle))
        assert transform["deskew_angle"] == 0.0
        assert np.array_equal(np.array(processed), np.array(ImageOps.grayscale(_tilted_line(angle))))


@pytest.mark.parametrize(("height", "factor"), [(40, 2), (20, 5), (95, 2), (96, 1)])
def test_a_crop_shorter_than_96_px_is_still_enlarged(height: int, factor: int) -> None:
    processed, transform = _preprocess_kraken_crop_with_metadata(Image.new("RGB", (300, height), "white"))
    assert processed.size == (300 * factor, height * factor)
    assert transform["scale_x"] == transform["scale_y"] == factor


def test_calamari_still_gets_a_binarised_crop() -> None:
    processed, _ = _preprocess_kraken_crop_with_metadata(_tilted_line(), binarise=True)
    assert set(np.unique(np.array(processed)).tolist()) <= {0, 255}


# ------------------------------------------------------------------- crops --


PAGE = Image.fromarray(np.random.default_rng(11).integers(0, 255, (500, 700, 3), dtype=np.uint8))
PAGE_B64 = _b64(PAGE)


def _whole_page_crop(page: Image.Image, region: OCRRegionInput, upscale: int) -> np.ndarray:
    """The crop as it used to be made: mask the whole page, then cut the window."""
    width, height = page.size
    if region.polygon is not None:
        x1, y1, x2, y2 = crop_agent._bbox_from_polygon(region.polygon)
        mask = Image.new("L", page.size, 0)
        ImageDraw.Draw(mask).polygon([(int(round(x)), int(round(y))) for x, y in region.polygon], fill=255)
        source = Image.new("RGB", page.size, (255, 255, 255))
        source.paste(page, mask=mask)
    else:
        x1, y1, x2, y2 = [int(round(v)) for v in region.bbox_xyxy or []]
        source = page
    x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
    x2, y2 = max(x1 + 1, min(x2, width)), max(y1 + 1, min(y2, height))
    box = crop_agent._expand_bbox(
        (x1, y1, x2, y2), width=width, height=height, line_like=crop_agent._looks_line_like(region.label)
    )
    crop = source.crop(box)
    if upscale > 1:
        crop = crop.resize((crop.width * upscale, crop.height * upscale), Image.Resampling.LANCZOS)
    return np.array(crop)


@pytest.mark.parametrize(
    "region",
    [
        OCRRegionInput(region_id="box", bbox_xyxy=[120, 200, 480, 240], label="Main script black"),
        OCRRegionInput(
            region_id="polygon",
            polygon=[[110.4, 205.2], [470, 198], [482.7, 238], [118, 246.5]],
            label="Main script black",
        ),
        OCRRegionInput(region_id="edge", polygon=[[-20, 10], [300, -15], [310, 60], [-5, 70]], label="MainZone"),
    ],
)
@pytest.mark.parametrize("upscale", [1, 2])
def test_cropping_a_decoded_page_gives_the_pixels_of_the_old_whole_page_mask(
    region: OCRRegionInput, upscale: int
) -> None:
    result = crop_page_region(decode_page(PAGE_B64), region, upscale_factor=upscale)
    assert np.array_equal(_pixels(result.crop_b64), _whole_page_crop(PAGE, region, upscale))


def test_crop_region_still_takes_an_encoded_page() -> None:
    region = OCRRegionInput(region_id="r", polygon=[[110, 205], [470, 198], [482, 238], [118, 246]])
    by_b64 = crop_region(PAGE_B64, region, upscale_factor=2)
    by_page = crop_page_region(decode_page(PAGE_B64), region, upscale_factor=2)
    assert (by_b64.region_id, by_b64.crop_box, by_b64.crop_b64) == (by_page.region_id, by_page.crop_box, by_page.crop_b64)


def test_a_page_that_is_not_an_image_is_refused() -> None:
    with pytest.raises(crop_agent.CropAgentError):
        decode_page(base64.b64encode(b"not an image").decode("ascii"))


# ------------------------------------------------------------------- agent --


class _EchoBackend:
    backend_name = "kraken_catmus"
    model_name = "CATMuS Medieval"

    def __init__(self) -> None:
        self.crop_heights: list[int] = []

    def recognize(self, crop_b64: str, metadata: Any) -> OCRBackendResult:
        with Image.open(io.BytesIO(base64.b64decode(crop_b64))) as image:
            self.crop_heights.append(image.height)
        return OCRBackendResult(
            text=f"line {metadata.region_id}",
            confidence=0.9,
            backend_name=self.backend_name,
            model_name=self.model_name,
            raw_metadata={"warnings": [], "flags": []},
            region_id=str(metadata.region_id),
            page_id=metadata.page_id,
        )


class _DummyClient:
    def list_models(self, force_refresh: bool = False) -> list[str]:
        return []


def _request(heights: list[int]) -> OCRExtractRequest:
    regions, top = [], 10
    for index, height in enumerate(heights):
        regions.append(
            OCRRegionInput(
                region_id=f"l{index}",
                bbox_xyxy=[20, top, 660, top + height],
                label="Main script black",
                reading_order=index,
            )
        )
        top += height + 5
    return OCRExtractRequest(
        page_id="p",
        image_b64=PAGE_B64,
        regions=regions,
        options=OCRExtractOptions(backend="kraken_catmus", apply_proofread=False),
    )


@pytest.fixture
def echo(monkeypatch: pytest.MonkeyPatch) -> _EchoBackend:
    backend = _EchoBackend()
    monkeypatch.setattr(ocr_agent, "build_backend_runtime", lambda *_args, **_kwargs: {"kraken_catmus": backend})
    monkeypatch.setattr(ocr_agent, "write_ocr_evidence_jsonl", lambda _record: None)
    return backend


def test_the_page_is_decoded_once_for_all_its_lines(echo: _EchoBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    decodes: list[int] = []

    def counting_decode(image_b64: str) -> Image.Image:
        decodes.append(1)
        return decode_page(image_b64)

    monkeypatch.setattr(ocr_agent, "decode_page", counting_decode)
    result = ocr_agent.run_ocr_extraction(_request([40, 40, 40, 40]), client=_DummyClient(), upscale_factor=2)
    assert result.text.splitlines() == ["line l0", "line l1", "line l2", "line l3"]
    assert len(decodes) == 1


def test_only_lines_shorter_than_a_recogniser_line_are_upscaled(echo: _EchoBackend) -> None:
    ocr_agent.run_ocr_extraction(_request([40, 150]), client=_DummyClient(), upscale_factor=2)
    short, tall = echo.crop_heights
    assert short > 2 * 40  # the padded 40 px line, doubled
    assert tall < 2 * 150  # a 150 px line, padded but not doubled


@pytest.mark.parametrize(
    ("region", "expected"),
    [
        (OCRRegionInput(region_id="a", bbox_xyxy=[0, 0, 500, 44]), 2),
        (OCRRegionInput(region_id="b", bbox_xyxy=[0, 0, 500, 218]), 1),
        (OCRRegionInput(region_id="c", polygon=[[0, 0], [500, 10], [500, 60], [0, 50]]), 2),
        (OCRRegionInput(region_id="d", polygon=[[0, 0], [500, 10], [500, 400], [0, 390]]), 1),
    ],
)
def test_region_upscale(region: OCRRegionInput, expected: int) -> None:
    assert ocr_agent._region_upscale(region, 2) == expected
