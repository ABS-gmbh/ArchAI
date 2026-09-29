"""Tests for the engine behind /ocr/extract_full_page.

The route used to send every page to GLM-OCR. The line-level Kraken path that
reads medieval hands far better could not take over, because of defects on the
way to it:

* The upload was written to a temp file with no extension. The layout detector
  picks its image loader by extension, so every RGB page within the size limits
  - 9 of 10 real manuscript pages - raised "No images or videos found" and was
  read with no layout at all.
* Text outside the columns (a library stamp under the text, marginal numbers,
  the edge of the facing page) was ordered into the body, so a watermark became
  the last line of a column.
* Kraken's CATMuS models emit NFD, which splits words in two for every tokenizer
  downstream.
"""

from __future__ import annotations

import asyncio
import base64
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.routers import ocr as ocr_router  # type: ignore[import-untyped]  # noqa: E402
from app.schemas.agents_ocr import (  # type: ignore[import-untyped]  # noqa: E402
    OCRExtractResponse,
    OCRProvenance,
    OCRRawOCRPayload,
    OCRRegionResult,
    SaiaFullPageExtractRequest,
)
from app.services import ocr_backends  # type: ignore[import-untyped]  # noqa: E402
from app.services.glm_ollama_ocr import GlmOllamaOcrResult  # type: ignore[import-untyped]  # noqa: E402


def _image_bytes(fmt: str, size: tuple[int, int] = (60, 40), mode: str = "RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, "white").save(buffer, format=fmt)
    return buffer.getvalue()


PAGE = _image_bytes("PNG", (1000, 1400))


def _coco(*boxes: tuple[str, float, float, float, float]) -> dict[str, Any]:
    names = sorted({name for name, *_ in boxes})
    ids = {name: index + 1 for index, name in enumerate(names)}
    return {
        "categories": [{"id": ids[name], "name": name} for name in names],
        "annotations": [
            {"id": f"a{index}", "category_id": ids[name], "bbox": [x, y, w, h]}
            for index, (name, x, y, w, h) in enumerate(boxes)
        ],
    }


COLUMN = ("Column", 100.0, 100.0, 700.0, 1000.0)


def _line(y: float, x: float = 120.0, w: float = 640.0) -> tuple[str, float, float, float, float]:
    return ("Main script black", x, y, w, 40.0)


# --------------------------------------------------------- segmentation input --


@pytest.mark.parametrize(
    ("fmt", "suffix"),
    [("JPEG", ".jpg"), ("PNG", ".png"), ("TIFF", ".tif"), ("BMP", ".bmp"), ("WEBP", ".webp")],
)
def test_the_upload_is_written_with_its_image_suffix(tmp_path: Path, fmt: str, suffix: str) -> None:
    data = _image_bytes(fmt)
    path = ocr_router._write_segmentation_input(tmp_path, data)
    assert path.suffix == suffix
    assert path.read_bytes() == data


def test_a_format_the_detector_cannot_load_is_re_encoded(tmp_path: Path) -> None:
    path = ocr_router._write_segmentation_input(tmp_path, _image_bytes("GIF", mode="P"))
    assert path.suffix == ".png"
    with Image.open(path) as image:
        assert (image.format, image.size) == ("PNG", (60, 40))


def test_bytes_that_are_not_an_image_still_reach_the_422(tmp_path: Path) -> None:
    path = ocr_router._write_segmentation_input(tmp_path, b"not an image")
    with pytest.raises(ocr_router.HTTPException) as caught:
        ocr_router._prepare_image_for_segmentation(str(path))
    assert caught.value.status_code == 422


def test_an_rgb_page_reaches_the_detector_as_a_loadable_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the RGB JPEG went through untouched, as a file with no suffix."""
    seen: list[str] = []

    def fake_segmentation(path: str, **_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        seen.append(path)
        assert Path(path).is_file()
        return {"images": [], "annotations": [], "categories": []}, {}

    monkeypatch.setattr(ocr_router, "run_single_segmentation", fake_segmentation)
    ocr_router._segment_page(_image_bytes("JPEG", (1200, 1600)))
    assert Path(seen[0]).suffix == ".jpg"


# ---------------------------------------------------------------------- lanes --


def test_text_outside_the_columns_leaves_the_body() -> None:
    coco = _coco(COLUMN, _line(150), _line(250), _line(350), ("Main script black", 300.0, 1250.0, 400.0, 40.0))
    body, outside = ocr_router._segmented_region_lanes(coco)
    assert [region.region_id for region in body] == ["a1", "a2", "a3"]
    assert [region.region_id for region in outside] == ["a4"]
    assert [region.reading_order for region in body] == [0, 1, 2]


def test_the_old_ordering_made_the_stamp_the_columns_last_line() -> None:
    """Precondition: any row under a column's x-span joined it."""
    coco = _coco(COLUMN, _line(150), _line(250), _line(350), ("Main script black", 300.0, 1250.0, 400.0, 40.0))
    assert [region.region_id for region in ocr_router._extract_segmented_regions(coco)][-1] == "a4"


def test_marginal_text_is_kept_in_page_order() -> None:
    coco = _coco(
        COLUMN,
        *[_line(150 + 100 * index) for index in range(5)],
        ("Main script black", 20.0, 600.0, 60.0, 30.0),
        ("Main script black", 20.0, 300.0, 60.0, 30.0),
    )
    _body, outside = ocr_router._segmented_region_lanes(coco)
    assert [region.region_id for region in outside] == ["a7", "a6"]
    assert [region.reading_order for region in outside] == [0, 1]


def test_a_line_just_over_the_column_edge_stays_in_the_body() -> None:
    """The detector's column box often clips the line ends it contains."""
    coco = _coco(COLUMN, _line(150), _line(250, x=640.0, w=330.0))  # centre 5 px right of the box
    body, outside = ocr_router._segmented_region_lanes(coco)
    assert (len(body), outside) == (2, [])


def test_a_page_whose_columns_were_missed_keeps_everything_in_the_body() -> None:
    """A two-page spread with one page's column undetected: most lines are outside."""
    left = [_line(150 + 100 * index) for index in range(3)]
    right = [("Main script black", 1000.0, 150.0 + 100 * index, 640.0, 40.0) for index in range(5)]
    coco = _coco(COLUMN, *left, *right)
    body, outside = ocr_router._segmented_region_lanes(coco)
    assert outside == []
    assert [region.region_id for region in body] == [
        region.region_id for region in ocr_router._extract_segmented_regions(coco)
    ]


def test_a_page_without_columns_keeps_the_legacy_order() -> None:
    coco = _coco(_line(150), _line(250), ("Main script black", 900.0, 150.0, 300.0, 40.0))
    body, outside = ocr_router._segmented_region_lanes(coco)
    assert outside == []
    assert [region.region_id for region in body] == [
        region.region_id for region in ocr_router._extract_segmented_regions(coco)
    ]


# ------------------------------------------------------------------ geometry --


def test_an_exif_rotated_photo_is_made_upright() -> None:
    image = Image.new("RGB", (300, 200), "white")
    exif = image.getexif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes())
    with Image.open(io.BytesIO(ocr_router._upright_page_bytes(buffer.getvalue()))) as upright:
        assert upright.size == (200, 300)


def test_a_page_without_orientation_is_passed_through() -> None:
    assert ocr_router._upright_page_bytes(PAGE) is PAGE


# ----------------------------------------------------------------- Kraken text --


def test_kraken_output_is_nfc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("kraken")
    from kraken import rpred

    model_path = tmp_path / "catmus.mlmodel"
    model_path.write_bytes(b"weights")
    monkeypatch.setattr(ocr_backends, "_load_kraken_model", lambda _path, _device: SimpleNamespace(seg_type="bbox"))
    records = [SimpleNamespace(prediction="mẽt en autre", confidences=[0.9, 0.8])]
    monkeypatch.setattr(rpred, "rpred", lambda *_args, **_kwargs: iter(records))
    backend = ocr_backends.KrakenBackend(
        backend_name="kraken_catmus", model_name="CATMuS Medieval", configured_path=str(model_path)
    )
    crop = base64.b64encode(_image_bytes("PNG", (400, 60))).decode("ascii")
    metadata = ocr_backends.OCRRecognitionMetadata(page_id=None, image_id=None, region_id="r1", bbox_xyxy=[0, 0, 400, 60])

    result = backend.recognize(crop, metadata)

    assert result.text == "mẽt en autre"


# ---------------------------------------------------------------------- route --


class _LineAgent:
    """Reads every region as its own id, and records the requests it served."""

    def __init__(self, *, reads: bool = True) -> None:
        self.reads = reads
        self.requests: list[Any] = []

    def run(self, payload: Any) -> OCRExtractResponse:
        self.requests.append(payload)
        lines = [f"line {region.region_id}" for region in payload.regions] if self.reads else []
        text = "\n".join(lines)
        return OCRExtractResponse(
            status="FULL",
            model="CATMuS Medieval",
            ocr_backend="kraken_catmus",
            fallbacksUsed=[],
            warnings=[],
            text=text,
            script_hint="latin_medieval",
            final_text=text,
            page_id=payload.page_id,
            image_id=payload.image_id,
            fallbacks=[],
            regions=[
                OCRRegionResult(
                    region_id=region.region_id,
                    text=f"line {region.region_id}",
                    quality=0.8,
                    confidence=0.9,
                    flags=[],
                    backend_name="kraken_catmus",
                    model_name="CATMuS Medieval",
                )
                for region in payload.regions
            ],
            provenance=OCRProvenance(
                crop_sha256="sha", prompt_version="test", agent_version="test", timestamp="2026-09-29T00:00:00+00:00"
            ),
            raw_ocr=OCRRawOCRPayload(lines=lines, text=text, script_hint="latin_medieval", confidence=0.9, warnings=[]),
            comparison_results=[],
            evidence_id="evidence",
            is_evidence=True,
            is_verified=False,
        )


def _glm_result(text: str = "glm page text") -> GlmOllamaOcrResult:
    return GlmOllamaOcrResult(
        text=text,
        lines=text.splitlines(),
        model_used="glm-ocr:latest",
        warnings=[],
        original_size_bytes=len(PAGE),
        original_width=1000,
        original_height=1400,
        processed_width=1000,
        processed_height=1400,
        processed_size_bytes=len(PAGE),
        preprocessing_applied=False,
        processed_variant_name="rgb_jpeg_1536",
        attempts_used=1,
        duration_seconds=0.1,
    )


@pytest.fixture
def route(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"agent": _LineAgent(), "glm_calls": 0, "pipeline": []}

    def fake_glm(*_args: Any, **_kwargs: Any) -> GlmOllamaOcrResult:
        state["glm_calls"] += 1
        return _glm_result()

    def fake_pipeline(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        state["pipeline"].append(kwargs["ocr_backend"])
        return {}

    monkeypatch.setattr(ocr_router, "_get_ocr_agent", lambda: state["agent"])
    monkeypatch.setattr(ocr_router, "run_glm_ollama_ocr", fake_glm)
    monkeypatch.setattr(ocr_router, "_run_full_page_post_ocr_pipeline", fake_pipeline)
    monkeypatch.setattr(ocr_router, "get_test_ocr_fixture", lambda _data: None)
    monkeypatch.setattr(ocr_router, "get_test_ocr_override", lambda _data: None)
    return state


def _extract(**fields: Any) -> Any:
    payload = SaiaFullPageExtractRequest(
        page_id="page-1", image_b64=base64.b64encode(PAGE).decode("ascii"), apply_proofread=False, **fields
    )
    return asyncio.run(ocr_router.ocr_extract_full_page(payload))


def test_the_route_reads_segmented_lines_by_default(route: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    coco = _coco(COLUMN, _line(150), _line(250), _line(350), ("Main script black", 300.0, 1250.0, 400.0, 40.0))
    monkeypatch.setattr(ocr_router, "_segment_page", lambda _data: coco)

    result = _extract()

    assert result.ocr_engine == "segmented"
    assert result.model_used == "CATMuS Medieval"
    assert result.lines == ["line a1", "line a2", "line a3"]
    assert result.secondary_lines == ["line a4"]
    assert (result.original_image_width, result.original_image_height) == (1000, 1400)
    assert route["glm_calls"] == 0
    assert route["pipeline"] == ["kraken_catmus"]
    body_request, secondary_request = route["agent"].requests
    assert body_request.options.backend == "auto"
    assert secondary_request.options.backend == "kraken_catmus"  # the lanes are read by one model


def _no_text_regions(_data: bytes) -> dict[str, Any]:
    return _coco(COLUMN, ("Illustrations", 120.0, 150.0, 400.0, 300.0))


def _missing_weights(_data: bytes) -> dict[str, Any]:
    raise RuntimeError("zone weights missing")


@pytest.mark.parametrize(
    ("segmentation", "reason"),
    [
        (_no_text_regions, "OCR_ENGINE_FALLBACK:no_text_regions"),
        (_missing_weights, "OCR_ENGINE_FALLBACK:segmented_failed:zone weights missing"),
    ],
)
def test_the_route_falls_back_to_glm_ocr(
    route: dict[str, Any], monkeypatch: pytest.MonkeyPatch, segmentation: Any, reason: str
) -> None:
    monkeypatch.setattr(ocr_router, "_segment_page", segmentation)

    result = _extract()

    assert (result.ocr_engine, result.text, route["glm_calls"]) == ("glmocr", "glm page text", 1)
    assert reason in result.warnings
    assert route["pipeline"] == ["glmocr"]


def test_a_page_the_recogniser_reads_as_nothing_goes_to_glm_ocr(
    route: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ocr_router, "_segment_page", lambda _data: _coco(COLUMN, _line(150)))
    route["agent"] = _LineAgent(reads=False)

    result = _extract()

    assert result.ocr_engine == "glmocr"
    assert "OCR_ENGINE_FALLBACK:no_text" in result.warnings


def test_an_explicit_glm_request_skips_segmentation(route: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_router, "_segment_page", lambda _data: pytest.fail("segmentation must not run"))
    assert _extract(ocr_backend="glmocr").ocr_engine == "glmocr"


def test_the_setting_restores_glm_ocr_for_auto(route: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_router._app_settings, "full_page_ocr_engine", "glmocr")
    monkeypatch.setattr(ocr_router, "_segment_page", lambda _data: pytest.fail("segmentation must not run"))
    assert _extract().ocr_engine == "glmocr"


def test_a_pinned_kraken_backend_is_passed_to_the_agent(route: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_router, "_segment_page", lambda _data: _coco(COLUMN, _line(150)))
    _extract(ocr_backend="kraken_cremma_medieval")
    assert route["agent"].requests[0].options.backend == "kraken_cremma_medieval"
