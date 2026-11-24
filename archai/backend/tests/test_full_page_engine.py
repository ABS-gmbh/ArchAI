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
import json
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


@pytest.mark.requires_models
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

    def run(self, payload: Any, *, progress: Any = None) -> OCRExtractResponse:
        self.requests.append(payload)
        self.progress_given = progress is not None
        for done in range(1, len(payload.regions) + 1):
            if progress is not None:
                progress(done, len(payload.regions))
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
        # The error's type, not its text (security review C8): the text is logged.
        (_missing_weights, "OCR_ENGINE_FALLBACK:segmented_failed:RuntimeError"),
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


class _PunctuatedAgent(_LineAgent):
    """Reads lines holding a question mark and an ellipsis, as a recogniser can."""

    def run(self, payload: Any, *, progress: Any = None) -> OCRExtractResponse:
        result = super().run(payload, progress=progress)
        text = "\n".join(f"Que diras tu? {region.region_id} … e" for region in payload.regions)
        return result.model_copy(update={"text": text, "final_text": text})


def test_the_segmented_route_keeps_question_marks_and_ellipses(route: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing on this route inserts uncertainty marks: a "?" is the manuscript's punctuation."""
    coco = _coco(COLUMN, _line(150), _line(250), ("Main script black", 300.0, 1250.0, 400.0, 40.0))
    monkeypatch.setattr(ocr_router, "_segment_page", lambda _data: coco)
    route["agent"] = _PunctuatedAgent()

    result = _extract()

    assert result.lines == ["Que diras tu? a1 … e", "Que diras tu? a2 … e"]
    assert result.text == "Que diras tu? a1 … e\nQue diras tu? a2 … e"
    assert result.secondary_lines == ["Que diras tu? a3 … e"]


def test_the_marker_stripping_used_by_the_other_paths_is_unchanged() -> None:
    assert ocr_router._strip_uncertainty_lines(["Que [ ]a? b […] c…", "  ", "x  y"]) == ["Que [ ]a b c", "x y"]
    assert ocr_router._strip_uncertainty_lines(["a?  b"], strip_markers=False) == ["a? b"]


# ------------------------------------------------------------ line regions --


class _BoxAgent(_LineAgent):
    """Like the real agent: each region keeps its input box and polygon, and its text is one line."""

    def __init__(self, *, texts: dict[str, str] | None = None, proofread: str | None = None) -> None:
        super().__init__()
        self.texts = texts or {}
        self.proofread = proofread

    def run(self, payload: Any, *, progress: Any = None) -> OCRExtractResponse:
        result = super().run(payload, progress=progress)
        regions = [
            OCRRegionResult(
                region_id=region.region_id,
                text=self.texts.get(region.region_id, f"line {region.region_id}"),
                quality=0.8,
                confidence=0.9,
                flags=[],
                bbox_xyxy=list(region.bbox_xyxy) if region.bbox_xyxy else None,
                polygon=region.polygon,
                reading_order=region.reading_order,
                backend_name="kraken_catmus",
                model_name="CATMuS Medieval",
            )
            for region in payload.regions
        ]
        text = self.proofread or "\n".join(region.text for region in regions)
        return result.model_copy(update={"regions": regions, "text": text, "final_text": text})


def _rotated_page() -> bytes:
    """A photo stored 1400x1000 with EXIF orientation 6: upright, it is 1000x1400."""
    image = Image.new("RGB", (1400, 1000), (200, 190, 170))
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def test_each_region_is_tied_to_its_span_of_text_and_its_box(route: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def segment(data: bytes) -> dict[str, Any]:
        with Image.open(io.BytesIO(data)) as page:
            seen["size"] = page.size  # the frame the boxes are in
        return _coco(COLUMN, _line(150), _line(250), _line(350), ("Main script black", 300.0, 1250.0, 400.0, 40.0))

    monkeypatch.setattr(ocr_router, "_segment_page", segment)
    route["agent"] = _BoxAgent(texts={"a2": "Que   diras tu?"})
    payload = SaiaFullPageExtractRequest(page_id="page-1", image_b64=base64.b64encode(_rotated_page()).decode("ascii"), apply_proofread=False)

    result = asyncio.run(ocr_router.ocr_extract_full_page(payload))

    assert seen["size"] == (1000, 1400) == (result.original_image_width, result.original_image_height)
    body = [line for line in result.line_regions if line.lane == "body"]
    assert [line.region_id for line in body] == ["a1", "a2", "a3"]
    for line in body:
        assert result.text[line.char_start : line.char_end] == line.text
    assert body[1].text == "Que diras tu?"  # cleaned as the response text is
    assert body[0].bbox_xyxy == [120.0, 150.0, 760.0, 190.0]  # upright-page pixels, as segmented
    assert all(0 <= x <= 1000 for line in body for x in line.bbox_xyxy[::2])
    assert all(0 <= y <= 1400 for line in body for y in line.bbox_xyxy[1::2])
    assert (body[0].confidence, body[0].backend_name, body[0].reading_order) == (0.9, "kraken_catmus", 0)
    [margin] = [line for line in result.line_regions if line.lane == "secondary"]
    assert (margin.region_id, margin.char_start, margin.char_end) == ("a4", None, None)
    assert margin.bbox_xyxy == [300.0, 1250.0, 700.0, 1290.0]


def test_a_region_rewritten_by_the_proofreader_gets_no_offsets(route: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_router, "_segment_page", lambda _data: _coco(COLUMN, _line(150), _line(250)))
    route["agent"] = _BoxAgent(proofread="line a1\nsomething else")
    result = _extract()
    first, second = result.line_regions
    assert result.text[first.char_start : first.char_end] == "line a1"
    assert (second.char_start, second.char_end) == (None, None)


def test_existing_fields_are_unchanged_by_line_regions(route: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    coco = _coco(COLUMN, _line(150), _line(250), ("Main script black", 300.0, 1250.0, 400.0, 40.0))
    monkeypatch.setattr(ocr_router, "_segment_page", lambda _data: coco)
    with_regions = _extract().model_dump(exclude={"line_regions"})
    monkeypatch.setattr(ocr_router, "_full_page_lines", lambda *args, **kwargs: [])
    without = _extract().model_dump(exclude={"line_regions"})
    assert json.dumps(with_regions, sort_keys=True) == json.dumps(without, sort_keys=True)
    assert with_regions["lines"] == ["line a1", "line a2"] and with_regions["secondary_lines"] == ["line a3"]


def test_glm_ocr_returns_no_line_regions(route: dict[str, Any]) -> None:
    result = _extract(ocr_backend="glmocr")
    assert result.ocr_engine == "glmocr" and result.line_regions is None


# ------------------------------------------------------------ client runs --


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from app.db import pipeline_db

    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    return pipeline_db


def _events(pipeline_db: Any, run_id: str) -> list[tuple[str, str]]:
    return [(event["stage"], event["event"]) for event in pipeline_db.list_events(run_id)]


def test_a_reserved_run_carries_the_request_trace_from_the_start(route: dict[str, Any], db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_router, "_segment_page", lambda _data: _coco(COLUMN, _line(150), _line(250)))
    calls: list[dict[str, Any]] = []

    def pipeline(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        db.update_run_fields(kwargs["run_id"], status="COMPLETED", current_stage="DONE")
        db.log_event(kwargs["run_id"], "DONE", "END", "done")
        return {"run_id": kwargs["run_id"]}

    monkeypatch.setattr(ocr_router, "_run_full_page_post_ocr_pipeline", pipeline)
    run_id = asyncio.run(ocr_router.ocr_reserve_run())["run_id"]
    assert db.get_run(run_id)["status"] == "RESERVED"

    result = _extract(run_id=run_id)

    assert result.ocr_engine == "segmented"
    assert result.run_id == run_id and calls[0]["run_id"] == run_id
    assert _events(db, run_id) == [
        ("RECEIVED", "START"), ("OCR_RUNNING", "START"), ("OCR_RUNNING", "INFO"), ("OCR_DONE", "END"), ("DONE", "END"),
    ]
    assert db.list_events(run_id)[2]["message"] == "lines 2/2"
    assert db.get_run(run_id)["status"] == "COMPLETED"
    assert db.get_run(run_id)["asset_ref"] == "page-1"


def test_an_unknown_or_used_run_id_is_refused(route: dict[str, Any], db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    monkeypatch.setattr(ocr_router, "_segment_page", lambda _data: _coco(COLUMN, _line(150)))
    monkeypatch.setattr(ocr_router, "_run_full_page_post_ocr_pipeline", lambda *a, **k: {"run_id": k.get("run_id")})
    with pytest.raises(HTTPException) as unknown:
        _extract(run_id="not-a-run")
    assert unknown.value.status_code == 404 and "POST /ocr/runs" in unknown.value.detail

    run_id = asyncio.run(ocr_router.ocr_reserve_run())["run_id"]
    _extract(run_id=run_id)
    with pytest.raises(HTTPException) as reused:
        _extract(run_id=run_id)
    assert reused.value.status_code == 409

    # Somebody else's run, created the ordinary way, cannot be written into either.
    other = db.create_run(asset_ref="someone-else", asset_sha256="sha")
    before = _events(db, other)
    with pytest.raises(HTTPException) as foreign:
        _extract(run_id=other)
    assert foreign.value.status_code == 409 and _events(db, other) == before
    assert db.get_run(other)["asset_ref"] == "someone-else"


def test_a_reserved_run_is_claimed_once(db: Any) -> None:
    run_id = db.reserve_run()
    assert db.claim_reserved_run(run_id, asset_ref="a", asset_sha256=None) == "claimed"
    assert db.claim_reserved_run(run_id, asset_ref="b", asset_sha256=None) == "used"
    assert db.claim_reserved_run("missing", asset_ref="c", asset_sha256=None) == "unknown"


def test_a_failed_request_ends_its_reserved_run(route: dict[str, Any], db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    async def glm_down(*_args: Any, **_kwargs: Any) -> Any:
        raise HTTPException(status_code=502, detail="GLM-OCR unreachable")

    monkeypatch.setattr(ocr_router, "_run_glm_full_page", glm_down)
    run_id = asyncio.run(ocr_router.ocr_reserve_run())["run_id"]
    with pytest.raises(HTTPException):
        _extract(run_id=run_id, ocr_backend="glmocr")
    run = db.get_run(run_id)
    assert run["status"] == "FAILED" and "GLM-OCR unreachable" in run["error"]
    assert _events(db, run_id)[-1] == ("FAILED", "ERROR")


def test_without_a_run_id_nothing_changes(route: dict[str, Any], db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_router, "_segment_page", lambda _data: _coco(COLUMN, _line(150)))
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(ocr_router, "_run_full_page_post_ocr_pipeline", lambda *a, **k: calls.append(k) or {})
    _extract()
    assert "run_id" not in calls[0]


def test_an_empty_reading_still_ends_the_reserved_run(db: Any) -> None:
    from app.schemas.agents_ocr import SaiaFullPageExtractResponse
    from app.services import trace_pipeline

    run_id = db.reserve_run()
    db.claim_reserved_run(run_id, asset_ref="page-1", asset_sha256=None)
    empty = SaiaFullPageExtractResponse(status="EMPTY", model_used="CATMuS Medieval", text="")
    payload = SaiaFullPageExtractRequest(page_id="page-1", image_b64=base64.b64encode(PAGE).decode("ascii"))
    assert trace_pipeline._run_full_page_post_ocr_pipeline(payload, empty, PAGE, ocr_backend="kraken_catmus", run_id=run_id) == {"run_id": run_id}
    assert db.get_run(run_id)["status"] == "COMPLETED"
    assert trace_pipeline._run_full_page_post_ocr_pipeline(payload, empty, PAGE, ocr_backend="kraken_catmus") == {}


# --------------------------------------------------- reservation bounds --


def _age_reservation(db: Any, run_id: str, seconds: int) -> None:
    from datetime import datetime, timedelta, timezone

    created = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    with db._connect() as conn:
        conn.execute("UPDATE pipeline_runs SET created_at=?, updated_at=? WHERE run_id=?", (created, created, run_id))
        conn.commit()


def test_an_expired_reservation_is_refused_and_removed(route: dict[str, Any], db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    from app.config import settings

    monkeypatch.setattr(settings, "ocr_run_reservation_ttl_seconds", 3600)
    monkeypatch.setattr(ocr_router, "_segment_page", lambda _data: _coco(COLUMN, _line(150)))
    stale = asyncio.run(ocr_router.ocr_reserve_run())["run_id"]
    _age_reservation(db, stale, 2 * 3600)
    with pytest.raises(HTTPException) as refused:
        _extract(run_id=stale)
    assert refused.value.status_code == 410 and "expired" in refused.value.detail
    assert db.get_run(stale)["status"] == "RESERVED"  # refusing does not claim it

    fresh = asyncio.run(ocr_router.ocr_reserve_run())["run_id"]  # the next reservation sweeps
    assert db.get_run(stale) is None
    assert db.get_run(fresh)["status"] == "RESERVED"


def test_the_sweep_never_touches_a_claimed_or_finished_run(db: Any) -> None:
    claimed = db.reserve_run()
    assert db.claim_reserved_run(claimed, asset_ref="page", asset_sha256=None) == "claimed"
    ordinary = db.create_run(asset_ref="other", asset_sha256=None)
    db.update_run_fields(ordinary, status="COMPLETED")
    for run_id in (claimed, ordinary):
        _age_reservation(db, run_id, 10 * 24 * 3600)
    old_reserved = db.reserve_run()
    _age_reservation(db, old_reserved, 10 * 24 * 3600)

    from datetime import datetime, timezone

    assert db.delete_expired_reservations(created_before=datetime.now(timezone.utc).isoformat()) == 1
    assert db.get_run(old_reserved) is None
    assert db.get_run(claimed)["status"] == "RUNNING" and db.get_run(ordinary)["status"] == "COMPLETED"


def test_too_many_waiting_reservations_answer_429(db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    from app.config import settings

    monkeypatch.setattr(settings, "ocr_run_reservation_max_live", 3)
    for _ in range(3):
        asyncio.run(ocr_router.ocr_reserve_run())
    with pytest.raises(HTTPException) as capped:
        asyncio.run(ocr_router.ocr_reserve_run())
    assert capped.value.status_code == 429 and "Too many reserved runs" in capped.value.detail
    assert db.count_reserved_runs() == 3


def test_retention_keeps_a_reserved_run_as_not_finished(db: Any) -> None:
    from app.services.run_retention import plan_retention

    reserved = db.reserve_run()
    _age_reservation(db, reserved, 400 * 24 * 3600)
    plan = plan_retention(180, protected=set())
    assert reserved in plan.kept.get("not_finished", []) and reserved not in plan.expired


# ------------------------------------------------------------ line progress --


def test_line_progress_is_throttled_to_every_five_lines_or_one_second(db: Any) -> None:
    run_id = db.create_run(asset_ref="page", asset_sha256=None)
    now = [0.0]
    events = ocr_router._LineProgressEvents(run_id, clock=lambda: now[0])
    for done in range(1, 13):
        now[0] += 0.1  # fast lines: only every fifth, and the last
        events(done, 12)
    slow = ocr_router._LineProgressEvents(run_id, clock=lambda: now[0])
    for done in range(1, 4):
        now[0] += 1.5  # slow lines: each one
        slow(done, 3)
    assert [e["message"] for e in db.list_events(run_id)] == ["lines 5/12", "lines 10/12", "lines 12/12", "lines 1/3", "lines 2/3", "lines 3/3"]


def test_a_five_line_page_streams_line_counts_into_its_reserved_run(route: dict[str, Any], db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_router, "_segment_page", lambda _data: _coco(COLUMN, *[_line(150 + 100 * i) for i in range(5)]))
    monkeypatch.setattr(ocr_router, "_run_full_page_post_ocr_pipeline", lambda *a, **k: {"run_id": k["run_id"]})
    ticks = iter(range(100))
    monkeypatch.setattr(ocr_router.time, "monotonic", lambda: float(next(ticks)) * 1.1)  # each line takes ~1 s
    run_id = asyncio.run(ocr_router.ocr_reserve_run())["run_id"]

    _extract(run_id=run_id)

    assert [(e["stage"], e["event"], e["message"]) for e in db.list_events(run_id)] == [
        ("RECEIVED", "START", "Full-page OCR request received."),
        ("OCR_RUNNING", "START", "Full-page OCR running."),
        ("OCR_RUNNING", "INFO", "lines 1/5"),
        ("OCR_RUNNING", "INFO", "lines 2/5"),
        ("OCR_RUNNING", "INFO", "lines 3/5"),
        ("OCR_RUNNING", "INFO", "lines 4/5"),
        ("OCR_RUNNING", "INFO", "lines 5/5"),
        ("OCR_DONE", "END", "Full-page OCR done (segmented)."),
    ]


def test_without_a_reserved_run_the_agent_gets_no_progress(route: dict[str, Any], db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_router, "_segment_page", lambda _data: _coco(COLUMN, _line(150), _line(250)))
    result = _extract()
    assert result.ocr_engine == "segmented" and route["agent"].progress_given is False


def test_a_reserved_run_logs_ocr_once_through_the_real_post_ocr_stages(route: dict[str, Any], db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The downstream stages must not log RECEIVED/OCR_RUNNING/OCR_DONE again into a reserved run."""
    from app.services import trace_pipeline

    monkeypatch.setattr(ocr_router, "_segment_page", lambda _data: _coco(COLUMN, _line(150), _line(250)))
    monkeypatch.setattr(ocr_router, "_run_full_page_post_ocr_pipeline", trace_pipeline._run_full_page_post_ocr_pipeline)
    monkeypatch.setattr(trace_pipeline, "_run_trace_analysis", lambda run_id, text: ([], [], [], {"skipped": True}))
    monkeypatch.setattr(trace_pipeline, "_run_authority_linking_stage", lambda run_id: None)
    monkeypatch.setattr(trace_pipeline, "_auto_index_run", lambda run_id: None)
    monkeypatch.setattr(trace_pipeline, "_build_mention_extraction_report", lambda *a, **k: None)
    monkeypatch.setattr(trace_pipeline, "_build_consolidated_report", lambda *a, **k: None)
    run_id = asyncio.run(ocr_router.ocr_reserve_run())["run_id"]

    _extract(run_id=run_id)

    stages = [(e["stage"], e["event"]) for e in db.list_events(run_id)]
    assert stages[:5] == [("RECEIVED", "START"), ("OCR_RUNNING", "START"), ("OCR_RUNNING", "INFO"), ("OCR_DONE", "END"), ("ANALYZE_RUNNING", "START")]
    assert stages.count(("RECEIVED", "START")) == 1 and stages.count(("OCR_DONE", "END")) == 1
    assert stages[-1] == ("DONE", "END") and db.get_run(run_id)["status"] == "COMPLETED"


# ------------------------------------------------------------- no chat key --

_CHAT_KEY_SETTINGS = ("chat_ai_api_key", "saia_api_key", "archai_chat_ai_api_key", "archai_saia_api_key")
_CHAT_KEY_ENV = ("CHAT_AI_API_KEY", "SAIA_API_KEY", "ARCHAI_CHAT_AI_API_KEY", "ARCHAI_SAIA_API_KEY")


@pytest.fixture
def no_chat_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import settings

    for name in _CHAT_KEY_SETTINGS:
        monkeypatch.setattr(settings, name, "")
    for name in _CHAT_KEY_ENV:
        monkeypatch.delenv(name, raising=False)


def test_the_segmented_route_reads_a_page_without_a_chat_key(no_chat_key: None, monkeypatch: pytest.MonkeyPatch) -> None:
    # Local OCR (layout + Kraken) never calls the chat provider, so a deployment
    # without a chat key must still read pages. The OCR agent built its chat
    # client up front, and that raised SaiaConfigError before any line was read.
    from app.services import trace_pipeline

    def recognize(self: Any, _crop_b64: str, metadata: Any) -> Any:
        return ocr_backends.OCRBackendResult(
            text=f"line {metadata.region_id}",
            confidence=0.9,
            backend_name=self.backend_name,
            model_name=self.model_name,
            raw_metadata={},
            region_id=metadata.region_id,
            page_id=metadata.page_id,
            script_hint=None,
        )

    monkeypatch.setattr(trace_pipeline, "_ocr_agent_instance", None)
    monkeypatch.setattr(ocr_backends.KrakenBackend, "recognize", recognize)
    monkeypatch.setattr(ocr_router, "_segment_page", lambda _data: _coco(COLUMN, _line(150), _line(250)))
    monkeypatch.setattr(ocr_router, "run_glm_ollama_ocr", lambda *_a, **_k: pytest.fail("must not fall back to GLM-OCR"))
    monkeypatch.setattr(ocr_router, "_run_full_page_post_ocr_pipeline", lambda *_a, **_k: {})
    monkeypatch.setattr(ocr_router, "get_test_ocr_fixture", lambda _data: None)
    monkeypatch.setattr(ocr_router, "get_test_ocr_override", lambda _data: None)

    result = _extract()

    assert result.ocr_engine == "segmented"
    assert result.lines == ["line a1", "line a2"]


# ------------------------------------------------------------ versal initials --

_VERSE = {
    "a1": ("C aldꝰ li roys qui mlt fu ber", 2.4),
    "a2": ("L a fist requerre ⁊ demander", 1.7),
    "a3": ("⁊ larges ⁊ cortoiz ⁊ franz", 2.0),
    "a4": ("M arcadigas anõ a uoit", 1.6),
    "a5": ("Qͥ gnͣt conseil uilain", 0.65),
    "a6": ("y nabele ⁊ marcadigas.", 1.6),
}


@pytest.mark.parametrize("enabled", [True, False])
def test_versal_initials_are_joined_in_the_region_texts_and_the_spans_stay_exact(
    enabled: bool, no_chat_key: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import settings
    from app.services import trace_pipeline

    def recognize(self: Any, _crop_b64: str, metadata: Any) -> Any:
        text, ratio = _VERSE[metadata.region_id]
        return ocr_backends.OCRBackendResult(
            text=text,
            confidence=0.9,
            backend_name=self.backend_name,
            model_name=self.model_name,
            raw_metadata={"leading_glyph": {"detached": True, "gap_ratio": ratio}},
            region_id=metadata.region_id,
            page_id=metadata.page_id,
            script_hint=None,
        )

    monkeypatch.setattr(settings, "ocr_join_detached_initials", enabled)
    monkeypatch.setattr(trace_pipeline, "_ocr_agent_instance", None)
    monkeypatch.setattr(ocr_backends.KrakenBackend, "recognize", recognize)
    monkeypatch.setattr(ocr_router, "_segment_page", lambda _data: _coco(COLUMN, *[_line(150 + 100 * i) for i in range(6)]))
    monkeypatch.setattr(ocr_router, "run_glm_ollama_ocr", lambda *_a, **_k: pytest.fail("must not fall back to GLM-OCR"))
    monkeypatch.setattr(ocr_router, "_run_full_page_post_ocr_pipeline", lambda *_a, **_k: {})
    monkeypatch.setattr(ocr_router, "get_test_ocr_fixture", lambda _data: None)
    monkeypatch.setattr(ocr_router, "get_test_ocr_override", lambda _data: None)

    result = _extract()

    expected = [
        "Caldꝰ li roys qui mlt fu ber",
        "La fist requerre ⁊ demander",
        "⁊ larges ⁊ cortoiz ⁊ franz",  # a word sign stays apart
        "Marcadigas anõ a uoit",
        "Qͥ gnͣt conseil uilain",  # a word at word spacing stays apart
        "ynabele ⁊ marcadigas.",
    ]
    assert result.lines == (expected if enabled else [text for text, _ in _VERSE.values()])
    body = [line for line in result.line_regions if line.lane == "body"]
    assert len(body) == 6
    for line in body:
        assert line.char_start is not None and result.text[line.char_start : line.char_end] == line.text


def test_an_unclaimed_reservation_expires_within_minutes_by_default(route: dict[str, Any], db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # One client could hold all 100 reservations for an hour (security review
    # C9). A reservation is claimed by the upload that follows it, so the
    # default TTL covers a slow upload, not an hour.
    from fastapi import HTTPException

    monkeypatch.setattr(ocr_router, "_segment_page", lambda _data: _coco(COLUMN, _line(150)))
    stale = asyncio.run(ocr_router.ocr_reserve_run())["run_id"]
    _age_reservation(db, stale, 11 * 60)
    with pytest.raises(HTTPException) as refused:
        _extract(run_id=stale)
    assert refused.value.status_code == 410
    fresh = asyncio.run(ocr_router.ocr_reserve_run())["run_id"]
    _age_reservation(db, fresh, 9 * 60)
    assert _extract(run_id=fresh).ocr_engine == "segmented"


def test_parallel_reservations_never_pass_the_cap(db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # Second review, E: the count and the insert were separate, and 160
    # parallel requests made 134 rows against a cap of 100.
    import threading

    from fastapi import HTTPException

    from app.config import settings

    monkeypatch.setattr(settings, "ocr_run_reservation_max_live", 10)
    start = threading.Barrier(40)
    outcomes: list[str] = []

    def reserve() -> None:
        start.wait()
        try:
            ocr_router._reserve_bounded_run()
            outcomes.append("reserved")
        except HTTPException as exc:
            outcomes.append(str(exc.status_code))

    threads = [threading.Thread(target=reserve) for _ in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert db.count_reserved_runs() == 10
    assert sorted(outcomes) == ["429"] * 30 + ["reserved"] * 10
