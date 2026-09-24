from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from archai_ocr.config import AppConfig
from archai_ocr.pipeline.page import Point, RecognizedLine, mean_confidence
from archai_ocr.pipeline.zone_lines import line_bands

_KRAKEN_INSTALL_HINT = (
    "Kraken is required for the recognition stage but is not installed in this "
    "environment. Install it with:  pip install 'kraken>=5.3.0'  "
    "(or `pip install -e .` from the repository root). "
    "Layout detection and cropping work without it; use --dry-run to validate "
    "configuration and inputs only."
)

try:
    from kraken import binarization, blla, pageseg, rpred
    from kraken.containers import BaselineLine, Segmentation
    from kraken.lib.models import load_any
    from kraken.lib.vgsl import TorchVGSLModel

    KRAKEN_AVAILABLE = True
    _KRAKEN_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - exercised only without kraken installed
    binarization = blla = pageseg = rpred = None
    load_any = TorchVGSLModel = None
    BaselineLine = Segmentation = None
    KRAKEN_AVAILABLE = False
    _KRAKEN_IMPORT_ERROR = exc

# "page": Kraken's baseline segmenter, for zones as large as a text block.
# "zone": ink-profile bands, for the small zones that segmenter finds nothing in.
Segmenter = Literal["page", "zone"]


def require_kraken() -> None:
    """Raise a user-facing error if the optional Kraken dependency is missing."""
    if not KRAKEN_AVAILABLE:
        raise RuntimeError(_KRAKEN_INSTALL_HINT) from _KRAKEN_IMPORT_ERROR


def recognize_crops(
    crop_paths: Sequence[Path],
    config: AppConfig,
    logger: logging.Logger,
) -> list[str]:
    """One transcription per crop: its non-empty lines, newline-joined."""
    return [
        "\n".join(line.text for line in lines if line.text)
        for lines in recognize_crop_lines(crop_paths, config, logger)
    ]


def recognize_crop_lines(
    crop_paths: Sequence[Path],
    config: AppConfig,
    logger: logging.Logger,
    *,
    segmenter: Segmenter = "page",
) -> list[list[RecognizedLine]]:
    """Recognise every crop, keeping each line's geometry and confidence.

    Geometry is in crop pixels. A crop that fails yields no lines rather than
    aborting the page, so the result stays aligned with *crop_paths*.
    """
    if not crop_paths:
        return []
    require_kraken()
    device = config.runtime.kraken_device
    rec_model = _recognition_model(config.weights.kraken_recognition, device)
    seg_model = (
        _segmentation_model(config.weights.kraken_segmentation, device, logger)
        if segmenter == "page"
        else None
    )

    results: list[list[RecognizedLine]] = []
    failed_regions = 0
    for index, crop_path in enumerate(crop_paths):
        try:
            with Image.open(crop_path) as im:
                crop_img = im.convert("L")
            if segmenter == "zone":
                segmentation = _zone_segmentation(crop_img)
            else:
                segmentation = _segment_crop(crop_img, seg_model)
            records = _run_recognition(rec_model, crop_img, segmentation)
        except Exception:
            # One unreadable region must not discard the rest of the page.
            failed_regions += 1
            logger.exception(
                "htr.region_failed",
                extra={"stage": "htr", "count": index, "output": str(crop_path)},
            )
            results.append([])
            continue
        # Each record carries its own geometry. The previous per-crop PAGE writer
        # paired texts with segmentation boxes by index after dropping empty
        # predictions, so every line after the first empty one got the wrong box.
        results.append([_line_from_record(record, logger) for record in records])

    if failed_regions:
        logger.warning(
            "htr.regions_failed",
            extra={"stage": "htr", "count": failed_regions},
        )
    return results


def _recognition_model(path: Path, device: str) -> Any:
    try:
        return _load_recognition_model(str(path), path.stat().st_mtime_ns, device)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Kraken recognition model. Expected a Kraken-compatible model file at: {path}"
        ) from exc


def _segmentation_model(path: Path, device: str, logger: logging.Logger) -> Any | None:
    if not path.exists():
        logger.warning(
            "Kraken segmentation model not found. Falling back to legacy segmentation.",
            extra={"stage": "htr"},
        )
        return None
    try:
        return _load_segmentation_model(str(path), path.stat().st_mtime_ns, device)
    except Exception:
        logger.warning(
            "Failed to load Kraken segmentation model. Falling back to legacy segmentation.",
            extra={"stage": "htr"},
        )
        return None


# Models were reloaded for every page (0.9 s per page for the shipped pair).
# The key includes the file's mtime, so replacing a model is never served stale.
@lru_cache(maxsize=4)
def _load_recognition_model(path: str, mtime_ns: int, device: str) -> Any:  # noqa: ARG001
    model = load_any(path)
    _try_move_to_device(model, device, logging.getLogger("archai_ocr"))
    return model


@lru_cache(maxsize=4)
def _load_segmentation_model(path: str, mtime_ns: int, device: str) -> Any:  # noqa: ARG001
    model = TorchVGSLModel.load_model(path)
    _try_move_to_device(model, device, logging.getLogger("archai_ocr"))
    return model


def _line_from_record(record: Any, logger: logging.Logger | None = None) -> RecognizedLine:
    """Text, mean confidence and geometry of one Kraken record, in crop pixels."""
    text = _extract_prediction_text(record, logger)
    confidences = _field(record, "confidences") or []
    baseline = _points(_field(record, "baseline"))
    boundary = _points(_field(record, "boundary"))
    if not boundary:
        boundary = _bbox_polygon(_field(record, "bbox"))
    return RecognizedLine(
        text=text,
        confidence=mean_confidence(confidences) if text else None,
        baseline=baseline,
        boundary=boundary,
    )


def _field(record: Any, name: str) -> Any:
    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


def _points(raw: Any) -> tuple[Point, ...]:
    """Coordinate pairs from Kraken's [[x, y], ...] (or dict) point lists."""
    if not raw:
        return ()
    points: list[Point] = []
    for point in raw:
        if isinstance(point, dict):
            points.append((int(point.get("x", 0)), int(point.get("y", 0))))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append((int(point[0]), int(point[1])))
    return tuple(points)


def _bbox_polygon(raw: Any) -> tuple[Point, ...]:
    """The rectangle of an (x1, y1, x2, y2) box, as the legacy segmenter reports."""
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return ()
    x1, y1, x2, y2 = (int(v) for v in raw[:4])
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


# Where the baseline falls in a band running from ascenders to descenders: about
# three quarters of the way down. Kraken centres the line image on it.
_BASELINE_DEPTH = 0.78


def _zone_segmentation(image: Image.Image) -> Any:
    """One straight, full-width baseline line per ink band of a small zone."""
    width = image.width
    lines = []
    for index, (top, bottom) in enumerate(line_bands(image)):
        baseline_y = top + int(_BASELINE_DEPTH * (bottom - top))
        lines.append(
            BaselineLine(
                id=f"line_{index}",
                baseline=[(0, baseline_y), (width - 1, baseline_y)],
                boundary=[(0, top), (width - 1, top), (width - 1, bottom - 1), (0, bottom - 1)],
            )
        )
    return Segmentation(
        type="baselines",
        imagename="zone",
        text_direction="horizontal-lr",
        script_detection=False,
        lines=lines,
        regions={},
        line_orders=[],
    )


def _segment_crop(image: Image.Image, seg_model: Any | None) -> Any:
    if seg_model is not None:
        try:
            return blla.segment(image, model=seg_model)
        except TypeError:
            return blla.segment(image, segmentation_model=seg_model)

    bin_img = binarization.nlbin(image)
    try:
        return pageseg.segment(bin_img)
    except TypeError:
        return pageseg.segment(bin_img, text_direction="horizontal-lr")


def _run_recognition(rec_model: Any, image: Image.Image, segmentation: Any) -> list[Any]:
    # kraken is untyped, so these closures are Callable[[], Any]; annotate the
    # tuple so strict mode does not treat each call as an untyped call.
    attempts: tuple[Callable[[], list[Any]], ...] = (
        lambda: list(rpred.rpred(rec_model, image, segmentation)),
        lambda: list(rpred.rpred(rec_model, image, bounds=segmentation)),
        lambda: list(rpred.rpred(network=rec_model, im=image, bounds=segmentation)),
    )
    last_exc: Exception | None = None
    for attempt in attempts:
        try:
            records: list[Any] = attempt()
        except TypeError as exc:
            last_exc = exc
            continue
        else:
            return records
    raise RuntimeError(f"Unable to run Kraken recognition with available call signatures: {last_exc}")


def _extract_prediction_text(record: Any, logger: logging.Logger | None = None) -> str:
    """Pull the recognized string out of a Kraken record.

    Anything we cannot positively identify yields an empty string. Falling back to
    str(record) put reprs such as "<kraken.rpred.ocr_record object at 0x...>" into
    the transcription, where they are indistinguishable from recognized text.
    """
    if record is None:
        return ""

    if isinstance(record, dict):
        text = record.get("prediction") or record.get("text")
        return str(text).strip() if text is not None else ""

    for attribute in ("prediction", "text"):
        value = getattr(record, attribute, None)
        if isinstance(value, str):
            return value.strip()
        if value is not None:
            return str(value).strip()

    if isinstance(record, str):
        return record.strip()

    if logger is not None:
        logger.warning(
            "htr.unrecognized_record_type",
            extra={"stage": "htr", "output": type(record).__name__},
        )
    return ""


def _try_move_to_device(model: Any, device: str, logger: logging.Logger) -> None:
    if not device:
        return
    if hasattr(model, "to"):
        try:
            model.to(device)
        except Exception:
            logger.warning(
                "Unable to move model to device; continuing with framework default.",
                extra={"stage": "htr"},
            )
