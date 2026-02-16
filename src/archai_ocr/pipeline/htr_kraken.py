from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree as ET

import logging

from PIL import Image
from kraken import binarization, blla, pageseg, rpred
from kraken.lib.models import load_any
from kraken.lib.vgsl import TorchVGSLModel

from archai_ocr.config import AppConfig


def recognize_crops(
    crop_paths: Sequence[Path],
    config: AppConfig,
    logger: logging.Logger,
) -> list[str]:
    try:
        rec_model = load_any(str(config.weights.kraken_recognition))
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Kraken recognition model. "
            f"Expected a Kraken-compatible model file at: {config.weights.kraken_recognition}"
        ) from exc
    _try_move_to_device(rec_model, config.runtime.kraken_device, logger)

    seg_model: Any | None = None
    if config.weights.kraken_segmentation.exists():
        try:
            seg_model = TorchVGSLModel.load_model(str(config.weights.kraken_segmentation))
            _try_move_to_device(seg_model, config.runtime.kraken_device, logger)
        except Exception:
            seg_model = None
            logger.warning(
                "Failed to load Kraken segmentation model. Falling back to legacy segmentation.",
                extra={"stage": "htr"},
            )
    else:
        logger.warning(
            "Kraken segmentation model not found. Falling back to legacy segmentation.",
            extra={"stage": "htr"},
        )

    region_texts: list[str] = []
    for crop_path in crop_paths:
        with Image.open(crop_path) as im:
            crop_img = im.convert("L")
            segmentation = _segment_crop(crop_img, seg_model)
            records = _run_recognition(rec_model, crop_img, segmentation)

        lines = [_extract_prediction_text(record) for record in records]
        lines = [line for line in lines if line]
        region_text = "\n".join(lines)
        region_texts.append(region_text)

        if config.runtime.write_page_xml:
            xml_path = crop_path.with_suffix(".xml")
            _write_page_xml(
                xml_path=xml_path,
                image_name=crop_path.name,
                image_size=crop_img.size,
                segmentation=segmentation,
                line_texts=lines,
            )

    return region_texts


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
    attempts = (
        lambda: list(rpred.rpred(rec_model, image, segmentation)),
        lambda: list(rpred.rpred(rec_model, image, bounds=segmentation)),
        lambda: list(rpred.rpred(network=rec_model, im=image, bounds=segmentation)),
    )
    last_exc: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Unable to run Kraken recognition with available call signatures: {last_exc}")


def _extract_prediction_text(record: Any) -> str:
    if record is None:
        return ""

    if isinstance(record, dict):
        text = record.get("prediction") or record.get("text")
        return str(text).strip() if text is not None else ""

    if hasattr(record, "prediction"):
        return str(getattr(record, "prediction")).strip()
    if hasattr(record, "text"):
        return str(getattr(record, "text")).strip()
    return str(record).strip()


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


def _write_page_xml(
    xml_path: Path,
    image_name: str,
    image_size: tuple[int, int],
    segmentation: Any,
    line_texts: Sequence[str],
) -> None:
    width, height = image_size
    namespace = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
    ET.register_namespace("", namespace)

    root = ET.Element("PcGts", {"xmlns": namespace})
    page = ET.SubElement(
        root,
        "Page",
        {
            "imageFilename": image_name,
            "imageWidth": str(width),
            "imageHeight": str(height),
        },
    )
    region = ET.SubElement(page, "TextRegion", {"id": "region_000"})
    ET.SubElement(region, "Coords", {"points": _bbox_to_points((0, 0, width, height))})

    line_boxes = _line_bboxes(segmentation)
    if not line_boxes:
        line_boxes = [(0, 0, width, height)] * max(1, len(line_texts))
    if line_boxes and len(line_boxes) < len(line_texts):
        line_boxes.extend([line_boxes[-1]] * (len(line_texts) - len(line_boxes)))

    for idx, text in enumerate(line_texts):
        box = line_boxes[idx] if idx < len(line_boxes) else (0, 0, width, height)
        text_line = ET.SubElement(region, "TextLine", {"id": f"line_{idx:03d}"})
        ET.SubElement(text_line, "Coords", {"points": _bbox_to_points(box)})
        text_equiv = ET.SubElement(text_line, "TextEquiv")
        unicode_text = ET.SubElement(text_equiv, "Unicode")
        unicode_text.text = text

    tree = ET.ElementTree(root)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def _line_bboxes(segmentation: Any) -> list[tuple[int, int, int, int]]:
    lines: Iterable[Any] = ()
    if isinstance(segmentation, dict):
        candidate = segmentation.get("lines") or segmentation.get("boxes") or []
        lines = candidate
    elif hasattr(segmentation, "lines"):
        lines = getattr(segmentation, "lines")

    out: list[tuple[int, int, int, int]] = []
    for line in lines:
        bbox = _extract_line_bbox(line)
        if bbox:
            out.append(bbox)
    return out


def _extract_line_bbox(line: Any) -> tuple[int, int, int, int] | None:
    if isinstance(line, dict):
        if "bbox" in line:
            return _normalize_bbox(line["bbox"])
        if "boundary" in line:
            return _boundary_to_bbox(line["boundary"])
    if hasattr(line, "bbox"):
        return _normalize_bbox(getattr(line, "bbox"))
    if hasattr(line, "boundary"):
        return _boundary_to_bbox(getattr(line, "boundary"))
    return None


def _normalize_bbox(raw_bbox: Any) -> tuple[int, int, int, int] | None:
    if raw_bbox is None:
        return None
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
        x1, y1, x2, y2 = raw_bbox[:4]
        return int(x1), int(y1), int(x2), int(y2)
    return None


def _boundary_to_bbox(boundary: Any) -> tuple[int, int, int, int] | None:
    if not boundary:
        return None
    points: list[tuple[int, int]] = []
    for point in boundary:
        if isinstance(point, dict):
            x = int(point.get("x", 0))
            y = int(point.get("y", 0))
            points.append((x, y))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append((int(point[0]), int(point[1])))
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_to_points(bbox: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox
    return f"{x1},{y1} {x2},{y1} {x2},{y2} {x1},{y2}"
