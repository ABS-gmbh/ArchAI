from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from archai_ocr.config import AppConfig
from archai_ocr.utils.coco_writer import RegionDict, write_layout_coco
from archai_ocr.utils.image_io import get_image_size

# Ultralytics installs missing optional packages with pip at runtime (see
# ultralytics.utils.checks.check_requirements, gated on YOLO_AUTOINSTALL which
# defaults to true). Silently mutating the environment mid-run breaks
# reproducibility for a research pipeline and requires network access at
# inference time. setdefault leaves an explicit operator choice intact, and it
# must run before ultralytics is imported — hence the deferred import below.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

_ULTRALYTICS_INSTALL_HINT = (
    "ultralytics is required for layout detection but is not installed. "
    "Install it with:  pip install 'ultralytics>=8.2.0'  (or `pip install -e .` "
    "from the repository root)."
)

try:
    from ultralytics import YOLO  # noqa: E402 - must follow the env guard above

    ULTRALYTICS_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - exercised only without ultralytics
    YOLO = None
    ULTRALYTICS_AVAILABLE = False
    _ULTRALYTICS_IMPORT_ERROR: ImportError | None = exc
else:
    _ULTRALYTICS_IMPORT_ERROR = None


def require_ultralytics() -> None:
    """Raise a user-facing error if the optional ultralytics dependency is missing.

    Guarding the import keeps this module's pure logic — reading order, NMS,
    geometry — importable and testable without the inference stack installed.
    """
    if not ULTRALYTICS_AVAILABLE:
        raise RuntimeError(_ULTRALYTICS_INSTALL_HINT) from _ULTRALYTICS_IMPORT_ERROR


@dataclass(frozen=True)
class RegionDetection:
    bbox: tuple[int, int, int, int]
    score: float
    class_id: int
    class_name: str

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    def to_dict(self) -> RegionDict:
        return {
            "bbox": list(self.bbox),
            "score": self.score,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@lru_cache(maxsize=4)
def _load_model(weights_path: str, mtime_ns: int) -> Any:  # noqa: ARG001 - mtime busts the cache
    """Load a YOLO model once per (path, mtime).

    The detector was previously constructed on every call, which reloaded weights
    from disk for each page. mtime_ns is part of the key so that replacing the
    weights file invalidates the cache rather than serving a stale model.
    """
    require_ultralytics()
    return YOLO(weights_path)


def load_layout_model(weights: Path) -> Any:
    require_ultralytics()
    return _load_model(str(weights), weights.stat().st_mtime_ns)


def detect_layout_regions(
    image_path: Path,
    run_dir: Path,
    config: AppConfig,
    logger: logging.Logger,
) -> list[RegionDetection]:
    model = load_layout_model(config.weights.layout_yolo)
    results = model.predict(
        source=str(image_path),
        conf=config.layout.confidence_threshold,
        iou=config.layout.iou_threshold,
        max_det=config.layout.max_regions,
        verbose=False,
    )
    if not results:
        _write_empty_coco(image_path=image_path, run_dir=run_dir, config=config)
        return []

    result = results[0]
    names = result.names if hasattr(result, "names") else model.names
    regions: list[RegionDetection] = []
    seen_classes: set[str] = set()
    # Match case-insensitively: config files and model vocabularies disagree on
    # casing often enough that an exact-match filter silently yields nothing.
    wanted = {name.casefold() for name in config.layout.main_text_classes}

    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls.item())
            class_name = _class_name(names, class_id)
            seen_classes.add(class_name)
            score = float(box.conf.item())
            if class_name.casefold() not in wanted:
                continue
            if score < config.layout.confidence_threshold:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            regions.append(
                RegionDetection(
                    bbox=(x1, y1, x2, y2),
                    score=score,
                    class_id=class_id,
                    class_name=class_name,
                )
            )

    # A main_text_class that matches nothing the model can emit yields a silently
    # empty transcription. Surface it instead.
    if not regions and seen_classes and not (wanted & {c.casefold() for c in seen_classes}):
        raise ValueError(
            f"layout.main_text_class {list(config.layout.main_text_classes)} matched none of the "
            f"classes this model can emit: {sorted(seen_classes)}. "
            f"Set layout.main_text_class to one of those names (the shipped weights use "
            f"SegmOnto zones, where body text is 'MainZone')."
        )

    kept, dropped = _drop_degenerate(regions, config.layout.min_region_size)
    if dropped:
        logger.warning("layout.degenerate_regions_dropped", extra={"stage": "layout", "count": dropped})

    deduped = _nms(kept, config.layout.iou_threshold)
    ordered = order_regions(
        deduped,
        mode=config.layout.reading_order,
        column_overlap_ratio=config.layout.column_overlap_ratio,
    )
    ordered = ordered[: config.layout.max_regions]

    write_layout_coco(
        image_path=image_path,
        image_size=get_image_size(image_path),
        regions=[region.to_dict() for region in ordered],
        output_path=run_dir / "layout_coco.json",
        category_name=config.layout.main_text_classes[0],
    )
    logger.info("layout.regions_detected", extra={"stage": "layout", "count": len(ordered)})
    return ordered


# ------------------------------------------------------------- reading order --


def order_regions(
    regions: Sequence[RegionDetection],
    mode: str = "column",
    column_overlap_ratio: float = 0.5,
) -> list[RegionDetection]:
    """Sort detected regions into human reading order.

    "simple" is the historical behaviour: a global top-to-bottom, left-to-right
    sort. On a two-column manuscript page that interleaves the columns, so the
    assembled transcription alternates between them line by line.

    "column" groups regions into columns first (by horizontal overlap), orders
    the columns left to right, then reads each column top to bottom.
    """
    if mode == "simple":
        return sorted(regions, key=lambda region: (region.bbox[1], region.bbox[0]))

    columns = _group_into_columns(regions, column_overlap_ratio)
    ordered: list[RegionDetection] = []
    for column in columns:
        ordered.extend(sorted(column, key=lambda region: (region.bbox[1], region.bbox[0])))
    return ordered


def _group_into_columns(
    regions: Sequence[RegionDetection],
    overlap_ratio: float,
) -> list[list[RegionDetection]]:
    """Cluster regions into columns, then order the columns left to right."""
    if not regions:
        return []

    # Seed clusters in left-to-right order so the first region of each cluster
    # defines a stable x-span to compare against.
    by_x = sorted(regions, key=lambda region: (region.bbox[0], region.bbox[1]))
    columns: list[list[RegionDetection]] = []
    spans: list[tuple[int, int]] = []

    for region in by_x:
        x1, _, x2, _ = region.bbox
        placed = False
        for index, (span_x1, span_x2) in enumerate(spans):
            if _horizontal_overlap_ratio((x1, x2), (span_x1, span_x2)) >= overlap_ratio:
                columns[index].append(region)
                # Widen the column span to the union of its members.
                spans[index] = (min(span_x1, x1), max(span_x2, x2))
                placed = True
                break
        if not placed:
            columns.append([region])
            spans.append((x1, x2))

    order = sorted(range(len(columns)), key=lambda i: (spans[i][0], spans[i][1]))
    return [columns[i] for i in order]


def _horizontal_overlap_ratio(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Overlap of two x-spans as a fraction of the narrower span."""
    a_x1, a_x2 = a
    b_x1, b_x2 = b
    overlap = min(a_x2, b_x2) - max(a_x1, b_x1)
    if overlap <= 0:
        return 0.0
    narrower = min(a_x2 - a_x1, b_x2 - b_x1)
    if narrower <= 0:
        return 0.0
    return overlap / narrower


# ------------------------------------------------------------------ filtering --


def _drop_degenerate(
    regions: Sequence[RegionDetection],
    min_size: int,
) -> tuple[list[RegionDetection], int]:
    """Remove zero-area and sliver boxes that would crash or poison recognition."""
    kept = [r for r in regions if r.width >= min_size and r.height >= min_size]
    return kept, len(regions) - len(kept)


def _write_empty_coco(image_path: Path, run_dir: Path, config: AppConfig) -> None:
    write_layout_coco(
        image_path=image_path,
        image_size=get_image_size(image_path),
        regions=[],
        output_path=run_dir / "layout_coco.json",
        category_name=config.layout.main_text_classes[0],
    )


def _class_name(names: dict[int, str] | list[str] | Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, str(class_id)))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _nms(regions: Sequence[RegionDetection], iou_threshold: float) -> list[RegionDetection]:
    if not regions:
        return []

    sorted_regions = sorted(regions, key=lambda region: region.score, reverse=True)
    kept: list[RegionDetection] = []

    while sorted_regions:
        current = sorted_regions.pop(0)
        kept.append(current)
        sorted_regions = [
            candidate for candidate in sorted_regions if _iou(current.bbox, candidate.bbox) < iou_threshold
        ]

    return kept


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom
