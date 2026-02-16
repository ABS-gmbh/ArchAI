from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import logging

from ultralytics import YOLO

from archai_ocr.config import AppConfig
from archai_ocr.utils.coco_writer import write_layout_coco
from archai_ocr.utils.image_io import get_image_size


@dataclass(frozen=True)
class RegionDetection:
    bbox: tuple[int, int, int, int]
    score: float
    class_id: int
    class_name: str

    def to_dict(self) -> dict[str, object]:
        return {
            "bbox": list(self.bbox),
            "score": self.score,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


def detect_layout_regions(
    image_path: Path,
    run_dir: Path,
    config: AppConfig,
    logger: logging.Logger,
) -> list[RegionDetection]:
    model = YOLO(str(config.weights.layout_yolo))
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

    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls.item())
            class_name = _class_name(names, class_id)
            score = float(box.conf.item())
            if class_name != config.layout.main_text_class:
                continue
            if score < config.layout.confidence_threshold:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            regions.append(
                RegionDetection(
                    bbox=(x1, y1, x2, y2),
                    score=score,
                    class_id=class_id,
                    class_name=class_name,
                )
            )

    deduped = _nms(regions, config.layout.iou_threshold)
    ordered = sorted(deduped, key=lambda region: (region.bbox[1], region.bbox[0]))
    ordered = ordered[: config.layout.max_regions]

    write_layout_coco(
        image_path=image_path,
        image_size=get_image_size(image_path),
        regions=[region.to_dict() for region in ordered],
        output_path=run_dir / "layout_coco.json",
        category_name=config.layout.main_text_class,
    )
    logger.info("layout.regions_detected", extra={"count": len(ordered)})
    return ordered


def _write_empty_coco(image_path: Path, run_dir: Path, config: AppConfig) -> None:
    write_layout_coco(
        image_path=image_path,
        image_size=get_image_size(image_path),
        regions=[],
        output_path=run_dir / "layout_coco.json",
        category_name=config.layout.main_text_class,
    )


def _class_name(names: dict[int, str] | list[str], class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, str(class_id)))
    if 0 <= class_id < len(names):
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
            candidate
            for candidate in sorted_regions
            if _iou(current.bbox, candidate.bbox) < iou_threshold
        ]

    return kept


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom
