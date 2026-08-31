from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TypedDict


class RegionDict(TypedDict):
    """Serialized form of a RegionDetection, as produced by RegionDetection.to_dict()."""

    bbox: list[int]
    score: float
    class_id: int
    class_name: str


def write_layout_coco(
    image_path: Path,
    image_size: tuple[int, int],
    regions: Iterable[RegionDict],
    output_path: Path,
    category_name: str,
) -> Path:
    """Write layout detections as a single-image COCO detection file.

    Input boxes are xyxy (as produced by the detector); COCO stores xywh.
    """
    width, height = image_size
    annotations: list[dict[str, Any]] = []

    for idx, region in enumerate(regions, start=1):
        x1, y1, x2, y2 = _as_xyxy(region["bbox"])
        bbox_w = max(0, x2 - x1)
        bbox_h = max(0, y2 - y1)
        annotations.append(
            {
                "id": idx,
                "image_id": 1,
                "category_id": 1,
                "bbox": [x1, y1, bbox_w, bbox_h],
                "area": float(bbox_w * bbox_h),
                "iscrowd": 0,
                # Boxes are axis-aligned, so the polygon is the rectangle itself.
                # Several COCO consumers require this key to be present.
                "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]],
                "score": float(region["score"]),
            }
        )

    payload = {
        "images": [
            {
                "id": 1,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        ],
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": category_name,
                "supercategory": "text",
            }
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _as_xyxy(bbox: Sequence[int]) -> tuple[int, int, int, int]:
    if len(bbox) < 4:
        raise ValueError(f"Region bbox must have 4 values (x1, y1, x2, y2), got {list(bbox)!r}.")
    x1, y1, x2, y2 = (int(v) for v in bbox[:4])
    return x1, y1, x2, y2
