from __future__ import annotations

from pathlib import Path
from typing import Iterable

import json


def write_layout_coco(
    image_path: Path,
    image_size: tuple[int, int],
    regions: Iterable[dict[str, object]],
    output_path: Path,
    category_name: str,
) -> Path:
    width, height = image_size
    annotations = []
    for idx, region in enumerate(regions, start=1):
        x1, y1, x2, y2 = region["bbox"]  # type: ignore[index]
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
