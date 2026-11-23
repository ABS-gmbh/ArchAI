"""draw_coco_on_image closes its figure even when drawing or saving fails.

pyplot keeps every open figure in a process-wide registry. The figure was
closed only after a successful save, so each failure left one in memory for
the life of the server.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from PIL import Image

from app.services.visualization import draw_coco_on_image

COCO = {
    "images": [{"id": 1, "file_name": "p.png", "width": 60, "height": 40}],
    "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 5, 20, 10], "segmentation": []}],
    "categories": [{"id": 1, "name": "MainZone"}],
}


@pytest.fixture
def page(tmp_path: Path) -> Path:
    path = tmp_path / "p.png"
    Image.new("RGB", (60, 40), "white").save(path)
    return path


def test_a_successful_drawing_leaves_no_figure_open(page: Path, tmp_path: Path) -> None:
    before = set(plt.get_fignums())
    out = draw_coco_on_image(str(page), COCO, ["MainZone"], str(tmp_path / "out.png"))
    assert Path(out).exists()
    assert set(plt.get_fignums()) == before


def test_a_failed_save_leaves_no_figure_open(page: Path, tmp_path: Path) -> None:
    before = set(plt.get_fignums())
    with pytest.raises(OSError):
        draw_coco_on_image(str(page), COCO, ["MainZone"], str(tmp_path / "missing" / "dir" / "out.png"))
    assert set(plt.get_fignums()) == before
