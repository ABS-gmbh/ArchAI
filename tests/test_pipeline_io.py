from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest
from PIL import Image

from archai_ocr.pipeline.assemble_text import assemble_text
from archai_ocr.pipeline.crop_regions import crop_regions
from archai_ocr.pipeline.layout_yolo import RegionDetection
from archai_ocr.utils.coco_writer import write_layout_coco
from archai_ocr.utils.image_io import get_image_size, validate_image_path


def region(x1: int, y1: int, x2: int, y2: int) -> RegionDetection:
    return RegionDetection(bbox=(x1, y1, x2, y2), score=0.9, class_id=0, class_name="main_text")


# ------------------------------------------------------------------ cropping --


def test_crop_writes_one_png_per_region(page_image: Path, tmp_path: Path) -> None:
    paths = crop_regions(page_image, [region(10, 10, 200, 100), region(10, 200, 200, 300)], tmp_path / "c")
    assert [p.name for p in paths] == ["region_000.png", "region_001.png"]
    assert all(p.exists() for p in paths)


def test_crop_clamps_to_page_bounds(page_image: Path, tmp_path: Path) -> None:
    """Padding must never push the box outside the image."""
    paths = crop_regions(page_image, [region(0, 0, 1000, 1400)], tmp_path / "c", padding=50)
    with Image.open(paths[0]) as im:
        assert im.size == (1000, 1400)


def test_crop_handles_inverted_bbox(page_image: Path, tmp_path: Path) -> None:
    """x2 < x1 must not produce a negative-size crop."""
    paths = crop_regions(page_image, [region(300, 200, 100, 50)], tmp_path / "c", padding=0)
    with Image.open(paths[0]) as im:
        assert im.size == (200, 150)


def test_crop_skips_degenerate_regions(page_image: Path, tmp_path: Path) -> None:
    paths = crop_regions(
        page_image,
        [region(10, 10, 200, 100), region(500, 500, 502, 502)],
        tmp_path / "c",
        padding=0,
        min_size=8,
    )
    assert len(paths) == 1


def test_crop_indices_track_the_region_list_not_the_output(page_image: Path, tmp_path: Path) -> None:
    """A skipped region must not renumber the ones after it."""
    paths = crop_regions(
        page_image,
        [region(500, 500, 501, 501), region(10, 10, 200, 100)],
        tmp_path / "c",
        padding=0,
        min_size=8,
    )
    assert [p.name for p in paths] == ["region_001.png"]


def test_crop_of_empty_region_list_creates_dir_and_returns_nothing(page_image: Path, tmp_path: Path) -> None:
    out = tmp_path / "c"
    assert crop_regions(page_image, [], out) == []
    assert out.is_dir()


# ------------------------------------------------------------------ assembly --


def test_assemble_joins_regions_with_a_blank_line(tmp_path: Path) -> None:
    out = assemble_text(["alpha", "beta"], tmp_path / "o.txt")
    assert out.read_text(encoding="utf-8") == "alpha\n\nbeta\n"


def test_assemble_drops_empty_and_whitespace_regions(tmp_path: Path) -> None:
    out = assemble_text(["alpha", "", "   ", "\n", "beta"], tmp_path / "o.txt")
    assert out.read_text(encoding="utf-8") == "alpha\n\nbeta\n"


def test_assemble_normalizes_to_nfc(tmp_path: Path) -> None:
    decomposed = "Rege" + "́" + "s"  # e + combining acute
    out = assemble_text([decomposed], tmp_path / "o.txt")
    text = out.read_text(encoding="utf-8")
    assert text == unicodedata.normalize("NFC", decomposed) + "\n"
    assert "́" not in text


def test_assemble_can_skip_normalization(tmp_path: Path) -> None:
    decomposed = "Regés"
    out = assemble_text([decomposed], tmp_path / "o.txt", normalize=None)
    assert "́" in out.read_text(encoding="utf-8")


def test_assemble_empty_input_writes_empty_file(tmp_path: Path) -> None:
    out = assemble_text([], tmp_path / "o.txt")
    assert out.read_text(encoding="utf-8") == ""


def test_assemble_creates_missing_parent_directories(tmp_path: Path) -> None:
    out = assemble_text(["x"], tmp_path / "deep" / "nested" / "o.txt")
    assert out.exists()


# ---------------------------------------------------------------------- coco --


def test_coco_bbox_is_xywh_not_xyxy(tmp_path: Path, page_image: Path) -> None:
    out = write_layout_coco(
        image_path=page_image,
        image_size=(1000, 1400),
        regions=[region(10, 20, 110, 220).to_dict()],
        output_path=tmp_path / "c.json",
        category_name="main_text",
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    annotation = payload["annotations"][0]
    assert annotation["bbox"] == [10, 20, 100, 200]
    assert annotation["area"] == pytest.approx(100 * 200)


def test_coco_has_required_top_level_keys(tmp_path: Path, page_image: Path) -> None:
    out = write_layout_coco(page_image, (1000, 1400), [], tmp_path / "c.json", "main_text")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert {"images", "annotations", "categories"} <= set(payload)
    assert payload["images"][0]["file_name"] == page_image.name


def test_coco_annotation_ids_start_at_one_and_increment(tmp_path: Path, page_image: Path) -> None:
    regions = [region(0, 0, 10, 10).to_dict(), region(20, 20, 30, 30).to_dict()]
    out = write_layout_coco(page_image, (100, 100), regions, tmp_path / "c.json", "main_text")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [a["id"] for a in payload["annotations"]] == [1, 2]


# ------------------------------------------------------------------ image io --


def test_validate_image_path_rejects_unsupported_extension(tmp_path: Path) -> None:
    bad = tmp_path / "notes.txt"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported image extension"):
        validate_image_path(bad)


def test_validate_image_path_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_image_path(tmp_path / "gone.png")


def test_validate_image_path_rejects_a_corrupt_image(tmp_path: Path) -> None:
    corrupt = tmp_path / "broken.png"
    corrupt.write_bytes(b"not a png")
    with pytest.raises(ValueError, match="Not a readable image file"):
        validate_image_path(corrupt)


def test_get_image_size(page_image: Path) -> None:
    assert get_image_size(page_image) == (1000, 1400)


def test_ultralytics_runtime_autoinstall_is_disabled() -> None:
    """Importing the layout module must not leave ultralytics free to pip install."""
    import os

    import archai_ocr.pipeline.layout_yolo  # noqa: F401

    assert os.environ["YOLO_AUTOINSTALL"] == "false"
