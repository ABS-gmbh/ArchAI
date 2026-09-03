"""Tests for layout geometry that crosses the segmentation/crop boundary.

Segmentation runs on a size-bounded copy of the page while every crop is taken
from the original upload. Coordinates were returned in the downscaled space with
no inverse scale, so on any page large enough to trigger the resize every region
crop was cut from the wrong pixels while still looking like a plausible box.
"""

from __future__ import annotations

import pytest

from app.core.constants import (
    ZONE_CLASSES_WITHOUT_COCO_MAPPING,
    catmus_zones_mapping,
    coco_class_mapping,
)
from app.core.image_batch_classes import UnmappedAnnotationClass, _mappable_annotations
from app.routers.predict import rescale_coco_to_original

# Every class the shipped zone detector can emit.
ZONE_MODEL_CLASSES = [
    "DigitizationArtefactZone", "DropCapitalZone", "GraphicZone", "MainZone",
    "MarginTextZone", "MusicZone", "NumberingZone", "QuireMarksZone",
    "RunningTitleZone", "StampZone", "TitlePageZone",
]


def coco(bbox: list[float], *, width: int, height: int) -> dict:
    return {
        "images": [{"id": 1, "width": width, "height": height}],
        "annotations": [
            {
                "id": 1,
                "bbox": list(bbox),
                "area": bbox[2] * bbox[3],
                "segmentation": [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]],
            }
        ],
    }


# ──────────────────────────────────────────────── coordinate rescaling ──


def test_bbox_is_restored_to_original_scale() -> None:
    scale = 0.5
    out = rescale_coco_to_original(coco([100.0, 200.0, 50.0, 30.0], width=500, height=600), scale)
    assert out["annotations"][0]["bbox"] == [200.0, 400.0, 100.0, 60.0]


def test_image_dimensions_are_restored() -> None:
    out = rescale_coco_to_original(coco([0.0, 0.0, 1.0, 1.0], width=500, height=600), 0.5)
    assert out["images"][0]["width"] == 1000
    assert out["images"][0]["height"] == 1200


def test_area_scales_by_the_square_of_the_factor() -> None:
    """Area is two-dimensional; scaling it linearly would understate it."""
    out = rescale_coco_to_original(coco([0.0, 0.0, 10.0, 10.0], width=100, height=100), 0.5)
    assert out["annotations"][0]["area"] == pytest.approx(400.0)


def test_segmentation_polygons_are_rescaled() -> None:
    out = rescale_coco_to_original(coco([10.0, 20.0, 5.0, 5.0], width=100, height=100), 0.5)
    assert out["annotations"][0]["segmentation"][0] == [20.0, 40.0, 30.0, 50.0]


def test_unscaled_page_is_returned_untouched() -> None:
    """scale == 1.0 must be a true no-op so normal pages are unaffected."""
    payload = coco([1.0, 2.0, 3.0, 4.0], width=100, height=100)
    assert rescale_coco_to_original(payload, 1.0) is payload


@pytest.mark.parametrize("bad_scale", [0.0, -1.0])
def test_nonsensical_scale_is_ignored_rather_than_dividing_by_zero(bad_scale: float) -> None:
    payload = coco([1.0, 2.0, 3.0, 4.0], width=100, height=100)
    assert rescale_coco_to_original(payload, bad_scale) is payload


def test_empty_or_missing_sections_do_not_raise() -> None:
    assert rescale_coco_to_original({}, 0.5) == {}
    assert rescale_coco_to_original({"annotations": []}, 0.5) == {"annotations": []}


def test_input_is_not_mutated() -> None:
    payload = coco([10.0, 10.0, 10.0, 10.0], width=100, height=100)
    rescale_coco_to_original(payload, 0.5)
    assert payload["annotations"][0]["bbox"] == [10.0, 10.0, 10.0, 10.0]


def test_realistic_oversized_page_offset() -> None:
    """A 9000x12000 page exceeds the 85 MP cap, so it is resized before segmenting."""
    scale = (85_000_000 / (9000 * 12000)) ** 0.5
    out = rescale_coco_to_original(
        coco([1000.0, 2000.0, 500.0, 300.0], width=round(9000 * scale), height=round(12000 * scale)),
        scale,
    )
    assert out["annotations"][0]["bbox"][0] == pytest.approx(1000.0 / scale)
    assert out["images"][0]["width"] == 9000


# ───────────────────────────────────────────── unmapped zone classes ──


class FakeAnnotation:
    def __init__(self, name: str) -> None:
        self.name = name

    def coco_class_name(self) -> str:
        return catmus_zones_mapping.get(self.name, self.name)

    def has_coco_mapping(self) -> bool:
        return self.coco_class_name() in coco_class_mapping


def test_every_zone_model_class_is_accounted_for() -> None:
    """An unhandled class raises KeyError and discards the whole page's layout."""
    unhandled = [
        name
        for name in ZONE_MODEL_CLASSES
        if catmus_zones_mapping.get(name, name) not in coco_class_mapping
        and name not in ZONE_CLASSES_WITHOUT_COCO_MAPPING
    ]
    assert not unhandled, f"zone classes with neither a mapping nor an exemption: {unhandled}"


def test_unmappable_annotation_is_dropped_and_the_rest_survive() -> None:
    """Regression: DigitizationArtefactZone previously took every region with it."""
    annotations = [
        FakeAnnotation("MainZone"),
        FakeAnnotation("DigitizationArtefactZone"),
        FakeAnnotation("MarginTextZone"),
        FakeAnnotation("DropCapitalZone"),
    ]
    kept = _mappable_annotations(annotations)
    assert [a.name for a in kept] == ["MainZone", "MarginTextZone", "DropCapitalZone"]


def test_all_unmappable_yields_empty_not_an_exception() -> None:
    assert _mappable_annotations([FakeAnnotation("DigitizationArtefactZone")]) == []


def test_to_coco_format_still_signals_a_programming_error() -> None:
    """Skipping is the caller's job; calling it directly on an unmapped class must be loud."""
    from app.core.image_batch_classes import Annotation

    annotation = Annotation.__new__(Annotation)
    annotation.name = "DigitizationArtefactZone"
    with pytest.raises(UnmappedAnnotationClass, match="no COCO category"):
        annotation.to_coco_format(1)
