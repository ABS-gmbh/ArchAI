from __future__ import annotations

import logging
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from archai_ocr.config import LayoutConfig, load_config
from archai_ocr.pipeline import layout_yolo
from archai_ocr.pipeline.layout_yolo import (
    RegionDetection,
    _drop_degenerate,
    _nms,
    detect_page_layout,
    order_regions,
    split_lanes,
)
from archai_ocr.pipeline.zones import (
    DEFAULT_SECONDARY_TEXT_CLASSES,
    LaneAssignment,
    zone_kind,
)

LAYOUT = LayoutConfig(
    main_text_classes=("MainZone",),
    confidence_threshold=0.25,
    iou_threshold=0.5,
    max_regions=50,
    crop_padding=5,
    reading_order="column",
    column_overlap_ratio=0.5,
    min_region_size=8,
)
LANES = LaneAssignment.from_names(LAYOUT.main_text_classes, LAYOUT.secondary_text_classes)


def det(name: str, bbox: tuple[int, int, int, int], score: float = 0.9) -> RegionDetection:
    return RegionDetection(bbox=bbox, score=score, class_id=0, class_name=name)


# ---------------------------------------------------------------------- zones --


def test_lane_assignment_is_case_insensitive() -> None:
    assert LANES.lane("mainzone") == "main"
    assert LANES.lane("MARGINTEXTZONE") == "secondary"
    assert LANES.lane("StampZone") == "layout"


def test_a_class_configured_as_main_is_never_also_secondary() -> None:
    lanes = LaneAssignment.from_names(["MainZone", "MarginTextZone"], DEFAULT_SECONDARY_TEXT_CLASSES)
    assert lanes.lane("MarginTextZone") == "main"


def test_zone_kinds_map_to_page_elements() -> None:
    assert zone_kind("MainZone").element == "TextRegion"
    assert zone_kind("StampZone").element == "GraphicRegion"
    assert zone_kind("MusicZone").element == "MusicRegion"
    assert zone_kind("DigitizationArtefactZone").element == "NoiseRegion"
    assert zone_kind("MarginTextZone").label == "Marginalia"


def test_unknown_classes_are_kept_as_text_regions() -> None:
    kind = zone_kind("main_text")
    assert (kind.element, kind.label) == ("TextRegion", "main_text")


# ---------------------------------------------------------------------- lanes --


def test_detections_are_split_into_lanes() -> None:
    layout = split_lanes(
        [
            det("MainZone", (100, 100, 800, 1200)),
            det("MarginTextZone", (850, 300, 980, 400), 0.6),
            det("NumberingZone", (900, 20, 960, 60), 0.3),
            det("StampZone", (50, 1250, 200, 1350)),
        ],
        LANES,
        LAYOUT,
    )
    assert [d.class_name for d in layout.main] == ["MainZone"]
    assert [d.class_name for d in layout.secondary] == ["NumberingZone", "MarginTextZone"]
    assert [d.class_name for d in layout.layout] == ["StampZone"]


def test_each_lane_applies_its_own_threshold() -> None:
    layout = split_lanes(
        [
            det("MainZone", (100, 100, 800, 1200), 0.2),  # below the 0.25 main threshold
            det("MainZone", (100, 1300, 800, 1400), 0.9),
            det("MarginTextZone", (850, 300, 980, 400), 0.15),  # above the 0.10 secondary one
            det("MarginTextZone", (850, 500, 980, 600), 0.05),
        ],
        LANES,
        LAYOUT,
    )
    assert [d.score for d in layout.main] == [0.9]
    assert [d.score for d in layout.secondary] == [0.15]


def test_one_strip_reported_as_two_secondary_classes_is_kept_once() -> None:
    layout = split_lanes(
        [
            det("MarginTextZone", (10, 1500, 1000, 1560), 0.18),
            det("RunningTitleZone", (12, 1502, 998, 1558), 0.12),
        ],
        LANES,
        LAYOUT,
    )
    assert [d.class_name for d in layout.secondary] == ["MarginTextZone"]


def test_secondary_text_inside_a_main_zone_is_left_to_that_zone() -> None:
    """An interlinear gloss is already recognised as lines of its main zone."""
    layout = split_lanes(
        [
            det("MainZone", (100, 100, 800, 1200)),
            det("MarginTextZone", (300, 400, 500, 440), 0.7),  # inside the body
            det("MarginTextZone", (780, 400, 980, 440), 0.7),  # mostly in the margin
        ],
        LANES,
        LAYOUT,
    )
    assert [d.bbox for d in layout.secondary] == [(780, 400, 980, 440)]


def test_the_secondary_lane_can_be_disabled() -> None:
    lanes = LaneAssignment.from_names(["MainZone"], [])
    layout = split_lanes([det("MarginTextZone", (850, 300, 980, 400), 0.9)], lanes, LAYOUT)
    assert layout.secondary == ()
    assert [d.class_name for d in layout.layout] == ["MarginTextZone"]


def test_degenerate_secondary_zones_are_dropped() -> None:
    layout = split_lanes([det("NumberingZone", (10, 10, 13, 40))], LANES, LAYOUT)
    assert layout.secondary == ()


def _legacy_main(detections: list[RegionDetection], layout: LayoutConfig) -> list[RegionDetection]:
    """The main-text path as it was before lanes existed."""
    wanted = {name.casefold() for name in layout.main_text_classes}
    kept = [
        d for d in detections if d.class_name.casefold() in wanted and d.score >= layout.confidence_threshold
    ]
    kept, _ = _drop_degenerate(kept, layout.min_region_size)
    ordered = order_regions(
        _nms(kept, layout.iou_threshold),
        mode=layout.reading_order,
        column_overlap_ratio=layout.column_overlap_ratio,
    )
    return ordered[: layout.max_regions]


@pytest.mark.parametrize("reading_order", ["column", "simple"])
def test_the_main_lane_is_exactly_the_legacy_main_text(reading_order: str) -> None:
    rng = random.Random(7)
    classes = ["MainZone", "MarginTextZone", "NumberingZone", "StampZone", "RunningTitleZone"]
    layout = replace(LAYOUT, reading_order=reading_order)
    for _ in range(300):
        detections = []
        for _ in range(rng.randrange(0, 12)):
            x1, y1 = rng.randrange(0, 900), rng.randrange(0, 1300)
            box = (x1, y1, x1 + rng.randrange(1, 400), y1 + rng.randrange(1, 400))
            detections.append(det(rng.choice(classes), box, round(rng.random(), 3)))
        assert list(split_lanes(detections, LANES, layout).main) == _legacy_main(detections, layout)


# ------------------------------------------------------------ model boundary --


class _Scalar:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class _Box:
    def __init__(self, class_id: int, score: float, xyxy: tuple[int, int, int, int]) -> None:
        self.cls = _Scalar(class_id)
        self.conf = _Scalar(score)
        self.xyxy = [_Tensor(xyxy)]


class _Tensor:
    def __init__(self, values: tuple[int, ...]) -> None:
        self._values = values

    def tolist(self) -> list[int]:
        return list(self._values)


class _FakeModel:
    names = {0: "MainZone", 1: "MarginTextZone", 2: "StampZone"}

    def __init__(self, boxes: list[_Box]) -> None:
        self._boxes = boxes
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        result = type("Result", (), {"names": self.names, "boxes": self._boxes})()
        return [result]


@pytest.fixture
def zone_config(tmp_path: Path, config_file: Path) -> Any:
    return load_config(config_file, overrides={"layout.main_text_class": "MainZone"})


def test_the_model_is_queried_once_at_the_lowest_lane_threshold(
    zone_config: Any, page_image: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _FakeModel([_Box(0, 0.95, (100, 100, 800, 1200)), _Box(1, 0.12, (850, 300, 980, 400))])
    monkeypatch.setattr(layout_yolo, "load_layout_model", lambda _weights: model)
    layout = detect_page_layout(page_image, tmp_path, zone_config, logging.getLogger("test"))
    assert [call["conf"] for call in model.calls] == [pytest.approx(0.10)]
    assert (len(layout.main), len(layout.secondary)) == (1, 1)


def test_a_page_without_body_text_is_not_a_configuration_error(
    zone_config: Any, page_image: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a class the model cannot emit at all is a misconfiguration."""
    model = _FakeModel([_Box(2, 0.9, (50, 50, 200, 200))])
    monkeypatch.setattr(layout_yolo, "load_layout_model", lambda _weights: model)
    layout = detect_page_layout(page_image, tmp_path, zone_config, logging.getLogger("test"))
    assert layout.main == ()
    assert [d.class_name for d in layout.layout] == ["StampZone"]


def test_a_main_class_outside_the_model_vocabulary_is_reported(
    config_file: Path, page_image: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(config_file)  # main_text_class "main_text": not in the vocabulary
    model = _FakeModel([_Box(0, 0.9, (50, 50, 200, 200))])
    monkeypatch.setattr(layout_yolo, "load_layout_model", lambda _weights: model)
    with pytest.raises(ValueError, match="matched none of the classes"):
        detect_page_layout(page_image, tmp_path, config, logging.getLogger("test"))


def test_the_coco_file_records_every_zone(
    zone_config: Any, page_image: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    model = _FakeModel(
        [
            _Box(0, 0.95, (100, 100, 800, 1200)),
            _Box(1, 0.6, (850, 300, 980, 400)),
            _Box(2, 0.9, (50, 1250, 200, 1350)),
        ]
    )
    monkeypatch.setattr(layout_yolo, "load_layout_model", lambda _weights: model)
    detect_page_layout(page_image, tmp_path, zone_config, logging.getLogger("test"))
    payload = json.loads((tmp_path / "layout_coco.json").read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in payload["categories"]}
    assert sorted(names[a["category_id"]] for a in payload["annotations"]) == [
        "MainZone",
        "MarginTextZone",
        "StampZone",
    ]
