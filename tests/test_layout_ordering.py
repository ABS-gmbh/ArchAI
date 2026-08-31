from __future__ import annotations

import pytest

from archai_ocr.pipeline.layout_yolo import (
    RegionDetection,
    _drop_degenerate,
    _horizontal_overlap_ratio,
    _iou,
    _nms,
    order_regions,
)


def region(x1: int, y1: int, x2: int, y2: int, name: str = "r", score: float = 0.9) -> RegionDetection:
    return RegionDetection(bbox=(x1, y1, x2, y2), score=score, class_id=0, class_name=name)


def names(regions: list[RegionDetection]) -> list[str]:
    return [r.class_name for r in regions]


# ------------------------------------------------------------- reading order --


@pytest.fixture
def two_column_page() -> list[RegionDetection]:
    """A two-column page, deliberately shuffled to prove ordering does the work."""
    return [
        region(500, 210, 800, 290, "R2"),
        region(100, 100, 400, 180, "L1"),
        region(500, 310, 800, 390, "R3"),
        region(100, 300, 400, 380, "L3"),
        region(500, 110, 800, 190, "R1"),
        region(100, 200, 400, 280, "L2"),
    ]


def test_column_order_reads_each_column_fully(two_column_page: list[RegionDetection]) -> None:
    assert names(order_regions(two_column_page, mode="column")) == [
        "L1",
        "L2",
        "L3",
        "R1",
        "R2",
        "R3",
    ]


def test_simple_order_interleaves_columns(two_column_page: list[RegionDetection]) -> None:
    """Documents the pre-0.2.0 behaviour that `simple` preserves for reproducibility."""
    assert names(order_regions(two_column_page, mode="simple")) == [
        "L1",
        "R1",
        "L2",
        "R2",
        "L3",
        "R3",
    ]


def test_single_column_order_is_top_to_bottom() -> None:
    page = [region(100, 300, 400, 380, "c"), region(100, 100, 400, 180, "a"), region(100, 200, 400, 280, "b")]
    assert names(order_regions(page, mode="column")) == ["a", "b", "c"]


def test_three_columns_order_left_to_right() -> None:
    page = [
        region(700, 100, 900, 200, "C1"),
        region(100, 100, 300, 200, "A1"),
        region(400, 100, 600, 200, "B1"),
    ]
    assert names(order_regions(page, mode="column")) == ["A1", "B1", "C1"]


def test_full_width_header_does_not_split_into_its_own_column() -> None:
    """A header spanning both columns overlaps each of them; it must not be lost."""
    page = [
        region(100, 50, 800, 90, "header"),
        region(100, 100, 400, 400, "left"),
        region(500, 100, 800, 400, "right"),
    ]
    ordered = names(order_regions(page, mode="column"))
    assert set(ordered) == {"header", "left", "right"}
    assert len(ordered) == 3


def test_ordering_is_stable_for_identical_boxes() -> None:
    page = [region(0, 0, 10, 10, "a"), region(0, 0, 10, 10, "b")]
    assert len(order_regions(page, mode="column")) == 2


def test_empty_input_returns_empty() -> None:
    assert order_regions([], mode="column") == []
    assert order_regions([], mode="simple") == []


def test_overlap_ratio_threshold_controls_column_splitting() -> None:
    # Two boxes overlapping by 50 px; the narrower is 100 px wide -> ratio 0.5
    page = [region(0, 0, 100, 50, "a"), region(50, 100, 150, 150, "b")]
    # A strict threshold keeps them apart as two columns.
    assert names(order_regions(page, mode="column", column_overlap_ratio=0.9)) == ["a", "b"]
    # A permissive threshold merges them into one column, read top to bottom.
    assert names(order_regions(page, mode="column", column_overlap_ratio=0.4)) == ["a", "b"]


# ------------------------------------------------------------------ geometry --


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((0, 100), (0, 100), 1.0),
        ((0, 100), (200, 300), 0.0),
        ((0, 100), (50, 150), 0.5),
        ((0, 100), (100, 200), 0.0),  # touching, not overlapping
    ],
)
def test_horizontal_overlap_ratio(a: tuple[int, int], b: tuple[int, int], expected: float) -> None:
    assert _horizontal_overlap_ratio(a, b) == pytest.approx(expected)


def test_iou_identical_boxes_is_one() -> None:
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero() -> None:
    assert _iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_iou_of_degenerate_box_is_zero_not_a_crash() -> None:
    assert _iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


def test_nms_keeps_highest_scoring_of_an_overlapping_pair() -> None:
    kept = _nms([region(0, 0, 100, 100, "low", 0.5), region(5, 5, 105, 105, "high", 0.95)], 0.5)
    assert names(kept) == ["high"]


def test_nms_keeps_both_when_they_barely_overlap() -> None:
    kept = _nms([region(0, 0, 100, 100, "a", 0.9), region(95, 95, 200, 200, "b", 0.8)], 0.5)
    assert len(kept) == 2


# ----------------------------------------------------------------- filtering --


def test_degenerate_regions_are_dropped() -> None:
    kept, dropped = _drop_degenerate(
        [region(0, 0, 100, 100, "ok"), region(10, 10, 10, 10, "zero"), region(0, 0, 200, 3, "sliver")],
        min_size=8,
    )
    assert names(kept) == ["ok"]
    assert dropped == 2


def test_nothing_dropped_when_all_regions_are_large_enough() -> None:
    kept, dropped = _drop_degenerate([region(0, 0, 100, 100, "ok")], min_size=8)
    assert dropped == 0 and len(kept) == 1
