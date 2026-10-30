"""Retry tilings read the whole page (audit SVC-04)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_backend = Path(__file__).resolve().parents[1]
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from app.services.seam_strategies import _make_grid, grid_shift  # type: ignore[import-untyped]  # noqa: E402


def _covered(intervals: list[tuple[int, int]], length: int) -> bool:
    """Do the half-open intervals cover [0, length)?"""
    reach = 0
    for start, end in sorted(intervals):
        if start > reach:
            return False
        reach = max(reach, end)
    return reach >= length


def _covers_page(boxes: list[tuple[int, int, int, int]], img_w: int, img_h: int) -> bool:
    # A grid is axis-aligned: it covers the page when its rows cover the height
    # and its columns the width.
    return _covered([(y1, y2) for _x1, y1, _x2, y2 in boxes], img_h) and _covered([(x1, x2) for x1, _y1, x2, _y2 in boxes], img_w)


@pytest.mark.parametrize(("img_w", "img_h"), [(2000, 3000), (2001, 3001), (1999, 2999), (757, 1123)])
@pytest.mark.parametrize(("rows", "cols"), [(1, 1), (2, 1), (3, 1), (2, 2), (4, 2)])
@pytest.mark.parametrize("offset", [0.0, 0.33])
def test_every_grid_reads_every_pixel_row_and_column(img_w: int, img_h: int, rows: int, cols: int, offset: float) -> None:
    boxes = _make_grid(rows, cols, img_w, img_h, 0.15, y_offset_frac=offset)
    assert _covers_page(boxes, img_w, img_h)
    assert min(y1 for _x1, y1, _x2, _y2 in boxes) == 0
    assert max(y2 for _x1, _y1, _x2, y2 in boxes) == img_h
    assert max(x2 for _x1, _y1, x2, _y2 in boxes) == img_w


def test_the_offset_grid_still_moves_the_seams() -> None:
    plain = _make_grid(3, 1, 2000, 3000, 0.15)
    shifted = _make_grid(3, 1, 2000, 3000, 0.15, y_offset_frac=0.33)
    assert [b[3] for b in plain[:-1]] != [b[3] for b in shifted[:-1]]


def test_each_grid_grid_shift_moves_to_covers_the_page() -> None:
    """Successive retries switch grids; each must still read the whole page."""
    prev: list[tuple[int, int, int, int]] = []
    for attempt in range(1, 8):
        plan = grid_shift(prev, 2001, 3001, attempt_idx=attempt)
        assert _covers_page(plan.tile_boxes, 2001, 3001), plan.grid
        prev = plan.tile_boxes
