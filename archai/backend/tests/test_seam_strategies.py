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


# ── seams are only where neighbouring tiles meet (audit SVC-03) ─────────

from app.services.seam_strategies import _seam_x_coords, _seam_y_coords, seam_band_crop  # type: ignore[import-untyped]  # noqa: E402


@pytest.mark.parametrize(
    ("rows", "cols", "ys", "xs"),
    [
        (2, 2, [1500], [1000]),
        (3, 1, [1000, 2000], []),
        (1, 3, [], [666, 1332]),
        (3, 2, [1000, 2000], [1000]),
        (1, 1, [], []),
    ],
)
def test_a_grid_has_only_its_own_seams(rows: int, cols: int, ys: list[int], xs: list[int]) -> None:
    grid = _make_grid(rows, cols, 2000, 3000, 0.15)
    assert _seam_y_coords(grid) == ys
    assert _seam_x_coords(grid) == xs


def test_two_text_columns_meet_at_one_vertical_seam() -> None:
    """Layout regions of uneven height side by side: one seam between them, none across."""
    left, right = (100, 200, 950, 2700), (1000, 260, 1900, 2500)
    assert _seam_x_coords([left, right]) == [975]
    assert _seam_y_coords([left, right]) == []


def test_seam_band_crop_cuts_only_at_the_real_seams() -> None:
    plan = seam_band_crop(_make_grid(2, 2, 2000, 3000, 0.15), 2000, 3000)
    assert plan.preproc["seam_ys"] == [1500]
    assert plan.preproc["seam_xs"] == [1000]
