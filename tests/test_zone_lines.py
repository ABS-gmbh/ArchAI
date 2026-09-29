from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from archai_ocr.pipeline.zone_lines import _moving_average, _otsu, line_bands


def zone(lines: list[tuple[int, int]], size: tuple[int, int] = (200, 120)) -> Image.Image:
    """A white crop with a dark 'text line' at each (top, bottom) row range."""
    image = Image.new("L", size, 235)
    draw = ImageDraw.Draw(image)
    for top, bottom in lines:
        # Broken strokes, like letters, rather than a solid bar.
        for x in range(10, size[0] - 10, 9):
            draw.rectangle((x, top, x + 5, bottom), fill=30)
    return image


def test_two_lines_separated_by_white_space_are_two_bands() -> None:
    bands = line_bands(zone([(20, 45), (70, 95)]))
    assert len(bands) == 2
    (top_1, bottom_1), (top_2, bottom_2) = bands
    assert top_1 <= 20 and bottom_1 <= top_2 and bottom_2 >= 95


def test_bands_partition_the_zone_so_lines_keep_their_ascenders() -> None:
    bands = line_bands(zone([(20, 45), (70, 95)]))
    assert bands[0][0] == 0 and bands[-1][1] == 120
    assert all(left[1] == right[0] for left, right in zip(bands, bands[1:], strict=False))


def test_a_single_line_is_never_cut_apart() -> None:
    assert line_bands(zone([(40, 80)])) == [(0, 120)]


def test_a_shallow_dip_inside_a_line_is_not_a_line_break() -> None:
    """Rows between an x-height band and its ascenders thin out but never empty."""
    image = zone([(30, 90)])
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 55, 200, 60), fill=235)  # a thin gap: 6 of 60 rows
    for x in range(10, 190, 18):  # but strokes still cross it
        draw.rectangle((x, 50, x + 5, 65), fill=30)
    assert len(line_bands(image)) == 1


def test_three_lines() -> None:
    assert len(line_bands(zone([(10, 30), (50, 70), (90, 110)]))) == 3


def test_a_blank_zone_is_one_band() -> None:
    assert line_bands(Image.new("L", (50, 40), 255)) == [(0, 40)]


def test_an_empty_image_has_no_bands() -> None:
    assert line_bands(Image.new("L", (0, 0))) == []


def test_otsu_separates_ink_from_parchment() -> None:
    histogram = [0] * 256
    histogram[30] = 100
    histogram[220] = 900
    assert 30 < _otsu(histogram) <= 220


def test_moving_average_matches_a_centred_zero_padded_box_filter() -> None:
    assert _moving_average([0, 3, 0, 0], 3) == pytest.approx([1.0, 1.0, 1.0, 0.0])


def test_zone_segmentation_is_a_kraken_baseline_segmentation() -> None:
    pytest.importorskip("kraken.containers")
    from archai_ocr.pipeline.htr_kraken import _zone_segmentation

    segmentation = _zone_segmentation(zone([(20, 45), (70, 95)]))
    assert segmentation.type == "baselines"
    assert len(segmentation.lines) == 2
    first = segmentation.lines[0]
    assert first.baseline[0][0] == 0 and first.baseline[-1][0] == 199
    assert first.boundary[0][1] == 0  # the first band starts at the top of the zone
