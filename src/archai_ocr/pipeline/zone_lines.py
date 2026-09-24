"""Finding the text lines inside a small zone, where Kraken's segmenter finds none.

Kraken's baseline segmenter is trained on whole pages. Run on the crop of a folio
number or a marginal note - often under 100 px high - it found a line in only 3
of the 20 secondary zones on ten real pages, and none of those lines produced
any text. Main-text zones are large enough for it and keep using it.

A small zone is read instead as horizontal bands of ink, separated at the
valleys of its row-wise ink profile. It is cut only where the profile falls
almost to nothing between two lines - below :data:`SPLIT_RATIO` of the peaks on
either side. On those ten pages that cut apart the only two-line reference
cleanly separated by white space ("apoc. / 21.9") and left every single-line
zone whole; a looser 0.25 already cut a single-line numbering zone in two.
Tightly written multi-line notes are not separated and are read as one line.

Pure PIL, so it runs where Kraken is not installed.
"""

from __future__ import annotations

from PIL import Image

# A valley splits two lines only when it is at most this share of the smaller
# neighbouring peak.
SPLIT_RATIO = 0.15
# Peaks below this share of the tallest are noise: a stray stroke, not a line.
_MIN_PEAK = 0.25
_MIN_BAND_PX = 4


def line_bands(image: Image.Image, *, split_ratio: float = SPLIT_RATIO) -> list[tuple[int, int]]:
    """Top and bottom row of each text line in *image*, as half-open ranges.

    Bands partition the image from top to bottom, so each line keeps the room
    its ascenders and descenders need. An image with no separable lines is one
    band covering all of it.
    """
    gray = image.convert("L")
    width, height = gray.size
    if width == 0 or height == 0:
        return []
    threshold = _otsu(gray.histogram())
    ink = gray.point(lambda value: 255 if value < threshold else 0)
    # A box-filtered resize to one pixel wide is the mean ink of each row; an
    # 8-bit image of width 1 serialises to exactly one byte per row.
    profile = [float(value) for value in ink.resize((1, height), Image.Resampling.BOX).tobytes()]
    smooth = _moving_average(profile, max(3, height // 30))
    tallest = max(smooth)
    if tallest <= 0:
        return [(0, height)]

    peaks = [
        row
        for row in range(1, height - 1)
        if smooth[row] >= smooth[row - 1]
        and smooth[row] > smooth[row + 1]
        and smooth[row] >= _MIN_PEAK * tallest
    ]
    cuts: list[int] = []
    previous: int | None = None
    for peak in peaks:
        if previous is None:
            previous = peak
            continue
        valley = min(range(previous, peak + 1), key=smooth.__getitem__)
        if smooth[valley] <= split_ratio * min(smooth[previous], smooth[peak]):
            cuts.append(valley)
            previous = peak
        elif smooth[peak] > smooth[previous]:
            previous = peak

    edges = [0, *cuts, height]
    bands = [
        (edges[i], edges[i + 1]) for i in range(len(edges) - 1) if edges[i + 1] - edges[i] >= _MIN_BAND_PX
    ]
    return bands or [(0, height)]


def _otsu(histogram: list[int]) -> int:
    """Otsu's threshold over a 256-bin grey-level histogram."""
    counts = histogram[:256]
    total = sum(counts)
    if total == 0:
        return 128
    weighted_total = sum(level * count for level, count in enumerate(counts))
    background = background_weighted = 0.0
    best_level, best_variance = 128, -1.0
    for level in range(1, 255):
        background += counts[level - 1]
        background_weighted += (level - 1) * counts[level - 1]
        foreground = total - background
        if background == 0 or foreground == 0:
            continue
        mean_background = background_weighted / background
        mean_foreground = (weighted_total - background_weighted) / foreground
        variance = background * foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_level, best_variance = level, variance
    return best_level


def _moving_average(values: list[float], window: int) -> list[float]:
    """Centred box filter, zero-padded at the ends: numpy's convolve(mode="same")."""
    length = len(values)
    offset = (window - 1) // 2
    padded = [0.0] * (window - 1) + values + [0.0] * (window - 1)
    full = [sum(padded[i : i + window]) / window for i in range(length + window - 1)]
    return full[offset : offset + length]
