"""The recognised page: regions by lane, lines with geometry and confidence.

Recognition previously returned one string per region, so everything the
recognizer knew about a line - where it is, its baseline, how confident it was -
was discarded before any output could use it. These records keep it, in page
coordinates, so the PAGE XML writer can place each line where it was read.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from archai_ocr.pipeline.layout_yolo import RegionDetection
from archai_ocr.pipeline.zones import Lane

Point = tuple[int, int]


@dataclass(frozen=True)
class RecognizedLine:
    """One recognised text line.

    ``confidence`` is the mean per-character confidence the recognizer reported.
    It ranks lines usefully but is not a calibrated probability of correctness.
    """

    text: str
    confidence: float | None = None
    baseline: tuple[Point, ...] = ()
    boundary: tuple[Point, ...] = ()

    def translated(self, dx: int, dy: int) -> RecognizedLine:
        """The same line with its geometry shifted by (dx, dy)."""
        return replace(
            self,
            baseline=tuple((x + dx, y + dy) for x, y in self.baseline),
            boundary=tuple((x + dx, y + dy) for x, y in self.boundary),
        )


@dataclass(frozen=True)
class TranscribedRegion:
    """A detected zone and, for text lanes, the lines recognised inside it."""

    region_id: str
    detection: RegionDetection
    lane: Lane
    lines: tuple[RecognizedLine, ...] = ()

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text)


@dataclass(frozen=True)
class PageTranscription:
    """Everything the pipeline found on one page, in output order.

    Regions are ordered main lane first (in reading order), then secondary, then
    layout-only zones, each by position.
    """

    image_name: str
    image_size: tuple[int, int]
    regions: tuple[TranscribedRegion, ...]

    def lane(self, lane: Lane) -> tuple[TranscribedRegion, ...]:
        return tuple(region for region in self.regions if region.lane == lane)


def mean_confidence(confidences: Sequence[float]) -> float | None:
    """Mean of per-character confidences, or None when there are none."""
    values = [float(value) for value in confidences]
    if not values:
        return None
    return sum(values) / len(values)
