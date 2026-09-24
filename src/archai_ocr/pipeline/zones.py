"""Which layout zones carry text, and how each is recorded in PAGE XML.

The shipped layout model emits the eleven SegmOnto zone classes. Until 0.3 only
MainZone was transcribed, so marginal glosses, running titles, foliation and
quire marks - most of the retrievable metadata about a codex - never reached the
output. Zones are now assigned to one of three lanes:

``main``
    Body text, in reading order. The only lane written to the page's ``.txt``,
    whose content is unchanged by this module.
``secondary``
    Text outside the body: transcribed, but kept out of the body's reading order
    so a gloss never splices itself into the middle of a sentence.
``layout``
    Everything else the model found - decoration, stamps, music, scanner
    artefacts. Recorded in PAGE XML for a complete layout, never recognised.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

Lane = Literal["main", "secondary", "layout"]

PageElement = Literal["TextRegion", "GraphicRegion", "MusicRegion", "NoiseRegion"]


@dataclass(frozen=True)
class ZoneKind:
    """How one layout class is represented in the outputs."""

    element: PageElement
    label: str


# Keyed by casefolded class name; matching elsewhere is case-insensitive too.
SEGMONTO_ZONES: dict[str, ZoneKind] = {
    "mainzone": ZoneKind("TextRegion", "Main text"),
    "margintextzone": ZoneKind("TextRegion", "Marginalia"),
    "runningtitlezone": ZoneKind("TextRegion", "Running title"),
    "numberingzone": ZoneKind("TextRegion", "Numbering"),
    "quiremarkszone": ZoneKind("TextRegion", "Quire marks"),
    "dropcapitalzone": ZoneKind("TextRegion", "Drop capital"),
    "titlepagezone": ZoneKind("TextRegion", "Title page"),
    "graphiczone": ZoneKind("GraphicRegion", "Graphic"),
    "stampzone": ZoneKind("GraphicRegion", "Stamp"),
    "musiczone": ZoneKind("MusicRegion", "Music"),
    "digitizationartefactzone": ZoneKind("NoiseRegion", "Digitization artefact"),
}

# Marginalia, running titles, foliation and quire marks are the text-bearing
# zones a manuscript page carries besides its body.
DEFAULT_SECONDARY_TEXT_CLASSES: tuple[str, ...] = (
    "MarginTextZone",
    "RunningTitleZone",
    "NumberingZone",
    "QuireMarksZone",
)


def zone_kind(class_name: str) -> ZoneKind:
    """The PAGE representation of *class_name*; unknown classes are text."""
    known = SEGMONTO_ZONES.get(class_name.casefold())
    if known is not None:
        return known
    # A custom model's vocabulary is unknown here. Treating its classes as text
    # regions keeps them visible and editable in PAGE-based correction tools.
    return ZoneKind("TextRegion", class_name)


@dataclass(frozen=True)
class LaneAssignment:
    """Maps a detected class to its lane from the configured class lists."""

    main_classes: frozenset[str]
    secondary_classes: frozenset[str]

    @classmethod
    def from_names(cls, main: Iterable[str], secondary: Iterable[str]) -> LaneAssignment:
        main_set = frozenset(name.casefold() for name in main)
        # A class configured as main text is never also transcribed as secondary
        # text: that would record the same lines twice.
        secondary_set = frozenset(name.casefold() for name in secondary) - main_set
        return cls(main_set, secondary_set)

    def lane(self, class_name: str) -> Lane:
        key = class_name.casefold()
        if key in self.main_classes:
            return "main"
        if key in self.secondary_classes:
            return "secondary"
        return "layout"
