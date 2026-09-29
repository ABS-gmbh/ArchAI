"""PAGE XML (2019-07-15) output for a whole recognised page.

PAGE is what eScriptorium, Transkribus, Kraken and OCR-D exchange, so a page
written here can be opened, corrected and exported again as ground truth - the
missing piece for measuring this pipeline's accuracy. The file holds every zone
the layout model found, each text line with its polygon, baseline and
confidence, and the reading order of the body text.

Two conventions are deliberate:

* Zone types are written as ``custom="structure {type:MarginTextZone;}"``, the
  form eScriptorium and Transkribus read. PAGE's own ``@type`` is left unset:
  Kraken's parser prefers it when present, and its vocabulary is coarser, so it
  would replace "MarginTextZone" with "marginalia" on import.
* The reading order lists the body (main lane) first, in reading order, then the
  secondary lane by position. Body text never interleaves with marginalia.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from archai_ocr import __version__
from archai_ocr.pipeline.assemble_text import NormalizationForm
from archai_ocr.pipeline.page import PageTranscription, Point, RecognizedLine, TranscribedRegion
from archai_ocr.pipeline.zones import zone_kind

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
SCHEMA_LOCATION = f"{PAGE_NS} {PAGE_NS}/pagecontent.xsd"


def write_page_xml(
    page: PageTranscription,
    output_path: Path,
    *,
    created: datetime | None = None,
    normalize: NormalizationForm | None = "NFC",
) -> Path:
    """Serialise *page* to *output_path* and return the path."""
    tree = page_xml_tree(page, created=created, normalize=normalize)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def page_xml_tree(
    page: PageTranscription,
    *,
    created: datetime | None = None,
    normalize: NormalizationForm | None = "NFC",
) -> ET.ElementTree:
    ET.register_namespace("", PAGE_NS)
    ET.register_namespace("xsi", XSI_NS)
    stamp = (created or datetime.now(UTC)).isoformat(timespec="seconds")
    width, height = page.image_size

    root = ET.Element(_tag("PcGts"), {f"{{{XSI_NS}}}schemaLocation": SCHEMA_LOCATION})
    metadata = ET.SubElement(root, _tag("Metadata"))
    ET.SubElement(metadata, _tag("Creator")).text = f"archai-ocr {__version__}"
    ET.SubElement(metadata, _tag("Created")).text = stamp
    ET.SubElement(metadata, _tag("LastChange")).text = stamp

    page_el = ET.SubElement(
        root,
        _tag("Page"),
        {"imageFilename": page.image_name, "imageWidth": str(width), "imageHeight": str(height)},
    )
    named = [(region.region_id, region) for region in page.regions]

    # The schema requires ReadingOrder before any region, and a group that
    # references at least one region, so it is omitted for a page without text.
    ordered = [region_id for region_id, region in named if region.lane in ("main", "secondary")]
    if ordered:
        reading_order = ET.SubElement(page_el, _tag("ReadingOrder"))
        group = ET.SubElement(
            reading_order,
            _tag("OrderedGroup"),
            {"id": "reading_order", "caption": "Main text in reading order, then secondary text"},
        )
        for index, region_id in enumerate(ordered):
            ET.SubElement(group, _tag("RegionRefIndexed"), {"index": str(index), "regionRef": region_id})

    for region_id, region in named:
        _region_element(page_el, region, region_id, (width, height), normalize)

    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def _region_element(
    parent: ET.Element,
    region: TranscribedRegion,
    region_id: str,
    page_size: tuple[int, int],
    normalize: NormalizationForm | None,
) -> None:
    kind = zone_kind(region.detection.class_name)
    element = ET.SubElement(
        parent,
        _tag(kind.element),
        {"id": region_id, "custom": f"structure {{type:{_custom_value(region.detection.class_name)};}}"},
    )
    x1, y1, x2, y2 = region.detection.bbox
    region_polygon = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    ET.SubElement(element, _tag("Coords"), {"points": _points(region_polygon, page_size)})
    if kind.element != "TextRegion":
        return

    for index, line in enumerate(region.lines):
        _line_element(element, line, f"{region_id}_line_{index:03d}", region_polygon, page_size, normalize)
    text = _normalize(region.text, normalize)
    if text:
        equiv = ET.SubElement(element, _tag("TextEquiv"))
        ET.SubElement(equiv, _tag("Unicode")).text = text


def _line_element(
    parent: ET.Element,
    line: RecognizedLine,
    line_id: str,
    fallback_polygon: Sequence[Point],
    page_size: tuple[int, int],
    normalize: NormalizationForm | None,
) -> None:
    element = ET.SubElement(parent, _tag("TextLine"), {"id": line_id})
    # Coords is mandatory for a TextLine; a line whose boundary was lost keeps
    # its region's rectangle rather than making the document invalid.
    polygon = line.boundary if len(line.boundary) >= 3 else fallback_polygon
    ET.SubElement(element, _tag("Coords"), {"points": _points(polygon, page_size)})
    if len(line.baseline) >= 2:
        ET.SubElement(element, _tag("Baseline"), {"points": _points(line.baseline, page_size)})
    attributes = {} if line.confidence is None else {"conf": f"{min(1.0, max(0.0, line.confidence)):.4f}"}
    equiv = ET.SubElement(element, _tag("TextEquiv"), attributes)
    ET.SubElement(equiv, _tag("Unicode")).text = _normalize(line.text, normalize)


def _points(points: Iterable[Point], page_size: tuple[int, int]) -> str:
    """PAGE PointsType: non-negative integers, clamped to the page."""
    width, height = page_size
    return " ".join(
        f"{min(max(int(x), 0), max(width - 1, 0))},{min(max(int(y), 0), max(height - 1, 0))}"
        for x, y in points
    )


def _custom_value(value: str) -> str:
    # The custom-attribute grammar splits on "{", "}", ";" and ":".
    return "".join(ch for ch in value if ch not in "{};:")


def _normalize(text: str, form: NormalizationForm | None) -> str:
    return unicodedata.normalize(form, text) if form else text


def _tag(name: str) -> str:
    return f"{{{PAGE_NS}}}{name}"
