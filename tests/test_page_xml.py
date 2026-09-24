from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from archai_ocr.pipeline.layout_yolo import RegionDetection
from archai_ocr.pipeline.page import PageTranscription, RecognizedLine, TranscribedRegion
from archai_ocr.pipeline.page_xml import PAGE_NS, page_xml_tree, write_page_xml

NS = {"p": PAGE_NS}
CREATED = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def region(
    region_id: str,
    name: str,
    bbox: tuple[int, int, int, int],
    lane: str,
    lines: tuple[RecognizedLine, ...] = (),
) -> TranscribedRegion:
    return TranscribedRegion(
        region_id=region_id,
        detection=RegionDetection(bbox=bbox, score=0.9, class_id=0, class_name=name),
        lane=lane,  # type: ignore[arg-type]
        lines=lines,
    )


def line(text: str, y: int, confidence: float | None = 0.93) -> RecognizedLine:
    return RecognizedLine(
        text=text,
        confidence=confidence,
        baseline=((110, y), (790, y)),
        boundary=((110, y - 30), (790, y - 30), (790, y + 8), (110, y + 8)),
    )


@pytest.fixture
def page() -> PageTranscription:
    return PageTranscription(
        image_name="folio_001r.jpg",
        image_size=(1000, 1400),
        regions=(
            region(
                "region_000",
                "MainZone",
                (100, 100, 800, 1200),
                "main",
                (line("In principio erat verbum", 160), line("", 220, None), line("et verbum erat", 280)),
            ),
            region(
                "secondary_000",
                "MarginTextZone",
                (850, 300, 980, 400),
                "secondary",
                (line("apoc. 21.9", 350),),
            ),
            region("layout_000", "StampZone", (50, 1250, 200, 1350), "layout"),
            region("layout_001", "DigitizationArtefactZone", (0, 1360, 1000, 1400), "layout"),
        ),
    )


def parse(page: PageTranscription) -> ET.Element:
    return page_xml_tree(page, created=CREATED).getroot()


def local(element: ET.Element) -> str:
    return element.tag.split("}")[-1]


def test_document_is_page_2019_with_schema_location(page: PageTranscription) -> None:
    root = parse(page)
    assert root.tag == f"{{{PAGE_NS}}}PcGts"
    location = root.get("{http://www.w3.org/2001/XMLSchema-instance}schemaLocation")
    assert location is not None and location.endswith("pagecontent.xsd")


def test_metadata_follows_the_schema_sequence(page: PageTranscription) -> None:
    metadata = parse(page).find("p:Metadata", NS)
    assert metadata is not None
    assert [local(child) for child in metadata] == ["Creator", "Created", "LastChange"]
    assert metadata.findtext("p:Created", namespaces=NS) == "2026-09-24T12:00:00+00:00"


def test_page_carries_image_name_and_size(page: PageTranscription) -> None:
    page_el = parse(page).find("p:Page", NS)
    assert page_el is not None
    assert (page_el.get("imageFilename"), page_el.get("imageWidth"), page_el.get("imageHeight")) == (
        "folio_001r.jpg",
        "1000",
        "1400",
    )


def test_reading_order_comes_first_and_lists_text_regions_only(page: PageTranscription) -> None:
    page_el = parse(page).find("p:Page", NS)
    assert page_el is not None
    assert local(page_el[0]) == "ReadingOrder"
    refs = [ref.get("regionRef") for ref in page_el.iterfind(".//p:RegionRefIndexed", NS)]
    assert refs == ["region_000", "secondary_000"]  # body first, then secondary text
    ids = {element.get("id") for element in page_el if element.get("id")}
    assert set(refs) <= ids


def test_zones_become_the_right_page_elements(page: PageTranscription) -> None:
    page_el = parse(page).find("p:Page", NS)
    assert page_el is not None
    elements = [(local(e), e.get("custom")) for e in page_el if local(e) != "ReadingOrder"]
    assert elements == [
        ("TextRegion", "structure {type:MainZone;}"),
        ("TextRegion", "structure {type:MarginTextZone;}"),
        ("GraphicRegion", "structure {type:StampZone;}"),
        ("NoiseRegion", "structure {type:DigitizationArtefactZone;}"),
    ]


def test_page_type_attribute_is_left_for_the_segmonto_name(page: PageTranscription) -> None:
    """Kraken's importer prefers @type and would replace MarginTextZone with it."""
    assert all(region.get("type") is None for region in parse(page).iter(f"{{{PAGE_NS}}}TextRegion"))


def test_text_region_children_follow_the_schema_sequence(page: PageTranscription) -> None:
    body = parse(page).find(".//p:TextRegion[@id='region_000']", NS)
    assert body is not None
    assert [local(child) for child in body] == ["Coords", "TextLine", "TextLine", "TextLine", "TextEquiv"]
    assert body.findtext("p:TextEquiv/p:Unicode", namespaces=NS) == "In principio erat verbum\net verbum erat"


def test_lines_carry_polygon_baseline_and_confidence(page: PageTranscription) -> None:
    first = parse(page).find(".//p:TextLine[@id='region_000_line_000']", NS)
    assert first is not None
    assert [local(child) for child in first] == ["Coords", "Baseline", "TextEquiv"]
    assert first.find("p:Coords", NS).get("points") == "110,130 790,130 790,168 110,168"  # type: ignore[union-attr]
    assert first.find("p:Baseline", NS).get("points") == "110,160 790,160"  # type: ignore[union-attr]
    assert first.find("p:TextEquiv", NS).get("conf") == "0.9300"  # type: ignore[union-attr]


def test_a_line_read_as_nothing_keeps_its_geometry(page: PageTranscription) -> None:
    empty = parse(page).find(".//p:TextLine[@id='region_000_line_001']", NS)
    assert empty is not None
    equiv = empty.find("p:TextEquiv", NS)
    assert equiv is not None and equiv.get("conf") is None
    assert (equiv.findtext("p:Unicode", namespaces=NS) or "") == ""


def test_coordinates_are_clamped_to_the_page() -> None:
    page = PageTranscription(
        image_name="p.png",
        image_size=(100, 100),
        regions=(region("region_000", "MainZone", (-5, -5, 150, 150), "main"),),
    )
    coords = parse(page).find(".//p:TextRegion/p:Coords", NS)
    assert coords is not None and coords.get("points") == "0,0 99,0 99,99 0,99"


def test_a_line_without_a_boundary_falls_back_to_its_region() -> None:
    page = PageTranscription(
        image_name="p.png",
        image_size=(1000, 1000),
        regions=(region("region_000", "MainZone", (10, 10, 500, 500), "main", (RecognizedLine("x"),)),),
    )
    coords = parse(page).find(".//p:TextLine/p:Coords", NS)
    assert coords is not None and coords.get("points") == "10,10 500,10 500,500 10,500"
    assert parse(page).find(".//p:TextLine/p:Baseline", NS) is None


def test_a_page_without_text_has_no_reading_order() -> None:
    page = PageTranscription(
        image_name="p.png",
        image_size=(100, 100),
        regions=(region("layout_000", "StampZone", (1, 1, 50, 50), "layout"),),
    )
    assert parse(page).find(".//p:ReadingOrder", NS) is None


def test_text_is_unicode_normalised_like_the_txt_output() -> None:
    decomposed = "Rege\u0301s"
    page = PageTranscription(
        image_name="p.png",
        image_size=(1000, 1000),
        regions=(region("region_000", "MainZone", (10, 10, 500, 500), "main", (line(decomposed, 100),)),),
    )
    unicode = parse(page).find(".//p:TextLine/p:TextEquiv/p:Unicode", NS)
    assert unicode is not None and unicode.text == "Reg\u00e9s"


def test_output_is_deterministic_for_a_fixed_timestamp(page: PageTranscription, tmp_path: Path) -> None:
    first = write_page_xml(page, tmp_path / "a.page.xml", created=CREATED).read_bytes()
    second = write_page_xml(page, tmp_path / "b.page.xml", created=CREATED).read_bytes()
    assert first == second
    assert first.startswith(b"<?xml version='1.0' encoding='utf-8'?>")


def test_kraken_reads_the_file_back(page: PageTranscription, tmp_path: Path) -> None:
    """eScriptorium imports PAGE through this parser; zone types must survive it."""
    xml = pytest.importorskip("kraken.lib.xml")
    path = write_page_xml(page, tmp_path / "folio.page.xml", created=CREATED)
    parsed = xml.XMLPage(path, filetype="page")
    assert set(parsed.regions) >= {"MainZone", "MarginTextZone"}
    texts = [line.text for line in parsed.get_sorted_lines()]
    assert "In principio erat verbum" in texts and "apoc. 21.9" in texts
    first = next(line for line in parsed.get_sorted_lines() if line.text == "In principio erat verbum")
    assert [tuple(point) for point in first.baseline] == [(110, 160), (790, 160)]
