"""process_image end to end, with the two model stages replaced by fakes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pytest

from archai_ocr import cli
from archai_ocr.config import load_config
from archai_ocr.pipeline import htr_kraken, layout_yolo
from archai_ocr.pipeline.layout_yolo import PageLayout, RegionDetection
from archai_ocr.pipeline.page import RecognizedLine
from archai_ocr.pipeline.page_xml import PAGE_NS

NS = {"p": PAGE_NS}


def det(name: str, bbox: tuple[int, int, int, int], score: float = 0.9) -> RegionDetection:
    return RegionDetection(bbox=bbox, score=score, class_id=0, class_name=name)


def line(text: str, y: int) -> RecognizedLine:
    return RecognizedLine(text=text, confidence=0.9, baseline=((10, y), (300, y)), boundary=())


@pytest.fixture
def run(config_file: Path, page_image: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Run process_image on a fixed layout; returns (txt_path, run_dir)."""

    def _run(
        layout: PageLayout, texts: dict[str, list[RecognizedLine]], **overrides: Any
    ) -> tuple[Path, Path]:
        config = load_config(config_file, overrides={"layout.main_text_class": "MainZone", **overrides})
        monkeypatch.setattr(layout_yolo, "detect_page_layout", lambda **_kwargs: layout)
        monkeypatch.setattr(
            htr_kraken,
            "recognize_crop_lines",
            lambda crop_paths, **_kwargs: [texts.get(path.name, []) for path in crop_paths],
        )
        txt_path = cli.process_image(page_image, config, logging.getLogger("test"))
        return txt_path, txt_path.parent

    return _run


LAYOUT = PageLayout(
    main=(det("MainZone", (100, 100, 800, 1200)),),
    secondary=(
        det("NumberingZone", (900, 20, 960, 60), 0.3),
        det("MarginTextZone", (850, 300, 980, 400), 0.6),
    ),
    layout=(det("StampZone", (50, 1250, 200, 1350)),),
)
TEXTS = {
    "region_000.png": [line("In principio erat verbum", 40), line("et verbum erat", 90)],
    "secondary_000.png": [line("28", 20)],
    "secondary_001.png": [line("apoc. 21.9", 30)],
}


def test_the_txt_holds_body_text_only(run: Any) -> None:
    txt_path, _ = run(LAYOUT, TEXTS)
    assert txt_path.read_text(encoding="utf-8") == "In principio erat verbum\net verbum erat\n"


def test_text_outside_the_body_gets_its_own_file(run: Any) -> None:
    txt_path, run_dir = run(LAYOUT, TEXTS)
    secondary = run_dir / f"{txt_path.stem}.secondary.txt"
    assert secondary.read_text(encoding="utf-8") == "[Numbering]\n28\n\n[Marginalia]\napoc. 21.9\n"


def test_page_xml_is_written_on_request_with_lines_in_page_coordinates(run: Any) -> None:
    txt_path, run_dir = run(LAYOUT, TEXTS, **{"runtime.write_page_xml": True})
    root = ET.parse(run_dir / f"{txt_path.stem}.page.xml").getroot()
    regions = [(r.get("id"), r.get("custom")) for r in root.find("p:Page", NS) if r.get("id")]  # type: ignore[union-attr]
    assert regions == [
        ("region_000", "structure {type:MainZone;}"),
        ("secondary_000", "structure {type:NumberingZone;}"),
        ("secondary_001", "structure {type:MarginTextZone;}"),
        ("layout_000", "structure {type:StampZone;}"),
    ]
    # The body crop starts at (95, 95): its 5 px padding off the (100, 100) box.
    baseline = root.find(".//p:TextLine[@id='region_000_line_000']/p:Baseline", NS)
    assert baseline is not None and baseline.get("points") == "105,135 395,135"


def test_page_xml_is_not_written_by_default(run: Any) -> None:
    txt_path, run_dir = run(LAYOUT, TEXTS)
    assert not (run_dir / f"{txt_path.stem}.page.xml").exists()


def test_a_page_with_only_secondary_text_still_yields_it(run: Any) -> None:
    layout = PageLayout(main=(), secondary=(det("NumberingZone", (900, 20, 960, 60)),))
    txt_path, run_dir = run(layout, {"secondary_000.png": [line("f. 1", 20)]})
    assert txt_path.read_text(encoding="utf-8") == ""
    assert (run_dir / f"{txt_path.stem}.secondary.txt").read_text(encoding="utf-8") == "[Numbering]\nf. 1\n"


def test_cli_flags_control_the_new_outputs() -> None:
    args = cli.build_parser().parse_args(["--image", "p.png", "--no-secondary-text", "--page-xml"])
    overrides = cli._build_overrides(args)
    assert overrides["layout.secondary_text_class"] == []
    assert overrides["runtime.write_page_xml"] is True
