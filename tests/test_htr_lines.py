from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from archai_ocr.config import load_config
from archai_ocr.pipeline import htr_kraken
from archai_ocr.pipeline.htr_kraken import _line_from_record, recognize_crop_lines, recognize_crops
from archai_ocr.pipeline.page import RecognizedLine


class Record:
    """The attributes of a kraken BaselineOCRRecord that the pipeline reads."""

    def __init__(self, prediction: str, confidences: list[float], y: int) -> None:
        self.prediction = prediction
        self.confidences = confidences
        self.baseline = [[10, y], [200, y]]
        self.boundary = [[10, y - 20], [200, y - 20], [200, y + 5], [10, y + 5]]


def test_a_record_keeps_its_text_confidence_and_geometry() -> None:
    line = _line_from_record(Record("  In principio  ", [0.9, 0.8], 40))
    assert line == RecognizedLine(
        text="In principio",
        confidence=pytest.approx(0.85),
        baseline=((10, 40), (200, 40)),
        boundary=((10, 20), (200, 20), (200, 45), (10, 45)),
    )


def test_a_legacy_bbox_record_becomes_a_rectangle() -> None:
    line = _line_from_record({"prediction": "et", "confidences": [1.0], "bbox": (5, 6, 50, 30)})
    assert line.boundary == ((5, 6), (50, 6), (50, 30), (5, 30))
    assert line.baseline == ()


def test_an_empty_prediction_has_no_confidence_but_keeps_its_place() -> None:
    line = _line_from_record(Record("", [0.2], 80))
    assert (line.text, line.confidence) == ("", None)
    assert line.baseline == ((10, 80), (200, 80))


def test_lines_are_translated_into_page_coordinates() -> None:
    line = _line_from_record(Record("x", [1.0], 40)).translated(95, 1000)
    assert line.baseline == ((105, 1040), (295, 1040))


@pytest.fixture
def config(config_file: Path) -> Any:
    return load_config(config_file)


@pytest.fixture
def crops(tmp_path: Path) -> list[Path]:
    paths = []
    for index in range(3):
        path = tmp_path / f"region_{index:03d}.png"
        Image.new("RGB", (220, 120), "white").save(path)
        paths.append(path)
    return paths


@pytest.fixture
def fake_kraken(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stand-ins for the kraken calls, so recognition runs without the models."""
    state: dict[str, Any] = {"records": {}, "fail": set()}
    monkeypatch.setattr(htr_kraken, "KRAKEN_AVAILABLE", True)
    monkeypatch.setattr(htr_kraken, "_recognition_model", lambda _path, _device: object())
    monkeypatch.setattr(htr_kraken, "_segmentation_model", lambda _path, _device, _logger: object())
    monkeypatch.setattr(htr_kraken, "_segment_crop", lambda _image, _model: "segmentation")

    calls = iter(range(100))

    def run(model: Any, image: Any, segmentation: Any) -> list[Any]:
        index = next(calls)
        if index in state["fail"]:
            raise RuntimeError("unreadable crop")
        return state["records"].get(index, [])

    monkeypatch.setattr(htr_kraken, "_run_recognition", run)
    return state


def test_every_line_keeps_its_own_geometry_after_an_empty_one(
    config: Any, crops: list[Path], fake_kraken: dict[str, Any]
) -> None:
    """Pairing texts with boxes by index after dropping empties shifted every box."""
    fake_kraken["records"][0] = [
        Record("prima", [0.9], 30),
        Record("", [0.1], 60),
        Record("tertia", [0.8], 90),
    ]
    lines = recognize_crop_lines(crops[:1], config, logging.getLogger("test"))
    assert [(line.text, line.baseline[0][1]) for line in lines[0]] == [
        ("prima", 30),
        ("", 60),
        ("tertia", 90),
    ]


def test_a_failing_crop_yields_no_lines_and_the_page_continues(
    config: Any, crops: list[Path], fake_kraken: dict[str, Any]
) -> None:
    fake_kraken["records"] = {0: [Record("a", [1.0], 30)], 2: [Record("c", [1.0], 30)]}
    fake_kraken["fail"] = {1}
    lines = recognize_crop_lines(crops, config, logging.getLogger("test"))
    assert [[line.text for line in crop] for crop in lines] == [["a"], [], ["c"]]


def test_recognize_crops_keeps_its_one_string_per_crop_contract(
    config: Any, crops: list[Path], fake_kraken: dict[str, Any]
) -> None:
    fake_kraken["records"] = {
        0: [Record("prima", [1.0], 30), Record("", [1.0], 60), Record("tertia", [1.0], 90)]
    }
    assert recognize_crops(crops[:2], config, logging.getLogger("test")) == ["prima\ntertia", ""]


def test_models_are_loaded_once_until_the_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model.mlmodel"
    model_path.write_bytes(b"weights")
    loads: list[str] = []
    monkeypatch.setattr(htr_kraken, "load_any", lambda path: loads.append(path) or object())
    htr_kraken._load_recognition_model.cache_clear()

    first = htr_kraken._recognition_model(model_path, "")
    assert htr_kraken._recognition_model(model_path, "") is first
    assert len(loads) == 1

    stat = model_path.stat()
    os.utime(model_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    htr_kraken._recognition_model(model_path, "")
    assert len(loads) == 2
    htr_kraken._load_recognition_model.cache_clear()


def test_a_broken_recognition_model_is_reported_with_its_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "broken.mlmodel"
    model_path.write_bytes(b"")

    def fail(path: str) -> Any:
        raise OSError("bad file")

    monkeypatch.setattr(htr_kraken, "load_any", fail)
    htr_kraken._load_recognition_model.cache_clear()
    with pytest.raises(RuntimeError, match="broken.mlmodel"):
        htr_kraken._recognition_model(model_path, "")
    htr_kraken._load_recognition_model.cache_clear()
