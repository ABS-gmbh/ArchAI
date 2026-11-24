"""scripts/benchmark_recognizers.py: --save-readings writes what calibrate_quality_gate reads."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]


def _load(name: str, monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location(name, BACKEND / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


def test_save_readings_writes_the_texts_in_the_calibration_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    br = _load("benchmark_recognizers", monkeypatch)
    images = tmp_path / "images"
    images.mkdir()
    pages = {"page-a": ("old_french", "li rois de france\nsi uint a paris"), "page-b": ("latin", "in principio erat uerbum")}
    manifest = tmp_path / "manifest.jsonl"
    lines = []
    for page_id, (hint, reference) in pages.items():
        content = f"stand-in image {page_id}".encode()  # never decoded: read_page is replaced
        (images / f"{page_id}.png").write_bytes(content)
        (tmp_path / f"{page_id}.gt.txt").write_text(reference, encoding="utf-8")
        lines.append(json.dumps({"page_id": page_id, "image_sha256": hashlib.sha256(content).hexdigest(), "reference": f"{page_id}.gt.txt", "language_hint": hint}))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def fake_read(engine: str, page) -> tuple[str, str]:
        if engine == "glmocr" and page.page_id == "page-b":
            raise RuntimeError("engine down")
        return f"{engine} read {page.reference}", "model"

    monkeypatch.setattr(br, "read_page", fake_read)
    # main() switches the OCR evidence log off for the run; put it back afterwards.
    monkeypatch.setattr(br.ocr_agent, "write_ocr_evidence_jsonl", br.ocr_agent.write_ocr_evidence_jsonl)
    monkeypatch.setenv("YOLO_VERBOSE", "False")
    out = tmp_path / "fresh.json"
    assert br.main(["--images", str(images), "--manifest", str(manifest), "--engines", "segmented", "glmocr", "--save-readings", str(out)]) == 0
    capsys.readouterr()

    saved = json.loads(out.read_text(encoding="utf-8"))
    by_id = {page["page_id"]: page for page in saved["pages"]}
    assert by_id["page-a"]["language"] == "french" and by_id["page-b"]["language"] == "latin"
    assert by_id["page-a"]["reference"] == pages["page-a"][1]
    assert by_id["page-a"]["readings"] == {
        "segmented": f"segmented read {pages['page-a'][1]}",
        "glmocr": f"glmocr read {pages['page-a'][1]}",
    }
    assert list(by_id["page-b"]["readings"]) == ["segmented"]  # the failed engine is left out

    rows = _load("calibrate_quality_gate", monkeypatch).calibrate_readings(out)
    assert {(row["page"], row["sample"]) for row in rows} >= {("page-a", "segmented"), ("page-a", "glmocr"), ("page-b", "segmented")}
