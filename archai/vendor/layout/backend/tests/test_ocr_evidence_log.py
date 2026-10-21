"""The OCR evidence log goes where it is configured, and never into the checkout from a test."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_backend = Path(__file__).resolve().parents[1]
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from app.config import Settings, settings  # type: ignore[import-untyped]  # noqa: E402
from app.services.ocr_evidence import OcrEvidenceRecord, write_ocr_evidence_jsonl  # type: ignore[import-untyped]  # noqa: E402

TRACKED = _backend / "data" / "evidence" / "ocr_evidence.jsonl"


def _record(text: str = "Linea una") -> OcrEvidenceRecord:
    return OcrEvidenceRecord(
        created_at="2026-10-21T00:00:00+00:00",
        image_sha256="0" * 64,
        image_ref="page-1",
        model="kraken_catmus",
        decoding={},
        prompt_version="v",
        pipeline_version="dev",
        script_hint="latin",
        confidence=0.9,
        warnings=[],
        raw_ocr_text=text,
    )


def test_the_default_path_is_the_backends_not_the_working_directorys(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    assert Path(Settings().ocr_evidence_path) == TRACKED


def test_the_path_can_be_set_from_the_environment(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("OCR_EVIDENCE_PATH", str(tmp_path / "log.jsonl"))
    assert Settings().ocr_evidence_path == str(tmp_path / "log.jsonl")


def test_records_are_appended_to_the_configured_file(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(settings, "ocr_evidence_path", str(tmp_path / "nested" / "log.jsonl"))
    write_ocr_evidence_jsonl(_record("prima"))
    write_ocr_evidence_jsonl(_record("secunda"))
    lines = (tmp_path / "nested" / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["raw_ocr_text"] for line in lines] == ["prima", "secunda"]


def test_the_suite_writes_outside_the_checkout() -> None:
    assert Path(settings.ocr_evidence_path).resolve() != TRACKED.resolve()
    assert _backend not in Path(settings.ocr_evidence_path).resolve().parents
