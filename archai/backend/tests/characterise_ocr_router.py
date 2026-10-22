"""What the OCR router's pipeline helpers produce on the reference pages.

``tests/test_ocr_router_characterisation.py`` compares the helpers with
``tests/fixtures/ocr_router_characterisation.json``, recorded before they were
moved out of ``app/routers/ocr.py``. To re-record after an intended change::

    python tests/characterise_ocr_router.py --write
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
FIXTURE = BACKEND / "tests" / "fixtures" / "ocr_router_characterisation.json"
READINGS = BACKEND / "eval" / "quality_gate" / "readings.json"
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_VOLATILE = {"created_at", "updated_at", "ts", "took_ms"}


def _texts() -> dict[str, str]:
    pages = json.loads(READINGS.read_text(encoding="utf-8"))["pages"]
    out: dict[str, str] = {}
    for page in pages:
        out[f"{page['page_id']}:reference"] = page["reference"]
        out[f"{page['page_id']}:catmus"] = page["readings"]["kraken_catmus"]
    return out


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in sorted(value.items()) if k not in _VOLATILE}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, str):
        return _UUID.sub("<uuid>", value)
    if isinstance(value, float):
        return round(value, 6)
    return value


_MODULES = ("language_detection", "mention_extraction", "mention_report", "trace_pipeline")


def helper(name: str) -> Any:
    """*name* from the service module it now lives in, or from the router before the move."""
    import importlib

    with contextlib.redirect_stdout(io.StringIO()):
        for module in _MODULES:
            try:
                found = importlib.import_module(f"app.services.{module}")
            except ModuleNotFoundError:
                continue
            if hasattr(found, name):
                return getattr(found, name)
        return getattr(importlib.import_module("app.routers.ocr"), name)


def record() -> dict[str, Any]:
    """Run the helpers on every text, each in a fresh database."""
    with contextlib.redirect_stdout(io.StringIO()):
        from app.db import pipeline_db
    # langdetect samples at random: unseeded, a confidence moves between runs.
    from langdetect import DetectorFactory

    out: dict[str, Any] = {}
    for name, text in _texts().items():
        DetectorFactory.seed = 0
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["ARCHAI_DB_PATH"] = str(Path(tmp) / "archai.sqlite")
            pipeline_db._DB_READY = False
            detected, confidence = helper("_detect_language_metadata")(text)
            mentions, candidates, debug = helper("_extract_mentions_from_text")(text)
            run_id = pipeline_db.create_run(asset_ref=name, asset_sha256=None)
            record: dict[str, Any] = {
                "language": [detected, confidence, helper("_normalize_detected_language")(detected), helper("_quality_language")(detected)],
                "mentions": mentions,
                "candidates": candidates,
                "debug": debug,
            }
            try:
                chunks, stored, candidate_rows, salvage = helper("_run_trace_analysis")(run_id, text)
            except Exception as exc:  # noqa: BLE001 - recorded: a failure is behaviour too
                record["trace_analysis_error"] = type(exc).__name__
            else:
                report = helper("_build_consolidated_report")(run_id, name, stored, salvage, None)
                record.update(
                    chunks=chunks,
                    stored_mentions=stored,
                    candidate_rows=candidate_rows,
                    decisions=pipeline_db.list_entity_decisions(run_id),
                    report=report.replace(run_id, "<run>"),
                )
            out[name] = _clean(record)
    os.environ.pop("ARCHAI_DB_PATH", None)
    pipeline_db._DB_READY = False
    return out


if __name__ == "__main__":
    if "--write" in sys.argv:
        FIXTURE.write_text(json.dumps(record(), ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {FIXTURE}")
    else:
        print(json.dumps(record(), ensure_ascii=False, indent=1)[:2000])
