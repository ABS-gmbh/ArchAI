"""The backend scripts/smoke_api.py is documented for is one its over-the-limit checks can pass on.

Since JSON bodies have their own limit (MAX_JSON_MEGABYTES, 48 MB by default),
MAX_UPLOAD_MEGABYTES=2 no longer covers "ocr body over the limit", which sends
a 4 MB JSON body: on the documented backend it answered 422, not 413. Found by
a live regression on 2026-11-24.
"""

from __future__ import annotations

import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
SMOKE = (BACKEND / "scripts" / "smoke_api.py").read_text(encoding="utf-8")
DEPLOYMENT = (BACKEND.parents[1] / "docs" / "deployment.md").read_text(encoding="utf-8")


def _documented(text: str, name: str) -> int:
    match = re.search(rf"{name}=(\d+)", text)
    assert match, f"{name} is not in the documented setup"
    return int(match.group(1))


def _sent_megabytes(pattern: str) -> int:
    match = re.search(pattern, SMOKE)
    assert match, pattern
    return int(match.group(1))


def test_the_json_body_sent_is_over_the_documented_json_limit() -> None:
    sent = _sent_megabytes(r'big = "A" \* \((\d+) << 20\)')
    for text in (SMOKE, DEPLOYMENT):
        assert _documented(text, "MAX_JSON_MEGABYTES") < sent


def test_the_upload_sent_is_over_the_documented_upload_limit() -> None:
    sent = _sent_megabytes(r'"predict/single over the limit", 413, .*?b"\\0" \* \((\d+) << 20\)')
    for text in (SMOKE, DEPLOYMENT):
        assert _documented(text, "MAX_UPLOAD_MEGABYTES") < sent
