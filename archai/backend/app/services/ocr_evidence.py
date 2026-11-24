from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import threading
from pathlib import Path

from app.config import settings

_LOGGER = logging.getLogger(__name__)
# Whether the last write failed: a failure is logged when a streak starts and
# the recovery when it ends, not on every OCR call in between.
_failing = False
_state_lock = threading.Lock()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class OcrEvidenceRecord:
    created_at: str
    image_sha256: str
    image_ref: str
    model: str
    decoding: dict
    prompt_version: str
    pipeline_version: str
    script_hint: str
    confidence: float
    warnings: list[str]
    raw_ocr_text: str
    final_text: str | None = None
    final_changes: list[str] | None = None


def write_ocr_evidence_jsonl(
    record: OcrEvidenceRecord,
    out_path: str | None = None,
) -> bool:
    """Append *record* to *out_path*, by default ``settings.ocr_evidence_path``.

    The default is read at call time, so a test (or OCR_EVIDENCE_PATH) can send
    the log elsewhere.

    Never raises: a lost record must not fail the OCR call that made it. A
    failure is logged as an ERROR when a streak of failures begins, and a
    WARNING when a write succeeds again, so an unwritable log is seen once,
    not on every call. Returns whether the record was written.
    """
    global _failing
    target = Path(out_path or settings.ocr_evidence_path)
    try:
        line = json.dumps(asdict(record), ensure_ascii=False) + "\n"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception as exc:
        with _state_lock:
            starts_a_streak, _failing = not _failing, True
        if starts_a_streak:
            _LOGGER.error(
                "The OCR evidence log cannot be written at %s (%s: %s). OCR goes on, but its calls are not "
                "recorded until a write succeeds again; this is logged once, not for every call.",
                target,
                type(exc).__name__,
                exc,
            )
        return False
    with _state_lock:
        recovered, _failing = _failing, False
    if recovered:
        _LOGGER.warning("The OCR evidence log at %s is being written again.", target)
    return True
