from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

LOGGER_NAME = "archai_ocr"

# Attributes present on every LogRecord; anything else a caller passes via
# extra={...} is application data and belongs in the JSON payload.
_STANDARD_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per record, preserving caller-supplied extras."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Previously only a fixed whitelist of keys survived, so any new extra
        # field was silently dropped.
        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_ATTRS or key.startswith("_"):
                continue
            payload[key] = value if _json_safe(value) else str(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # ensure_ascii=False keeps medieval Latin/French diacritics readable in logs
        # instead of rendering them as \uXXXX escapes.
        return json.dumps(payload, ensure_ascii=False, default=str)


def _json_safe(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), list, dict))


def setup_logging(level: str = "INFO", *, stream: Any | None = None) -> logging.Logger:
    """Configure the archai_ocr logger.

    Scoped to this package's logger rather than the root logger: the previous
    version called root.handlers.clear(), which discarded handlers installed by
    any host application importing this package, and routed third-party library
    logs (ultralytics, kraken, torch) through the JSON formatter as well.
    """
    normalized = level.upper()
    if normalized not in logging._nameToLevel:  # noqa: SLF001 - the public API has no validator
        raise ValueError(
            f"Invalid log level {level!r}. Choose one of: {', '.join(sorted(logging._nameToLevel))}."  # noqa: SLF001
        )

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(normalized)

    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())

    for existing in list(logger.handlers):
        logger.removeHandler(existing)
    logger.addHandler(handler)
    # Do not re-emit through the root logger's handlers.
    logger.propagate = False
    return logger


@contextmanager
def log_stage(logger: logging.Logger, stage: str, **extra: object) -> Iterator[None]:
    logger.info("stage.start", extra={"stage": stage, **extra})
    started = time.perf_counter()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        elapsed = time.perf_counter() - started
        logger.info(
            "stage.end",
            extra={"stage": stage, "duration_s": round(elapsed, 3), "ok": not failed, **extra},
        )
