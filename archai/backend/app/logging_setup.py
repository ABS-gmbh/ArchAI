"""Logging for the backend process: one handler on the root logger, set from LOG_LEVEL and LOG_FORMAT.

Nothing configured logging before. Under uvicorn the app's loggers (app.*,
archai.*) had no handler of their own, so Python's last-resort handler printed
WARNING and above as bare messages and every INFO record - indexing, linking,
task cleanup - was dropped. uvicorn's own loggers keep their handlers and do
not propagate, so nothing is printed twice.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, TextIO

_HANDLER_NAME = "archai-backend"
_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_STANDARD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with any extra={...} fields kept."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str, fmt: str, *, stream: TextIO | None = None) -> logging.Handler:
    """Install (or replace) the backend's handler on the root logger and return it."""
    root = logging.getLogger()
    for existing in list(root.handlers):
        if existing.get_name() == _HANDLER_NAME:
            root.removeHandler(existing)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(_TEXT_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
    return handler
