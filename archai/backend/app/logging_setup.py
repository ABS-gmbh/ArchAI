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
import re
import sys
from datetime import datetime, timezone
from typing import Any, TextIO

_HANDLER_NAME = "archai-backend"
_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_STANDARD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with any extra={...} fields kept."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
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


_TOKEN_IN_QUERY = re.compile(r"([?&]token=)[^&\s\"]*")


class RedactTokenFilter(logging.Filter):
    """Blank the value of a ?token= query parameter before a record is written.

    The two event-stream routes accept the API token in the query string
    (app/auth.py), and uvicorn's access log writes each request's full path.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(
                _TOKEN_IN_QUERY.sub(r"\1[redacted]", arg) if isinstance(arg, str) else arg for arg in record.args
            )
        if isinstance(record.msg, str):
            record.msg = _TOKEN_IN_QUERY.sub(r"\1[redacted]", record.msg)
        return True


def install_token_redaction() -> None:
    """Attach the redaction to uvicorn's access logger, once."""
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(existing, RedactTokenFilter) for existing in access.filters):
        access.addFilter(RedactTokenFilter())


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
    install_token_redaction()
    return handler
