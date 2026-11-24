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
from urllib.parse import unquote_plus
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


# A query runs to the next whitespace: the request target cannot hold any
# (uvicorn's parsers refuse it), while a raw '"' or '#' can be in it (httptools
# 0.9 and h11 0.16 both pass a raw quote into the query, h11 a raw '#' too).
# Stopping at either, as this did, left "?a=\"&token=<value>" in clear.
_QUERY = re.compile(r"\?(\S*)")
_PAIR_SEPARATOR = re.compile(r"([&;])")


def _redact_query(match: re.Match[str]) -> str:
    """Blank every parameter whose name, once decoded, is "token" in any case."""
    parts = _PAIR_SEPARATOR.split(match.group(1))
    for index in range(0, len(parts), 2):
        name, equals, _ = parts[index].partition("=")
        if equals and unquote_plus(name).strip().lower() == "token":
            parts[index] = f"{name}=[redacted]"
    return "?" + "".join(parts)


def redact_token(text: str) -> str:
    return _QUERY.sub(_redact_query, text)


class RedactTokenFilter(logging.Filter):
    """Blank the value of a ?token= query parameter before a record is written.

    The two event-stream routes accept the API token in the query string
    (app/auth.py), and uvicorn's access log writes each request's full path.
    The parameter's name is decoded before it is compared: "?%74oken=" used to
    be written in clear, because only the literal "token=" was matched.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(redact_token(arg) if isinstance(arg, str) else arg for arg in record.args)
        if isinstance(record.msg, str):
            record.msg = redact_token(record.msg)
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
