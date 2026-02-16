from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone

import json
import logging
import time
from typing import Iterator


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("stage", "duration_s", "image", "output", "count"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def setup_logging(level: str = "INFO") -> None:
    logger = logging.getLogger()
    logger.setLevel(level.upper())

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    logger.handlers.clear()
    logger.addHandler(handler)


@contextmanager
def log_stage(logger: logging.Logger, stage: str, **extra: object) -> Iterator[None]:
    logger.info("stage.start", extra={"stage": stage, **extra})
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        logger.info(
            "stage.end",
            extra={"stage": stage, "duration_s": round(elapsed, 3), **extra},
        )
