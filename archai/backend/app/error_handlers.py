"""Error answers the app gives itself, bounded and in the API's own shape.

FastAPI's 422 for a request that fails validation repeats each wrong value
("input") in full. A second security review (2026-11-24) sent a 200 MB body
and got a 209.7 MB answer; within MAX_JSON_MEGABYTES the copy still doubles
what one request costs. The 422 now repeats a wrong value only while it is
small, and lists at most MAX_VALIDATION_ERRORS errors.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

MAX_ECHOED_INPUT_CHARS = 200
MAX_VALIDATION_ERRORS = 20


def _small(value: Any) -> bool:
    try:
        return len(json.dumps(jsonable_encoder(value))) <= MAX_ECHOED_INPUT_CHARS
    except (TypeError, ValueError):
        return False


def bounded_validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bounded = []
    for error in errors[:MAX_VALIDATION_ERRORS]:
        error = dict(error)
        if "input" in error and not _small(error["input"]):
            del error["input"]
        if "ctx" in error and not _small(error["ctx"]):
            del error["ctx"]
        bounded.append(error)
    return bounded


async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(bounded_validation_errors(list(exc.errors())))})


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _validation_error)  # type: ignore[arg-type]
