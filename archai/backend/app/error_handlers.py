"""Error answers the app gives itself, bounded and in the API's own shape: the
422 for a request that fails validation, and the 500 for an unhandled exception.

FastAPI's 422 for a request that fails validation repeats each wrong value
("input") in full. A second security review (2026-11-24) sent a 200 MB body
and got a 209.7 MB answer; within MAX_JSON_MEGABYTES the copy still doubles
what one request costs. The 422 now repeats a wrong value only while it is
small, and lists at most MAX_VALIDATION_ERRORS errors.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, MutableMapping
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


# An exception no route handles. Starlette answers it from ServerErrorMiddleware,
# outermost, outside every middleware the app adds: plain text, without
# X-Content-Type-Options or the CORS headers a browser needs to read it (second
# security review, 2026-11-24). UnhandledErrors sits inside them instead, logs
# the exception with its traceback and answers this sentence in JSON.
UNEXPECTED_ERROR_DETAIL = "The server could not complete this request. Its log has the details."

_logger = logging.getLogger("app.errors")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class UnhandledErrors:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception:
            _logger.exception("Unhandled error in %s %s", scope.get("method", ""), scope.get("path", ""))
            if started:
                return  # part of an answer went out; the connection ends with what was sent
            body = json.dumps({"detail": UNEXPECTED_ERROR_DETAIL}).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
                }
            )
            await send({"type": "http.response.body", "body": body})
