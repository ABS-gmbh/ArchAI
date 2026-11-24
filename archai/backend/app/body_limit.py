"""Refuse request bodies larger than the upload limit, before they are read into memory.

Multipart uploads to /predict are capped while they stream to disk, but the
JSON routes - /ocr/*, /agents/ocr, /chat/label-analysis, which carry the page
as base64 - had no limit: Starlette reads the whole body into memory, then it
is parsed into a str and decoded into bytes, several copies of whatever a
client sends. Every body is now limited to MAX_UPLOAD_MEGABYTES plus the base64
overhead (4/3) and 1 MB for the JSON around it, and a larger one is answered
413 - from Content-Length when the client sends it, and by counting the bytes
as they arrive otherwise.

That upload limit (about 1.3 GB by default) applies to multipart bodies sent
to a route that takes an upload (an UploadFile parameter: /predict/single and
/predict/batch), found from the routes themselves. Every other body - JSON, or
a multipart body sent to any other route - is limited to MAX_JSON_MEGABYTES
(48 MB by default). The limit used to follow the Content-Type alone, so a 200
MB body sent to /chat/completions as multipart was read whole (second security
review, 2026-11-24). JSON bodies are limited to MAX_JSON_MEGABYTES because
the largest JSON request the frontend sends carries one page image as a base64
data URL, and an A4 page scanned at 600 dpi and saved as PNG (27 MB) makes a
36 MB body; the same page as JPEG makes 10.5 MB.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Iterable, MutableMapping, Sequence
from typing import Any

from fastapi import APIRouter, params
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute
from starlette.routing import BaseRoute, compile_path

from app.config import settings

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def max_request_bytes() -> int:
    upload = max(1, int(settings.max_upload_megabytes)) * (1 << 20)
    return upload * 4 // 3 + (1 << 20)


def max_json_bytes() -> int:
    return max(1, int(settings.max_json_megabytes)) * (1 << 20)


def takes_upload(route: BaseRoute) -> bool:
    """True for a route with a File or UploadFile parameter, its dependencies' included."""
    return isinstance(route, APIRoute) and _takes_file(route.dependant)


def _takes_file(dependant: Dependant) -> bool:
    return any(isinstance(field.field_info, params.File) for field in dependant.body_params) or any(
        _takes_file(sub) for sub in dependant.dependencies
    )


def upload_paths(routers: Iterable[tuple[str, APIRouter]]) -> tuple[re.Pattern[str], ...]:
    """The path patterns of the routes that take an upload, for (prefix, router) pairs."""
    return tuple(
        compile_path(prefix + route.path)[0]
        for prefix, router in routers
        for route in router.routes
        if isinstance(route, APIRoute) and takes_upload(route)
    )


def _route_path(scope: Scope) -> str:
    # As Starlette matches routes: the path below root_path.
    path, root = scope.get("path", ""), scope.get("root_path", "")
    return path[len(root) :] if root and path.startswith(root) else path


def _limit_for(scope: Scope, uploads: Sequence[re.Pattern[str]]) -> int:
    content_type = dict(scope.get("headers") or []).get(b"content-type", b"")
    if content_type.lower().startswith(b"multipart/form-data"):
        path = _route_path(scope)
        if any(pattern.match(path) for pattern in uploads):
            return max_request_bytes()
    return max_json_bytes()


class RequestBodyLimit:
    def __init__(self, app: ASGIApp, upload_paths: Sequence[re.Pattern[str]] = ()) -> None:
        self.app = app
        self.upload_paths = tuple(upload_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = _limit_for(scope, self.upload_paths)
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await _send_413(send, limit)
            return

        received = 0
        started = False
        refused = False

        async def counting_receive() -> Message:
            # Over the limit, the 413 goes out at once and the app is told the
            # client left: raising here instead is caught by FastAPI's body
            # parsing and turned into a 400.
            nonlocal received, refused
            if refused:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    refused = True
                    if not started:
                        await _send_413(send, limit)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal started
            if refused:
                return  # the 413 has been sent; nothing the app says now reaches the client
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except Exception:
            if not refused:
                raise


async def _send_413(send: Send, limit: int) -> None:
    body = json.dumps(
        {"detail": f"The request is larger than the server accepts ({limit // (1 << 20)} MB)."}
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})
