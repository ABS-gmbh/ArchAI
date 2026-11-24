"""Refuse request bodies larger than the upload limit, before they are read into memory.

Multipart uploads to /predict are capped while they stream to disk, but the
JSON routes - /ocr/*, /agents/ocr, /chat/label-analysis, which carry the page
as base64 - had no limit: Starlette reads the whole body into memory, then it
is parsed into a str and decoded into bytes, several copies of whatever a
client sends. Every body is now limited to MAX_UPLOAD_MEGABYTES plus the base64
overhead (4/3) and 1 MB for the JSON around it, and a larger one is answered
413 - from Content-Length when the client sends it, and by counting the bytes
as they arrive otherwise.

That upload limit (about 1.3 GB by default) applies to multipart bodies only.
Every other body - JSON - is limited to MAX_JSON_MEGABYTES (48 MB by default):
the largest JSON request the frontend sends carries one page image as a base64
data URL, and an A4 page scanned at 600 dpi and saved as PNG (27 MB) makes a
36 MB body; the same page as JPEG makes 10.5 MB.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

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


def _limit_for(scope: Scope) -> int:
    content_type = dict(scope.get("headers") or []).get(b"content-type", b"")
    if content_type.lower().startswith(b"multipart/form-data"):
        return max_request_bytes()
    return max_json_bytes()


class RequestBodyLimit:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = _limit_for(scope)
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
