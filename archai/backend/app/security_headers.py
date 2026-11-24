"""Headers every backend response carries.

X-Content-Type-Options: nosniff keeps a browser from second-guessing a
response's Content-Type: a COCO download, an annotated image or an error body
is handled as what it says it is. Pure ASGI, so a streamed answer (server-sent
events, a ZIP) is not buffered: the header is added to the start of the
response only.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from starlette.datastructures import MutableHeaders

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

HEADERS = {"X-Content-Type-Options": "nosniff"}


class SecurityHeaders:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in HEADERS.items():
                    headers.setdefault(name, value)
            await send(message)

        await self.app(scope, receive, send_with_headers)
