"""Optional bearer-token authentication for the whole API (off unless API_TOKEN is set).

With API_TOKEN empty, the default, nothing changes: every route answers
anyone who can reach it, which is only safe when the backend is reachable
from localhost alone (startup logs a warning saying so). With API_TOKEN set,
every request except GET /api/health and CORS preflights is checked by
ApiTokenGate, an ASGI middleware, before the app receives a byte of its body:
a client without the token cannot make the server read or parse an upload
(tests/test_token_before_body.py). Every router but health is also included
with require_api_token as a dependency (app/main.py), the same check a second
time; a route cannot be added without it, and tests/test_api_token.py fails if
one is. The analytics routes keep their own login on top.

The token is accepted in an ``X-API-Token: <token>`` header or as
``Authorization: Bearer <token>``, compared in constant time. X-API-Token
exists because the analytics routes already use ``Authorization: Bearer``
for their session token: a client of those sends both headers, and the
frontend's server-side proxy should always use X-API-Token. The two server-sent-event streams a browser opens with
EventSource, which cannot send headers, also accept it as a ``token`` query
parameter, and only those two:
GET /api/ocr/trace/stream/{run_id} and GET /api/predict/batch/{task_id}/progress.
The token is never logged; uvicorn's access log has it redacted
(app/logging_setup.py).
"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Awaitable, Callable, Iterable, MutableMapping
from typing import Any
from urllib.parse import parse_qsl

from fastapi import HTTPException, Request

from app.config import settings

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

# The routes where ?token= is accepted: EventSource cannot set headers.
_STREAM_PATHS = (
    re.compile(r"^/api/ocr/trace/stream/[^/]+$"),
    re.compile(r"^/api/predict/batch/[^/]+/progress$"),
)
_DETAIL = "Authentication required: send the API token as 'X-API-Token: <token>' or 'Authorization: Bearer <token>'."
_CHALLENGE = 'Bearer realm="archai"'
_UNAUTHORIZED = HTTPException(status_code=401, detail=_DETAIL, headers={"WWW-Authenticate": _CHALLENGE})
# Answered without the token: monitors and load balancers poll health.
_OPEN_PATHS = frozenset({"/api/health"})


def _matches(candidate: str, token: str) -> bool:
    return secrets.compare_digest(candidate.encode(), token.encode())


def has_api_token(method: str, path: str, headers: Iterable[tuple[str, str]], query_string: str) -> bool:
    """True when API_TOKEN is unset or the request carries it (headers are lower-cased names)."""
    token = settings.api_token
    if not token:
        return True
    values = dict(headers)
    if _matches(values.get("x-api-token", "").strip(), token):
        return True
    scheme, _, credentials = values.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and _matches(credentials.strip(), token):
        return True
    if method == "GET" and any(pattern.match(path) for pattern in _STREAM_PATHS):
        query = dict(parse_qsl(query_string, keep_blank_values=True))
        if _matches(query.get("token", ""), token):
            return True
    return False


def require_api_token(request: Request) -> None:
    """FastAPI dependency: pass when API_TOKEN is unset or the request carries it."""
    if not has_api_token(request.method, request.url.path, request.headers.items(), request.url.query):
        raise _UNAUTHORIZED


class ApiTokenGate:
    """Refuse a request without the API token before its body is received.

    Sits inside CORSMiddleware, so a 401 carries the CORS headers a browser
    needs to read it, and preflights are answered by CORS before they get here.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not settings.api_token or _exempt(scope):
            await self.app(scope, receive, send)
            return
        headers = [(name.decode("latin-1").lower(), value.decode("latin-1")) for name, value in scope.get("headers") or []]
        query = (scope.get("query_string") or b"").decode("latin-1")
        if has_api_token(scope["method"], scope["path"], headers, query):
            await self.app(scope, receive, send)
            return
        body = json.dumps({"detail": _DETAIL}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", _CHALLENGE.encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _exempt(scope: Scope) -> bool:
    path = scope.get("path", "")
    if path in _OPEN_PATHS or path in _open_doc_paths():
        return True
    # A CORS preflight carries no credentials by design (CORSMiddleware answers it).
    headers = dict(scope.get("headers") or [])
    return scope.get("method") == "OPTIONS" and b"access-control-request-method" in headers


def _open_doc_paths() -> frozenset[str]:
    from app.config import api_docs_enabled

    if not api_docs_enabled(settings):
        return frozenset()
    return frozenset({"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"})
