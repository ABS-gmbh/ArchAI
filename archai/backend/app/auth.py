"""Optional bearer-token authentication for the whole API (off unless API_TOKEN is set).

With API_TOKEN empty, the default, nothing changes: every route answers
anyone who can reach it, which is only safe when the backend is reachable
from localhost alone (startup logs a warning saying so). With API_TOKEN set,
every router except /api/health is included with require_api_token as a
dependency (app/main.py). A route cannot be added without it, and
tests/test_api_token.py fails if one is. The analytics routes keep their own
login on top.

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

import re
import secrets

from fastapi import HTTPException, Request

from app.config import settings

# The routes where ?token= is accepted: EventSource cannot set headers.
_STREAM_PATHS = (
    re.compile(r"^/api/ocr/trace/stream/[^/]+$"),
    re.compile(r"^/api/predict/batch/[^/]+/progress$"),
)
_UNAUTHORIZED = HTTPException(
    status_code=401,
    detail="Authentication required: send the API token as 'X-API-Token: <token>' or 'Authorization: Bearer <token>'.",
    headers={"WWW-Authenticate": 'Bearer realm="archai"'},
)


def _matches(candidate: str, token: str) -> bool:
    return secrets.compare_digest(candidate.encode(), token.encode())


def require_api_token(request: Request) -> None:
    """FastAPI dependency: pass when API_TOKEN is unset or the request carries it."""
    token = settings.api_token
    if not token:
        return
    if _matches(request.headers.get("x-api-token", "").strip(), token):
        return
    scheme, _, credentials = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and _matches(credentials.strip(), token):
        return
    if request.method == "GET" and any(pattern.match(request.url.path) for pattern in _STREAM_PATHS):
        if _matches(request.query_params.get("token", ""), token):
            return
    raise _UNAUTHORIZED
