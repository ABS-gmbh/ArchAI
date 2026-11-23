"""Health check: is the process up, and can it reach what it depends on.

``status`` is "ok" when every check passes, "degraded" when only an optional
dependency is down (the app still serves, without that feature), and
"unavailable" with HTTP 503 when the pipeline database cannot be used, since
no OCR run can be recorded without it.

- ``database`` (required): the pipeline SQLite database answers a query.
- ``ollama`` (optional): the GLM-OCR host answers /api/version. Without it the
  GLM-OCR engine and the segmented engine's fallback fail.
- ``chat_api_key`` (optional): a SAIA / Chat AI key is configured. Whether it
  is valid is not checked - that would cost a request per probe - and the key
  itself is never returned.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import requests
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import pipeline_db
from app.services import inference
from app.services.saia_client import SaiaClient, SaiaConfigError

router = APIRouter(tags=["health"])

_OLLAMA_TIMEOUT_SECONDS = 2.0


def _check_database() -> dict[str, Any]:
    try:
        pipeline_db._init_db_if_needed()
        with pipeline_db._connect() as conn:
            conn.execute("SELECT 1").fetchone()
    except (sqlite3.Error, OSError) as exc:
        return {"ok": False, "detail": f"Pipeline database unavailable: {type(exc).__name__}: {exc}"}
    return {"ok": True}


def _check_ollama() -> dict[str, Any]:
    host = str(settings.glmocr_ollama_host or "").rstrip("/")
    if not host:
        return {"ok": False, "detail": "GLMOCR_OLLAMA_HOST is not set."}
    try:
        response = requests.get(f"{host}/api/version", timeout=_OLLAMA_TIMEOUT_SECONDS)
        response.raise_for_status()
        version = response.json().get("version")
    except (requests.RequestException, ValueError, AttributeError) as exc:
        return {"ok": False, "host": host, "detail": f"Ollama not reachable: {type(exc).__name__}"}
    return {"ok": True, "host": host, "version": version}


def _check_chat_api_key() -> dict[str, Any]:
    try:
        SaiaClient._resolve_api_key()
    except SaiaConfigError:
        return {"ok": False, "detail": "No chat API key configured (CHAT_AI_API_KEY or SAIA_API_KEY)."}
    return {"ok": True, "detail": "A key is configured; it is not validated here."}


# A plain def: the checks are a SQLite query and an HTTP request, which FastAPI
# runs in its threadpool instead of on the event loop.
@router.get("/health")
def health_check() -> JSONResponse:
    checks = {
        "database": _check_database(),
        "ollama": _check_ollama(),
        "chat_api_key": _check_chat_api_key(),
    }
    if not checks["database"]["ok"]:
        status, code = "unavailable", 503
    elif all(check["ok"] for check in checks.values()):
        status, code = "ok", 200
    else:
        status, code = "degraded", 200
    return JSONResponse(
        status_code=code,
        content={
            "status": status,
            "models_loaded": inference._model_pool is not None,
            "checks": checks,
        },
    )
