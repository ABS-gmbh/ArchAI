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

The endpoint is unauthenticated, so each check returns only ``ok``, a fixed
``detail`` and, for Ollama, the version. Exception text (a SQLite error can
carry the database path) and the Ollama host go to the log instead.
"""

from __future__ import annotations

import logging
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
_LOGGER = logging.getLogger(__name__)

_OLLAMA_TIMEOUT_SECONDS = 2.0

# The last state logged for each check. The UI polls this endpoint, and a
# warning on every poll while Ollama is down filled the log; now a check logs
# when it goes down and when it comes back.
_last_ok: dict[str, bool] = {}


def _log_transition(name: str, ok: bool, message: str, *args: object) -> None:
    previous = _last_ok.get(name)
    _last_ok[name] = ok
    if ok and previous is False:
        _LOGGER.info("Health check: %s is back.", name)
    elif not ok and previous is not False:
        _LOGGER.warning(message, *args)
    elif not ok:
        _LOGGER.debug(message, *args)


def _check_database() -> dict[str, Any]:
    try:
        pipeline_db._init_db_if_needed()
        with pipeline_db._connect() as conn:
            conn.execute("SELECT 1").fetchone()
    except (sqlite3.Error, OSError) as exc:
        _log_transition("database", False, "Health check: the pipeline database is unavailable: %s: %s", type(exc).__name__, exc)
        return {"ok": False, "detail": "The pipeline database is unavailable."}
    _log_transition("database", True, "")
    return {"ok": True, "detail": "The pipeline database answers."}


def _check_ollama() -> dict[str, Any]:
    host = str(settings.glmocr_ollama_host or "").rstrip("/")
    if not host:
        return {"ok": False, "detail": "No Ollama host is configured (GLMOCR_OLLAMA_HOST)."}
    try:
        response = requests.get(f"{host}/api/version", timeout=_OLLAMA_TIMEOUT_SECONDS)
        response.raise_for_status()
        version = response.json().get("version")
    except (requests.RequestException, ValueError, AttributeError) as exc:
        _log_transition("ollama", False, "Health check: Ollama at %s is not reachable: %s: %s", host, type(exc).__name__, exc)
        return {"ok": False, "detail": "Ollama is not reachable; GLM-OCR is unavailable."}
    _log_transition("ollama", True, "")
    return {"ok": True, "detail": "Ollama answers.", "version": version}


def _check_chat_api_key() -> dict[str, Any]:
    try:
        SaiaClient._resolve_api_key()
    except SaiaConfigError:
        return {"ok": False, "detail": "No chat API key is configured; chat and translation are unavailable."}
    return {"ok": True, "detail": "A chat API key is configured; it is not validated here."}


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
