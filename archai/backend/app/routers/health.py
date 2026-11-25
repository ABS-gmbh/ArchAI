"""Health check: is the process up, and can it reach what it depends on.

``status`` is "ok" when every check passes, "degraded" when only an optional
dependency is down (the app still serves, without that feature), and
"unavailable" with HTTP 503 when the pipeline database cannot be used, since
no OCR run can be recorded without it.

- ``database`` (required): the pipeline SQLite database answers a query.
- ``ollama`` (optional): the GLM-OCR host answers /api/version. Without it the
  GLM-OCR engine and the segmented engine's fallback fail.
- ``chat_api_key`` (optional): a SAIA / Chat AI key is configured, and what
  the provider last answered about it on the calls the app makes anyway
  (provider_key.py): refused (401/403), accepted, or not asked yet. A refused
  key is "degraded". Health sends nothing to the provider, and never returns
  the key, the host or the provider's text.
- ``layout_models`` (optional): the three YOLO layout weights are in MODEL_DIR
  and ultralytics is installed. Without them the segmented engine cannot read
  a page (GLM-OCR still can) and /api/predict fails.
- ``line_recognizer`` (optional): the CATMuS Medieval Kraken recogniser is
  where KRAKEN_CATMUS_MODEL_PATH points and kraken is installed.
- ``search_index`` (optional): pages read can be indexed for search. The
  default RAG_EMBEDDING_MODEL is a provider model reached with the chat key;
  without a key every page was read and not indexed, and only each run's
  Steps said so (live regression, 2026-11-24). RAG_EMBEDDING_MODEL= (empty)
  indexes with ChromaDB's local embedder instead, in collections of its own
  (rag_store: the two spaces never mix, so pages indexed in one are not found
  from the other until read again).

``limits`` are the request limits a client needs to refuse a page before
sending it: the JSON body limit in bytes (MAX_JSON_MEGABYTES) and the most
pixels an image may have (OCR_MAX_DECODE_PIXELS). The workspace sent an 85 MB
page before the server refused it as over 48 MB (live regression, 2026-11-24).

``models_loaded`` is true when both model checks pass: everything the
segmented engine needs is installed. The models themselves are loaded on first
use. (It used to report whether a worker process pool existed, which said
nothing about the weights.)

The endpoint is unauthenticated, so each check returns only ``ok``, a fixed
``detail`` (the model checks name missing files and packages, never paths)
and, for Ollama, the version. Exception text (a SQLite error can
carry the database path) and the Ollama host go to the log instead.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from typing import Any

import requests
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import pipeline_db
from app.body_limit import max_json_bytes
from app.services import model_files, provider_key
from app.services.model_router import (
    TASK_MODEL_SETTINGS,
    get_task_model_assignments,
    second_model_for_checking,
    task_models_not_offered,
)
from app.services.image_limits import max_decode_pixels
from app.services.saia_client import SaiaClient, SaiaConfigError

router = APIRouter(tags=["health"])

CHAT_KEY_REFUSED = (
    "The chat provider refused this server's key: chat and translation are unavailable until it is replaced."
)
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


# The Ollama answer is reused for this long. The endpoint is tokenless and the
# check is a blocking request with a 2 s timeout, so each call held a worker
# thread: 200 calls against a host that never answers made an authenticated
# request wait 9.9 s (security review C5). Now one call at a time asks, the
# others get the last answer at once, and the answer is kept a few seconds.
_OLLAMA_CACHE_SECONDS = 5.0
_ollama_lock = threading.Lock()
# (host, monotonic time of the answer, the check's result)
_ollama_cache: tuple[str, float, dict[str, Any]] | None = None


def _check_ollama() -> dict[str, Any]:
    global _ollama_cache
    host = str(settings.glmocr_ollama_host or "").rstrip("/")
    if not host:
        return {"ok": False, "detail": "No Ollama host is configured (GLMOCR_OLLAMA_HOST)."}
    cached = _ollama_cache if _ollama_cache is not None and _ollama_cache[0] == host else None
    if cached is not None and time.monotonic() - cached[1] < _OLLAMA_CACHE_SECONDS:
        return cached[2]
    if not _ollama_lock.acquire(blocking=False):
        # Another call is asking Ollama now: answer with what is known.
        if cached is not None:
            return cached[2]
        return {"ok": False, "detail": "Ollama is being checked; ask again in a moment."}
    try:
        result = _ask_ollama(host)
        _ollama_cache = (host, time.monotonic(), result)
        return result
    finally:
        _ollama_lock.release()


def _ask_ollama(host: str) -> dict[str, Any]:
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
    # What the provider last answered about the key, from the calls the app
    # makes anyway (provider_key.py): health itself sends nothing.
    said = provider_key.state()
    if said == "refused":
        return {"ok": False, "detail": CHAT_KEY_REFUSED}
    if said == "accepted":
        return {"ok": True, "detail": "A chat API key is configured and the provider accepted it."}
    return {"ok": True, "detail": "A chat API key is configured; the provider has not been asked yet."}


def _check_layout_models() -> dict[str, Any]:
    missing = model_files.missing_layout()
    if missing:
        return {
            "ok": False,
            "detail": "Pages cannot be read with the segmented engine: the layout models are not installed "
            f"({', '.join(missing)}). See docs/model_setup.md.",
        }
    return {"ok": True, "detail": "The layout models are installed."}


def _check_line_recognizer() -> dict[str, Any]:
    missing = model_files.missing_recogniser()
    if missing:
        return {
            "ok": False,
            "detail": "Lines cannot be read with Kraken: the line recogniser is not installed "
            f"({', '.join(missing)}). See docs/model_setup.md.",
        }
    return {"ok": True, "detail": "The line recogniser is installed."}


def _check_search_index() -> dict[str, Any]:
    if not settings.rag_auto_index:
        return {"ok": True, "detail": "Pages are not indexed automatically (RAG_AUTO_INDEX is off)."}
    if not str(settings.rag_embedding_model or "").strip():
        return {"ok": True, "detail": "Pages are indexed with the local embedder."}
    try:
        SaiaClient._resolve_api_key()
    except SaiaConfigError:
        return {
            "ok": False,
            "detail": "Pages are read but not indexed for search: the embedding model needs the chat key. Set the key, "
            "or set RAG_EMBEDDING_MODEL= to index with the local embedder (see docs/deployment.md).",
        }
    if provider_key.state() == "refused":
        return {
            "ok": False,
            "detail": "Pages are read but not indexed for search: the chat provider refused this server's key, which the "
            "embedding model needs. Replace the key, or set RAG_EMBEDDING_MODEL= to index with the local embedder "
            "(see docs/deployment.md).",
        }
    # Said from the same remembered state as chat_api_key, so the two agree: it
    # said "the key is not validated here" next to "the provider accepted it".
    # The embedding model is not checked against the provider's model list:
    # multilingual-e5-large-instruct, the default, is not in that list and
    # answers anyway (measured 2026-11-25), so a list check would call a
    # working setup broken.
    if provider_key.state() == "accepted":
        return {"ok": True, "detail": "Pages are indexed with the provider's embedding model; the provider accepted the key."}
    return {
        "ok": True,
        "detail": "Pages are indexed with the provider's embedding model; the provider has not been asked about the key yet.",
    }


# What each provider-served task is called, and who its readers are, for the
# sentences below. The setting names are model_router.TASK_MODEL_SETTINGS.
_TASK_WORDS = {
    "chat_rag_model": ("chat model", "questions use"),
    "translation_model": ("translation model", "translations use"),
    "label_visual_model": ("label-analysis model", "label analysis uses"),
    "label_visual_fallback_model": ("second label-analysis model", "label analysis uses"),
    "verifier_model": ("answer-checking model", ""),
}


def _instead(task: str, model: str, offered: tuple[str, ...]) -> str:
    """What the task does instead, as its own code path would choose (chat_ai, label_analysis_agent)."""
    from app.agents.label_analysis_agent import label_retry_model
    from app.services.chat_ai import _choose_retry_model

    chat_model = get_task_model_assignments().chat_rag_model
    if task == "verifier_model":
        # A second offered model checks, never the one that wrote the answer
        # (model_router.second_model_for_checking); only without one does the
        # chat model check itself, which is weaker, and the sentence says so.
        second = second_model_for_checking(chat_model, offered, exclude=[model])
        if second:
            return f"answers are checked by {second} instead"
        if chat_model.lower() in {m.lower() for m in offered}:
            return "answers are checked by the same model that wrote them, which is weaker than a second model"
        return "answers are given without that check"
    if task in ("label_visual_model", "label_visual_fallback_model"):
        used = label_retry_model(model, list(offered), prefer_vision=True)
    else:
        stage = {"chat_rag_model": "rag_chat", "translation_model": "translation"}[task]
        used = _choose_retry_model(model, list(offered), False, stage=stage)
    uses = _TASK_WORDS[task][1]  # "translations use", "label analysis uses"
    if not used:
        return f"{uses.rsplit(' ', 1)[0]} cannot be done until it is replaced"
    if used == chat_model:
        return f"{uses} the chat model instead"
    return f"{uses} {used} instead"


def _check_task_models() -> dict[str, Any]:
    # From the provider's own list as the app last fetched it (provider_key.py):
    # health asks the provider nothing.
    offered = provider_key.offered()
    if offered is None:
        return {"ok": True, "detail": "The provider's models have not been listed yet."}
    missing = task_models_not_offered(offered)
    if not missing:
        return {"ok": True, "detail": "The provider offers every configured model."}
    sentences = [
        f"The configured {_TASK_WORDS[task][0]} is not offered by the provider; {_instead(task, model, offered)}. "
        f"Set {TASK_MODEL_SETTINGS[task]} to one of the provider's models."
        for task, model in missing.items()
    ]
    return {"ok": False, "detail": " ".join(sentences)}


# A plain def: the checks are a SQLite query and an HTTP request, which FastAPI
# runs in its threadpool instead of on the event loop.
@router.get("/health")
def health_check() -> JSONResponse:
    checks = {
        "database": _check_database(),
        "ollama": _check_ollama(),
        "chat_api_key": _check_chat_api_key(),
        "layout_models": _check_layout_models(),
        "line_recognizer": _check_line_recognizer(),
        "search_index": _check_search_index(),
        "task_models": _check_task_models(),
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
            "models_loaded": checks["layout_models"]["ok"] and checks["line_recognizer"]["ok"],
            "checks": checks,
            "limits": {"json_bytes": max_json_bytes(), "decode_pixels": max_decode_pixels()},
        },
    )
