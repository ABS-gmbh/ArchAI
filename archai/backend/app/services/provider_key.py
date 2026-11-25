"""What the chat provider last said about this server's key.

Found on a user's installation (2026-11-25): the provider refused the
configured key (401 "Unknown API key"), and nothing said so until a question
failed. The model list fell back to configured names, health said a key was
configured, and every page went unindexed while search_index said ok.

Every OpenAI-compatible client the backend makes (chat_ai, saia_client, the
embeddings in rag_store) is built with http_client(), whose response hook
notes the status of each answer: 401 or 403 is "refused", a 2xx "accepted",
anything else leaves the last word as it was (a 429 or a 5xx says nothing
about the key). So the calls the app makes anyway (the model list at workspace
load, chat, translation, embeddings, SAIA OCR) keep it current, and health
reads it without asking the provider. In process memory, behind a lock;
"untried" again after a restart, when the key may have changed.
"""

from __future__ import annotations

import threading
from typing import Any, Literal

KeyState = Literal["untried", "accepted", "refused"]

_lock = threading.Lock()
_state: KeyState = "untried"
# The provider's own model list as last fetched (chat_ai, saia_client), or None.
_offered: tuple[str, ...] | None = None


def state() -> KeyState:
    with _lock:
        return _state


def observe_status(status: int) -> None:
    """Note one answer from the provider: 401/403 refused, 2xx accepted, others nothing."""
    global _state
    if status in (401, 403):
        new: KeyState | None = "refused"
    elif 200 <= status < 300:
        new = "accepted"
    else:
        new = None
    if new is not None:
        with _lock:
            _state = new


def reset() -> None:
    global _state, _offered
    with _lock:
        _state = "untried"
        _offered = None


def record_offered(models: list[str]) -> None:
    """The provider's model list, as a call the app made anyway fetched it."""
    global _offered
    with _lock:
        _offered = tuple(models)


def offered() -> tuple[str, ...] | None:
    with _lock:
        return _offered


def forget_offered() -> None:
    global _offered
    with _lock:
        _offered = None


# A 403 that names a model is the provider declining that model for this key
# ("Not authorized to access this Model: llama-3.3-70b-instruct"), not the key
# itself: the key works for the models it does offer (2026-11-25).
_MODEL_NOT_OFFERED = ("not authorized to access this model", "model is not available", "model not offered")


def _on_response(response: Any) -> None:
    status = int(getattr(response, "status_code", 0) or 0)
    if status == 403:
        try:
            response.read()
            text = str(response.text or "").lower()
        except Exception:
            text = ""
        if "model" in text:
            return
    observe_status(status)


def is_model_not_offered(exc: BaseException) -> bool:
    """Whether *exc* is the provider declining the requested model: a 403 that says so. A 401 never is."""
    if getattr(exc, "status_code", None) != 403:
        return False
    text = str(exc).lower()
    return any(phrase in text for phrase in _MODEL_NOT_OFFERED)


def http_client() -> Any:
    """The http client for an OpenAI-compatible client: the SDK's defaults, and the hook above.

    None (the SDK then builds its own, unobserved) where the openai package has
    no DefaultHttpxClient, as with the stand-ins some tests install.
    """
    try:
        from openai import DefaultHttpxClient
    except ImportError:
        return None
    return DefaultHttpxClient(event_hooks={"response": [_on_response]})


def is_refusal(exc: BaseException) -> bool:
    """Whether *exc*, or one of its causes, is the provider refusing the key."""
    import openai

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return True
        current = current.__cause__ or current.__context__
    return False
