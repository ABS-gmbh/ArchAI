"""Turn provider-side thinking off for the models that think by default.

Measured against the provider (2026-11-25): qwen3.5-397b, qwen3.6-35b,
qwen3.8-27b and glm-5.3 think before they answer unless told not to. With
thinking on they returned empty text at max_tokens 200 and took 10 to 48 s;
with extra_body={"chat_template_kwargs": {"enable_thinking": False}}
qwen3.6-35b answered an image question in 0.5 s and qwen3.5-397b in 1.3 s.

Every chat completion the backend asks for goes through create() below: the
chat answer (streamed and not), translation, the verifier, and through
saia_client label analysis, the OCR proofreader and the provider OCR. While
CHAT_DISABLE_THINKING is on, every model is sent the kwarg except one whose
name starts with a prefix in CHAT_DISABLE_THINKING_EXCEPT, so a model the
provider adds later gets thinking off without a release. Asked across the
provider's list (28 requests, 2026-11-25), models that do not think accept
the kwarg and ignore it; glm-5.3 accepts it and keeps thinking; only
mistral-large-4-1t-a52b refuses it, with 400 "chat_template is not
supported", hence the default exception "mistral-large".

A model the list does not name may still refuse it. On a 400 that names
chat_template the call is made once more without the kwarg, and when that
succeeds the model is remembered for the rest of the process and not sent it
again, so a provider change cannot break chat. Any other 400 is raised as it
came, without a second call.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from app.config import settings

_KWARG = "chat_template_kwargs"
_THINKING_OFF = {"enable_thinking": False}

_lock = threading.Lock()
# Models whose provider answered the kwarg with 400 and then answered without it.
_rejected: set[str] = set()


def _excepted() -> tuple[str, ...]:
    raw = str(getattr(settings, "chat_disable_thinking_except", "") or "")
    return tuple(p.strip().lower() for p in raw.split(",") if p.strip())


def applies(model: str) -> bool:
    """Whether a call to *model* is sent the thinking-off kwarg."""
    if not bool(getattr(settings, "chat_disable_thinking", False)):
        return False
    name = str(model or "").strip().lower()
    if not name:
        return False
    with _lock:
        if name in _rejected:
            return False
    excepted = _excepted()
    return not (excepted and name.startswith(excepted))


def rejected() -> set[str]:
    with _lock:
        return set(_rejected)


def reset() -> None:
    with _lock:
        _rejected.clear()


def _with_kwarg(kwargs: dict[str, Any]) -> dict[str, Any]:
    extra = dict(kwargs.get("extra_body") or {})
    extra[_KWARG] = dict(_THINKING_OFF)
    return {**kwargs, "extra_body": extra}


def _without_kwarg(kwargs: dict[str, Any]) -> dict[str, Any]:
    extra = {k: v for k, v in dict(kwargs.get("extra_body") or {}).items() if k != _KWARG}
    out = {k: v for k, v in kwargs.items() if k != "extra_body"}
    if extra:
        out["extra_body"] = extra
    return out


def _refuses_the_kwarg(exc: BaseException) -> bool:
    """A 400 that names the chat template: the provider refusing the kwarg for this model."""
    return getattr(exc, "status_code", None) == 400 and "chat_template" in str(exc).lower()


def create(create_fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Call *create_fn* (a client's chat.completions.create) with thinking off where it applies."""
    model = str(kwargs.get("model") or "")
    if not applies(model):
        return create_fn(**kwargs)
    try:
        return create_fn(**_with_kwarg(kwargs))
    except Exception as exc:
        if not _refuses_the_kwarg(exc):
            raise
    response = create_fn(**_without_kwarg(kwargs))
    with _lock:
        _rejected.add(model.strip().lower())
    return response
