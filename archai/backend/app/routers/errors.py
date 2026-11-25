"""How routes turn an exception they did not expect into a response.

The UI shows ``detail`` verbatim, and the routes used to put the exception's
text there: a file path, an upstream response body, a host. Now the
exception goes to the log and the client gets a sentence it can act on:

* the provider is not set up here (no key): the routes' own status, and a
  sentence for the reader; the settings to set go to the log, once per
  failure streak (app/services/setup_notices.py);
* the provider refused this server's key: 502;
* the provider did not answer in time: 504;
* the provider is rate-limiting: 429, with the wait it asked for (Retry-After) in words and in the header;
* the model spent its whole answer on thinking (provider_thinking): 502;
* anything else: 500, naming what failed and that the details are logged.

Routes that already answer configuration errors with their own status keep
doing so, with the reader's sentence; HTTPExceptions pass through. The
configuration message itself ("Chat API key not configured. Set
CHAT_AI_API_KEY ...") was shown to readers until the live regression of
2026-11-24; it names settings only an operator can change.
"""

from __future__ import annotations

import logging
import socket
from typing import NoReturn

import openai
import requests
from fastapi import HTTPException

from app.services.provider_thinking import ThinkingOnlyAnswer
from app.services.public_errors import rate_limited

_LOGGER = logging.getLogger("app.routers")

_TIMEOUTS: tuple[type[BaseException], ...] = (
    openai.APITimeoutError,
    requests.Timeout,
    socket.timeout,
    TimeoutError,
)


NOT_SET_UP = "Chat is not set up on this server. Tell whoever runs it."
REFUSED_KEY = "The chat provider refused this server's key. Tell whoever runs it."
_NOT_SET_UP_BY_PROVIDER = {
    "chat provider": NOT_SET_UP,
    "OCR provider": "Reading with the OCR provider is not set up on this server. Tell whoever runs it.",
    "embedding provider": "Search is not set up on this server. Tell whoever runs it.",
}


def _refused_key(provider: str) -> str:
    return REFUSED_KEY if provider == "chat provider" else f"The {provider} refused this server's key. Tell whoever runs it."


def provider_failure(exc: BaseException, *, provider: str) -> HTTPException | None:
    """The response for a provider failure a person can act on, or None for anything else.

    The agents wrap provider errors in their own ("OCR failed on model ...:
    {exc}"), so the chain of causes is searched, not just *exc*.
    """
    for current in _chain(exc):
        if isinstance(current, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return HTTPException(status_code=502, detail=_refused_key(provider))
        if isinstance(current, _TIMEOUTS):
            return HTTPException(status_code=504, detail=f"The {provider} did not answer in time. Try again.")
        if isinstance(current, ThinkingOnlyAnswer):
            return HTTPException(
                status_code=502,
                detail=f"The model {current.model} spent its whole answer on thinking and gave no answer. "
                "Ask again, or choose another model.",
            )
        if isinstance(current, openai.RateLimitError):
            sentence, wait = rate_limited(current, provider=provider) or ("", None)
            # The provider's own wait, when it gave one, for the reader and for clients that read the header.
            headers = {"Retry-After": str(wait)} if wait is not None else None
            return HTTPException(status_code=429, detail=sentence, headers=headers)
    return None


def _chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def configuration_message(exc: BaseException, *, provider: str = "chat provider") -> str | None:
    """The reader's sentence for a configuration error in *exc*'s chain, or None.

    The error's own message ("Chat API key not configured. Set CHAT_AI_API_KEY
    ...") names the settings to set: it goes to the server log once per
    failure streak, not to the reader.
    """
    from app.services import setup_notices
    from app.services.chat_ai import ChatConfigError
    from app.services.saia_client import SaiaConfigError

    for current in _chain(exc):
        if isinstance(current, (SaiaConfigError, ChatConfigError)):
            setup_notices.report_once(setup_notices.CHAT_KEY, str(current))
            return _NOT_SET_UP_BY_PROVIDER.get(provider, f"The {provider} is not set up on this server. Tell whoever runs it.")
    return None


def unexpected_error(exc: BaseException, *, what: str, provider: str | None = None) -> NoReturn:
    """Log *exc* and raise the response for it; *what* names the failed operation, as "Indexing"."""
    configuration = configuration_message(exc, provider=provider or "chat provider")
    if configuration is not None:
        raise HTTPException(status_code=500, detail=configuration) from exc
    if provider is not None:
        known = provider_failure(exc, provider=provider)
        if known is not None:
            _LOGGER.warning("%s failed: %s: %s", what, type(exc).__name__, exc)
            raise known from exc
    _LOGGER.error("%s failed", what, exc_info=exc)
    raise HTTPException(status_code=500, detail=f"{what} failed on the server. The details are in the server log.") from exc


def stream_error_message(exc: BaseException, *, provider: str = "chat provider") -> str:
    """The sentence for an error inside a response that is already streaming."""
    configuration = configuration_message(exc, provider=provider)
    if configuration is not None:
        return configuration
    known = provider_failure(exc, provider=provider)
    if known is not None:
        _LOGGER.warning("Streamed chat failed: %s: %s", type(exc).__name__, exc)
        return str(known.detail)
    _LOGGER.error("Streamed chat failed", exc_info=exc)
    return "The chat answer failed on the server. The details are in the server log."
