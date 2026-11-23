"""How routes turn an exception they did not expect into a response.

The UI shows ``detail`` verbatim, and the routes used to put the exception's
text there: a file path, an upstream response body, a host. Now the
exception goes to the log and the client gets a sentence it can act on:

* the provider rejected the API key: 502;
* the provider did not answer in time: 504;
* the provider is rate-limiting: 429;
* anything else: 500, naming what failed and that the details are logged.

Routes that already answer configuration errors with their own status keep
doing so. One that reaches the catch-all keeps its message (it names the
variable to set and holds no secret), and HTTPExceptions pass through.
"""

from __future__ import annotations

import logging
import socket
from typing import NoReturn

import openai
import requests
from fastapi import HTTPException

_LOGGER = logging.getLogger("app.routers")

_TIMEOUTS: tuple[type[BaseException], ...] = (
    openai.APITimeoutError,
    requests.Timeout,
    socket.timeout,
    TimeoutError,
)


def provider_failure(exc: BaseException, *, provider: str) -> HTTPException | None:
    """The response for a provider failure a person can act on, or None for anything else.

    The agents wrap provider errors in their own ("OCR failed on model ...:
    {exc}"), so the chain of causes is searched, not just *exc*.
    """
    for current in _chain(exc):
        if isinstance(current, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return HTTPException(status_code=502, detail=f"The {provider} rejected the API key.")
        if isinstance(current, _TIMEOUTS):
            return HTTPException(status_code=504, detail=f"The {provider} did not answer in time. Try again.")
        if isinstance(current, openai.RateLimitError):
            return HTTPException(
                status_code=429, detail=f"The {provider} is limiting requests right now. Try again in a minute."
            )
    return None


def _chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def configuration_message(exc: BaseException) -> str | None:
    """The message of a configuration error in *exc*'s chain ("Chat API key not configured. ..."), if any.

    These are written by this code base, name the variable to set and carry no
    secret, so they are passed on as they are.
    """
    from app.services.chat_ai import ChatConfigError
    from app.services.saia_client import SaiaConfigError

    for current in _chain(exc):
        if isinstance(current, (SaiaConfigError, ChatConfigError)):
            return str(current)
    return None


def unexpected_error(exc: BaseException, *, what: str, provider: str | None = None) -> NoReturn:
    """Log *exc* and raise the response for it; *what* names the failed operation, as "Indexing"."""
    configuration = configuration_message(exc)
    if configuration is not None:
        _LOGGER.warning("%s failed: %s", what, configuration)
        raise HTTPException(status_code=500, detail=configuration) from exc
    if provider is not None:
        known = provider_failure(exc, provider=provider)
        if known is not None:
            _LOGGER.warning("%s failed: %s: %s", what, type(exc).__name__, exc)
            raise known from exc
    _LOGGER.error("%s failed", what, exc_info=exc)
    raise HTTPException(status_code=500, detail=f"{what} failed on the server. The details are in the server log.") from exc
