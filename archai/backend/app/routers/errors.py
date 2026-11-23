"""How routes turn an exception they did not expect into a response.

The UI shows ``detail`` verbatim, and the routes used to put the exception's
text there: a file path, an upstream response body, a host. Now the
exception goes to the log and the client gets a sentence it can act on:

* the provider rejected the API key: 502;
* the provider did not answer in time: 504;
* the provider is rate-limiting: 429;
* anything else: 500, naming what failed and that the details are logged.

Configuration errors and HTTPExceptions are handled by each route before it
calls :func:`unexpected_error`, so their messages and status codes stay.
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
    """The response for a provider failure a person can act on, or None for anything else."""
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return HTTPException(status_code=502, detail=f"The {provider} rejected the API key.")
    if isinstance(exc, _TIMEOUTS):
        return HTTPException(status_code=504, detail=f"The {provider} did not answer in time. Try again.")
    if isinstance(exc, openai.RateLimitError):
        return HTTPException(
            status_code=429, detail=f"The {provider} is limiting requests right now. Try again in a minute."
        )
    return None


def unexpected_error(exc: BaseException, *, what: str, provider: str | None = None) -> NoReturn:
    """Log *exc* and raise the response for it; *what* names the failed operation, as "Indexing"."""
    if provider is not None:
        known = provider_failure(exc, provider=provider)
        if known is not None:
            _LOGGER.warning("%s failed: %s: %s", what, type(exc).__name__, exc)
            raise known from exc
    _LOGGER.error("%s failed", what, exc_info=exc)
    raise HTTPException(status_code=500, detail=f"{what} failed on the server. The details are in the server log.") from exc
