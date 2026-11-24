"""What a client is told when something fails on the server.

Run events, run.error, response warnings, batch progress and fallback reasons
carried the exception's own text: an absolute model path, a missing module's
name, an upstream message (security review C8). The exception now goes to the
log, and the client gets a fixed sentence naming what failed and the type of
the error. Configuration errors this code base writes ("Chat API key not
configured. Set CHAT_AI_API_KEY ...") name the variable to set and hold no
secret, so they are passed on as they are.
"""

from __future__ import annotations

import logging

_LOGGER = logging.getLogger("app.errors")


def public_failure(what: str, exc: BaseException, *, logger: logging.Logger | None = None) -> str:
    """Log *exc* with its traceback and return a sentence for the client: "<what> failed (<Type>). ..."."""
    from app.routers.errors import configuration_message

    configuration = configuration_message(exc)
    if configuration is not None:
        return configuration
    (logger or _LOGGER).warning("%s failed", what, exc_info=exc)
    return f"{what} failed ({type(exc).__name__}). The details are in the server log."
