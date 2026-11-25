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
import math

_LOGGER = logging.getLogger("app.errors")


class RequestRefused(ValueError):
    """A request this code base refuses, raised with a fixed sentence the client is shown as it is.

    The routes that answer a ValueError's text as their 422 or 404 catch this
    instead: a ValueError raised by a library inside the call (PIL, pydantic
    building an internal model) carried its own text - values, paths - to the
    client. Now it falls through to the route's fixed-sentence path and its
    text goes to the log (second security review). A ValueError subclass, so
    code that catches ValueError around these calls still catches it.
    """


def public_failure(what: str, exc: BaseException, *, logger: logging.Logger | None = None) -> str:
    """Log *exc* with its traceback and return a sentence for the client: "<what> failed (<Type>). ..."."""
    from app.routers.errors import configuration_message

    configuration = configuration_message(exc)
    if configuration is not None:
        return configuration
    (logger or _LOGGER).warning("%s failed", what, exc_info=exc)
    return f"{what} failed ({type(exc).__name__}). The details are in the server log."


def retry_after_wait(response: object) -> float | None:
    """The wait a response's Retry-After header asks for (seconds or an HTTP date), 0 for "now", or None."""
    headers = getattr(response, "headers", None)
    raw = str(headers.get("retry-after") or "").strip() if headers is not None else ""
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        from datetime import datetime, timezone
        from email.utils import parsedate_to_datetime

        try:
            seconds = (parsedate_to_datetime(raw) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError):
            return None
    if seconds != seconds or seconds == float("inf"):  # NaN, infinity
        return None
    return max(0.0, seconds)


def _retry_after_seconds(response: object) -> int | None:
    """The wait to tell the reader, in whole seconds, or None when the provider gave none worth saying."""
    seconds = retry_after_wait(response)
    if seconds is None or seconds <= 0:
        return None
    return max(1, math.ceil(seconds))


def _wait_in_words(seconds: int) -> str:
    if seconds < 90:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    minutes = math.ceil(seconds / 60)
    if minutes < 90:
        return f"{minutes} minutes"
    hours = math.ceil(minutes / 60)
    return f"{hours} hours"


def rate_limited(exc: BaseException, *, provider: str = "chat provider") -> tuple[str, int | None] | None:
    """The reader's sentence for a provider's 429 in *exc*'s chain, and its wait in seconds; None if there is none.

    The provider limits each user (10 requests in 10 s, 400 an hour). When it
    says how long to wait (Retry-After), the sentence says it; otherwise it
    says to try again in a minute, as before.
    """
    import openai

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, openai.RateLimitError):
            wait = _retry_after_seconds(getattr(current, "response", None))
            when = _wait_in_words(wait) if wait is not None else "a minute"
            return f"The {provider} is limiting requests right now. Try again in {when}.", wait
        current = current.__cause__ or current.__context__
    return None
