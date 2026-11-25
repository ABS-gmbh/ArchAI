"""When a call to the chat provider is tried again: decided here, once, for every client.

Measured against a stand-in (2026-11-25): the OpenAI SDK retried every 429
and 5xx twice by itself, waiting what Retry-After asked, so a 429 cost three
calls and, with Retry-After: 6, 12 s before the reader heard anything; on top
of that the chat answer and label analysis retried a 502 themselves, and the
chat answer then tried a second model the same way: 18 calls for one
question. Worker 2 counted 38 and 60 unrequested extra chat calls in two runs
of 79 questions, all against the provider's 400-an-hour budget per user.

Every OpenAI-compatible client the backend builds (chat_ai, saia_client, the
embeddings in rag_store) now has max_retries=0, and every call to the
provider goes through call() below (chat completions through
provider_thinking.create, the embeddings, the model lists). It makes the
call again at most once:

* after a 429 whose Retry-After asks for at most
  PROVIDER_RETRY_MAX_WAIT_SECONDS (default 5: the 10-requests-in-10-s
  limit), after that wait; a longer wait, or none given (the hourly cap),
  is raised at once, and the reader is told how long to wait
  (public_errors.rate_limited);
* after a 5xx, or a connection that could not be made, after a short pause;
* after nothing else: a timeout (the request may still be running) or any
  other status is raised as it came.

A streamed create raises on the status before it returns the stream, so the
same wrapper covers it; once chunks arrive nothing is sent again.

The decision is made here, around the call, not in an httpx transport: a
transport passed to the http client switches off its HTTP_PROXY /
HTTPS_PROXY handling (httpx: `allow_env_proxies = trust_env and transport is
None`), and a server that reaches the provider only through a proxy would
stop reaching it (tests/test_provider_proxy.py).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from app.config import settings

log = logging.getLogger(__name__)

T = TypeVar("T")

# The pause before a 5xx or a refused connection is tried again.
_PAUSE_SECONDS = 0.5


def _wait_before_retry(exc: BaseException) -> float | None:
    """Seconds to wait before making the failed call again, or None not to."""
    import openai

    from app.services.public_errors import retry_after_wait

    if isinstance(exc, openai.RateLimitError):
        wait = retry_after_wait(getattr(exc, "response", None))
        limit = float(getattr(settings, "provider_retry_max_wait_seconds", 5.0))
        if wait is None or wait > limit:
            return None
        return float(wait)
    if isinstance(exc, openai.APIStatusError) and 500 <= int(getattr(exc, "status_code", 0) or 0) < 600:
        return _PAUSE_SECONDS
    if isinstance(exc, openai.APIConnectionError) and not isinstance(exc, openai.APITimeoutError):
        return _PAUSE_SECONDS
    return None


def call(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """``fn(*args, **kwargs)``, made once more when the provider's failure says that may help."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        wait = _wait_before_retry(exc)
        if wait is None:
            raise
        log.info("provider call failed (%s); trying once more in %.1f s", type(exc).__name__, wait)
    time.sleep(wait)
    return fn(*args, **kwargs)
