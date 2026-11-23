"""Failed-login limit for the analytics login, per client address, in process memory.

After ANALYTICS_LOGIN_MAX_FAILURES failed logins from one address within
ANALYTICS_LOGIN_WINDOW_SECONDS, further attempts from it are refused with 429
until the oldest failure leaves the window, even with the right password. A
successful login clears the address. The address is the one the app sees:
behind a reverse proxy, uvicorn must be told to trust the proxy's
X-Forwarded-For (--forwarded-allow-ips), or every client shares the proxy's
address; otherwise let the proxy do the limiting (docs/deployment.md).

In process memory, like the rest of the backend's state: one worker, one
limiter (docs/deployment.md).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

# Addresses with no failure left in the window are dropped once the table is
# this large, so a scan from many addresses cannot grow it without bound.
_PRUNE_AT = 10_000


class FailedLoginLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._failures: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def retry_after(self, address: str, *, max_failures: int, window_seconds: float) -> float | None:
        """Seconds until *address* may try again, or None if it may try now."""
        now = self._clock()
        with self._lock:
            failures = self._recent(address, now, window_seconds)
            if len(failures) < max_failures:
                return None
            return max(0.0, failures[0] + window_seconds - now)

    def record_failure(self, address: str, *, window_seconds: float) -> None:
        now = self._clock()
        with self._lock:
            self._recent(address, now, window_seconds).append(now)
            if len(self._failures) > _PRUNE_AT:
                for key in [k for k, v in self._failures.items() if not v or v[-1] <= now - window_seconds]:
                    del self._failures[key]

    def reset(self, address: str) -> None:
        with self._lock:
            self._failures.pop(address, None)

    def _recent(self, address: str, now: float, window_seconds: float) -> deque[float]:
        failures = self._failures.setdefault(address, deque())
        while failures and failures[0] <= now - window_seconds:
            failures.popleft()
        return failures
