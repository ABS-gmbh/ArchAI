"""Failed-login limit for the analytics login, in process memory.

Two limits apply to every attempt, within ANALYTICS_LOGIN_WINDOW_SECONDS:
ANALYTICS_LOGIN_MAX_FAILURES failures from one client address, and
ANALYTICS_LOGIN_MAX_FAILURES_PER_USERNAME failures for one user name from any
address. Past either, attempts are refused with 429 until the oldest failure
leaves the window, even with the right password. A successful login clears both.

The user-name limit is the one an attacker cannot step around: a client that
changes its address for every guess (a botnet, IPv6, or a forwarded-for header
someone trusts) still meets it. Its cost is that the same guesses lock the
operator out of analytics until the window passes (docs/deployment.md).

An attempt is counted as a failure when it is checked, in one step under the
lock, and the count is taken back if the password was right: checking first
and recording after let about 40 parallel guesses through before the first
failure was recorded. The table is ordered by each key's last failure, so
expired keys leave from the front without a scan, and it holds at most
max_entries keys; past that the key whose last failure is oldest is dropped.

In process memory, like the rest of the backend's state: one worker, one
limiter (docs/deployment.md).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping

_MAX_ENTRIES = 10_000


class FailedLoginLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic, *, max_entries: int = _MAX_ENTRIES) -> None:
        self._clock = clock
        self._max_entries = max(1, int(max_entries))
        self._failures: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._failures)

    def attempt(self, limits: Mapping[str, int], *, window_seconds: float) -> float | None:
        """Count an attempt against each key's limit, or refuse it.

        *limits* maps each key (an address, a user name) to the failures it may
        have in the window. Returns None when the attempt may go ahead (it is
        then counted as a failure until ``succeeded`` takes it back), or the
        seconds until every key that is full has room again.
        """
        now = self._clock()
        with self._lock:
            self._expire(now, window_seconds)
            waits = [
                failures[0] + window_seconds - now
                for key, max_failures in limits.items()
                if len(failures := self._recent(key, now, window_seconds)) >= max_failures
            ]
            if waits:
                return max(0.0, max(waits))
            for key in limits:
                self._failures.setdefault(key, deque()).append(now)
                self._failures.move_to_end(key)
            while len(self._failures) > self._max_entries:
                self._failures.popitem(last=False)
            return None

    def succeeded(self, keys: Iterable[str]) -> None:
        """The attempt was right: clear these keys, the attempt's own count included."""
        with self._lock:
            for key in keys:
                self._failures.pop(key, None)

    def _recent(self, key: str, now: float, window_seconds: float) -> deque[float]:
        failures = self._failures.get(key)
        if failures is None:
            return deque()
        while failures and failures[0] <= now - window_seconds:
            failures.popleft()
        return failures

    def _expire(self, now: float, window_seconds: float) -> None:
        # Ordered by last failure: the first key still in the window ends the sweep.
        while self._failures:
            key, failures = next(iter(self._failures.items()))
            if failures and failures[-1] > now - window_seconds:
                break
            del self._failures[key]
