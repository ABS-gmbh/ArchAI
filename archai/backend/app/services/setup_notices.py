"""Log a setup problem once per failure streak, not on every request.

Without a chat key every question, translation and label analysis fails the
same way. The reader is told only that chat is not set up; which settings to
set goes to the server log, once, until the key is found again (the next
failure after that starts a new streak and is logged again).
"""

from __future__ import annotations

import logging
import threading

_LOGGER = logging.getLogger("app.setup")
# The chat provider's key (chat_ai, saia_client), shared by chat, translation,
# label analysis and the vision OCR agents.
CHAT_KEY = "chat key"
_lock = threading.Lock()
_reported: set[str] = set()


def report_once(problem: str, message: str) -> None:
    """Log *message* as a WARNING unless *problem* was already reported in this streak."""
    with _lock:
        if problem in _reported:
            return
        _reported.add(problem)
    _LOGGER.warning("%s", message)


def resolved(problem: str) -> None:
    """*problem* is gone (a key was found): its next failure is logged again."""
    with _lock:
        _reported.discard(problem)


def reset() -> None:
    with _lock:
        _reported.clear()
