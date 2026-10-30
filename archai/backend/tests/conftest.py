"""Test-wide guards for the backend suite.

No test may reach the network: Wikidata, SAIA, GWDG and Ollama are stubbed in
the tests that use them, and one that is not would pass or fail with the
service rather than with the code. Outside connections raise
``NetworkAccessInTest``; loopback and Unix sockets are allowed.

Nor may a test write the OCR evidence log: every OCR run appends to it, and the
suite's fake pages went into the copy in the checkout. It goes to a file of the
session's own.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.config import settings  # noqa: E402

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


class NetworkAccessInTest(RuntimeError):
    pass


def _check(sock: socket.socket, address: object) -> None:
    if sock.family == getattr(socket, "AF_UNIX", None):
        return
    host = address[0] if isinstance(address, tuple) else address
    if str(host) not in _LOOPBACK:
        raise NetworkAccessInTest(f"a test tried to connect to {address!r}; stub the client instead")


def _connect(self: socket.socket, address: object) -> None:
    _check(self, address)
    return _real_connect(self, address)


def _connect_ex(self: socket.socket, address: object) -> int:
    _check(self, address)
    return _real_connect_ex(self, address)


@pytest.fixture(autouse=True, scope="session")
def _no_network():
    socket.socket.connect = _connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _connect_ex  # type: ignore[method-assign]
    yield
    socket.socket.connect = _real_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _real_connect_ex  # type: ignore[method-assign]



@pytest.fixture(autouse=True, scope="session")
def _evidence_log_outside_the_checkout(tmp_path_factory: pytest.TempPathFactory):
    real = settings.ocr_evidence_path
    settings.ocr_evidence_path = str(tmp_path_factory.mktemp("evidence") / "ocr_evidence.jsonl")
    yield
    settings.ocr_evidence_path = real


@pytest.fixture(autouse=True, scope="session")
def _authority_cache_outside_the_checkout(tmp_path_factory: pytest.TempPathFactory):
    """Tests feed the linker fake candidates, which it caches: keep them out of app/.data."""
    import os

    from app.services import wikidata_client

    real = os.environ.get("ARCHAI_WIKIDATA_CACHE")
    os.environ["ARCHAI_WIKIDATA_CACHE"] = str(tmp_path_factory.mktemp("authority_cache") / "cache.sqlite")
    wikidata_client._CACHE_READY = False
    yield
    if real is None:
        os.environ.pop("ARCHAI_WIKIDATA_CACHE", None)
    else:
        os.environ["ARCHAI_WIKIDATA_CACHE"] = real
    wikidata_client._CACHE_READY = False
