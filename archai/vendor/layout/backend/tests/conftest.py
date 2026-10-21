"""Test-wide guards for the backend suite.

No test may reach the network: Wikidata, SAIA, GWDG and Ollama are stubbed in
the tests that use them, and one that is not would pass or fail with the
service rather than with the code. Outside connections raise
``NetworkAccessInTest``; loopback and Unix sockets are allowed.
"""

from __future__ import annotations

import socket

import pytest

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

