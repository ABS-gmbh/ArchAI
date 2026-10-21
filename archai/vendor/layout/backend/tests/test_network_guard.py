"""The suite cannot reach the network: see tests/conftest.py."""

from __future__ import annotations

import socket

import pytest


def test_an_outside_connection_is_refused() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(RuntimeError, match="stub the client"):
            sock.connect(("192.0.2.1", 80))  # TEST-NET-1: never routed


def test_loopback_is_allowed() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=2):
            pass
