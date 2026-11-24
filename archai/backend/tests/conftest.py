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


@pytest.fixture(autouse=True, scope="session")
def _live_stores_outside_the_checkout(tmp_path_factory: pytest.TempPathFactory):
    """Point the pipeline DB, the Chroma store and the task directory at temp paths.

    Their defaults are app/archai.sqlite, .data/chroma and .tasks in the
    backend directory - the live data of whoever runs the suite from their
    checkout. Tests that need a store set their own path; any test that forgot
    to (test_chat_api.py and test_entity_decision_spans.py did) wrote into the
    live database and vector store.
    """
    import os

    from app.db import pipeline_db
    from app.services import file_manager

    root = tmp_path_factory.mktemp("stores")
    real_db = os.environ.get("ARCHAI_DB_PATH")
    real_chroma = settings.chroma_persist_dir
    real_tasks = file_manager.TASK_BASE_DIR

    os.environ["ARCHAI_DB_PATH"] = str(root / "archai.sqlite")
    pipeline_db._DB_READY = False
    settings.chroma_persist_dir = str(root / "chroma")
    file_manager.TASK_BASE_DIR = str(root / "tasks")
    yield _store_paths()
    if real_db is None:
        os.environ.pop("ARCHAI_DB_PATH", None)
    else:
        os.environ["ARCHAI_DB_PATH"] = real_db
    pipeline_db._DB_READY = False
    settings.chroma_persist_dir = real_chroma
    file_manager.TASK_BASE_DIR = real_tasks
    rag_store = sys.modules.get("app.services.rag_store")
    if rag_store is not None:
        rag_store._CLIENT = None


def _store_paths() -> dict[str, str | None]:
    import os

    from app.services import file_manager

    return {
        "ARCHAI_DB_PATH": os.environ.get("ARCHAI_DB_PATH"),
        "settings.chroma_persist_dir": settings.chroma_persist_dir,
        "file_manager.TASK_BASE_DIR": file_manager.TASK_BASE_DIR,
    }


@pytest.fixture(autouse=True)
def _stores_stay_redirected(_live_stores_outside_the_checkout: dict[str, str | None]):
    """Fail the test that leaves a store path changed, rather than the ones after it.

    characterise_ocr_router.py removed ARCHAI_DB_PATH when it finished, and
    every later test went back to the live app/archai.sqlite.
    """
    yield
    changed = {
        name: value
        for name, value in _store_paths().items()
        if value != _live_stores_outside_the_checkout[name]
    }
    if changed:
        pytest.fail(f"test left store paths pointing elsewhere: {changed}")


@pytest.fixture(autouse=True)
def _fresh_login_limiter():
    """Each test starts with no failed analytics logins on record.

    Assigned directly rather than through monkeypatch: requesting monkeypatch
    here would change the order in which the store-path guard is torn down.
    """
    from app.routers import analytics
    from app.services.login_limiter import FailedLoginLimiter

    analytics._login_limiter = FailedLoginLimiter()
    yield


@pytest.fixture(autouse=True)
def _no_held_tasks():
    """A test that stubs a batch out never releases its task; the next test starts with none held."""
    yield
    from app.services import file_manager

    held = getattr(file_manager, "_held", None)
    if held is not None:
        with file_manager._lock:
            held.clear()
