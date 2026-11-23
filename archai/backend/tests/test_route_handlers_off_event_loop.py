"""Route handlers that do blocking work must not run on the event loop.

An ``async def`` handler runs on the event loop itself, so a synchronous call
inside it - SQLite, a file read, an HTTP request - freezes every other request
on the process until it returns. FastAPI runs a plain ``def`` handler in its
threadpool instead.

Eighteen handlers were ``async def`` without a single ``await``: authority
linking (Wikidata, VIAF and GeoNames requests with 10-15 s timeouts), RAG
indexing and search (embedding requests), the trace snapshot and tables
(SQLite), evidence spans, analytics, downloads and batch results (task state
read from disk). An ``async def`` with no ``await`` can only be trivial or
blocking, so the structural check below refuses it outright.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROUTERS = Path(__file__).resolve().parents[1] / "app" / "routers"
_ROUTE_DECORATORS = {"get", "post", "put", "patch", "delete"}


def _async_routes_without_await() -> list[str]:
    found = []
    for path in sorted(ROUTERS.glob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            is_route = any(
                isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr in _ROUTE_DECORATORS
                for d in node.decorator_list
            )
            if is_route and not any(isinstance(n, ast.Await) for n in ast.walk(node)):
                found.append(f"{path.name}:{node.lineno} {node.name}")
    return found


def test_no_async_route_lacks_an_await() -> None:
    assert _async_routes_without_await() == []


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def test_authority_linking_route_runs_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.routers import authority
    from app.services import authority_linking

    seen: dict[str, bool] = {}

    def fake_linking(run_id: str, force_refresh: bool = False) -> dict:
        seen["on_loop"] = _on_event_loop()
        return {"run_id": run_id}

    from app.db import pipeline_db

    monkeypatch.setattr(authority_linking, "run_authority_linking", fake_linking)
    run_id = pipeline_db.create_run(asset_ref="page-1")
    app = FastAPI()
    app.include_router(authority.router)
    response = TestClient(app).post(f"/authority/link/{run_id}")
    assert response.status_code == 200
    assert seen == {"on_loop": False}


def test_index_route_runs_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.routers import index

    seen: dict[str, bool] = {}

    def fake_index_run(run_id: str) -> dict:
        seen["on_loop"] = _on_event_loop()
        return {"run_id": run_id}

    monkeypatch.setattr(index.rag_store, "index_run", fake_index_run)
    app = FastAPI()
    app.include_router(index.router)
    response = TestClient(app).post("/index/run/run-1")
    assert response.status_code == 200
    assert seen == {"on_loop": False}


def test_trace_snapshot_route_runs_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.routers import ocr

    seen: dict[str, bool] = {}

    def fake_snapshot(run_id: str) -> dict:
        seen["on_loop"] = _on_event_loop()
        return {"run": {"run_id": run_id}}

    monkeypatch.setattr(ocr, "_build_trace_snapshot", fake_snapshot)
    app = FastAPI()
    app.include_router(ocr.router)
    response = TestClient(app).get("/ocr/trace/run-1")
    assert response.status_code == 200
    assert seen == {"on_loop": False}
