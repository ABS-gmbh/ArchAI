"""Index and authority routes check their input instead of passing it through."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import authority, index


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(index.router)
    app.include_router(authority.router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("params", ["query=arthur&top_k=-1", "query=arthur&top_k=0", "query=arthur&top_k=100000", "query="])
def test_search_parameters_are_bounded(client: TestClient, params: str) -> None:
    assert client.post(f"/index/search?{params}").status_code == 422


def test_unknown_runs_are_404(client: TestClient) -> None:
    """Both answered 200 with an empty result for a run that does not exist."""
    linked = client.post("/authority/link/no-such-run")
    status = client.get("/index/run/no-such-run/status")
    assert (linked.status_code, status.status_code) == (404, 404)
    assert linked.json()["detail"] == "Run not found: no-such-run"
