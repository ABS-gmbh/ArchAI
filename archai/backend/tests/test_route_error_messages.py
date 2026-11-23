"""Unexpected exceptions reach the log, not the response; actionable ones keep their message.

The UI shows ``detail`` verbatim. The routes used to put the exception's text
there, so a file path, an upstream URL or a response body could reach the
browser.
"""

from __future__ import annotations

import logging
import socket

import httpx
import openai
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.routers import index
from app.routers.errors import provider_failure, unexpected_error

LEAKY = "could not open /srv/archai/private/archai.sqlite via https://internal.example/v1?key=abc"


def _client(*routers) -> TestClient:
    app = FastAPI()
    for router in routers:
        app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


def _openai_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://internal.example/v1/chat/completions")
    return cls(LEAKY, response=httpx.Response(status, request=request), body=None)


# ───────────────────────────────────────────────────────── the helper ──


@pytest.mark.parametrize(
    ("exc", "status", "phrase"),
    [
        (_openai_error(openai.AuthenticationError, 401), 502, "rejected the API key"),
        (_openai_error(openai.PermissionDeniedError, 403), 502, "rejected the API key"),
        (_openai_error(openai.RateLimitError, 429), 429, "limiting requests"),
        (openai.APITimeoutError(httpx.Request("POST", "https://internal.example/v1")), 504, "did not answer in time"),
        (socket.timeout(LEAKY), 504, "did not answer in time"),
    ],
)
def test_provider_failures_a_person_can_act_on(exc: BaseException, status: int, phrase: str) -> None:
    response = provider_failure(exc, provider="chat provider")
    assert response is not None and response.status_code == status and phrase in response.detail
    assert "internal.example" not in response.detail


def test_anything_else_is_a_fixed_500_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR), pytest.raises(HTTPException) as raised:
        unexpected_error(RuntimeError(LEAKY), what="Indexing", provider="embedding provider")
    assert raised.value.status_code == 500
    assert raised.value.detail == "Indexing failed on the server. The details are in the server log."
    assert "/srv/archai/private" in caplog.text


# ──────────────────────────────────────────────────────────── index ──


@pytest.mark.parametrize(
    ("method", "path", "target"),
    [
        ("post", "/index/run/{run}", "index_run"),
        ("get", "/index/run/{run}/status", "is_run_indexed"),
        ("delete", "/index/run/{run}", "delete_run"),
        ("post", "/index/search?query=arthur", "retrieve_chunks"),
    ],
)
def test_index_routes_keep_exception_text_out_of_the_response(
    monkeypatch: pytest.MonkeyPatch, method: str, path: str, target: str
) -> None:
    from app.db import pipeline_db

    def boom(*args, **kwargs):
        raise RuntimeError(LEAKY)

    monkeypatch.setattr(index.rag_store, target, boom)
    run_id = pipeline_db.create_run(asset_ref="page")
    response = getattr(_client(index.router), method)(path.format(run=run_id))
    assert response.status_code == 500
    assert "/srv/archai" not in response.text and "internal.example" not in response.text
    assert response.json()["detail"].endswith("failed on the server. The details are in the server log.")


def test_index_keeps_its_not_found_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(run_id: str):
        raise ValueError(f"Pipeline run {run_id!r} not found.")

    monkeypatch.setattr(index.rag_store, "index_run", missing)
    response = _client(index.router).post("/index/run/nope")
    assert (response.status_code, response.json()["detail"]) == (404, "Pipeline run 'nope' not found.")


def test_an_embedding_provider_rejecting_the_key_is_a_502(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.db import pipeline_db

    def rejected(run_id: str):
        raise _openai_error(openai.AuthenticationError, 401)

    monkeypatch.setattr(index.rag_store, "index_run", rejected)
    response = _client(index.router).post(f"/index/run/{pipeline_db.create_run(asset_ref='p')}")
    assert response.status_code == 502
    assert response.json()["detail"] == "The embedding provider rejected the API key."


# ──────────────────────────────────────────────────────── authority ──


@pytest.mark.parametrize(
    ("method", "path", "target", "what"),
    [
        ("post", "/authority/link/{run}", "run_authority_linking", "Authority linking"),
        ("get", "/authority/report/{run}", "build_report_from_db", "The linking report"),
    ],
)
def test_authority_routes_keep_exception_text_out_of_the_response(
    monkeypatch: pytest.MonkeyPatch, method: str, path: str, target: str, what: str
) -> None:
    from app.db import pipeline_db
    from app.routers import authority
    from app.services import authority_linking

    def boom(*args, **kwargs):
        raise RuntimeError(LEAKY)

    monkeypatch.setattr(authority_linking, target, boom)
    response = getattr(_client(authority.router), method)(path.format(run=pipeline_db.create_run(asset_ref="p")))
    assert response.status_code == 500
    assert "/srv/archai" not in response.text and "internal.example" not in response.text
    assert response.json()["detail"] == f"{what} failed on the server. The details are in the server log."


def test_the_report_keeps_its_not_found_message(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.routers import authority
    from app.services import authority_linking

    def missing(run_id: str):
        raise ValueError(f"Run not found: {run_id}")

    monkeypatch.setattr(authority_linking, "build_report_from_db", missing)
    response = _client(authority.router).get("/authority/report/nope")
    assert (response.status_code, response.json()["detail"]) == (404, "Run not found: nope")
