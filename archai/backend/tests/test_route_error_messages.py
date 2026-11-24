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
        (_openai_error(openai.AuthenticationError, 401), 502, "refused this server's key"),
        (_openai_error(openai.PermissionDeniedError, 403), 502, "refused this server's key"),
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
    from app.services.public_errors import RequestRefused

    def missing(run_id: str):
        raise RequestRefused(f"Pipeline run {run_id!r} not found.")

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
    assert response.json()["detail"] == "The embedding provider refused this server's key. Tell whoever runs it."


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

    from app.services.public_errors import RequestRefused

    def missing(run_id: str):
        raise RequestRefused(f"Run not found: {run_id}")

    monkeypatch.setattr(authority_linking, "build_report_from_db", missing)
    response = _client(authority.router).get("/authority/report/nope")
    assert (response.status_code, response.json()["detail"]) == (404, "Run not found: nope")


# ──────────────────────────────────────────────────────── rag_debug ──


@pytest.mark.parametrize(
    ("path", "module", "target", "what"),
    [
        ("/rag/debug/retrieve?query=arthur", "app.services.rag_store", "retrieve_debug", "Retrieval"),
        (
            "/rag/debug/evidence-preview?query=arthur&run_id=r1",
            "app.services.chat_ai",
            "build_rag_evidence_for_debug",
            "The evidence preview",
        ),
    ],
)
def test_rag_debug_routes_keep_exception_text_out_of_the_response(
    monkeypatch: pytest.MonkeyPatch, path: str, module: str, target: str, what: str
) -> None:
    import importlib

    from app.routers import rag_debug

    def boom(*args, **kwargs):
        raise RuntimeError(LEAKY)

    monkeypatch.setenv("ARCHAI_DEBUG_RAG", "1")
    monkeypatch.setattr(importlib.import_module(module), target, boom)
    response = _client(rag_debug.router).get(path)
    assert response.status_code == 500
    assert "/srv/archai" not in response.text and "internal.example" not in response.text
    assert response.json()["detail"] == f"{what} failed on the server. The details are in the server log."


def test_rag_debug_still_explains_how_to_enable_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.routers import rag_debug

    monkeypatch.delenv("ARCHAI_DEBUG_RAG", raising=False)
    response = _client(rag_debug.router).get("/rag/debug/retrieve?query=arthur")
    assert response.status_code == 404
    assert response.json()["detail"] == "RAG debug endpoint disabled.  Set ARCHAI_DEBUG_RAG=1 to enable."


# ────────────────────────────────────────────────────────────── ocr ──


def _png_b64() -> str:
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (60, 40), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _Agent:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def run(self, payload):
        raise self.exc


def _ocr_extract(monkeypatch: pytest.MonkeyPatch, exc: BaseException):
    from app.routers import ocr

    monkeypatch.setattr(ocr, "_get_ocr_agent", lambda: _Agent(exc))
    return _client(ocr.router).post("/ocr/extract", json={"image_b64": _png_b64()})


def test_ocr_extract_keeps_the_agents_error_text_out(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.agents.ocr_agent import OCRAgentError

    response = _ocr_extract(monkeypatch, OCRAgentError(f"OCR failed on model internvl: {LEAKY}"))
    assert response.status_code == 500
    assert "/srv/archai" not in response.text and "internal.example" not in response.text
    assert response.json()["detail"] == "OCR extraction failed on the server. The details are in the server log."


def test_ocr_extract_reports_a_rejected_key_through_the_agents_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.agents.ocr_agent import OCRAgentError

    try:
        try:
            raise _openai_error(openai.AuthenticationError, 401)
        except openai.AuthenticationError as inner:
            raise OCRAgentError(f"OCR failed on model internvl: {inner}") from inner
    except OCRAgentError as wrapped:
        response = _ocr_extract(monkeypatch, wrapped)
    assert (response.status_code, response.json()["detail"]) == (502, "The OCR provider refused this server's key. Tell whoever runs it.")


def test_ocr_extract_tells_the_reader_the_ocr_provider_is_not_set_up(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """It answered the configuration text, naming CHAT_AI_API_KEY, to the reader; that goes to the log now."""
    from app.services import setup_notices
    from app.services.saia_client import SaiaConfigError

    setup_notices.reset()
    message = "Chat API key not configured. Set CHAT_AI_API_KEY (or SAIA_API_KEY) in backend/.env or environment."
    with caplog.at_level(logging.WARNING):
        first = _ocr_extract(monkeypatch, SaiaConfigError(message))
        second = _ocr_extract(monkeypatch, SaiaConfigError(message))
    sentence = "Reading with the OCR provider is not set up on this server. Tell whoever runs it."
    assert (first.status_code, first.json()["detail"]) == (422, sentence)
    assert second.json()["detail"] == sentence
    assert "CHAT_AI_API_KEY" not in first.text
    assert sum("CHAT_AI_API_KEY" in record.getMessage() for record in caplog.records) == 1


@pytest.mark.parametrize("path", ["/ocr/extract_full_page", "/ocr/saia"])
def test_glm_failures_keep_the_ollama_host_out(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    from app.routers import ocr
    from app.services.glm_ollama_ocr import GlmOllamaOcrError

    def refuse(*args, **kwargs):
        raise GlmOllamaOcrError(f"Ollama request failed at http://internal.example:11434/api/chat: {LEAKY}")

    monkeypatch.setattr(ocr, "run_glm_ollama_ocr", refuse)
    response = _client(ocr.router).post(
        path, json={"image_b64": _png_b64(), "ocr_backend": "glmocr", "apply_proofread": False}
    )
    assert response.status_code == 502
    assert "internal.example" not in response.text and "/srv/archai" not in response.text
    assert response.json()["detail"] == "GLM-OCR (Ollama) could not read the page. The details are in the server log."


def _trace(monkeypatch: pytest.MonkeyPatch, *, key: str):
    from app.config import settings
    from app.routers import ocr
    from app.services import trace_pipeline

    # Agents are cached per process; one built with a key would hide its absence.
    monkeypatch.setattr(ocr, "_saia_ocr_agent_instance", None)
    monkeypatch.setattr(trace_pipeline, "_ocr_agent_instance", None)
    monkeypatch.setattr(ocr, "run_glm_ollama_ocr", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(LEAKY)))
    for name in ("saia_api_key", "archai_chat_ai_api_key", "archai_saia_api_key"):
        monkeypatch.setattr(settings, name, "")
    monkeypatch.setattr(settings, "chat_ai_api_key", key)
    for name in ("CHAT_AI_API_KEY", "SAIA_API_KEY", "ARCHAI_CHAT_AI_API_KEY", "ARCHAI_SAIA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return _client(ocr.router).post(
        "/ocr/page_with_trace", json={"image_b64": _png_b64(), "ocr_backend": "glmocr", "apply_proofread": False}
    )


def test_a_failed_trace_run_keeps_its_id_and_hides_the_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _trace(monkeypatch, key="test-key")
    detail = response.json()["detail"]
    assert response.status_code == 500 and detail["run_id"]
    assert "/srv/archai" not in response.text and "internal.example" not in response.text
    assert detail["error"] == "The OCR run failed on the server. The details are in the server log."


def test_a_trace_run_without_a_key_still_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _trace(monkeypatch, key="")
    assert response.status_code == 500
    assert response.json()["detail"]["error"] == (
        "Reading with the OCR provider is not set up on this server. Tell whoever runs it."
    )


# ───────────────────────────────────────────────────────────── chat ──


def _raise(exc: BaseException):
    def raiser(*args, **kwargs):
        raise exc

    return raiser


MESSAGES = {"messages": [{"role": "user", "content": "hi"}]}


@pytest.mark.parametrize(
    ("call", "target", "what"),
    [
        (lambda c: c.get("/chat/models"), "list_available_models", "Listing the chat models"),
        (lambda c: c.post("/chat/completions", json={**MESSAGES, "stream": False}), "create_chat_completion", "The chat completion"),
    ],
)
def test_chat_routes_keep_exception_text_out(monkeypatch: pytest.MonkeyPatch, call, target: str, what: str) -> None:
    from app.routers import chat

    monkeypatch.setattr(chat, target, _raise(RuntimeError(LEAKY)))
    response = call(_client(chat.router))
    assert response.status_code == 500
    assert "/srv/archai" not in response.text and "internal.example" not in response.text
    assert response.json()["detail"] == f"{what} failed on the server. The details are in the server log."


def test_label_analysis_keeps_exception_text_out(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.routers import chat

    monkeypatch.setattr(chat, "_get_label_analysis_agent", lambda: _Agent(RuntimeError(LEAKY)))
    response = _client(chat.router).post(
        "/chat/label-analysis", json={"question": "q", "label_name": "MainZone", "image_b64": _png_b64()}
    )
    assert response.status_code == 500
    assert "/srv/archai" not in response.text
    assert response.json()["detail"] == "The label analysis failed on the server. The details are in the server log."


def test_a_rejected_chat_key_is_a_502(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.routers import chat

    monkeypatch.setattr(chat, "create_chat_completion", _raise(_openai_error(openai.AuthenticationError, 401)))
    response = _client(chat.router).post("/chat/completions", json={**MESSAGES, "stream": False})
    assert (response.status_code, response.json()["detail"]) == (502, "The chat provider refused this server's key. Tell whoever runs it.")


def test_the_chat_configuration_message_is_for_the_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """The setting names go to the log (tests/test_chat_not_set_up.py), not to the reader."""
    from app.routers import chat
    from app.services.chat_ai import ChatConfigError

    message = "Chat API key not configured. Set CHAT_AI_API_KEY (or SAIA_API_KEY) in backend/.env or environment."
    monkeypatch.setattr(chat, "list_available_models", _raise(ChatConfigError(message)))
    response = _client(chat.router).get("/chat/models")
    assert (response.status_code, response.json()["detail"]) == (422, "Chat is not set up on this server. Tell whoever runs it.")


def test_a_streamed_failure_keeps_exception_text_out(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.routers import chat

    def failing_stream(**kwargs):
        yield {"type": "delta", "delta": "In"}
        raise RuntimeError(LEAKY)

    monkeypatch.setattr(chat, "stream_chat_completion", failing_stream)
    response = _client(chat.router).post("/chat/completions", json={**MESSAGES, "stream": True})
    assert response.status_code == 200
    assert "/srv/archai" not in response.text and "internal.example" not in response.text
    assert "The chat answer failed on the server. The details are in the server log." in response.text
