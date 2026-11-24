"""Without a chat key, only the paths that call the chat provider fail.

The segmented route's own test is in test_full_page_engine.py; /api/health
reporting the missing key as a degraded feature is in test_health.py.
"""

from __future__ import annotations

import base64
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.schemas.agents_ocr import OCRExtractRequest
from app.services.saia_client import LazySaiaClient, SaiaConfigError

_KEY_MESSAGE = "Chat API key not configured"


@pytest.fixture(autouse=True)
def no_chat_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("chat_ai_api_key", "saia_api_key", "archai_chat_ai_api_key", "archai_saia_api_key"):
        monkeypatch.setattr(settings, name, "")
    for name in ("CHAT_AI_API_KEY", "SAIA_API_KEY", "ARCHAI_CHAT_AI_API_KEY", "ARCHAI_SAIA_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_the_ocr_agent_is_built_without_a_key() -> None:
    from app.agents.ocr_agent import OcrAgent

    agent = OcrAgent()
    assert isinstance(agent.client, LazySaiaClient)


def test_the_lazy_client_raises_the_key_error_on_first_use() -> None:
    client = LazySaiaClient()
    with pytest.raises(SaiaConfigError, match=_KEY_MESSAGE):
        client.list_models()


def test_tiled_vlm_ocr_still_fails_with_the_key_error() -> None:
    # The whole-image path asks the provider for its models first.
    from app.agents.ocr_agent import OcrAgent

    page = base64.b64encode(b"not decoded: the key check comes first").decode("ascii")
    with pytest.raises(SaiaConfigError, match=_KEY_MESSAGE):
        OcrAgent().run(OCRExtractRequest(image_b64=page))


def test_chat_models_answers_at_once_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    from app.routers import chat

    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: pytest.fail("no client may be built without a key"))
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    started = time.monotonic()
    response = TestClient(app).get("/api/chat/models")
    assert response.status_code == 422
    assert response.json()["detail"].startswith(_KEY_MESSAGE)
    assert time.monotonic() - started < 2.0
