"""A model the provider will not serve falls back, and is said once in the log.

Found on a user's installation (2026-11-25): the provider answers the default
translation model with `403 - Not authorized to access this Model:
llama-3.3-70b-instruct` and no longer lists it, and the app did not recognise
that 403 as a model it cannot use, so every translation failed instead of
falling back to the chat model, as fallback_models_for_stage prescribes. A
refused key is a 401 and must never be taken for it.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.config import settings
from app.services import chat_ai, provider_key, saia_client, setup_notices

KEY = "k" * 32
# What the provider lists today (2026-11-25).
OFFERED = [
    "deepseek-v4-flash-0731", "deepseek-v4.1-flash-1017", "eurollm-22b-instruct-2512", "gemma-4-31b-it", "glm-5.3",
    "glm-5.3-flash", "meta-llama-3.1-8b-instruct", "mistral-large-4-1t-a52b", "qwen3-30b-a3b-instruct-2507",
    "qwen3-coder-next", "qwen3-omni-30b-a3b-instruct", "qwen3.5-397b-a17b", "qwen3.6-35b-a3b", "qwen3.8-27b",
]


class _Provider(BaseHTTPRequestHandler):
    key_refused = False
    asked: list[str] = []
    content = "Translated."

    def _send(self, status: int, body: object) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # /models
        if self.key_refused:
            return self._send(401, {"error": {"message": "Unknown API key"}})
        self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in OFFERED]})

    def do_POST(self) -> None:  # /chat/completions
        payload = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
        model = str(payload.get("model"))
        type(self).asked.append(model)
        if self.key_refused:
            return self._send(401, {"error": {"message": "Unknown API key"}})
        if model not in OFFERED:
            return self._send(403, {"message": f"Not authorized to access this Model: {model}"})
        self._send(200, {"id": "x", "object": "chat.completion", "created": 0, "model": model,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": self.content}, "finish_reason": "stop"}]})

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_Provider]]:
    handler = type("Provider", (_Provider,), {"key_refused": False, "asked": [], "content": "Translated."})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    monkeypatch.setattr(settings, "chat_ai_api_key", KEY)
    for name in ("chat_ai_base_url", "archai_chat_ai_base_url"):
        monkeypatch.setattr(settings, name, base)
    # A configured translation model the provider no longer offers, as on the
    # user's installation (since then TRANSLATION_MODEL defaults to empty: the
    # chat model translates).
    monkeypatch.setattr(settings, "translation_model", "llama-3.3-70b-instruct")
    provider_key.reset()
    setup_notices.reset()
    # The label agent and the SAIA model list are cached per process; they must reach this stand-in.
    from app.routers import chat as chat_router
    from app.services.saia_client import SaiaClient

    monkeypatch.setattr(chat_router, "_label_analysis_agent_instance", None)
    monkeypatch.setattr(SaiaClient, "_shared_models_cache", None)
    yield handler
    server.shutdown()
    provider_key.reset()
    setup_notices.reset()


TRANSLATE = {"chat_stage": "translation"}
MESSAGES = [{"role": "user", "content": "Translate: Caldus de Lausanne"}]


def test_a_translation_model_not_offered_falls_back_to_the_chat_model(provider: type[_Provider]) -> None:
    assert settings.translation_model == "llama-3.3-70b-instruct"  # configured, and not offered
    result = chat_ai.create_chat_completion(messages=MESSAGES, context=TRANSLATE)
    assert result["text"] == "Translated."
    assert result["model"] == settings.chat_rag_model  # the answer says which model produced it
    assert provider.asked[0] == "llama-3.3-70b-instruct"
    # The 403 was about the model, not the key.
    assert provider_key.state() == "accepted"


def test_the_log_says_once_per_model_which_was_used_instead(provider: type[_Provider], caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        chat_ai.create_chat_completion(messages=MESSAGES, context=TRANSLATE)
        chat_ai.create_chat_completion(messages=MESSAGES, context=TRANSLATE)
    lines = [r.getMessage() for r in caplog.records if "llama-3.3-70b-instruct" in r.getMessage()]
    assert len(lines) == 1, lines
    assert "not offered" in lines[0] and settings.chat_rag_model in lines[0]


def test_a_streamed_translation_falls_back_too(provider: type[_Provider]) -> None:
    events = list(chat_ai.stream_chat_completion(messages=MESSAGES, context=TRANSLATE))
    done = [e for e in events if e.get("type") == "done"]
    assert done and done[0]["model"] == settings.chat_rag_model


def test_a_refused_key_is_never_taken_for_a_model_not_offered(provider: type[_Provider]) -> None:
    provider.key_refused = True
    with pytest.raises(Exception):
        chat_ai.create_chat_completion(messages=MESSAGES, context=TRANSLATE)
    assert provider.asked == ["llama-3.3-70b-instruct"]  # no fallback round on a 401
    assert provider_key.state() == "refused"


def test_the_shapes_of_the_errors() -> None:
    import httpx
    import openai

    request = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")

    def error(cls: type[openai.APIStatusError], status: int, body: object) -> openai.APIStatusError:
        return cls(f"Error code: {status} - {body}", response=httpx.Response(status, request=request), body=body)

    not_offered = error(openai.PermissionDeniedError, 403, {"message": "Not authorized to access this Model: llama-3.3-70b-instruct"})
    refused = error(openai.AuthenticationError, 401, {"message": "Unknown API key"})
    other_403 = error(openai.PermissionDeniedError, 403, {"message": "Forbidden"})
    for check in (chat_ai._is_model_not_found_error, saia_client.is_model_not_found_error):
        assert check(not_offered) is True
        assert check(refused) is False
        assert check(other_403) is False


# ── (b) The operator is told, from the provider's own list ──────────────

TRANSLATION_SENTENCE = (
    "The configured translation model is not offered by the provider; translations use the chat model instead. "
    "Set TRANSLATION_MODEL to one of the provider's models."
)


def _health_task_models() -> dict:
    from fastapi.testclient import TestClient

    import app.main as main

    return TestClient(main.app).get("/api/health").json()["checks"]["task_models"]


def test_nothing_is_said_before_the_provider_has_listed_its_models(provider: type[_Provider]) -> None:
    provider_key.forget_offered()
    assert _health_task_models() == {"ok": True, "detail": "The provider's models have not been listed yet."}


def test_the_model_list_marks_each_configured_model_not_offered(provider: type[_Provider]) -> None:
    listed = chat_ai.list_available_models()
    assert listed["provider"] == "ok"
    assert listed["not_offered"] == {
        "translation_model": "llama-3.3-70b-instruct",
        "label_visual_model": "qwen3-vl-30b-a3b-instruct",
        "label_visual_fallback_model": "internvl3.5-30b-a3b",
        "verifier_model": "qwen3-235b-a22b",  # PALEOGRAPHY_VERIFICATION_MODEL's default, not listed either
    }


def test_health_names_each_task_whose_model_is_not_offered(provider: type[_Provider]) -> None:
    chat_ai.list_available_models()
    gets_before = sum(1 for model in provider.asked)  # POSTs only; GETs counted below
    check = _health_task_models()
    assert check["ok"] is False
    assert TRANSLATION_SENTENCE in check["detail"]
    assert "LABEL_VISUAL_MODEL" in check["detail"]
    assert "PALEOGRAPHY_VERIFICATION_MODEL" in check["detail"]
    assert (
        "The configured label-analysis model is not offered by the provider; label analysis uses "
        "gemma-4-31b-it instead. Set LABEL_VISUAL_MODEL to one of the provider's models."
    ) in check["detail"]
    assert "Set PALEOGRAPHY_VERIFICATION_MODEL to one of the provider's models." in check["detail"]
    assert "CHAT_RAG_MODEL" not in check["detail"]  # the chat model is offered
    assert sum(1 for model in provider.asked) == gets_before


def test_health_sends_nothing_for_it(provider: type[_Provider], monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(chat_ai, "_raw_server_models", lambda client: calls.append(1) or [])
    _health_task_models()
    _health_task_models()
    assert calls == []


def test_every_configured_model_offered_is_ok(provider: type[_Provider], monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("translation_model", "label_visual_model", "label_visual_fallback_model", "paleography_verification_model"):
        monkeypatch.setattr(settings, name, "qwen3-omni-30b-a3b-instruct")
    chat_ai.list_available_models()
    assert _health_task_models() == {"ok": True, "detail": "The provider offers every configured model."}


# ── (c) Label analysis and the provider OCR say so ──────────────────────

def _png_b64() -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (200, 120), "white").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_label_analysis_falls_back_and_the_log_says_so_once(provider: type[_Provider], caplog: pytest.LogCaptureFixture) -> None:
    from fastapi.testclient import TestClient

    import app.main as main

    client = TestClient(main.app)
    body = {
        "question": "Which letter is this initial?",
        "label_name": "Plain initial - Black",
        "image_b64": _png_b64(),
        "regions": [{"region_id": "r1", "bbox_xyxy": [10, 10, 100, 60], "polygons": []}],
    }
    with caplog.at_level(logging.WARNING):
        first = client.post("/api/chat/label-analysis", json=body)
        client.post("/api/chat/label-analysis", json=body)
    assert first.status_code == 200
    assert first.json()["model_used"] != "qwen3-vl-30b-a3b-instruct"  # the answer names the model used
    lines = [r.getMessage() for r in caplog.records if "qwen3-vl-30b-a3b-instruct" in r.getMessage() and "not offered" in r.getMessage()]
    assert len(lines) == 1, lines


def test_the_provider_ocr_says_its_model_is_not_offered(provider: type[_Provider], caplog: pytest.LogCaptureFixture) -> None:
    from fastapi.testclient import TestClient

    import app.main as main

    with caplog.at_level(logging.WARNING):
        first = TestClient(main.app).post("/api/ocr/extract", json={"image_b64": _png_b64()})
        TestClient(main.app).post("/api/ocr/extract", json={"image_b64": _png_b64()})
    assert first.status_code == 422
    assert first.json()["detail"] == (
        "Reading with the OCR provider is not possible: the provider does not offer the model it reads with. "
        "Tell whoever runs the server."
    )
    lines = [r.getMessage() for r in caplog.records if "internvl3.5-30b-a3b" in r.getMessage() and "not offered" in r.getMessage()]
    assert len(lines) == 1, lines


def test_an_empty_translation_model_is_the_chat_model_and_never_reported(provider: type[_Provider], monkeypatch: pytest.MonkeyPatch) -> None:
    """TRANSLATION_MODEL empty means the chat model translates (the user's choice, 2026-11-25): the
    comparison is of the resolved assignment, so it is not "not offered", and nothing falls back."""
    monkeypatch.setattr(settings, "translation_model", "")
    listed = chat_ai.list_available_models()
    assert "translation_model" not in listed["not_offered"]
    assert "translation model" not in _health_task_models()["detail"]
    result = chat_ai.create_chat_completion(messages=MESSAGES, context=TRANSLATE)
    assert result["model"] == settings.chat_rag_model
    assert provider.asked == [settings.chat_rag_model]  # asked once, with the chat model



# ── The verifier falls back to the chat model, and says it checked itself ─

VERDICT = json.dumps({"assessment": "supported", "corrected_answer": "Lausanne.", "notes": [], "citations_checked": []})


def _verify(answer_model: str) -> dict:
    return chat_ai._run_paleography_verification(
        question="Who is named?",
        draft_text="Lausanne.",
        context={"ocr_run_id": "run-1", "transcript": "Caldus de Lausanne"},
        evidence_text="[OCR_CHUNK_EVIDENCE] Caldus de Lausanne",
        answer_model=answer_model,
    )


def test_the_verifier_falls_back_to_a_second_fast_model_not_the_one_that_answered(
    provider: type[_Provider], caplog: pytest.LogCaptureFixture
) -> None:
    provider.content = VERDICT
    assert settings.paleography_verification_model == "qwen3-235b-a22b"  # the default, not offered
    with caplog.at_level(logging.WARNING):
        verification = _verify(settings.chat_rag_model)
        _verify(settings.chat_rag_model)
    assert provider.asked[:2] == ["qwen3-235b-a22b", "deepseek-v4.1-flash-1017"]  # the first fast one offered
    assert verification["assessment"] == "supported"
    assert verification["model_used"] == "deepseek-v4.1-flash-1017"
    assert verification["self_check"] is False
    lines = [r.getMessage() for r in caplog.records if "qwen3-235b-a22b" in r.getMessage() and "not offered" in r.getMessage()]
    assert len(lines) == 1, lines


def test_a_verifier_configured_as_the_answering_model_is_replaced_by_a_second_one(
    provider: type[_Provider], monkeypatch: pytest.MonkeyPatch
) -> None:
    provider.content = VERDICT
    monkeypatch.setattr(settings, "paleography_verification_model", settings.chat_rag_model)
    verification = _verify(settings.chat_rag_model)
    assert verification["model_used"] != settings.chat_rag_model
    assert verification["self_check"] is False


def test_only_when_no_other_model_is_offered_does_the_answering_model_check_itself(provider: type[_Provider]) -> None:
    provider.content = VERDICT
    saved = list(OFFERED)
    OFFERED[:] = [settings.chat_rag_model]
    provider_key.forget_offered()
    try:
        verification = _verify(settings.chat_rag_model)
    finally:
        OFFERED[:] = saved
    assert verification["model_used"] == settings.chat_rag_model
    assert verification["self_check"] is True


def test_a_second_model_is_not_a_self_check(provider: type[_Provider], monkeypatch: pytest.MonkeyPatch) -> None:
    provider.content = VERDICT
    monkeypatch.setattr(settings, "paleography_verification_model", "qwen3.8-27b")
    verification = _verify(settings.chat_rag_model)
    assert verification["model_used"] == "qwen3.8-27b"
    assert verification["self_check"] is False


def test_health_says_which_second_model_checks_answers(provider: type[_Provider]) -> None:
    chat_ai.list_available_models()
    assert (
        "The configured answer-checking model is not offered by the provider; answers are checked by "
        "deepseek-v4.1-flash-1017 instead. Set PALEOGRAPHY_VERIFICATION_MODEL to one of the provider's models."
    ) in _health_task_models()["detail"]



# ── One vision test for label analysis and the provider OCR ─────────────

def test_label_analysis_and_the_ocr_agent_share_one_vision_test() -> None:
    from app.agents import label_analysis_agent, ocr_agent
    from app.services import model_router

    assert ocr_agent.is_vision_model is model_router.is_vision_model
    assert label_analysis_agent.LabelAnalysisAgent._is_vision_model is model_router.is_vision_model
    # The provider's documentation (2026-11-25): six of today's 14 take images; mistral-large does not.
    assert [m for m in OFFERED if model_router.is_vision_model(m)] == [
        "gemma-4-31b-it", "glm-5.3-flash", "qwen3-omni-30b-a3b-instruct",
        "qwen3.5-397b-a17b", "qwen3.6-35b-a3b", "qwen3.8-27b",
    ]
    assert not model_router.is_vision_model("mistral-large-4-1t-a52b")


def test_label_analysis_now_falls_back_to_a_model_its_test_calls_vision_capable() -> None:
    from app.agents.label_analysis_agent import label_retry_model

    # Today's catalogue, read in order: the first listed that the shared test accepts.
    assert label_retry_model("qwen3-vl-30b-a3b-instruct", OFFERED, prefer_vision=True) == "gemma-4-31b-it"



def test_an_image_never_goes_to_a_text_model(provider: type[_Provider]) -> None:
    from fastapi.testclient import TestClient

    import app.main as main
    from app.agents.label_analysis_agent import NO_VISION_MODEL, label_retry_model

    text_only = [m for m in OFFERED if m not in {"gemma-4-31b-it", "glm-5.3-flash", "qwen3-omni-30b-a3b-instruct", "qwen3.5-397b-a17b", "qwen3.6-35b-a3b", "qwen3.8-27b"}]
    assert label_retry_model("qwen3-vl-30b-a3b-instruct", text_only, prefer_vision=True) is None
    # A question about the text may still go to a text model.
    assert label_retry_model("qwen3-vl-30b-a3b-instruct", text_only, prefer_vision=False) is not None

    OFFERED[:] , saved = text_only, list(OFFERED)
    try:
        from app.services.saia_client import SaiaClient

        SaiaClient._shared_models_cache = None
        response = TestClient(main.app).post("/api/chat/label-analysis", json={
            "question": "Which letter is this initial?",
            "label_name": "Plain initial - Black",
            "image_b64": _png_b64(),
            "regions": [{"region_id": "r1", "bbox_xyxy": [10, 10, 100, 60], "polygons": []}],
        })
    finally:
        OFFERED[:] = saved
        SaiaClient._shared_models_cache = None
    assert response.status_code == 422
    assert response.json()["detail"] == NO_VISION_MODEL
    assert not any(m in text_only for m in provider.asked)  # no text model was sent the image
