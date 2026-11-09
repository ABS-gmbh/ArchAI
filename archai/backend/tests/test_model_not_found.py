"""Only a missing model counts as MODEL_NOT_FOUND (audit SVC-22).

is_model_not_found_error matched "invalid_request_error", the type of every
400 from an OpenAI-compatible server, and any "404" in the message. An
oversized image was reported as a missing model, and the same request was
retried on each fallback model.
"""

from __future__ import annotations

import httpx
import openai
import pytest

from app.services.saia_client import is_model_not_found_error


def _api_error(cls, status: int, message: str, code: str | None, type_: str):
    request = httpx.Request("POST", "https://saia.example/v1/chat/completions")
    body = {"message": message, "type": type_, "code": code}
    response = httpx.Response(status, request=request, json={"error": body})
    return cls(message, response=response, body=body)


@pytest.mark.parametrize("error", [
    _api_error(openai.NotFoundError, 404, "The model `qwen-x` does not exist", "model_not_found", "invalid_request_error"),
    _api_error(openai.BadRequestError, 400, "unknown model qwen-x", None, "invalid_request_error"),
    RuntimeError("Model not found: qwen-x"),
])
def test_a_missing_model_is_recognised(error):
    assert is_model_not_found_error(error)


@pytest.mark.parametrize("error", [
    _api_error(openai.BadRequestError, 400, "Image too large: 40404 pixels over the limit", None, "invalid_request_error"),
    _api_error(openai.BadRequestError, 400, "This model's maximum context length is 32768 tokens", "context_length_exceeded", "invalid_request_error"),
    RuntimeError("upstream returned 502 after 404 ms"),
])
def test_other_request_errors_are_not(error):
    assert not is_model_not_found_error(error)
