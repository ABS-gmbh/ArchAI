"""The helpers the OCR agents shared as copies (audit AG-17)."""

from __future__ import annotations

from typing import Any

import pytest

from app.agents import _llm_common, ocr_agent, saia_ocr_agent


class _Client:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None and "response_format" in kwargs:
            raise self.error
        return {"text": "ok"}


def test_json_object_is_asked_for_first() -> None:
    client = _Client()
    assert _llm_common.chat_completion_preferring_json_object(client, messages=[], model="m", top_p=1.0) == {"text": "ok"}
    assert client.calls == [{"response_format": {"type": "json_object"}, "messages": [], "model": "m", "top_p": 1.0}]


def test_a_rejected_format_is_retried_without_it_and_other_errors_raise() -> None:
    client = _Client(RuntimeError("unsupported response_format json_object"))
    _llm_common.chat_completion_preferring_json_object(client, messages=[], model="m")
    assert [("response_format" in call) for call in client.calls] == [True, False]
    with pytest.raises(RuntimeError, match="boom"):
        _llm_common.chat_completion_preferring_json_object(_Client(RuntimeError("boom")), messages=[], model="m")


@pytest.mark.parametrize("module", [ocr_agent, saia_ocr_agent])
def test_both_agents_use_the_one_size_check(module: Any) -> None:
    assert module._is_image_too_large_error(RuntimeError("Image size 9000x9000 exceeds limit"))
    assert module._is_image_too_large_error(RuntimeError("DecompressionBomb: decompression bomb"))
    assert not module._is_image_too_large_error(RuntimeError("timeout"))
