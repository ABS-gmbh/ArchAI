"""With nothing listening at the Ollama host, GLM-OCR gives up after one variant.

It used to try all six image variants twice each, with a 2 s pause between
attempts: 12 s per request to report that the server is not there, although
every variant goes to the same host.
"""

from __future__ import annotations

import io
import socket
from typing import Any

import pytest
import requests
from PIL import Image

from app.services import glm_ollama_ocr
from app.services.glm_ollama_ocr import GlmOllamaOcrError, run_glm_ollama_ocr


def _page() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (300, 200), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def posts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    real_post = requests.post

    def counting_post(url: str, **kwargs: Any):
        calls.append(url)
        return real_post(url, **kwargs)

    monkeypatch.setattr(glm_ollama_ocr.requests, "post", counting_post)
    monkeypatch.setattr(glm_ollama_ocr.time, "sleep", lambda seconds: None)
    return calls


def test_a_refused_connection_stops_after_the_first_variant(posts: list[str]) -> None:
    host = f"http://127.0.0.1:{_closed_port()}"
    with pytest.raises(GlmOllamaOcrError, match="Ollama is not reachable"):
        run_glm_ollama_ocr(_page(), image_ref="page", host=host, retries=2)
    # One variant, two attempts; no other variant tried.
    assert len(posts) == 2


def test_a_dropped_connection_still_tries_the_other_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def dropped(url: str, **kwargs: Any):
        calls.append(url)
        raise requests.ConnectionError("('Connection aborted.', RemoteDisconnected('Remote end closed connection'))")

    monkeypatch.setattr(glm_ollama_ocr.requests, "post", dropped)
    monkeypatch.setattr(glm_ollama_ocr.time, "sleep", lambda seconds: None)
    with pytest.raises(GlmOllamaOcrError, match="all OCR image variants failed"):
        run_glm_ollama_ocr(_page(), image_ref="page", host="http://127.0.0.1:1", retries=2)
    assert len(calls) > 2
