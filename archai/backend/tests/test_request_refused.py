"""Routes answer only this code base's own refusals with the exception's text (second security review).

The handlers that turned a bare ValueError into a 422 or 404 with its text
also echoed a ValueError raised by a library inside the call. They now catch
RequestRefused, raised with fixed sentences where the code refuses a request;
a library's ValueError falls through to the route's fixed-sentence answer and
its text goes to the log.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image


def _pil_value_error(*_args: Any, **_kwargs: Any) -> None:
    """A real PIL ValueError ("not enough image data"), as a library raises it inside a call."""
    Image.frombytes("RGB", (64, 64), b"\x00" * 10)


def _png_b64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 32), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _client(router: Any) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


def _region_ocr(monkeypatch: pytest.MonkeyPatch, fail: Callable[..., Any]) -> Any:
    from app.routers import ocr

    class Agent:
        def run(self, _payload: Any) -> Any:
            return fail()

    monkeypatch.setattr(ocr, "_get_ocr_agent", lambda: Agent())
    return _client(ocr.router).post(
        "/ocr/extract", json={"image_b64": _png_b64(), "page_id": "p", "ocr_backend": "kraken", "regions": [{"region_id": "r", "bbox_xyxy": [0, 0, 10, 10]}]}
    )


def test_a_library_value_error_inside_region_ocr_is_not_echoed(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR):
        response = _region_ocr(monkeypatch, _pil_value_error)
    assert response.status_code == 500
    assert "not enough image data" not in response.text
    assert response.json()["detail"] == "OCR extraction failed on the server. The details are in the server log."
    assert "not enough image data" in caplog.text


def test_a_refusal_of_this_code_base_is_still_answered_with_its_sentence(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services.public_errors import RequestRefused

    def refuse() -> None:
        raise RequestRefused("Provide image_b64 or cropped_image_b64.")

    response = _region_ocr(monkeypatch, refuse)
    assert (response.status_code, response.json()["detail"]) == (422, "Provide image_b64 or cropped_image_b64.")


def test_a_page_region_ocr_cannot_read_is_a_422_not_a_500() -> None:
    from app.routers import ocr

    response = _client(ocr.router).post(
        "/ocr/extract",
        json={"image_b64": base64.b64encode(b"hello world").decode("ascii"), "page_id": "p", "ocr_backend": "kraken", "regions": [{"region_id": "r", "bbox_xyxy": [0, 0, 5, 5]}]},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "The uploaded file is not an image the server can read. Use PNG, JPEG or TIFF."


@pytest.mark.parametrize(
    ("route", "method", "path", "target"),
    [
        ("index", "post", "/index/run/r1", ("app.services.rag_store", "index_run")),
        ("authority", "get", "/authority/report/r1", ("app.services.authority_linking", "build_report_from_db")),
    ],
)
def test_a_library_value_error_is_not_a_404_with_its_text(monkeypatch: pytest.MonkeyPatch, route: str, method: str, path: str, target: tuple[str, str]) -> None:
    import importlib

    router = importlib.import_module(f"app.routers.{route}").router
    monkeypatch.setattr(importlib.import_module(target[0]), target[1], _pil_value_error)
    response = getattr(_client(router), method)(path)
    assert response.status_code == 500
    assert "not enough image data" not in response.text


def test_a_library_value_error_in_a_chat_completion_is_not_echoed(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.routers import chat

    monkeypatch.setattr(chat, "create_chat_completion", _pil_value_error)
    response = _client(chat.router).post("/chat/completions", json={"stream": False, "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 500
    assert "not enough image data" not in response.text
