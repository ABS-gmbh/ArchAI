"""Request bodies over the upload limit are refused with 413 before they are read whole."""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel
from starlette.routing import compile_path

from app.body_limit import RequestBodyLimit, max_json_bytes, max_request_bytes
from app.config import settings


class Payload(BaseModel):
    image_b64: str


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(settings, "max_upload_megabytes", 1)
    monkeypatch.setattr(settings, "max_json_megabytes", 1)
    app = FastAPI()
    app.add_middleware(RequestBodyLimit)

    @app.post("/echo")
    def echo(payload: Payload) -> dict[str, int]:
        return {"length": len(payload.image_b64)}

    return TestClient(app)


def test_the_limit_allows_a_full_upload_in_base64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_upload_megabytes", 3)
    assert max_request_bytes() >= 3 * (1 << 20) * 4 // 3


def test_a_body_within_the_limit_passes(client: TestClient) -> None:
    response = client.post("/echo", json={"image_b64": "A" * 1000})
    assert response.status_code == 200 and response.json() == {"length": 1000}


def test_a_declared_oversized_body_is_a_413(client: TestClient) -> None:
    response = client.post("/echo", json={"image_b64": "A" * (max_json_bytes() + 10)})
    assert response.status_code == 413
    assert "larger than the server accepts" in response.json()["detail"]


def test_an_undeclared_oversized_body_is_counted(client: TestClient) -> None:
    """A chunked body has no Content-Length; the bytes are counted as they arrive."""
    chunk = b"A" * (256 * 1024)

    def stream():
        yield b'{"image_b64": "'
        for _ in range(max_json_bytes() // len(chunk) + 2):
            yield chunk
        yield b'"}'

    response = client.post("/echo", content=stream(), headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_the_app_installs_the_limit() -> None:
    import app.main as main

    assert any(m.cls is RequestBodyLimit for m in main.app.user_middleware)


def test_a_multipart_upload_is_held_to_the_upload_limit_not_the_json_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_upload_megabytes", 3)
    monkeypatch.setattr(settings, "max_json_megabytes", 1)
    app = FastAPI()
    # Only the paths given as uploads get the upload limit (tests/test_body_limit_by_route.py).
    app.add_middleware(RequestBodyLimit, upload_paths=[compile_path("/upload")[0]])

    @app.post("/upload")
    async def upload(request: Request) -> dict[str, int]:
        return {"length": len(await request.body())}

    client = TestClient(app)
    body = b"x" * (2 * (1 << 20))
    multipart = client.post("/upload", content=body, headers={"content-type": "multipart/form-data; boundary=b"})
    assert multipart.status_code == 200 and multipart.json() == {"length": len(body)}
    as_json = client.post("/upload", content=body, headers={"content-type": "application/json"})
    assert as_json.status_code == 413
    assert max_request_bytes() > 3 * (1 << 20)
