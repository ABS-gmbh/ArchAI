"""The upload limit is chosen by the route, not by the Content-Type the client sends.

A second security review (2026-11-24) sent a 200 MB body to /api/chat/completions
with `Content-Type: multipart/form-data`: it got the upload limit (about 1.3 GB),
was read whole and answered 422 with a 209.7 MB response, because the
validation error echoed the body back. Only the routes that take an UploadFile
get the upload limit now; every other route gets MAX_JSON_MEGABYTES whatever
its content type, and a validation error no longer echoes what was sent.
"""

from __future__ import annotations

import inspect
import typing

import pytest
from fastapi import APIRouter, FastAPI, File, UploadFile
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.routing import compile_path

import app.main as main
from app.body_limit import RequestBodyLimit, takes_upload, upload_paths
from app.config import settings

MB = 1 << 20


def _annotation_mentions_upload(annotation: object) -> bool:
    if annotation is UploadFile:
        return True
    return any(_annotation_mentions_upload(arg) for arg in typing.get_args(annotation))


def _endpoint_takes_upload(route: APIRoute) -> bool:
    hints = typing.get_type_hints(route.endpoint)
    return any(
        _annotation_mentions_upload(hints.get(name)) for name in inspect.signature(route.endpoint).parameters
    )


def _app_upload_patterns() -> tuple[str, ...]:
    limit = next(m for m in main.app.user_middleware if m.cls is RequestBodyLimit)
    return tuple(pattern.pattern for pattern in limit.kwargs["upload_paths"])


def test_every_route_with_an_upload_parameter_gets_the_upload_limit_and_no_other() -> None:
    routes = [
        (main.API_PREFIX + route.path, route)
        for router in main.OPEN_ROUTERS + main.PROTECTED_ROUTERS
        for route in router.routes
        if isinstance(route, APIRoute)
    ]
    assert len(routes) > 30
    expected = {path for path, route in routes if _endpoint_takes_upload(route)}
    assert expected == {"/api/predict/single", "/api/predict/batch"}
    assert {path for path, route in routes if takes_upload(route)} == expected
    # The app's middleware has exactly these, and so does the API's own description.
    assert sorted(_app_upload_patterns()) == sorted(compile_path(path)[0].pattern for path in expected)
    spec = main.app.openapi()
    multipart = {
        path
        for path, operations in spec["paths"].items()
        for operation in operations.values()
        if "multipart/form-data" in (operation.get("requestBody") or {}).get("content", {})
    }
    assert multipart == expected


@pytest.fixture
def small_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_upload_megabytes", 3)
    monkeypatch.setattr(settings, "max_json_megabytes", 1)


def test_a_json_route_sent_a_multipart_content_type_gets_the_json_limit(small_limits: None) -> None:
    client = TestClient(main.app)
    body = b"x" * (2 * MB)
    for content_type in ("multipart/form-data; boundary=x", "application/json", "text/plain"):
        response = client.post("/api/chat/completions", content=body, headers={"content-type": content_type})
        assert response.status_code == 413, content_type


def test_an_upload_route_keeps_the_upload_limit_for_multipart_only(small_limits: None) -> None:
    router = APIRouter()

    @router.post("/upload")
    async def upload(image: UploadFile = File(...)) -> dict[str, int]:
        return {"length": len(await image.read())}

    app = FastAPI()
    app.add_middleware(RequestBodyLimit, upload_paths=upload_paths([("", router)]))
    app.include_router(router)

    client = TestClient(app)
    data = b"x" * (2 * MB)
    response = client.post("/upload", files={"image": ("p.png", data, "image/png")})
    assert response.status_code == 200 and response.json() == {"length": len(data)}
    as_json = client.post("/upload", content=data, headers={"content-type": "application/json"})
    assert as_json.status_code == 413


def test_a_validation_error_does_not_echo_a_large_body() -> None:
    client = TestClient(main.app)
    sent = "x" * 500_000
    response = client.post("/api/chat/completions", json={"messages": sent})
    assert response.status_code == 422
    assert len(response.content) < 5_000
    detail = response.json()["detail"]
    assert detail and all({"type", "loc", "msg"} <= set(error) for error in detail)


def test_a_small_input_is_still_shown_in_the_validation_error() -> None:
    client = TestClient(main.app)
    response = client.post("/api/chat/completions", json={"messages": 12})
    assert response.status_code == 422
    assert any(error.get("input") == 12 for error in response.json()["detail"])
