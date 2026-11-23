"""Chat request bounds are checked by the route, not discovered upstream."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import chat


@pytest.mark.parametrize("temperature", [-0.5, 2.5, 50])
def test_an_out_of_range_temperature_is_a_422(temperature: float) -> None:
    """SAIA refused it, and the route turned the refusal into a 500."""
    app = FastAPI()
    app.include_router(chat.router)
    response = TestClient(app, raise_server_exceptions=False).post(
        "/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": False, "temperature": temperature},
    )
    assert response.status_code == 422
    assert "temperature" in response.text
