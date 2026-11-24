"""The analytics session token: a subject that must be the operator, and a word about a short secret.

Found by a security review: a token signed with the right secret was accepted
for any subject, "someone-else" included; and JWT_SECRET had no minimum length
(API_TOKEN has one), so "secret" signed every session without a word. A short
secret is logged at startup (test_jwt_secret_startup.py), not refused: an
installation that has one keeps starting.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt

from app.config import Settings, settings
from app.routers import analytics

SECRET = "s" * 40


def test_a_short_secret_does_not_stop_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "secret")
    assert Settings(_env_file=None).jwt_secret == "secret"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(settings, "analytics_username", "operator")
    monkeypatch.setattr(settings, "analytics_password", "the-right-password")
    monkeypatch.setattr(settings, "jwt_secret", SECRET)
    app = FastAPI()
    app.include_router(analytics.router)
    return TestClient(app)


def _token(subject: str) -> str:
    expires = datetime.now(timezone.utc) + timedelta(minutes=5)
    return jwt.encode({"sub": subject, "exp": expires}, SECRET, algorithm="HS256")


def test_a_token_for_another_subject_is_refused(client: TestClient) -> None:
    response = client.get("/analytics/data", headers={"Authorization": f"Bearer {_token('someone-else')}"})
    assert response.status_code == 401


def test_the_operators_own_token_is_accepted(client: TestClient) -> None:
    login = client.post("/analytics/login", json={"username": "operator", "password": "the-right-password"})
    assert login.status_code == 200
    response = client.get("/analytics/data", headers={"Authorization": f"Bearer {login.json()['token']}"})
    assert response.status_code == 200
    assert client.get("/analytics/data", headers={"Authorization": f"Bearer {_token('operator')}"}).status_code == 200
