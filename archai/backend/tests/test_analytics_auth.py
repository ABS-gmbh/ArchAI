"""Analytics auth has no built-in credentials and fails closed (audit SEC-01).

The audit found admin / layout2024 and a fixed JWT secret as defaults in
app/config.py. They were removed earlier; these tests keep them out.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt

from app.config import Settings
from app.routers import analytics


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(analytics.router)
    return TestClient(app)


def test_there_are_no_default_credentials_or_secret():
    fields = Settings.model_fields
    assert (fields["analytics_username"].default, fields["analytics_password"].default) == ("", "")
    assert fields["jwt_secret"].default == ""


def test_login_is_refused_until_credentials_are_configured(client, monkeypatch):
    monkeypatch.setattr(analytics.settings, "analytics_username", "")
    monkeypatch.setattr(analytics.settings, "analytics_password", "")
    response = client.post("/analytics/login", json={"username": "admin", "password": "layout2024"})
    assert response.status_code == 503


def test_wrong_and_right_credentials(client, monkeypatch):
    monkeypatch.setattr(analytics.settings, "analytics_username", "operator")
    monkeypatch.setattr(analytics.settings, "analytics_password", "s3cret-pass")
    assert client.post("/analytics/login", json={"username": "operator", "password": "nope"}).status_code == 401
    ok = client.post("/analytics/login", json={"username": "operator", "password": "s3cret-pass"})
    assert ok.status_code == 200 and ok.json()["token"]


def test_a_token_signed_with_the_old_default_secret_is_rejected(client, monkeypatch):
    monkeypatch.setattr(analytics.settings, "analytics_username", "operator")
    monkeypatch.setattr(analytics.settings, "analytics_password", "s3cret-pass")
    monkeypatch.setattr(analytics.settings, "jwt_secret", "")
    forged = jwt.encode(
        {"sub": "admin", "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
        "change-this-to-a-random-string",
        algorithm="HS256",
    )
    response = client.get("/analytics/data", headers={"Authorization": f"Bearer {forged}"})
    assert response.status_code == 401
