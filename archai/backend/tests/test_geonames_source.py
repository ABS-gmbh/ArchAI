"""GeoNames is off until an account is configured, and its errors are logged (audit AL-12).

The default username was the shared "demo" account, which is over its daily
quota: every search came back as {"status": {"message": "the daily limit of
20000 credits for demo has been exceeded...", "value": 18}} (checked live on
2026-10-30), was read as zero candidates, and was still counted as a GeoNames
call. The default was also plain HTTP, with the username in the query string.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings, settings
from app.services import authority_sources

QUOTA = {"status": {"message": "the daily limit of 20000 credits for demo has been exceeded.", "value": 18}}
ROW = {"geonameId": 2988507, "toponymName": "Paris", "countryName": "France", "adminName1": "Île-de-France", "fcodeName": "capital of a political entity"}


def _record(monkeypatch, payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake(url: str, params: dict[str, Any], *, timeout: int = 10) -> dict[str, Any]:
        calls.append((url, params))
        return payload

    monkeypatch.setattr(authority_sources, "_http_json", fake)
    return calls


def test_the_defaults_are_no_account_over_https():
    assert Settings.model_fields["geonames_username"].default == ""
    assert Settings.model_fields["geonames_base_url"].default.startswith("https://")


def test_without_an_account_nothing_is_requested(monkeypatch):
    monkeypatch.setattr(settings, "geonames_username", "")
    calls = _record(monkeypatch, {"geonames": [ROW]})
    assert authority_sources.search_geonames("Paris", ent_type="place") == []
    assert calls == []


def test_an_error_status_is_logged_not_read_as_a_miss(monkeypatch, caplog):
    monkeypatch.setattr(settings, "geonames_username", "demo")
    _record(monkeypatch, QUOTA)
    with caplog.at_level(logging.WARNING, logger="archai.authority_sources"):
        assert authority_sources.search_geonames("Paris", ent_type="place") == []
    assert "daily limit" in caplog.text


def test_rows_are_read_with_an_account(monkeypatch):
    monkeypatch.setattr(settings, "geonames_username", "archai")
    monkeypatch.setattr(settings, "geonames_base_url", "https://secure.geonames.org")
    calls = _record(monkeypatch, {"geonames": [ROW]})
    rows = authority_sources.search_geonames("Paris", ent_type="place")
    assert [row["label"] for row in rows] == ["Paris"]
    assert calls[0][0] == "https://secure.geonames.org/searchJSON"
    assert calls[0][1]["username"] == "archai"
