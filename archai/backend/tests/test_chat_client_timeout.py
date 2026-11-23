"""The chat client has a bounded timeout, set by CHAT_AI_TIMEOUT_SECONDS.

It was built with no timeout, so a chat or translation request that the server
never answered waited the OpenAI SDK's default of 600 s, three times over with
the SDK's two retries, while the HTTP request that triggered it hung.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings, settings
from app.services import chat_ai


def test_the_client_carries_the_configured_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "chat_ai_api_key", "test-key")
    monkeypatch.setattr(settings, "chat_ai_timeout_seconds", 42.0)
    client = chat_ai._create_client()
    assert client.timeout == 42.0


def test_the_default_is_well_below_the_sdk_default() -> None:
    assert 0 < Settings.model_fields["chat_ai_timeout_seconds"].default < 600


def test_it_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_AI_TIMEOUT_SECONDS", "75")
    assert Settings(_env_file=None).chat_ai_timeout_seconds == 75


@pytest.mark.parametrize("raw", ["0", "-5"])
def test_a_non_positive_timeout_is_refused_by_name(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("CHAT_AI_TIMEOUT_SECONDS", raw)
    with pytest.raises(ValidationError, match="CHAT_AI_TIMEOUT_SECONDS must be greater than 0"):
        Settings(_env_file=None)


def test_a_non_number_is_refused_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_AI_TIMEOUT_SECONDS", "slow")
    with pytest.raises(ValidationError, match="chat_ai_timeout_seconds"):
        Settings(_env_file=None)
