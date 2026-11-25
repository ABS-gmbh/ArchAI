"""Which model answers when nothing is configured.

Chosen from a measurement against the provider (2026-11-25, 79 questions on
the same retrieved evidence, thinking off): qwen3.5-397b-a17b answered 61 of
the 62 answerable questions correctly and none wrongly, the former default
qwen3-30b-a3b-instruct-2507 47 correctly and 4 wrongly. The backend's default,
its fallback when the setting is empty, .env.example and the frontend's
default must name the same model, and it must be one that is asked not to
think: with thinking on its first token came after a median 20 s.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.config import Settings, settings
from app.services import model_router, provider_thinking

CHAT_MODEL = "qwen3.5-397b-a17b"
BACKEND = Path(__file__).resolve().parents[1]


def test_the_default_chat_model_is_the_measured_one() -> None:
    assert Settings.model_fields["chat_rag_model"].default == CHAT_MODEL
    assert Settings.model_fields["archai_chat_ai_model"].default == CHAT_MODEL


def test_an_empty_setting_falls_back_to_the_same_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "chat_rag_model", "")
    monkeypatch.setattr(settings, "archai_chat_ai_model", "")
    assert model_router.get_task_model_assignments().chat_rag_model == CHAT_MODEL


def test_the_example_env_names_the_same_model() -> None:
    example = (BACKEND / ".env.example").read_text(encoding="utf-8")
    for name in ("CHAT_RAG_MODEL", "ARCHAI_CHAT_AI_MODEL"):
        assert re.search(rf"^{name}=(.*)$", example, re.MULTILINE).group(1).strip() == CHAT_MODEL


def test_the_frontend_starts_with_the_same_model() -> None:
    workspace = BACKEND.parent / "frontend" / "src" / "components" / "workspace" / "DocumentChatWorkspace.tsx"
    match = re.search(r'^const DEFAULT_CHAT_MODEL_ID = "([^"]+)";$', workspace.read_text(encoding="utf-8"), re.MULTILINE)
    assert match and match.group(1) == CHAT_MODEL


def test_the_default_chat_model_is_asked_not_to_think(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "chat_disable_thinking", Settings.model_fields["chat_disable_thinking"].default)
    monkeypatch.setattr(
        settings, "chat_disable_thinking_except", Settings.model_fields["chat_disable_thinking_except"].default
    )
    provider_thinking.reset()
    assert provider_thinking.applies(CHAT_MODEL)
