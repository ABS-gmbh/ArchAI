"""Which model translates when nothing is configured.

The default used to name llama-3.3-70b-instruct. The provider stopped
offering it (403 "Not authorized to access this Model", 2026-11-25), so with
the default configuration every translation failed. Translations now use the
chat model unless TRANSLATION_MODEL names another.
"""

from __future__ import annotations

import pytest

from app.config import Settings, settings
from app.services import model_router


def test_the_default_names_no_model_of_its_own() -> None:
    assert Settings.model_fields["translation_model"].default == ""


def test_without_a_setting_the_chat_model_translates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "translation_model", "")
    monkeypatch.setattr(settings, "chat_rag_model", "the-chat-model")
    assignments = model_router.get_task_model_assignments()
    assert assignments.translation_model == "the-chat-model"
    assert model_router.model_for_chat_stage("translation") == "the-chat-model"


def test_a_configured_translation_model_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "translation_model", " a-translation-model ")
    monkeypatch.setattr(settings, "chat_rag_model", "the-chat-model")
    assert model_router.get_task_model_assignments().translation_model == "a-translation-model"
