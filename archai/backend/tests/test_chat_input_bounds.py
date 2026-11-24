"""A chat message cannot stall the process: the inspection regex is linear and content is bounded."""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from app.services import chat_ai


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("show me the rag evidence", True),
        ("Please DISPLAY the retrieval", True),
        ("what did you retrieve", True),
        ("inspect\nretrieval", False),  # trigger and target on one line, as before
        ("evidence first, then show", False),  # the target must follow the trigger
        ("showcase the evidence", False),  # whole words only
        ("who is named in the text?", False),
    ],
)
def test_inspection_queries_are_recognised_as_before(text: str, expected: bool) -> None:
    assert chat_ai._is_rag_inspection_query(text) is expected


def test_the_inspection_check_is_linear_on_adversarial_input() -> None:
    # "show " repeated made the old pattern backtrack quadratically: 5.4 s at 80k
    # characters, holding the GIL, so every other request stalled with it.
    started = time.perf_counter()
    assert chat_ai._is_rag_inspection_query("show " * 16000) is False
    assert chat_ai._is_rag_inspection_query("show " * 16000 + "evidence") in (True, False)
    assert time.perf_counter() - started < 0.2


def test_message_content_is_bounded_with_a_sentence_the_workspace_shows() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.config import settings
    from app.routers import chat

    limit = settings.chat_max_message_chars
    assert limit >= 200_000  # ten times the largest message the workspace builds, at least
    app = FastAPI()
    app.include_router(chat.router)
    client = TestClient(app)
    for content in ("a" * (limit + 1), [{"type": "text", "text": "a" * (limit + 1)}]):
        response = client.post("/chat/completions", json={"stream": False, "messages": [{"role": "user", "content": content}]})
        assert response.status_code == 422
        assert response.json()["detail"] == f"The message is longer than the server accepts ({limit} characters)."
    # An attached page image is bounded by the request body limit, not by this one.
    image = [{"type": "text", "text": "read"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * (limit * 2)}}]
    assert chat.ChatCompletionRequest(messages=[{"role": "user", "content": image}]).messages[0].text_length() == 4
