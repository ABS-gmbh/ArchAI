"""Chat endpoints for ArchAI workspace."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.public_errors import RequestRefused
from app.config import settings
from app.agents.label_analysis_agent import LabelAnalysisAgent, LabelAnalysisAgentError
from app.routers.errors import configuration_message, stream_error_message, unexpected_error
from app.services.chat_ai import (
    ChatConfigError,
    create_chat_completion,
    list_available_models,
    stream_chat_completion,
)
from app.services.saia_client import SaiaConfigError

router = APIRouter(tags=["chat"])
_label_analysis_agent_instance: LabelAnalysisAgent | None = None


def _get_label_analysis_agent() -> LabelAnalysisAgent:
    global _label_analysis_agent_instance
    if _label_analysis_agent_instance is None:
        _label_analysis_agent_instance = LabelAnalysisAgent()
    return _label_analysis_agent_instance


class ChatMessage(BaseModel):
    role: str = Field(default="user")
    content: Any

    def text_length(self) -> int:
        """Characters of text the message carries: the string, or all its text parts together.

        The longest part was counted, so forty parts of 200,000 characters each
        passed the cap (second security review, L).
        """
        if isinstance(self.content, str):
            return len(self.content)
        if isinstance(self.content, list):
            return sum(
                len(part["text"]) for part in self.content if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        return 0

    def shape_problem(self) -> str | None:
        """A sentence when the content is neither text nor a list of parts with string text, else None.

        A part whose text was a list passed every check and went on to the
        provider (second security review, L).
        """
        if self.content is None or isinstance(self.content, str):
            return None
        if not isinstance(self.content, list):
            return "A message's content must be text or a list of parts."
        for part in self.content:
            if not isinstance(part, dict):
                return "A message's content must be text or a list of parts."
            if "text" in part and not isinstance(part["text"], str):
                return "A text part's text must be a string."
        return None


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    # The OpenAI-compatible range; outside it SAIA refused the call and the route answered 500.
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    stream: bool = True
    context: dict[str, Any] | None = None


class LabelRegionPayload(BaseModel):
    region_id: str | None = None
    bbox_xyxy: list[float] | None = None
    polygons: list[list[float]] = Field(default_factory=list)


class LabelAnalysisRequest(BaseModel):
    question: str
    label_name: str
    image_b64: str
    regions: list[LabelRegionPayload] = Field(default_factory=list)
    filename: str | None = None
    page_id: str | None = None
    document_id: str | None = None


class LabelAnalysisResponse(BaseModel):
    status: str
    text: str
    label_name: str
    analysis_mode: str | None = None
    model_used: str
    warnings: list[str] = Field(default_factory=list)
    region_count: int
    crop_image_b64: str
    crop_bounds_xyxy: list[int] = Field(default_factory=list)
    ocr_text: str | None = None
    stage_metadata: dict[str, Any] | None = None
    inspection: dict[str, Any] | None = None


@router.get("/chat/models")
async def chat_models() -> dict[str, Any]:
    try:
        # Off the event loop, like every call into the blocking SAIA client
        # here; it froze the process for the whole round trip (audit CHAT-001).
        return await asyncio.to_thread(list_available_models)
    except ChatConfigError as exc:
        raise HTTPException(status_code=422, detail=configuration_message(exc)) from exc
    except Exception as exc:
        unexpected_error(exc, what="Listing the chat models", provider="chat provider")


def _sse_events(payload: ChatCompletionRequest) -> Iterator[str]:
    try:
        for event in stream_chat_completion(
            messages=[msg.model_dump() for msg in payload.messages],
            model=payload.model,
            temperature=payload.temperature,
            context=payload.context,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    except Exception as exc:
        # The stream has started, so the error travels as an event, not a status;
        # it says what a person can act on and leaves the rest to the log.
        error_event = {"type": "error", "error": stream_error_message(exc)}
        yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"


def _check_message_lengths(payload: ChatCompletionRequest) -> None:
    """Refuse a message over CHAT_MAX_MESSAGE_CHARS with a sentence the workspace shows as is.

    Checked here, not as a field validator, so the 422's detail is that sentence
    rather than a list of validation errors.
    """
    for message in payload.messages:
        problem = message.shape_problem()
        if problem is not None:
            raise HTTPException(status_code=422, detail=problem)
    limit = int(settings.chat_max_message_chars)
    if any(message.text_length() > limit for message in payload.messages):
        raise HTTPException(status_code=422, detail=f"The message is longer than the server accepts ({limit} characters).")


@router.post("/chat/completions")
async def chat_completions(payload: ChatCompletionRequest) -> Any:
    _check_message_lengths(payload)
    try:
        if payload.stream:
            return StreamingResponse(
                _sse_events(payload),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        return await asyncio.to_thread(
            create_chat_completion,
            messages=[msg.model_dump() for msg in payload.messages],
            model=payload.model,
            temperature=payload.temperature,
            context=payload.context,
        )
    except RequestRefused as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ChatConfigError as exc:
        raise HTTPException(status_code=422, detail=configuration_message(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        unexpected_error(exc, what="The chat completion", provider="chat provider")


@router.post("/chat/label-analysis", response_model=LabelAnalysisResponse)
async def chat_label_analysis(payload: LabelAnalysisRequest) -> LabelAnalysisResponse:
    try:
        return await asyncio.to_thread(_get_label_analysis_agent().run, payload)
    except LabelAnalysisAgentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ChatConfigError as exc:
        raise HTTPException(status_code=422, detail=configuration_message(exc)) from exc
    except SaiaConfigError as exc:
        raise HTTPException(status_code=422, detail=configuration_message(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        unexpected_error(exc, what="The label analysis", provider="chat provider")
