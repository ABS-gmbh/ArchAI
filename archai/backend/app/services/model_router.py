from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Sequence
from typing import Any

from app.config import settings


@dataclass(frozen=True)
class TaskModelAssignments:
    ocr_model: str
    chat_rag_model: str
    translation_model: str
    label_visual_model: str
    label_visual_fallback_model: str
    verifier_model: str
    embedding_model: str

    def as_payload(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def get_task_model_assignments() -> TaskModelAssignments:
    chat_rag_model = str(
        settings.chat_rag_model
        or settings.archai_chat_ai_model
        or "qwen3.5-397b-a17b"
    ).strip() or "qwen3.5-397b-a17b"
    # No model of its own unless TRANSLATION_MODEL names one: the chat model translates.
    translation_model = str(settings.translation_model or "").strip() or chat_rag_model
    label_visual_model = str(
        settings.label_visual_model
        or settings.saia_label_analysis_model
        or "qwen3-vl-30b-a3b-instruct"
    ).strip() or "qwen3-vl-30b-a3b-instruct"
    label_visual_fallback_model = str(
        settings.label_visual_fallback_model
        or "internvl3.5-30b-a3b"
    ).strip() or "internvl3.5-30b-a3b"
    verifier_model = str(
        settings.paleography_verification_model
        or chat_rag_model
        or "qwen3-235b-a22b"
    ).strip() or chat_rag_model
    embedding_model = str(
        settings.rag_embedding_model
        or "multilingual-e5-large-instruct"
    ).strip() or "multilingual-e5-large-instruct"
    ocr_model = str(
        settings.glmocr_ollama_model
        or "glm-ocr:latest"
    ).strip() or "glm-ocr:latest"

    return TaskModelAssignments(
        ocr_model=ocr_model,
        chat_rag_model=chat_rag_model,
        translation_model=translation_model,
        label_visual_model=label_visual_model,
        label_visual_fallback_model=label_visual_fallback_model,
        verifier_model=verifier_model,
        embedding_model=embedding_model,
    )


def chat_stage_from_context(context: dict[str, Any] | None) -> str:
    if not context:
        return "rag_chat"
    stage = str(
        context.get("chat_stage")
        or context.get("task")
        or context.get("mode")
        or "rag_chat"
    ).strip().lower()
    return stage or "rag_chat"


def model_for_chat_stage(stage: str | None, *, prefer_vision: bool = False) -> str:
    assignments = get_task_model_assignments()
    normalized = str(stage or "rag_chat").strip().lower() or "rag_chat"
    if normalized == "translation":
        return assignments.translation_model
    if normalized in {"label_visual", "visual_label_analysis", "visual_chat"}:
        return assignments.label_visual_model
    if normalized in {"verification", "verifier", "paleography_verification"}:
        return assignments.verifier_model
    if prefer_vision:
        return assignments.label_visual_model
    return assignments.chat_rag_model


def fallback_models_for_stage(stage: str | None, *, prefer_vision: bool = False) -> list[str]:
    assignments = get_task_model_assignments()
    normalized = str(stage or "rag_chat").strip().lower() or "rag_chat"

    ordered: list[str] = []
    if normalized == "translation":
        ordered.extend([assignments.translation_model, assignments.chat_rag_model])
    elif normalized in {"label_visual", "visual_label_analysis", "visual_chat"}:
        ordered.extend([assignments.label_visual_model, assignments.label_visual_fallback_model, assignments.chat_rag_model])
    elif normalized in {"verification", "verifier", "paleography_verification"}:
        ordered.extend([assignments.verifier_model, assignments.chat_rag_model])
    else:
        ordered.append(assignments.chat_rag_model)
        if prefer_vision:
            ordered.extend([assignments.label_visual_model, assignments.label_visual_fallback_model])

    deduped: list[str] = []
    seen: set[str] = set()
    for model_id in ordered:
        clean = str(model_id or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        deduped.append(clean)
    return deduped


# The task models the chat provider serves, and the setting that names each
# (GLM-OCR runs on Ollama; the embedding model answers without being listed).
TASK_MODEL_SETTINGS: dict[str, str] = {
    "chat_rag_model": "CHAT_RAG_MODEL",
    "translation_model": "TRANSLATION_MODEL",
    "label_visual_model": "LABEL_VISUAL_MODEL",
    "label_visual_fallback_model": "LABEL_VISUAL_FALLBACK_MODEL",
    "verifier_model": "PALEOGRAPHY_VERIFICATION_MODEL",
}


def task_models_not_offered(offered: Sequence[str]) -> dict[str, str]:
    """Each configured task model missing from the provider's own list *offered*, by task."""
    available = {str(model).lower() for model in offered}
    payload = get_task_model_assignments().as_payload()
    return {
        task: payload[task]
        for task in TASK_MODEL_SETTINGS
        if payload.get(task) and payload[task].lower() not in available
    }


# Models that read images. Shared by label analysis and the provider OCR (they
# had a test each; label analysis's found no vision model in the provider's
# catalogue and fell back to a text model: 2026-11-25).
#
# Known by name, from the provider's documentation (2026-11-25), and the
# earlier defaults. Most of the provider's vision models carry no telling
# fragment, so a name test alone does not find them. Add a model here when the
# provider documents that it takes images; whether one really accepts an image
# through this API is known only by sending it one.
KNOWN_VISION_MODELS = frozenset(
    {
        "qwen3.5-397b-a17b",
        "qwen3.8-27b",
        "qwen3.6-35b-a3b",
        "gemma-4-31b-it",
        "qwen3-omni-30b-a3b-instruct",
        "glm-5.3-flash",
        "qwen3-vl-30b-a3b-instruct",
        "internvl3.5-30b-a3b",
    }
)
# And by name fragment, for models not listed above. "mistral" was one, and
# matched mistral-large-4-1t-a52b, which the documentation lists as text only.
VISION_MODEL_FRAGMENTS = ("vl", "vision", "internvl", "gemma", "omni", "pixtral")


def is_vision_model(model_id: str) -> bool:
    key = (model_id or "").strip().lower()
    return key in KNOWN_VISION_MODELS or any(fragment in key for fragment in VISION_MODEL_FRAGMENTS)


# Models that check answers when the configured verifier cannot: it is not
# offered, or it is the model that wrote the answer (a model checking its own
# answer is weaker than a second model). Fast ones first, as the user asked
# (2026-11-25: "make the pipeline fast"). Chosen by name from the provider's
# documentation ("flash" variants, few active parameters) and not yet timed:
# worker 2's measurements will set this order and the verifier's default.
FAST_VERIFIER_MODELS = ("deepseek-v4.1-flash-1017", "deepseek-v4-flash-0731", "glm-5.3-flash", "qwen3.6-35b-a3b")


def second_model_for_checking(answering_model: str, offered: Sequence[str], *, exclude: Sequence[str] = ()) -> str | None:
    """A model from *offered* to check an answer written by *answering_model*, never that model; None if there is none."""
    by_lower = {str(model).lower(): str(model) for model in offered}
    skip = {str(answering_model or "").lower(), *(str(model).lower() for model in exclude)}
    for model in FAST_VERIFIER_MODELS:
        if model.lower() in by_lower and model.lower() not in skip:
            return by_lower[model.lower()]
    for model in offered:
        if str(model).lower() not in skip:
            return str(model)
    return None
