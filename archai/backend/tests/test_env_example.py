"""Every variable .env.example names is one the backend reads.

SAIA_MODEL_PROBE was listed with a value and read nowhere, so setting it did
nothing. A variable counts as read when it is a Settings field or the code
looks it up with os.getenv / os.environ.get.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.config import Settings

BACKEND = Path(__file__).resolve().parents[1]
_ASSIGNMENT = re.compile(r"^#?\s*([A-Z][A-Z0-9_]+)=", re.MULTILINE)
_ENV_LOOKUP = re.compile(r"os\.(?:getenv|environ\.get)\(\s*[\"']([A-Z][A-Z0-9_]+)[\"']")


def _read_by_the_backend() -> set[str]:
    names = {name.upper() for name in Settings.model_fields}
    for path in (BACKEND / "app").rglob("*.py"):
        names.update(_ENV_LOOKUP.findall(path.read_text(encoding="utf-8")))
    return names


def test_every_example_variable_is_read() -> None:
    listed = set(_ASSIGNMENT.findall((BACKEND / ".env.example").read_text(encoding="utf-8")))
    assert listed, "no variables found in .env.example"
    assert sorted(listed - _read_by_the_backend()) == []


def test_the_operational_settings_are_documented() -> None:
    listed = set(_ASSIGNMENT.findall((BACKEND / ".env.example").read_text(encoding="utf-8")))
    operational = {
        "ARCHAI_DB_PATH",
        "CHROMA_PERSIST_DIR",
        "MAX_UPLOAD_MEGABYTES",
        "MAX_EXTRACTED_MEGABYTES",
        "TASK_TTL_MINUTES",
        "LOG_LEVEL",
        "LOG_FORMAT",
        "CHAT_AI_TIMEOUT_SECONDS",
        "ENABLE_TEST_OCR_FIXTURES",
        "OCR_RUN_RESERVATION_TTL_SECONDS",
        "OCR_RUN_RESERVATION_MAX_LIVE",
        "OCR_TILE_CONCURRENCY",
        "OCR_BACKEND_DEFAULT",
        "GLMOCR_DEVICE",
        "RAG_TOP_K",
        "RAG_CHUNK_WINDOW_LINES",
        "RAG_CHUNK_OVERLAP_LINES",
        "RAG_LEXICAL_FUSION",
        "RAG_LEXICAL_WEIGHT",
        "RAG_LEXICAL_WEIGHT_LOCAL",
        "RAG_LEXICAL_MAX_CHUNKS",
    }
    assert sorted(operational - listed) == []
