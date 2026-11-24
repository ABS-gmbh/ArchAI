from __future__ import annotations

import sqlite3

from fastapi import APIRouter, HTTPException, Query

from app.agents.evidence_store_agent import create_span, list_spans
from app.schemas.agents_ocr import EvidenceSpanCreateRequest, EvidenceSpanRecord

router = APIRouter(tags=["evidence"])


@router.post("/evidence/spans", response_model=EvidenceSpanRecord)
def create_evidence_span(payload: EvidenceSpanCreateRequest) -> EvidenceSpanRecord:
    try:
        return create_span(payload)
    except sqlite3.IntegrityError as exc:
        # A span must belong to an existing run, and its chunk and mention, when
        # given, must exist too. A missing or unknown run_id answered 500.
        raise HTTPException(
            status_code=422,
            detail=(
                "The evidence span must belong to an existing OCR run (run_id), and its chunk_id "
                "and mention_id, if given, must exist in that run."
            ),
        ) from exc


# Reads that returned every row are capped (security review S3): the first
# `limit` rows in their usual order, 1,000 unless asked otherwise.
DEFAULT_READ_LIMIT = 1000
MAX_READ_LIMIT = 10000


@router.get("/evidence/spans/{page_id}", response_model=list[EvidenceSpanRecord])
def get_evidence_spans(
    page_id: str, limit: int = Query(default=DEFAULT_READ_LIMIT, ge=1, le=MAX_READ_LIMIT)
) -> list[EvidenceSpanRecord]:
    """The page's evidence spans, oldest first: at most *limit* (default 1,000, at most 10,000)."""
    return list_spans(page_id, limit=limit)
