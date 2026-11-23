from __future__ import annotations

import sqlite3

from fastapi import APIRouter, HTTPException

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


@router.get("/evidence/spans/{page_id}", response_model=list[EvidenceSpanRecord])
def get_evidence_spans(page_id: str) -> list[EvidenceSpanRecord]:
    return list_spans(page_id)
