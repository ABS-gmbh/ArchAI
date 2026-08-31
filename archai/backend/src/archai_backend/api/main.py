"""ArchAI prototype backend — MOCK IMPLEMENTATION, NOT A PIPELINE.

This package is scaffolding: 41 of its 47 modules contain only
`# TODO: implement`. The ingest endpoint does not run OCR. It calls
`_simulate_pipeline`, which sleeps through fake stage names and inserts
hardcoded English sentences (`_SIMULATED_SPAN_TEXTS`) as document spans.

Nothing here is derived from a manuscript image. The real pipelines are:
  * archai/vendor/layout/backend  — the production FastAPI OCR service
  * src/archai_ocr                — the YOLO + Kraken command-line pipeline

Because the fabricated output is indistinguishable from genuine results, the
mock refuses to run unless ARCHAI_ALLOW_MOCK_BACKEND=1 is set explicitly.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from archai_backend.api.schemas.requests import ChatRequest
from archai_backend.api.schemas.responses import ChatResponse
from archai_backend.config.logging import configure_logging
from archai_backend.config.settings import RAW_IMAGES_DIR
from archai_backend.store import db


configure_logging()

# Explicit opt-in required: see the module docstring.
MOCK_BACKEND_ENABLED = os.getenv("ARCHAI_ALLOW_MOCK_BACKEND") == "1"

app = FastAPI(
    title="ArchAI Backend (MOCK — returns fabricated data)",
    description=(
        "Prototype scaffolding. Ingest returns hardcoded text, not OCR output. "
        "Set ARCHAI_ALLOW_MOCK_BACKEND=1 to enable. Use "
        "archai/vendor/layout/backend for the real service."
    ),
)


def _require_mock_opt_in() -> None:
    if not MOCK_BACKEND_ENABLED:
        raise HTTPException(
            status_code=501,
            detail=(
                "This prototype backend is a mock: it returns fabricated spans "
                "rather than running OCR. Set ARCHAI_ALLOW_MOCK_BACKEND=1 to run it "
                "anyway, or use archai/vendor/layout/backend for the real pipeline."
            ),
        )

db.init_db()


def _save_upload(doc_id: str, page_id: str, upload: UploadFile) -> str:
    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(upload.filename or "image").suffix or ".bin"
    dest_dir = RAW_IMAGES_DIR / doc_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{page_id}{ext}"
    with dest_path.open("wb") as f:
        f.write(upload.file.read())
    return str(dest_path)


def _simulate_pipeline(job_id: str, doc_id: str, page_id: str) -> None:
    stages = [
        ("segmented_stub", 15, "Segmenting"),
        ("cropped_stub", 35, "Cropping"),
        ("htr_stub", 55, "Recognizing"),
        ("spanned", 75, "Building spans"),
        ("indexed", 90, "Indexing"),
        ("ready", 100, "Ready"),
    ]
    for stage, progress, message in stages:
        time.sleep(0.6)
        status = "running" if stage != "ready" else "complete"
        db.update_job(job_id, status=status, stage=stage, progress=progress, message=message)
        if stage == "spanned":
            _create_spans(doc_id, page_id)


# Fabricated text. Not derived from any image. See the module docstring.
_SIMULATED_SPAN_TEXTS = [
    "The document notes a transfer recorded in the ledger.",
    "A marginal note references a shipment date and location.",
    "The scribe lists three witnesses to the agreement.",
    "The entry mentions a payment of four florins.",
    "A correction indicates the date was later amended.",
    "The record describes a boundary near the river bend.",
    "An endorsement cites approval by the magistrate.",
    "The notation includes a seal impression remark.",
]


def _create_spans(doc_id: str, page_id: str) -> None:
    span_texts = list(_SIMULATED_SPAN_TEXTS)
    random.shuffle(span_texts)
    count = random.randint(3, 8)
    for i in range(count):
        span_id = str(uuid4())
        x1 = round(random.uniform(10, 100), 2)
        y1 = round(random.uniform(10, 200), 2)
        x2 = round(x1 + random.uniform(100, 300), 2)
        y2 = round(y1 + random.uniform(20, 60), 2)
        db.insert_span(
            span_id=span_id,
            doc_id=doc_id,
            page_id=page_id,
            text=span_texts[i],
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest/image")
def ingest_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_id: Optional[str] = None,
    page_id: Optional[str] = None,
) -> dict:
    # This endpoint fabricates its results; refuse unless explicitly opted in.
    _require_mock_opt_in()

    doc_id = doc_id or str(uuid4())
    page_id = page_id or str(uuid4())

    image_path = _save_upload(doc_id, page_id, file)

    db.create_document(doc_id)
    db.create_page(page_id, doc_id, image_path)

    job_id = str(uuid4())
    db.create_job(job_id, doc_id, page_id)

    background_tasks.add_task(_simulate_pipeline, job_id, doc_id, page_id)

    return {"job_id": job_id, "doc_id": doc_id, "page_id": page_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    row = db.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "status": row["status"],
        "stage": row["stage"],
        "progress": row["progress"],
        "message": row["message"],
    }


@app.get("/pages/{page_id}/image")
def get_page_image(page_id: str, doc_id: str) -> FileResponse:
    image_path = db.get_page_image_path(doc_id, page_id)
    if not image_path:
        raise HTTPException(status_code=404, detail="Page not found")
    return FileResponse(image_path)


@app.get("/evidence/span/{span_id}")
def get_span(span_id: str) -> dict:
    row = db.get_span(span_id)
    if not row:
        raise HTTPException(status_code=404, detail="Span not found")
    return {
        "span_id": row["span_id"],
        "doc_id": row["doc_id"],
        "page_id": row["page_id"],
        "text": row["text"],
        "bbox_xyxy": [row["x1"], row["y1"], row["x2"], row["y2"]],
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    spans = db.get_spans(request.doc_id, request.page_id, limit=5)
    if not spans:
        return ChatResponse(
            answer="No evidence spans are available yet for this document.",
            citations=[],
        )

    sentences = []
    citations = []
    for row in spans:
        sentences.append(f"{row['text']} [{row['span_id']}]")
        citations.append(row["span_id"])

    answer = " ".join(sentences)
    return ChatResponse(answer=answer, citations=citations)
