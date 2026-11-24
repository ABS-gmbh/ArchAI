"""Index router — trigger and inspect RAG indexing for pipeline runs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.services.public_errors import RequestRefused
from app.db import pipeline_db

from app.routers.errors import unexpected_error
from app.services import rag_store

router = APIRouter(tags=["index"])


@router.post("/index/run/{run_id}")
def index_run(run_id: str) -> dict[str, Any]:
    """Embed all chunks for *run_id* and upsert into the vector store."""
    try:
        return rag_store.index_run(run_id)
    except RequestRefused as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        unexpected_error(exc, what="Indexing", provider="embedding provider")


@router.get("/index/run/{run_id}/status")
def index_status(run_id: str) -> dict[str, Any]:
    """Check whether *run_id* has been indexed."""
    # An unknown run reported zero chunks, indexed or not, with 200.
    if pipeline_db.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    try:
        return rag_store.is_run_indexed(run_id)
    except Exception as exc:
        unexpected_error(exc, what="Status check", provider="embedding provider")


@router.delete("/index/run/{run_id}")
def delete_index(run_id: str) -> dict[str, Any]:
    """Remove all vectors for *run_id* from the store."""
    try:
        deleted = rag_store.delete_run(run_id)
        return {"run_id": run_id, "deleted": deleted}
    except Exception as exc:
        unexpected_error(exc, what="Deletion", provider="embedding provider")


@router.post("/index/search")
def search_index(
    query: str = Query(..., min_length=1),
    # Bounded like /rag/debug/retrieve: a negative or huge top_k reached the
    # vector store as n_results.
    top_k: int | None = Query(None, ge=1, le=50),
    run_id: str | None = None,
    asset_ref: str | None = None,
) -> dict[str, Any]:
    """Retrieve the most relevant chunks for a free-text *query*.

    Optionally filter by *run_id* and/or *asset_ref*.
    """
    try:
        run_ids = [run_id] if run_id else None
        hits = rag_store.retrieve_chunks(
            query, top_k=top_k, run_ids=run_ids, asset_ref=asset_ref,
        )
        entity_hits = rag_store.retrieve_entities(
            query, top_k=top_k, run_ids=run_ids, asset_ref=asset_ref,
        )
        return {
            "query": query,
            "top_k": top_k or 5,
            "mode": hits.mode,
            "notes": [*hits.notes, *entity_hits.notes],
            "results": list(hits),
            "entity_results": list(entity_hits),
        }
    except Exception as exc:
        unexpected_error(exc, what="Search", provider="embedding provider")
