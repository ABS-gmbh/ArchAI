"""Authority-linking router — trigger and inspect entity linking for pipeline runs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.services.public_errors import RequestRefused
from app.routers.errors import unexpected_error

router = APIRouter(tags=["authority-linking"])


# Plain ``def`` handlers, so FastAPI runs them in its threadpool: linking makes
# blocking Wikidata/VIAF/GeoNames requests with multi-second timeouts, and as
# ``async def`` it froze every other request on the process while it ran.
@router.post("/authority/link/{run_id}")
def link_run(run_id: str, force_refresh: bool = False) -> dict[str, Any]:
    """Run authority linking (Wikidata search + scoring) for all mentions in *run_id*."""
    from app.db import pipeline_db
    from app.services.authority_linking import run_authority_linking

    # run_authority_linking answers an unknown run with an empty result, so the
    # route returned 200 "no mentions found" for a run that does not exist.
    if pipeline_db.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    try:
        return run_authority_linking(run_id, force_refresh=force_refresh)
    except RequestRefused as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        unexpected_error(exc, what="Authority linking")


@router.get("/authority/report/{run_id}")
def linking_report(run_id: str) -> dict[str, Any]:
    """Return a pre-built linking report for *run_id*.

    Reads existing candidates from the DB (does NOT re-run linking).
    """
    from app.services.authority_linking import build_report_from_db

    try:
        return build_report_from_db(run_id)
    except RequestRefused as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        unexpected_error(exc, what="The linking report")
