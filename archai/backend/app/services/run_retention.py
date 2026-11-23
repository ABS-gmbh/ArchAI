"""Retention for pipeline runs (audit DB-07): which runs may go, and removing them.

Nothing here runs unless asked. scripts/prune_runs.py reports by default and
deletes only with --apply; the scheduled sweep runs only when
RUN_RETENTION_DAYS is above 0, and it is 0 by default.

A run is deleted only when every one of these holds:

* it was last updated more than ``older_than_days`` days ago;
* it has finished (COMPLETED, FAILED or FAILED_QUALITY) - a run in progress
  is never touched, however old;
* it has a page identity - its image hash, or its asset_ref without one, as
  for search. A run with neither cannot be matched to its page and is kept;
* it is not one of the runs a page relies on: the page's newest run, its
  newest COMPLETED run (the newest may have failed), and the run search
  serves for it (pipeline_db.searchable_runs, used directly so the two rules
  cannot drift);
* no OCR benchmark reference points at it - deleting the run would delete
  the reference text with it;
* it carries no annotation: an evidence span the pipeline did not write
  (meta_json.source other than ocr_chunk, authority_linking or
  authority_linking.deferred), which may be a person's work. Only the script,
  with --include-annotated, may delete these; the sweep never does;
* its id does not appear in the thesis showcase bundle.

Each kept run is reported with the first rule that kept it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.db import pipeline_db

_LOGGER = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "FAILED_QUALITY"})
# meta_json.source of the evidence spans the pipeline writes itself; any other
# span was posted to /api/evidence/spans and counts as an annotation.
PIPELINE_SPAN_SOURCES = ("ocr_chunk", "authority_linking", "authority_linking.deferred")
SHOWCASE_DIR = Path(__file__).resolve().parents[4] / "artifacts" / "thesis_showcase"
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


@dataclass
class RetentionPlan:
    cutoff: str
    expired: list[str] = field(default_factory=list)
    kept: dict[str, list[str]] = field(default_factory=dict)

    def keep(self, reason: str, run_id: str) -> None:
        self.kept.setdefault(reason, []).append(run_id)


def showcase_run_ids(directory: Path = SHOWCASE_DIR) -> set[str]:
    """Every id-shaped string in the showcase bundle's text files."""
    found: set[str] = set()
    if not directory.is_dir():
        return found
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".csv", ".md", ".txt"}:
            try:
                found.update(_UUID.findall(path.read_text(encoding="utf-8", errors="ignore")))
            except OSError:
                continue
    return found


def plan_retention(
    older_than_days: int,
    *,
    now: datetime | None = None,
    protected: set[str] | None = None,
    include_annotated: bool = False,
) -> RetentionPlan:
    """Decide which runs are expired, without deleting anything."""
    if older_than_days < 1:
        raise ValueError(f"older_than_days must be at least 1, got {older_than_days}")
    cutoff = ((now or datetime.now(UTC)) - timedelta(days=older_than_days)).isoformat()
    protected = showcase_run_ids() if protected is None else protected
    plan = RetentionPlan(cutoff=cutoff)

    pipeline_db._init_db_if_needed()
    with pipeline_db._connect() as conn:
        rows = conn.execute(
            "SELECT run_id, status, updated_at, created_at, asset_ref, asset_sha256 FROM pipeline_runs "
            "ORDER BY created_at DESC, run_id DESC"
        ).fetchall()
        benchmarked = {row[0] for row in conn.execute("SELECT DISTINCT run_id FROM ocr_benchmark_references")}
        placeholders = ",".join("?" for _ in PIPELINE_SPAN_SOURCES)
        annotated = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT run_id FROM evidence_spans WHERE NOT ("
                f"meta_json IS NOT NULL AND json_valid(meta_json) "
                f"AND json_extract(meta_json, '$.source') IN ({placeholders}))",
                PIPELINE_SPAN_SOURCES,
            )
        }

    searchable = {str(row["run_id"]) for row in pipeline_db.searchable_runs()}
    newest_of_page: set[str] = set()
    newest_completed_of_page: set[str] = set()
    seen_pages: set[str] = set()
    seen_completed_pages: set[str] = set()
    for row in rows:
        page = _page_key(row)
        if page is None:
            continue
        if page not in seen_pages:
            seen_pages.add(page)
            newest_of_page.add(row["run_id"])
        if str(row["status"] or "").upper() == "COMPLETED" and page not in seen_completed_pages:
            seen_completed_pages.add(page)
            newest_completed_of_page.add(row["run_id"])

    for row in rows:
        run_id = row["run_id"]
        if str(row["updated_at"] or row["created_at"]) >= cutoff:
            plan.keep("recent", run_id)
        elif str(row["status"] or "").upper() not in TERMINAL_STATUSES:
            plan.keep("not_finished", run_id)
        elif _page_key(row) is None:
            plan.keep("no_page_key", run_id)
        elif run_id in newest_of_page:
            plan.keep("newest_of_page", run_id)
        elif run_id in newest_completed_of_page:
            plan.keep("newest_completed_of_page", run_id)
        elif run_id in searchable:
            plan.keep("searchable", run_id)
        elif run_id in benchmarked:
            plan.keep("benchmark_reference", run_id)
        elif run_id in annotated and not include_annotated:
            plan.keep("annotated", run_id)
        elif run_id in protected:
            plan.keep("thesis_showcase", run_id)
        else:
            plan.expired.append(run_id)
    return plan


def _page_key(row: Any) -> str | None:
    """The page a run analysed: its image hash, else its asset_ref; None when it has neither."""
    sha = str(row["asset_sha256"] or "").strip()
    if sha:
        return sha
    ref = str(row["asset_ref"] or "").strip()
    return f"asset_ref:{ref}" if ref else None


def delete_runs(run_ids: list[str], *, cutoff: str) -> dict[str, int]:
    """Delete each run's vectors, then the run itself.

    Each run is read again just before it is deleted and skipped if it is no
    longer finished or was updated after *cutoff*: a re-run since planning
    makes it current again. Vectors go first, so a failure in between leaves a
    run without vectors, which can be re-indexed, never vectors without a run;
    a run whose vectors cannot be deleted is kept.
    """
    from app.services import rag_store

    runs = vectors = 0
    for run_id in run_ids:
        current = pipeline_db.get_run(run_id)
        if current is None:
            continue
        if str(current.get("status") or "").upper() not in TERMINAL_STATUSES or str(
            current.get("updated_at") or current.get("created_at")
        ) >= cutoff:
            _LOGGER.info("Retention: run %s changed since planning; keeping it", run_id)
            continue
        try:
            vectors += rag_store.delete_run(run_id)
        except Exception:  # noqa: BLE001 - a store failure must not leave the row behind silently
            _LOGGER.exception("Retention: could not delete the vectors of run %s; keeping the run", run_id)
            continue
        if pipeline_db.delete_run(run_id):
            runs += 1
            _LOGGER.info("Retention: deleted run %s", run_id)
    return {"runs": runs, "vectors": vectors}
