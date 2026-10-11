"""Grade stored runs with the current quality gate and record whether they may be searched.

Usage::

    python scripts/regrade_runs.py                       # report what would change
    python scripts/regrade_runs.py --apply               # record the decisions
    python scripts/regrade_runs.py --apply --include-ungraded
    python scripts/regrade_runs.py --db app/archai.sqlite

The vector store searches a run only when the quality gate allowed search on it
(``pipeline_db.searchable_runs``). The pipeline now records that decision on the
run; runs from before fall back to their latest quality report, written by the
gate of the day. Before ABS-gmbh/ArchAI#17 that gate refused 71 of 72 held-out
transcriptions, faithful ones included, so most older runs would stay
unsearchable until analysed again.

This grades each such run's base text - the proofread text when there is one,
the OCR text otherwise, as the pipeline chunks it - the way the pipeline does
(language detection, quality report, ``token_search_allowed``), and with
``--apply`` records the decision and logs a ``REGRADED`` event on the run. A
run is graded as tiled when it has OCR attempts, which only the tiled path
writes.

Runs never graded at all (no quality report) are left alone unless
``--include-ungraded``: in the databases this was written against they are test
runs from before the gate existed. Runs with a recorded decision are left alone
unless ``--all``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import pipeline_db  # noqa: E402


def candidates(*, include_ungraded: bool, everything: bool) -> list[dict[str, Any]]:
    sql = (
        "SELECT r.run_id, r.asset_ref, r.ocr_text, r.proofread_text, r.search_allowed, "
        "(SELECT q.token_search_allowed FROM ocr_quality_reports q WHERE q.run_id = r.run_id "
        "ORDER BY q.pass_idx DESC, q.created_at DESC LIMIT 1) AS report_allowed, "
        "EXISTS (SELECT 1 FROM ocr_attempts a WHERE a.run_id = r.run_id) AS tiled "
        "FROM pipeline_runs r WHERE EXISTS (SELECT 1 FROM chunks c WHERE c.run_id = r.run_id) "
        "ORDER BY r.created_at"
    )
    pipeline_db._init_db_if_needed()
    with pipeline_db._connect() as conn:
        rows = [dict(row) for row in conn.execute(sql).fetchall()]
    return [
        row
        for row in rows
        if (everything or row["search_allowed"] is None) and (include_ungraded or row["report_allowed"] is not None)
    ]


def grade(text: str, *, tiled: bool) -> dict[str, Any]:
    """The decision the pipeline takes on *text*: language, label, and whether it may be searched."""
    from app.routers.ocr import _detect_language_metadata, _normalize_detected_language, _quality_language
    from app.services.ocr_quality import compute_quality_report

    detected, _confidence = _detect_language_metadata(text)
    language = _normalize_detected_language(detected)
    report = compute_quality_report(text, run_id="regrade", pass_idx=0, language=_quality_language(language), tiled=tiled)
    return {"language": language, "label": report.quality_label, "allowed": bool(report.token_search_allowed)}


def regrade(*, apply: bool, include_ungraded: bool, everything: bool) -> list[dict[str, Any]]:
    out = []
    for row in candidates(include_ungraded=include_ungraded, everything=everything):
        text = str(row["proofread_text"] or "").strip() or str(row["ocr_text"] or "").strip()
        if not text:
            continue
        # The OCR router prints progress while it loads; keep stdout for the report.
        with contextlib.redirect_stdout(sys.stderr):
            verdict = grade(text, tiled=bool(row["tiled"]))
        before = row["search_allowed"] if row["search_allowed"] is not None else row["report_allowed"]
        result = {
            "run_id": row["run_id"],
            "asset_ref": row["asset_ref"],
            "before": None if before is None else bool(before),
            **verdict,
        }
        if apply:
            pipeline_db.update_run_fields(row["run_id"], search_allowed=1 if verdict["allowed"] else 0)
            pipeline_db.log_event(
                row["run_id"],
                "REGRADED",
                "INFO",
                f"search_allowed={verdict['allowed']} quality={verdict['label']} by the current quality gate",
            )
        out.append(result)
    return out


def _render(results: list[dict[str, Any]], *, applied: bool) -> str:
    def word(value: bool | None) -> str:
        return {True: "searchable", False: "refused", None: "ungraded"}[value]

    lines = [f"{'run':10s} {'asset_ref':44s} {'language':14s} {'label':10s} {'before':10s} after"]
    for item in results:
        lines.append(
            f"{item['run_id'][:8]:10s} {item['asset_ref'][:44]:44s} {item['language']:14s} {item['label']:10s} "
            f"{word(item['before']):10s} {word(item['allowed'])}"
        )
    changed = sum(item["before"] != item["allowed"] for item in results)
    now = sum(item["allowed"] for item in results)
    lines.append("")
    lines.append(
        f"{len(results)} runs graded, {now} searchable under the current gate, {changed} decisions changed"
        + ("" if applied else " (report only; --apply records them)")
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--db", type=Path, help="pipeline database (default: ARCHAI_DB_PATH or app/archai.sqlite)")
    parser.add_argument("--apply", action="store_true", help="record the decisions on the runs")
    parser.add_argument("--include-ungraded", action="store_true", help="also grade runs that never had a quality report")
    parser.add_argument("--all", action="store_true", dest="everything", help="also re-grade runs with a recorded decision")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.db:
        if not args.db.exists():
            print(f"error: {args.db} does not exist", file=sys.stderr)
            return 2
        os.environ["ARCHAI_DB_PATH"] = str(args.db)
    results = regrade(apply=args.apply, include_ungraded=args.include_ungraded, everything=args.everything)
    print(json.dumps(results, indent=2, ensure_ascii=False) if args.json else _render(results, applied=args.apply))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
