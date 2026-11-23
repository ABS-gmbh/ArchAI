"""Report, and on request delete, pipeline runs past a retention window (audit DB-07).

Usage::

    python scripts/prune_runs.py --older-than-days 180              # report only
    python scripts/prune_runs.py --older-than-days 180 --apply      # delete them
    python scripts/prune_runs.py --older-than-days 180 --db /path/archai.sqlite --json

Without --apply nothing is deleted: the report is the way to preview a policy.
Which runs are kept, and why, is set out in app/services/run_retention.py: runs
updated within the window, runs not finished, runs with no page identity, each
page's newest run, newest completed run and searched run, runs with an OCR
benchmark reference, annotated runs (unless --include-annotated) and runs the
thesis showcase bundle names. --apply removes each expired run's
vectors, then the run and everything that cascades from it. --vacuum then
compacts the database file; it locks the database while it runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.db import pipeline_db  # noqa: E402
from app.services import run_retention  # noqa: E402


def run(
    older_than_days: int, *, apply: bool = False, include_annotated: bool = False, vacuum: bool = False
) -> dict[str, Any]:
    plan = run_retention.plan_retention(older_than_days, include_annotated=include_annotated)
    report: dict[str, Any] = {
        "database": str(pipeline_db._db_config().path),
        "cutoff": plan.cutoff,
        "expired": len(plan.expired),
        "kept": {reason: len(run_ids) for reason, run_ids in sorted(plan.kept.items())},
        "expired_run_ids": plan.expired,
        "applied": apply,
    }
    if apply:
        report["deleted"] = run_retention.delete_runs(plan.expired, cutoff=plan.cutoff)
        if vacuum:
            with sqlite3.connect(pipeline_db._db_config().path) as conn:
                conn.execute("VACUUM")
            report["vacuumed"] = True
    return report


def _render(report: dict[str, Any]) -> str:
    lines = [
        f"database: {report['database']}",
        f"cutoff (last update before): {report['cutoff']}",
        f"expired: {report['expired']}",
        "kept:",
        *(f"  {reason:22s} {count}" for reason, count in report["kept"].items()),
    ]
    if report["applied"]:
        deleted = report["deleted"]
        lines.append(f"deleted: {deleted['runs']} runs, {deleted['vectors']} vectors")
    else:
        lines.append("report only: nothing was deleted (add --apply to delete the expired runs)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--older-than-days", type=int, required=True, help="retention window in days (at least 1)")
    parser.add_argument("--apply", action="store_true", help="delete the expired runs")
    parser.add_argument("--include-annotated", action="store_true", help="also expire runs with annotations")
    parser.add_argument("--vacuum", action="store_true", help="with --apply, compact the database afterwards")
    parser.add_argument("--db", type=Path, help="pipeline database (default: ARCHAI_DB_PATH or app/archai.sqlite)")
    parser.add_argument("--chroma", type=Path, help="ChromaDB directory (default: settings.chroma_persist_dir)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.older_than_days < 1:
        parser.error("--older-than-days must be at least 1")
    if args.db:
        if not args.db.exists():
            print(f"error: {args.db} does not exist", file=sys.stderr)
            return 2
        os.environ["ARCHAI_DB_PATH"] = str(args.db)
        pipeline_db._DB_READY = False
    if args.chroma:
        settings.chroma_persist_dir = str(args.chroma)
    report = run(
        args.older_than_days, apply=args.apply, include_annotated=args.include_annotated, vacuum=args.vacuum
    )
    print(json.dumps(report, indent=2) if args.json else _render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
