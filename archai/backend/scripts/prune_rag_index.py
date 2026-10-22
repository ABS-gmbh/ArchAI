"""Audit the vector store against the pipeline database, and prune what retrieval can no longer use.

Usage::

    python scripts/prune_rag_index.py                       # report only
    python scripts/prune_rag_index.py --apply               # delete unusable vectors
    python scripts/prune_rag_index.py --apply --drop-other-spaces
    python scripts/prune_rag_index.py --db app/archai.sqlite --chroma .data/chroma

Retrieval now searches only the searchable runs (``pipeline_db.searchable_runs``):
runs the quality gate allowed search on, the newest of each page. Vectors of any
other run are never returned, but stores built by older code are full of them:
copies of every page from each earlier analysis, runs the gate refused, chunks
since deleted. This reports, for each collection, how many vectors belong to

* ``searchable`` - a searchable run, and a chunk the database still holds;
* ``superseded`` - an earlier run of a page with a newer searchable run;
* ``refused`` - a run the gate did not allow search on, or never graded;
* ``orphaned`` - a run or chunk the database no longer holds.

``--apply`` deletes all but ``searchable`` from the collections of the configured
embedding space; an explicitly requested older run is re-indexed on demand.
Collections of other embedding spaces are listed and, with
``--drop-other-spaces``, dropped: retrieval no longer falls back to them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.db import pipeline_db  # noqa: E402
from app.services import rag_store  # noqa: E402


def _classify(
    collection: Any,
    *,
    searchable: set[str],
    approved: set[str],
    known_runs: set[str],
    chunk_ids: set[str] | None,
) -> dict[str, list[str]]:
    """Sort a collection's vector ids into the four buckets of the module docstring.

    *approved* holds the runs the gate allowed search on; *searchable* the
    newest of them for each page.
    """
    got = collection.get(include=["metadatas"])
    buckets: dict[str, list[str]] = {"searchable": [], "superseded": [], "refused": [], "orphaned": []}
    for doc_id, meta in zip(got.get("ids") or [], got.get("metadatas") or [], strict=True):
        run_id = str((meta or {}).get("run_id") or "")
        if run_id not in known_runs or (chunk_ids is not None and str(doc_id) not in chunk_ids):
            buckets["orphaned"].append(str(doc_id))
        elif run_id in searchable:
            buckets["searchable"].append(str(doc_id))
        elif run_id in approved:
            buckets["superseded"].append(str(doc_id))
        else:
            buckets["refused"].append(str(doc_id))
    return buckets


def audit(*, apply: bool, drop_other_spaces: bool) -> dict[str, Any]:
    searchable = {row["run_id"] for row in pipeline_db.searchable_runs()}
    pipeline_db._init_db_if_needed()
    with pipeline_db._connect() as conn:
        known_runs = {str(row[0]) for row in conn.execute("SELECT run_id FROM pipeline_runs")}
        chunk_ids = {str(row[0]) for row in conn.execute("SELECT chunk_id FROM chunks")}
    # Approved on their own, whether or not a newer run of their page supersedes them.
    approved = {row["run_id"] for row in pipeline_db.searchable_runs(sorted(known_runs))}
    space = rag_store._space_key()
    own = {
        rag_store._active_chunk_collection_name(space): chunk_ids,
        rag_store._active_entity_collection_name(space): None,
    }
    client = rag_store._chroma_client()
    report: dict[str, Any] = {"embedding_space": space, "searchable_runs": len(searchable), "collections": {}}
    for item in client.list_collections():
        name = str(getattr(item, "name", item))
        if not (name.startswith(settings.rag_collection_name) or name.startswith(settings.rag_entity_collection_name)):
            continue
        collection = client.get_collection(name)
        is_chunks = name.startswith(settings.rag_collection_name) and not name.startswith(settings.rag_entity_collection_name)
        buckets = _classify(
            collection,
            searchable=searchable,
            approved=approved,
            known_runs=known_runs,
            chunk_ids=chunk_ids if is_chunks else None,
        )
        entry: dict[str, Any] = {
            "vectors": sum(len(ids) for ids in buckets.values()),
            **{bucket: len(ids) for bucket, ids in buckets.items()},
            "configured_space": name in own,
        }
        if is_chunks:
            got = collection.get(include=["documents"])
            texts = Counter(" ".join((doc or "").split()) for doc in got.get("documents") or [])
            entry["exact_duplicate_vectors"] = sum(count for count in texts.values() if count > 1)
        if apply and name in own:
            doomed = [doc_id for bucket in ("superseded", "refused", "orphaned") for doc_id in buckets[bucket]]
            for start in range(0, len(doomed), 500):
                collection.delete(ids=doomed[start : start + 500])
            entry["deleted"] = len(doomed)
        elif apply and drop_other_spaces and name not in own:
            client.delete_collection(name)
            entry["dropped"] = True
        report["collections"][name] = entry
    return report


def _render(report: dict[str, Any]) -> str:
    lines = [
        f"embedding space: {report['embedding_space']}",
        f"searchable runs (newest gate-approved run of each page): {report['searchable_runs']}",
        "",
        f"{'collection':52s} {'vectors':>7s} {'searchable':>10s} {'superseded':>10s} {'refused':>8s} {'orphaned':>8s} {'dup text':>8s}",
    ]
    for name, entry in report["collections"].items():
        marker = "" if entry["configured_space"] else "  (other space)"
        action = " deleted %d" % entry["deleted"] if "deleted" in entry else (" dropped" if entry.get("dropped") else "")
        lines.append(
            f"{name:52s} {entry['vectors']:7d} {entry['searchable']:10d} {entry['superseded']:10d} "
            f"{entry['refused']:8d} {entry['orphaned']:8d} {entry.get('exact_duplicate_vectors', 0):8d}{marker}{action}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--db", type=Path, help="pipeline database (default: ARCHAI_DB_PATH or app/archai.sqlite)")
    parser.add_argument("--chroma", type=Path, help="ChromaDB directory (default: settings.chroma_persist_dir)")
    parser.add_argument("--apply", action="store_true", help="delete vectors retrieval can no longer use")
    parser.add_argument("--drop-other-spaces", action="store_true", help="with --apply, drop collections of other embedding spaces")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.db:
        if not args.db.exists():
            print(f"error: {args.db} does not exist", file=sys.stderr)
            return 2
        os.environ["ARCHAI_DB_PATH"] = str(args.db)
    if args.chroma:
        settings.chroma_persist_dir = str(args.chroma)
    report = audit(apply=args.apply, drop_other_spaces=args.drop_other_spaces)
    print(json.dumps(report, indent=2) if args.json else _render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
