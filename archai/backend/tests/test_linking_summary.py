"""The linking step's summary accounts for every name found, skipped ones included.

On the Antitus page the Steps said "6 names" were found, then "2 linked, 3 not
linked, 0 ambiguous": the sixth, a date ("lan de grace Mille anq"), is never
looked up and was left out of the summary (live regression, 2026-11-24). Both
counts were right; the summary now says "skipped=1".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.db import pipeline_db
from app.services import authority_linking, trace_pipeline

TEXT = "La satyre megere. Le bon pere euesque prince de lausanne. En lan de grace Mille anq cens."
MENTIONS = [
    ("La satyre megere.", "work"),
    ("pere", "title"),
    ("euesque", "title"),
    ("lausanne", "place"),
    ("lan de grace Mille anq", "date"),
    ("xq", "person"),
]
SUMMARY = re.compile(r"^Authority linking complete: linked=(\d+), unresolved=(\d+), ambiguous=(\d+), skipped=(\d+)\.$")


def test_the_summary_parts_add_up_to_the_names_found(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    run_id = pipeline_db.create_run(asset_ref="antitus", asset_sha256=None)
    pipeline_db.update_run_fields(run_id, proofread_text=TEXT, ocr_text=TEXT)
    rows = []
    for surface, ent_type in MENTIONS:
        start = TEXT.find(surface) if surface in TEXT else 0
        rows.append({"surface": surface, "ent_type": ent_type, "start_offset": start, "end_offset": start + len(surface), "method": "test", "confidence": 0.9})
    pipeline_db.insert_entity_mentions(run_id, rows)

    # No record anywhere, and nothing cached.
    monkeypatch.setattr(authority_linking, "cache_check", lambda *args, **kwargs: (None, "miss"))
    for search in ("search_wikidata", "search_viaf", "search_geonames"):
        monkeypatch.setattr(authority_linking, search, lambda *args, **kwargs: [])
    trace_pipeline._run_authority_linking_stage(run_id)

    messages = [e["message"] for e in pipeline_db.list_events(run_id) if e["stage"] == "AUTHORITY_LINKING_DONE"]
    assert len(messages) == 1, messages
    match = SUMMARY.match(messages[0])
    assert match, messages[0]
    linked, unresolved, ambiguous, skipped = (int(part) for part in match.groups())
    assert linked + unresolved + ambiguous + skipped == len(MENTIONS)
    assert skipped >= 1  # the date
