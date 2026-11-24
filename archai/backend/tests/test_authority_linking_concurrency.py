"""Two linking runs for the same pipeline run must not interleave.

run_authority_linking clears a run's candidates and links, spends seconds to
minutes on Wikidata/VIAF/GeoNames lookups, then inserts candidates with fresh
uuid4 ids. Two calls for one run - the manual POST /authority/link/{run_id}
next to the pipeline's own linking stage, or two clicks - interleaved as clear,
clear, insert, insert and left every candidate twice.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from app.db import pipeline_db
from app.services import authority_linking


def _seed_run(text: str) -> str:
    run_id = pipeline_db.create_run(asset_ref="page-1", asset_sha256="sha")
    with pipeline_db._connect() as conn:
        conn.execute(
            "UPDATE pipeline_runs SET ocr_text=?, proofread_text=?, warnings_json=? WHERE run_id=?",
            (text, text, json.dumps({"quality_label": "HIGH"}), run_id),
        )
        conn.commit()
    pipeline_db.insert_chunks(
        run_id, [{"chunk_id": f"{run_id}-c", "idx": 0, "start_offset": 0, "end_offset": len(text), "text": text}]
    )
    pipeline_db.insert_entity_mentions(
        run_id,
        [
            {
                "mention_id": f"{run_id}-m",
                "chunk_id": f"{run_id}-c",
                "start_offset": 0,
                "end_offset": 6,
                "surface": "Merlin",
                "norm": "merlin",
                "label": "PERSON",
                "ent_type": "person",
                "confidence": 0.92,
                "method": "test",
            }
        ],
    )
    return run_id


def _candidate_count(run_id: str) -> int:
    with pipeline_db._connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM entity_candidates WHERE mention_id IN "
            "(SELECT mention_id FROM entity_mentions WHERE run_id=?)",
            (run_id,),
        ).fetchone()[0]


@pytest.fixture
def slow_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)

    def search_wikidata(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        time.sleep(0.3)  # the network round trip, during which the other call clears
        return [
            {
                "qid": "Q111",
                "label": "Merlin",
                "description": "medieval legendary wizard arthurian",
                "url": "https://www.wikidata.org/wiki/Q111",
            }
        ]

    monkeypatch.setattr(authority_linking, "cache_check", lambda *args, **kwargs: (None, "miss"))
    monkeypatch.setattr(authority_linking, "search_wikidata", search_wikidata)
    monkeypatch.setattr(authority_linking, "search_viaf", lambda *args, **kwargs: [])
    monkeypatch.setattr(authority_linking, "search_geonames", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        authority_linking,
        "enrich_wikidata_item",
        lambda qid: {
            "instance_of_qids": ["Q5"],
            "canonical_label": "Merlin",
            "description": "medieval legendary wizard arthurian",
            "aliases": [],
            "country_qids": [],
            "admin_qids": [],
        },
    )
    monkeypatch.setattr(authority_linking, "is_type_compatible", lambda *args, **kwargs: True)


def _in_threads(*run_ids: str) -> list[BaseException]:
    errors: list[BaseException] = []

    def link(run_id: str) -> None:
        try:
            authority_linking.run_authority_linking(run_id, top_k=3)
        except BaseException as exc:  # noqa: BLE001 - reported by the test
            errors.append(exc)

    threads = [threading.Thread(target=link, args=(run_id,)) for run_id in run_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return errors


def test_concurrent_calls_for_one_run_leave_one_set_of_candidates(slow_sources: None) -> None:
    text = "Merlin medieval legendary wizard arthurian Merlinus"
    alone = _seed_run(text)
    authority_linking.run_authority_linking(alone, top_k=3)
    expected = _candidate_count(alone)
    assert expected > 0

    together = _seed_run(text)
    assert _in_threads(together, together) == []
    assert _candidate_count(together) == expected


def test_different_runs_still_link_in_parallel(slow_sources: None, monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Merlin medieval legendary wizard arthurian Merlinus"
    first, second = _seed_run(text), _seed_run(text)

    # Both runs must be inside their Wikidata lookup at the same moment. If
    # runs were serialised, the first would wait here for a second that cannot
    # start, and the barrier would break. No clock is compared: on a loaded CI
    # runner, "two runs take less than 1.5 times one run" failed by 2 ms.
    together = threading.Barrier(2, timeout=10)
    waited: set[int] = set()
    lookup = authority_linking.search_wikidata

    def search_wikidata(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        thread = threading.get_ident()
        if thread not in waited:
            waited.add(thread)
            together.wait()
        return lookup(*args, **kwargs)

    monkeypatch.setattr(authority_linking, "search_wikidata", search_wikidata)
    assert _in_threads(first, second) == []
    # A lookup's own error handling swallows a broken barrier, so ask the barrier.
    assert len(waited) == 2 and not together.broken
    assert _candidate_count(first) == _candidate_count(second) > 0
