"""Authority lookups share a deadline, and the selected VIAF record is not fetched twice (audit AL-03).

as_completed waited without a timeout, and leaving the executor's with block
waited for every future, so one lookup that never returned held the linking
run, and with it a threadpool thread, indefinitely.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest

from app.db import pipeline_db
from app.services import authority_linking

WIKIDATA_HIT = {"qid": "Q111", "label": "Merlin", "description": "medieval legendary wizard arthurian", "url": "u"}


@pytest.fixture
def stuck_viaf(monkeypatch: pytest.MonkeyPatch):
    """A VIAF lookup that does not return until the test is over."""
    release = threading.Event()
    calls: list[str] = []

    def search_viaf(query: str, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append(query)
        release.wait(30)
        return []

    monkeypatch.setattr(authority_linking, "search_viaf", search_viaf)
    monkeypatch.setattr(authority_linking, "_lookup_deadline_seconds", lambda sources, top_k: 0.3)
    monkeypatch.setattr(authority_linking, "cache_get", lambda *args, **kwargs: None)
    monkeypatch.setattr(authority_linking, "cache_put", lambda *args, **kwargs: None)
    yield calls
    release.set()


def test_a_source_that_never_answers_is_skipped_at_the_deadline(stuck_viaf: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(authority_linking, "search_wikidata", lambda query, **kwargs: [dict(WIKIDATA_HIT)])
    timed_out: list[str] = []
    started = time.monotonic()
    results, search_calls, viaf_calls, _ = authority_linking._search_all_sources_for_query(
        "Merlin", ent_type="person", top_k=3, wikidata_mode="api", timed_out=timed_out
    )
    assert time.monotonic() - started < 1.5
    assert results["wikidata"] and results["viaf"] == []
    assert (search_calls, viaf_calls) == (1, 0)
    assert timed_out == ["viaf"]


def _seed_run() -> str:
    text = "Merlin medieval legendary wizard arthurian Merlinus"
    run_id = pipeline_db.create_run(asset_ref="page-1", asset_sha256="sha")
    with pipeline_db._connect() as conn:
        conn.execute(
            "UPDATE pipeline_runs SET ocr_text=?, proofread_text=?, warnings_json=? WHERE run_id=?",
            (text, text, json.dumps({"quality_label": "HIGH"}), run_id),
        )
    pipeline_db.insert_chunks(run_id, [{"chunk_id": "c1", "idx": 0, "start_offset": 0, "end_offset": len(text), "text": text}])
    pipeline_db.insert_entity_mentions(
        run_id,
        [{"mention_id": "m1", "chunk_id": "c1", "start_offset": 0, "end_offset": 6, "surface": "Merlin",
          "norm": "merlin", "label": "PERSON", "ent_type": "person", "confidence": 0.92, "method": "test"}],
    )
    return run_id


def test_a_linking_run_finishes_and_says_which_source_was_skipped(
    stuck_viaf: list[str], tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    monkeypatch.setattr(authority_linking, "cache_check", lambda *args, **kwargs: (None, "miss"))
    monkeypatch.setattr(authority_linking, "search_wikidata", lambda query, **kwargs: [dict(WIKIDATA_HIT)])
    monkeypatch.setattr(authority_linking, "search_geonames", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        authority_linking,
        "enrich_wikidata_item",
        lambda qid: {"instance_of_qids": ["Q5"], "canonical_label": "Merlin", "description": "wizard",
                     "aliases": [], "country_qids": [], "admin_qids": []},
    )
    monkeypatch.setattr(authority_linking, "is_type_compatible", lambda *args, **kwargs: True)
    run_id = _seed_run()

    started = time.monotonic()
    result = authority_linking.run_authority_linking(run_id, top_k=3)
    elapsed = time.monotonic() - started

    assert stuck_viaf, "the stuck source was queried"
    assert elapsed < 0.3 * len(stuck_viaf) + 2.0
    assert result["source_timeouts"] == {"viaf": len(stuck_viaf)}
    assert "sources that did not answer in time (lookups skipped): viaf=" in authority_linking.build_linking_report(result)
    events = [(e["stage"], e["event"], e["message"]) for e in pipeline_db.list_events(run_id)]
    assert any(stage == "AUTHORITY_LINKING" and event == "WARN" and "viaf x" in message for stage, event, message in events)


def test_the_deadline_follows_the_request_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "geonames_timeout_seconds", 7)
    deadline = authority_linking._lookup_deadline_seconds
    assert deadline(["wikidata"], 5) == 15.0 + 2.0
    assert deadline(["viaf"], 5) == 10.0 * 6 + 2.0  # AutoSuggest plus one record per hit
    assert deadline(["geonames"], 5) == 7.0 + 2.0
    assert deadline(["wikidata", "geonames"], 5) == 17.0


def test_a_selected_viaf_candidate_reuses_its_record(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[str] = []
    monkeypatch.setattr(authority_linking, "_fetch_viaf_profile", lambda viaf_id: fetched.append(viaf_id) or {"aliases": []})
    candidate = {
        "source": "viaf",
        "viaf_id": "12345",
        "aliases": [{"lang": "", "value": "Merlinus"}],
        "name_type": "Personal",
        "titles": {"work": []},
    }
    profile = authority_linking._viaf_profile_for(candidate)
    assert fetched == []
    assert profile["aliases"] == [{"lang": "", "value": "Merlinus"}]
    assert (profile["name_type"], profile["titles"]) == ("Personal", {"work": []})
    assert profile["source_url"].endswith("/12345")

    authority_linking._viaf_profile_for({**candidate, "partial": True})
    authority_linking._viaf_profile_for({"source": "wikidata", "qid": "Q1", "viaf_id": "999"})
    assert fetched == ["12345", "999"]
