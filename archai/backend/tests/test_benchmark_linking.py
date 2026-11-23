"""scripts/benchmark_linking.py: scoring, record/replay, politeness. No network."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bl():
    spec = importlib.util.spec_from_file_location("benchmark_linking", BACKEND / "scripts" / "benchmark_linking.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


LINKABLE = {"surface": "lausanne", "type": "place", "authority": {"status": "linkable", "qids": ["Q807"], "viaf": ["147349797"]}}
NO_RECORD = {"surface": "Ynabele", "type": "person", "authority": {"status": "none", "qids": [], "viaf": []}}


def _link(start: int, end: int, status: str = "linked", qid: str = "", viaf: str = "") -> dict[str, Any]:
    return {"start_offset": start, "end_offset": end, "link_status": status, "wikidata_qid": qid, "viaf_id": viaf}


@pytest.mark.parametrize(
    ("entity", "links", "outcome"),
    [
        (LINKABLE, [_link(0, 8, qid="Q807")], "correct"),
        (LINKABLE, [_link(0, 8, qid="", viaf="147349797")], "correct"),
        (LINKABLE, [_link(0, 8, qid="Q39")], "wrong"),
        (LINKABLE, [_link(0, 8, status="unresolved")], "abstained"),
        (LINKABLE, [_link(20, 30, qid="Q807")], "missed-mention"),
        (NO_RECORD, [_link(0, 7, status="unresolved")], "correct"),
        (NO_RECORD, [_link(0, 7, qid="Q5")], "wrong"),
    ],
)
def test_score_entity(bl, entity: dict[str, Any], links: list[dict[str, Any]], outcome: str) -> None:
    assert bl.score_entity(entity, (0, len(entity["surface"])), links)["outcome"] == outcome


def test_request_keys_ignore_parameter_order_and_empty_values(bl) -> None:
    assert bl.request_key("https://x/api", {"b": "2", "a": "1", "c": ""}) == bl.request_key("https://x/api", {"a": "1", "b": "2"})
    assert bl.request_key("https://viaf.org/viaf/1", {}) == "https://viaf.org/viaf/1"


def test_replay_answers_from_the_cache_and_counts_misses(bl) -> None:
    source = bl.Source({"https://x/api?a=1": {"ok": True}}, live=False, user_agent="test")
    assert source.fetch("https://x/api", {"a": "1"}) == {"ok": True}
    assert source.fetch("https://x/api", {"a": "2"}) is None
    assert source.misses == ["https://x/api?a=2"] and source.requests == 0


def test_recording_stops_on_429_and_paces_requests(bl, monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    calls: list[str] = []

    def refuse(request, timeout=20):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(bl.urllib.request, "urlopen", refuse)
    source = bl.Source({}, live=True, user_agent="Archai-OCR-Pipeline/1.0", min_interval=0.0)
    with pytest.raises(SystemExit, match="429"):
        source.fetch("https://www.wikidata.org/w/api.php", {"action": "wbsearchentities"})
    assert len(calls) == 1


def test_recorded_records_are_trimmed_to_what_the_linker_reads(bl) -> None:
    viaf = {"ns1:VIAFCluster": {"ns1:viafID": 1, "ns1:nameType": "Personal", "ns1:x400s": {"big": 1},
                                 "ns1:mainHeadings": {"ns1:data": [{"ns1:text": "A", "ns1:sources": {"ns1:s": ["LC"]}}]}}}
    assert bl._trim_viaf(viaf) == {"ns1:VIAFCluster": {"ns1:viafID": 1, "ns1:nameType": "Personal", "ns1:mainHeadings": {"ns1:data": [{"ns1:text": "A"}]}}}
    wikidata = {"entities": {"Q1": {"id": "Q1", "labels": {"en": {"value": "x"}}, "sitelinks": {"enwiki": {}},
                                    "claims": {"P31": [{"mainsnak": {"p": 1}, "rank": "normal", "references": [1]}], "P18": [{}]}}}}
    trimmed = bl._trim_wikidata(wikidata)["entities"]["Q1"]
    assert set(trimmed) == {"id", "labels", "claims"}
    assert trimmed["claims"] == {"P31": [{"mainsnak": {"p": 1}, "rank": "normal"}]}


def test_the_committed_cache_replays_the_reported_numbers_offline(bl) -> None:
    """The recorded responses stand in for Wikidata and VIAF: no request leaves the process."""
    result = bl.run()
    assert result["cache_misses"] == [] and result["live_requests"] == 0
    assert result["n"] == 13
    assert result["totals"] == {"correct": 7, "wrong": 0, "abstained": 1, "missed-mention": 5}
    lausanne = next(row for row in result["pages"]["middle-french-antitus"] if row["surface"] == "lausanne")
    assert (lausanne["outcome"], lausanne["linked_to"]) == ("correct", "Q807")


def test_gold_authority_annotations_are_complete() -> None:
    gold = json.loads((BACKEND / "eval/entities/gold.json").read_text(encoding="utf-8"))
    for page in gold["pages"]:
        for entity in page["entities"]:
            authority = entity["authority"]
            assert authority["status"] in {"linkable", "none", "n/a", "ambiguous"}, entity
            assert authority["how"], entity
            if authority["status"] == "linkable":
                assert authority["qids"] or authority["viaf"], entity
