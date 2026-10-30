"""VIAF and GeoNames answers are cached and their calls capped per run (audit AL-03).

Each VIAF search is one AutoSuggest request plus one record request per hit,
up to 10 s each, and it ran for every query variant of every person and work
mention: no cache, and no cap - the Wikidata cap did not apply to it, and once
Wikidata's budget was spent person, work and place mentions kept searching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services import authority_linking, authority_sources

VIAF_HIT = {"source": "viaf", "authority_id": "12345", "viaf_id": "12345", "label": "Merlin", "aliases": []}


def _counting(monkeypatch: Any, name: str, payload: list[dict[str, Any]]) -> list[str]:
    calls: list[str] = []

    def search(query: str, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append(query)
        return [dict(item) for item in payload]

    monkeypatch.setattr(authority_linking, name, search)
    return calls


def _search(query: str, ent_type: str = "person", **kwargs: Any):
    return authority_linking._search_all_sources_for_query(
        query, ent_type=ent_type, top_k=3, wikidata_mode="skip", **kwargs
    )


def test_a_viaf_answer_is_cached(monkeypatch: Any) -> None:
    calls = _counting(monkeypatch, "search_viaf", [VIAF_HIT])
    first, _, viaf_calls_first, _ = _search("Merlin cache test")
    second, _, viaf_calls_second, _ = _search("Merlin cache test")
    assert calls == ["Merlin cache test"]
    assert (viaf_calls_first, viaf_calls_second) == (1, 0)
    assert first["viaf"][0]["viaf_id"] == second["viaf"][0]["viaf_id"] == "12345"


def test_a_partial_viaf_answer_is_not_cached(monkeypatch: Any) -> None:
    calls = _counting(monkeypatch, "search_viaf", [{**VIAF_HIT, "partial": True}])
    _search("Merlin partial test")
    _search("Merlin partial test")
    assert len(calls) == 2


def test_a_geonames_answer_is_cached(monkeypatch: Any) -> None:
    calls = _counting(monkeypatch, "search_geonames", [{"source": "geonames", "geonames_id": "1", "label": "Paris"}])
    _search("Paris cache test", ent_type="place")
    _search("Paris cache test", ent_type="place")
    assert calls == ["Paris cache test"]


def test_past_its_budget_a_source_is_not_called_but_its_cache_is_read(monkeypatch: Any) -> None:
    calls = _counting(monkeypatch, "search_viaf", [VIAF_HIT])
    _search("Merlin budget test")
    results, _, viaf_calls, _ = _search("Merlin budget test", allow_viaf=False)
    assert (len(calls), viaf_calls) == (1, 0)
    assert results["viaf"][0]["viaf_id"] == "12345"
    assert _search("Lancelot budget test", allow_viaf=False)[0]["viaf"] == []
    assert len(calls) == 1


def test_search_viaf_marks_answers_whose_records_failed(monkeypatch: Any) -> None:
    def http_json(url: str, params: dict[str, Any], *, timeout: int = 10) -> dict[str, Any]:
        if "AutoSuggest" in url:
            return {"result": [{"viafid": "1", "term": "Merlin"}, {"viafid": "2", "term": "Merlinus"}]}
        return {"nameType": "Personal"} if url.endswith("/1/viaf.json") else {}

    monkeypatch.setattr(authority_sources, "_http_json", http_json)
    assert all(item.get("partial") for item in authority_sources.search_viaf("Merlin", ent_type="person"))

    monkeypatch.setattr(authority_sources, "_http_json", lambda url, params, timeout=10: (
        {"result": [{"viafid": "1", "term": "Merlin"}]} if "AutoSuggest" in url else {"nameType": "Personal"}))
    assert not any(item.get("partial") for item in authority_sources.search_viaf("Merlin", ent_type="person"))


def test_a_run_stops_searching_once_every_source_is_spent(monkeypatch: Any, tmp_path: Path) -> None:
    from tests.test_authority_linking_multisource import _seed_run

    run_id = _seed_run(tmp_path, monkeypatch, surface="Merlin", ent_type="person",
                       base_text="Merlin medieval legendary wizard")
    monkeypatch.setattr(authority_linking, "_MAX_SEARCH_CALLS_PER_RUN", 0)
    monkeypatch.setattr(authority_linking, "_MAX_VIAF_CALLS_PER_RUN", 0)
    monkeypatch.setattr(authority_linking, "cache_check", lambda *args, **kwargs: (None, "miss"))
    viaf_calls = _counting(monkeypatch, "search_viaf", [VIAF_HIT])
    _counting(monkeypatch, "search_wikidata", [])

    result = authority_linking.run_authority_linking(run_id, top_k=3)

    assert viaf_calls == []
    assert result["api_calls_viaf"] == 0
