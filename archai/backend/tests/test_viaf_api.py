"""VIAF is queried at the endpoints OCLC documents, and its JSON is read as it comes back.

OpenAPI "VIAF API" 1.22.1 (https://developer.api.oclc.org/viaf-api, checked
2026-11-23): a cluster is GET https://viaf.org/viaf/{id} with an Accept header,
and AutoSuggest is GET https://viaf.org/viaf/AutoSuggest?query=. The code
fetched https://viaf.org/viaf/{id}/viaf.json, which is not documented and
answered 403, so every record came back empty. The fixtures are live responses
of that day for "austen" and cluster 102333412, trimmed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.services import authority_linking, authority_sources

FIXTURES = Path(__file__).parent / "fixtures" / "viaf"
AUTOSUGGEST = json.loads((FIXTURES / "autosuggest_austen.json").read_text(encoding="utf-8"))
CLUSTER = json.loads((FIXTURES / "cluster_102333412.json").read_text(encoding="utf-8"))


@pytest.fixture
def requests(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def http_json(url: str, params: dict[str, Any], *, timeout: int = 10, headers: Any = None) -> dict[str, Any]:
        seen.append({"url": url, "params": dict(params), "headers": dict(headers or {})})
        if url == "https://viaf.org/viaf/AutoSuggest":
            return AUTOSUGGEST
        if url == "https://viaf.org/viaf/102333412":
            return CLUSTER
        return {}

    monkeypatch.setattr(authority_sources, "_http_json", http_json)
    return seen


def test_a_cluster_is_fetched_from_the_documented_endpoint_as_json(requests: list[dict[str, Any]]) -> None:
    record = authority_sources._fetch_viaf_record("102333412")
    assert requests == [{"url": "https://viaf.org/viaf/102333412", "params": {}, "headers": {"Accept": "application/json"}}]
    assert record["nameType"] == "Personal"
    assert [item["text"] for item in record["mainHeadings"]["data"]] == ["Austen, Jane, 1775-1817.", "Austen, Jane"]
    assert record["viafID"] == 102333412
    assert not any(key.startswith(("ns1:", "xmlns")) for key in record)


def test_search_viaf_builds_candidates_from_autosuggest_and_the_cluster(requests: list[dict[str, Any]]) -> None:
    candidates = authority_sources.search_viaf("austen", k=1, ent_type="person")
    assert requests[0] == {
        "url": "https://viaf.org/viaf/AutoSuggest",
        "params": {"query": "austen"},
        "headers": {"Accept": "application/json"},
    }
    [jane] = candidates
    assert jane["viaf_id"] == "102333412"
    assert jane["label"] == "Austen, Jane, 1775-1817"
    assert jane["name_type"] == "Personal"
    assert jane["instance_of_qids"] == authority_sources.viaf_type_qids("Personal") != []
    assert [alias["value"] for alias in jane["aliases"]] == ["Austen, Jane, 1775-1817.", "Austen, Jane"]
    assert jane["titles"] == {"work": [{"title": "Anna Elliotová"}, {"title": "Anne Elliot Roman"}]}
    assert jane["url"] == "https://viaf.org/viaf/102333412"
    assert "partial" not in jane


def test_the_linker_profile_reads_the_same_record(requests: list[dict[str, Any]]) -> None:
    profile = authority_linking._fetch_viaf_profile("102333412")
    assert profile["source_url"] == "https://viaf.org/viaf/102333412"
    assert profile["name_type"] == "Personal"
    assert [alias["value"] for alias in profile["aliases"]] == ["Austen, Jane, 1775-1817.", "Austen, Jane"]


def test_a_failed_record_marks_the_candidates_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        authority_sources,
        "_http_json",
        lambda url, params, timeout=10, headers=None: AUTOSUGGEST if url.endswith("/AutoSuggest") else {},
    )
    candidates = authority_sources.search_viaf("austen", k=2, ent_type="person")
    assert len(candidates) == 2 and all(item["partial"] for item in candidates)


def test_no_code_path_uses_the_retired_viaf_json_url() -> None:
    for module in (authority_sources, authority_linking):
        assert '/viaf.json"' not in Path(module.__file__).read_text(encoding="utf-8")  # as a URL literal
