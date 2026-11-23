"""A VIAF candidate's type comes from its record's nameType (audit AL-02).

Fixtures are live VIAF responses of 2026-11-23, trimmed: the Jane Austen
cluster (Personal) and AutoSuggest/cluster answers for "Chanson de Roland"
(UniformTitleWork, and one UniformTitleExpression).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.services import authority_sources
from app.services.authority_sources import viaf_type_qids
from app.services.wikidata_client import is_type_compatible

FIXTURES = Path(__file__).parent / "fixtures" / "viaf"


def _load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _cluster(name: str) -> dict[str, Any]:
    return authority_sources._strip_viaf_namespace(_load(name))["VIAFCluster"]


def test_a_personal_cluster_is_compatible_with_a_person_and_not_a_place() -> None:
    qids = viaf_type_qids(_cluster("cluster_102333412.json")["nameType"])
    assert qids == ["Q5"]
    assert is_type_compatible("person", qids, description="Personal | Austen, Jane, 1775-1817")
    assert not is_type_compatible("place", qids)


def test_a_uniform_title_work_is_compatible_with_a_work_mention() -> None:
    cluster = _cluster("cluster_5836165272458110690007.json")
    assert cluster["nameType"] == "UniformTitleWork"
    assert is_type_compatible("work", viaf_type_qids(cluster["nameType"]))
    work_row = _load("autosuggest_chanson_de_roland.json")["result"][0]
    assert work_row["nametype"] == "uniformtitlework"
    assert viaf_type_qids(work_row["nametype"]) == viaf_type_qids(cluster["nameType"]) == ["Q7725634"]
    assert not is_type_compatible("person", viaf_type_qids(cluster["nameType"]))


@pytest.mark.parametrize("name_type", ["uniformtitleexpression", "UniformTitleExpression", "", "Unknown", None])
def test_an_expression_or_an_unknown_name_type_stays_incompatible(name_type: Any) -> None:
    qids = viaf_type_qids(name_type)
    assert qids == []
    for ent_type in ("work", "place"):
        assert not is_type_compatible(ent_type, qids)
    assert not is_type_compatible("person", qids, description="VIAF authority record")


def test_search_viaf_types_a_work_candidate_from_its_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    def http_json(url: str, params: dict[str, Any], *, timeout: int = 10, headers: Any = None) -> dict[str, Any]:
        if url.endswith("/AutoSuggest"):
            return _load("autosuggest_chanson_de_roland.json")
        if url.endswith("/5836165272458110690007"):
            return _load("cluster_5836165272458110690007.json")
        return {}

    monkeypatch.setattr(authority_sources, "_http_json", http_json)
    work, expression = authority_sources.search_viaf("chanson de roland", k=2, ent_type="work")
    assert (work["name_type"], work["instance_of_qids"]) == ("UniformTitleWork", ["Q7725634"])
    # The expression's cluster was not served, so its type comes from AutoSuggest's nametype: untyped.
    assert expression["instance_of_qids"] == []
