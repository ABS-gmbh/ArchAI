"""scripts/benchmark_end_to_end.py: the matching rules and the link outcomes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def e2e():
    spec = importlib.util.spec_from_file_location("benchmark_end_to_end", BACKEND / "scripts" / "benchmark_end_to_end.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("benchmark_end_to_end", module)
    spec.loader.exec_module(module)
    return module


def test_exact_and_fuzzy_rules(e2e) -> None:
    assert e2e.exact_form("Ame\nde  monfalcon") == e2e.exact_form("ame de monfalcon")
    assert e2e.fuzzy_match("mͣrcadigas", "Roys mͣrcadigas")  # one key contains the other
    assert e2e.fuzzy_match("Marcadigas", "marcadigas")
    assert e2e.fuzzy_match("lan de grace Mille cinq cens", "lan de grace Mille anq")  # within 1/3
    assert not e2e.fuzzy_match("iohannis", "R Rian")
    assert not e2e.fuzzy_match("Caldꝰ", "Roys")


def _gold(surface: str, status: str, qids: list[str] | None = None, type_: str = "person") -> dict:
    return {"surface": surface, "type": type_, "authority": {"status": status, "qids": qids or [], "viaf": []}}


def test_link_outcomes(e2e) -> None:
    gold = [
        _gold("lausanne", "linkable", ["Q807"], "place"),
        _gold("Antitus", "linkable", ["Q30729001"]),
        _gold("iohannis", "linkable", ["Q328804"]),
        _gold("Marcadigas", "none"),
        _gold("hyerusalem", "linkable", ["Q1218"], "place"),
    ]
    mentions = [
        {"mention_id": 1, "surface": "lausanne", "ent_type": "place", "start_offset": 0},
        {"mention_id": 2, "surface": "Antitus", "ent_type": "person", "start_offset": 10},
        {"mention_id": 3, "surface": "euangelistæ iohannis", "ent_type": "person", "start_offset": 20},
        {"mention_id": 4, "surface": "marcadigas", "ent_type": "person", "start_offset": 30},
        {"mention_id": 5, "surface": "R Rian", "ent_type": "person", "start_offset": 40},
        {"mention_id": 6, "surface": "prince", "ent_type": "title", "start_offset": 50},
    ]
    links = [
        {"mention_id": 1, "link_status": "linked", "wikidata_qid": "Q807", "authority_source": "wikidata"},
        {"mention_id": 2, "link_status": "linked", "wikidata_qid": "Q1", "authority_source": "wikidata"},
        {"mention_id": 3, "link_status": "unresolved"},
    ]
    score = e2e.score_page(gold, mentions, links)
    outcomes = {row["surface"]: row["link"] for row in score["entities"]}
    assert outcomes == {"lausanne": "correct", "Antitus": "wrong", "iohannis": "abstained", "Marcadigas": "correct", "hyerusalem": "missed"}
    assert score["spurious"] == [{"surface": "R Rian", "type": "person", "method": None}]  # titles are not scored
    assert (score["exact"], score["fuzzy"], score["mentions_scored_types"]) == (3, 4, 5)
