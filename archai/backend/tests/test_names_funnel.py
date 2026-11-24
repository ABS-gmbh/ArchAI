"""The funnel puts a linkable gold name at the first stage it does not pass (scripts/benchmark_names_funnel.py)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_names_funnel as funnel  # noqa: E402

GOLD = {"qids": ["Q31966"], "viaf": ["264405984"], "outcome": "abstained"}


def _mention(**fields: Any) -> dict[str, Any]:
    base = {"queries": ["Marc"], "hits": [], "prefilter_in": [], "prefilter_out": [], "enriched": [], "scored": [], "final_reason": ""}
    return {**base, **fields}


def _hit(ident: str, source: str = "wikidata", rank: int = 1) -> dict[str, Any]:
    return {"query": "Marc", "source": source, "rank": rank, "id": ident, "label": "x"}


def _scored(ident: str, score: float, *, type_ok: bool = True, name_sim: float = 1.0, source: str = "wikidata") -> dict[str, Any]:
    return {"qid": ident if source == "wikidata" else "", "viaf": ident if source == "viaf" else "", "source": source,
            "label": ident, "score": score, "type_ok": type_ok, "name_sim": name_sim}


def stage(mentions: list[dict[str, Any]], **entity: Any) -> str:
    return funnel.assign_stage({**GOLD, **entity}, mentions)[0]


def test_no_mention_is_not_extracted() -> None:
    assert stage([], outcome="missed-mention") == "1 not extracted"


def test_a_gate_before_the_lookup_is_never_searched() -> None:
    assert stage([_mention(queries=[], final_reason="low_evidence_place: cap=False ...")]) == "2 extracted, never searched"


def test_the_gold_record_missing_from_every_source() -> None:
    assert stage([_mention(hits=[_hit("Q1"), _hit("123", "viaf")])]) == "3 gold record not in the results"


def test_a_wikidata_hit_never_enriched_or_scored() -> None:
    mention = _mention(hits=[_hit("Q31966", rank=4)], prefilter_in=["Q1", "Q31966"], prefilter_out=["Q1", "Q2", "Q3", "Q4", "Q31966"],
                       enriched=[{"qid": "Q1"}, {"qid": "Q2"}, {"qid": "Q3"}], scored=[_scored("Q1", 0.8), _scored("Q31966", 0.35, type_ok=False)])
    name, info = funnel.assign_stage(GOLD, [mention])
    assert name == "4 in the results, not enriched" and info["pre_sort_rank"] == 5


def test_enriched_but_type_incompatible() -> None:
    mention = _mention(hits=[_hit("Q31966")], enriched=[{"qid": "Q31966"}], scored=[_scored("Q31966", 0.4, type_ok=False)])
    assert stage([mention]) == "5 enriched, type-incompatible or name too far"


def test_scored_but_lost_on_the_margin() -> None:
    mention = _mention(hits=[_hit("264405984", "viaf")], scored=[_scored("999", 0.83, source="viaf"), _scored("264405984", 0.79, source="viaf")],
                       final_reason="margin too small")
    name, info = funnel.assign_stage(GOLD, [mention])
    assert name == "6 scored, not selected" and info["top"][1] == 0.83 and info["runner_up"][1] == 0.79


def test_the_mention_that_got_furthest_counts() -> None:
    gated = _mention(queries=[], final_reason="low_evidence_place")
    searched = _mention(hits=[_hit("Q1")])
    assert stage([gated, searched]) == "3 gold record not in the results"


def test_linked_and_linked_elsewhere() -> None:
    assert stage([_mention()], outcome="correct") == "7 linked"
    assert stage([_mention()], outcome="wrong", linked_to="Q83241") == "7 linked to another record"
