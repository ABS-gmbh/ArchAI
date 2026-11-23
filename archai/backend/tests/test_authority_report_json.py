"""The JSON linking report carries what the linker stored: score components and the link trace.

Built from a real linking run on the middle-french-antitus reference page,
replayed from eval/entities/linking_cache.json (no network).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("benchmark_linking_for_report", BACKEND / "scripts" / "benchmark_linking.py")
    bl = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bl
    spec.loader.exec_module(bl)
    import os

    from app.db import pipeline_db
    from app.services import wikidata_client
    from app.services.authority_linking import build_report_from_db

    saved = {key: os.environ.get(key) for key in ("ARCHAI_DB_PATH", "ARCHAI_WIKIDATA_CACHE")}
    source = bl.Source(json.loads((BACKEND / "eval/entities/linking_cache.json").read_text(encoding="utf-8")), live=False, user_agent="test")
    undo = bl.install(source)
    try:
        page = next(p["reference"] for p in json.loads((BACKEND / "eval/quality_gate/readings.json").read_text())["pages"] if p["page_id"] == "middle-french-antitus")
        work = tmp_path_factory.mktemp("report")
        links = bl.link_page("middle-french-antitus", page, work)
        built = build_report_from_db(str(links[0]["run_id"]))
    finally:
        undo()
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        pipeline_db._DB_READY = False
        wikidata_client._CACHE_READY = False
        sys.modules.pop(spec.name, None)
    assert not source.misses
    return built


def _mention(report: dict[str, Any], surface: str) -> dict[str, Any]:
    return next(m for m in report["mention_results"] if m["surface"].startswith(surface))


def test_the_selected_candidate_carries_its_score_components(report: dict[str, Any]) -> None:
    lausanne = _mention(report, "lausanne")
    assert lausanne["status"] == "linked" and lausanne["selected"]["qid"] == "Q807"
    breakdown = lausanne["selected"]["score_breakdown"]
    assert breakdown["label_similarity"] == 1.0
    assert {"alias_match_quality", "document_context_compatibility", "final_score"} <= set(breakdown)


def test_every_top_candidate_carries_its_score_components(report: dict[str, Any]) -> None:
    satyre = _mention(report, "La satyre megere")
    assert satyre["top_candidates"], satyre
    for candidate in satyre["top_candidates"]:
        assert isinstance(candidate["score_breakdown"], dict)
        assert candidate["score_breakdown"]["final_score"] == pytest.approx(candidate["score"], abs=1e-3)


def test_existing_fields_are_unchanged(report: dict[str, Any]) -> None:
    candidate = _mention(report, "lausanne")["top_candidates"][0]
    assert set(candidate) - {"score_breakdown"} == {"source", "authority_id", "qid", "label", "score", "type_compatible"}
    selected = _mention(report, "lausanne")["selected"]
    assert set(selected) - {"score_breakdown"} == {
        "source", "authority_id", "qid", "label", "description", "score", "viaf_id", "geonames_id", "type_compatible",
    }


def test_each_mention_carries_its_link_trace_and_the_placeholders_are_filled(report: dict[str, Any]) -> None:
    satyre = _mention(report, "La satyre megere")
    trace = satyre["link_trace"]
    assert trace["status"] == "linked" and trace["reason"].startswith("best score = 0.894")
    assert [q["query"] for q in trace["query_details"]] == ["La satyre megere", "satyre megere"]
    first = trace["query_details"][0]
    assert first["source_hit_counts"]["wikidata"] == 1 and first["raw_hits"][0]["qid"] == "Q135111996"
    assert "raw_hits_by_source" not in first
    # These were [] and None although the trace held them.
    assert satyre["query_details"] == trace["query_details"]
    assert satyre["queries_attempted"] == ["La satyre megere", "satyre megere"]
    assert satyre["name_likeness"] == trace["name_likeness"] is not None


def test_the_trace_is_capped(report: dict[str, Any]) -> None:
    from app.services.authority_linking import _trimmed_link_trace

    stored = {
        "status": "unresolved",
        "query_details": [
            {"query": f"q{i}", "raw_hits": [{"qid": f"Q{j}", "label": "x", "extra": "dropped"} for j in range(12)], "raw_hits_by_source": {"wikidata": []}}
            for i in range(14)
        ],
        "top_candidates": [{"qid": f"Q{i}"} for i in range(9)],
    }
    trimmed = _trimmed_link_trace(stored)
    assert len(trimmed["query_details"]) == 10
    assert all(len(q["raw_hits"]) == 5 and q["raw_hits_total"] == 12 for q in trimmed["query_details"])
    assert all("extra" not in hit for q in trimmed["query_details"] for hit in q["raw_hits"])
    assert len(trimmed["top_candidates"]) == 5
    assert _trimmed_link_trace(None) is None and _trimmed_link_trace("garbage") is None
    for mention in report["mention_results"]:
        assert len(json.dumps(mention["link_trace"] or {})) < 20_000
