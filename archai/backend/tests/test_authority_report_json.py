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
