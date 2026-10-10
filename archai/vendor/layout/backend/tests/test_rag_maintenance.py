"""The scripts that bring an existing vector store in line: regrade_runs and prune_rag_index."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import zlib
from pathlib import Path
from typing import Any

import pytest

from app.config import settings
from app.db import pipeline_db
from app.services import rag_store

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
READINGS = Path(__file__).resolve().parents[1] / "eval" / "quality_gate" / "readings.json"
FAITHFUL = next(page for page in json.loads(READINGS.read_text(encoding="utf-8"))["pages"] if page["page_id"] == "old-french")["reference"]
REVERSED = "\n".join(" ".join(word[::-1] for word in line.split()) for line in FAITHFUL.splitlines())


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def regrade() -> Any:
    yield _load("regrade_runs")
    sys.modules.pop("regrade_runs", None)


@pytest.fixture(scope="module")
def prune() -> Any:
    yield _load("prune_rag_index")
    sys.modules.pop("prune_rag_index", None)


def _embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        vector = [0.0] * 32
        for i in range(len(text) - 2):
            vector[zlib.crc32(text[i : i + 3].encode()) % 32] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vectors.append([v / norm for v in vector])
    return vectors


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(rag_store, "_CLIENT", None)
    monkeypatch.setattr(settings, "rag_embedding_model", "test-embedder")
    monkeypatch.setattr(rag_store, "_embed", _embed)


def _run(text: str, *, sha: str, search_allowed: int | None = None, report_allowed: bool | None = None) -> str:
    run_id = pipeline_db.create_run(asset_ref=f"page-{sha}", asset_sha256=sha)
    lines = [line for line in text.splitlines() if line.strip()][:6]
    pipeline_db.insert_chunks(
        run_id,
        [{"chunk_id": f"{run_id}:{i}", "idx": i, "start_offset": 0, "end_offset": len(line), "text": line} for i, line in enumerate(lines)],
    )
    pipeline_db.update_run_fields(run_id, ocr_text=text)
    if search_allowed is not None:
        pipeline_db.update_run_fields(run_id, search_allowed=search_allowed)
    if report_allowed is not None:
        pipeline_db.insert_ocr_quality_report(run_id, {"pass_idx": 0, "token_search_allowed": report_allowed})
    return run_id


# ── regrade_runs ────────────────────────────────────────────────────────


def test_regrade_restores_a_faithful_page_the_old_gate_refused(store: None, regrade: Any) -> None:
    refused_by_old_gate = _run(FAITHFUL, sha="a", report_allowed=False)
    passed_by_old_gate = _run(REVERSED, sha="b", report_allowed=True)

    report = {item["run_id"]: item for item in regrade.regrade(apply=False, include_ungraded=False, everything=False)}
    assert (report[refused_by_old_gate]["before"], report[refused_by_old_gate]["allowed"]) == (False, True)
    assert (report[passed_by_old_gate]["before"], report[passed_by_old_gate]["allowed"]) == (True, False)
    assert pipeline_db.get_run(refused_by_old_gate)["search_allowed"] is None  # a report changes nothing

    regrade.regrade(apply=True, include_ungraded=False, everything=False)
    assert pipeline_db.get_run(refused_by_old_gate)["search_allowed"] == 1
    assert pipeline_db.get_run(passed_by_old_gate)["search_allowed"] == 0
    assert any(event["stage"] == "REGRADED" for event in pipeline_db.list_events(refused_by_old_gate))


def test_regrade_leaves_ungraded_and_decided_runs_alone_unless_asked(store: None, regrade: Any) -> None:
    ungraded = _run(FAITHFUL, sha="u")
    decided = _run(REVERSED, sha="d", search_allowed=1)
    assert regrade.regrade(apply=False, include_ungraded=False, everything=False) == []
    assert {item["run_id"] for item in regrade.regrade(apply=False, include_ungraded=True, everything=False)} == {ungraded}
    everything = regrade.regrade(apply=False, include_ungraded=True, everything=True)
    assert {item["run_id"] for item in everything} == {ungraded, decided}


# ── prune_rag_index ─────────────────────────────────────────────────────


def _put(collection: Any, run_id: str, ids: list[str]) -> None:
    collection.upsert(ids=ids, documents=ids, metadatas=[{"run_id": run_id} for _ in ids], embeddings=_embed(ids))


def test_prune_sorts_vectors_and_deletes_only_what_retrieval_cannot_use(store: None, prune: Any) -> None:
    older = _run(FAITHFUL, sha="page", search_allowed=1)
    newer = _run(FAITHFUL, sha="page", search_allowed=1)
    refused = _run(REVERSED, sha="other", search_allowed=0)
    chunks = rag_store._collection(backend_key=rag_store._space_key())
    _put(chunks, older, [f"{older}:0", f"{older}:1"])
    _put(chunks, newer, [f"{newer}:0", f"{newer}:1", f"{newer}:2"])
    _put(chunks, refused, [f"{refused}:0"])
    _put(chunks, newer, ["chunk-since-deleted"])
    _put(chunks, "run-since-deleted", ["gone:0"])
    other_space = rag_store._collection(backend_key=rag_store._LOCAL_EMBED_BACKEND)
    _put(other_space, newer, [f"{newer}:0"])

    report = prune.audit(apply=False, drop_other_spaces=False)
    own = report["collections"][rag_store._active_chunk_collection_name("test_embedder")]
    assert {key: own[key] for key in ("searchable", "superseded", "refused", "orphaned")} == {
        "searchable": 3,
        "superseded": 2,
        "refused": 1,
        "orphaned": 2,
    }
    assert report["collections"][settings.rag_collection_name]["configured_space"] is False
    assert chunks.count() == 8

    prune.audit(apply=True, drop_other_spaces=False)
    assert sorted(chunks.get()["ids"]) == [f"{newer}:0", f"{newer}:1", f"{newer}:2"]
    names = {collection.name for collection in rag_store._chroma_client().list_collections()}
    assert settings.rag_collection_name in names

    prune.audit(apply=True, drop_other_spaces=True)
    names = {collection.name for collection in rag_store._chroma_client().list_collections()}
    assert settings.rag_collection_name not in names
