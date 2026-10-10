"""retrieve_chunks fuses the dense ranking with character n-gram BM25."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from app.config import settings
from app.db import pipeline_db
from app.services import rag_store

PAGE = {
    "c0": "Incipit liber primus de natura rerum",
    "c1": "Oremus. dn̄s uobiscū et cū sp̄u tuo",
    "c2": "Explicit hic totum pro xp̄o da mihi potum",
    "c3": "In principio erat uerbum et uerbum erat apud deum",
}
# A dense retriever that has the prayer (c1) last, as the deployed embedding
# model does with abbreviated Latin.
DENSE_ORDER = ["c0", "c2", "c3", "c1"]


def _insert_run(asset_ref: str, chunks: dict[str, str]) -> str:
    run_id = pipeline_db.create_run(asset_ref=asset_ref, asset_sha256=f"sha-{asset_ref}")
    offset = 0
    rows = []
    for idx, (chunk_id, text) in enumerate(chunks.items()):
        rows.append({"chunk_id": chunk_id, "idx": idx, "start_offset": offset, "end_offset": offset + len(text), "text": text})
        offset += len(text) + 1
    pipeline_db.insert_chunks(run_id, rows)
    pipeline_db.update_run_fields(run_id, search_allowed=1)
    return run_id


class _Collection:
    """The dense side: ranks the page in DENSE_ORDER, whatever the query."""

    def __init__(self, records: dict[str, dict[str, Any]], *, restrict_by_ids: bool = True) -> None:
        self.records = records
        self.restrict_by_ids = restrict_by_ids
        self.queries = 0

    def _matches(self, meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
        if not where:
            return True
        wanted = where["run_id"]
        return meta["run_id"] in wanted["$in"] if isinstance(wanted, dict) else meta["run_id"] == wanted

    def query(self, *, n_results, where=None, include=None, ids=None, query_texts=None, query_embeddings=None):
        self.queries += 1
        if ids is not None and not self.restrict_by_ids:
            raise TypeError("query() got an unexpected keyword argument 'ids'")
        order = [cid for cid in [*DENSE_ORDER, "other"] if self._matches(self.records[cid]["metadata"], where)]
        if ids is not None:
            order = [cid for cid in order if cid in ids]
        chosen = order[:n_results]
        return {
            "ids": [chosen],
            "documents": [[self.records[cid]["document"] for cid in chosen]],
            "metadatas": [[self.records[cid]["metadata"] for cid in chosen]],
            "distances": [[0.1 * [*DENSE_ORDER, "other"].index(cid) for cid in chosen]],
        }


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    run_1 = _insert_run("page-1", PAGE)
    run_2 = _insert_run("page-2", {"other": "Dominus vobiscum"})
    records = {
        cid: {"document": text, "metadata": {"run_id": run_1, "asset_ref": "page-1", "chunk_idx": i}}
        for i, (cid, text) in enumerate(PAGE.items())
    }
    records["other"] = {"document": "Dominus vobiscum", "metadata": {"run_id": run_2, "asset_ref": "page-2", "chunk_idx": 0}}
    col = _Collection(records)
    monkeypatch.setattr(rag_store, "_embed", lambda texts: None)
    monkeypatch.setattr(rag_store, "_ensure_chunk_runs_indexed", lambda run_ids: None)
    monkeypatch.setattr(rag_store, "_collection", lambda client=None, *, backend_key=None: col)
    monkeypatch.setattr(settings, "rag_lexical_fusion", True)
    monkeypatch.setattr(settings, "rag_fusion_pool", 50)
    monkeypatch.setattr(settings, "rag_lexical_max_chunks", 5000)
    return {"run_1": run_1, "run_2": run_2, "collection": col}


def _ids(hits: list[dict[str, Any]]) -> list[str]:
    return [hit["chunk_id"] for hit in hits]


def test_a_lexical_match_the_dense_ranking_buries_reaches_the_top(runs: dict[str, Any]) -> None:
    hits = rag_store.retrieve_chunks("Dominus vobiscum", top_k=2, run_ids=[runs["run_1"]])
    assert hits.mode == "hybrid"
    assert "c1" in _ids(hits)
    prayer = next(hit for hit in hits if hit["chunk_id"] == "c1")
    assert (prayer["dense_rank"], prayer["lexical_rank"]) == (4, 1)
    assert prayer["distance"] == pytest.approx(0.3)


def test_scope_is_respected_by_the_lexical_side(runs: dict[str, Any]) -> None:
    """run-2 holds an exact match, but the query is scoped to run-1."""
    hits = rag_store.retrieve_chunks("Dominus vobiscum", top_k=5, run_ids=[runs["run_1"]])
    assert "other" not in _ids(hits)


def test_no_lexical_overlap_leaves_the_dense_order(runs: dict[str, Any]) -> None:
    hits = rag_store.retrieve_chunks("θεος", top_k=4, run_ids=[runs["run_1"]])
    assert _ids(hits) == DENSE_ORDER
    assert all(hit["lexical_rank"] is None for hit in hits)


def test_fusion_off_returns_the_dense_ranking_unchanged(runs: dict[str, Any], monkeypatch) -> None:
    monkeypatch.setattr(settings, "rag_lexical_fusion", False)
    hits = rag_store.retrieve_chunks("Dominus vobiscum", top_k=2, run_ids=[runs["run_1"]])
    assert hits.mode == "dense"
    assert _ids(hits) == DENSE_ORDER[:2]
    assert "fusion_score" not in hits[0]


def test_lexical_failure_falls_back_to_dense(runs: dict[str, Any], monkeypatch, caplog) -> None:
    def unavailable(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("database locked")

    monkeypatch.setattr(pipeline_db, "list_chunks_for_runs", unavailable)
    with caplog.at_level(logging.WARNING, logger=rag_store.log.name):
        hits = rag_store.retrieve_chunks("Dominus vobiscum", top_k=2, run_ids=[runs["run_1"]])
    assert (hits.mode, _ids(hits)) == ("dense", DENSE_ORDER[:2])
    assert "Lexical retrieval failed" in caplog.text
    assert any("lexical ranking failed" in note for note in hits.notes)


def test_scopes_above_the_cap_skip_the_lexical_side(runs: dict[str, Any], monkeypatch) -> None:
    monkeypatch.setattr(settings, "rag_lexical_max_chunks", 3)
    hits = rag_store.retrieve_chunks("Dominus vobiscum", top_k=4, run_ids=[runs["run_1"]])
    assert (hits.mode, _ids(hits)) == ("dense", DENSE_ORDER)
    assert any("rag_lexical_max_chunks=3" in note for note in hits.notes)


def test_lexical_only_hits_get_a_dense_distance(runs: dict[str, Any], monkeypatch) -> None:
    """With a dense pool smaller than the scope, c1 is found only lexically."""
    monkeypatch.setattr(settings, "rag_fusion_pool", 1)
    fused = rag_store.retrieve_chunks("Dominus vobiscum", top_k=2, run_ids=[runs["run_1"]])
    # Dense pool [c0, c2], lexical [c1]: c0 and c1 tie at 1/61, first-seen order.
    assert _ids(fused) == ["c0", "c1"]
    prayer = fused[1]
    assert prayer["dense_rank"] is None
    assert prayer["distance"] == pytest.approx(0.3)


def test_missing_distance_is_shown_as_unavailable(runs: dict[str, Any], monkeypatch) -> None:
    """Chroma before 1.0 cannot restrict a query to ids; the block must still render."""
    runs["collection"].restrict_by_ids = False
    monkeypatch.setattr(settings, "rag_fusion_pool", 1)
    hits = rag_store.retrieve_chunks("Dominus vobiscum", top_k=2, run_ids=[runs["run_1"]])
    prayer = next(hit for hit in hits if hit["chunk_id"] == "c1")
    assert prayer["distance"] is None
    assert "retrieval_score: n/a" in rag_store.format_evidence_blocks([prayer])


def test_an_unembeddable_query_is_answered_lexically(runs: dict[str, Any], monkeypatch) -> None:
    """The provider is down: BM25 still ranks the page, and no other space is queried."""

    def down(_texts: list[str]) -> list[list[float]]:
        raise rag_store.EmbeddingUnavailable("multilingual-e5-large-instruct: 401 Unauthorized")

    monkeypatch.setattr(rag_store, "_embed", down)
    hits = rag_store.retrieve_chunks("Dominus vobiscum", top_k=2, run_ids=[runs["run_1"]])
    assert hits.mode == "lexical"
    assert _ids(hits)[0] == "c1"
    assert hits[0]["distance"] is None and hits[0]["lexical_rank"] == 1
    assert runs["collection"].queries == 0
    assert any("401 Unauthorized" in note for note in hits.notes)


def test_an_unembeddable_query_with_fusion_off_finds_nothing(runs: dict[str, Any], monkeypatch) -> None:
    def down(_texts: list[str]) -> list[list[float]]:
        raise rag_store.EmbeddingUnavailable("provider down")

    monkeypatch.setattr(rag_store, "_embed", down)
    monkeypatch.setattr(settings, "rag_lexical_fusion", False)
    hits = rag_store.retrieve_chunks("Dominus vobiscum", top_k=2, run_ids=[runs["run_1"]])
    assert (list(hits), hits.mode) == ([], "none")
