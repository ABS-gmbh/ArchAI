"""retrieve_chunks fuses the dense ranking with character n-gram BM25."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from app.config import settings
from app.services import rag_store

PAGE = {
    "c0": "Incipit liber primus de natura rerum",
    "c1": "Oremus. dn̄s uobiscū et cū sp̄u tuo",
    "c2": "Explicit hic totum pro xp̄o da mihi potum",
    "c3": "In principio erat uerbum et uerbum erat apud deum",
}
# A dense retriever that has the prayer (c1) last, as the deployed embedding
# model does with abbreviated Latin.
DENSE_ORDER = ["c0", "c2", "c3", "c1"]


class _Collection:
    def __init__(self, *, restrict_by_ids: bool = True, fail_get: bool = False) -> None:
        self.records = {
            cid: {"document": text, "metadata": {"run_id": "run-1", "asset_ref": "page-1", "chunk_idx": i}}
            for i, (cid, text) in enumerate(PAGE.items())
        }
        self.records["other"] = {
            "document": "Dominus vobiscum",
            "metadata": {"run_id": "run-2", "asset_ref": "page-2", "chunk_idx": 0},
        }
        self.restrict_by_ids = restrict_by_ids
        self.fail_get = fail_get

    def _matches(self, meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
        return not where or all(meta.get(key) == value for key, value in where.items())

    def get(self, *, where=None, limit=None, include=None) -> dict[str, Any]:
        if self.fail_get:
            raise RuntimeError("store unavailable")
        rows = [(cid, r) for cid, r in self.records.items() if self._matches(r["metadata"], where)][:limit]
        return {
            "ids": [cid for cid, _ in rows],
            "documents": [r["document"] for _, r in rows],
            "metadatas": [r["metadata"] for _, r in rows],
        }

    def query(self, *, n_results, where=None, include=None, ids=None, query_texts=None, query_embeddings=None):
        if ids is not None and not self.restrict_by_ids:
            raise TypeError("query() got an unexpected keyword argument 'ids'")
        order = [cid for cid in DENSE_ORDER + ["other"] if self._matches(self.records[cid]["metadata"], where)]
        if ids is not None:
            order = [cid for cid in order if cid in ids]
        chosen = order[:n_results]
        return {
            "ids": [chosen],
            "documents": [[self.records[cid]["document"] for cid in chosen]],
            "metadatas": [[self.records[cid]["metadata"] for cid in chosen]],
            "distances": [[0.1 * (DENSE_ORDER + ["other"]).index(cid) for cid in chosen]],
        }


@pytest.fixture
def collection(monkeypatch: pytest.MonkeyPatch) -> _Collection:
    col = _Collection()
    monkeypatch.setattr(rag_store, "_provider_embed", lambda texts: (None, rag_store._LOCAL_EMBED_BACKEND))
    monkeypatch.setattr(rag_store, "_ensure_chunk_runs_indexed", lambda run_ids, *, backend_key: None)
    monkeypatch.setattr(rag_store, "_collection", lambda client=None, *, backend_key=None: col)
    monkeypatch.setattr(settings, "rag_lexical_fusion", True)
    monkeypatch.setattr(settings, "rag_fusion_pool", 50)
    monkeypatch.setattr(settings, "rag_lexical_max_chunks", 5000)
    return col


def _ids(hits: list[dict[str, Any]]) -> list[str]:
    return [hit["chunk_id"] for hit in hits]


def test_a_lexical_match_the_dense_ranking_buries_reaches_the_top(collection: _Collection) -> None:
    hits = rag_store.retrieve_chunks("Dominus vobiscum", top_k=2, run_ids=["run-1"])
    assert "c1" in _ids(hits)
    prayer = next(hit for hit in hits if hit["chunk_id"] == "c1")
    assert (prayer["dense_rank"], prayer["lexical_rank"]) == (4, 1)
    assert prayer["distance"] == pytest.approx(0.3)


def test_scope_is_respected_by_the_lexical_side(collection: _Collection) -> None:
    """run-2 holds an exact match, but the query is scoped to run-1."""
    hits = rag_store.retrieve_chunks("Dominus vobiscum", top_k=5, run_ids=["run-1"])
    assert "other" not in _ids(hits)


def test_no_lexical_overlap_leaves_the_dense_order(collection: _Collection) -> None:
    hits = rag_store.retrieve_chunks("θεος", top_k=4, run_ids=["run-1"])
    assert _ids(hits) == DENSE_ORDER
    assert all(hit["lexical_rank"] is None for hit in hits)


def test_fusion_off_returns_the_dense_ranking_unchanged(collection: _Collection, monkeypatch) -> None:
    monkeypatch.setattr(settings, "rag_lexical_fusion", False)
    hits = rag_store.retrieve_chunks("Dominus vobiscum", top_k=2, run_ids=["run-1"])
    assert _ids(hits) == DENSE_ORDER[:2]
    assert "fusion_score" not in hits[0]


def test_lexical_failure_falls_back_to_dense(collection: _Collection, caplog) -> None:
    collection.fail_get = True
    with caplog.at_level(logging.WARNING, logger=rag_store.log.name):
        hits = rag_store.retrieve_chunks("Dominus vobiscum", top_k=2, run_ids=["run-1"])
    assert _ids(hits) == DENSE_ORDER[:2]
    assert "Lexical retrieval failed" in caplog.text


def test_scopes_above_the_cap_skip_the_lexical_side(collection: _Collection, monkeypatch) -> None:
    monkeypatch.setattr(settings, "rag_lexical_max_chunks", 3)
    hits = rag_store.retrieve_chunks("Dominus vobiscum", top_k=4, run_ids=["run-1"])
    assert _ids(hits) == DENSE_ORDER


def test_lexical_only_hits_get_a_dense_distance(collection: _Collection, monkeypatch) -> None:
    """With a dense pool smaller than the scope, c1 is found only lexically."""
    monkeypatch.setattr(settings, "rag_fusion_pool", 1)
    fused = rag_store.retrieve_chunks("Dominus vobiscum", top_k=2, run_ids=["run-1"])
    # Dense pool [c0, c2], lexical [c1]: c0 and c1 tie at 1/61, first-seen order.
    assert _ids(fused) == ["c0", "c1"]
    prayer = fused[1]
    assert prayer["dense_rank"] is None
    assert prayer["distance"] == pytest.approx(0.3)


def test_missing_distance_is_shown_as_unavailable(collection: _Collection, monkeypatch) -> None:
    """Chroma before 1.0 cannot restrict a query to ids; the block must still render."""
    collection.restrict_by_ids = False
    monkeypatch.setattr(settings, "rag_fusion_pool", 1)
    hits = rag_store.retrieve_chunks("Dominus vobiscum", top_k=2, run_ids=["run-1"])
    prayer = next(hit for hit in hits if hit["chunk_id"] == "c1")
    assert prayer["distance"] is None
    assert "retrieval_score: n/a" in rag_store.format_evidence_blocks([prayer])
