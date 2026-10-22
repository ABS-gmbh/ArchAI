"""Tests for character n-gram BM25 and reciprocal rank fusion."""

from __future__ import annotations

import math

import pytest

from app.services.lexical_retrieval import LexicalIndex, reciprocal_rank_fusion, search_terms

# ── terms ───────────────────────────────────────────────────────────────


def test_tokens_are_padded_so_word_edges_are_terms() -> None:
    assert search_terms("rois") == [" roi", "rois", "ois "]


def test_short_tokens_are_kept_whole() -> None:
    assert search_terms("et") == [" et "]


def test_terms_come_from_the_search_key() -> None:
    """Abbreviated, orthographic and line-broken forms meet their expansions."""
    assert search_terms("dn̄s") == search_terms("dominus")
    assert search_terms("vir") == search_terms("uir")
    assert search_terms("Iohannes") == search_terms("johannes")
    assert search_terms("domi-\nnus") == search_terms("dominus")


def test_empty_text_has_no_terms() -> None:
    assert search_terms("") == []
    assert search_terms("   ") == []


# ── BM25 ────────────────────────────────────────────────────────────────


def test_bm25_matches_a_hand_computation() -> None:
    index = LexicalIndex([("d1", "a b"), ("d2", "a a c")], terms=str.split)
    ((doc, score),) = index.search("c")
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    norm = 1.2 * (1 - 0.75 + 0.75 * 3 / 2.5)
    assert doc == "d2"
    assert score == pytest.approx(idf * 1 * 2.2 / (1 + norm))


def test_documents_sharing_no_term_are_omitted() -> None:
    """An empty lexical ranking is what leaves the dense ranking untouched."""
    index = LexicalIndex([("d1", "alpha"), ("d2", "beta")], terms=str.split)
    assert [doc for doc, _ in index.search("alpha")] == ["d1"]
    assert index.search("gamma") == []
    assert index.search("") == []


def test_rarer_terms_weigh_more() -> None:
    index = LexicalIndex([("common", "x y"), ("rare", "x z"), ("other", "x w")], terms=str.split)
    assert index.search("x z")[0][0] == "rare"


def test_ties_keep_document_order_and_limit_applies() -> None:
    index = LexicalIndex([("first", "x"), ("second", "x"), ("third", "x")], terms=str.split)
    assert [doc for doc, _ in index.search("x")] == ["first", "second", "third"]
    assert [doc for doc, _ in index.search("x", limit=2)] == ["first", "second"]


def test_empty_index_is_searchable() -> None:
    index = LexicalIndex([])
    assert len(index) == 0
    assert index.search("anything") == []


def test_abbreviated_diplomatic_text_is_found_by_its_expansion() -> None:
    docs = [
        ("prayer", "Oremus. dn̄s uobiscū et cū sp̄u tuo"),
        ("rubric", "Incipit liber primus de natura rerum"),
        ("colophon", "Explicit hic totum pro xp̄o da mihi potum"),
    ]
    assert LexicalIndex(docs).search("Dominus vobiscum")[0][0] == "prayer"


def test_a_passage_survives_a_different_ocr_reading() -> None:
    """Real lines from two OCR readings of one page; no word matches exactly."""
    page = [
        ("c0", "age fu enfez tendu e ala roys qu'uncz fu le Lait\nrequerez demaider por"),
        ("c1", "I gnt confeil vilam deucre\nlon auor gent a"),
        ("c2", "les nocz gus Quer arrez telement\nsi redit on"),
    ]
    assert LexicalIndex(page).search("a qite confel vilain defaure")[0][0] == "c1"


# ── fusion ──────────────────────────────────────────────────────────────


def test_rrf_matches_a_hand_computation() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]], k=60)
    assert [item for item, _ in fused] == ["a", "c", "b"]
    assert dict(fused)["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert dict(fused)["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert dict(fused)["b"] == pytest.approx(1 / 62)


def test_rrf_rewards_agreement_over_a_single_first_place() -> None:
    fused = reciprocal_rank_fusion([["solo", "both"], ["x", "both"], ["y", "both"]])
    assert fused[0][0] == "both"


def test_rrf_with_one_empty_ranking_keeps_the_other() -> None:
    assert [item for item, _ in reciprocal_rank_fusion([["a", "b", "c"], []])] == ["a", "b", "c"]


def test_rrf_ties_keep_first_seen_order() -> None:
    assert [item for item, _ in reciprocal_rank_fusion([["a", "b"], ["b", "a"]])] == ["a", "b"]


def test_rrf_rejects_negative_k() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)
