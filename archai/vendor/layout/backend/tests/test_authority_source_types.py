"""Tests for type compatibility of non-Wikidata authority sources.

is_type_compatible is precision-first: with no P31 values it returns False, so
untyped Wikidata items are never auto-linked. VIAF and GeoNames candidates were
constructed with instance_of_qids=[], so every one of them was judged
incompatible - the two sources were queried on every person, work and place
mention and could never contribute a link.

Separately, the type check was a veto on the TOP-ranked candidate rather than a
filter over the field, so one incompatible candidate landing first made the whole
mention unresolvable even when a perfectly typed candidate sat directly beneath.
"""

from __future__ import annotations

import pytest

from app.services.authority_sources import (
    _GEONAMES_CLASS_QIDS,
    _GEONAMES_FCODE_QIDS,
    geonames_type_qids,
    viaf_type_qids,
)
from app.services.entity_scoring import disambiguate
from app.services.wikidata_client import is_type_compatible


def candidate(cid: str, score: float, *, compatible: bool, label: str = "Lausanne") -> dict:
    return {"id": cid, "qid": cid, "label": label, "score": score, "type_compatible": compatible}


# ───────────────────────────────────────── sources assert their type ──


@pytest.mark.parametrize("fcode", sorted(_GEONAMES_FCODE_QIDS))
def test_every_geonames_feature_code_is_place_compatible(fcode: str) -> None:
    assert is_type_compatible("place", geonames_type_qids(fcode))


@pytest.mark.parametrize("fclass", sorted(_GEONAMES_CLASS_QIDS))
def test_every_geonames_feature_class_is_place_compatible(fclass: str) -> None:
    assert is_type_compatible("place", geonames_type_qids("UNKNOWN_CODE", fclass))


def test_an_unknown_geonames_code_still_reads_as_a_place() -> None:
    """A GeoNames hit is a place by construction."""
    assert is_type_compatible("place", geonames_type_qids("", ""))


@pytest.mark.parametrize(
    ("name_type", "ent_type"),
    [("Personal", "person"), ("Corporate", "org"), ("UniformTitle", "work"), ("Geographic", "place")],
)
def test_viaf_name_types_map_to_compatible_qids(name_type: str, ent_type: str) -> None:
    assert is_type_compatible(ent_type, viaf_type_qids(name_type))


def test_viaf_name_type_matching_ignores_case_and_separators() -> None:
    assert viaf_type_qids("uniform-title") == viaf_type_qids("UniformTitle")


def test_an_unknown_viaf_name_type_asserts_nothing() -> None:
    """Better to stay untyped than to invent a type."""
    assert viaf_type_qids("SomethingElse") == []


def test_untyped_wikidata_items_are_still_rejected() -> None:
    """The precision-first rule must survive for Wikidata itself."""
    assert is_type_compatible("person", []) is False
    assert is_type_compatible("place", []) is False


# ─────────────────────────────────── the gate filters, not vetoes ──


def test_a_typed_candidate_below_an_untyped_one_is_reachable() -> None:
    """Regression: the mention was unresolvable because ranked[0] was incompatible."""
    result = disambiguate(
        [candidate("X1", 0.95, compatible=False), candidate("Q806", 0.90, compatible=True)],
        ocr_quality="HIGH",
    )
    assert result["status"] == "linked"
    assert result["selected"]["id"] == "Q806"


def test_incompatible_candidates_are_never_selected() -> None:
    result = disambiguate(
        [candidate("X1", 0.99, compatible=False), candidate("Q806", 0.90, compatible=True)],
        ocr_quality="HIGH",
    )
    assert result["selected"]["type_compatible"] is True


def test_all_incompatible_still_resolves_to_nothing() -> None:
    result = disambiguate(
        [candidate("X1", 0.95, compatible=False), candidate("X2", 0.80, compatible=False)],
        ocr_quality="HIGH",
    )
    assert result["status"] == "unresolved"
    assert "type_incompatible" in result["reason"]


def test_the_full_candidate_list_is_still_reported() -> None:
    """Filtering affects selection, not the audit trail."""
    result = disambiguate(
        [candidate("X1", 0.95, compatible=False), candidate("Q806", 0.90, compatible=True)],
        ocr_quality="HIGH",
    )
    assert len(result["all"]) == 2


def test_margin_is_measured_against_the_next_eligible_candidate() -> None:
    """An incompatible runner-up must not manufacture ambiguity."""
    result = disambiguate(
        [
            candidate("X1", 0.93, compatible=False),
            candidate("Q1", 0.92, compatible=True),
            candidate("Q2", 0.40, compatible=True, label="Other"),
        ],
        ocr_quality="HIGH",
    )
    assert result["status"] == "linked"
    assert result["selected"]["id"] == "Q1"


def test_genuine_ambiguity_between_eligible_candidates_is_still_caught() -> None:
    result = disambiguate(
        [candidate("Q1", 0.92, compatible=True), candidate("Q2", 0.90, compatible=True, label="Other")],
        ocr_quality="HIGH",
    )
    assert result["status"] == "ambiguous"


def test_an_empty_candidate_list_is_unresolved() -> None:
    assert disambiguate([], ocr_quality="HIGH")["status"] == "unresolved"
