"""disambiguate applies the MIN_STRING_SIMILARITY gate it documents (audit AL-07).

The tiers defined MIN_STRING_SIMILARITY (0.60 / 0.65 / 0.70) and the docstring
listed it as gate 3, but nothing read it: a candidate whose name barely
resembled the mention could be linked on its type, source and context bonuses.
The gate now filters candidates like the type gate does, on the better of the
label and best-alias similarity in the score breakdown.
"""

from __future__ import annotations

from app.services.authority_scoring import _score_candidate
from app.services.entity_scoring import QUALITY_THRESHOLDS, disambiguate


def candidate(qid: str, score: float, *, label: float, alias: float = 0.0) -> dict:
    return {
        "qid": qid,
        "score": score,
        "type_compatible": True,
        "score_breakdown": {"label_similarity": label, "alias_match_quality": alias},
    }


def test_a_high_score_with_an_unlike_name_is_not_linked():
    result = disambiguate([candidate("Q1", 0.97, label=0.43)], ocr_quality="HIGH")
    assert result["status"] == "unresolved"
    assert result["selected"] is None
    assert "name similarity" in result["reason"]


def test_a_like_name_below_it_is_linked_instead():
    result = disambiguate(
        [candidate("Q1", 0.97, label=0.43), candidate("Q2", 0.90, label=0.95)],
        ocr_quality="HIGH",
    )
    assert (result["status"], result["selected"]["qid"]) == ("linked", "Q2")


def test_a_matching_alias_passes_the_gate():
    """A Latin or Old French form is often an alias, not the label: Artus for Arthur."""
    result = disambiguate([candidate("Q1", 0.95, label=0.40, alias=0.88)], ocr_quality="HIGH")
    assert result["status"] == "linked"


def test_the_threshold_follows_the_ocr_quality_tier():
    for tier, values in QUALITY_THRESHOLDS.items():
        floor = values["MIN_STRING_SIMILARITY"]
        at = disambiguate([candidate("Q1", 0.99, label=floor)], ocr_quality=tier)
        below = disambiguate([candidate("Q1", 0.99, label=floor - 0.01)], ocr_quality=tier)
        assert (at["status"], below["status"]) == ("linked", "unresolved"), tier


def test_the_margin_is_measured_against_eligible_candidates_only():
    result = disambiguate(
        [candidate("Q1", 0.95, label=0.95), candidate("Q2", 0.94, label=0.30)],
        ocr_quality="HIGH",
    )
    assert (result["status"], result["selected"]["qid"]) == ("linked", "Q1")


def test_a_candidate_without_a_breakdown_is_not_gated():
    result = disambiguate([{"qid": "Q1", "score": 0.95, "type_compatible": True}], ocr_quality="HIGH")
    assert result["status"] == "linked"


def test_the_scorer_records_the_similarities_the_gate_reads():
    _score, parts = _score_candidate(
        mention={"surface": "Artus", "ent_type": "person"},
        candidate={"label": "Arthur", "aliases": [{"value": "Artus"}], "description": "legendary king"},
        context="",
        canonical_norm="",
        domain_bonus=0.0,
        type_ok=True,
        resolved_tokens=set(),
    )
    assert 0.8 < parts["label_similarity"] < 1.0
    assert parts["alias_match_quality"] == 1.0
