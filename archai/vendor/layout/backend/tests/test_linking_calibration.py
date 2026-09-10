"""Calibration tests for authority linking.

The composite scorer could not produce a link. Its 0.25-weighted term, named
alias_sim, was fed context_similarity(context, description) - a Jaccard over
token SETS between a 200-400 character Old French context window and a 2-10
token English Wikidata gloss, which measures 0.00-0.03 however well the
candidate fits. That capped compute_score at 0.75 and the full composite at
0.7936, below every AUTO_SELECT threshold (0.80 / 0.85 / 0.90).

The same context quantity was then added a SECOND time by the orchestrator at
weight 0.08, so context contributed 0.25 of the total while measuring ~0.014,
and the genuine alias match - 1.0 on an exact hit - got only 0.12.

The fix is calibration of the scorer, not relaxation of the thresholds: a
correct candidate must clear the bar while a wrong one still fails it.
"""

from __future__ import annotations

import pytest

from app.services.authority_linking import _score_candidate
from app.services.entity_scoring import (
    QUALITY_THRESHOLDS,
    compute_score,
    context_coverage,
    context_similarity,
)

CONTEXT = (
    "Et quant li rois ot ce oi il fu mout dolenz et corrociez si comme cil qui mout "
    "amoit le chevalier Lancelot del Lac et por ce que il estoit ses compaignons de la "
    "Table Reonde si li dist que il ne feroit mie ce que il li avoit requis"
)
DESCRIPTION = "knight of the Round Table in Arthurian legend"


def perfect_score() -> float:
    """Surface equals label, type-compatible, exact alias hit, domain saturated."""
    return compute_score(
        "Lancelot", "Lancelot", CONTEXT, DESCRIPTION,
        type_compatible=True, domain_bonus=0.25, alias_sim=1.0,
    )


# ─────────────────────────────────────────────── the calibration bug ──


@pytest.mark.parametrize("tier", ["HIGH", "MEDIUM", "LOW"])
def test_a_perfect_candidate_clears_every_threshold(tier: str) -> None:
    """Regression: the achievable maximum was 0.7936, below all three."""
    assert perfect_score() >= QUALITY_THRESHOLDS[tier]["AUTO_SELECT_THRESHOLD"]


def test_weights_sum_to_one_so_a_perfect_score_can_reach_one() -> None:
    from app.services.entity_scoring import (
        _W_ALIAS, _W_CONTEXT, _W_DOMAIN, _W_LABEL, _W_TYPE,
    )

    assert _W_LABEL + _W_ALIAS + _W_TYPE + _W_CONTEXT + _W_DOMAIN == pytest.approx(1.0)


def test_thresholds_were_not_lowered_to_make_this_pass() -> None:
    """The scorer was fixed, not the bar moved."""
    assert QUALITY_THRESHOLDS["HIGH"]["AUTO_SELECT_THRESHOLD"] == 0.80
    assert QUALITY_THRESHOLDS["MEDIUM"]["AUTO_SELECT_THRESHOLD"] == 0.85
    assert QUALITY_THRESHOLDS["LOW"]["AUTO_SELECT_THRESHOLD"] == 0.90


# ──────────────────────────────────────────────── discrimination ──


def test_a_wrong_candidate_is_still_rejected() -> None:
    """Calibration must not come at the cost of precision."""
    wrong = compute_score(
        "Lancelot", "Lanzarote", CONTEXT, "island in the Canary Islands",
        type_compatible=True, domain_bonus=0.0, alias_sim=0.3,
    )
    assert wrong < QUALITY_THRESHOLDS["HIGH"]["AUTO_SELECT_THRESHOLD"]


def test_the_margin_between_right_and_wrong_exceeds_min_margin() -> None:
    wrong = compute_score(
        "Lancelot", "Lanzarote", CONTEXT, "island in the Canary Islands",
        type_compatible=True, domain_bonus=0.0, alias_sim=0.3,
    )
    assert perfect_score() - wrong >= QUALITY_THRESHOLDS["HIGH"]["MIN_MARGIN"]


def test_type_incompatibility_remains_a_hard_penalty() -> None:
    incompatible = compute_score(
        "Lancelot", "Lancelot", CONTEXT, "commune in France",
        type_compatible=False, domain_bonus=0.0, alias_sim=1.0,
    )
    assert incompatible < QUALITY_THRESHOLDS["HIGH"]["AUTO_SELECT_THRESHOLD"]


# ────────────────────────────────────────── the alias signal itself ──


def test_alias_similarity_actually_moves_the_score() -> None:
    """The 0.25 weight was dead: it received a quantity stuck near 0.01."""
    low = compute_score("Lancelot", "Lancelot", CONTEXT, DESCRIPTION, alias_sim=0.0)
    high = compute_score("Lancelot", "Lancelot", CONTEXT, DESCRIPTION, alias_sim=1.0)
    assert high - low == pytest.approx(0.25, abs=0.01)


def test_aliases_are_used_when_no_explicit_similarity_is_given() -> None:
    scored = compute_score(
        "Lancelot du Lac", "Lancelot", CONTEXT, DESCRIPTION,
        candidate_aliases=["Lancelot du Lac", "Lanval"],
    )
    assert scored > compute_score("Lancelot du Lac", "Lancelot", CONTEXT, DESCRIPTION)


# ───────────────────────────────────────────── the context signal ──


def test_context_coverage_is_not_swamped_by_context_length() -> None:
    """Jaccard's union is dominated by context tokens the gloss cannot contain."""
    assert context_coverage(CONTEXT, DESCRIPTION) > context_similarity(CONTEXT, DESCRIPTION)


def test_a_fully_supported_description_scores_highly() -> None:
    assert context_coverage(CONTEXT, "Round Table Lancelot") >= 0.6


def test_an_unsupported_description_scores_zero() -> None:
    assert context_coverage(CONTEXT, "quantum chromodynamics") == 0.0


def test_context_coverage_handles_empty_input() -> None:
    assert context_coverage("", DESCRIPTION) == 0.0
    assert context_coverage(CONTEXT, "") == 0.0


# ──────────────────────────────────────── full orchestrator path ──


def candidate(label: str, description: str, aliases: list[str]) -> dict:
    return {
        "id": "Q1", "label": label, "description": description,
        "aliases": [{"value": a} for a in aliases], "source": "wikidata",
    }


def test_score_candidate_links_a_correct_mention() -> None:
    score, detail = _score_candidate(
        mention={"surface": "Lancelot", "label": "person", "ent_type": "person"},
        candidate=candidate("Lancelot", DESCRIPTION, ["Lancelot du Lac", "Lancelot"]),
        context=CONTEXT, canonical_norm="lancelot", domain_bonus=0.25,
        type_ok=True, resolved_tokens=set(),
    )
    assert score >= QUALITY_THRESHOLDS["LOW"]["AUTO_SELECT_THRESHOLD"]
    assert detail["alias_match_quality"] == pytest.approx(1.0)


def test_score_candidate_still_separates_a_wrong_mention() -> None:
    correct, _ = _score_candidate(
        mention={"surface": "Lancelot", "label": "person", "ent_type": "person"},
        candidate=candidate("Lancelot", DESCRIPTION, ["Lancelot"]),
        context=CONTEXT, canonical_norm="lancelot", domain_bonus=0.25,
        type_ok=True, resolved_tokens=set(),
    )
    wrong, _ = _score_candidate(
        mention={"surface": "Lancelot", "label": "person", "ent_type": "person"},
        candidate=candidate("Guinevere", "queen consort of King Arthur", ["Guenievre"]),
        context=CONTEXT, canonical_norm="lancelot", domain_bonus=0.25,
        type_ok=True, resolved_tokens=set(),
    )
    assert correct > wrong
    assert correct - wrong >= QUALITY_THRESHOLDS["HIGH"]["MIN_MARGIN"]
