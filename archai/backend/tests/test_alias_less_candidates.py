"""A candidate with no aliases is scored on its label, not on a 0.0 alias match.

Before, an alias-less candidate was capped near 0.75, under every auto-select
threshold, while the same candidate listing its label as an alias got the full
alias weight. Measured on the recorded linking cache: 21 candidates over 5
mentions, 8 alias-less; 1 crosses 0.80 with the fallback, and it is the right
entity (La satyre megere, Q135111996).
"""

from __future__ import annotations

import json
from pathlib import Path

from app.services import authority_scoring

BACKEND = Path(__file__).resolve().parents[1]
ANTITUS = next(p["reference"] for p in json.loads((BACKEND / "eval/quality_gate/readings.json").read_text())["pages"] if p["page_id"] == "middle-french-antitus")
SATYRE = {"label": "La satyre megere", "description": "work by Antitus Faure", "source": "wikidata", "aliases": [], "instance_of_qids": ["Q7725634"]}


def _score(candidate: dict, *, surface: str = "La satyre megere.", ent_type: str = "work", type_ok: bool = True) -> float:
    score, _ = authority_scoring._score_candidate(
        mention={"surface": surface, "ent_type": ent_type},
        candidate=candidate,
        context=ANTITUS[:400],
        canonical_norm="",
        domain_bonus=0.0,
        type_ok=type_ok,
        resolved_tokens=set(),
    )
    return score


def test_an_alias_less_exact_label_clears_the_high_threshold() -> None:
    assert _score(SATYRE) >= 0.80  # was 0.674


def test_alias_less_scores_like_a_label_repeated_as_alias() -> None:
    repeated = {**SATYRE, "aliases": [{"lang": "fr", "value": "La satyre megere"}]}
    assert _score(SATYRE) == _score(repeated)


def test_a_candidate_with_aliases_scores_exactly_as_before() -> None:
    """Its alias match is still its best alias's own similarity, not the label's."""
    with_alias = {**SATYRE, "aliases": [{"lang": "en", "value": "Satire of Megaera"}, {"lang": "fr", "value": "Satyre"}]}
    _, breakdown = authority_scoring._score_candidate(
        mention={"surface": "La satyre megere.", "ent_type": "work"},
        candidate=with_alias,
        context=ANTITUS[:400],
        canonical_norm="",
        domain_bonus=0.0,
        type_ok=True,
        resolved_tokens=set(),
    )
    expected = max(authority_scoring.string_similarity("La satyre megere.", a) for a in ("Satire of Megaera", "Satyre"))
    assert breakdown["alias_match_quality"] == round(expected, 4)
    assert expected < breakdown["label_similarity"]


def test_an_alias_less_type_incompatible_candidate_still_does_not_link() -> None:
    """'prince' -> Q16280695 in the recorded run: 0.465 before, 0.476 with the fallback."""
    prince = {"label": "Prince", "description": "family name", "source": "wikidata", "aliases": []}
    assert _score(prince, surface="prince", ent_type="title", type_ok=False) < 0.80
