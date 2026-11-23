"""A VIAF candidate's work titles reach the context text (not the key "work")."""

from __future__ import annotations

from app.services import authority_scoring
from app.services.authority_scoring import _title_texts

VIAF_TITLES = {"work": [{"title": "La satyre megere"}, {"title": "Epistres"}]}


def test_viaf_titles_give_their_title_strings() -> None:
    assert _title_texts(VIAF_TITLES) == ["La satyre megere", "Epistres"]
    assert _title_texts({"work": {"title": "Seule"}}) == ["Seule"]  # one work is not wrapped in a list
    assert _title_texts(["A", "B"]) == ["A", "B"]
    assert _title_texts(None) == [] and _title_texts({}) == []


def test_the_context_score_sees_the_titles(monkeypatch) -> None:
    seen: list[str] = []
    real = authority_scoring.context_coverage

    def spy(context: str, description_text: str) -> float:
        seen.append(description_text)
        return real(context, description_text)

    monkeypatch.setattr(authority_scoring, "context_coverage", spy)
    authority_scoring._score_candidate(
        mention={"surface": "Antitus", "ent_type": "person"},
        candidate={"label": "Antitus Faure", "source": "viaf", "titles": VIAF_TITLES, "aliases": []},
        context="La satyre megere Antitus",
        canonical_norm="",
        domain_bonus=0.0,
        type_ok=True,
        resolved_tokens=set(),
    )
    assert "La satyre megere" in seen[0]
    assert "work" not in seen[0].split()
