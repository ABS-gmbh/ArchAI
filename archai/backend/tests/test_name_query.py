"""A person is asked for by name alone, and one name is never enough to link (held-out B-dev, cause 2).

A mention span carries what stood around the name: a title ("Sainct Pierre"),
a role ("Rey Rodrigo"), a word of the next sentence ("beati iohannis obtutibus
presentatus"). The authorities were asked for the whole span. The query is
now the name; the span shown stays as extracted, and the decision trace
records the query. A name reduced to one given name ("Pierre") matches
every bearer of it, so it links only when the page bears out the candidate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.db import pipeline_db
from app.services import authority_linking
from app.services import authority_scoring


@pytest.mark.parametrize(
    ("span", "name"),
    [
        ("Sainct Pierre", "Pierre"),
        ("sainct Frãcois lasche", "Frãcois"),
        ("beati iohannis obtutibus presentatus", "iohannis"),
        ("sancte iohannes interueni quesumus", "iohannes"),
        ("Rey Rodrigo", "Rodrigo"),
        ("Çiro Rey", "Çiro"),
        ("Aussi de Bar", "Bar"),
        ("Dominos Franciscum", "Franciscum"),
        ("Domini Martini Iusti", "Martini Iusti"),
        ("DISCIPLINIS BEATI COLUMBANI FRUDITUS", "COLUMBANI"),
        ("EST IOHANNES SUMMUS AMICUS DEI", "IOHANNES"),
        ("Sancti Marci. In", "Marci"),
        ("Theob̃. Walt̃", "Theob Walt"),  # a point after an abbreviated word does not end the name
        ("Rog̾o de Renni", "Rogo de Renni"),  # marks dropped, not turned into spaces ("Rog o")
        ("Roys mͣrcadigas", "marcadigas"),  # a superscript letter is a letter
        ("Thomas de hereforde", "Thomas de hereforde"),  # a particle keeps the next word
        ("Rommant de la Rose", "Rommant de la Rose"),
        ("Pierre Gringoire", "Pierre Gringoire"),
    ],
)
def test_the_name_is_cut_out_of_the_span(span: str, name: str) -> None:
    assert authority_scoring._name_query(span) == name


def test_a_span_with_nothing_but_titles_is_queried_as_it_is() -> None:
    assert authority_scoring._name_query("Sancti") == "Sancti"


def _seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, surface: str, text: str) -> str:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    run_id = pipeline_db.create_run(asset_ref="page-1", asset_sha256="sha")
    with pipeline_db._connect() as conn:
        conn.execute(
            "UPDATE pipeline_runs SET ocr_text=?, proofread_text=?, warnings_json=? WHERE run_id=?",
            (text, text, json.dumps({"quality_label": "HIGH"}), run_id),
        )
        conn.commit()
    pipeline_db.insert_chunks(run_id, [{"chunk_id": "c1", "idx": 0, "start_offset": 0, "end_offset": len(text), "text": text}])
    start = text.index(surface)
    pipeline_db.insert_entity_mentions(
        run_id,
        [
            {
                "mention_id": "m1", "chunk_id": "c1", "start_offset": start, "end_offset": start + len(surface),
                "surface": surface, "norm": surface.lower(), "label": "PERSON", "ent_type": "person",
                "confidence": 0.9, "method": "rule:proper_name_sequence",
            }
        ],
    )
    return run_id


def _breakdown(row: dict[str, Any]) -> dict[str, Any]:
    value = row["score_breakdown_json"]
    return value if isinstance(value, dict) else json.loads(value)


def _one_candidate(monkeypatch: pytest.MonkeyPatch, *, label: str, description: str, asked: list[str]) -> None:
    """Wikidata answers every query with one person of that exact name; the other sources with nothing."""

    def search(query: str, **_kwargs: Any) -> list[dict[str, Any]]:
        asked.append(query)
        return [{"qid": "Q1", "label": label, "description": description, "url": "https://www.wikidata.org/wiki/Q1"}]

    monkeypatch.setattr(authority_linking, "cache_check", lambda *args, **kwargs: (None, "miss"))
    monkeypatch.setattr(authority_linking, "search_wikidata", search)
    monkeypatch.setattr(authority_linking, "search_viaf", lambda *args, **kwargs: [])
    monkeypatch.setattr(authority_linking, "search_geonames", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        authority_linking,
        "enrich_wikidata_item",
        lambda qid: {"instance_of_qids": ["Q5"], "canonical_label": label, "description": description, "aliases": [], "country_qids": [], "admin_qids": []},
    )
    monkeypatch.setattr(authority_linking, "is_type_compatible", lambda *args, **kwargs: True)


def test_the_authorities_are_asked_for_the_name_and_the_trace_records_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = _seed(tmp_path, monkeypatch, surface="Rey Rodrigo Diaz", text="Et el Rey Rodrigo Diaz de Bivar, caballero de Castilla, fue a Valencia.")
    asked: list[str] = []
    _one_candidate(monkeypatch, label="Rodrigo Diaz", description="Castilian knight", asked=asked)
    authority_linking.run_authority_linking(run_id, top_k=3)
    assert asked and all("Rey" not in query for query in asked)
    row = pipeline_db.list_mention_links_for_run(run_id)[0]
    breakdown = _breakdown(row)
    assert breakdown["mention_surface"] == "Rey Rodrigo Diaz"  # the span shown is unchanged
    assert breakdown["name_query"] == "Rodrigo Diaz"


def test_one_name_does_not_link_on_the_name_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # On B-dev "Sainct Pierre" gave Pierre Abélard 0.81, over the threshold;
    # here the only candidate, so no margin can hold it back.
    run_id = _seed(tmp_path, monkeypatch, surface="Sainct Pierre", text="¶ Sainct Pierre print le glaiue de sainct Pol")
    _one_candidate(monkeypatch, label="Pierre", description="French theologian and philosopher", asked=[])
    authority_linking.run_authority_linking(run_id, top_k=3)
    row = pipeline_db.list_mention_links_for_run(run_id)[0]
    assert not row.get("wikidata_qid")
    breakdown = _breakdown(row)
    assert "one name ('Pierre')" in breakdown["reason"]


def test_one_name_links_when_the_page_bears_the_candidate_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = _seed(tmp_path, monkeypatch, surface="Sainct Pierre", text="¶ Sainct Pierre apostle print le glaiue de sainct Pol")
    _one_candidate(monkeypatch, label="Pierre", description="apostle of Jesus", asked=[])
    authority_linking.run_authority_linking(run_id, top_k=3)
    assert pipeline_db.list_mention_links_for_run(run_id)[0].get("wikidata_qid") == "Q1"


def test_the_name_alone_never_counts_as_support() -> None:
    # A VIAF description repeats the label: "Personal | Pierre".
    candidate = {"label": "Pierre", "description": "Personal | Pierre", "canonical_description": "VIAF authority record"}
    assert not authority_linking._context_supports("Sainct Pierre print le glaiue", candidate, "Pierre")
