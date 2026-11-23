"""rag_store reads authority aliases and assertions through pipeline_db's public API (RAG-04).

It reached into the private ``_connect()``, which skips schema initialisation:
on a fresh database the entity index failed with "no such table:
authority_aliases" unless some other call had happened to create the schema.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.db import pipeline_db
from app.services import rag_store


@pytest.fixture
def fresh_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)


def test_aliases_load_on_a_database_nothing_has_initialised(fresh_db: None) -> None:
    assert rag_store._load_authority_aliases(["Q1"]) == {"Q1": []}
    assert rag_store._load_authority_assertions(["Q1"]) == {"Q1": []}


def _entity(entity_id: str) -> None:
    pipeline_db.upsert_authority_entity({"entity_id": entity_id, "authority_source": "wikidata", "authority_id": entity_id})


def test_aliases_and_assertions_round_trip(fresh_db: None) -> None:
    _entity("Q1")
    pipeline_db.replace_authority_aliases("Q1", [{"alias": "Arthur"}, {"alias": "Artus"}, {"alias": " "}])
    pipeline_db.replace_authority_source_assertions(
        "Q1",
        [
            {"source_name": "wikidata", "property_name": "P31", "property_value": "Q5"},
            {"source_name": "viaf", "property_name": "name", "property_value": ""},
        ],
    )
    assert rag_store._load_authority_aliases(["Q1", "Q2", ""]) == {"Q1": ["Arthur", "Artus"], "Q2": []}
    assert rag_store._load_authority_assertions(["Q1"]) == {"Q1": ["viaf name", "wikidata P31 Q5"]}


def test_more_ids_than_one_sqlite_batch(fresh_db: None) -> None:
    ids = [f"Q{i}" for i in range(1, 1201)]
    _entity("Q1200")
    pipeline_db.replace_authority_aliases("Q1200", [{"alias": "Last"}])
    aliases = rag_store._load_authority_aliases(ids)
    assert len(aliases) == 1200
    assert aliases["Q1200"] == ["Last"]


def test_rag_store_no_longer_touches_the_private_connection() -> None:
    source = Path(rag_store.__file__).read_text(encoding="utf-8")
    assert "pipeline_db._connect" not in source
