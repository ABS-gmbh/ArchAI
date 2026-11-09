"""Wikidata searches are cached per language (audit AL-13).

The cache key left out the search language, though both _cache_key and
search_wikidata's docstring said it was included: a search for the same
surface in a second language read back the first language's results.
"""

from __future__ import annotations

from app.services import wikidata_client


def _fake_search(monkeypatch) -> list[str]:
    languages: list[str] = []

    def http_get(url, params):
        languages.append(params["language"])
        return {"search": [{"id": "Q45720", "label": f"Arthur ({params['language']})", "description": ""}]}

    monkeypatch.setattr(wikidata_client, "_http_get", http_get)
    monkeypatch.setattr(wikidata_client, "_rate_limit", lambda: None)
    return languages


def test_each_language_gets_its_own_cached_search(monkeypatch):
    """Audit AL-13: an English search read back the French one's cached labels."""
    languages = _fake_search(monkeypatch)
    french = wikidata_client.search_wikidata("Artus lang test", language="fr")
    english = wikidata_client.search_wikidata("Artus lang test", language="en")
    again = wikidata_client.search_wikidata("Artus lang test", language="en")
    assert languages == ["fr", "en"]
    assert (french[0]["label"], english[0]["label"], again[0]["label"]) == (
        "Arthur (fr)", "Arthur (en)", "Arthur (en)"
    )


def test_the_linker_checks_the_cache_its_searches_fill(monkeypatch):
    _fake_search(monkeypatch)
    wikidata_client.search_wikidata("Gauvain lang test")
    _, status = wikidata_client.cache_check(
        "wikidata", "Gauvain lang test", max_age_hours=6.0, lang=wikidata_client.DEFAULT_SEARCH_LANGUAGE
    )
    assert status == "hit"
