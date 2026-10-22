"""Tests that a failed Wikidata fetch is never cached as an answer.

_http_get swallowed every failure - HTTP error, timeout, dropped connection,
malformed JSON - and returned {}. enrich_wikidata_item could not tell that from
"this entity genuinely has no claims", so it built an all-empty enrichment and
cached it.

Two guards that should have caught it both missed:

* cache_put refuses to store empty results, but the call site passes [result] -
  a one-element list holding an empty dict, which is not empty.
* cache_get expires empty cached entries against max_age_hours, but the call
  site passed no TTL, and non-empty entries are returned regardless of age.

With instance_of_qids empty, is_type_compatible then rejects the candidate. One
transient network failure made that entity permanently unlinkable.
"""

from __future__ import annotations

import pytest

import app.services.wikidata_client as wikidata_client
from app.services.wikidata_client import WikidataUnavailable, is_type_compatible


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    """Give every test its own cache DB and no rate limiting.

    These tests assert on cache behaviour, so sharing the real on-disk cache
    would make them order-dependent: a QID written by one test would be served
    from cache in the next.
    """
    monkeypatch.setattr(wikidata_client, "_rate_limit", lambda: None)
    monkeypatch.setattr(wikidata_client, "_cache_path", lambda: tmp_path / "wikidata_cache.sqlite")
    monkeypatch.setattr(wikidata_client, "_CACHE_READY", False)


def entity_payload(qid: str, p31: str = "Q5") -> dict:
    return {"entities": {qid: {"claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": p31}}}}]}}}}


def test_http_failures_raise_rather_than_look_like_empty_answers(monkeypatch) -> None:
    """The root cause: {} meant both 'no claims' and 'request failed'."""
    import urllib.request

    def boom(*_args, **_kwargs):
        raise OSError("connection reset")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(WikidataUnavailable):
        wikidata_client._http_get("https://example.invalid", {"a": "b"})


def test_a_failed_enrichment_is_not_cached(monkeypatch) -> None:
    """Regression: the failure was stored and served forever."""
    calls = {"n": 0}

    def always_down(_url, _params):
        calls["n"] += 1
        raise WikidataUnavailable("outage")

    monkeypatch.setattr(wikidata_client, "_http_get", always_down)
    qid = "Q_TEST_NOT_CACHED"
    wikidata_client.enrich_wikidata_item(qid)
    wikidata_client.enrich_wikidata_item(qid)
    assert calls["n"] == 2, "the second call must retry rather than serve a cached failure"


def test_an_entity_recovers_after_a_transient_outage(monkeypatch) -> None:
    calls = {"n": 0}
    down = {"value": True}
    qid = "Q_TEST_RECOVERS"

    def flaky(_url, _params):
        calls["n"] += 1
        if down["value"]:
            raise WikidataUnavailable("outage")
        return entity_payload(qid)

    monkeypatch.setattr(wikidata_client, "_http_get", flaky)
    assert wikidata_client.enrich_wikidata_item(qid).get("instance_of_qids") in (None, [])

    down["value"] = False
    recovered = wikidata_client.enrich_wikidata_item(qid)
    assert recovered["instance_of_qids"] == ["Q5"]
    assert is_type_compatible("person", recovered["instance_of_qids"]) is True


def test_a_failed_enrichment_returns_an_empty_mapping(monkeypatch) -> None:
    """Callers must not see a half-built record that looks authoritative."""
    monkeypatch.setattr(
        wikidata_client, "_http_get", lambda _u, _p: (_ for _ in ()).throw(WikidataUnavailable("x"))
    )
    assert wikidata_client.enrich_wikidata_item("Q_TEST_EMPTY") == {}


def test_a_successful_enrichment_is_still_cached(monkeypatch) -> None:
    """The fix must not disable caching for real answers."""
    calls = {"n": 0}
    qid = "Q_TEST_CACHED_OK"

    def once(_url, _params):
        calls["n"] += 1
        return entity_payload(qid)

    monkeypatch.setattr(wikidata_client, "_http_get", once)
    first = wikidata_client.enrich_wikidata_item(qid)
    second = wikidata_client.enrich_wikidata_item(qid)
    assert first["instance_of_qids"] == second["instance_of_qids"] == ["Q5"]
    assert calls["n"] == 1, "a successful result should be served from cache"


def test_search_does_not_cache_an_outage(monkeypatch) -> None:
    calls = {"n": 0}

    def always_down(_url, _params):
        calls["n"] += 1
        raise WikidataUnavailable("outage")

    monkeypatch.setattr(wikidata_client, "_http_get", always_down)
    assert wikidata_client.search_wikidata("Lancelot") == []
    wikidata_client.search_wikidata("Lancelot")
    assert calls["n"] == 2, "a failed search must not be cached as 'no candidates'"


def test_empty_enrichments_carry_a_ttl() -> None:
    """A genuinely sparse entity should be re-asked eventually, not written off."""
    assert wikidata_client._ENRICH_EMPTY_TTL_HOURS > 0
