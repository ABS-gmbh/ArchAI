"""Score authority linking on the hand-annotated reference pages.

Usage::

    python scripts/benchmark_linking.py                    # replay eval/entities/linking_cache.json
    python scripts/benchmark_linking.py --record           # ask Wikidata and VIAF, and rewrite the cache
    python scripts/benchmark_linking.py --json

``eval/entities/gold.json`` gives, for each named entity of the reference
pages, the Wikidata QIDs and VIAF ids a correct link may name (``authority``;
see ``eval/entities/README.md``). Each page with entities is loaded into a
throwaway pipeline database and authority cache in a temporary directory
(never ``app/archai.sqlite``), its mentions extracted as the pipeline extracts
them, and ``run_authority_linking`` run on it. Every gold entity is then scored:

- ``correct``: a mention overlapping it was linked to an accepted QID or VIAF
  id, or, for an entity with no authority record (``none``), left unlinked;
- ``wrong``: linked to anything else, or linked at all where no record exists;
- ``abstained``: a mention was found, but not linked, although a record exists;
- ``missed-mention``: no extracted mention overlaps it.
Dates (``n/a``) are not scored. The report gives k/n per page and in total;
with 13 scored entities there is no headline rate.

Linking makes live Wikidata and VIAF requests. ``--record`` makes them once,
politely (the app's User-Agent, at most one request a second, stopping on a
429 or 403), and stores every response in the cache file, trimmed to what the
linker reads: of a Wikidata entity its labels, descriptions, aliases and the
main values of P31, P214, P1566, P625, P17 and P131; of a VIAF cluster its
viafID, nameType, and the text of its mainHeadings and titles. Without
``--record`` the responses come from that file, so the numbers are
reproducible offline, and a request missing from it is answered as a failed
one (counted as ``cache_misses``). GeoNames is not used: it needs a
GEONAMES_USERNAME.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOLD = ROOT / "eval" / "entities" / "gold.json"
READINGS = ROOT / "eval" / "quality_gate" / "readings.json"
CACHE = ROOT / "eval" / "entities" / "linking_cache.json"
OUTCOMES = ("correct", "wrong", "abstained", "missed-mention")

# Of a VIAF cluster record the linker reads these keys only; the rest (often
# hundreds of kB) is dropped from the cache.
_VIAF_KEYS = ("viafID", "nameType", "mainHeadings", "titles")


def request_key(url: str, params: dict[str, Any] | None) -> str:
    """One string per request: the URL and its sorted non-empty parameters."""
    items = sorted((str(k), str(v)) for k, v in (params or {}).items() if v is not None and str(v) != "")
    return f"{url}?{urllib.parse.urlencode(items)}" if items else url


# Of a wbgetentities answer wikidata_client reads labels, descriptions, aliases
# and these claims' main values; references and qualifiers are dropped.
_WIKIDATA_CLAIMS = ("P31", "P214", "P1566", "P625", "P17", "P131")


def _trim_wikidata(payload: Any) -> Any:
    if not isinstance(payload, dict) or not isinstance(payload.get("entities"), dict):
        return payload
    entities = {}
    for qid, entity in payload["entities"].items():
        if not isinstance(entity, dict):
            entities[qid] = entity
            continue
        claims = entity.get("claims") or {}
        entities[qid] = {
            **{key: entity[key] for key in ("id", "type", "labels", "descriptions", "aliases") if key in entity},
            "claims": {
                prop: [{"mainsnak": claim.get("mainsnak"), "rank": claim.get("rank")} for claim in claims.get(prop, [])]
                for prop in _WIKIDATA_CLAIMS
                if prop in claims
            },
        }
    return {**payload, "entities": entities}


def _trim_viaf(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    for wrapper, cluster in payload.items():
        if wrapper.endswith("VIAFCluster") and isinstance(cluster, dict):
            kept = {k: v for k, v in cluster.items() if k.split(":", 1)[-1] in _VIAF_KEYS}
            # Headings and titles each carry their sources' ids, which nothing reads.
            for key, value in kept.items():
                name = key.split(":", 1)[-1]
                if name in ("mainHeadings", "titles") and isinstance(value, dict):
                    kept[key] = {
                        inner: [
                            {k: v for k, v in item.items() if k.split(":", 1)[-1] in ("text", "title")}
                            if isinstance(item, dict) else item
                            for item in (items if isinstance(items, list) else [items])
                        ]
                        for inner, items in value.items()
                    }
            return {wrapper: kept}
    return payload


class Source:
    """Answers the linker's HTTP calls: live (recording them) or from a cache."""

    def __init__(self, cache: dict[str, Any], *, live: bool, user_agent: str, min_interval: float = 1.0) -> None:
        self.cache = cache
        self.live = live
        self.user_agent = user_agent
        self.min_interval = min_interval
        self.requests = 0
        self.misses: list[str] = []
        self._last = 0.0

    def fetch(self, url: str, params: dict[str, Any] | None, headers: dict[str, str] | None = None) -> Any:
        key = request_key(url, params)
        if key in self.cache:
            return self.cache[key]
        if not self.live:
            self.misses.append(key)
            return None
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        self.requests += 1
        request = urllib.request.Request(key, headers={"User-Agent": self.user_agent, **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload: Any = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                raise SystemExit(f"stopped: HTTP {exc.code} for {key} (the service refused; not retrying)") from exc
            payload = None
        except Exception:  # noqa: BLE001 - a failed request is recorded as failed
            payload = None
        if "viaf.org/viaf/" in url and "AutoSuggest" not in url:
            payload = _trim_viaf(payload)
        elif "wbgetentities" in key:
            payload = _trim_wikidata(payload)
        self.cache[key] = payload
        return payload


def install(source: Source) -> Callable[[], None]:
    """Route wikidata_client and authority_sources through *source*; returns the undo."""
    from app.services import authority_sources, wikidata_client

    real = (wikidata_client._http_get, authority_sources._http_json, wikidata_client._rate_limit)

    def http_get(url: str, params: dict[str, str]) -> dict[str, Any]:
        payload = source.fetch(url, params)
        if payload is None:
            raise wikidata_client.WikidataUnavailable(f"no answer for {url}")
        return payload

    def http_json(url: str, params: dict[str, Any], *, timeout: int = 10, headers: dict[str, str] | None = None) -> dict[str, Any]:
        payload = source.fetch(url, params, headers)
        return payload if isinstance(payload, dict) else {}

    wikidata_client._http_get = http_get
    authority_sources._http_json = http_json
    wikidata_client._rate_limit = lambda: None  # Source paces live requests itself

    def undo() -> None:
        wikidata_client._http_get, authority_sources._http_json, wikidata_client._rate_limit = real

    return undo


def link_page(page_id: str, text: str, workdir: Path) -> list[dict[str, Any]]:
    """Extract and link the mentions of *text* in a throwaway database; the mention links."""
    os.environ["ARCHAI_DB_PATH"] = str(workdir / "archai.sqlite")
    os.environ["ARCHAI_WIKIDATA_CACHE"] = str(workdir / "authority_cache.sqlite")
    from app.config import settings
    from app.db import pipeline_db
    from app.services.authority_linking import run_authority_linking
    from app.services.chunking import build_window_chunks
    from app.services.mention_extraction import _extract_mentions_from_text

    from app.services import wikidata_client

    pipeline_db._DB_READY = False
    wikidata_client._CACHE_READY = False  # the new page's authority cache gets its table
    run_id = pipeline_db.create_run(asset_ref=f"eval:{page_id}", asset_sha256=f"eval-{page_id}")
    pipeline_db.update_run_fields(run_id, ocr_text=text, search_allowed=1)
    windows = build_window_chunks(
        text,
        window_lines=int(settings.rag_chunk_window_lines),
        overlap_lines=min(int(settings.rag_chunk_overlap_lines), int(settings.rag_chunk_window_lines) - 1),
    )
    chunks = pipeline_db.insert_chunks(run_id, [{**w, "chunk_id": f"{page_id}:{w['idx']}"} for w in windows])
    mentions, _candidates, _debug = _extract_mentions_from_text(text)
    for mention in mentions:
        start, end = int(mention["start_offset"]), int(mention["end_offset"])
        mention["chunk_id"] = next(
            (c["chunk_id"] for c in chunks if c["start_offset"] <= start and end <= c["end_offset"]),
            next((c["chunk_id"] for c in chunks if c["start_offset"] < end and start < c["end_offset"]), None),
        )
    pipeline_db.insert_entity_mentions(run_id, mentions)
    run_authority_linking(run_id, top_k=5)
    return pipeline_db.list_mention_links_for_run(run_id)


def _span(text: str, entity: dict[str, Any]) -> tuple[int, int]:
    start = -1
    for _ in range(int(entity.get("occurrence", 1))):
        start = text.find(entity["surface"], start + 1)
    if start < 0:
        raise ValueError(f"gold surface {entity['surface']!r} not in the page")
    return start, start + len(entity["surface"])


def score_entity(entity: dict[str, Any], span: tuple[int, int], links: list[dict[str, Any]]) -> dict[str, Any]:
    authority = entity["authority"]
    start, end = span
    overlapping = [link for link in links if int(link["start_offset"]) < end and start < int(link["end_offset"])]
    linked = [link for link in overlapping if str(link.get("link_status") or "").lower() == "linked"]
    row = {"surface": entity["surface"], "type": entity["type"], "status": authority["status"]}
    if not overlapping:
        return {**row, "outcome": "missed-mention", "linked_to": None}
    if not linked:
        return {**row, "outcome": "correct" if authority["status"] == "none" else "abstained", "linked_to": None}
    link = linked[0]
    qid, viaf = str(link.get("wikidata_qid") or ""), str(link.get("viaf_id") or "")
    accepted = (qid and qid in authority["qids"]) or (viaf and viaf in authority["viaf"])
    label = qid or (f"viaf:{viaf}" if viaf else str(link.get("authority_id") or link.get("entity_id") or "?"))
    return {**row, "outcome": "correct" if accepted else "wrong", "linked_to": label}


def run(*, cache_path: Path = CACHE, record: bool = False) -> dict[str, Any]:
    from app.services import wikidata_client

    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    texts = {page["page_id"]: page["reference"] for page in json.loads(READINGS.read_text(encoding="utf-8"))["pages"]}
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() and not record else {}
    source = Source(cache, live=record, user_agent=wikidata_client._USER_AGENT)
    undo = install(source)
    saved_env = {key: os.environ.get(key) for key in ("ARCHAI_DB_PATH", "ARCHAI_WIKIDATA_CACHE")}
    pages: dict[str, list[dict[str, Any]]] = {}
    try:
        for page in gold["pages"]:
            scored = [e for e in page["entities"] if e["authority"]["status"] != "n/a"]
            if not scored:
                continue
            text = texts[page["page_id"]]
            with tempfile.TemporaryDirectory(prefix="archai-linking-") as tmp:
                links = link_page(page["page_id"], text, Path(tmp))
            pages[page["page_id"]] = [score_entity(e, _span(text, e), links) for e in scored]
    finally:
        undo()
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        # The schema flags describe the throwaway files; the restored paths
        # must be initialised afresh on their next use.
        from app.db import pipeline_db

        pipeline_db._DB_READY = False
        wikidata_client._CACHE_READY = False
    if record:
        cache_path.write_text(json.dumps(source.cache, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    totals: dict[str, int] = defaultdict(int)
    for rows in pages.values():
        for row in rows:
            totals[row["outcome"]] += 1
    return {
        "pages": pages,
        "totals": {outcome: totals[outcome] for outcome in OUTCOMES},
        "n": sum(totals.values()),
        "live_requests": source.requests,
        "cache_misses": sorted(set(source.misses)),
    }


def _render(result: dict[str, Any]) -> str:
    lines = []
    for page_id, rows in result["pages"].items():
        counts = defaultdict(int)
        for row in rows:
            counts[row["outcome"]] += 1
        lines.append(f"{page_id} (n={len(rows)}): " + ", ".join(f"{o} {counts[o]}/{len(rows)}" for o in OUTCOMES if counts[o]))
        for row in rows:
            target = f" -> {row['linked_to']}" if row["linked_to"] else ""
            lines.append(f"  {row['surface']!r:32} {row['type']:7} gold={row['status']:8} {row['outcome']}{target}")
    n = result["n"]
    lines.append("total (n=%d): " % n + ", ".join(f"{o} {result['totals'][o]}/{n}" for o in OUTCOMES))
    if result["cache_misses"]:
        lines.append(f"cache misses ({len(result['cache_misses'])}, answered as failed requests): rerun with --record")
    if result["live_requests"]:
        lines.append(f"live requests: {result['live_requests']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--record", action="store_true", help="make the live requests and rewrite the cache")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run(cache_path=args.cache, record=args.record)
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else _render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
