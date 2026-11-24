"""Where the linkable names of the held-out set stop on their way to a link: a funnel.

Usage::

    python scripts/benchmark_names_funnel.py                 # B-dev, as shipped
    python scripts/benchmark_names_funnel.py --oracle        # the gold spans given as the mentions
    python scripts/benchmark_names_funnel.py --deeper        # and where a longer or other-language search finds the gold record
    python scripts/benchmark_names_funnel.py --json --out rows.json

Every sure gold name with an authority record (status ``linkable``) is put at
the first stage it does not pass:

1. not extracted: no mention overlaps it;
2. extracted, never searched: a gate before the lookup stopped every
   overlapping mention (``low_evidence_place``, name-likeness, a role);
3. gold record not in the search results, from any source;
4. in the results, but dropped by the Wikidata pre-filter or never enriched
   (the linker enriches at most three Wikidata candidates a mention);
5. enriched, but type-incompatible or under the name-similarity floor;
6. scored, not selected (threshold, margin, the one-name check);
7. linked, or linked to another record.

Of several overlapping mentions the one that got furthest counts. The run is
offline: it replays ``heldout_linking_cache.json.gz`` and, for the requests
only these probes make (the gold spans as mentions, searches asking for 50
results or in other languages), ``heldout_funnel_probes.json.gz``. A request
in neither is reported as a miss, never made. ``eval/entities/README.md``,
"Where linkable names stop", gives the result.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PROBES = ROOT / "eval" / "entities" / "heldout_funnel_probes.json.gz"
STAGES = (
    "1 not extracted",
    "2 extracted, never searched",
    "3 gold record not in the results",
    "4 in the results, not enriched",
    "5 enriched, type-incompatible or name too far",
    "6 scored, not selected",
    "7 linked to another record",
    "7 linked",
)
DEEPER_LANGUAGES = ("fr", "en", "la", "es", "it")


def _is_gold(entity: dict[str, Any], ident: Any) -> bool:
    return bool(ident) and (ident in entity["qids"] or str(ident) in {str(v) for v in entity["viaf"]})


def _mention_stage(entity: dict[str, Any], mention: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not mention.get("queries"):
        return STAGES[1], {"gate": mention.get("final_reason")}
    gold_hits = [h for h in mention.get("hits", []) if _is_gold(entity, h["id"])]
    if not gold_hits:
        return STAGES[2], {"queries": mention["queries"], "hits": len(mention.get("hits", []))}
    info: dict[str, Any] = {"hit": gold_hits[0]}
    scored = [c for c in mention.get("scored", []) if _is_gold(entity, c.get("qid")) or _is_gold(entity, c.get("viaf"))]
    wikidata_only = all(h["source"] == "wikidata" for h in gold_hits)
    if wikidata_only:
        qid = gold_hits[0]["id"]
        kept = mention.get("prefilter_out", [])
        info.update({"pre_sort_rank": kept.index(qid) + 1 if qid in kept else None, "wikidata_candidates": len(kept)})
    if wikidata_only and not scored:
        return STAGES[3], info
    if not scored:
        return STAGES[5], {**info, "note": "gold record not in the scored list"}
    gold = max(scored, key=lambda c: c.get("score") or 0.0)
    info.update({"gold_score": gold.get("score"), "type_ok": gold.get("type_ok"), "name_sim": gold.get("name_sim")})
    if wikidata_only and gold.get("qid") not in {e["qid"] for e in mention.get("enriched", [])} and not gold.get("type_ok"):
        return STAGES[3], info
    if not gold.get("type_ok") or (gold.get("name_sim") or 0.0) < mention.get("min_similarity", 0.60):
        return STAGES[4], info
    ranked = sorted(mention.get("scored", []), key=lambda c: -(c.get("score") or 0.0))
    info.update(
        {
            "top": (ranked[0].get("label"), ranked[0].get("score")),
            "runner_up": (ranked[1].get("label"), ranked[1].get("score")) if len(ranked) > 1 else None,
            "reason": mention.get("final_reason"),
        }
    )
    return STAGES[5], info


def assign_stage(entity: dict[str, Any], mentions: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """The stage a linkable gold name stops at, from its overlapping mentions' records (see the module docstring)."""
    if entity.get("outcome") == "correct":
        return STAGES[7], {}
    if entity.get("outcome") == "wrong":
        return STAGES[6], {"linked_to": entity.get("linked_to")}
    if not mentions:
        return STAGES[0], {}
    return max((_mention_stage(entity, m) for m in mentions), key=lambda item: STAGES.index(item[0]))


def _instrument(record: dict[str, Any]) -> None:
    """Wrap the linker's steps so each mention's queries, hits, enrichment and scores are kept."""
    from app.services import authority_linking as al
    from app.services import entity_scoring as es

    def wrap(name: str, after):
        real = getattr(al, name)

        def wrapper(*args, **kwargs):
            result = real(*args, **kwargs)
            after(result, *args, **kwargs)
            return result

        setattr(al, name, wrapper)

    def queries(result, query, method):
        record["cur"] = {"queries": list(result), "hits": [], "prefilter_in": [], "prefilter_out": [], "enriched": [], "scored": []}

    def hits(result, query, **_kwargs):
        for source, items in result[0].items():
            for rank, item in enumerate(items, 1):
                record["cur"]["hits"].append(
                    {"query": query, "source": source, "rank": rank, "label": item.get("label"),
                     "id": item.get("qid") or item.get("viaf_id") or item.get("authority_id") or item.get("geonames_id")}
                )

    def prefilter(result, candidates, surface, ent_type):
        record["cur"]["prefilter_in"] = [c.get("qid") for c in candidates]
        record["cur"]["prefilter_out"] = [c.get("qid") for c in result]

    def enriched(result, qid):
        record["cur"]["enriched"].append({"qid": qid, "viaf": result.get("viaf_id")})

    def disambiguated(result, scored, **_kwargs):
        record["cur"]["scored"] = [
            {"qid": c.get("qid"), "viaf": c.get("viaf_id"), "source": c.get("source"), "label": c.get("label"),
             "score": c.get("score"), "type_ok": c.get("type_compatible"), "name_sim": round(es._name_similarity(c), 4)}
            for c in result["all"]
        ]

    def breakdown(result, **kwargs):
        mention_id = kwargs["mention"].get("mention_id")
        record["by_mention"][mention_id] = {**(record["cur"] or {}), "final_reason": kwargs.get("reason"), "status": kwargs.get("status")}
        record["cur"] = None

    wrap("_build_query_variants", queries)
    wrap("_search_all_sources_for_query", hits)
    wrap("_prefilter_candidates", prefilter)
    wrap("enrich_wikidata_item", enriched)
    wrap("disambiguate", disambiguated)
    wrap("_score_breakdown_for_mention", breakdown)


def _oracle(entities: list[dict[str, Any]]):
    import benchmark_names_heldout as bn

    def extract(text: str):
        mentions = []
        for entity in entities:
            if entity.get("unsure") or entity["type"] not in bn.TYPES:
                continue
            start, end = bn._span(text, entity)
            mentions.append({"start_offset": start, "end_offset": end, "surface": text[start:end], "norm": text[start:end].lower(),
                             "label": entity["type"].upper(), "ent_type": entity["type"], "confidence": 0.9, "method": "oracle", "notes": None})
        return mentions, [], {}

    return extract


def _cache() -> dict[str, Any]:
    import benchmark_names_heldout as bn

    return {**bn._read_cache(bn.CACHE), **bn._read_cache(PROBES)}


def run(split: str = "dev", *, oracle: bool = False) -> tuple[list[dict[str, Any]], int]:
    """Each sure linkable gold name with its stage and evidence, and the count of cache misses."""
    import benchmark_linking as bl
    import benchmark_names_heldout as bn
    from app.services import mention_extraction, wikidata_client

    record: dict[str, Any] = {"cur": None, "by_mention": {}}
    _instrument(record)
    pages = {p["page_id"]: p for p in json.loads(bn.PAGES.read_text(encoding="utf-8"))["pages"]}
    gold = [g for g in json.loads(bn.GOLD.read_text(encoding="utf-8"))["pages"] if split == "all" or g["split"] == split]
    source = bl.Source(_cache(), live=False, user_agent=wikidata_client._USER_AGENT)
    undo = bl.install(source)
    real_extract = mention_extraction._extract_mentions_from_text
    rows = []
    try:
        for g in gold:
            text = pages[g["page_id"]]["reference"]
            record["by_mention"].clear()
            if oracle:
                mention_extraction._extract_mentions_from_text = _oracle(g["entities"])
            with tempfile.TemporaryDirectory(prefix="archai-funnel-") as tmp:
                mentions, links = bn.extract_and_link(g["page_id"], text, Path(tmp))
            scores = bn.score_page(text, g["entities"], mentions, links)["entities"]
            for entity, score in zip(g["entities"], scores):
                if entity.get("unsure") or entity["authority"]["status"] != "linkable":
                    continue
                start, end = bn._span(text, entity)
                overlapping = [m for m in mentions if int(m["start_offset"]) < end and start < int(m["end_offset"])]
                row = {"page_id": g["page_id"], "language": pages[g["page_id"]]["language"], "surface": entity["surface"],
                       "type": entity["type"], "qids": entity["authority"]["qids"], "viaf": entity["authority"]["viaf"],
                       "outcome": score.get("outcome"), "linked_to": score.get("linked_to"),
                       "mentions": [{"surface": m["surface"], **record["by_mention"].get(m["mention_id"], {})} for m in overlapping]}
                row["stage"], row["evidence"] = assign_stage(row, row["mentions"])
                rows.append(row)
    finally:
        mention_extraction._extract_mentions_from_text = real_extract
        undo()
    return rows, len(set(source.misses))


def deeper(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For names whose gold record was not in the results: its best rank in a 50-result Wikidata search in five languages, and in VIAF's full AutoSuggest answer."""
    import benchmark_linking as bl
    from app.services import wikidata_client

    cache = _cache()
    os.environ["ARCHAI_WIKIDATA_CACHE"] = str(Path(tempfile.mkdtemp(prefix="archai-funnel-")) / "wd.sqlite")
    wikidata_client._CACHE_READY = False
    source = bl.Source(cache, live=False, user_agent=wikidata_client._USER_AGENT)
    undo = bl.install(source)
    out = []
    try:
        for row in rows:
            if row["stage"] != STAGES[2]:
                continue
            best: dict[str, Any] = {"wikidata": None, "viaf": None}
            misses_before = len(source.misses)
            for query in {q for m in row["mentions"] for q in m.get("queries") or []}:
                for language in DEEPER_LANGUAGES:
                    for rank, hit in enumerate(wikidata_client.search_wikidata(query, k=50, force_refresh=True, language=language), 1):
                        if _is_gold(row, hit.get("qid")) and (best["wikidata"] is None or rank < best["wikidata"]["rank"]):
                            best["wikidata"] = {"rank": rank, "query": query, "language": language}
                key = next((k for k in cache if "AutoSuggest" in k and urllib.parse.unquote_plus(k.split("query=")[-1]) == query), None)
                for rank, hit in enumerate(((cache.get(key) or {}).get("result") or []) if key else [], 1):
                    if _is_gold(row, hit.get("viafid")) and (best["viaf"] is None or rank < best["viaf"]["rank"]):
                        best["viaf"] = {"rank": rank, "query": query}
            # Only the names as shipped were probed; any other query is a miss, not a "not found".
            out.append({"surface": row["surface"], "page_id": row["page_id"], "probed": len(source.misses) == misses_before, **best})
    finally:
        undo()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--oracle", action="store_true", help="give the gold spans as the mentions")
    parser.add_argument("--deeper", action="store_true", help="also probe longer and other-language searches")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    logging.disable(logging.WARNING)
    rows, misses = run(args.split, oracle=args.oracle)
    result: dict[str, Any] = {"stages": dict(Counter(r["stage"] for r in rows)), "names": len(rows), "cache_misses": misses}
    if args.deeper:
        result["deeper"] = deeper(rows)
    if args.out:
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1, default=str))
        return 0
    for stage in STAGES:
        print(f"{stage:48s} {result['stages'].get(stage, 0)}")
    print(f"linkable names {len(rows)}, cache misses {misses}")
    for row in rows:
        if row["stage"] in (STAGES[0], STAGES[7]):
            continue
        print(f"  [{row['stage']}] {row['surface']!r} ({row['language']}) {json.dumps(row['evidence'], ensure_ascii=False, default=str)[:240]}")
    for probe in result.get("deeper", []):
        if probe["probed"]:
            print(f"  deeper {probe['surface']!r}: wikidata {probe['wikidata']}, viaf {probe['viaf']}")
        else:
            print(f"  deeper {probe['surface']!r}: not probed (no recorded response)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
