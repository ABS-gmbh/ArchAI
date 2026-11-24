"""Score name extraction and authority linking on the held-out name set (B-dev, B-test).

Usage::

    python scripts/benchmark_names_heldout.py --split dev                 # replay the recorded authority responses
    python scripts/benchmark_names_heldout.py --split all --record        # ask Wikidata and VIAF, and save the answers
    python scripts/benchmark_names_heldout.py --split dev --json --out rows.json

``eval/entities/heldout_pages.json`` holds pseudo-pages of CATMuS Medieval test
manuscripts, ``eval/entities/heldout_gold.json`` their gold mentions and
authority ids (``eval/entities/README.md``, "Held-out name set"). Each page's
reference text goes, in a throwaway database, through the pipeline's mention
extractor and ``run_authority_linking``, as in ``benchmark_linking.py``, and is
scored against the gold:

- mentions: a gold mention is **found** when an extracted mention of the gold
  types overlaps it, **typed** when that mention also has its type. Extracted
  mentions of those types that overlap no gold mention are **spurious**. Gold
  mentions marked ``unsure`` are counted apart and never make an overlapping
  mention spurious.
- links, per gold mention with authority status ``linkable`` or ``none``:
  correct, wrong, abstained or missed, as in ``benchmark_linking.score_entity``.
  ``uncertain`` mentions are scored as found or missed only.

Intervals (95%) are resampled over manuscripts, not mentions. Authority
requests replay ``eval/entities/heldout_linking_cache.json.gz`` (gzip; a plain
``.json`` path is read as is); ``--record`` makes them live (at most one a
second) and rewrites it.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PAGES = ROOT / "eval" / "entities" / "heldout_pages.json"
GOLD = ROOT / "eval" / "entities" / "heldout_gold.json"
CACHE = ROOT / "eval" / "entities" / "heldout_linking_cache.json.gz"
TYPES = ("person", "place", "work")


def _read_cache(path: Path) -> dict[str, Any]:
    """The recorded responses; a .gz file is read through gzip."""
    if not path.is_file():
        return {}
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(raw.decode("utf-8"))


def _write_cache(path: Path, cache: dict[str, Any]) -> None:
    text = json.dumps(cache, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    data = text.encode("utf-8")
    path.write_bytes(gzip.compress(data, compresslevel=9, mtime=0) if path.suffix == ".gz" else data)


def _span(text: str, entity: dict[str, Any]) -> tuple[int, int]:
    start = -1
    for _ in range(int(entity.get("occurrence", 1))):
        start = text.find(entity["surface"], start + 1)
    if start < 0:
        raise ValueError(f"gold surface {entity['surface']!r} not in the page")
    return start, start + len(entity["surface"])


def extract_and_link(page_id: str, text: str, workdir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The extracted mentions and their links for *text*, in a throwaway database."""
    os.environ["ARCHAI_DB_PATH"] = str(workdir / "archai.sqlite")
    os.environ["ARCHAI_WIKIDATA_CACHE"] = str(workdir / "authority_cache.sqlite")
    from app.config import settings
    from app.db import pipeline_db
    from app.services import wikidata_client
    from app.services.authority_linking import run_authority_linking
    from app.services.chunking import build_window_chunks
    from app.services.mention_extraction import _extract_mentions_from_text

    pipeline_db._DB_READY = False
    wikidata_client._CACHE_READY = False
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
    return pipeline_db.list_entity_mentions(run_id), pipeline_db.list_mention_links_for_run(run_id)


def score_page(text: str, gold: list[dict[str, Any]], mentions: list[dict[str, Any]], links: list[dict[str, Any]]) -> dict[str, Any]:
    import benchmark_linking as bl

    scored = [m for m in mentions if str(m.get("ent_type")) in TYPES]
    spans = [_span(text, e) for e in gold]
    rows = []
    for entity, (start, end) in zip(gold, spans):
        hits = [m for m in scored if int(m["start_offset"]) < end and start < int(m["end_offset"])]
        row = {
            "surface": entity["surface"], "type": entity["type"], "unsure": bool(entity.get("unsure")),
            "status": entity["authority"]["status"], "found": bool(hits),
            "typed": any(str(m.get("ent_type")) == entity["type"] for m in hits),
            "mention": hits[0]["surface"] if hits else None,
        }
        if entity["authority"]["status"] in ("linkable", "none"):
            row.update({k: v for k, v in bl.score_entity(entity, (start, end), links).items() if k in ("outcome", "linked_to")})
        rows.append(row)
    spurious = [
        {"surface": m["surface"], "type": m.get("ent_type"), "method": m.get("method")}
        for m in scored
        if not any(int(m["start_offset"]) < e and s < int(m["end_offset"]) for s, e in spans)
    ]
    return {"entities": rows, "spurious": spurious, "extracted": len(scored)}


def _boot(units: list[tuple[int, int]], *, seed: int = 11, n: int = 1000) -> tuple[float, float] | None:
    """95% interval of sum(a)/sum(b) over manuscripts (units are per-manuscript (a, b))."""
    units = [u for u in units if u[1]]
    if len(units) < 2:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(n):
        pick = [rng.choice(units) for _ in units]
        total = sum(b for _, b in pick)
        values.append(sum(a for a, _ in pick) / total if total else 0.0)
    values.sort()
    return values[int(0.025 * n)], values[int(0.975 * n) - 1]


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    def table(rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_ms: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_ms[r["shelfmark"]].append(r)
        def per_ms(fn):
            return [fn(rs) for rs in by_ms.values()]
        sure = lambda rs: [e for r in rs for e in r["score"]["entities"] if not e["unsure"]]  # noqa: E731
        found = per_ms(lambda rs: (sum(e["found"] for e in sure(rs)), len(sure(rs))))
        spurious = sum(len(r["score"]["spurious"]) for r in rows)
        extracted = sum(r["score"]["extracted"] for r in rows)
        precision = per_ms(lambda rs: (sum(r["score"]["extracted"] - len(r["score"]["spurious"]) for r in rs), sum(r["score"]["extracted"] for r in rs)))
        linkrows = [e for e in sure(rows) if "outcome" in e]
        outcomes = Counter(e["outcome"] for e in linkrows)
        correct = per_ms(lambda rs: (sum(e.get("outcome") == "correct" for e in sure(rs) if "outcome" in e), sum(1 for e in sure(rs) if "outcome" in e)))
        gold_sure = sum(b for _, b in found)
        return {
            "pages": len(rows), "manuscripts": len(by_ms), "gold": gold_sure,
            "gold_unsure": sum(1 for r in rows for e in r["score"]["entities"] if e["unsure"]),
            "found": sum(a for a, _ in found), "found_ci": _boot(found),
            "typed": sum(e["typed"] for e in sure(rows)),
            "extracted": extracted, "spurious": spurious, "precision_ci": _boot(precision),
            "links": {k: outcomes.get(k, 0) for k in ("correct", "wrong", "abstained", "missed-mention")}, "links_n": len(linkrows),
            "correct_ci": _boot(correct),
        }
    out = {"all": table(results)}
    for language in sorted({r["language"] for r in results}):
        out[language] = table([r for r in results if r["language"] == language])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--pages", type=Path, default=PAGES)
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", type=Path, help="write every page's scored rows here")
    args = parser.parse_args(argv)

    import benchmark_linking as bl
    from app.db import pipeline_db
    from app.services import wikidata_client

    pages = {p["page_id"]: p for p in json.loads(args.pages.read_text(encoding="utf-8"))["pages"]}
    gold = [g for g in json.loads(args.gold.read_text(encoding="utf-8"))["pages"] if args.split == "all" or g["split"] == args.split]
    cache = _read_cache(args.cache)
    source = bl.Source(cache, live=args.record, user_agent=wikidata_client._USER_AGENT)
    undo = bl.install(source)
    saved_env = {key: os.environ.get(key) for key in ("ARCHAI_DB_PATH", "ARCHAI_WIKIDATA_CACHE")}
    results = []
    try:
        for g in gold:
            page = pages[g["page_id"]]
            with tempfile.TemporaryDirectory(prefix="archai-names-") as tmp:
                mentions, links = extract_and_link(g["page_id"], page["reference"], Path(tmp))
            results.append({"page_id": g["page_id"], "split": g["split"], "language": page["language"], "shelfmark": page["shelfmark"],
                            "score": score_page(page["reference"], g["entities"], mentions, links)})
    finally:
        undo()
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        pipeline_db._DB_READY = False
        wikidata_client._CACHE_READY = False
    if args.record:
        _write_cache(args.cache, source.cache)
    summary = summarise(results)
    summary["requests"] = {"live": source.requests, "cache_misses": len(set(source.misses))}
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0
    for language, t in summary.items():
        if language == "requests":
            continue
        ci = lambda c: f"[{c[0]:.2f}, {c[1]:.2f}]" if c else "[-]"  # noqa: E731
        print(
            f"{language:12s} pages {t['pages']:2d} ms {t['manuscripts']:2d} | gold {t['gold']:3d} (+{t['gold_unsure']} unsure) found {t['found']:3d} "
            f"{ci(t['found_ci'])} typed {t['typed']:3d} | extracted {t['extracted']:3d} spurious {t['spurious']:3d} precision {ci(t['precision_ci'])} | "
            f"links n={t['links_n']:3d} " + " ".join(f"{k} {v}" for k, v in t["links"].items()) + f" correct {ci(t['correct_ci'])}"
        )
    print(f"authority requests: live {source.requests}, cache misses {len(set(source.misses))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
