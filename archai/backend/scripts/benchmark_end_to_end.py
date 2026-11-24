"""Score the whole pipeline, from page image to linked names, on the reference pages.

Usage::

    python scripts/benchmark_end_to_end.py --images DIR --out-dir OUT --record   # live authority requests, saved
    python scripts/benchmark_end_to_end.py --images DIR --out-dir OUT --cache OUT/authority_responses.json
    python scripts/benchmark_end_to_end.py --reference-only --out-dir OUT --cache FILE

Each page of ``eval/recognizers/manifest.jsonl`` (matched by SHA-256 in
``--images``) is read twice, each time in a throwaway database, vector store,
authority cache and task directory under ``--out-dir``, never ``app/.data``:

- ``ocr``: ``ocr_router._extract_full_page``, the full-page route's own
  function: the segmented engine (GLM-OCR fallback if it fails), then
  ``_run_full_page_post_ocr_pipeline`` with the quality gate, mention
  extraction, authority linking and indexing.
- ``reference``: the same post-OCR pipeline on the page's reference
  transcription, so OCR errors are separated from extraction and linking ones.

Scored against ``eval/entities/gold.json``, per page and overall:

- ``cer``: of the OCR text against the reference (diplomatic policy), and the
  gate's verdict (``quality_label``, ``downstream_mode``, ``search_allowed``).
- mentions of the gold types (person, place, work, date), one-to-one, each gold
  entity in page order taking the first unmatched mention that matches it:
  - ``exact``: the same surface after NFC, whitespace collapsing and casefolding;
  - ``fuzzy``: the same on the search key (``build_search_key``: abbreviations
    expanded, u/v and i/j folded), one key containing the other (4 or more
    characters), or a character error rate of 1/3 or less between the keys.
  ``spurious`` are mentions of those types that match no gold entity. Types are
  not required to agree; ``type_agrees`` counts the fuzzy matches that do.
- links, per gold entity with an authority status (dates are not linked),
  through its fuzzy-matched mention, as in ``benchmark_linking.py``:
  ``correct`` (an accepted Wikidata QID or VIAF id; for an entity with no
  record, left unlinked), ``wrong``, ``abstained`` (a mention, but no link,
  where a record exists) or ``missed`` (no matching mention). Each link names
  the authority that answered.

Authority requests (Wikidata, VIAF) go through ``benchmark_linking.Source``:
live with ``--record`` (at most one a second, stopping on a 429 or 403), saved
to ``OUT/authority_responses.json``; replayed from ``--cache`` otherwise.
GeoNames needs a username (``GEONAMES_USERNAME``) and is not asked unless one
is set. Embeddings use ChromaDB's local model, so nothing calls the chat or
embedding provider.

Five pages and one annotator: the counts are reported as k/n, without a rate.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

GOLD = ROOT / "eval" / "entities" / "gold.json"
SCORED_TYPES = ("person", "place", "work", "date")
LINK_OUTCOMES = ("correct", "wrong", "abstained", "missed")


# ── matching ────────────────────────────────────────────────────────────


def exact_form(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip().casefold()


def key_form(text: str) -> str:
    from app.services.medieval_text import build_search_key

    return re.sub(r"\s+", " ", build_search_key(re.sub(r"\s+", " ", text or ""))).strip()


def fuzzy_match(gold: str, mention: str) -> bool:
    from app.services.evaluation import cer

    a, b = key_form(gold), key_form(mention)
    if not a or not b:
        return False
    if a == b:
        return True
    if (len(a) >= 4 and a in b) or (len(b) >= 4 and b in a):
        return True
    return cer(a, b).rate <= 1 / 3


def match(gold: list[dict[str, Any]], mentions: list[dict[str, Any]], rule) -> dict[int, int]:
    """Gold index -> mention index, one-to-one, gold in page order taking the first unmatched match."""
    taken: set[int] = set()
    out: dict[int, int] = {}
    for gi, entity in enumerate(gold):
        for mi, mention in enumerate(mentions):
            if mi not in taken and rule(entity["surface"], str(mention.get("surface") or "")):
                out[gi] = mi
                taken.add(mi)
                break
    return out


def score_page(gold: list[dict[str, Any]], mentions: list[dict[str, Any]], links: list[dict[str, Any]]) -> dict[str, Any]:
    scored = sorted(
        (m for m in mentions if str(m.get("ent_type")) in SCORED_TYPES), key=lambda m: int(m.get("start_offset") or 0)
    )
    exact = match(gold, scored, lambda g, m: exact_form(g) == exact_form(m))
    fuzzy = match(gold, scored, fuzzy_match)
    links_by_mention: dict[Any, list[dict[str, Any]]] = {}
    for link in links:
        links_by_mention.setdefault(link.get("mention_id"), []).append(link)

    entities = []
    for gi, entity in enumerate(gold):
        authority = entity.get("authority") or {}
        row: dict[str, Any] = {
            "surface": entity["surface"],
            "type": entity["type"],
            "status": authority.get("status"),
            "exact": gi in exact,
            "fuzzy": gi in fuzzy,
            "mention": None,
            "link": None,
        }
        mention = scored[fuzzy[gi]] if gi in fuzzy else None
        if mention is not None:
            row["mention"] = {"surface": mention.get("surface"), "type": mention.get("ent_type"), "method": mention.get("method")}
            row["type_agrees"] = str(mention.get("ent_type")) == entity["type"]
        if authority.get("status") in (None, "n/a"):
            entities.append(row)
            continue
        if mention is None:
            row["link"] = "missed"
        else:
            linked = [
                link
                for link in links_by_mention.get(mention.get("mention_id"), [])
                if str(link.get("link_status") or "").lower() == "linked"
            ]
            if not linked:
                row["link"] = "correct" if authority["status"] == "none" else "abstained"
            else:
                link = linked[0]
                qid, viaf = str(link.get("wikidata_qid") or ""), str(link.get("viaf_id") or "")
                accepted = bool((qid and qid in authority.get("qids", [])) or (viaf and viaf in authority.get("viaf", [])))
                row["link"] = "correct" if accepted else "wrong"
                row["linked_to"] = {
                    "authority_source": link.get("authority_source"),
                    "authority_id": link.get("authority_id"),
                    "wikidata_qid": qid or None,
                    "viaf_id": viaf or None,
                    "geonames_id": link.get("geonames_id"),
                    "label": link.get("canonical_label"),
                    "confidence": link.get("confidence"),
                }
        entities.append(row)

    matched = set(fuzzy.values())
    spurious = [
        {"surface": m.get("surface"), "type": m.get("ent_type"), "method": m.get("method")}
        for mi, m in enumerate(scored)
        if mi not in matched
    ]
    return {
        "gold": len(gold),
        "mentions_scored_types": len(scored),
        "mentions_all_types": len(mentions),
        "exact": len(exact),
        "fuzzy": len(fuzzy),
        "type_agrees": sum(1 for e in entities if e.get("type_agrees")),
        "spurious": spurious,
        "links": dict(Counter(e["link"] for e in entities if e["link"])),
        "entities": entities,
    }


# ── pipeline ────────────────────────────────────────────────────────────


def isolate(workdir: Path) -> None:
    """Point every store the pipeline writes at *workdir*."""
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ["ARCHAI_DB_PATH"] = str(workdir / "archai.sqlite")
    os.environ["ARCHAI_WIKIDATA_CACHE"] = str(workdir / "authority_cache.sqlite")
    from app.config import settings
    from app.db import pipeline_db
    from app.services import rag_store, wikidata_client

    settings.chroma_persist_dir = str(workdir / "chroma")
    settings.task_base_dir = str(workdir / "tasks")
    settings.ocr_evidence_path = str(workdir / "ocr_evidence.jsonl")
    settings.rag_embedding_model = ""  # ChromaDB's local model: no provider call
    settings.enable_test_ocr_fixtures = False
    pipeline_db._DB_READY = False
    wikidata_client._CACHE_READY = False
    rag_store._CLIENT = None


def run_record(run_id: str) -> dict[str, Any]:
    from app.db import pipeline_db

    run = pipeline_db.get_run(run_id) or {}
    events = [
        {"stage": e.get("stage"), "level": e.get("level"), "message": e.get("message")}
        for e in (pipeline_db.list_events(run_id) if hasattr(pipeline_db, "list_events") else [])
    ]
    return {
        "status": run.get("status"),
        "search_allowed": run.get("search_allowed"),
        "mentions": pipeline_db.list_entity_mentions(run_id),
        "links": pipeline_db.list_mention_links_for_run(run_id),
        "events": events,
    }


def read_ocr(page: Any) -> tuple[dict[str, Any], str]:
    from app.routers import ocr as ocr_router
    from app.schemas.agents_ocr import SaiaFullPageExtractRequest

    data = page.image.read_bytes()
    payload = SaiaFullPageExtractRequest(
        page_id=page.page_id,
        image_b64=base64.b64encode(data).decode("ascii"),
        language_hint=page.language_hint,
        apply_proofread=False,
        ocr_backend="auto",
    )
    response = asyncio.run(ocr_router._extract_full_page(payload, data, None))
    return response.model_dump(exclude={"line_regions", "comparison_runs"}), str(response.text or "")


def read_reference(page: Any, image_bytes: bytes) -> dict[str, Any]:
    from app.routers import ocr as ocr_router
    from app.schemas.agents_ocr import SaiaFullPageExtractRequest, SaiaFullPageExtractResponse
    from app.services import trace_pipeline

    text = page.reference.strip()
    detected, confidence = ocr_router._detect_language_metadata(text)
    if detected == "unknown" and page.language_hint:
        detected = ocr_router._normalize_detected_language(str(page.language_hint))
    payload = SaiaFullPageExtractRequest(
        page_id=f"{page.page_id}:reference",
        image_b64=base64.b64encode(image_bytes).decode("ascii"),
        language_hint=page.language_hint,
        apply_proofread=False,
    )
    response = SaiaFullPageExtractResponse(
        status="FULL",
        model_used="reference transcription",
        detected_language=ocr_router._normalize_detected_language(detected),
        language_confidence=confidence,
        lines=[line for line in text.splitlines() if line.strip()],
        text=text,
    )
    fields = trace_pipeline._run_full_page_post_ocr_pipeline(payload, response, image_bytes, ocr_backend="reference")
    return {**response.model_dump(exclude={"line_regions", "comparison_runs"}), **fields}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--images", type=Path, help="directory holding the page images (matched by SHA-256)")
    parser.add_argument("--manifest", type=Path, default=ROOT / "eval" / "recognizers" / "manifest.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True, help="run outputs and the throwaway stores")
    parser.add_argument("--cache", type=Path, help="replay authority responses from this file")
    parser.add_argument("--record", action="store_true", help="make live authority requests and save them")
    parser.add_argument("--reference-only", action="store_true", help="skip OCR (no images needed)")
    args = parser.parse_args(argv)

    import benchmark_linking as bl
    import benchmark_recognizers as br
    from app.agents import ocr_agent
    from app.services import wikidata_client
    from app.services.evaluation import DIPLOMATIC, cer

    out = args.out_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    gold_pages = {p["page_id"]: p["entities"] for p in json.loads(GOLD.read_text(encoding="utf-8"))["pages"]}
    if args.reference_only:
        pages = []
        for raw in args.manifest.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                record = json.loads(raw)
                reference = (args.manifest.parent / record["reference"]).read_text(encoding="utf-8")
                pages.append(br.Page(record["page_id"], Path("/dev/null"), reference, record.get("language_hint")))
    else:
        if args.images is None:
            parser.error("--images is required unless --reference-only")
        pages = br.load_pages(args.manifest, args.images.expanduser())

    cache = json.loads(args.cache.read_text(encoding="utf-8")) if args.cache and args.cache.is_file() else {}
    source = bl.Source(cache, live=args.record, user_agent=wikidata_client._USER_AGENT)
    used: Counter[str] = Counter()
    fetch = source.fetch

    def counted_fetch(url: str, params: Any, headers: Any = None) -> Any:
        payload = fetch(url, params, headers)
        used[bl.request_key(url, params)] += 1
        return payload

    source.fetch = counted_fetch  # type: ignore[method-assign]
    undo = bl.install(source)
    ocr_agent.write_ocr_evidence_jsonl = lambda _record: None  # a benchmark is not evidence about a document

    modes = ("reference",) if args.reference_only else ("ocr", "reference")
    results: list[dict[str, Any]] = []
    try:
        for page in pages:
            # Without an image the run is keyed by a stand-in; the reference pipeline never decodes it.
            image_bytes = f"reference:{page.page_id}".encode() if args.reference_only else page.image.read_bytes()
            for mode in modes:
                isolate(out / "stores" / f"{page.page_id}-{mode}")
                started = time.time()
                row: dict[str, Any] = {"page_id": page.page_id, "mode": mode}
                try:
                    if mode == "ocr":
                        response, text = read_ocr(page)
                    else:
                        response = read_reference(page, image_bytes)
                        text = page.reference.strip()
                    run_id = response.get("run_id")
                    record = run_record(run_id) if run_id else {"mentions": [], "links": [], "events": []}
                    row.update(
                        {
                            "run_id": run_id,
                            "ocr_engine": response.get("ocr_engine"),
                            "model_used": response.get("model_used"),
                            "lines": len(response.get("lines") or []),
                            "cer": cer(page.reference, text, DIPLOMATIC).rate,
                            "quality_label": response.get("quality_label"),
                            "downstream_mode": response.get("downstream_mode"),
                            "search_allowed": record.get("search_allowed"),
                            "warnings": response.get("warnings"),
                            "score": score_page(gold_pages.get(page.page_id, []), record["mentions"], record["links"]),
                            "events": record["events"],
                            "text": text,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - a page that fails is reported, not fatal
                    row["error"] = f"{type(exc).__name__}: {exc}"
                row["seconds"] = round(time.time() - started, 1)
                results.append(row)
                print(render_row(row), flush=True)
    finally:
        undo()

    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    if args.record:
        (out / "authority_responses.json").write_text(
            json.dumps(source.cache, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
    hosts = Counter(re.sub(r"^https?://([^/?]+).*$", r"\1", key) for key in used)
    failed = sum(1 for key in used if not source.cache.get(key))
    counts = {
        "lookups": sum(used.values()),
        "distinct_requests": len(used),
        "by_host": dict(sorted(hosts.items())),
        "live_requests": source.requests,
        "empty_or_failed": failed,
        "cache_misses": len(set(source.misses)),
    }
    (out / "authority_counts.json").write_text(json.dumps(counts, indent=1) + "\n", encoding="utf-8")
    print(
        f"\nauthority: {counts['lookups']} lookups, {counts['distinct_requests']} distinct requests "
        f"({', '.join(f'{h} {n}' for h, n in counts['by_host'].items())}), live {counts['live_requests']}, "
        f"empty or failed {failed}, cache misses {counts['cache_misses']}"
    )
    print(render_totals(results, modes))
    return 0


# ── report ──────────────────────────────────────────────────────────────


def render_row(row: dict[str, Any]) -> str:
    if row.get("error"):
        return f"{row['page_id']:22s} {row['mode']:9s} ERROR {row['error'][:160]}"
    s = row["score"]
    links = ", ".join(f"{k} {s['links'].get(k, 0)}" for k in LINK_OUTCOMES if s["links"].get(k))
    return (
        f"{row['page_id']:22s} {row['mode']:9s} CER {row['cer']:.3f} {row['quality_label']}/{row['downstream_mode']} "
        f"search={row['search_allowed']} gold {s['gold']} exact {s['exact']} fuzzy {s['fuzzy']} "
        f"spurious {len(s['spurious'])} links: {links or '-'} ({row['seconds']}s)"
    )


def render_totals(results: list[dict[str, Any]], modes: tuple[str, ...]) -> str:
    lines = []
    for mode in modes:
        rows = [r for r in results if r["mode"] == mode and not r.get("error")]
        gold = sum(r["score"]["gold"] for r in rows)
        exact = sum(r["score"]["exact"] for r in rows)
        fuzzy = sum(r["score"]["fuzzy"] for r in rows)
        spurious = sum(len(r["score"]["spurious"]) for r in rows)
        links = Counter()
        for r in rows:
            links.update(r["score"]["links"])
        n_links = sum(links.values())
        lines.append(
            f"{mode}: mentions exact {exact}/{gold}, fuzzy {fuzzy}/{gold}, spurious {spurious}; "
            f"links (n={n_links}) " + ", ".join(f"{k} {links.get(k, 0)}/{n_links}" for k in LINK_OUTCOMES)
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
