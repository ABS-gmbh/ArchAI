"""Score entity-mention extraction against hand-annotated reference pages.

Usage::

    python scripts/benchmark_entities.py
    python scripts/benchmark_entities.py --json

``eval/entities/gold.json`` marks the named entities - people, places, works,
dates - in the five reference transcriptions of
``eval/quality_gate/readings.json``. Each page is run through the full-page
pipeline's mention extractor (``_extract_mentions_from_text``) and every
mention of those four types is scored. A mention counts as found when it
overlaps an unmatched gold span of the same type, so "Roys mͣrcadigas" finds
"mͣrcadigas"; anything else is a false positive. Titles and roles are not
scored.

The extractor's anchors were written for one page, ``middle-french-antitus``
(Lausanne, "pere", "prince", "euesque"), so totals are also reported without
it. Two pages contain no named entity at all: there, every mention is wrong.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "eval" / "entities" / "gold.json"
READINGS = ROOT / "eval" / "quality_gate" / "readings.json"
TUNED_ON = "middle-french-antitus"


def _extractor() -> Any:
    # The OCR router prints its backend set-up while it loads.
    with contextlib.redirect_stdout(io.StringIO()):
        from app.routers.ocr import _extract_mentions_from_text
    return _extract_mentions_from_text


def gold_spans(text: str, entities: list[dict[str, Any]]) -> list[tuple[int, int, str, str]]:
    spans = []
    for entity in entities:
        start = -1
        for _ in range(int(entity.get("occurrence", 1))):
            start = text.find(entity["surface"], start + 1)
        if start < 0:
            raise ValueError(f"gold surface {entity['surface']!r} not in the page")
        spans.append((start, start + len(entity["surface"]), entity["type"], entity["surface"]))
    return spans


def score_page(text: str, entities: list[dict[str, Any]], types: set[str], extract: Any) -> dict[str, Any]:
    gold = gold_spans(text, entities)
    mentions, _candidates, _debug = extract(text)
    predicted = sorted(
        (int(m["start_offset"]), int(m["end_offset"]), str(m["ent_type"]), str(m["surface"]))
        for m in mentions
        if str(m.get("ent_type")) in types
    )
    matched: set[int] = set()
    true_positives, false_positives = [], []
    for start, end, ent_type, surface in predicted:
        hit = next(
            (i for i, (gs, ge, gt, _gsurf) in enumerate(gold) if i not in matched and gt == ent_type and start < ge and gs < end),
            None,
        )
        if hit is None:
            false_positives.append({"surface": surface, "type": ent_type})
        else:
            matched.add(hit)
            true_positives.append({"surface": surface, "type": ent_type, "gold": gold[hit][3]})
    missed = [{"surface": g[3], "type": g[2]} for i, g in enumerate(gold) if i not in matched]
    return {"tp": len(true_positives), "fp": len(false_positives), "fn": len(missed), "found": true_positives, "false_positives": false_positives, "missed": missed}


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def run() -> dict[str, Any]:
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    texts = {page["page_id"]: page["reference"] for page in json.loads(READINGS.read_text(encoding="utf-8"))["pages"]}
    types = set(gold["types"])
    extract = _extractor()
    pages = {page["page_id"]: score_page(texts[page["page_id"]], page["entities"], types, extract) for page in gold["pages"]}

    def total(keep: Any) -> dict[str, Any]:
        tp = sum(p["tp"] for pid, p in pages.items() if keep(pid))
        fp = sum(p["fp"] for pid, p in pages.items() if keep(pid))
        fn = sum(p["fn"] for pid, p in pages.items() if keep(pid))
        return {"tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)}

    return {"pages": pages, "all": total(lambda _pid: True), "without_tuned_page": total(lambda pid: pid != TUNED_ON)}


def _render(result: dict[str, Any]) -> str:
    lines = [f"{'page':24s} {'tp':>3s} {'fp':>3s} {'fn':>3s}  false positives / missed"]
    for pid, page in result["pages"].items():
        fps = ", ".join(f"{m['surface']!r}:{m['type']}" for m in page["false_positives"]) or "-"
        missed = ", ".join(f"{m['surface']!r}:{m['type']}" for m in page["missed"]) or "-"
        lines.append(f"{pid:24s} {page['tp']:3d} {page['fp']:3d} {page['fn']:3d}  fp: {fps} | missed: {missed}")
    for name, label in (("all", "all five pages"), ("without_tuned_page", f"without {TUNED_ON}")):
        t = result[name]
        lines.append(
            f"{label:24s} {t['tp']:3d} {t['fp']:3d} {t['fn']:3d}  precision {t['precision']:.2f}  recall {t['recall']:.2f}  F1 {t['f1']:.2f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else _render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
