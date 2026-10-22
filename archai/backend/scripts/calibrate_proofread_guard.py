"""Measure the proofreader guard on corrections and inventions of known kind.

Usage::

    python scripts/calibrate_proofread_guard.py
    python scripts/calibrate_proofread_guard.py --json

No proofreader output with known truth exists, so the cases are built from
``eval/quality_gate/readings.json``. Each recogniser's lines are aligned to the
reference lines of the same page, and each reading line is "proofread" into:

* ``correction`` - its reference line: what a perfect proofreader would write;
* ``invented word`` - the reference line with one word of four letters or more
  replaced by a word of similar length from another page;
* ``swapped line`` - a reference line of similar length from another page:
  fluent, in a real language, and not what the page says;
* ``gap filled`` - the reading with one word turned into "[…]", "proofread"
  into the reading as it was: an uncertain reading guessed at.

``judge_line`` should keep the first and revert the rest. The page-level test
then invents a share of a page's lines and asks whether the text that survives
contains any invented line, for this guard and for ``check_proofread_delta``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.ocr_proofreader_agent import check_proofread_delta  # noqa: E402
from app.services.proofread_guard import _align_lines, guard_proofread, judge_line  # noqa: E402

READINGS = Path(__file__).resolve().parents[1] / "eval" / "quality_gate" / "readings.json"
RECOGNISERS = ("kraken_catmus", "kraken_cremma_medieval", "kraken_mccatmus")


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def line_cases(pages: list[dict[str, Any]], seed: int = 1) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    words = {p["page_id"]: re.findall(r"\w{4,}", p["reference"]) for p in pages}
    others = {p["page_id"]: [(q["page_id"], line) for q in pages if q["page_id"] != p["page_id"] for line in _lines(q["reference"])] for p in pages}
    cases = []
    for page in pages:
        foreign_words = [w for pid, ws in words.items() if pid != page["page_id"] for w in ws]
        for recogniser in RECOGNISERS:
            for raw_group, ref_group in _align_lines(_lines(page["readings"][recogniser]), _lines(page["reference"])):
                if len(raw_group) != 1 or len(ref_group) != 1:
                    continue
                raw, ref = raw_group[0], ref_group[0]
                cases.append({"recogniser": recogniser, "kind": "correction", "raw": raw, "proof": ref})
                tokens = ref.split()
                long_ones = [k for k, t in enumerate(tokens) if len(re.sub(r"\W", "", t)) >= 4]
                if long_ones:
                    k = rng.choice(long_ones)
                    similar = [w for w in foreign_words if abs(len(w) - len(tokens[k])) <= 2] or foreign_words
                    invented = " ".join([*tokens[:k], rng.choice(similar), *tokens[k + 1 :]])
                    cases.append({"recogniser": recogniser, "kind": "invented word", "raw": raw, "proof": invented})
                similar_lines = [line for _pid, line in others[page["page_id"]] if abs(len(line) - len(ref)) <= max(4, len(ref) // 4)]
                if similar_lines:
                    cases.append({"recogniser": recogniser, "kind": "swapped line", "raw": raw, "proof": rng.choice(similar_lines)})
                raw_tokens = raw.split()
                if len(raw_tokens) >= 3:
                    k = rng.randrange(len(raw_tokens))
                    gapped = " ".join([*raw_tokens[:k], "[…]", *raw_tokens[k + 1 :]])
                    cases.append({"recogniser": recogniser, "kind": "gap filled", "raw": gapped, "proof": raw})
    return cases


def page_cases(pages: list[dict[str, Any]], seed: int = 2) -> list[dict[str, Any]]:
    """Proofread = reference with a share of its lines replaced by lines of other pages."""
    rng = random.Random(seed)
    out = []
    for page in pages:
        ref = _lines(page["reference"])
        foreign = [line for q in pages if q["page_id"] != page["page_id"] for line in _lines(q["reference"])]
        for share in (0.0, 0.1, 0.25, 0.5):
            proof = list(ref)
            invented = set(rng.sample(range(len(ref)), round(share * len(ref))))
            for k in invented:
                proof[k] = rng.choice(foreign)
            for recogniser in RECOGNISERS:
                out.append({"page": page["page_id"], "recogniser": recogniser, "share": share, "raw": page["readings"][recogniser], "proof": "\n".join(proof), "invented": [proof[k] for k in invented]})
    return out


def run() -> dict[str, Any]:
    pages = json.loads(READINGS.read_text(encoding="utf-8"))["pages"]
    rows: dict[str, dict[str, list[int]]] = {}
    for case in line_cases(pages):
        kept = judge_line(case["raw"], case["proof"]) is None
        rows.setdefault(case["kind"], {}).setdefault(case["recogniser"], [0, 0])
        rows[case["kind"]][case["recogniser"]][0] += kept
        rows[case["kind"]][case["recogniser"]][1] += 1
    pages_out: dict[str, dict[str, list[int]]] = {}
    for case in page_cases(pages):
        survived_new = any(line in guard_proofread(case["raw"], case["proof"]).text.splitlines() for line in case["invented"])
        old_accepts = check_proofread_delta(case["raw"], case["proof"]).accepted
        survived_old = bool(case["invented"]) and old_accepts
        key = f"{case['share']:.0%} of lines invented"
        bucket = pages_out.setdefault(key, {"line guard": [0, 0], "check_proofread_delta": [0, 0]})
        bucket["line guard"][0] += survived_new
        bucket["line guard"][1] += 1
        bucket["check_proofread_delta"][0] += survived_old
        bucket["check_proofread_delta"][1] += 1
    return {"lines": rows, "pages": pages_out}


def _render(result: dict[str, Any]) -> str:
    lines = ["Lines kept by judge_line (corrections should be kept, the rest reverted):"]
    for kind, by in result["lines"].items():
        cells = "  ".join(f"{rec.split('_', 1)[1]:16s} {k / n:6.1%} of {n:3d}" for rec, (k, n) in by.items())
        lines.append(f"  {kind:14s} {cells}")
    lines.append("")
    lines.append("Pages where an invented line survives (15 pages per row: 5 pages x 3 recognisers):")
    for key, by in result["pages"].items():
        lines.append(f"  {key:24s} " + "  ".join(f"{name}: {s}/{n}" for name, (s, n) in by.items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run()
    print(json.dumps(result, indent=2) if args.json else _render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
