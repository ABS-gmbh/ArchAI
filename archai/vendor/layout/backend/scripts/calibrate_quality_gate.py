"""Check the OCR quality gate against transcriptions of known quality.

Usage::

    python scripts/calibrate_quality_gate.py
    python scripts/calibrate_quality_gate.py --db app/archai.sqlite

The gate decides whether a transcription may feed entity extraction, authority
linking, the search index and translation. Its verdicts are checked on:

* ``eval/quality_gate/readings.json``: five reference transcriptions (CER 0),
  each recogniser's reading of the same page, and synthetic corruptions of
  each reference - characters substituted, inserted or deleted at a set rate,
  every word reversed, every word replaced by random letters - with the
  character error rate of each against its reference;
* with ``--db``, every distinct transcription of 300 characters or more in a
  pipeline database, held out, and three corruptions of each: noise at 50%,
  reversed words, random letters. There is no reference there, so only the
  contrast and the verdicts are reported.

A verdict is BLOCKED when entity extraction and search are not allowed.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import statistics
import sys
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.routers.ocr import _detect_language_metadata, _normalize_detected_language, _quality_language  # noqa: E402
from app.services.lexicon_trust import best_fit_plausibility  # noqa: E402
from app.services.ocr_quality import compute_quality_report  # noqa: E402
from app.services.pipeline_hardening import enforce_quality_gates  # noqa: E402

DEFAULT_READINGS = Path(__file__).resolve().parents[1] / "eval" / "quality_gate" / "readings.json"
_LETTERS = "abcdefghilmnopqrstuvxyz"


@dataclass(frozen=True)
class Verdict:
    language: str
    label: str
    contrast: float
    allowed: bool


def verdict(text: str) -> Verdict:
    """The gate's decision on *text*, taken the way the post-OCR pipeline takes it."""
    detected, _confidence = _detect_language_metadata(text)
    language = _normalize_detected_language(detected)
    report = compute_quality_report(
        text, run_id="calibration", pass_idx=0, language=_quality_language(language), tiled=False
    )
    plausibility = best_fit_plausibility(text, language) if language != "unknown" else None
    gates = enforce_quality_gates(report, run_id="calibration", lexical_plausibility=plausibility)
    blocked = {"token_ner", "token_search"} & set(gates["blocked_stages"])
    allowed = bool(report.ner_allowed and report.token_search_allowed and not blocked)
    return Verdict(language, report.quality_label, report.lexical_contrast, allowed)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Edit distance over reference length, after NFC and whitespace folding."""
    ref = " ".join(unicodedata.normalize("NFC", reference).split())
    hyp = " ".join(unicodedata.normalize("NFC", hypothesis).split())
    if not ref:
        return 0.0 if not hyp else float("inf")
    target = np.fromiter((ord(ch) for ch in hyp), dtype=np.int64, count=len(hyp))
    offsets = np.arange(len(hyp) + 1)
    previous = offsets.copy()
    for row, ch in enumerate(ref, start=1):
        current = np.empty_like(previous)
        current[0] = row
        current[1:] = np.minimum(previous[:-1] + (target != ord(ch)), previous[1:] + 1)
        # Insertions: current[j] = min(current[j], current[j - 1] + 1), run as a prefix minimum.
        previous = np.minimum.accumulate(current - offsets) + offsets
    return float(previous[-1]) / len(ref)


def corrupt(text: str, rate: float, seed: int) -> str:
    """Substitute (60%), insert after (20%) or delete (20%) each letter with probability *rate*."""
    rng = random.Random(seed)
    out: list[str] = []
    for ch in text:
        if ch.isalpha() and rng.random() < rate:
            draw = rng.random()
            if draw < 0.6:
                out.append(rng.choice(_LETTERS))
            elif draw < 0.8:
                out.append(ch + rng.choice(_LETTERS))
        else:
            out.append(ch)
    return "".join(out)


def reverse_words(text: str) -> str:
    return "\n".join(" ".join(word[::-1] for word in line.split()) for line in text.splitlines())


def random_letters(text: str, seed: int) -> str:
    rng = random.Random(seed)
    return "\n".join(
        " ".join("".join(rng.choice(_LETTERS) for _ in word) for word in line.split()) for line in text.splitlines()
    )


SYNTHETIC: tuple[tuple[str, Callable[[str, int], str]], ...] = (
    ("noise 10%", lambda text, seed: corrupt(text, 0.1, seed)),
    ("noise 20%", lambda text, seed: corrupt(text, 0.2, seed)),
    ("noise 30%", lambda text, seed: corrupt(text, 0.3, seed)),
    ("noise 50%", lambda text, seed: corrupt(text, 0.5, seed)),
    ("noise 70%", lambda text, seed: corrupt(text, 0.7, seed)),
    ("reversed words", lambda text, _seed: reverse_words(text)),
    ("random letters", random_letters),
)


def calibrate_readings(path: Path) -> list[dict[str, object]]:
    pages = json.loads(path.read_text(encoding="utf-8"))["pages"]
    rows: list[dict[str, object]] = []
    for index, page in enumerate(pages):
        reference = page["reference"]
        samples = [("reference", reference), *page["readings"].items()]
        samples += [(name, make(reference, index)) for name, make in SYNTHETIC]
        for name, text in samples:
            outcome = verdict(text)
            rows.append(
                {
                    "page": page["page_id"],
                    "sample": name,
                    "cer": character_error_rate(reference, text),
                    "expected_language": page["language"],
                    **outcome.__dict__,
                }
            )
    return rows


def calibrate_database(db_path: Path) -> list[dict[str, object]]:
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute("SELECT ocr_text, proofread_text FROM pipeline_runs").fetchall()
    seen: set[str] = set()
    texts: list[str] = []
    for fields in rows:
        for text in fields:
            text = (text or "").strip()
            key = " ".join(text.split())
            if len(text) >= 300 and key not in seen:
                seen.add(key)
                texts.append(text)
    results: list[dict[str, object]] = []
    for index, text in enumerate(texts):
        variants = [("as transcribed", text)]
        variants += [(name, make(text, index)) for name, make in SYNTHETIC if name in ("noise 50%", "reversed words", "random letters")]
        for name, variant in variants:
            results.append({"text": index, "sample": name, **verdict(variant).__dict__})
    return results


def _render_readings(rows: list[dict[str, object]]) -> str:
    lines = [f"{'page':22s} {'sample':24s} {'CER':>7s} {'language':14s} {'contrast':>8s} {'label':10s} verdict"]
    for row in rows:
        contrast = f"{row['contrast']:.2f}" if float(row["contrast"]) >= 0 else "-"
        lines.append(
            f"{row['page']:22s} {row['sample']:24s} {float(row['cer']):7.1%} {row['language']:14s} {contrast:>8s} "
            f"{row['label']:10s} {'allowed' if row['allowed'] else 'BLOCKED'}"
        )
    return "\n".join(lines)


def _render_database(rows: list[dict[str, object]]) -> str:
    lines = [f"{'sample':16s} {'n':>4s} {'contrast p5':>11s} {'median':>7s} {'p95':>6s} {'blocked':>8s} {'UNRELIABLE':>11s}"]
    for sample in dict.fromkeys(str(row["sample"]) for row in rows):
        own = [row for row in rows if row["sample"] == sample]
        contrasts = sorted(float(row["contrast"]) for row in own if float(row["contrast"]) >= 0)
        q = lambda p: contrasts[min(len(contrasts) - 1, int(p * len(contrasts)))] if contrasts else float("nan")  # noqa: E731
        median = statistics.median(contrasts) if contrasts else float("nan")
        lines.append(
            f"{sample:16s} {len(own):4d} {q(0.05):11.2f} {median:7.2f} {q(0.95):6.2f} "
            f"{sum(not row['allowed'] for row in own):8d} {sum(row['label'] == 'UNRELIABLE' for row in own):11d}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--readings", type=Path, default=DEFAULT_READINGS)
    parser.add_argument("--db", type=Path, help="a pipeline database, for the held-out check")
    parser.add_argument("--json", action="store_true", help="print JSON instead of tables")
    args = parser.parse_args(argv)

    readings = calibrate_readings(args.readings)
    held_out = calibrate_database(args.db) if args.db else []
    if args.json:
        print(json.dumps({"readings": readings, "held_out": held_out}, ensure_ascii=False, indent=2))
        return 0
    print(_render_readings(readings))
    if held_out:
        print()
        print(f"Held out: {len({row['text'] for row in held_out})} transcriptions from {args.db}")
        print(_render_database(held_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
