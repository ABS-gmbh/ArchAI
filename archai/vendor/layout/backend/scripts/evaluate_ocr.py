"""Score OCR output against reference transcriptions.

Usage::

    python scripts/evaluate_ocr.py --manifest eval/manifest.jsonl
    python scripts/evaluate_ocr.py --manifest eval/manifest.jsonl --policy lenient --json

The manifest is JSON Lines, one page per line::

    {"page_id": "fmb-cb-0001_001r", "reference": "gold/fmb-cb-0001_001r.gt.txt",
     "hypothesis": "runs/2026-09-24/fmb-cb-0001_001r.txt"}

Paths are resolved relative to the manifest. Reference files should hold a
diplomatic transcription, one manuscript line per line, with no headers or zone
labels - anything that is not transcription is scored as text.

Exit status is 0 on success and 2 when the manifest cannot be read, so the tool
can gate CI once a gold set exists.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.evaluation import (  # noqa: E402
    DIPLOMATIC,
    LENIENT,
    CorpusReport,
    evaluate_corpus,
)

POLICIES = {"diplomatic": DIPLOMATIC, "lenient": LENIENT}


class ManifestError(ValueError):
    pass


def load_manifest(path: Path) -> list[tuple[str, str, str]]:
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")
    base = path.parent
    triples: list[tuple[str, str, str]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{path}:{number}: not valid JSON ({exc.msg})") from exc
        missing = [key for key in ("page_id", "reference", "hypothesis") if key not in record]
        if missing:
            raise ManifestError(f"{path}:{number}: missing {', '.join(missing)}")
        texts = []
        for key in ("reference", "hypothesis"):
            file_path = (base / record[key]).resolve()
            if not file_path.is_file():
                raise ManifestError(f"{path}:{number}: {key} file not found: {file_path}")
            texts.append(file_path.read_text(encoding="utf-8"))
        triples.append((str(record["page_id"]), texts[0], texts[1]))
    if not triples:
        raise ManifestError(f"{path}: no pages listed")
    return triples


def _pct(value: float) -> str:
    return "inf" if not math.isfinite(value) else f"{value * 100:6.2f}%"


def _interval(bounds: tuple[float, float]) -> str:
    low, high = bounds
    if math.isnan(low):
        return "n/a"
    return f"[{_pct(low).strip()}, {_pct(high).strip()}]"


def render(report: CorpusReport, *, worst: int = 5) -> str:
    pct = int(round(report.confidence * 100))
    lines = [
        f"pages scored     {len(report.pages)}",
        f"normalisation    {report.policy.describe()}",
        "",
        f"{'':<6}{'micro':>9}  {pct}% CI{'':<15}{'macro':>9}  {pct}% CI",
    ]
    for name, rate in (("CER", report.cer), ("WER", report.wer)):
        lines.append(
            f"{name:<6}{_pct(rate.micro):>9}  {_interval(rate.micro_ci):<21}"
            f"{_pct(rate.macro):>9}  {_interval(rate.macro_ci)}"
        )
    lines += [
        "",
        "micro = total edits / total reference characters (long pages weigh more)",
        "macro = mean of per-page rates (every page weighs the same)",
    ]
    breakdown = report.char_breakdown
    if breakdown.get("substitutions") is not None:
        lines += [
            "",
            "character edits  "
            f"{breakdown['substitutions']} substitutions, "
            f"{breakdown['deletions']} deletions, "
            f"{breakdown['insertions']} insertions",
        ]
    if len(report.pages) > 1:
        lines += ["", f"worst {min(worst, len(report.pages))} pages by CER"]
        for page in report.worst_pages(worst):
            lines.append(f"  {page.page_id:<40} {_pct(page.char.rate)}")
    if len(report.pages) < 2:
        lines += [
            "",
            "note: no confidence interval can be estimated from a single page",
        ]
    elif len(report.pages) < 10:
        lines += [
            "",
            f"note: with only {len(report.pages)} pages the interval is wide; treat it as indicative",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="diplomatic")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed, for reproducible intervals")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    try:
        triples = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = evaluate_corpus(
        triples, policy=POLICIES[args.policy], confidence=args.confidence, seed=args.seed
    )
    print(json.dumps(report.to_dict(), indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
