"""Measure the recognisers, the quality gate and Kraken's confidence on held-out medieval lines.

Usage::

    python scripts/benchmark_heldout.py sample --data DIR --manifest OUT/manifest.jsonl [--seed 20261124]
    python scripts/benchmark_heldout.py run    --data DIR --manifest OUT/manifest.jsonl --out OUT/lines.jsonl
    python scripts/benchmark_heldout.py report --lines OUT/lines.jsonl [--json OUT/report.json]

The data is the test split of the CATMuS Medieval dataset (Hugging Face
``CATMuS/medieval``, CC-BY 4.0), one parquet file per manuscript, downloaded
to DIR (``test/*.parquet``). Its test manuscripts are disjoint from its train
and dev manuscripts, and none of them was in the train or dev split of
release 1.5.1, the data behind the CATMuS Medieval 1.5.0 Kraken model the
backend routes to (see ``eval/heldout/README.md`` for what that does and does
not prove, and for the other two models).

- ``sample`` draws, per manuscript, up to ``--blocks`` blocks of ``--block``
  consecutive main-text lines (file order) at random offsets, from ``--seed``.
  It writes a manifest of line ids (``<file>#<row>``) with their metadata and a
  hash of their reference text, not the images. The same seed and data give
  the same manifest; ``run`` checks the hashes.
- ``run`` reads each line image through the Kraken backends the full-page route
  uses (``KrakenBackend.recognize``, same preprocessing and model files, the
  whole line image as the crop), for CATMuS Medieval, CREMMA Medieval and
  McCATMuS, and writes the text, the confidence and the CER of every line.
- ``report`` gives micro CER by language, century and script, with 95%
  intervals resampled over manuscripts, not lines; each other model against
  CATMuS Medieval as a paired difference; the quality gate on pseudo-pages (one
  per block, its lines joined), verdict against page CER; and Kraken's
  confidence against CER at line and page level.

Not tested here: the detached-initial join, which needs the page layout that
line crops do not have.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import random
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

ENGINES = ("kraken_catmus", "kraken_cremma_medieval", "kraken_mccatmus")
MAIN_TEXT = "MainZone"


def _sha(text: str) -> str:
    return hashlib.sha1(unicodedata.normalize("NFC", text).encode("utf-8")).hexdigest()[:16]


def _files(data: Path) -> list[Path]:
    return sorted((data / "test").glob("*.parquet"))


# ── sample ──────────────────────────────────────────────────────────────


def sample(data: Path, manifest: Path, *, seed: int, block: int, blocks: int) -> None:
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    by_ms: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in _files(data):
        table = pq.read_table(path, columns=["text", "language", "century", "script_type", "shelfmark", "region", "line_type"])
        for row, rec in enumerate(table.to_pylist()):
            if rec["region"] != MAIN_TEXT or not str(rec["text"] or "").strip():
                continue
            by_ms[rec["shelfmark"]].append({"line_id": f"{path.name}#{row}", **{k: rec[k] for k in ("language", "century", "script_type", "shelfmark")}, "text_sha": _sha(rec["text"])})
    out = []
    for shelfmark in sorted(by_ms):
        lines = by_ms[shelfmark]
        if len(lines) <= block:
            starts = [0]
        else:
            # Non-overlapping blocks at random offsets.
            slots = list(range(0, len(lines) - block + 1, block))
            starts = sorted(rng.sample(slots, min(blocks, len(slots))))
        for b, start in enumerate(starts):
            for line in lines[start : start + block]:
                out.append({**line, "block": f"{shelfmark}#{b}"})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    print(f"{len(out)} lines from {len(by_ms)} manuscripts, {len({x['block'] for x in out})} blocks -> {manifest}")


# ── run ─────────────────────────────────────────────────────────────────


def run(data: Path, manifest: Path, out: Path) -> None:
    import pyarrow.parquet as pq
    from PIL import Image

    from app.agents import ocr_agent
    from app.services import ocr_backends
    from app.services.evaluation import DIPLOMATIC, cer

    ocr_agent.write_ocr_evidence_jsonl = lambda _r: None
    backends = {
        "kraken_catmus": ocr_backends.KrakenCatmusBackend(),
        "kraken_cremma_medieval": ocr_backends.KrakenCremmaMedievalBackend(),
        "kraken_mccatmus": ocr_backends.KrakenMcCatmusBackend(),
    }
    entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_file[entry["line_id"].split("#")[0]].append(entry)
    done = set()
    if out.exists():
        done = {(r["line_id"], r["engine"]) for r in map(json.loads, out.read_text(encoding="utf-8").splitlines()) if r}
    started = time.time()
    with out.open("a", encoding="utf-8") as handle:
        for name, group in sorted(by_file.items()):
            table = pq.read_table(data / "test" / name, columns=["text", "im"])
            for entry in group:
                row = int(entry["line_id"].split("#")[1])
                reference = str(table.column("text")[row].as_py())
                if _sha(reference) != entry["text_sha"]:
                    raise SystemExit(f"{entry['line_id']}: the data differs from the manifest (text hash)")
                image_bytes = table.column("im")[row].as_py()["bytes"]
                with Image.open(io.BytesIO(image_bytes)) as image:
                    width, height = image.size
                    buffer = io.BytesIO()
                    image.convert("RGB").save(buffer, format="PNG")
                crop_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
                metadata = ocr_backends.OCRRecognitionMetadata(
                    page_id=entry["block"], image_id=None, region_id=entry["line_id"],
                    bbox_xyxy=[0.0, 0.0, float(width), float(height)], crop_box=(0.0, 0.0, float(width), float(height)),
                )
                for engine, backend in backends.items():
                    if (entry["line_id"], engine) in done:
                        continue
                    try:
                        result = backend.recognize(crop_b64, metadata)
                        text, confidence, error = result.text, result.confidence, None
                    except Exception as exc:  # noqa: BLE001 - a line that fails is recorded
                        text, confidence, error = "", None, f"{type(exc).__name__}: {exc}"
                    counts = cer(unicodedata.normalize("NFC", reference), text, DIPLOMATIC)
                    handle.write(json.dumps({
                        **{k: entry[k] for k in ("line_id", "block", "language", "century", "script_type", "shelfmark")},
                        "engine": engine, "reference": unicodedata.normalize("NFC", reference), "text": text,
                        "confidence": confidence, "errors": counts.distance, "chars": counts.reference_length, "error": error,
                    }, ensure_ascii=False) + "\n")
                handle.flush()
            print(f"{name}: {len(group)} lines ({time.time() - started:.0f} s)", flush=True)


# ── report ──────────────────────────────────────────────────────────────


def _micro(rows: list[dict[str, Any]]) -> float:
    chars = sum(r["chars"] for r in rows)
    return sum(r["errors"] for r in rows) / chars if chars else float("nan")


def _boot(rows_by_ms: dict[str, list[dict[str, Any]]], stat, *, seed: int = 7, n: int = 1000) -> tuple[float, float]:
    rng = random.Random(seed)
    keys = sorted(rows_by_ms)
    values = []
    for _ in range(n):
        pick = [rng.choice(keys) for _ in keys]
        values.append(stat([r for k in pick for r in rows_by_ms[k]]))
    values = sorted(v for v in values if v == v)
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def _spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else None


def report(lines_path: Path, json_out: Path | None) -> None:
    import calibrate_quality_gate as gate

    rows = [json.loads(line) for line in lines_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result: dict[str, Any] = {"lines": len({r["line_id"] for r in rows}), "manuscripts": len({r["shelfmark"] for r in rows})}
    print(f"{result['lines']} lines, {result['manuscripts']} manuscripts, {len({r['block'] for r in rows})} pseudo-pages")

    def by_ms(rs):
        out = defaultdict(list)
        for r in rs:
            out[r["shelfmark"]].append(r)
        return out

    def century_band(c: int) -> str:
        return "8-11" if c <= 11 else str(c)

    strata = {"all": lambda r: "all", "language": lambda r: r["language"], "century": lambda r: century_band(int(r["century"])), "script": lambda r: r["script_type"]}
    tables: dict[str, Any] = {}
    for name, key in strata.items():
        table = {}
        for value in sorted({key(r) for r in rows}, key=str):
            cell = {}
            sub = [r for r in rows if key(r) == value]
            for engine in ENGINES:
                er = [r for r in sub if r["engine"] == engine]
                ms = by_ms(er)
                lo, hi = _boot(ms, _micro)
                cell[engine] = {"cer": _micro(er), "ci": [lo, hi], "lines": len(er), "manuscripts": len(ms)}
            # Paired difference against CATMuS Medieval, resampled over manuscripts.
            base = by_ms([r for r in sub if r["engine"] == "kraken_catmus"])
            for engine in ENGINES[1:]:
                other = by_ms([r for r in sub if r["engine"] == engine])
                both = {k: base[k] + other[k] for k in base if k in other}
                diff = lambda rs: _micro([r for r in rs if r["engine"] == engine]) - _micro([r for r in rs if r["engine"] == "kraken_catmus"])  # noqa: E731
                cell[engine]["minus_catmus"] = {"mean": diff([r for v in both.values() for r in v]), "ci": list(_boot(both, diff))}
            table[value] = cell
        tables[name] = table
    result["cer"] = tables
    for name, table in tables.items():
        print(f"\nMicro CER by {name} (95% CI over manuscripts; lines / manuscripts):")
        for value, cell in table.items():
            parts = []
            for engine in ENGINES:
                c = cell[engine]
                extra = ""
                if "minus_catmus" in c:
                    d = c["minus_catmus"]
                    extra = f" Δ{d['mean']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]"
                parts.append(f"{engine.replace('kraken_', '')} {c['cer']:.3f} [{c['ci'][0]:.3f},{c['ci'][1]:.3f}]{extra}")
            n = cell[ENGINES[0]]
            print(f"  {str(value):34s} ({n['lines']}/{n['manuscripts']})  " + " | ".join(parts))

    # The gate on pseudo-pages: each block's lines joined, as one page.
    pages = []
    for (block, engine), rs in sorted(_group(rows, lambda r: (r["block"], r["engine"])).items()):
        reference = "\n".join(r["reference"] for r in rs)
        text = "\n".join(r["text"] for r in rs)
        verdict = gate.verdict(text)
        pages.append({
            "block": block, "engine": engine, "language": rs[0]["language"], "lines": len(rs),
            "cer": gate.character_error_rate(reference, text), "label": verdict.label, "allowed": verdict.allowed,
            "detected": verdict.language, "confidence": statistics.mean([r["confidence"] for r in rs if r["confidence"] is not None] or [float("nan")]),
        })
    result["pages"] = pages
    print("\nQuality gate on pseudo-pages (blocked at CER <= 10% / allowed at CER >= 25%):")
    for engine in ("all", *ENGINES):
        for language in ["all", *sorted({p["language"] for p in pages})]:
            ps = [p for p in pages if (engine == "all" or p["engine"] == engine) and (language == "all" or p["language"] == language)]
            if not ps or (engine != "all" and language != "all" and len(ps) < 3):
                continue
            low = [p for p in ps if p["cer"] <= 0.10]
            high = [p for p in ps if p["cer"] >= 0.25]
            if engine == "all" or language == "all":
                print(f"  {engine:24s} {language:16s} pages {len(ps):3d}  blocked at <=10%: {sum(not p['allowed'] for p in low)}/{len(low)}  allowed at >=25%: {sum(p['allowed'] for p in high)}/{len(high)}")

    # Kraken confidence against CER.
    result["confidence"] = {}
    print("\nKraken confidence against CER (Spearman; negative = confidence falls as CER rises):")
    for engine in ENGINES:
        er = [r for r in rows if r["engine"] == engine and r["confidence"] is not None and r["chars"]]
        line_rho = _spearman([r["confidence"] for r in er], [r["errors"] / r["chars"] for r in er])
        ps = [p for p in pages if p["engine"] == engine and p["confidence"] == p["confidence"]]
        page_rho = _spearman([p["confidence"] for p in ps], [p["cer"] for p in ps])
        confs = sorted(r["confidence"] for r in er)
        result["confidence"][engine] = {"line_rho": line_rho, "page_rho": page_rho, "lines": len(er), "pages": len(ps), "p5": confs[len(confs) // 20], "p95": confs[-len(confs) // 20]}
        rho = lambda value: "n/a" if value is None else f"{value:+.3f}"  # noqa: E731 - too few points, or nothing to rank
        print(f"  {engine:24s} line rho {rho(line_rho)} over {len(er)} lines; page rho {rho(page_rho)} over {len(ps)} pseudo-pages; confidence p5-p95 {confs[len(confs)//20]:.3f}-{confs[-len(confs)//20]:.3f}")
    if json_out:
        json_out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


def _group(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    for rs in out.values():
        rs.sort(key=lambda r: int(r["line_id"].split("#")[1]))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--seed", type=int, default=20261124)
    s.add_argument("--block", type=int, default=30)
    s.add_argument("--blocks", type=int, default=2)
    r = sub.add_parser("run")
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("report")
    p.add_argument("--lines", type=Path, required=True)
    p.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.command == "sample":
        sample(args.data, args.manifest, seed=args.seed, block=args.block, blocks=args.blocks)
    elif args.command == "run":
        run(args.data, args.manifest, args.out)
    else:
        report(args.lines, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
