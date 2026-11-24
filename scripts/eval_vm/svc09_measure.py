"""SVC-09: does Kraken's confidence track CER, and does the pooling matter?

Runs the segmented full-page route (as benchmark_recognizers does) with
kraken.rpred.rpred wrapped, so every record's per-character confidences are
kept per crop. Then, per engine and page:

- page CER (diplomatic) of the route's text;
- three page confidences: "route" (what the route reports), "current"
  (per crop: mean over all characters of all its records, then mean over
  crops - the code as it is), "per_line" (per crop: mean over its records of
  each record's mean, then mean over crops), "char_pooled" (mean over every
  character of the page);
- per record (line): its mean confidence and its CER against the closest
  reference line (minimum CER over the page's reference lines; a heuristic,
  since output and reference lines are not aligned one to one).

Writes JSON to argv[2] and prints Spearman correlations.
Usage (from archai/backend): python ../../scripts/eval_vm/svc09_measure.py ~/pages out.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import kraken.rpred as kr  # noqa: E402

import benchmark_recognizers as br  # noqa: E402
from app.agents import ocr_agent  # noqa: E402
from app.services import ocr_backends  # noqa: E402
from app.services.evaluation import DIPLOMATIC, cer  # noqa: E402

ocr_agent.write_ocr_evidence_jsonl = lambda _r: None

CROPS: list[list[tuple[str, list[float]]]] = []
_orig_rpred = kr.rpred


def _rpred(*args, **kwargs):
    records = list(_orig_rpred(*args, **kwargs))
    CROPS.append([(str(getattr(r, "prediction", "") or ""), [float(c) for c in (getattr(r, "confidences", []) or [])]) for r in records])
    return iter(records)


kr.rpred = _rpred


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def ranks(xs):
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


def spearman(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None, len(pairs)
    ra, rb = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    ma, mb = mean(ra), mean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return (cov / (va * vb) if va and vb else None), len(pairs)


def main() -> None:
    pages = br.load_pages(br.DEFAULT_MANIFEST, Path(sys.argv[1]).expanduser())
    engines = ["segmented", "segmented:kraken_cremma_medieval", "segmented:kraken_mccatmus"]
    results = []
    for engine in engines:
        for page in pages:
            CROPS.clear()
            started = time.time()
            text, model = br.read_page(engine, page)
            crops = [c for c in CROPS if c]
            ref_lines = [line for line in page.reference.splitlines() if line.strip()]
            current = mean(mean(c for _t, cs in crop for c in cs) for crop in crops if any(cs for _t, cs in crop))
            per_line = mean(mean(mean(cs) for _t, cs in crop if cs) for crop in crops if any(cs for _t, cs in crop))
            pooled = mean(c for crop in crops for _t, cs in crop for c in cs)
            lines = []
            for crop in crops:
                for rec_text, cs in crop:
                    if not rec_text.strip() or not cs:
                        continue
                    line_cer = min(cer(ref, rec_text, DIPLOMATIC).rate for ref in ref_lines)
                    lines.append({"conf": mean(cs), "cer": line_cer, "chars": len(rec_text), "records_in_crop": len(crop)})
            results.append(
                {
                    "engine": engine,
                    "page_id": page.page_id,
                    "model": model,
                    "page_cer": cer(page.reference, text, DIPLOMATIC).rate,
                    "conf_current": current,
                    "conf_per_line": per_line,
                    "conf_char_pooled": pooled,
                    "crops": len(crops),
                    "crops_with_several_records": sum(1 for c in crops if len(c) > 1),
                    "records": sum(len(c) for c in crops),
                    "seconds": round(time.time() - started, 1),
                    "lines": lines,
                }
            )
            r = results[-1]
            print(f"{engine:34s} {page.page_id:22s} CER {r['page_cer']:.3f} conf cur {r['conf_current']:.4f} per-line {r['conf_per_line']:.4f} pooled {r['conf_char_pooled']:.4f} crops {r['crops']} multi {r['crops_with_several_records']}", flush=True)
    Path(sys.argv[2]).write_text(json.dumps(results, indent=1), encoding="utf-8")

    print("\nPage level (Spearman of confidence against page CER; negative = confidence falls as CER rises):")
    for key in ("conf_current", "conf_per_line", "conf_char_pooled"):
        rho, n = spearman([r[key] for r in results], [r["page_cer"] for r in results])
        print(f"  {key:18s} rho {rho:+.3f} over {n} engine-pages" if rho is not None else f"  {key}: n={n}")
    print("\nLine level (Spearman of a line's mean confidence against its CER to the closest reference line):")
    all_lines = [l for r in results for l in r["lines"]]
    for engine in engines + ["all"]:
        ls = all_lines if engine == "all" else [l for r in results if r["engine"] == engine for l in r["lines"]]
        rho, n = spearman([l["conf"] for l in ls], [l["cer"] for l in ls])
        print(f"  {engine:34s} rho {rho:+.3f} over {n} lines" if rho is not None else f"  {engine}: n={n}")
    multi = sum(r["crops_with_several_records"] for r in results)
    crops = sum(r["crops"] for r in results)
    print(f"\nCrops with more than one Kraken record (where the pooling can differ): {multi} of {crops}")
    diffs = [abs(r["conf_current"] - r["conf_per_line"]) for r in results]
    print(f"Largest |current - per_line| page confidence difference: {max(diffs):.5f}")


if __name__ == "__main__":
    main()
