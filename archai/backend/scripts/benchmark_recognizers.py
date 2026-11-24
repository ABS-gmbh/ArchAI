"""Compare the full-page OCR engines on pages that have reference transcriptions.

Usage::

    python scripts/benchmark_recognizers.py --images /path/to/page/images
    python scripts/benchmark_recognizers.py --images DIR --engines segmented glmocr --json
    python scripts/benchmark_recognizers.py --images DIR --save-readings fresh_readings.json

Each engine reads a page exactly as /ocr/extract_full_page would - layout
segmentation then line recognition ("segmented", optionally pinned to one
backend as "segmented:<backend>"), or GLM-OCR on the whole page ("glmocr") -
but nothing is written to the pipeline database or the OCR evidence log.

Pages are matched to the manifest by SHA-256, so image file names do not
matter. Reported per page and engine: character error rate under the
diplomatic policy (case, punctuation and abbreviation marks kept), word error
rate, and the CER of the search keys the retrieval index is built from - the
view that decides whether a query finds the page at all.

``--save-readings PATH`` also writes the text each engine read, in the format
of ``eval/quality_gate/readings.json``, so ``scripts/calibrate_quality_gate.py
--readings PATH`` can check the gate on fresh readings. Readings are keyed by
engine name; an engine that failed on a page is left out of that page.

Exit status is 0 on success, 2 when the manifest or an image is missing.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Keep the layout detector's per-image log lines out of the report.
os.environ.setdefault("YOLO_VERBOSE", "False")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents import ocr_agent  # noqa: E402
from app.routers import ocr as ocr_router  # noqa: E402
from app.schemas.agents_ocr import SaiaFullPageExtractRequest  # noqa: E402
from app.services.evaluation import DIPLOMATIC, ErrorCounts, cer, wer  # noqa: E402
from app.services.medieval_text import build_search_key, rejoin_line_breaks  # noqa: E402

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "eval" / "recognizers" / "manifest.jsonl"
DEFAULT_ENGINES = (
    "segmented",
    "segmented:kraken_cremma_medieval",
    "segmented:kraken_mccatmus",
    "glmocr",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


class BenchmarkError(ValueError):
    pass


@dataclass(frozen=True)
class Page:
    page_id: str
    image: Path
    reference: str
    language_hint: str | None


@dataclass(frozen=True)
class Reading:
    engine: str
    page_id: str
    model: str
    cer: float
    wer: float
    key_cer: float
    seconds: float
    characters: int
    char_errors: int
    error: str | None = None


def load_pages(manifest: Path, images: Path) -> list[Page]:
    if not manifest.is_file():
        raise BenchmarkError(f"manifest not found: {manifest}")
    if not images.is_dir():
        raise BenchmarkError(f"image directory not found: {images}")
    by_digest = {
        hashlib.sha256(path.read_bytes()).hexdigest(): path
        for path in sorted(images.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    pages: list[Page] = []
    for number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        record = json.loads(line)
        image = by_digest.get(str(record["image_sha256"]))
        if image is None:
            raise BenchmarkError(f"{manifest}:{number}: no image in {images} has SHA-256 {record['image_sha256']}")
        reference = (manifest.parent / str(record["reference"])).read_text(encoding="utf-8")
        pages.append(Page(str(record["page_id"]), image, reference, record.get("language_hint")))
    return pages


def read_page(engine: str, page: Page) -> tuple[str, str]:
    """Text and model name, as /ocr/extract_full_page produces them for ``engine``."""
    kind, _, backend = engine.partition(":")
    data = page.image.read_bytes()
    payload = SaiaFullPageExtractRequest(
        page_id=page.page_id,
        image_b64=base64.b64encode(data).decode("ascii"),
        language_hint=page.language_hint,
        apply_proofread=False,
        ocr_backend=backend or ("glmocr" if kind == "glmocr" else "auto"),
    )
    if kind == "segmented":
        response, _backend = ocr_router._run_segmented_full_page(payload, data)
    elif kind == "glmocr":
        response = asyncio.run(ocr_router._run_glm_full_page(payload, data))
    else:
        raise BenchmarkError(f"unknown engine: {engine}")
    return response.text, response.model_used


def search_key(text: str) -> str:
    return build_search_key(rejoin_line_breaks(text))


def score(engine: str, page: Page) -> Reading:
    return read_and_score(engine, page)[0]


def read_and_score(engine: str, page: Page) -> tuple[Reading, str]:
    """The scored reading and the text the engine read ("" if it failed)."""
    started = time.perf_counter()
    try:
        text, model = read_page(engine, page)
        error = None
    except Exception as exc:  # an engine that cannot run is reported, not fatal
        text, model, error = "", "", f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - started
    chars: ErrorCounts = cer(page.reference, text, DIPLOMATIC)
    reading = Reading(
        engine=engine,
        page_id=page.page_id,
        model=model,
        cer=chars.rate,
        wer=wer(page.reference, text, DIPLOMATIC).rate,
        key_cer=cer(search_key(page.reference), search_key(text)).rate,
        seconds=round(seconds, 2),
        characters=chars.reference_length,
        char_errors=chars.distance,
        error=error,
    )
    return reading, text


def readings_file(pages: list[Page], results: list[tuple[Reading, str]]) -> dict:
    """The texts read, in the format of eval/quality_gate/readings.json."""
    out = []
    for page in pages:
        texts = {r.engine: text for r, text in results if r.page_id == page.page_id and not r.error}
        # language_hint "old_french" -> "french", as readings.json records the language.
        language = (page.language_hint or "unknown").rsplit("_", 1)[-1]
        out.append({"page_id": page.page_id, "language": language, "reference": page.reference, "readings": texts})
    return {"description": "Readings saved by scripts/benchmark_recognizers.py --save-readings.", "pages": out}


def render(readings: list[Reading]) -> str:
    rows = [f"{'engine':34s} {'page':14s} {'model':18s} {'CER':>7s} {'WER':>7s} {'key CER':>8s} {'seconds':>8s}"]
    for item in readings:
        if item.error:
            rows.append(f"{item.engine:34s} {item.page_id:14s} failed: {item.error[:80]}")
            continue
        rows.append(
            f"{item.engine:34s} {item.page_id:14s} {item.model[:18]:18s} "
            f"{item.cer:7.1%} {item.wer:7.1%} {item.key_cer:8.1%} {item.seconds:8.1f}"
        )
    rows.append("")
    rows.append("Over all pages: CER micro-averaged, and seconds in total:")
    for engine in dict.fromkeys(item.engine for item in readings):
        own = [item for item in readings if item.engine == engine and not item.error]
        characters = sum(item.characters for item in own)
        if characters and len(own) == len({item.page_id for item in readings}):
            rate = sum(item.char_errors for item in own) / characters
            rows.append(f"  {engine:32s} {rate:7.1%} {sum(item.seconds for item in own):8.1f}")
        else:
            rows.append(f"  {engine:32s} {'incomplete':>7s}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--images", type=Path, required=True, help="directory holding the page images")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--engines", nargs="+", default=list(DEFAULT_ENGINES))
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.add_argument("--save-readings", type=Path, help="also write the texts read, in readings.json format")
    args = parser.parse_args(argv)

    try:
        pages = load_pages(args.manifest, args.images)
    except (BenchmarkError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # A benchmark run is not evidence about a document.
    ocr_agent.write_ocr_evidence_jsonl = lambda _record: None  # type: ignore[assignment]

    # The layout models print progress to stdout; keep it for the report alone.
    with contextlib.redirect_stdout(sys.stderr):
        results = [read_and_score(engine, page) for engine in args.engines for page in pages]
    readings = [reading for reading, _text in results]
    if args.save_readings:
        args.save_readings.write_text(
            json.dumps(readings_file(pages, results), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if args.json:
        print(json.dumps([asdict(item) for item in readings], ensure_ascii=False, indent=2))
    else:
        print(render(readings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
