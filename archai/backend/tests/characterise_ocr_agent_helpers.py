"""What the OCR agent's parsing and tiling helpers give on fixed inputs.

``tests/test_ocr_agent_helpers_characterisation.py`` compares them with
``tests/fixtures/ocr_agent_helpers_characterisation.json``, recorded before
they were moved out of ``app/agents/ocr_agent.py``. Encoded tiles are recorded
by size and mean level, which do not depend on the Pillow build. To re-record
after an intended change::

    python tests/characterise_ocr_agent_helpers.py --write
"""

from __future__ import annotations

import base64
import dataclasses
import importlib
import io
import json
import random
import sys
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
FIXTURE = BACKEND / "tests" / "fixtures" / "ocr_agent_helpers_characterisation.json"
READINGS = BACKEND / "eval" / "quality_gate" / "readings.json"
_MODULES = ("app.agents.ocr_text_parsing", "app.agents.ocr_tiles")


def helper(name: str) -> Any:
    for module in _MODULES:
        try:
            found = importlib.import_module(module)
        except ModuleNotFoundError:
            continue
        if hasattr(found, name):
            return getattr(found, name)
    return getattr(importlib.import_module("app.agents.ocr_agent"), name)


def _level(b64: str) -> list[Any]:
    from PIL import Image

    image = Image.open(io.BytesIO(base64.b64decode(b64)))
    data = image.convert("L").tobytes()
    return [image.size, round(sum(data) / max(1, len(data)))]


def _clean(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        record = dataclasses.asdict(value)
        if "image_b64" in record:
            record["image_b64"] = _level(record["image_b64"])
        return _clean(record)
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def _text_record(text: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if line.strip()]
    wrapped = json.dumps({"region_id": "r1", "lines": lines, "text": text, "script_hint": "latin", "confidence": 0.71, "warnings": []}, ensure_ascii=False)
    tile_results = [
        helper("OCRTileResult")(tile_id=f"t{i}", bbox_xyxy=[0.0, float(i * 100), 100.0, float(i * 100 + 90)], lines=lines[i::3], text="\n".join(lines[i::3]), confidence=0.5 + i / 10, script_hint="latin", warnings=[], flags=[], model="m")
        for i in range(3)
    ]
    return {
        "sanitized": helper("sanitize_ocr_text")(text),
        "sanitized_fenced": helper("sanitize_ocr_text")("```json\n" + wrapped + "\n```"),
        "junk": [helper("_looks_like_pattern_junk")(line) for line in lines[:12]],
        "latin_lines": [helper("_sanitize_line_for_latin")(line, latin_lock=True) for line in lines[:12]],
        "normalized": helper("_normalize_lines")(list(lines), latin_lock=True),
        "normalized_unlocked": helper("_normalize_lines")(list(lines), latin_lock=False),
        "parsed": helper("parse_ocr_json")(wrapped, expected_region_id="r1"),
        "parsed_strict": helper("parse_ocr_json")(wrapped, latin_lock=False, strict_keys=True),
        "parsed_raw": helper("parse_ocr_json")(text),
        "quality": helper("score_text_quality")(text),
        "merged_lines": helper("_merge_tile_lines")(tile_results),
        "merged_confidence": helper("_merge_confidence")(tile_results),
        "language": helper("best_fit_language")(text),
        "region_quality": [helper("_region_quality_value")(text, c, h) for c in (None, 0.4, 0.9) for h in (None, "latin", "old_french")],
        "changes": helper("_summarize_final_changes")(text, text.replace("⁊", "et")),
    }


def _page_b64(size: tuple[int, int], seed: int) -> str:
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    image = Image.new("RGB", size, (240, 232, 210))
    draw = ImageDraw.Draw(image)
    for _ in range(rng.randrange(20, 60)):
        x, y = rng.randrange(size[0] // 8, size[0] * 7 // 8), rng.randrange(size[1] // 10, size[1] * 9 // 10)
        draw.rectangle([x, y, x + rng.randrange(30, 200), y + rng.randrange(6, 18)], fill=(30, 25, 20))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _tile_record(size: tuple[int, int], seed: int) -> dict[str, Any]:
    from PIL import Image

    from app.schemas.agents_ocr import OCRExtractRequest, OCRRegionInput

    b64 = _page_b64(size, seed)
    source = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    boxes = helper("_build_tile_boxes")(source)
    tiles = helper("_boxes_to_tiles")(source, boxes, upscale_factor=2)
    rng = random.Random(seed)
    regions = [OCRRegionInput(region_id=f"r{i}", bbox_xyxy=[rng.randrange(0, size[0] // 2), rng.randrange(0, size[1] // 2), rng.randrange(size[0] // 2, size[0]), rng.randrange(size[1] // 2, size[1])]) for i in range(rng.randrange(0, 4))]
    request_tiles, request_image = helper("_make_tiles_from_payload")(OCRExtractRequest(page_id="p", image_b64=b64, regions=regions), upscale_factor=2)
    # The page comes back PNG-encoded; its bytes depend on the platform's zlib.
    payload_tiles = [request_tiles, _level(request_image) if request_image else request_image]
    resized = helper("_resize_crop_for_ocr")(source.crop((0, 0, size[0] // 2, size[1] // 2)), requested_scale=2.0)
    return {
        "block": helper("_detect_text_block_bbox")(source),
        "boxes": boxes,
        "tiles": tiles,
        "payload_tiles": payload_tiles,
        "resized": [resized[0].size, resized[1]],
        "downscaled": [helper("_downscale_tile_for_retry")(t) for t in tiles[:2]],
    }


def record() -> dict[str, Any]:
    pages = json.loads(READINGS.read_text(encoding="utf-8"))["pages"]
    texts = {f"{p['page_id']}:{name}": text for p in pages for name, text in {"reference": p["reference"], **p["readings"]}.items()}
    return _clean(
        {
            "texts": {name: _text_record(text) for name, text in texts.items()},
            "tiles": {f"{w}x{h}:{seed}": _tile_record((w, h), seed) for seed, (w, h) in enumerate(((800, 1100), (1600, 1200), (400, 2400), (2600, 3400)))},
            "models": helper("choose_models")(["gemma-3-27b-it", "internvl3.5-30b-a3b", "qwen3-vl-30b-a3b-instruct"], ["qwen3-vl-30b-a3b-instruct", "missing", "internvl3.5-30b-a3b"]),
        }
    )


if __name__ == "__main__":
    if "--write" in sys.argv:
        FIXTURE.write_text(json.dumps(record(), ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {FIXTURE}")
    else:
        print(json.dumps(record(), ensure_ascii=False)[:1500])
