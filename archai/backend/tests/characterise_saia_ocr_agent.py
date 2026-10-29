"""What the SAIA OCR agent's module-level helpers produce on fixed inputs.

``tests/test_saia_ocr_agent_characterisation.py`` compares them with
``tests/fixtures/saia_ocr_agent_characterisation.json``, recorded before they
were moved out of ``app/agents/saia_ocr_agent.py``. To re-record after an
intended change::

    python tests/characterise_saia_ocr_agent.py --write
"""

from __future__ import annotations

import base64
import importlib
import io
import json
import random
import sys
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
FIXTURE = BACKEND / "tests" / "fixtures" / "saia_ocr_agent_characterisation.json"
READINGS = BACKEND / "eval" / "quality_gate" / "readings.json"
_MODULES = ("app.services.ocr_sanity", "app.services.ocr_tiling", "app.services.ocr_image_prep")


def helper(name: str) -> Any:
    """*name* from the module it now lives in, or from the agent before the move."""
    for module in _MODULES:
        try:
            found = importlib.import_module(module)
        except ModuleNotFoundError:
            continue
        if hasattr(found, name):
            return getattr(found, name)
    return getattr(importlib.import_module("app.agents.saia_ocr_agent"), name)


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def _texts() -> dict[str, str]:
    out = {}
    for page in json.loads(READINGS.read_text(encoding="utf-8"))["pages"]:
        for name, text in {"reference": page["reference"], **page["readings"]}.items():
            out[f"{page['page_id']}:{name}"] = text
    return out


def _text_record(text: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if line.strip()]
    sanity = helper("compute_sanity")(text)
    script = helper("_detect_script_hint")(text)
    record: dict[str, Any] = {
        "sanity": sanity,
        "label": helper("_quality_label_from_sanity")(sanity),
        "sanity_metrics": helper("_ocr_sanity_metrics")(text),
        "is_sane": helper("_is_sane_ocr")(text),
        "junk_ratio": helper("_junk_ratio")(text),
        "script_hint": script,
        "fallback_language": helper("_fallback_detected_language")(script, text),
        "sanitized": helper("sanitize_lines")(list(lines), "latin", script),
        "sanitized_french": helper("sanitize_lines")(list(lines), "old_french", script),
        "diplomatic": helper("enforce_diplomatic_uncertainty")(list(lines), script, "latin"),
        "gate": helper("quality_gate_enforce")(list(lines)),
        "markers": helper("force_uncertainty_markers")(list(lines), sanity),
        "confidence": helper("sanity_adjust_confidence")(0.8, sanity),
        "summary": helper("format_sanity_summary")(sanity),
        "caps": helper("_apply_ocr_confidence_caps")(text, 0.9, []),
        "repaired": helper("_repair_lines_and_text")(lines, text),
        "dup_ratio": helper("_line_dup_ratio")(lines),
        "garbage_ratio": helper("_garbage_ratio")(text),
        "drift": helper("_script_drift_ratio")(text, script),
        "hallucinated": helper("_looks_hallucinated_passage")(text, lines),
        "low_quality": helper("_is_low_text_quality")(text, lines, script),
        "parsed": helper("_parse_ocr_payload")(json.dumps({"lines": lines, "text": text, "script_hint": script, "confidence": 0.7, "warnings": []}, ensure_ascii=False)),
    }
    return record


def _geometry_record(seed: int) -> dict[str, Any]:
    from app.schemas.agents_ocr import SaiaOCRLocationSuggestion

    rng = random.Random(seed)
    width, height = 2000, 3000
    suggestions = []
    for i in range(rng.randrange(3, 14)):
        x, y = rng.uniform(0, 1700), rng.uniform(0, 2700)
        suggestions.append(
            SaiaOCRLocationSuggestion(
                region_id=f"r{i}",
                category=rng.choice(["Main script black", "Main script coloured", "Gloss", "Page Number"]),
                bbox_xywh=[x, y, rng.uniform(40, 300), rng.uniform(20, 300)],
            )
        )
    rects = helper("_suggestions_to_rects")(suggestions, width, height)
    return {
        "described": helper("_format_location_suggestions")(suggestions),
        "rects": rects,
        "merged": helper("_merge_overlapping_rects")(list(rects)),
        "merged_simple": helper("_merge_overlapping_rects_simple")(list(rects)),
        "blocks": helper("_rects_to_block_rects")(list(rects), width, height),
        "padded": [helper("_pad_rect")(r, width, height) for r in rects],
        "ordered": helper("_order_tiles_reading")(list(rects), width),
    }


def _image_record(size: tuple[int, int], seed: int) -> dict[str, Any]:
    from PIL import Image

    rng = random.Random(seed)
    image = Image.new("RGB", size, (235, 225, 200))
    pixels = image.load()
    for _ in range(size[0] * size[1] // 40):
        pixels[rng.randrange(size[0]), rng.randrange(size[1])] = (rng.randrange(60), rng.randrange(60), rng.randrange(60))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    # Encoded PNG bytes depend on the Pillow and zlib build, and resampled
    # pixels on the Pillow version: a resized image is recorded by its size,
    # mode and mean level, which do not.
    def pixels(b64: str) -> list[Any]:
        decoded = Image.open(io.BytesIO(base64.b64decode(b64)))
        return resampled(decoded)

    def resampled(img: Any) -> list[Any]:
        data = img.convert("L").tobytes()
        return [img.size, img.mode, round(sum(data) / max(1, len(data)))]

    enhanced = helper("_enhance_manuscript_image")(helper("_open_rgb_image_from_b64")(b64))
    resized, changed = helper("_resize_for_limits")(image)
    prepared, warnings = helper("_prepare_image_for_ocr")(b64)
    downscaled = helper("_downscale_image_b64_for_retry")(b64)
    crop = helper("_crop_and_maybe_upscale")(image, (10, 10, size[0] // 2, size[1] // 3))
    return {
        "enhanced": resampled(enhanced),
        "resized": [resampled(resized), changed],
        "prepared": [pixels(prepared), warnings],
        "downscaled": pixels(downscaled) if downscaled else None,
        "crop": resampled(crop),
        "encoded": pixels(helper("_encode_png_base64")(image)),
    }


def record() -> dict[str, Any]:
    return _clean(
        {
            "texts": {name: _text_record(text) for name, text in _texts().items()},
            "geometry": {str(seed): _geometry_record(seed) for seed in range(12)},
            "images": {f"{w}x{h}": _image_record((w, h), w + h) for w, h in ((300, 420), (1200, 800), (64, 2400))},
        }
    )


if __name__ == "__main__":
    if "--write" in sys.argv:
        FIXTURE.parent.mkdir(exist_ok=True)
        FIXTURE.write_text(json.dumps(record(), ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {FIXTURE}")
    else:
        print(json.dumps(record(), ensure_ascii=False)[:1500])
