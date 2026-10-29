from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from typing import Any, Sequence

from PIL import Image

from app.agents.base import BaseAgent
from app.agents.crop_agent import crop_page_region, decode_image_bytes, decode_page
from app.agents.ocr_proofreader_agent import OcrProofreaderAgent
from app.config import settings
from app.schemas.agents_ocr import (
    OCRExtractAnyResponse,
    OCRComparisonResult,
    OCRExtractRequest,
    OCRExtractResponse,
    OCRExtractSimpleResponse,
    OCRFallback,
    OCRProvenance,
    OCRRawOCRPayload,
    OCRRegionResult,
)
from app.services.ocr_backends import (
    OCRBackendError,
    OCRBackendPlan,
    OCRRecognitionMetadata,
    build_backend_runtime,
    select_backend_plan,
)
from app.services.ocr_evidence import (
    OcrEvidenceRecord,
    now_iso,
    sha256_bytes,
    write_ocr_evidence_jsonl,
)

from app.services.saia_client import SaiaClient, is_model_not_found_error
from app.agents.ocr_agent_errors import (
    OCRAgentError,
)
from app.agents.ocr_tiles import (
    OCRTile,
    _downscale_tile_for_retry,
    _make_tiles_from_payload,
)
from app.agents.ocr_text_parsing import (
    ALLOWED_SCRIPT_HINTS,
    OCRTileResult,
    _contains_script_drift,
    _detect_script_hint,
    _merge_confidence,
    _merge_tile_lines,
    _region_quality_value,
    _summarize_final_changes,
    parse_ocr_json,
    score_text_quality,
)

AGENT_VERSION = "5.0.0"
OCR_PROMPT_VERSION = "paleo_ocr_tiled_v1"
LOCKED_OCR_MODEL = "internvl3.5-30b-a3b"
PIPELINE_VERSION = str(
    os.getenv("ARCHAI_PIPELINE_VERSION")
    or os.getenv("ARCHAI_GIT_SHA")
    or os.getenv("GIT_SHA")
    or "dev"
)
PROMPT_VERSION = OCR_PROMPT_VERSION
OCR_DECODING_TEMPERATURE = 0.0
OCR_DECODING_TOP_P = 1.0
OCR_DECODING_MAX_TOKENS = 3000
OCR_IMAGE_SIZE_RETRY_LIMIT = max(
    0,
    int(settings.ocr_image_size_retry_limit or os.getenv("OCR_IMAGE_SIZE_RETRY_LIMIT", "2") or "2"),
)
PALEO_OCR_SYSTEM_PROMPT = """You are a professional paleographer producing diplomatic OCR for historical manuscripts.

Core rule: Transcribe ONLY glyphs that are clearly visible in the provided image. NEVER invent, continue, normalize, or “repair” text from memory, context, or known sources. If uncertain, keep what you see and mark uncertainty.

Reading order
- If the page is two-column, output left column top→bottom, then right column top→bottom.
- If single-column, output top→bottom.
- Ignore decorations/frames except when they contain letters.
- Treat a large decorated initial as the first letter(s) of the line’s first word when clearly visible.

Uncertainty markers (MUST be exact)
- Use ? for unclear character(s) inside a word.
- Use […] for unclear spans (one or more words / longer missing sequence).
- NEVER replace […] with guessed content.
- NEVER remove ?/[…] unless the image itself makes the reading deterministic.

Script / character discipline (STRICT)
First decide script_hint from this set:
- insular_old_english
- latin_medieval
- unknown

If script_hint=insular_old_english, avoid drifting into modern Latin-looking substitutions. Use these conventions consistently when clearly visible:
- Use þ for thorn, ð for eth, ƿ for wynn.
- Use 7 for the tironian et / “and” sign when written as that symbol.
- If you see a macron/overline and can represent it, use combining macron (e.g., ā = a + U+0304). If you cannot do it reliably, keep the base letter and add ? or […] rather than inventing expansions.

Do not output characters from other scripts (Greek/Cyrillic/CJK). If you catch yourself about to output non-Latin characters, STOP and use ? / […] instead.

Output contract (NON-NEGOTIABLE)
Return exactly one JSON object with keys: lines, text, script_hint, confidence, warnings.
- lines is an array of strings, one manuscript line per entry.
- text MUST equal "\\n".join(lines).
- confidence is a float 0–1 reflecting how readable the page is overall.
- warnings is an array of strings (may be empty).

If almost nothing is readable: return lines=[] and text="".

If you recognize a famous passage (scripture/boilerplate): treat that recognition as a hallucination risk and only transcribe what is visibly legible; put […] everywhere else.

Internal OCR method
- First inspect the visible strokes, letterforms, spacing, abbreviation marks, and line structure.
- Next infer the most likely script family, language family, and genre from the visible evidence alone.
- Only then use familiar formulas or known passages as soft anchors to validate a reading.
- Anchor phrases are checks, not sources. Never complete a passage from memory.
- A good reading stays in the right script family, preserves plausible word boundaries and morphology, and matches the visible strokes."""

PALEO_OCR_USER_PROMPT = '''Return JSON only.

Schema rules:
- Keys must be exactly: lines, text, script_hint, confidence, warnings
- One manuscript line per entry in lines
- text must equal "\\n".join(lines)
- Preserve original spelling, abbreviations, punctuation, capitalization, and line breaks
- Do not translate, normalize, modernize, or expand abbreviations
- Do not output coordinates, markdown, commentary, or extra keys
- Do not output tile ids, list numbers, region ids, or any prompt metadata as transcription
- Use uncertainty markers exactly: ? and […] (no alternatives)

Quality rules:
- Prefer ? / […] over guessing.
- Do NOT introduce characters from other scripts (Greek/Cyrillic/CJK). If uncertain, use ? / […].
- If page is two-column: output left column fully first, then right column.
- If the crop resembles a known biblical or liturgical phrase, verify it against the visible strokes before using it; never copy a memorized phrase.

If nothing readable is visible, return lines=[] and text="".'''

OCR_JSON_REPAIR_PROMPT = (
    "Return VALID JSON only. Keys must be exactly: lines, text, script_hint, confidence, warnings. "
    "No extra keys. text must equal \"\\n\".join(lines). "
    "script_hint must be one of insular_old_english, latin_medieval, unknown. "
    "Use only ? and […] for uncertainty."
)

_DEGENERATE_REPEAT_MIN_LEN = 8
_DEGENERATE_REPEAT_THRESHOLD = 2


def _repeat_signature(text: str) -> str:
    compact = re.sub(r"\s+", " ", str(text or "").strip().lower())
    compact = re.sub(r"^[\d\W_]+", "", compact)
    compact = re.sub(r"[\W_]+", " ", compact)
    return compact.strip()


def _is_degenerate_repeat(signature: str, seen_count: int) -> bool:
    return bool(signature) and len(signature) >= _DEGENERATE_REPEAT_MIN_LEN and seen_count >= (_DEGENERATE_REPEAT_THRESHOLD - 1)


def _suppress_degenerate_backend_result(
    backend_result,
    *,
    signature: str,
    prior_count: int,
):
    raw_metadata = dict(backend_result.raw_metadata or {})
    raw_metadata["suppressed_text"] = backend_result.text
    raw_metadata["degenerate_repeat_signature"] = signature[:160]
    raw_metadata["degenerate_repeat_prior_count"] = prior_count
    warnings = list(raw_metadata.get("warnings") or [])
    warnings.append("DEGENERATE_REPEAT_OUTPUT")
    raw_metadata["warnings"] = list(dict.fromkeys(warnings))
    flags = [str(item) for item in raw_metadata.get("flags", [])]
    flags.append("DEGENERATE_REPEAT_OUTPUT")
    raw_metadata["flags"] = list(dict.fromkeys(flags))
    return backend_result.__class__(
        text="",
        confidence=0.0,
        backend_name=backend_result.backend_name,
        model_name=backend_result.model_name,
        raw_metadata=raw_metadata,
        region_id=backend_result.region_id,
        page_id=backend_result.page_id,
        script_hint=backend_result.script_hint,
    )


class OcrAgent(BaseAgent):
    name = "ocr-agent"

    def __init__(self) -> None:
        self.client = SaiaClient()
        self.upscale_factor = max(
            1,
            int(settings.ocr_crop_upscale or os.getenv("OCR_CROP_UPSCALE", "2") or "2"),
        )

    def run(self, payload: OCRExtractRequest) -> OCRExtractAnyResponse:
        return run_ocr_extraction(payload, client=self.client, upscale_factor=self.upscale_factor)


def _parse_preference_env(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            return []
    return [item.strip() for item in text.split(",") if item.strip()]


def resolve_model_preferences(explicit: Sequence[str] | None = None, prefer_model: str | None = None) -> list[str]:
    _ = explicit
    _ = prefer_model
    return [LOCKED_OCR_MODEL]


def is_vision_model(model_id: str) -> bool:
    key = (model_id or "").lower()
    return any(token in key for token in ("vl", "vision", "internvl", "gemma", "mistral", "omni"))


def choose_models(available_models: Sequence[str], preferred_models: Sequence[str]) -> list[str]:
    by_lower = {model.lower(): model for model in available_models}
    ordered: list[str] = []
    seen: set[str] = set()

    for preferred in preferred_models:
        resolved = by_lower.get(preferred.lower())
        if not resolved:
            continue
        key = resolved.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(resolved)

    if preferred_models:
        return ordered

    if ordered:
        return ordered

    for model in available_models:
        key = model.lower()
        if key in seen:
            continue
        if is_vision_model(model):
            seen.add(key)
            ordered.append(model)

    return ordered


def _latin_lock_from_hint(language_hint: str | None) -> bool:
    hint = str(language_hint or "").lower()
    return any(token in hint for token in ("la", "latin", "insular", "old_english", "old english"))


def _has_strict_schema_error(payload: dict[str, object]) -> bool:
    warnings = {str(item) for item in payload.get("warnings") or []}
    return bool({"INVALID_JSON_RESPONSE", "INVALID_SCHEMA_KEYS"} & warnings)


def _has_script_drift_warning(payload: dict[str, object]) -> bool:
    warnings = {str(item) for item in payload.get("warnings") or []}
    return "SCRIPT_DRIFT" in warnings


def _is_image_too_large_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        ("image size" in text and "exceeds limit" in text)
        or "decompression bomb" in text
        or ("too large" in text and "pixels" in text)
    )


def _call_ocr_model(
    client: SaiaClient,
    model: str,
    tile_b64: str,
    *,
    latin_lock: bool,
) -> dict[str, object]:
    messages = build_ocr_messages(tile_b64)
    response = _chat_completion_with_optional_json_format(client=client, model=model, messages=messages)
    parsed = parse_ocr_json(str(response.get("text") or ""), latin_lock=latin_lock, strict_keys=True)

    needs_repair = _has_strict_schema_error(parsed) or _has_script_drift_warning(parsed)
    if not needs_repair:
        return parsed

    repair_messages = build_ocr_messages(
        tile_b64,
        user_prompt=(
            OCR_JSON_REPAIR_PROMPT
            + (
                " Remove Greek/Cyrillic/CJK drift from output; if uncertain, use ? or […]."
                if _has_script_drift_warning(parsed)
                else ""
            )
        ),
    )
    repair_response = _chat_completion_with_optional_json_format(client=client, model=model, messages=repair_messages)
    repaired = parse_ocr_json(str(repair_response.get("text") or ""), latin_lock=latin_lock, strict_keys=True)

    if _has_strict_schema_error(repaired):
        if not _has_strict_schema_error(parsed):
            chosen = parsed
        else:
            parsed_quality = score_text_quality(str(parsed.get("text") or ""))
            repaired_quality = score_text_quality(str(repaired.get("text") or ""))
            chosen = repaired if repaired_quality > parsed_quality else parsed
    else:
        chosen = repaired

    if _has_script_drift_warning(parsed) or _has_script_drift_warning(repaired):
        warnings = [str(item) for item in (chosen.get("warnings") or [])]
        if "SCRIPT_DRIFT" not in warnings:
            warnings.append("SCRIPT_DRIFT")
        chosen["warnings"] = warnings
    return chosen


def _chat_completion_with_optional_json_format(
    *,
    client: SaiaClient,
    model: str,
    messages: list[dict[str, object]],
) -> dict[str, object]:
    try:
        return client.chat_completion(
            model=model,
            temperature=OCR_DECODING_TEMPERATURE,
            top_p=OCR_DECODING_TOP_P,
            max_tokens=OCR_DECODING_MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=messages,
        )
    except Exception as exc:
        text = str(exc).lower()
        if "response_format" not in text and "json_object" not in text:
            raise
        return client.chat_completion(
            model=model,
            temperature=OCR_DECODING_TEMPERATURE,
            top_p=OCR_DECODING_TOP_P,
            max_tokens=OCR_DECODING_MAX_TOKENS,
            messages=messages,
        )


def build_ocr_messages(tile_b64: str, *, user_prompt: str = PALEO_OCR_USER_PROMPT) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": PALEO_OCR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tile_b64}"}},
            ],
        },
    ]


def _merge_script_hint(tile_results: list[OCRTileResult], merged_text: str) -> str:
    weights: dict[str, float] = {}
    for tile in tile_results:
        hint = tile.script_hint if tile.script_hint in ALLOWED_SCRIPT_HINTS else "unknown"
        if hint == "unknown":
            continue
        weights[hint] = weights.get(hint, 0.0) + max(0.05, tile.confidence)

    if weights:
        return sorted(weights.items(), key=lambda item: item[1], reverse=True)[0][0]

    detected = _detect_script_hint(merged_text)
    return detected if detected in ALLOWED_SCRIPT_HINTS else "unknown"


def _source_sha256(image_b64: str) -> str:
    return hashlib.sha256(decode_image_bytes(image_b64)).hexdigest()


def _iter_region_inputs(payload: OCRExtractRequest) -> list:
    if payload.regions:
        return list(payload.regions)
    if payload.region is not None:
        return [payload.region]
    return []


def _ordered_region_inputs(payload: OCRExtractRequest) -> list:
    regions = _iter_region_inputs(payload)
    if not regions:
        return []
    if any(getattr(region, "reading_order", None) is not None for region in regions):
        return sorted(
            regions,
            key=lambda region: (
                getattr(region, "reading_order", None) is None,
                int(getattr(region, "reading_order", 0) or 0),
            ),
        )
    return regions


# Every Kraken model here resizes a line to 120 px high before reading it, so a
# taller line gains nothing from being doubled first. On e-codices scans (median
# line 218-404 px high) encoding the doubled crops took longer than recognising
# them, and skipping it left CER where it was (7.60% against 7.73% over the
# reference pages of scripts/benchmark_recognizers.py).
_UPSCALE_BELOW_LINE_HEIGHT = 120.0


def _region_upscale(region: Any, upscale_factor: int) -> int:
    """The configured crop upscale for a region shorter than a recogniser line, else 1."""
    if region.bbox_xyxy is not None:
        height = float(region.bbox_xyxy[3]) - float(region.bbox_xyxy[1])
    elif region.polygon:
        ys = [float(point[1]) for point in region.polygon]
        height = max(ys) - min(ys)
    else:
        return upscale_factor
    return upscale_factor if height < _UPSCALE_BELOW_LINE_HEIGHT else 1


def _reorder_attempt_backends_from_samples(
    *,
    plan: OCRBackendPlan,
    payload: OCRExtractRequest,
    runtime: dict[str, Any],
    regions_input: list,
    page: Image.Image,
    upscale_factor: int,
) -> tuple[OCRBackendPlan, dict[int, tuple[str, str, OCRRecognitionMetadata]], dict[int, dict[str, Any]], dict[int, dict[str, str]]]:
    if len(plan.attempt_backends) < 2 or not settings.ocr_backend_election:
        return plan, {}, defaultdict(dict), defaultdict(dict)

    explicit_backend = str(payload.options.backend or settings.ocr_backend_default or "auto").strip().lower()
    if explicit_backend != "auto":
        return plan, {}, defaultdict(dict), defaultdict(dict)

    sample_count = min(6, len(regions_input))
    if sample_count < 2:
        return plan, {}, defaultdict(dict), defaultdict(dict)

    prepared_regions: dict[int, tuple[str, str, OCRRecognitionMetadata]] = {}
    prefetched_results: dict[int, dict[str, Any]] = defaultdict(dict)
    prefetched_errors: dict[int, dict[str, str]] = defaultdict(dict)
    backend_scores: dict[str, list[float]] = defaultdict(list)

    for index in range(sample_count):
        region = regions_input[index]
        _crop = crop_page_region(page, region, upscale_factor=_region_upscale(region, upscale_factor))
        region_id, crop_b64 = _crop.region_id, _crop.crop_b64
        metadata = OCRRecognitionMetadata(
            page_id=payload.page_id,
            image_id=payload.image_id,
            region_id=region_id,
            label=getattr(region, "label", None),
            bbox_xyxy=list(region.bbox_xyxy) if region.bbox_xyxy is not None else None,
            polygon=[list(point) for point in region.polygon] if region.polygon is not None else None,
            language_hint=payload.options.language_hint or (payload.metadata.language if payload.metadata else None),
            year=payload.metadata.year if payload.metadata else None,
            place_or_origin=payload.metadata.place_or_origin if payload.metadata else None,
            script_family=payload.metadata.script_family if payload.metadata else None,
            document_type=payload.metadata.document_type if payload.metadata else None,
            notes=payload.metadata.notes if payload.metadata else None,
            crop_box=_crop.crop_box,
            crop_upscale=_crop.upscale_factor,
        )
        prepared_regions[index] = (region_id, crop_b64, metadata)
        for backend_id in plan.attempt_backends:
            try:
                backend_result = runtime[backend_id].recognize(crop_b64, metadata)
                prefetched_results[index][backend_id] = backend_result
                backend_scores[backend_id].append(
                    _region_quality_value(
                        backend_result.text,
                        backend_result.confidence,
                        payload.options.language_hint,
                    )
                )
            except OCRBackendError as exc:
                prefetched_errors[index][backend_id] = str(exc)

    ranked_backends = sorted(
        plan.attempt_backends,
        key=lambda backend_id: (
            -(sum(backend_scores.get(backend_id, [])) / max(1, len(backend_scores.get(backend_id, [])))),
            plan.attempt_backends.index(backend_id),
        ),
    )
    reordered_plan = OCRBackendPlan(
        primary_backend=ranked_backends[0],
        attempt_backends=tuple(ranked_backends),
        comparison_backends=plan.comparison_backends,
    )
    return reordered_plan, prepared_regions, prefetched_results, prefetched_errors


def _run_segmented_ocr_extraction(
    payload: OCRExtractRequest,
    *,
    client: SaiaClient,
    upscale_factor: int,
) -> OCRExtractAnyResponse:
    regions_input = _ordered_region_inputs(payload)
    if not regions_input:
        raise OCRAgentError("No OCR regions were provided for segmented extraction.")

    source_b64 = str(payload.image_b64 or payload.cropped_image_b64 or "").strip()
    # Decoded once: every region is cropped from the same page.
    page = decode_page(source_b64)
    plan = select_backend_plan(
        explicit_backend=payload.options.backend or settings.ocr_backend_default,
        language_hint=payload.options.language_hint,
        compare_backends=payload.options.compare_backends,
    )
    runtime = build_backend_runtime(
        [*plan.attempt_backends, *plan.comparison_backends],
        saia_client=client,
        quality_floor=payload.options.quality_floor,
        max_fallbacks=payload.options.max_fallbacks,
    )
    plan, prefetched_regions, prefetched_results, prefetched_errors = _reorder_attempt_backends_from_samples(
        plan=plan,
        payload=payload,
        runtime=runtime,
        regions_input=regions_input,
        page=page,
        upscale_factor=upscale_factor,
    )

    region_results: list[OCRRegionResult] = []
    comparison_results: list[OCRComparisonResult] = []
    merged_lines: list[str] = []
    warnings: list[str] = []
    fallback_records: list[OCRFallback] = []
    fallbacks_used: list[str] = []
    script_hints: list[str] = []
    confidences: list[float] = []
    selected_backend_names: list[str] = []
    selected_model_names: list[str] = []
    final_model = plan.primary_backend
    backend_repeat_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for index, region in enumerate(regions_input):
        if index in prefetched_regions:
            region_id, crop_b64, metadata = prefetched_regions[index]
        else:
            _crop = crop_page_region(page, region, upscale_factor=_region_upscale(region, upscale_factor))
            region_id, crop_b64 = _crop.region_id, _crop.crop_b64
            metadata = OCRRecognitionMetadata(
                page_id=payload.page_id,
                image_id=payload.image_id,
                region_id=region_id,
                label=getattr(region, "label", None),
                bbox_xyxy=list(region.bbox_xyxy) if region.bbox_xyxy is not None else None,
                polygon=[list(point) for point in region.polygon] if region.polygon is not None else None,
                language_hint=payload.options.language_hint or (payload.metadata.language if payload.metadata else None),
                year=payload.metadata.year if payload.metadata else None,
                place_or_origin=payload.metadata.place_or_origin if payload.metadata else None,
                script_family=payload.metadata.script_family if payload.metadata else None,
                document_type=payload.metadata.document_type if payload.metadata else None,
                notes=payload.metadata.notes if payload.metadata else None,
                crop_box=_crop.crop_box,
                crop_upscale=_crop.upscale_factor,
            )

        selected_backend_id: str | None = None
        selected_result = None
        selected_quality = -1.0
        attempt_cache: dict[str, Any] = {}
        attempt_errors: dict[str, str] = {}
        region_flags: list[str] = []

        for backend_id in plan.attempt_backends:
            backend = runtime[backend_id]
            if index in prefetched_results and backend_id in prefetched_results[index]:
                backend_result = prefetched_results[index][backend_id]
            elif index in prefetched_errors and backend_id in prefetched_errors[index]:
                attempt_errors[backend_id] = prefetched_errors[index][backend_id]
                fallback_records.append(OCRFallback(model=backend_id, reason=f"BACKEND_ERROR:{prefetched_errors[index][backend_id]}"))
                fallbacks_used.append(backend_id)
                region_flags.append(f"BACKEND_ERROR:{backend_id}")
                continue
            else:
                try:
                    backend_result = backend.recognize(crop_b64, metadata)
                except OCRBackendError as exc:
                    attempt_errors[backend_id] = str(exc)
                    fallback_records.append(OCRFallback(model=backend_id, reason=f"BACKEND_ERROR:{exc}"))
                    fallbacks_used.append(backend_id)
                    region_flags.append(f"BACKEND_ERROR:{backend_id}")
                    continue

            signature = _repeat_signature(backend_result.text)
            prior_count = backend_repeat_counts[backend_id][signature] if signature else 0
            if _is_degenerate_repeat(signature, prior_count):
                backend_result = _suppress_degenerate_backend_result(
                    backend_result,
                    signature=signature,
                    prior_count=prior_count,
                )
                fallback_records.append(
                    OCRFallback(model=backend_id, reason=f"DEGENERATE_REPEAT:{signature[:48]}")
                )
                fallbacks_used.append(backend_id)
                region_flags.append(f"DEGENERATE_REPEAT:{backend_id}")
            if signature:
                backend_repeat_counts[backend_id][signature] += 1

            attempt_cache[backend_id] = backend_result
            current_quality = _region_quality_value(
                backend_result.text,
                backend_result.confidence,
                payload.options.language_hint,
            )
            if selected_result is None or current_quality > selected_quality:
                selected_result = backend_result
                selected_backend_id = backend_id
                selected_quality = current_quality

            # The quality score is not comparable across models, so by default it
            # only reports: the first backend to read the line keeps it.
            if backend_result.text.strip() and (
                not settings.ocr_backend_election or current_quality >= payload.options.quality_floor
            ):
                break

            if backend_id != plan.attempt_backends[-1]:
                fallback_records.append(OCRFallback(model=backend_id, reason=f"LOW_QUALITY:{current_quality:.2f}"))
                fallbacks_used.append(backend_id)
                region_flags.append(f"LOW_BACKEND_QUALITY:{backend_id}:{current_quality:.2f}")

        if selected_result is None:
            comparison_backend_ids = tuple(dict.fromkeys([*plan.attempt_backends, *plan.comparison_backends]))
            for backend_id in comparison_backend_ids:
                if backend_id in attempt_cache:
                    backend_result = attempt_cache[backend_id]
                    comparison_results.append(
                        OCRComparisonResult(
                            page_id=payload.page_id,
                            region_id=region_id,
                            backend_name=backend_result.backend_name,
                            model_name=backend_result.model_name,
                            text=backend_result.text,
                            confidence=backend_result.confidence,
                            selected=False,
                            raw_metadata=dict(backend_result.raw_metadata or {}),
                        )
                    )
                    continue
                if backend_id in attempt_errors:
                    comparison_results.append(
                        OCRComparisonResult(
                            page_id=payload.page_id,
                            region_id=region_id,
                            backend_name=backend_id,
                            model_name=backend_id,
                            text="",
                            confidence=0.0,
                            selected=False,
                            raw_metadata={"error": attempt_errors[backend_id]},
                        )
                    )
                    continue
                try:
                    backend_result = runtime[backend_id].recognize(crop_b64, metadata)
                except OCRBackendError as exc:
                    attempt_errors[backend_id] = str(exc)
                    comparison_results.append(
                        OCRComparisonResult(
                            page_id=payload.page_id,
                            region_id=region_id,
                            backend_name=backend_id,
                            model_name=backend_id,
                            text="",
                            confidence=0.0,
                            selected=False,
                            raw_metadata={"error": str(exc)},
                        )
                    )
                    continue

                comparison_results.append(
                    OCRComparisonResult(
                        page_id=payload.page_id,
                        region_id=region_id,
                        backend_name=backend_result.backend_name,
                        model_name=backend_result.model_name,
                        text=backend_result.text,
                        confidence=backend_result.confidence,
                        selected=False,
                        raw_metadata=dict(backend_result.raw_metadata or {}),
                    )
                )
            region_results.append(
                OCRRegionResult(
                    region_id=region_id,
                    text="",
                    quality=0.0,
                    confidence=0.0,
                    flags=["EMPTY_TEXT", *region_flags],
                    bbox_xyxy=list(region.bbox_xyxy) if region.bbox_xyxy is not None else None,
                    polygon=[list(point) for point in region.polygon] if region.polygon is not None else None,
                    label=getattr(region, "label", None),
                    reading_order=getattr(region, "reading_order", index),
                    backend_name=selected_backend_id,
                    model_name=None,
                    raw_metadata={"backend_errors": dict(attempt_errors)} if attempt_errors else None,
                )
            )
            warnings.append(f"EMPTY_TEXT:{region_id}")
            continue

        final_model = selected_result.model_name
        selected_backend_names.append(selected_result.backend_name)
        selected_model_names.append(selected_result.model_name)
        if selected_result.script_hint:
            script_hints.append(selected_result.script_hint)
        if selected_result.confidence is not None:
            confidences.append(float(selected_result.confidence))
        if selected_result.text.strip():
            merged_lines.extend([line for line in selected_result.text.splitlines() if line.strip()])

        raw_meta = dict(selected_result.raw_metadata or {})
        raw_warnings = list(raw_meta.get("warnings") or [])
        warnings.extend(f"{item}:{region_id}" for item in raw_warnings)
        raw_flags = [str(item) for item in raw_meta.get("flags", [])]
        warnings.extend(f"{item}:{region_id}" for item in raw_flags)
        region_results.append(
            OCRRegionResult(
                region_id=region_id,
                text=selected_result.text,
                quality=selected_quality,
                confidence=selected_result.confidence,
                flags=list(dict.fromkeys([*region_flags, *raw_flags])),
                bbox_xyxy=list(region.bbox_xyxy) if region.bbox_xyxy is not None else None,
                polygon=[list(point) for point in region.polygon] if region.polygon is not None else None,
                label=getattr(region, "label", None),
                reading_order=getattr(region, "reading_order", index),
                backend_name=selected_result.backend_name,
                model_name=selected_result.model_name,
                raw_metadata=raw_meta,
            )
        )
        comparison_results.append(
            OCRComparisonResult(
                page_id=payload.page_id,
                region_id=region_id,
                backend_name=selected_result.backend_name,
                model_name=selected_result.model_name,
                text=selected_result.text,
                confidence=selected_result.confidence,
                selected=True,
                raw_metadata=raw_meta,
            )
        )

        for backend_id in plan.comparison_backends:
            if backend_id in attempt_cache:
                backend_result = attempt_cache[backend_id]
            else:
                try:
                    backend_result = runtime[backend_id].recognize(crop_b64, metadata)
                except OCRBackendError as exc:
                    comparison_results.append(
                        OCRComparisonResult(
                            page_id=payload.page_id,
                            region_id=region_id,
                            backend_name=backend_id,
                            model_name=backend_id,
                            text="",
                            confidence=0.0,
                            selected=False,
                            raw_metadata={"error": str(exc)},
                        )
                    )
                    continue

            comparison_results.append(
                OCRComparisonResult(
                    page_id=payload.page_id,
                    region_id=region_id,
                    backend_name=backend_result.backend_name,
                    model_name=backend_result.model_name,
                    text=backend_result.text,
                    confidence=backend_result.confidence,
                    selected=False,
                    raw_metadata=dict(backend_result.raw_metadata or {}),
                )
            )

    merged_text = "\n".join(merged_lines).strip()
    merged_confidence = round((sum(confidences) / len(confidences)) if confidences else 0.0, 4)
    merged_script_hint = "unknown"
    final_backend = Counter(selected_backend_names).most_common(1)[0][0] if selected_backend_names else plan.primary_backend
    if selected_model_names:
        final_model = Counter(selected_model_names).most_common(1)[0][0]
    if script_hints:
        merged_script_hint = Counter(script_hints).most_common(1)[0][0]
    if merged_script_hint not in ALLOWED_SCRIPT_HINTS:
        merged_script_hint = _detect_script_hint(merged_text)
    if merged_script_hint not in ALLOWED_SCRIPT_HINTS:
        merged_script_hint = "unknown"
    if merged_script_hint == "unknown":
        fallback_hint = _detect_script_hint(merged_text)
        if fallback_hint in ALLOWED_SCRIPT_HINTS:
            merged_script_hint = fallback_hint

    raw_ocr = OCRRawOCRPayload(
        lines=merged_lines,
        text=merged_text,
        script_hint=merged_script_hint if merged_script_hint in ALLOWED_SCRIPT_HINTS else "unknown",
        confidence=merged_confidence,
        warnings=sorted(set(warnings)),
    )

    final_text = raw_ocr.text
    should_proofread = payload.options.apply_proofread and final_backend == "saia"
    if should_proofread and raw_ocr.text:
        proofreader = OcrProofreaderAgent(client=client, model_override=LOCKED_OCR_MODEL)
        final_text = proofreader.proofread(raw_ocr.text, raw_ocr.script_hint)
        if raw_ocr.text and not final_text:
            final_text = raw_ocr.text

    source_image_bytes = decode_image_bytes(source_b64)
    evidence_id = sha256_bytes(source_image_bytes)
    image_ref = str(payload.image_id or payload.page_id or "inline_image")
    evidence_record = OcrEvidenceRecord(
        created_at=now_iso(),
        image_sha256=evidence_id,
        image_ref=image_ref,
        model=final_model,
        decoding={
            "temperature": OCR_DECODING_TEMPERATURE,
            "top_p": OCR_DECODING_TOP_P,
            "max_tokens": OCR_DECODING_MAX_TOKENS,
        },
        prompt_version=OCR_PROMPT_VERSION,
        pipeline_version=PIPELINE_VERSION,
        script_hint=raw_ocr.script_hint,
        confidence=float(raw_ocr.confidence),
        warnings=list(raw_ocr.warnings),
        raw_ocr_text=raw_ocr.text,
        final_text=final_text or None,
        final_changes=_summarize_final_changes(raw_ocr.text, final_text),
    )
    try:
        write_ocr_evidence_jsonl(evidence_record)
    except Exception:
        pass

    if payload.mode == "simple":
        return OCRExtractSimpleResponse(
            text=final_text,
            script_hint=raw_ocr.script_hint,
            evidence_id=evidence_id,
            is_evidence=True,
            is_verified=False,
        )

    status: str = "FULL"
    if not final_text:
        status = "FAILED"
    elif raw_ocr.warnings or fallback_records or any(item.flags for item in region_results):
        status = "PARTIAL"

    provenance = OCRProvenance(
        crop_sha256=_source_sha256(source_b64),
        prompt_version=PROMPT_VERSION,
        agent_version=AGENT_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    return OCRExtractResponse(
        status=status,  # type: ignore[arg-type]
        model=final_model,
        ocr_backend=final_backend,  # type: ignore[arg-type]
        fallbacksUsed=list(dict.fromkeys(fallbacks_used)),
        warnings=sorted(set(warnings)),
        text=final_text,
        script_hint=raw_ocr.script_hint,
        final_text=final_text,
        page_id=payload.page_id,
        image_id=payload.image_id,
        fallbacks=fallback_records,
        regions=region_results,
        provenance=provenance,
        raw_ocr=raw_ocr,
        comparison_results=comparison_results,
        evidence_id=evidence_id,
        is_evidence=True,
        is_verified=False,
    )


def _ocr_tile_with_model_candidates(
    *,
    tile: OCRTile,
    saia: SaiaClient,
    selected_models: list[str],
    quality_floor: float,
    latin_lock: bool,
) -> tuple[OCRTileResult, list[OCRFallback], list[str]]:
    fallback_records: list[OCRFallback] = []
    fallbacks_used: list[str] = []

    best_lines: list[str] = []
    best_text = ""
    best_quality = -1.0
    best_flags: list[str] = []
    best_model = selected_models[0]
    best_script_hint = "unknown"
    best_confidence = 0.0
    best_warnings: list[str] = []

    for idx, model in enumerate(selected_models):
        current_tile = tile
        resize_retry_count = 0
        resized_after_limit = False
        ocr_payload: dict[str, object] | None = None

        while True:
            try:
                ocr_payload = _call_ocr_model(saia, model, current_tile.image_b64, latin_lock=latin_lock)
                break
            except Exception as exc:
                if _is_image_too_large_error(exc) and resize_retry_count < OCR_IMAGE_SIZE_RETRY_LIMIT:
                    shrunk_tile = _downscale_tile_for_retry(current_tile)
                    if shrunk_tile is not None:
                        current_tile = shrunk_tile
                        resize_retry_count += 1
                        resized_after_limit = True
                        continue

                if idx + 1 < len(selected_models):
                    reason = "MODEL_NOT_FOUND" if is_model_not_found_error(exc) else f"MODEL_ERROR:{exc}"
                    fallback_records.append(OCRFallback(model=model, reason=reason))
                    fallbacks_used.append(model)
                    break
                raise OCRAgentError(f"OCR failed on model {model}: {exc}") from exc

        if ocr_payload is None:
            continue

        lines = [str(item) for item in (ocr_payload.get("lines") or []) if str(item).strip()]
        text = str(ocr_payload.get("text") or "").strip()
        if not text and lines:
            text = "\n".join(lines)

        quality = score_text_quality(text)
        model_warnings = [str(item) for item in (ocr_payload.get("warnings") or []) if str(item).strip()]
        if tile.resized_for_limits or current_tile.resized_for_limits:
            model_warnings.append("auto_resized_for_limit")
        if resized_after_limit:
            model_warnings.append("auto_retry_downscale")
        model_warnings = list(dict.fromkeys(model_warnings))
        model_confidence = float(ocr_payload.get("confidence") or 0.0)
        model_script_hint = str(ocr_payload.get("script_hint") or "unknown").lower()
        non_latin_ratio = float(ocr_payload.get("non_latin_ratio") or 0.0)

        flags: list[str] = []
        if tile.resized_for_limits or current_tile.resized_for_limits:
            flags.append("AUTO_RESIZED_FOR_LIMIT")
        if resized_after_limit:
            flags.append("AUTO_RETRY_DOWNSCALE")
        if not text:
            flags.append("EMPTY_TEXT")
        if quality < quality_floor:
            flags.append("LOW_TEXT_QUALITY")
        if "INVALID_JSON_RESPONSE" in model_warnings or "INVALID_SCHEMA_KEYS" in model_warnings:
            flags.append("INVALID_JSON_RESPONSE")
        if "meta_text_removed" in model_warnings:
            flags.append("META_TEXT_DETECTED")
            quality *= 0.1
        if "pattern_junk_removed" in model_warnings:
            flags.append("PATTERN_JUNK_REMOVED")
            quality *= 0.6
        if non_latin_ratio > 0.0 or _contains_script_drift(text):
            flags.append("NON_LATIN_CHARS_DETECTED")
            quality *= 0.2
        if "SCRIPT_DRIFT" in model_warnings:
            flags.append("SCRIPT_DRIFT")
            quality *= 0.2
        if latin_lock and model_script_hint == "unknown" and _contains_script_drift(text):
            flags.append("SCRIPT_MISMATCH")
            quality *= 0.1
        if model_confidence < 0.35 and text:
            flags.append("LOW_MODEL_CONFIDENCE")
            quality *= 0.6
        if text:
            uncertainty_ratio = (text.count("?") + text.count("[…]")) / max(1, len(text))
            if uncertainty_ratio > 0.35:
                flags.append("HIGH_UNCERTAINTY")
                quality *= 0.7

        if quality > best_quality:
            best_lines = lines
            best_text = text
            best_quality = quality
            best_flags = list(flags)
            best_model = model
            best_script_hint = model_script_hint if model_script_hint in ALLOWED_SCRIPT_HINTS else "unknown"
            best_confidence = max(0.0, min(1.0, model_confidence))
            best_warnings = list(model_warnings)

        force_retry = any(
            flag in {"INVALID_JSON_RESPONSE", "NON_LATIN_CHARS_DETECTED", "SCRIPT_MISMATCH", "SCRIPT_DRIFT"}
            for flag in flags
        )
        if quality >= quality_floor and not force_retry:
            break

        if idx + 1 < len(selected_models):
            fallback_records.append(OCRFallback(model=model, reason=f"LOW_QUALITY:{quality:.2f}"))
            fallbacks_used.append(model)

    if {"NON_LATIN_CHARS_DETECTED", "SCRIPT_MISMATCH", "SCRIPT_DRIFT"} & set(best_flags):
        best_lines = []
        best_text = ""
        best_flags.append("REGION_TEXT_DROPPED")

    return (
        OCRTileResult(
            tile_id=tile.tile_id,
            bbox_xyxy=tile.bbox_xyxy,
            lines=best_lines,
            text=best_text,
            confidence=round(max(0.0, min(1.0, best_confidence)), 4),
            script_hint=best_script_hint,
            warnings=best_warnings,
            flags=best_flags,
            model=best_model,
        ),
        fallback_records,
        fallbacks_used,
    )


def run_ocr_extraction(
    payload: OCRExtractRequest,
    *,
    client: SaiaClient | None = None,
    upscale_factor: int = 2,
) -> OCRExtractAnyResponse:
    saia = client or SaiaClient()

    if _iter_region_inputs(payload):
        return _run_segmented_ocr_extraction(payload, client=saia, upscale_factor=upscale_factor)

    available_models = saia.list_models()
    preferred_models = resolve_model_preferences(
        explicit=payload.options.model_preference,
        prefer_model=payload.prefer_model,
    )
    candidate_models = choose_models(available_models, preferred_models)
    if not candidate_models:
        raise OCRAgentError("No image-capable model is available from SAIA /models for OCR extraction.")

    max_attempts = min(len(candidate_models), 1 + payload.options.max_fallbacks)
    selected_models = candidate_models[:max_attempts]

    tiles, source_b64 = _make_tiles_from_payload(payload, upscale_factor=upscale_factor)
    latin_lock = _latin_lock_from_hint(payload.options.language_hint)

    tile_results: list[OCRTileResult] = []
    fallback_records: list[OCRFallback] = []
    fallbacks_used: list[str] = []

    for tile in tiles:
        tile_result, tile_fallbacks, tile_fallback_models = _ocr_tile_with_model_candidates(
            tile=tile,
            saia=saia,
            selected_models=selected_models,
            quality_floor=payload.options.quality_floor,
            latin_lock=latin_lock,
        )
        tile_results.append(tile_result)
        fallback_records.extend(tile_fallbacks)
        fallbacks_used.extend(tile_fallback_models)

    merged_lines, deduped_count = _merge_tile_lines(tile_results)
    merged_text = "\n".join(merged_lines).strip()
    merged_script_hint = _merge_script_hint(tile_results, merged_text)
    merged_confidence = round(_merge_confidence(tile_results), 4)

    warnings: list[str] = []
    if deduped_count > 0:
        warnings.append("DEDUPLICATED_OVERLAPS")
    for tile in tile_results:
        for warning in tile.warnings:
            warnings.append(f"{warning}:{tile.tile_id}")
        if "LOW_TEXT_QUALITY" in tile.flags:
            warnings.append(f"LOW_TEXT_QUALITY:{tile.tile_id}")
        if "INVALID_JSON_RESPONSE" in tile.flags:
            warnings.append(f"INVALID_JSON_RESPONSE:{tile.tile_id}")
        if "NON_LATIN_CHARS_DETECTED" in tile.flags:
            warnings.append(f"NON_LATIN_CHARS_DETECTED:{tile.tile_id}")
        if "SCRIPT_DRIFT" in tile.flags:
            warnings.append(f"SCRIPT_DRIFT:{tile.tile_id}")
        if "SCRIPT_MISMATCH" in tile.flags:
            warnings.append(f"SCRIPT_MISMATCH:{tile.tile_id}")
        if "META_TEXT_DETECTED" in tile.flags:
            warnings.append(f"META_TEXT_DETECTED:{tile.tile_id}")
        if "LOW_MODEL_CONFIDENCE" in tile.flags:
            warnings.append(f"LOW_MODEL_CONFIDENCE:{tile.tile_id}")

    if not merged_text:
        warnings.append("NO_READABLE_TEXT")

    raw_ocr = OCRRawOCRPayload(
        lines=merged_lines,
        text=merged_text,
        script_hint=merged_script_hint if merged_script_hint in ALLOWED_SCRIPT_HINTS else "unknown",
        confidence=merged_confidence,
        warnings=sorted(set(warnings)),
    )

    model_counts = Counter([tile.model for tile in tile_results if tile.model])
    final_model = model_counts.most_common(1)[0][0] if model_counts else selected_models[0]

    final_text = raw_ocr.text
    if payload.options.apply_proofread and raw_ocr.text:
        proofreader = OcrProofreaderAgent(client=saia, model_override=final_model)
        final_text = proofreader.proofread(raw_ocr.text, raw_ocr.script_hint)
        if raw_ocr.text and not final_text:
            final_text = raw_ocr.text

    source_image_bytes = decode_image_bytes(source_b64)
    evidence_id = sha256_bytes(source_image_bytes)
    image_ref = str(payload.image_id or payload.page_id or "inline_image")
    evidence_record = OcrEvidenceRecord(
        created_at=now_iso(),
        image_sha256=evidence_id,
        image_ref=image_ref,
        model=final_model,
        decoding={
            "temperature": OCR_DECODING_TEMPERATURE,
            "top_p": OCR_DECODING_TOP_P,
            "max_tokens": OCR_DECODING_MAX_TOKENS,
        },
        prompt_version=OCR_PROMPT_VERSION,
        pipeline_version=PIPELINE_VERSION,
        script_hint=raw_ocr.script_hint,
        confidence=float(raw_ocr.confidence),
        warnings=list(raw_ocr.warnings),
        raw_ocr_text=raw_ocr.text,
        final_text=final_text or None,
        final_changes=_summarize_final_changes(raw_ocr.text, final_text),
    )
    try:
        write_ocr_evidence_jsonl(evidence_record)
    except Exception:
        pass

    if payload.mode == "simple":
        return OCRExtractSimpleResponse(
            text=final_text,
            script_hint=raw_ocr.script_hint,
            evidence_id=evidence_id,
            is_evidence=True,
            is_verified=False,
        )

    status: str = "FULL"
    if not final_text:
        status = "FAILED"
    elif raw_ocr.warnings or fallback_records or any(tile.flags for tile in tile_results):
        status = "PARTIAL"

    provenance = OCRProvenance(
        crop_sha256=_source_sha256(source_b64),
        prompt_version=PROMPT_VERSION,
        agent_version=AGENT_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    regions = [
        OCRRegionResult(
            region_id=tile.tile_id,
            text=tile.text,
            quality=round(max(0.0, min(1.0, tile.confidence)), 4),
            flags=tile.flags,
            bbox_xyxy=tile.bbox_xyxy,
            polygon=None,
        )
        for tile in tile_results
    ]

    return OCRExtractResponse(
        status=status,  # type: ignore[arg-type]
        model=final_model,
        fallbacksUsed=list(dict.fromkeys(fallbacks_used)),
        warnings=sorted(set(warnings)),
        text=final_text,
        script_hint=raw_ocr.script_hint,
        final_text=final_text,
        page_id=payload.page_id,
        image_id=payload.image_id,
        fallbacks=fallback_records,
        regions=regions,
        provenance=provenance,
        raw_ocr=raw_ocr,
        evidence_id=evidence_id,
        is_evidence=True,
        is_verified=False,
    )
