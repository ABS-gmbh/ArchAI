from __future__ import annotations

import asyncio
import base64
from collections import defaultdict
import hashlib
import io
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.agents.crop_agent import decode_image_bytes
from app.agents.ocr_agent import OCRAgentError
from app.agents.ocr_proofreader_agent import OcrProofreaderAgent
from app.agents.saia_ocr_agent import SaiaOCRAgent
from app.agents.segmentation_agent import run_single_segmentation
from app.core.constants import FINAL_CLASSES
from app.db.pipeline_db import (
    count_chunks,
    count_entity_mentions,
    create_run,
    get_run,
    insert_ocr_benchmark_reference,
    insert_entity_mentions,
    list_chunks,
    list_entity_mentions,
    list_events,
    log_event,
    table_view_for_chunks,
    table_view_for_entity_attempts,
    table_view_for_entity_candidates,
    table_view_for_entity_decisions,
    table_view_for_entity_mentions,
    table_view_for_events,
    table_view_for_ocr_backend_results,
    table_view_for_run,
    update_run_fields,
)
from app.routers.predict import _prepare_image_for_segmentation, rescale_coco_to_original
from app.schemas.agents_ocr import OCRComparisonSummary, OCRExtractAnyResponse, OCRExtractOptions, OCRExtractRequest
from app.schemas.agents_ocr import (
    OCRExtractResponse,
    OCRRegionInput,
    SaiaFullPageExtractRequest,
    SaiaFullPageExtractResponse,
    SaiaOCRLocationSuggestion,
    SaiaOCRRequest,
    SaiaOCRResponse,
)
from app.config import settings as _app_settings
from app.services.glm_ollama_ocr import (
    GlmOllamaOcrError,
    GlmOllamaOcrResult,
    build_glm_ocr_prompt,
    run_glm_ollama_ocr,
)
from app.services.test_ocr_overrides import get_test_ocr_fixture, get_test_ocr_override
from app.services.saia_client import SaiaConfigError
from app.services.lexicon_trust import best_fit_plausibility as _lexical_plausibility
from app.services.ocr_quality import (
    compute_quality_report,
    check_mention_recall,
    OCRQualityReport,
    EffectiveQuality,
    build_effective_quality,
    compute_cross_pass_stability,
    apply_uncertainty_markers,
)
from app.services.ocr_quality_config import (
    CROSS_PASS_STABILITY_MIN,
    LEADING_FRAG_HARD_LIMIT,
    MAX_OCR_ATTEMPTS as _MAX_OCR_ATTEMPTS,
    SEAM_FRAG_HARD_LIMIT,
    frag_gate_value,
)
from app.services.pipeline_hardening import (
    enforce_quality_gates,
    extract_high_recall_mentions,
    format_gate_report,
    proofreading_quality_guard,
)
from app.services.seam_strategies import (
    TilingPlan,
    default_plan_from_suggestions,
    is_noop_retry,
    plan_to_suggestions,
    select_retry_strategy,
)
from app.db.pipeline_db import (
    insert_ocr_quality_report,
    insert_ocr_attempt,
    list_ocr_attempts,
)
from app.services.language_detection import (
    _detect_language_metadata,
    _normalize_detected_language,
    _quality_language,
)
from app.services.mention_extraction import (
    _CANONICAL_WORKS,
    _PLACE_ANCHORS,
    _SALVAGE_TOKEN_RE,
    _SALVAGE_TRIGGERS,
    _TITLE_ANCHORS,
)
from app.services.mention_report import (
    _build_consolidated_report,
    _build_mention_extraction_report,
)
from app.services.trace_pipeline import (
    _assign_chunk_id_for_span,
    _auto_index_run,
    _get_ocr_agent,
    _persist_deferred_authority_links,
    _record_search_decision,
    _run_authority_linking_stage,
    _run_full_page_post_ocr_pipeline,
    _run_segmented_trace_pipeline,
    _run_trace_analysis,
)

router = APIRouter(tags=["ocr"])
_saia_ocr_agent_instance: SaiaOCRAgent | None = None


def _get_saia_ocr_agent() -> SaiaOCRAgent:
    global _saia_ocr_agent_instance
    if _saia_ocr_agent_instance is None:
        _saia_ocr_agent_instance = SaiaOCRAgent()
    return _saia_ocr_agent_instance


_TEXT_LABEL_INCLUDE_TOKENS = (
    "main script",
    "main text",
    "main script block",
    "variant script",
    "plain initial",
    "historiated",
    "inhabited",
    "embellished",
    "dropcapital",
    "defaultline",
)
_TEXT_LABEL_EXCLUDE_TOKENS = (
    "border",
    "column",
    "table",
    "diagram",
    "illustration",
    "graphic",
    "music",
    "zone",
    "gloss",
    "header",
    "catchword",
    "page number",
    "quire",
)

_PARTIAL_WARNING_KEYS = {
    "LOW_TEXT_QUALITY",
    "INVALID_OCR_JSON",
    "JSON_REPAIR_RETRY",
    "SCRIPT_DRIFT",
    "HALLUCINATION_SUSPECTED",
    "OCR_FAILED_ALL_MODELS",
    "PROOFREAD_FAILED",
    "PROOFREAD_EMPTY_RESULT",
    "PROOFREAD_SKIPPED_LOW_QUALITY",
    "SEGMENTATION_FAILED",
}
def _levenshtein(s: str, t: str) -> int:
    """Compute Levenshtein edit distance (no external deps)."""
    n, m = len(s), len(t)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def _has_entity_cues(base_text: str) -> bool:
    """Return True if the text has any cues suggesting named entities exist.

    Cues: trigger words, title anchors, place anchors, or any token >=8 chars
    not in stopwords.
    """
    value = str(base_text or "").lower()
    if not value.strip():
        return False
    tokens = set(_SALVAGE_TOKEN_RE.findall(value))
    # Check trigger words
    if tokens & set(_SALVAGE_TRIGGERS.keys()):
        return True
    # Check title / place anchors
    if tokens & _TITLE_ANCHORS:
        return True
    if tokens & _PLACE_ANCHORS:
        return True
    # Check canonical works (exact match only for cue detection)
    if tokens & set(_CANONICAL_WORKS.keys()):
        return True
    return False


def _is_french_family_language(value: str | None) -> bool:
    return str(value or "") in {"french", "old_french", "middle_french", "anglo_norman"}


def _is_relevant_text_label(label: str) -> bool:
    key = str(label or "").lower().strip()
    if not key:
        return False
    if any(token in key for token in _TEXT_LABEL_EXCLUDE_TOKENS):
        return False
    return any(token in key for token in _TEXT_LABEL_INCLUDE_TOKENS)


def _extract_location_suggestions(coco: dict[str, Any] | None) -> list[SaiaOCRLocationSuggestion]:
    if not isinstance(coco, dict):
        return []

    categories = coco.get("categories")
    annotations = coco.get("annotations")
    if not isinstance(categories, list) or not isinstance(annotations, list):
        return []

    category_by_id: dict[int, str] = {}
    for category in categories:
        if not isinstance(category, dict):
            continue
        try:
            category_id = int(category.get("id"))
        except Exception:
            continue
        category_name = str(category.get("name") or "").strip()
        if category_name:
            category_by_id[category_id] = category_name

    suggestions: list[SaiaOCRLocationSuggestion] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        try:
            x, y, w, h = [float(bbox[idx]) for idx in range(4)]
        except Exception:
            continue
        if not all(value >= 0 for value in (w, h)):
            continue
        if w < 8 or h < 8:
            continue

        category_name = category_by_id.get(int(annotation.get("category_id") or -1), "")
        if not _is_relevant_text_label(category_name):
            continue

        region_id = str(annotation.get("id") or f"region-{len(suggestions) + 1}")
        suggestions.append(
            SaiaOCRLocationSuggestion(
                region_id=region_id,
                category=category_name,
                bbox_xywh=[x, y, w, h],
            )
        )

    suggestions.sort(key=lambda item: (item.bbox_xywh[1], item.bbox_xywh[0]))
    return suggestions[:80]


def _annotation_polygon(annotation: dict[str, Any]) -> list[list[float]] | None:
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, list) or not segmentation:
        return None
    coords = segmentation[0] if isinstance(segmentation[0], list) else segmentation
    if not isinstance(coords, list) or len(coords) < 6 or len(coords) % 2 != 0:
        return None
    points: list[list[float]] = []
    for idx in range(0, len(coords), 2):
        try:
            points.append([float(coords[idx]), float(coords[idx + 1])])
        except Exception:
            return None
    return points or None


_TextRow = tuple[float, float, str, tuple[float, float, float, float], OCRRegionInput]


def _is_column_like_label(label: str) -> bool:
    key = str(label or "").strip().lower()
    return any(token in key for token in ("column", "mainzone", "main zone", "text zone", "main text area"))


def _bbox_overlap_ratio(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    inter_x1 = max(left[0], right[0])
    inter_y1 = max(left[1], right[1])
    inter_x2 = min(left[2], right[2])
    inter_y2 = min(left[3], right[3])
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    left_area = max(1.0, (left[2] - left[0]) * (left[3] - left[1]))
    return inter_area / left_area


def _cluster_rows_by_columns(
    text_rows: list[_TextRow],
    column_boxes: list[tuple[float, float, float, float]],
) -> list[_TextRow]:
    if not text_rows:
        return []

    assigned: dict[int, list[_TextRow]] = {index: [] for index in range(len(column_boxes))}
    overflow: list[_TextRow] = []

    for row in text_rows:
        bbox = row[3]
        center_x = (bbox[0] + bbox[2]) / 2.0
        best_index = -1
        best_score = -1.0
        for index, column_bbox in enumerate(column_boxes):
            contains_center = column_bbox[0] <= center_x <= column_bbox[2]
            overlap = _bbox_overlap_ratio(bbox, column_bbox)
            score = overlap + (0.25 if contains_center else 0.0)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index >= 0 and best_score > 0.05:
            assigned[best_index].append(row)
        else:
            overflow.append(row)

    ordered_rows: list[_TextRow] = []
    for index, column_bbox in enumerate(column_boxes):
        column_rows = sorted(assigned[index], key=lambda item: (item[3][1], item[3][0]))
        ordered_rows.extend(column_rows)
    ordered_rows.extend(sorted(overflow, key=lambda item: (item[3][1], item[3][0])))
    return ordered_rows


def _cluster_rows_without_columns(text_rows: list[_TextRow]) -> list[_TextRow]:
    if not text_rows:
        return []
    sorted_rows = sorted(text_rows, key=lambda item: ((item[3][0] + item[3][2]) / 2.0, item[3][1]))
    page_width = max((row[3][2] for row in sorted_rows), default=0.0) - min((row[3][0] for row in sorted_rows), default=0.0)
    merge_gap = max(80.0, page_width * 0.08)
    columns: list[dict[str, Any]] = []
    for row in sorted_rows:
        bbox = row[3]
        center_x = (bbox[0] + bbox[2]) / 2.0
        matched = None
        matched_distance = None
        for column in columns:
            overlap = min(bbox[2], column["x2"]) - max(bbox[0], column["x1"])
            if overlap > 0:
                distance = abs(center_x - column["center_x"])
            else:
                distance = max(bbox[0] - column["x2"], column["x1"] - bbox[2], 0.0)
            if distance <= merge_gap and (matched_distance is None or distance < matched_distance):
                matched = column
                matched_distance = distance
        if matched is None:
            columns.append({
                "x1": bbox[0],
                "x2": bbox[2],
                "center_x": center_x,
                "rows": [row],
            })
            continue
        matched["rows"].append(row)
        matched["x1"] = min(matched["x1"], bbox[0])
        matched["x2"] = max(matched["x2"], bbox[2])
        matched["center_x"] = sum((item[3][0] + item[3][2]) / 2.0 for item in matched["rows"]) / len(matched["rows"])

    columns.sort(key=lambda column: column["x1"])
    ordered_rows: list[_TextRow] = []
    for column in columns:
        ordered_rows.extend(sorted(column["rows"], key=lambda item: (item[3][1], item[3][0])))
    return ordered_rows


def _collect_text_rows(
    coco: dict[str, Any] | None,
) -> tuple[list[_TextRow], list[tuple[float, float, float, float]]]:
    """Text-bearing detections and the column (main zone) boxes of a COCO page."""
    if not isinstance(coco, dict):
        return [], []

    categories = coco.get("categories")
    annotations = coco.get("annotations")
    if not isinstance(categories, list) or not isinstance(annotations, list):
        return [], []

    category_by_id: dict[int, str] = {}
    for category in categories:
        if not isinstance(category, dict):
            continue
        try:
            category_id = int(category.get("id"))
        except Exception:
            continue
        category_name = str(category.get("name") or "").strip()
        if category_name:
            category_by_id[category_id] = category_name

    rows: list[_TextRow] = []
    column_boxes: list[tuple[float, float, float, float]] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        try:
            x, y, w, h = [float(bbox[idx]) for idx in range(4)]
        except Exception:
            continue
        if w < 8 or h < 8:
            continue

        category_name = category_by_id.get(int(annotation.get("category_id") or -1), "")
        bbox_xyxy = (x, y, x + w, y + h)
        if _is_column_like_label(category_name):
            column_boxes.append(bbox_xyxy)
            continue
        if not _is_relevant_text_label(category_name):
            continue

        polygon = _annotation_polygon(annotation)
        region = OCRRegionInput(
            region_id=str(annotation.get("id") or f"region-{len(rows) + 1}"),
            bbox_xyxy=[x, y, x + w, y + h],
            polygon=polygon,
            label=category_name or None,
            reading_order=len(rows),
        )
        rows.append((y, x, category_name, bbox_xyxy, region))

    preferred_rows = [
        item for item in rows
        if any(token in item[2].lower() for token in ("line", "main script", "variant script"))
    ]
    column_boxes.sort(key=lambda item: (item[0], item[1]))
    return preferred_rows or rows, column_boxes


def _ordered_regions(
    rows: list[_TextRow],
    column_boxes: list[tuple[float, float, float, float]],
) -> list[OCRRegionInput]:
    ordered = _cluster_rows_by_columns(rows, column_boxes) if column_boxes else _cluster_rows_without_columns(rows)
    return [
        region.model_copy(update={"reading_order": idx})
        for idx, (_y, _x, _label, _bbox, region) in enumerate(ordered)
    ]


def _extract_segmented_regions(coco: dict[str, Any] | None) -> list[OCRRegionInput]:
    rows, column_boxes = _collect_text_rows(coco)
    return _ordered_regions(rows, column_boxes)


# Text detected outside every column box - a library watermark, marginal line
# numbers, a folio number, a pencil note, the edge of the facing page - is kept
# out of the body only while it is a minority of the page's rows. On nine of ten
# real pages it was 0-30% of the rows, none of it the page's own text. On the
# tenth a strip of the facing page shows, its column undetected: 60%. That looks
# the same as a two-page spread with one column missed, whose rows ARE body text,
# so above this share the columns are taken to be incomplete and the old order
# is kept.
_MAX_OUTSIDE_SHARE = 1 / 3
# A row belongs to a column when its centre lies inside the column box grown by
# this share of the box's size. The ordering heuristic above is laxer on purpose -
# any row under a column's x-span joins it - which is how the library watermark at
# the foot of a page used to become that column's last line.
_COLUMN_MARGIN = 0.02


def _row_in_columns(row: _TextRow, column_boxes: list[tuple[float, float, float, float]]) -> bool:
    x1, y1, x2, y2 = row[3]
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    for cx1, cy1, cx2, cy2 in column_boxes:
        margin_x, margin_y = _COLUMN_MARGIN * (cx2 - cx1), _COLUMN_MARGIN * (cy2 - cy1)
        if cx1 - margin_x <= center_x <= cx2 + margin_x and cy1 - margin_y <= center_y <= cy2 + margin_y:
            return True
    return False


def _segmented_region_lanes(
    coco: dict[str, Any] | None,
) -> tuple[list[OCRRegionInput], list[OCRRegionInput]]:
    """Body regions in reading order, and text regions outside every column.

    Without column boxes, or when the rows outside them are too many to be page
    furniture, everything stays in the body in the legacy order.
    """
    rows, column_boxes = _collect_text_rows(coco)
    if not column_boxes:
        return _ordered_regions(rows, column_boxes), []
    inside = [row for row in rows if _row_in_columns(row, column_boxes)]
    outside = [row for row in rows if not _row_in_columns(row, column_boxes)]
    if not outside or len(outside) > _MAX_OUTSIDE_SHARE * len(rows):
        return _ordered_regions(rows, column_boxes), []
    by_position = sorted(outside, key=lambda item: (item[3][1], item[3][0]))
    return (
        _ordered_regions(inside, column_boxes),
        [row[4].model_copy(update={"reading_order": idx}) for idx, row in enumerate(by_position)],
    )


def _regions_from_location_suggestions(suggestions: list[SaiaOCRLocationSuggestion]) -> list[OCRRegionInput]:
    regions: list[OCRRegionInput] = []
    for idx, suggestion in enumerate(suggestions):
        bbox = list(getattr(suggestion, "bbox_xywh", []) or [])
        if len(bbox) < 4:
            continue
        x, y, w, h = [float(item) for item in bbox[:4]]
        regions.append(
            OCRRegionInput(
                region_id=suggestion.region_id or f"region-{idx + 1}",
                bbox_xyxy=[x, y, x + w, y + h],
                label=suggestion.category,
                reading_order=idx,
            )
        )
    return regions


# Ultralytics picks an image loader by file extension. The upload used to be
# written as "ocr_full_page_input", with none, and _prepare_image_for_segmentation
# passes an RGB image within its size limits through untouched - so the detector
# raised "No images or videos found" on every such page and the route fell back to
# OCR without layout. Measured: 9 of 10 real manuscript pages, every RGB upload.
_SEGMENTATION_SUFFIXES = {"JPEG": ".jpg", "MPO": ".jpg", "PNG": ".png", "TIFF": ".tif", "BMP": ".bmp", "WEBP": ".webp"}


def _write_segmentation_input(directory: Path, image_bytes: bytes) -> Path:
    """Write an upload where the layout detector can read it: with a matching suffix."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            suffix = _SEGMENTATION_SUFFIXES.get(str(image.format or "").upper())
            path = directory / f"ocr_full_page_input{suffix or '.png'}"
            if suffix is None:
                # A format the detector cannot load by extension is re-encoded.
                image.convert("RGB").save(path, format="PNG")
                return path
    except (OSError, ValueError):
        # Not an image: _prepare_image_for_segmentation rejects it with a 422.
        path = directory / "ocr_full_page_input"
    path.write_bytes(image_bytes)
    return path


def _segment_page(image_bytes: bytes) -> dict[str, Any]:
    """Run the layout models on an uploaded page; the COCO is in original pixels."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        source_path = _write_segmentation_input(Path(tmp_dir), image_bytes)
        prepared = _prepare_image_for_segmentation(str(source_path))
        coco, _stats = run_single_segmentation(
            prepared.path,
            confidence=0.25,
            iou=0.3,
            selected_classes=FINAL_CLASSES,
            annotated_output_path=str(Path(tmp_dir) / "annotated.jpg"),
        )
    # Crops downstream are taken from the original bytes, so bring the geometry
    # back out of the downscaled segmentation space.
    return rescale_coco_to_original(coco, prepared.scale)


def _run_segmentation_for_suggestions(image_bytes: bytes) -> list[SaiaOCRLocationSuggestion]:
    return _extract_location_suggestions(_segment_page(image_bytes))


def _run_segmentation_for_regions(image_bytes: bytes) -> list[OCRRegionInput]:
    return _extract_segmented_regions(_segment_page(image_bytes))


def _resolve_full_page_status(*, text: str, confidence: float | None, warnings: list[str]) -> str:
    if not str(text or "").strip():
        return "EMPTY"

    upper_warnings = {str(item).upper() for item in warnings}
    if any(key in upper_warnings for key in _PARTIAL_WARNING_KEYS):
        return "PARTIAL"

    if confidence is not None and confidence < 0.4:
        return "PARTIAL"

    return "FULL"


def _latest_completed_stage(events: list[dict[str, Any]]) -> str | None:
    for item in reversed(events):
        if str(item.get("event") or "").upper() == "END":
            return str(item.get("stage") or "")
    return None


def _build_trace_snapshot(run_id: str) -> dict[str, Any] | None:
    run = get_run(run_id)
    if run is None:
        return None
    events = list_events(run_id)
    chunks_preview = list_chunks(run_id, limit=20)
    mentions_preview = list_entity_mentions(run_id, limit=20)
    chunks_count = int(run.get("chunks_count") or count_chunks(run_id))
    mentions_count = int(run.get("mentions_count") or count_entity_mentions(run_id))
    return {
        "run_id": run_id,
        "run": run,
        "events": events,
        "latest_completed_stage": _latest_completed_stage(events),
        "chunks_count": chunks_count,
        "mentions_count": mentions_count,
        "chunks_preview": chunks_preview,
        "top_mentions": mentions_preview,
    }


def _map_extract_script_hint_to_full_page(value: str | None) -> str:
    hint = str(value or "unknown").strip().lower()
    if hint == "latin_medieval":
        return "latin"
    if hint in {"latin", "greek", "cyrillic", "mixed", "unknown"}:
        return hint
    return "unknown"


def _strip_uncertainty_markers(text: str) -> str:
    return (
        str(text or "")
        .replace("[…]", "")
        .replace("[...]", "")
        .replace("…", "")
        .replace("?", "")
    )


def _strip_uncertainty_lines(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        value = re.sub(r"\s{2,}", " ", _strip_uncertainty_markers(line)).strip()
        if value:
            cleaned.append(value)
    return cleaned


def _build_segmented_extract_request(
    payload: SaiaFullPageExtractRequest,
    regions: list[OCRRegionInput],
) -> OCRExtractRequest:
    effective_language_hint = str(
        payload.language_hint
        or (payload.metadata.language if payload.metadata and payload.metadata.language else "unknown")
    )
    return OCRExtractRequest(
        image_id=payload.image_id,
        page_id=payload.page_id,
        image_b64=payload.image_b64,
        regions=regions,
        mode="full",
        options=OCRExtractOptions(
            language_hint=effective_language_hint,
            apply_proofread=payload.apply_proofread,
            backend=payload.ocr_backend,
            compare_backends=list(payload.compare_backends),
        ),
        metadata=payload.metadata,
        benchmark_text=payload.benchmark_text,
        benchmark_source=payload.benchmark_source,
    )


def _build_masked_text_page_b64(image_bytes: bytes, regions: list[OCRRegionInput]) -> str:
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as source:
        buffer = io.BytesIO()
        source.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _glm_region_bbox(region: OCRRegionInput) -> tuple[float, float, float, float] | None:
    if region.bbox_xyxy and len(region.bbox_xyxy) >= 4:
        x1, y1, x2, y2 = [float(value) for value in region.bbox_xyxy[:4]]
        return (x1, y1, x2, y2)
    if region.polygon:
        xs = [float(point[0]) for point in region.polygon]
        ys = [float(point[1]) for point in region.polygon]
        if xs and ys:
            return (min(xs), min(ys), max(xs), max(ys))
    return None


def _group_regions_for_glm_blocks(regions: list[OCRRegionInput]) -> list[list[OCRRegionInput]]:
    annotated: list[tuple[int, float, float, float, float, OCRRegionInput]] = []
    for index, region in enumerate(regions):
        bbox = _glm_region_bbox(region)
        if bbox is None:
            continue
        order = int(region.reading_order if region.reading_order is not None else index)
        annotated.append((order, bbox[0], bbox[1], bbox[2], bbox[3], region))
    if not annotated:
        return []

    groups: list[dict[str, Any]] = []
    for row in sorted(annotated, key=lambda item: (item[1], item[0])):
        _order, x1, _y1, x2, _y2, region = row
        center_x = (x1 + x2) / 2.0
        matched: dict[str, Any] | None = None
        best_score = (-1.0, -1.0)
        for group in groups:
            overlap = max(0.0, min(x2, group["x2"]) - max(x1, group["x1"]))
            contains = group["x1"] <= center_x <= group["x2"]
            distance = abs(center_x - group["center_x"])
            width_ref = max(1.0, x2 - x1, group["x2"] - group["x1"])
            near = distance <= width_ref * 0.75
            if not contains and overlap <= 0.0 and not near:
                continue
            score = (1.0 if contains else 0.0, overlap - distance * 0.01)
            if score > best_score:
                matched = group
                best_score = score
        if matched is None:
            groups.append({"x1": x1, "x2": x2, "center_x": center_x, "regions": [region]})
            continue
        matched["x1"] = min(matched["x1"], x1)
        matched["x2"] = max(matched["x2"], x2)
        matched["center_x"] = (matched["x1"] + matched["x2"]) / 2.0
        matched["regions"].append(region)

    ordered_groups = sorted(groups, key=lambda item: item["x1"])
    return [
        sorted(
            list(group["regions"]),
            key=lambda region: (
                int(region.reading_order if region.reading_order is not None else 10**9),
                (_glm_region_bbox(region) or (0.0, 0.0, 0.0, 0.0))[1],
                (_glm_region_bbox(region) or (0.0, 0.0, 0.0, 0.0))[0],
            ),
        )
        for group in ordered_groups
    ]


def _build_glm_block_images(image_bytes: bytes, regions: list[OCRRegionInput]) -> list[dict[str, Any]]:
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as source:
        base = source.convert("RGB")
    width, height = base.size
    region_groups = _group_regions_for_glm_blocks(regions)
    if not region_groups:
        return [{"region_id": "__glm_full_page__", "image_b64": _build_masked_text_page_b64(image_bytes, regions), "mode": "full_page"}]

    blocks: list[dict[str, Any]] = []
    for index, group in enumerate(region_groups):
        boxes = [_glm_region_bbox(region) for region in group]
        valid_boxes = [bbox for bbox in boxes if bbox is not None]
        if not valid_boxes:
            continue
        x1 = min(bbox[0] for bbox in valid_boxes)
        y1 = min(bbox[1] for bbox in valid_boxes)
        x2 = max(bbox[2] for bbox in valid_boxes)
        y2 = max(bbox[3] for bbox in valid_boxes)
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        pad_x = max(24, int(round(box_w * 0.05)))
        pad_y = max(24, int(round(box_h * 0.08)))
        crop_box = (
            max(0, int(round(x1 - pad_x))),
            max(0, int(round(y1 - pad_y))),
            min(width, int(round(x2 + pad_x))),
            min(height, int(round(y2 + pad_y))),
        )
        buffer = io.BytesIO()
        base.crop(crop_box).save(buffer, format="PNG")
        blocks.append(
            {
                "region_id": f"__glm_block_{index + 1}__",
                "image_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "mode": "column_block" if len(region_groups) > 1 else "page_block",
                "bbox_xyxy": list(crop_box),
            }
        )
    return blocks or [{"region_id": "__glm_full_page__", "image_b64": _build_masked_text_page_b64(image_bytes, regions), "mode": "full_page"}]


def _run_glmocr_full_page_summary(
    payload: SaiaFullPageExtractRequest,
    image_bytes: bytes,
    regions: list[OCRRegionInput],
    *,
    selected: bool,
) -> OCRComparisonSummary:
    from app.services.ocr_backends import GlmOcrBackend, OCRBackendError, OCRRecognitionMetadata

    try:
        backend = GlmOcrBackend()
        block_results: list[Any] = []
        warnings: list[str] = []
        for block in _build_glm_block_images(image_bytes, regions):
            result = backend.recognize(
                block["image_b64"],
                OCRRecognitionMetadata(
                    page_id=payload.page_id,
                    image_id=payload.image_id,
                    region_id=str(block["region_id"]),
                    label=str(block.get("mode") or "glm_block"),
                    bbox_xyxy=list(block.get("bbox_xyxy") or []) or None,
                    language_hint=payload.language_hint or (payload.metadata.language if payload.metadata else None),
                    script_hint_seed=payload.script_hint_seed,
                    year=payload.metadata.year if payload.metadata else None,
                    place_or_origin=payload.metadata.place_or_origin if payload.metadata else None,
                    script_family=payload.metadata.script_family if payload.metadata else None,
                    document_type=payload.metadata.document_type if payload.metadata else None,
                    notes=payload.metadata.notes if payload.metadata else None,
                ),
            )
            block_results.append((block, result))
        raw_text = "\n".join(str(result.text or "").strip() for _block, result in block_results if str(result.text or "").strip()).strip()
        text_value = _strip_uncertainty_markers(raw_text)
        lines_value = _strip_uncertainty_lines(text_value.splitlines())
        text_value = "\n".join(lines_value).strip()
        if not text_value:
            warnings.append("EMPTY_TEXT")
        if len(block_results) > 1:
            warnings.append(f"GLM_BLOCKS:{len(block_results)}")
        notes = ["experimental document-level OCR; using original page/column crops rather than line crops"]
        for block, _result in block_results:
            notes.append(f"{block['region_id']}:{block['mode']}")
        return OCRComparisonSummary(
            backend_name="glmocr",
            model_name="GLM-OCR-0.9B",
            selected=selected,
            text=text_value,
            lines=lines_value,
            confidence=None,
            warnings=warnings,
            language_hint=payload.language_hint or (payload.metadata.language if payload.metadata else None),
            script_family=payload.metadata.script_family if payload.metadata else None,
            notes=notes,
        )
    except OCRBackendError as exc:
        return OCRComparisonSummary(
            backend_name="glmocr",
            model_name="GLM-OCR-0.9B",
            selected=selected,
            text="",
            lines=[],
            confidence=0.0,
            warnings=[str(exc)],
            language_hint=payload.language_hint or (payload.metadata.language if payload.metadata else None),
            script_family=payload.metadata.script_family if payload.metadata else None,
            notes=["experimental document-level OCR"],
        )


def _full_page_response_from_comparison_summary(
    summary: OCRComparisonSummary,
    payload: SaiaFullPageExtractRequest,
) -> SaiaFullPageExtractResponse:
    text_value = str(summary.text or "").strip()
    lines_value = [line for line in summary.lines if str(line or "").strip()]
    if not text_value and lines_value:
        text_value = "\n".join(lines_value).strip()
    detected_language, language_confidence = _detect_language_metadata(text_value)
    if detected_language == "unknown" and payload.language_hint:
        detected_language = _normalize_detected_language(str(payload.language_hint))
    return SaiaFullPageExtractResponse(
        status=_resolve_full_page_status(text=text_value, confidence=summary.confidence, warnings=list(summary.warnings)),  # type: ignore[arg-type]
        model_used=summary.model_name,
        fallbacks_used=[],
        detected_language=_normalize_detected_language(detected_language),
        language_confidence=language_confidence,
        script_hint=_map_extract_script_hint_to_full_page(payload.script_hint_seed),  # type: ignore[arg-type]
        confidence=summary.confidence,
        warnings=list(summary.warnings),
        lines=lines_value,
        text=text_value,
        fallbacks=[],
        comparison_runs=[summary],
    )


def _full_page_response_from_glm_ollama_result(
    result: GlmOllamaOcrResult,
    payload: SaiaFullPageExtractRequest,
) -> SaiaFullPageExtractResponse:
    warnings = list(result.warnings)
    if payload.compare_backends:
        warnings.append("COMPARE_BACKENDS_IGNORED")
    fallbacks_used = [
        warning.split(":", 1)[1]
        for warning in warnings
        if warning.startswith("OCR_VARIANT_FALLBACK:")
    ]
    if result.attempts_used > 1:
        fallbacks_used.append(f"retry_attempt:{result.attempts_used}")
    summary = OCRComparisonSummary(
        backend_name="glmocr",
        model_name=result.model_used,
        selected=True,
        text=result.text,
        lines=list(result.lines),
        confidence=None,
        warnings=warnings,
        language_hint=payload.language_hint or (payload.metadata.language if payload.metadata else None),
        script_family=payload.metadata.script_family if payload.metadata else None,
        notes=[
            f"processed_variant={result.processed_variant_name}",
            f"processed_dimensions={result.processed_width}x{result.processed_height}",
            f"processed_size_bytes={result.processed_size_bytes}",
        ],
    )
    response = _full_page_response_from_comparison_summary(summary, payload)
    return response.model_copy(
        update={
            "fallbacks_used": fallbacks_used,
            "warnings": warnings,
            "original_image_size_bytes": result.original_size_bytes,
            "original_image_width": result.original_width,
            "original_image_height": result.original_height,
            "processed_image_size_bytes": result.processed_size_bytes,
            "processed_image_width": result.processed_width,
            "processed_image_height": result.processed_height,
            "preprocessing_applied": result.preprocessing_applied,
            "processed_variant_name": result.processed_variant_name,
            "ocr_attempts_used": result.attempts_used,
        }
    )


def _build_comparison_runs(result: OCRExtractResponse, requested_backend: str) -> list[OCRComparisonSummary]:
    order_map = {
        region.region_id: int(region.reading_order if region.reading_order is not None else index)
        for index, region in enumerate(result.regions)
    }
    selected_text = str(result.final_text or result.text or "").strip()
    selected_lines = [line for line in selected_text.splitlines() if line.strip()]
    selected_confidence = None
    if result.raw_ocr is not None:
        selected_confidence = float(result.raw_ocr.confidence)
    selected_meta = result.regions[0].raw_metadata if result.regions and result.regions[0].raw_metadata else {}

    runs: list[OCRComparisonSummary] = [
        OCRComparisonSummary(
            backend_name="kraken_family" if requested_backend == "auto" and str(result.ocr_backend or "").startswith("kraken") else str(result.ocr_backend or requested_backend or "selected"),
            model_name=result.model,
            selected=True,
            text=selected_text,
            lines=selected_lines,
            confidence=selected_confidence,
            warnings=list(result.warnings),
            language_hint=str(selected_meta.get("normalized_language_hint") or selected_meta.get("language_hint") or "") or None,
            script_family=str(selected_meta.get("script_family") or "") or None,
            notes=[str(item) for item in list(selected_meta.get("notes") or []) if str(item).strip()],
        )
    ]

    grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for item in result.comparison_results:
        if item.selected:
            continue
        grouped[(item.backend_name, item.model_name)].append(item)

    for (backend_name, model_name), items in grouped.items():
        ordered_items = sorted(items, key=lambda item: order_map.get(item.region_id, 10**9))
        text_value = "\n".join(str(item.text or "").strip() for item in ordered_items if str(item.text or "").strip()).strip()
        lines_value = [line for line in text_value.splitlines() if line.strip()]
        warnings = [
            str(item.raw_metadata.get("error"))
            for item in ordered_items
            if item.raw_metadata and item.raw_metadata.get("error")
        ]
        first_meta = next((item.raw_metadata for item in ordered_items if item.raw_metadata), {}) or {}
        confidence_values = [float(item.confidence) for item in ordered_items if item.confidence is not None]
        confidence = (sum(confidence_values) / len(confidence_values)) if confidence_values else None
        runs.append(
            OCRComparisonSummary(
                backend_name=backend_name,
                model_name=model_name,
                selected=False,
                text=text_value,
                lines=lines_value,
                confidence=confidence,
                warnings=warnings,
                language_hint=str(first_meta.get("normalized_language_hint") or first_meta.get("language_hint") or "") or None,
                script_family=str(first_meta.get("script_family") or "") or None,
                notes=[str(item) for item in list(first_meta.get("notes") or []) if str(item).strip()],
            )
        )
    return runs


def _full_page_response_from_extract(result: OCRExtractResponse) -> SaiaFullPageExtractResponse:
    text_value = str(result.final_text or result.text or "")
    text_value = _strip_uncertainty_markers(text_value)
    lines_value = _strip_uncertainty_lines([line for line in text_value.splitlines() if line.strip()])
    text_value = "\n".join(lines_value).strip()
    detected_language, language_confidence = _detect_language_metadata(text_value)
    confidence_value = None
    if result.raw_ocr is not None:
        confidence_value = float(result.raw_ocr.confidence)
    elif result.regions:
        region_conf = [float(region.confidence) for region in result.regions if region.confidence is not None]
        confidence_value = (sum(region_conf) / len(region_conf)) if region_conf else None

    warnings = list(dict.fromkeys(result.warnings))
    return SaiaFullPageExtractResponse(
        status=_resolve_full_page_status(text=text_value, confidence=confidence_value, warnings=warnings),  # type: ignore[arg-type]
        model_used=result.model,
        fallbacks_used=list(result.fallbacksUsed),
        detected_language=_normalize_detected_language(detected_language),
        language_confidence=language_confidence,
        script_hint=_map_extract_script_hint_to_full_page(result.script_hint),  # type: ignore[arg-type]
        confidence=confidence_value,
        warnings=warnings,
        lines=lines_value,
        text=text_value,
        fallbacks=[],
        comparison_runs=[],
    )


@router.post("/ocr/extract", response_model=OCRExtractAnyResponse)
async def ocr_extract(
    payload: OCRExtractRequest,
    mode: str | None = Query(default=None, pattern="^(full|simple)$"),
) -> OCRExtractAnyResponse:
    effective_payload = payload
    if mode in {"full", "simple"} and payload.mode != mode:
        effective_payload = payload.model_copy(update={"mode": mode})
    try:
        return _get_ocr_agent().run(effective_payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SaiaConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OCRAgentError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"OCR extraction failed: {exc}") from exc


@router.post("/ocr/saia", response_model=SaiaOCRResponse)
async def ocr_saia(payload: SaiaOCRRequest) -> SaiaOCRResponse:
    try:
        full_page_payload = SaiaFullPageExtractRequest(
            image_id=payload.image_id,
            page_id=payload.page_id,
            image_b64=payload.image_b64,
            script_hint_seed=payload.script_hint_seed,
            apply_proofread=payload.apply_proofread,
            location_suggestions=list(payload.location_suggestions),
            ocr_backend="glmocr",
        )
        response = await ocr_extract_full_page(full_page_payload)
        return SaiaOCRResponse(
            status="FAIL" if response.status == "EMPTY" else response.status,
            model_used=response.model_used,
            fallbacks=[],
            fallbacks_used=list(response.fallbacks_used),
            warnings=list(response.warnings),
            lines=list(response.lines),
            text=response.text,
            script_hint=response.script_hint,
            detected_language=response.detected_language,
            confidence=float(response.confidence or 0.0),
            raw_json={
                "status": response.status,
                "model_used": response.model_used,
                "warnings": list(response.warnings),
                "lines": list(response.lines),
                "text": response.text,
                "script_hint": response.script_hint,
                "detected_language": response.detected_language,
                "confidence": response.confidence,
                "processed_image_size_bytes": response.processed_image_size_bytes,
                "processed_image_width": response.processed_image_width,
                "processed_image_height": response.processed_image_height,
            },
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GlmOllamaOcrError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"GLM OCR failed: {exc}") from exc


class _SegmentedPageUnread(Exception):
    """The segmented engine read no text on a page; GLM-OCR reads it instead."""


def _full_page_engine(payload: SaiaFullPageExtractRequest) -> str:
    """"segmented" (layout, then line recognition) or "glmocr" (whole-page VLM)."""
    backend = str(payload.ocr_backend or "auto").strip().lower()
    if backend == "glmocr":
        return "glmocr"
    if backend == "auto" and str(_app_settings.full_page_ocr_engine or "").strip().lower() == "glmocr":
        return "glmocr"
    return "segmented"


def _upright_page_bytes(image_bytes: bytes) -> bytes:
    """Apply an EXIF orientation, so layout boxes and line crops share one frame.

    OpenCV, which loads the page for the layout models, honours the tag; PIL,
    which cuts the crops, does not. A rotated phone photo would otherwise be
    segmented upright and cropped sideways.
    """
    from PIL import Image, ImageOps

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.getexif().get(0x0112, 1) in (0, 1):
                return image_bytes
            upright = ImageOps.exif_transpose(image)
            buffer = io.BytesIO()
            upright.save(buffer, format="PNG")
            return buffer.getvalue()
    except (OSError, ValueError):
        return image_bytes


def _secondary_lines(result: OCRExtractAnyResponse) -> list[str]:
    if not isinstance(result, OCRExtractResponse):
        return []
    return _strip_uncertainty_lines(str(result.final_text or result.text or "").splitlines())


def _run_segmented_full_page(
    payload: SaiaFullPageExtractRequest,
    image_bytes: bytes,
) -> tuple[SaiaFullPageExtractResponse, str]:
    """Read a page line by line: layout segmentation, then the OCR agent's plan.

    Returns the response and the backend that read the body. Text detected
    outside every column is read separately into ``secondary_lines``, so a
    library stamp or a folio number never lands in the middle of the text.
    """
    page_bytes = _upright_page_bytes(image_bytes)
    if page_bytes is not image_bytes:
        payload = payload.model_copy(update={"image_b64": base64.b64encode(page_bytes).decode("ascii")})
    body, outside = _segmented_region_lanes(_segment_page(page_bytes))
    if not body:
        raise _SegmentedPageUnread("no_text_regions")

    agent = _get_ocr_agent()
    result = agent.run(_build_segmented_extract_request(payload, body))
    if not isinstance(result, OCRExtractResponse) or not str(result.final_text or result.text or "").strip():
        raise _SegmentedPageUnread("no_text")
    backend = str(result.ocr_backend or payload.ocr_backend or "auto")

    warnings: list[str] = []
    secondary: list[str] = []
    if outside:
        # Pinned to the body's backend, so both lanes are read by one model.
        pinned = payload.model_copy(update={"ocr_backend": backend, "compare_backends": []})
        try:
            secondary = _secondary_lines(agent.run(_build_segmented_extract_request(pinned, outside)))
        except Exception as exc:
            warnings.append(f"SECONDARY_TEXT_FAILED:{exc}")

    from PIL import Image

    with Image.open(io.BytesIO(page_bytes)) as image:
        width, height = image.size
    response = _full_page_response_from_extract(result)
    return (
        response.model_copy(
            update={
                "ocr_engine": "segmented",
                "warnings": list(dict.fromkeys([*response.warnings, *warnings])),
                "secondary_lines": secondary,
                "comparison_runs": _build_comparison_runs(result, str(payload.ocr_backend or "auto")),
                "original_image_size_bytes": len(image_bytes),
                "original_image_width": width,
                "original_image_height": height,
            }
        ),
        backend,
    )


async def _run_glm_full_page(
    payload: SaiaFullPageExtractRequest,
    image_bytes: bytes,
) -> SaiaFullPageExtractResponse:
    try:
        result = get_test_ocr_override(image_bytes)
        if result is None:
            glm_prompt = build_glm_ocr_prompt(
                language_hint=payload.language_hint,
                script_hint_seed=payload.script_hint_seed,
                metadata=payload.metadata,
            )
            result = await asyncio.to_thread(
                run_glm_ollama_ocr,
                image_bytes,
                image_ref=(
                    str(payload.page_id or "").strip()
                    or str(payload.image_id or "").strip()
                    or str(payload.document_id or "").strip()
                    or "inline:image_b64"
                ),
                prompt=glm_prompt,
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GlmOllamaOcrError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Full-page OCR failed: {exc}") from exc
    response = _full_page_response_from_glm_ollama_result(result, payload)
    return response.model_copy(update={"ocr_engine": "glmocr"})


@router.post("/ocr/extract_full_page", response_model=SaiaFullPageExtractResponse)
async def ocr_extract_full_page(payload: SaiaFullPageExtractRequest) -> SaiaFullPageExtractResponse:
    try:
        image_bytes = decode_image_bytes(payload.image_b64 or "")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image_b64 payload: {exc}") from exc
    fixture = get_test_ocr_fixture(image_bytes)
    if fixture is not None and fixture.wait_seconds > 0:
        await asyncio.sleep(float(fixture.wait_seconds))

    response: SaiaFullPageExtractResponse | None = None
    ocr_backend = "glmocr"
    fallback_warnings: list[str] = []
    # A canned fixture is a GLM-OCR result; it never goes through segmentation.
    if fixture is None and _full_page_engine(payload) == "segmented":
        try:
            response, ocr_backend = await asyncio.to_thread(_run_segmented_full_page, payload, image_bytes)
        except _SegmentedPageUnread as exc:
            fallback_warnings.append(f"OCR_ENGINE_FALLBACK:{exc}")
        except Exception as exc:
            fallback_warnings.append(f"OCR_ENGINE_FALLBACK:segmented_failed:{str(exc)[:200]}")
    if response is None:
        response = await _run_glm_full_page(payload, image_bytes)
        ocr_backend = "glmocr"
        if fallback_warnings:
            response = response.model_copy(update={"warnings": [*response.warnings, *fallback_warnings]})

    try:
        pipeline_fields = await asyncio.to_thread(
            _run_full_page_post_ocr_pipeline,
            payload,
            response,
            image_bytes,
            ocr_backend=ocr_backend,
        )
    except Exception as exc:  # pragma: no cover
        pipeline_fields = {"warnings": [f"POST_OCR_PIPELINE_FAILED:{exc}"]}

    merged_warnings = list(dict.fromkeys([*list(response.warnings), *list(pipeline_fields.pop("warnings", []) or [])]))
    return response.model_copy(update={**pipeline_fields, "warnings": merged_warnings})


# ── Quality label ranking helper ──────────────────────────────────────

_QUALITY_RANK = {"HIGH": 0, "OK": 1, "RISKY": 2, "UNRELIABLE": 3}


def _quality_rank(label: str) -> int:
    """Return numeric rank for quality label (lower = better)."""
    return _QUALITY_RANK.get(label, 4)


@router.post("/ocr/page_with_trace")
async def ocr_page_with_trace(payload: SaiaFullPageExtractRequest) -> dict[str, Any]:
    """Run single-page OCR pipeline with trace logging.

    Implements hard quality gate enforcement with effective seam retry:
      - OCR runs in a retry loop (max_attempts configurable, default 3)
      - Attempt 0: normal OCR with standard tiling
      - Attempt 1+: seam-aware retry using strategy selection
        (grid_shift → seam_band_crop → expand_overlap)
      - NO-OP guard: if retry produces identical text_sha256 or tile_boxes,
        the retry is marked as failed/no-op and the next strategy is tried
      - After each attempt, quality gates are evaluated
      - If any blocking gate fails, downstream stages are SKIPPED
      - If all attempts fail gates, run is marked FAILED_QUALITY
    """
    # ── Configurable retry params ─────────────────────────────────
    MAX_OCR_ATTEMPTS = _MAX_OCR_ATTEMPTS  # from ocr_quality_config

    try:
        image_bytes = decode_image_bytes(payload.image_b64 or "")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image_b64 payload: {exc}") from exc

    fixture = get_test_ocr_fixture(image_bytes)
    if fixture is not None and fixture.wait_seconds > 0:
        await asyncio.sleep(float(fixture.wait_seconds))

    test_override = get_test_ocr_override(image_bytes)
    if test_override is not None:
        response = _full_page_response_from_glm_ollama_result(test_override, payload)
        try:
            pipeline_fields = await asyncio.to_thread(
                _run_full_page_post_ocr_pipeline,
                payload,
                response,
                image_bytes,
                ocr_backend="glmocr",
            )
        except Exception as exc:  # pragma: no cover
            pipeline_fields = {"warnings": [f"POST_OCR_PIPELINE_FAILED:{exc}"]}

        merged_warnings = list(
            dict.fromkeys([*list(response.warnings), *list(pipeline_fields.get("warnings", []) or [])])
        )
        final_text = str(pipeline_fields.get("proofread_text") or response.text or "")
        confidence = float(response.confidence or 0.0)
        quality_label = pipeline_fields.get("quality_label")
        downstream_mode = pipeline_fields.get("downstream_mode")
        return {
            "run_id": pipeline_fields.get("run_id"),
            "status": pipeline_fields.get("status", "COMPLETED"),
            "ocr_result": {
                "lines": list(response.lines),
                "text": response.text,
                "script_hint": response.script_hint,
                "detected_language": response.detected_language,
                "confidence": confidence,
                "warnings": merged_warnings,
                "quality_label": quality_label,
                "sanity_metrics": {},
                "quality_label_v2": quality_label,
                "downstream_mode": downstream_mode,
            },
            "proofread_text": final_text,
            "detected_language": response.detected_language,
            "final_confidence": response.confidence,
            "quality_label": quality_label,
            "quality_label_v2": quality_label,
            "effective_quality": None,
            "downstream_mode": downstream_mode,
            "sanity_metrics": {},
            "quality_gates": None,
            "ocr_attempts": [],
            "mention_recall": None,
            "chunks_count": pipeline_fields.get("chunks_count"),
            "mentions_count": pipeline_fields.get("mentions_count"),
            "top_mentions": [],
            "mention_report": pipeline_fields.get("mention_report"),
            "linking_result": None,
            "consolidated_report": pipeline_fields.get("consolidated_report"),
            "authority_report": pipeline_fields.get("authority_report"),
            "ocr_backend_results": [],
        }

    # Get image dimensions for strategy computation
    import io
    from PIL import Image as _PILImage
    with _PILImage.open(io.BytesIO(image_bytes)) as _pil_img:
        img_w, img_h = _pil_img.size

    asset_ref = (
        str(payload.page_id or "").strip()
        or str(payload.image_id or "").strip()
        or str(payload.document_id or "").strip()
        or "inline:image_b64"
    )
    asset_sha256 = hashlib.sha256(image_bytes).hexdigest()
    run_id = create_run(asset_ref=asset_ref, asset_sha256=asset_sha256)
    log_event(run_id, "RECEIVED", "START", "OCR trace run received.")
    if str(payload.benchmark_text or "").strip():
        insert_ocr_benchmark_reference(
            run_id,
            page_id=payload.page_id,
            source_label=str(payload.benchmark_source or "reference").strip() or "reference",
            reference_text=str(payload.benchmark_text),
        )
        log_event(run_id, "RECEIVED", "INFO", "Stored benchmark/reference text for later OCR comparison.")

    location_suggestions = list(payload.location_suggestions)
    base_warnings: list[str] = []
    if not location_suggestions:
        try:
            location_suggestions = await asyncio.to_thread(_run_segmentation_for_suggestions, image_bytes)
            log_event(run_id, "RECEIVED", "INFO",
                      f"Segmentation location suggestions prepared: {len(location_suggestions)}")
        except Exception as exc:
            base_warnings.append(f"SEGMENTATION_FAILED:{exc}")
            log_event(run_id, "RECEIVED", "ERROR", f"Segmentation failed: {exc}")
    else:
        log_event(run_id, "RECEIVED", "INFO",
                  f"Using provided location suggestions: {len(location_suggestions)}")

    structured_regions = list(payload.regions) or _regions_from_location_suggestions(location_suggestions)
    if not structured_regions and not payload.location_suggestions:
        try:
            structured_regions = await asyncio.to_thread(_run_segmentation_for_regions, image_bytes)
            if structured_regions:
                log_event(run_id, "RECEIVED", "INFO", f"Segmented OCR regions prepared: {len(structured_regions)}")
        except Exception as exc:
            base_warnings.append(f"SEGMENTATION_REGIONS_FAILED:{exc}")
            log_event(run_id, "RECEIVED", "WARN", f"Structured region preparation failed: {exc}")

    if structured_regions:
        try:
            return await _run_segmented_trace_pipeline(
                run_id=run_id,
                asset_ref=asset_ref,
                payload=payload,
                extract_payload=_build_segmented_extract_request(payload, structured_regions),
            )
        except Exception as exc:
            err = str(exc)
            update_run_fields(run_id, status="FAILED", current_stage="FAILED", error=err)
            log_event(run_id, "FAILED", "ERROR", err)
            raise HTTPException(status_code=500, detail={"run_id": run_id, "error": err}) from exc

    try:
        # ══════════════════════════════════════════════════════════════
        # OCR RETRY LOOP — attempt 0..MAX_OCR_ATTEMPTS-1
        # Each attempt: run OCR → compute quality → evaluate gates
        # NO-OP guard: if text_sha256 or tile_boxes identical, switch strategy
        # If gates pass → proceed to downstream. If fail → retry or stop.
        # ══════════════════════════════════════════════════════════════
        from app.agents.saia_ocr_agent import (
            compute_sanity, _quality_label_from_sanity,
            quality_gate_enforce, sanity_adjust_confidence,
            force_uncertainty_markers, format_sanity_summary,
        )

        best_ocr_result: SaiaOCRResponse | None = None
        best_ocr_payload: dict[str, Any] | None = None
        best_quality_report: OCRQualityReport | None = None
        best_gate_decisions: dict[str, Any] | None = None
        best_effective_quality: EffectiveQuality | None = None
        best_attempt_idx: int = -1
        gates_ever_passed: bool = False

        # Track previous attempt for NO-OP detection
        prev_text_sha256: str = ""
        prev_tiling_plan: TilingPlan | None = None

        # Build initial tiling plan from location_suggestions
        suggestion_dicts = [
            {"region_id": getattr(s, "region_id", None),
             "category": getattr(s, "category", None),
             "bbox_xywh": list(getattr(s, "bbox_xywh", []))}
            for s in location_suggestions
        ]
        current_plan = default_plan_from_suggestions(suggestion_dicts, img_w, img_h)

        for attempt_idx in range(MAX_OCR_ATTEMPTS):
            # ── Determine tiling strategy for this attempt ────────────
            if attempt_idx == 0:
                effective_suggestions = location_suggestions
            else:
                # Select a retry strategy that changes tile_boxes
                assert prev_tiling_plan is not None
                current_plan = select_retry_strategy(
                    prev_tiling_plan, img_w, img_h,
                    prev_text_sha256=prev_text_sha256,
                    attempt_idx=attempt_idx,
                )
                # Convert TilingPlan → SaiaOCRLocationSuggestion objects
                retry_dicts = plan_to_suggestions(current_plan, img_w, img_h)
                effective_suggestions = [
                    SaiaOCRLocationSuggestion(**d) for d in retry_dicts
                ]
                log_event(run_id, "OCR_RUNNING", "INFO",
                          f"Retry strategy={current_plan.strategy} "
                          f"grid={current_plan.grid} "
                          f"tiles={len(current_plan.tile_boxes)} "
                          f"overlap={current_plan.overlap_pct:.0%}")

            log_event(run_id, "OCR_RUNNING", "START",
                      f"OCR attempt {attempt_idx}/{MAX_OCR_ATTEMPTS-1} "
                      f"strategy={current_plan.strategy}")
            update_run_fields(run_id, status="RUNNING",
                              current_stage="OCR_RUNNING")

            # ── Run OCR ───────────────────────────────────────────────
            ocr_result = await asyncio.to_thread(
                _get_saia_ocr_agent().extract,
                SaiaOCRRequest(
                    image_id=payload.image_id,
                    page_id=payload.page_id,
                    image_b64=payload.image_b64,
                    script_hint_seed=payload.script_hint_seed,
                    apply_proofread=False,
                    location_suggestions=effective_suggestions,
                ),
            )
            log_event(run_id, "OCR_DONE", "END",
                      f"OCR attempt {attempt_idx} complete, "
                      f"model={ocr_result.model_used}")

            # ── Log tile info ─────────────────────────────────────────
            raw_blob = ocr_result.raw_json if isinstance(ocr_result.raw_json, dict) else {}
            if "tile_count" in raw_blob:
                log_event(run_id, "OCR_DONE", "INFO",
                          f"Tile-based OCR: {raw_blob['tile_count']} tiles "
                          f"(strategy={current_plan.strategy}, grid={current_plan.grid}).")
            if raw_blob.get("column_split"):
                log_event(run_id, "OCR_DONE", "INFO",
                          f"Column-split OCR: {raw_blob.get('columns', 2)} cols.")

            # ── Legacy sanity metrics ─────────────────────────────────
            ocr_sanity = compute_sanity(str(ocr_result.text or ""))
            quality_label = raw_blob.get("quality_label") or _quality_label_from_sanity(ocr_sanity)
            sanity_metrics = raw_blob.get("sanity_metrics") or ocr_sanity
            sanity_summary = format_sanity_summary(sanity_metrics)

            log_event(run_id, "OCR_SANITY", "INFO",
                      f"attempt={attempt_idx} quality_label={quality_label} "
                      f"single_char={ocr_sanity.get('single_char_ratio',0):.3f} "
                      f"weird={ocr_sanity.get('weird_ratio',0):.3f} "
                      f"leading_frag={ocr_sanity.get('leading_fragment_ratio',0):.3f}")

            ocr_payload: dict[str, Any] = {
                "lines": list(ocr_result.lines),
                "text": str(ocr_result.text or ""),
                "script_hint": str(ocr_result.script_hint or "unknown"),
                "detected_language": _normalize_detected_language(ocr_result.detected_language),
                "confidence": float(ocr_result.confidence) if ocr_result.confidence is not None else 0.0,
                "warnings": list(dict.fromkeys([*base_warnings, *ocr_result.warnings, sanity_summary])),
                "quality_label": quality_label,
                "sanity_metrics": sanity_metrics,
            }

            # ── Compute text hash for NO-OP detection ─────────────────
            text_sha256 = hashlib.sha256(
                (ocr_payload["text"] or "").encode()
            ).hexdigest()
            current_plan.text_sha256 = text_sha256

            # ── NO-OP guard + auto-escalation ─────────────────────────
            noop_detected = False
            if attempt_idx > 0 and prev_tiling_plan is not None:
                noop_detected = is_noop_retry(
                    prev_text_sha256, text_sha256,
                    prev_tiling_plan, current_plan,
                )
                if noop_detected:
                    log_event(run_id, "OCR_DONE", "WARN",
                              f"NO-OP detected on attempt {attempt_idx}: "
                              f"text_sha256 {'identical' if text_sha256 == prev_text_sha256 else 'different'}, "
                              f"tile_boxes {'identical' if current_plan.boxes_signature() == prev_tiling_plan.boxes_signature() else 'different'}. "
                              f"Auto-escalating to next strategy.")
                    # Auto-escalate: immediately pick a different strategy
                    escalated_plan = select_retry_strategy(
                        current_plan, img_w, img_h,
                        prev_text_sha256=text_sha256,
                        attempt_idx=attempt_idx + 10,  # high idx to force fallback chain
                    )
                    if escalated_plan.boxes_signature() != current_plan.boxes_signature():
                        log_event(run_id, "OCR_DONE", "INFO",
                                  f"NOOP escalation: re-running OCR with "
                                  f"strategy={escalated_plan.strategy} "
                                  f"grid={escalated_plan.grid}")
                        current_plan = escalated_plan
                        retry_dicts = plan_to_suggestions(current_plan, img_w, img_h)
                        effective_suggestions = [
                            SaiaOCRLocationSuggestion(**d) for d in retry_dicts
                        ]
                        ocr_result = await asyncio.to_thread(
                            _get_saia_ocr_agent().extract,
                            SaiaOCRRequest(
                                image_id=payload.image_id,
                                page_id=payload.page_id,
                                image_b64=payload.image_b64,
                                script_hint_seed=payload.script_hint_seed,
                                apply_proofread=False,
                                location_suggestions=effective_suggestions,
                            ),
                        )
                        # Recompute text hash after escalation
                        ocr_payload["lines"] = list(ocr_result.lines)
                        ocr_payload["text"] = str(ocr_result.text or "")
                        text_sha256 = hashlib.sha256(
                            (ocr_payload["text"] or "").encode()
                        ).hexdigest()
                        current_plan.text_sha256 = text_sha256
                        noop_detected = False  # no longer a NOOP
                        log_event(run_id, "OCR_DONE", "INFO",
                                  f"NOOP escalation OCR complete, new text_hash={text_sha256[:12]}")
                    else:
                        log_event(run_id, "OCR_DONE", "WARN",
                                  f"NOOP escalation failed to change geometry. "
                                  f"Recording as NOOP attempt.")

            # ── Quality report (language-aware when a profile exists) ──
            prev_tokens = None
            if best_ocr_payload is not None:
                prev_tokens = [t for t in (best_ocr_payload["text"] or "").split() if t]
            ocr_quality_report = compute_quality_report(
                ocr_payload["text"],
                run_id=run_id,
                pass_idx=attempt_idx,
                previous_pass_tokens=prev_tokens,
                language=_quality_language(ocr_payload.get("detected_language")),
            )
            hardened_quality_label = ocr_quality_report.quality_label
            insert_ocr_quality_report(run_id, ocr_quality_report.to_dict())

            log_event(run_id, "QUALITY_REPORT", "INFO",
                      f"attempt={attempt_idx} quality_label_v2={hardened_quality_label} "
                      f"gibberish={ocr_quality_report.gibberish_score:.4f} "
                      f"lead_frag={ocr_quality_report.leading_fragment_ratio:.4f} "
                      f"seam_frag={ocr_quality_report.seam_fragment_ratio:.4f} "
                      f"nwl={ocr_quality_report.non_wordlike_frac:.4f} "
                      f"entropy={ocr_quality_report.char_entropy:.4f}")

            # ── Enforce quality gates ─────────────────────────────────
            _lex_lang = ocr_payload.get("detected_language", "unknown")
            _lex_score = _lexical_plausibility(ocr_payload["text"], _lex_lang) if _lex_lang != "unknown" else None
            gate_decisions = enforce_quality_gates(
                ocr_quality_report, run_id=run_id,
                lexical_plausibility=_lex_score,
            )
            downstream_mode = gate_decisions["downstream_mode"]
            log_event(run_id, "QUALITY_GATES", "INFO", format_gate_report(gate_decisions))

            has_blocking_failure = bool(gate_decisions.get("blocked_stages"))
            all_gates_passed = all(
                g["passed"] for g in gate_decisions.get("gates", {}).values()
            )

            # ── Build unified EffectiveQuality ────────────────────────
            effective_quality = build_effective_quality(
                ocr_quality_report, gate_decisions,
                confidence=ocr_payload["confidence"],
            )

            # ── Record attempt in DB ──────────────────────────────────
            eff_frag = frag_gate_value(
                ocr_quality_report.leading_fragment_ratio,
                ocr_quality_report.seam_fragment_ratio,
            )
            retry_reason = "initial" if attempt_idx == 0 else (
                "seam_retry" if gate_decisions.get("seam_retry_required") else "quality_retry"
            )
            insert_ocr_attempt(run_id, {
                "attempt_idx": attempt_idx,
                "tiling_strategy": current_plan.strategy,
                "tile_grid": current_plan.grid,
                "overlap_pct": current_plan.overlap_pct,
                "tile_count": len(current_plan.tile_boxes) or raw_blob.get("tile_count"),
                "tile_boxes_json": current_plan.tile_boxes,
                "preproc_json": current_plan.preproc,
                "model_used": ocr_result.model_used,
                "text_sha256": text_sha256,
                "text_hash": text_sha256[:16],
                "quality_label": hardened_quality_label,
                "effective_quality_json": effective_quality.to_dict(),
                "gibberish_score": ocr_quality_report.gibberish_score,
                "leading_fragment_ratio": ocr_quality_report.leading_fragment_ratio,
                "seam_fragment_ratio": ocr_quality_report.seam_fragment_ratio,
                "non_wordlike_frac": ocr_quality_report.non_wordlike_frac,
                "char_entropy": ocr_quality_report.char_entropy,
                "uncertainty_density": ocr_quality_report.uncertainty_density,
                "cross_pass_stability": ocr_quality_report.cross_pass_stability,
                "gates_passed": all_gates_passed,
                "noop_detected": noop_detected,
                "decision": "NOOP" if noop_detected else ("PASS" if all_gates_passed else "FAIL"),
                "ocr_text": ocr_payload["text"][:2000] if ocr_payload["text"] else None,
                "detail_json": {
                    "downstream_mode": downstream_mode,
                    "blocked_stages": gate_decisions.get("blocked_stages", []),
                    "seam_retry_required": gate_decisions.get("seam_retry_required", False),
                    "noop_detected": noop_detected,
                    "reason": retry_reason,
                    "lead_frag": ocr_quality_report.leading_fragment_ratio,
                    "seam_frag": ocr_quality_report.seam_fragment_ratio,
                    "max_frag_used_for_gate": eff_frag,
                    "cross_pass_stability": ocr_quality_report.cross_pass_stability,
                    "thresholds": {
                        "LEADING_FRAG_HARD_LIMIT": LEADING_FRAG_HARD_LIMIT,
                        "SEAM_FRAG_HARD_LIMIT": SEAM_FRAG_HARD_LIMIT,
                        "CROSS_PASS_STABILITY_MIN": CROSS_PASS_STABILITY_MIN,
                    },
                },
            })

            # ── Track best attempt ────────────────────────────────────
            is_better = (
                best_quality_report is None
                or _quality_rank(hardened_quality_label) < _quality_rank(best_quality_report.quality_label)
                or (
                    _quality_rank(hardened_quality_label) == _quality_rank(best_quality_report.quality_label)
                    and ocr_quality_report.gibberish_score < best_quality_report.gibberish_score
                )
            )
            if is_better:
                best_ocr_result = ocr_result
                best_ocr_payload = ocr_payload
                best_quality_report = ocr_quality_report
                best_gate_decisions = gate_decisions
                best_effective_quality = effective_quality
                best_attempt_idx = attempt_idx

            if all_gates_passed:
                gates_ever_passed = True
                log_event(run_id, "QUALITY_GATES", "INFO",
                          f"All gates PASSED on attempt {attempt_idx}. "
                          f"Proceeding to downstream stages.")
                break  # no need to retry

            # Gates failed — decide whether to retry
            prev_text_sha256 = text_sha256
            prev_tiling_plan = current_plan

            if attempt_idx < MAX_OCR_ATTEMPTS - 1:
                if gate_decisions.get("seam_retry_required"):
                    log_event(run_id, "QUALITY_GATES", "INFO",
                              f"Gates FAILED on attempt {attempt_idx} "
                              f"(seam_retry_required). Scheduling retry with "
                              f"strategy selection (grid_shift/seam_band_crop/expand).")
                    update_run_fields(run_id, current_stage="SEAM_RETRY_PENDING")
                elif has_blocking_failure:
                    log_event(run_id, "QUALITY_GATES", "INFO",
                              f"Gates FAILED on attempt {attempt_idx} "
                              f"(blocked_stages={gate_decisions['blocked_stages']}). "
                              f"Scheduling retry.")
                    update_run_fields(run_id, current_stage="QUALITY_RETRY_PENDING")
                else:
                    log_event(run_id, "QUALITY_GATES", "INFO",
                              f"Quality {hardened_quality_label} on attempt "
                              f"{attempt_idx}. Scheduling retry.")
                    update_run_fields(run_id, current_stage="QUALITY_RETRY_PENDING")
            else:
                log_event(run_id, "QUALITY_GATES", "INFO",
                          f"All {MAX_OCR_ATTEMPTS} attempts exhausted. "
                          f"Best attempt={best_attempt_idx}, "
                          f"quality={best_quality_report.quality_label if best_quality_report else 'UNKNOWN'}.")

        # ══════════════════════════════════════════════════════════════
        # POST-RETRY: Use best attempt results for the rest of the pipeline
        # ══════════════════════════════════════════════════════════════
        assert best_ocr_result is not None
        assert best_ocr_payload is not None
        assert best_quality_report is not None
        assert best_gate_decisions is not None
        assert best_effective_quality is not None

        ocr_result = best_ocr_result
        ocr_payload = best_ocr_payload
        ocr_quality_report = best_quality_report
        gate_decisions = best_gate_decisions
        effective_quality = best_effective_quality
        hardened_quality_label = ocr_quality_report.quality_label
        downstream_mode = gate_decisions["downstream_mode"]

        # ── Cross-pass stability: compute if not yet done and borderline ──
        # If attempt 0 is not HIGH, or near fragment thresholds, or
        # seam_retry_required, compute stability via a controlled perturbation.
        if ocr_quality_report.cross_pass_stability < 0:
            needs_stability = (
                hardened_quality_label not in ("HIGH",)
                or ocr_quality_report.seam_retry_required
                or frag_gate_value(
                    ocr_quality_report.leading_fragment_ratio,
                    ocr_quality_report.seam_fragment_ratio,
                ) >= SEAM_FRAG_HARD_LIMIT * 0.7
            )
            if needs_stability and best_attempt_idx >= 0 and prev_tiling_plan is not None:
                # Use a small perturbation plan for the stability pass
                from app.services.seam_strategies import expand_overlap as _expand_for_stability
                stability_plan = _expand_for_stability(
                    current_plan.tile_boxes, img_w, img_h,
                    expand_pct=0.10, attempt_idx=99,
                )
                stability_dicts = plan_to_suggestions(stability_plan, img_w, img_h)
                stability_suggestions = [
                    SaiaOCRLocationSuggestion(**d) for d in stability_dicts
                ]
                try:
                    stability_result = await asyncio.to_thread(
                        _get_saia_ocr_agent().extract,
                        SaiaOCRRequest(
                            image_id=payload.image_id,
                            page_id=payload.page_id,
                            image_b64=payload.image_b64,
                            script_hint_seed=payload.script_hint_seed,
                            apply_proofread=False,
                            location_suggestions=stability_suggestions,
                        ),
                    )
                    stability_text = str(stability_result.text or "")
                    cross_pass_stab = compute_cross_pass_stability(
                        ocr_payload["text"], stability_text,
                    )
                    ocr_quality_report.cross_pass_stability = cross_pass_stab
                    # Update effective quality and gate decisions with new stability
                    _stab_lex_lang = ocr_payload.get("detected_language", "unknown")
                    _stab_lex_score = _lexical_plausibility(ocr_payload["text"], _stab_lex_lang) if _stab_lex_lang != "unknown" else None
                    gate_decisions = enforce_quality_gates(
                        ocr_quality_report, run_id=run_id,
                        lexical_plausibility=_stab_lex_score,
                    )
                    effective_quality = build_effective_quality(
                        ocr_quality_report, gate_decisions,
                        confidence=ocr_payload["confidence"],
                    )
                    hardened_quality_label = ocr_quality_report.quality_label
                    downstream_mode = gate_decisions["downstream_mode"]
                    log_event(run_id, "CROSS_PASS_STABILITY", "INFO",
                              f"Cross-pass stability computed: {cross_pass_stab:.4f} "
                              f"(threshold={CROSS_PASS_STABILITY_MIN})")

                    # ── Apply uncertainty markers if warranted ────────
                    eff_frag_val = frag_gate_value(
                        ocr_quality_report.leading_fragment_ratio,
                        ocr_quality_report.seam_fragment_ratio,
                    )
                    processed_text, markers_count = apply_uncertainty_markers(
                        ocr_payload["text"],
                        cross_pass_text=stability_text,
                        cross_pass_stability=cross_pass_stab,
                        frag_gate_val=eff_frag_val,
                        uncertainty_dens=ocr_quality_report.uncertainty_density,
                    )
                    if markers_count > 0:
                        ocr_payload["text"] = processed_text
                        ocr_payload["lines"] = processed_text.splitlines()
                        log_event(run_id, "UNCERTAINTY_ENFORCEMENT", "INFO",
                                  f"Inserted {markers_count} uncertainty markers.")
                except Exception as stab_exc:
                    log_event(run_id, "CROSS_PASS_STABILITY", "WARN",
                              f"Stability pass failed: {stab_exc}")

        # Overlay hardened fields
        ocr_payload["quality_label_v2"] = hardened_quality_label
        ocr_payload["downstream_mode"] = downstream_mode
        # Add seam warning if relevant
        if gate_decisions.get("seam_retry_required"):
            ocr_payload["warnings"] = list(dict.fromkeys(
                [*ocr_payload["warnings"], "SEAM_RETRY_REQUIRED"]
            ))

        update_run_fields(
            run_id,
            current_stage="OCR_DONE",
            script_hint=ocr_payload["script_hint"],
            detected_language=ocr_payload["detected_language"],
            confidence=ocr_payload["confidence"],
            warnings_json=json.dumps({
                "warnings": ocr_payload["warnings"],
                "quality_label": quality_label,
                "quality_label_v2": hardened_quality_label,
                "sanity_metrics": ocr_payload.get("sanity_metrics", {}),
            }, ensure_ascii=False),
            ocr_lines_json=json.dumps(ocr_payload["lines"], ensure_ascii=False),
            ocr_text=ocr_payload["text"],
        )

        # ══════════════════════════════════════════════════════════════
        # HARD GATE CHECK — stop if any blocking gate failed
        # If gates never passed after all attempts, mark FAILED_QUALITY
        # and return WITHOUT running downstream stages.
        # ══════════════════════════════════════════════════════════════
        has_blocking = bool(gate_decisions.get("blocked_stages"))

        if not gates_ever_passed and has_blocking:
            blocked_stages = gate_decisions.get("blocked_stages", [])
            failure_reason = (
                f"Quality gates FAILED after {MAX_OCR_ATTEMPTS} attempts. "
                f"quality_label_v2={hardened_quality_label}, "
                f"blocked_stages={blocked_stages}, "
                f"gibberish={ocr_quality_report.gibberish_score:.4f}, "
                f"lead_frag={ocr_quality_report.leading_fragment_ratio:.4f}, "
                f"seam_frag={ocr_quality_report.seam_fragment_ratio:.4f}, "
                f"nwl={ocr_quality_report.non_wordlike_frac:.4f}"
            )
            log_event(run_id, "QUALITY_BLOCKED", "ERROR", failure_reason)
            update_run_fields(
                run_id,
                status="FAILED_QUALITY",
                current_stage="QUALITY_BLOCKED",
                error=failure_reason,
                proofread_text=ocr_payload["text"],  # keep best text
            )

            return {
                "run_id": run_id,
                "status": "FAILED_QUALITY",
                "ocr_result": ocr_payload,
                "proofread_text": None,
                "detected_language": ocr_payload["detected_language"],
                "final_confidence": ocr_payload["confidence"],
                "quality_label": ocr_payload.get("quality_label", quality_label),
                "quality_label_v2": hardened_quality_label,
                "effective_quality": effective_quality.to_dict(),
                "downstream_mode": downstream_mode,
                "sanity_metrics": ocr_payload.get("sanity_metrics", {}),
                "quality_gates": gate_decisions,
                "ocr_attempts": list_ocr_attempts(run_id),
                "failure_reason": failure_reason,
                # Downstream stages NOT executed
                "chunks_count": 0,
                "mentions_count": 0,
                "top_mentions": [],
                "mention_report": None,
                "mention_recall": None,
                "linking_result": None,
                "consolidated_report": None,
            }

        # ══════════════════════════════════════════════════════════════
        # DOWNSTREAM: PROOFREAD (only if gates allow)
        # ══════════════════════════════════════════════════════════════
        log_event(run_id, "PROOFREAD_RUNNING", "START", "Calling proofreader agent.")
        update_run_fields(run_id, current_stage="PROOFREAD_RUNNING")

        proofread_text = ocr_payload["text"]
        if payload.apply_proofread and ocr_payload["text"].strip():
            proofreader = OcrProofreaderAgent(
                client=_get_saia_ocr_agent().client,
                model_override=ocr_result.model_used or None,
            )
            proofread_text = await asyncio.to_thread(
                proofreader.proofread,
                ocr_payload["text"],
                ocr_payload["script_hint"],
                ocr_payload["detected_language"],
            )
            if not str(proofread_text or "").strip():
                proofread_text = ocr_payload["text"]
                log_event(run_id, "PROOFREAD_RUNNING", "INFO",
                          "Proofreader returned empty; using OCR text.")
        else:
            log_event(run_id, "PROOFREAD_RUNNING", "INFO",
                      "Proofreader skipped by request or empty OCR.")

        # ── Proofreading quality guard ────────────────────────────────
        if payload.apply_proofread and str(proofread_text or "").strip() and proofread_text != ocr_payload["text"]:
            guarded_text, proof_accepted, proof_reason = proofreading_quality_guard(
                ocr_payload["text"], proofread_text, ocr_quality_report,
            )
            if not proof_accepted:
                proofread_text = guarded_text
                ocr_payload["warnings"] = list(dict.fromkeys(
                    [*ocr_payload["warnings"], f"PROOFREAD_REJECTED:{proof_reason}"]
                ))
                log_event(run_id, "PROOFREAD_RUNNING", "INFO",
                          f"Proofreading rejected: {proof_reason}")
            else:
                log_event(run_id, "PROOFREAD_RUNNING", "INFO",
                          f"Proofreading accepted: {proof_reason}")

        # ── Post-proofread quality gate (legacy sanity) ───────────────
        final_text_for_gate = str(proofread_text or "").strip()
        if final_text_for_gate:
            post_sanity = compute_sanity(final_text_for_gate)
            if post_sanity["single_char_ratio"] > 0.10 or post_sanity["weird_ratio"] > 0.15:
                gated_lines, mask_count = quality_gate_enforce(
                    [ln for ln in final_text_for_gate.splitlines() if ln.strip()]
                )
                if mask_count > 0:
                    proofread_text = "\n".join(gated_lines).strip()
                    ocr_payload["warnings"] = list(dict.fromkeys(
                        [*ocr_payload["warnings"], f"QUALITY_GATE_MASKED:{mask_count}_lines"]
                    ))
                    log_event(run_id, "PROOFREAD_RUNNING", "INFO",
                              f"Quality gate masked {mask_count} suspicious lines.")
                    post_sanity = compute_sanity(proofread_text)

            forced_lines, forced_count = force_uncertainty_markers(
                [ln for ln in (proofread_text or "").splitlines() if ln.strip()],
                post_sanity,
            )
            if forced_count > 0:
                proofread_text = "\n".join(forced_lines).strip()
                ocr_payload["warnings"] = list(dict.fromkeys(
                    [*ocr_payload["warnings"], f"FORCED_UNCERTAINTY:{forced_count}"]
                ))
                log_event(run_id, "PROOFREAD_RUNNING", "INFO",
                          f"Forced {forced_count} uncertainty markers.")
                post_sanity = compute_sanity(proofread_text)
            quality_label = _quality_label_from_sanity(post_sanity)
            sanity_metrics = post_sanity
            ocr_payload["confidence"] = sanity_adjust_confidence(
                ocr_payload["confidence"], post_sanity
            )
            ocr_payload["quality_label"] = quality_label
            ocr_payload["sanity_metrics"] = sanity_metrics
            sanity_summary = format_sanity_summary(post_sanity)
            ocr_payload["warnings"] = [
                w for w in ocr_payload["warnings"]
                if not str(w).startswith("SANITY ")
            ]
            ocr_payload["warnings"] = list(dict.fromkeys(
                [*ocr_payload["warnings"], sanity_summary]
            ))

        log_event(run_id, "PROOFREAD_DONE", "END", "Proofreading complete.")

        # ── Post-proofread quality report update ──────────────────────
        final_pipeline_text = str(proofread_text or "").strip() or str(ocr_payload["text"] or "")
        if final_pipeline_text and final_pipeline_text != ocr_payload["text"]:
            post_proof_report = compute_quality_report(
                final_pipeline_text,
                run_id=run_id,
                pass_idx=best_attempt_idx + 10,  # offset to distinguish from OCR passes
                previous_pass_tokens=[
                    t for t in (ocr_payload["text"] or "").split() if t
                ],
                language=_quality_language(ocr_payload.get("detected_language")),
            )
            hardened_quality_label = post_proof_report.quality_label
            insert_ocr_quality_report(run_id, post_proof_report.to_dict())
            _pp_lex_lang = ocr_payload.get("detected_language", "unknown")
            _pp_lex_score = _lexical_plausibility(final_pipeline_text, _pp_lex_lang) if _pp_lex_lang != "unknown" else None
            gate_decisions = enforce_quality_gates(
                post_proof_report, run_id=run_id,
                lexical_plausibility=_pp_lex_score,
            )
            downstream_mode = gate_decisions["downstream_mode"]
            ocr_payload["quality_label_v2"] = hardened_quality_label
            ocr_payload["downstream_mode"] = downstream_mode
            ocr_quality_report = post_proof_report
            log_event(run_id, "QUALITY_REPORT", "INFO",
                      f"post_proofread quality_label_v2={hardened_quality_label}")

        update_run_fields(
            run_id,
            current_stage="PROOFREAD_DONE",
            proofread_text=proofread_text,
            confidence=ocr_payload["confidence"],
            warnings_json=json.dumps({
                "warnings": ocr_payload["warnings"],
                "quality_label": ocr_payload.get("quality_label", quality_label),
                "quality_label_v2": hardened_quality_label,
            }, ensure_ascii=False),
        )

        # ══════════════════════════════════════════════════════════════
        # DOWNSTREAM: ANALYZE (check ner_allowed)
        # ══════════════════════════════════════════════════════════════
        if not gate_decisions.get("ner_allowed", True):
            log_event(
                run_id,
                "ANALYZE_DEGRADED",
                "INFO",
                f"NER gate blocked full analysis; running degraded mention capture only "
                f"(quality={hardened_quality_label})",
            )
            update_run_fields(run_id, current_stage="ANALYZE_RUNNING")
            base_text_source = "proofread" if str(proofread_text or "").strip() else "ocr"
            base_text = str(proofread_text or "").strip() if base_text_source == "proofread" else str(ocr_payload["text"] or "")
            chunks, mentions, _candidates, salvage_debug = _run_trace_analysis(run_id, base_text)
            salvage_debug = dict(salvage_debug or {})
            salvage_debug["degraded_mode"] = True
            salvage_debug["reason"] = "ner_not_allowed_degraded_capture"
            chunks_count = len(chunks)
            mentions_count = len(mentions)
            update_run_fields(
                run_id,
                current_stage="ANALYZE_DONE",
                base_text_source=base_text_source,
                chunks_count=chunks_count,
                mentions_count=mentions_count,
            )
            log_event(run_id, "ANALYZE_DONE", "END",
                      f"degraded_capture chunks={chunks_count}, mentions={mentions_count}")
        else:
            log_event(run_id, "ANALYZE_RUNNING", "START",
                      "Running chunk/entity analysis.")
            update_run_fields(run_id, current_stage="ANALYZE_RUNNING")

            base_text_source = "proofread" if str(proofread_text or "").strip() else "ocr"
            base_text = str(proofread_text or "").strip() if base_text_source == "proofread" else str(ocr_payload["text"] or "")
            chunks, mentions, _candidates, salvage_debug = _run_trace_analysis(
                run_id, base_text,
            )
            chunks_count = len(chunks)
            mentions_count = len(mentions)

            # ── Mention recall check + high-recall fallback ───────────
            mention_recall_result = check_mention_recall(
                base_text, mentions_count, hardened_quality_label,
            )
            if not mention_recall_result["mention_recall_ok"]:
                ocr_payload["warnings"] = list(dict.fromkeys(
                    [*ocr_payload["warnings"], mention_recall_result["reason"]]
                ))
                log_event(run_id, "QUALITY_GATES", "INFO",
                          mention_recall_result["reason"])

            if mention_recall_result.get("trigger_high_recall"):
                hr_mentions = extract_high_recall_mentions(
                    base_text,
                    script=ocr_quality_report.script_family,
                    quality_label=hardened_quality_label,
                )
                if hr_mentions:
                    existing_spans = {
                        (m.get("start_offset"), m.get("end_offset"))
                        for m in mentions
                    }
                    new_hr = [
                        m for m in hr_mentions
                        if (m.get("start_offset"), m.get("end_offset")) not in existing_spans
                    ]
                    if new_hr:
                        for m in new_hr:
                            m["chunk_id"] = _assign_chunk_id_for_span(
                                int(m.get("start_offset", 0)),
                                int(m.get("end_offset", 0)),
                                chunks,
                            )
                        new_rows = insert_entity_mentions(run_id, new_hr)
                        mentions.extend(new_rows)
                        mentions_count = len(mentions)
                        log_event(run_id, "ANALYZE_RUNNING", "INFO",
                                  f"High-recall extraction added {len(new_rows)} mentions.")
                        salvage_debug["high_recall_added"] = len(new_rows)

            update_run_fields(
                run_id,
                current_stage="ANALYZE_DONE",
                base_text_source=base_text_source,
                chunks_count=chunks_count,
                mentions_count=mentions_count,
            )
            log_event(run_id, "ANALYZE_DONE", "END",
                      f"chunks={chunks_count}, mentions={mentions_count}")

        # Build mention extraction report
        mention_report = _build_mention_extraction_report(
            run_id, asset_ref, mentions, salvage_debug,
        ) if mentions or not salvage_debug.get("skipped") else None

        # ── Mention recall (for response) ─────────────────────────────
        if gate_decisions.get("ner_allowed", True):
            base_text = str(proofread_text or "").strip() or str(ocr_payload["text"] or "")
            mention_recall = check_mention_recall(
                base_text, mentions_count, hardened_quality_label,
            )
        else:
            mention_recall = {
                "mention_recall_ok": True,
                "reason": "Degraded mention capture only (ner_allowed=False)",
                "trigger_high_recall": False,
            }

        log_event(run_id, "STORED", "END", "Run results persisted in SQLite.")
        update_run_fields(run_id, current_stage="STORED", proofread_text=proofread_text)

        # ══════════════════════════════════════════════════════════════
        # DOWNSTREAM: AUTHORITY LINKING (check token_search_allowed)
        # ══════════════════════════════════════════════════════════════
        if not gate_decisions.get("token_search_allowed", True):
            log_event(run_id, "LINKING_SKIPPED", "INFO",
                      f"Authority linking SKIPPED: token_search_allowed=False "
                      f"(quality={hardened_quality_label})")
            linking_result, _authority_report = _persist_deferred_authority_links(
                run_id,
                reason=f"token_search_allowed=False quality={hardened_quality_label}",
            )
        else:
            # Authority linking is synchronous: urllib with 10-15s timeouts per request
            # plus a 0.25s time.sleep rate limiter between them. Called directly from an
            # async function it blocks the one event loop for the whole stage, freezing
            # every other request on the process - the trace endpoint, the SSE stream,
            # /health - not just this run. The GLM path already does this via to_thread.
            linking_result = await asyncio.to_thread(_run_authority_linking_stage, run_id)

        # Build consolidated report
        consolidated_report = _build_consolidated_report(
            run_id, asset_ref, mentions, salvage_debug, linking_result,
        ) if mentions or linking_result else None

        # ══════════════════════════════════════════════════════════════
        # DOWNSTREAM: AUTO-INDEX (check token_search_allowed)
        # ══════════════════════════════════════════════════════════════
        _record_search_decision(run_id, bool(gate_decisions.get("token_search_allowed", True)))
        if not gate_decisions.get("token_search_allowed", True):
            log_event(run_id, "INDEX_SKIPPED", "INFO",
                      f"RAG auto-index SKIPPED: token_search_allowed=False "
                      f"(quality={hardened_quality_label})")
        else:
            _auto_index_run(run_id)

        log_event(run_id, "DONE", "END", "Pipeline completed successfully.")
        update_run_fields(run_id, status="COMPLETED", current_stage="DONE", error=None)

        return {
            "run_id": run_id,
            "status": "COMPLETED",
            "ocr_result": ocr_payload,
            "proofread_text": proofread_text,
            "detected_language": ocr_payload["detected_language"],
            "final_confidence": ocr_payload["confidence"],
            "quality_label": ocr_payload.get("quality_label", quality_label),
            "quality_label_v2": hardened_quality_label,
            "effective_quality": effective_quality.to_dict(),
            "downstream_mode": downstream_mode,
            "sanity_metrics": ocr_payload.get("sanity_metrics", {}),
            "quality_gates": gate_decisions,
            "ocr_attempts": list_ocr_attempts(run_id),
            "mention_recall": mention_recall,
            "chunks_count": chunks_count,
            "mentions_count": mentions_count,
            "top_mentions": mentions[:20],
            "mention_report": mention_report,
            "linking_result": linking_result,
            "consolidated_report": consolidated_report,
        }
    except Exception as exc:
        err = str(exc)
        update_run_fields(run_id, status="FAILED", current_stage="FAILED", error=err)
        log_event(run_id, "FAILED", "ERROR", err)
        raise HTTPException(status_code=500, detail={"run_id": run_id, "error": err}) from exc


@router.get("/ocr/trace/{run_id}")
async def ocr_trace_snapshot(run_id: str) -> dict[str, Any]:
    snapshot = _build_trace_snapshot(run_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return snapshot


@router.get("/ocr/trace/stream/{run_id}")
async def ocr_trace_stream(run_id: str) -> StreamingResponse:
    if get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    async def event_stream():
        last_signature: tuple[Any, ...] | None = None
        while True:
            snapshot = _build_trace_snapshot(run_id)
            if snapshot is None:
                payload = {"run_id": run_id, "status": "FAILED", "error": "Run disappeared from database."}
                yield f"event: snapshot\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                break

            run = snapshot.get("run") or {}
            events = snapshot.get("events") or []
            signature = (
                run.get("updated_at"),
                run.get("current_stage"),
                run.get("status"),
                len(events),
            )
            if signature != last_signature:
                yield f"event: snapshot\ndata: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
                last_signature = signature

            if str(run.get("status") or "").upper() in {"COMPLETED", "FAILED"}:
                break
            await asyncio.sleep(0.4)

        yield "event: end\ndata: done\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/ocr/trace/{run_id}/tables")
async def ocr_trace_tables(run_id: str) -> dict[str, Any]:
    if get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return {
        "run_id": run_id,
        "tables": [
            table_view_for_run(run_id),
            table_view_for_events(run_id),
            table_view_for_ocr_backend_results(run_id),
            table_view_for_chunks(run_id),
            table_view_for_entity_mentions(run_id),
            table_view_for_entity_candidates(run_id),
            table_view_for_entity_decisions(run_id),
            table_view_for_entity_attempts(run_id),
        ],
    }
