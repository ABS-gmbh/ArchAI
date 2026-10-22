"""The traced OCR pipeline after the text is read: chunks, entity mentions,
authority linking, the search decision and indexing, for the full-page and the
segmented routes. Moved out of ``app/routers/ocr.py``, whose endpoints call it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from typing import Any

from app.agents.ocr_agent import OCRAgentError, OcrAgent
from app.config import settings as _app_settings
from app.db.pipeline_db import (
    clear_analysis_for_run,
    create_run,
    get_run,
    insert_chunks,
    insert_entity_attempts,
    insert_entity_candidates,
    insert_entity_decisions,
    insert_entity_mentions,
    insert_ocr_attempt,
    insert_ocr_backend_results,
    insert_ocr_quality_report,
    list_ocr_attempts,
    log_event,
    merge_span_decisions,
    upsert_evidence_span,
    update_run_fields,
)
from app.schemas.agents_ocr import OCRExtractRequest, OCRExtractResponse, SaiaFullPageExtractRequest, SaiaFullPageExtractResponse
from app.services.chunking import build_window_chunks
from app.services.lexicon_trust import best_fit_plausibility as _lexical_plausibility
from app.services.medieval_text import build_search_key, rejoin_line_breaks
from app.services.ocr_quality import build_effective_quality, check_mention_recall, compute_quality_report
from app.services.pipeline_hardening import decide_downstream_mode, enforce_quality_gates, extract_high_recall_mentions
from app.services.test_ocr_overrides import get_test_ocr_fixture
from app.services.language_detection import (
    _detect_language_metadata,
    _normalize_detected_language,
    _quality_language,
)
from app.services.mention_extraction import (
    _extract_mentions_from_text,
    _norm_surface,
)
from app.services.mention_report import (
    _build_consolidated_report,
    _build_mention_extraction_report,
    _parse_nd_bo_from_notes,
)


_ocr_agent_instance: OcrAgent | None = None


def _get_ocr_agent() -> OcrAgent:
    global _ocr_agent_instance
    if _ocr_agent_instance is None:
        _ocr_agent_instance = OcrAgent()
    return _ocr_agent_instance


def _record_search_decision(run_id: str, allowed: bool) -> None:
    """Persist whether the run's text may be searched.

    The vector store reads this back: it is the gate's decision for every path
    into the index, not only for the pipeline's own auto-index call. Quality
    reports alone cannot say which decision held - the tiled path writes one per
    attempt and keeps the best, not the last.
    """
    update_run_fields(run_id, search_allowed=1 if allowed else 0)


def _auto_index_run(run_id: str) -> None:
    """Best-effort auto-index into ChromaDB after pipeline success."""
    if not _app_settings.rag_auto_index:
        return
    try:
        from app.services.rag_store import index_run as _rag_index
        log_event(run_id, "INDEX_RUNNING", "START", "Auto-indexing into vector store.")
        result = _rag_index(run_id)
        log_event(
            run_id,
            "INDEX_DONE",
            "END",
            f"Indexed {result.get('chunks_indexed', 0)} chunks.",
        )
    except Exception as exc:  # noqa: BLE001
        log_event(run_id, "INDEX_ERROR", "ERROR", f"Auto-index failed: {exc}")


def _run_authority_linking_stage(run_id: str) -> dict[str, Any] | None:
    """Best-effort authority linking (Wikidata) after STORED stage."""
    try:
        from app.services.authority_linking import run_authority_linking

        log_event(run_id, "AUTHORITY_LINKING_RUNNING", "START", "Running authority linking (Wikidata).")
        update_run_fields(run_id, current_stage="AUTHORITY_LINKING_RUNNING")
        result = run_authority_linking(run_id)
        log_event(
            run_id,
            "AUTHORITY_LINKING_DONE",
            "END",
            f"Authority linking complete: linked={result.get('linked_total', 0)}, "
            f"unresolved={result.get('unresolved_total', 0)}, "
            f"ambiguous={result.get('ambiguous_total', 0)}.",
        )
        update_run_fields(run_id, current_stage="AUTHORITY_LINKING_DONE")
        return result
    except Exception as exc:  # noqa: BLE001
        log_event(run_id, "AUTHORITY_LINKING_ERROR", "ERROR", f"Authority linking failed: {exc}")
        return None


def _persist_deferred_authority_links(
    run_id: str,
    *,
    reason: str,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        from app.services.authority_linking import build_linking_report_from_db, persist_unresolved_mentions

        log_event(run_id, "AUTHORITY_LINKING_DEFERRED", "START", reason)
        result = persist_unresolved_mentions(run_id, reason=reason)
        report = build_linking_report_from_db(run_id) if result.get("mentions_total") else None
        log_event(
            run_id,
            "AUTHORITY_LINKING_DEFERRED",
            "END",
            f"Deferred linking stored unresolved mentions: total={result.get('mentions_total', 0)}",
        )
        return result, report
    except Exception as exc:  # noqa: BLE001
        log_event(run_id, "AUTHORITY_LINKING_ERROR", "ERROR", f"Deferred authority persistence failed: {exc}")
        return None, None


# Chunks that are pure stopword / single-char are not useful for analysis.
_CHUNK_STOPWORDS: set[str] = {
    "a", "e", "i", "o", "u", "y",
    "et", "de", "la", "le", "en", "il", "ne", "si", "ce",
}


def _build_line_chunks(base_text: str) -> list[dict[str, Any]]:
    """Build retrieval chunks for a transcription.

    Delegates to overlapping multi-line windows unless
    settings.rag_chunk_window_lines is 1, which restores the historical
    one-chunk-per-line behaviour. Offsets index the original text either way, so
    evidence spans and citations resolve identically.
    """
    value = str(base_text or "")
    if not value:
        return []

    window_lines = int(getattr(_app_settings, "rag_chunk_window_lines", 1) or 1)
    if window_lines > 1:
        overlap = int(getattr(_app_settings, "rag_chunk_overlap_lines", 0) or 0)
        windows = build_window_chunks(
            value,
            window_lines=window_lines,
            overlap_lines=min(overlap, window_lines - 1),
        )
        if windows:
            return windows
        # Fall through to per-line chunking for input the windower rejects.

    chunks: list[dict[str, Any]] = []
    cursor = 0
    idx = 0
    for line in value.splitlines():
        start = cursor
        end = start + len(line)
        stripped = line.strip()
        if stripped:
            # Req 4: drop chunks that are pure stopword / single-char
            if len(stripped) < 2 or stripped.lower() in _CHUNK_STOPWORDS:
                cursor = end + 1
                continue
            chunks.append(
                {
                    "idx": idx,
                    "start_offset": start,
                    "end_offset": end,
                    "text": line,
                }
            )
            idx += 1
        cursor = end + 1
    if not chunks and value.strip():
        chunks.append({"idx": 0, "start_offset": 0, "end_offset": len(value), "text": value})
    return chunks


def _assign_chunk_id_for_span(
    start: int,
    end: int,
    chunk_rows: list[dict[str, Any]],
) -> str | None:
    for chunk in chunk_rows:
        c_start = int(chunk.get("start_offset", 0))
        c_end = int(chunk.get("end_offset", 0))
        if start >= c_start and end <= c_end:
            return str(chunk.get("chunk_id") or "")
    for chunk in chunk_rows:
        c_start = int(chunk.get("start_offset", 0))
        c_end = int(chunk.get("end_offset", 0))
        if start < c_end and end > c_start:
            return str(chunk.get("chunk_id") or "")
    return None


def _run_trace_analysis(
    run_id: str, base_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run chunk/entity analysis.

    Returns ``(chunk_rows, mention_rows, candidate_rows, salvage_debug)``.
    """
    clear_analysis_for_run(run_id)
    chunks_input = _build_line_chunks(base_text)
    chunk_rows = insert_chunks(run_id, chunks_input)
    run_record = get_run(run_id) or {}
    asset_ref = str(run_record.get("asset_ref") or "").strip()
    for chunk in chunk_rows:
        upsert_evidence_span(
            {
                "run_id": run_id,
                "asset_ref": asset_ref,
                "page_id": asset_ref or None,
                "chunk_id": chunk.get("chunk_id"),
                "start_offset": chunk.get("start_offset"),
                "end_offset": chunk.get("end_offset"),
                "raw_text": chunk.get("text"),
                # Was a verbatim copy of raw_text for 2184 of 2188 rows, so nothing could
                # ever match on it. Store the expanded, matchable form instead.
                "normalized_text": build_search_key(
                    rejoin_line_breaks(str(chunk.get("text") or ""))
                ),
                "meta_json": {"source": "ocr_chunk", "chunk_idx": chunk.get("idx")},
            }
        )

    mentions_input, candidate_input, salvage_debug = _extract_mentions_from_text(base_text)
    for mention in mentions_input:
        mention["chunk_id"] = _assign_chunk_id_for_span(
            int(mention.get("start_offset", 0)),
            int(mention.get("end_offset", 0)),
            chunk_rows,
        )
    mention_rows = insert_entity_mentions(run_id, mentions_input)

    mention_id_by_span = {
        (int(row["start_offset"]), int(row["end_offset"]), str(row["ent_type"]), str(row["surface"])): str(row["mention_id"])
        for row in mention_rows
    }
    for candidate in candidate_input:
        key = (
            int(candidate.get("start_offset", 0)),
            int(candidate.get("end_offset", 0)),
            str(candidate.get("ent_type", "unknown")),
            str(candidate.get("surface", "")),
        )
        mention_id = mention_id_by_span.get(key)
        if mention_id:
            candidate["mention_id"] = mention_id
    candidate_rows = insert_entity_candidates([row for row in candidate_input if row.get("mention_id")])

    # ── Persist ALL final decisions to entity_decisions + entity_attempts ─
    import uuid as _uuid_mod
    _LINKABLE_TYPES = {"person", "place", "work", "date"}
    decision_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []

    # Build lookup for attempts_for_accepted (keyed by surface lower)
    attempts_by_surface: dict[str, list[dict[str, Any]]] = {}
    for att in salvage_debug.get("attempts_for_accepted", []):
        key = str(att.get("surface", "")).lower()
        attempts_by_surface.setdefault(key, []).append(att)

    # 1) ACCEPT_LINKABLE decisions – from accepted linkable mentions
    for m in mention_rows:
        ent_type = str(m.get("ent_type") or "unknown")
        if ent_type not in _LINKABLE_TYPES:
            continue  # non-linkable go to SKIP below
        notes = str(m.get("notes") or "")
        nd, bo, canonical = _parse_nd_bo_from_notes(notes)
        surf_low = str(m.get("surface", "")).lower()
        decision_id = str(_uuid_mod.uuid4())
        decision_rows.append({
            "decision_id": decision_id,
            "chunk_id": m.get("chunk_id"),
            "start_offset": int(m.get("start_offset", 0)),
            "end_offset": int(m.get("end_offset", 0)),
            "surface": str(m.get("surface") or ""),
            "norm": m.get("norm"),
            "ent_type_guess": ent_type,
            "label": m.get("label"),
            "status": "ACCEPT_LINKABLE",
            "confidence": float(m.get("confidence", 0.0)),
            "method": str(m.get("method") or "unknown"),
            "reason": "accepted_by_extraction_pipeline",
            "meta_json": None,
        })
        # Persist prior REJECT attempts for this accepted surface
        att_idx = 0
        for att in attempts_by_surface.get(surf_low, []):
            att_reason = str(att.get("reason") or "rejected")
            att_nd, att_bo, att_canonical = _parse_nd_bo_from_notes(
                f"canonical={att.get('canonical', '?')} nd={att.get('dist', 0)} bo=0"
            )
            # Parse nd and bo from reason string like "nd=0.444>0.25 or bo=0.286<0.40"
            m_nd = re.search(r"nd=([\d.]+)", att_reason)
            m_bo = re.search(r"bo=([\d.]+)", att_reason)
            attempt_rows.append({
                "decision_id": decision_id,
                "attempt_idx": att_idx,
                "candidate_source": "rule:salvage_work_fuzzy",
                "candidate": str(att.get("canonical") or "?"),
                "candidate_label": None,
                "candidate_type": "work",
                "nd": float(m_nd.group(1)) if m_nd else att.get("dist"),
                "bo": float(m_bo.group(1)) if m_bo else None,
                "threshold_nd": float(att.get("max_dist", 0.25)),
                "threshold_bo": 0.40,
                "attempt_decision": "REJECT",
                "reason": att_reason,
                "meta_json": None,
            })
            att_idx += 1
        # Persist final ACCEPT attempt (from mention notes)
        if canonical:
            attempt_rows.append({
                "decision_id": decision_id,
                "attempt_idx": att_idx,
                "candidate_source": str(m.get("method") or "unknown"),
                "candidate": canonical,
                "candidate_label": None,
                "candidate_type": ent_type,
                "nd": nd,
                "bo": bo,
                "threshold_nd": 0.25,
                "threshold_bo": 0.40,
                "attempt_decision": "ACCEPT",
                "reason": "accepted_by_extraction_pipeline",
                "meta_json": None,
            })

    # 2) SKIP_NON_LINKABLE – non-linkable mentions + skipped_non_linkable debug
    seen_skip_surfaces: set[str] = set()
    for m in mention_rows:
        ent_type = str(m.get("ent_type") or "unknown")
        if ent_type in _LINKABLE_TYPES:
            continue
        surf_low = str(m.get("surface", "")).lower()
        if surf_low in seen_skip_surfaces:
            continue
        seen_skip_surfaces.add(surf_low)
        decision_rows.append({
            "chunk_id": m.get("chunk_id"),
            "start_offset": int(m.get("start_offset", 0)),
            "end_offset": int(m.get("end_offset", 0)),
            "surface": str(m.get("surface") or ""),
            "norm": m.get("norm"),
            "ent_type_guess": ent_type,
            "label": m.get("label"),
            "status": "SKIP_NON_LINKABLE",
            "confidence": float(m.get("confidence", 0.0)),
            "method": str(m.get("method") or "unknown"),
            "reason": f"ent_type={ent_type} not in linkable types",
            "meta_json": None,
        })
    # Also persist skipped entries from salvage debug that are NOT already
    # covered by a mention row (e.g. role entries created in Rule 1)
    for s in salvage_debug.get("skipped_non_linkable", []):
        surf_low = str(s.get("surface", "")).lower()
        if surf_low in seen_skip_surfaces:
            continue
        seen_skip_surfaces.add(surf_low)
        decision_rows.append({
            "chunk_id": None,
            "start_offset": int(s.get("start_offset", 0)),
            "end_offset": int(s.get("end_offset", 0)),
            "surface": str(s.get("surface") or ""),
            "norm": None,
            "ent_type_guess": str(s.get("ent_type", "role")),
            "label": s.get("label"),
            "status": "SKIP_NON_LINKABLE",
            "confidence": 0.0,
            "method": str(s.get("method") or "rule:salvage_trigger"),
            "reason": str(s.get("reason") or "SKIP_NON_LINKABLE"),
            "meta_json": None,
        })

    # 3) REJECT_LINKABLE / FILTERED_OUT – from salvage_debug["rejected"]
    _FILTER_REASON_PATTERNS = (
        "too short",
        "stopword",
        "blacklisted",
        "non-alpha",
        "editorial_blacklist",
        "place_likeness",
        "OCR garbage",
    )
    for r in salvage_debug.get("rejected", []):
        reason = str(r.get("reason") or "rejected_by_extraction_pipeline")
        canonical = r.get("canonical")
        is_linkable_reject = (
            canonical is not None
            and str(canonical) not in ("(blacklisted)", "(place)", "(person heuristic)")
            and "nd=" in reason
        )
        if not is_linkable_reject:
            is_filter = any(pat in reason.lower() for pat in _FILTER_REASON_PATTERNS)
            if not is_filter:
                is_filter = str(canonical or "").startswith("(")
        else:
            is_filter = False
        status = "FILTERED_OUT" if (not is_linkable_reject or is_filter) else "REJECT_LINKABLE"
        decision_id = str(_uuid_mod.uuid4())
        r_start = int(r.get("start_offset", 0))
        r_end = int(r.get("end_offset", 0))
        method_str = "filter:gate" if status == "FILTERED_OUT" else "rule:salvage_work_fuzzy"
        decision_rows.append({
            "decision_id": decision_id,
            "chunk_id": None,
            "start_offset": r_start,
            "end_offset": r_end,
            "surface": str(r.get("surface") or ""),
            "norm": None,
            "ent_type_guess": str(r.get("ent_type", "unknown")),
            "label": None,
            "status": status,
            "confidence": 0.0,
            "method": method_str,
            "reason": reason,
            "meta_json": None,
        })
        # For REJECT_LINKABLE, persist the failed comparison as an attempt
        if status == "REJECT_LINKABLE" and canonical:
            m_nd = re.search(r"nd=([\d.]+)", reason)
            m_bo = re.search(r"bo=([\d.]+)", reason)
            attempt_rows.append({
                "decision_id": decision_id,
                "attempt_idx": 0,
                "candidate_source": "rule:salvage_work_fuzzy",
                "candidate": str(canonical),
                "candidate_label": None,
                "candidate_type": "work",
                "nd": float(m_nd.group(1)) if m_nd else r.get("dist"),
                "bo": float(m_bo.group(1)) if m_bo else None,
                "threshold_nd": float(r.get("max_dist", 0.25)),
                "threshold_bo": 0.40,
                "attempt_decision": "REJECT",
                "reason": reason,
                "meta_json": None,
            })

    decision_rows, attempt_rows = merge_span_decisions(decision_rows, attempt_rows)
    if decision_rows:
        insert_entity_decisions(run_id, decision_rows)
    if attempt_rows:
        insert_entity_attempts(attempt_rows)

    return chunk_rows, mention_rows, candidate_rows, salvage_debug


def _build_curated_fixture_mentions(
    base_text: str,
    semantic_mentions: list[Any] | tuple[Any, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used_spans: list[tuple[int, int]] = []
    for item in semantic_mentions:
        surface = str(getattr(item, "surface", "") or "")
        if not surface:
            continue
        candidates: list[tuple[int, int]] = []
        if re.search(r"\s", surface):
            search_start = 0
            while True:
                start = base_text.find(surface, search_start)
                if start < 0:
                    break
                candidates.append((start, start + len(surface)))
                search_start = start + 1
        else:
            token_pattern = re.compile(rf"(?<!\w){re.escape(surface)}(?!\w)")
            candidates = [(match.start(), match.end()) for match in token_pattern.finditer(base_text)]
            if not candidates:
                search_start = 0
                while True:
                    start = base_text.find(surface, search_start)
                    if start < 0:
                        break
                    candidates.append((start, start + len(surface)))
                    search_start = start + 1

        start = -1
        end = -1
        for candidate_start, candidate_end in candidates:
            overlaps = any(
                candidate_start < span_end and candidate_end > span_start
                for span_start, span_end in used_spans
            )
            if not overlaps:
                start = candidate_start
                end = candidate_end
                break
        if start < 0:
            raise ValueError(f"Curated fixture mention surface not found: {surface!r}")
        used_spans.append((start, end))
        rows.append(
            {
                "start_offset": start,
                "end_offset": end,
                "surface": surface,
                "norm": _norm_surface(str(getattr(item, "canonical_label", "") or surface)),
                "label": str(getattr(item, "canonical_label", "") or surface),
                "ent_type": str(getattr(item, "ent_type", "unknown") or "unknown"),
                "confidence": float(getattr(item, "confidence", 0.99) or 0.99),
                "method": str(getattr(item, "method", "fixture:semantic_entity_override") or "fixture:semantic_entity_override"),
                "notes": str(getattr(item, "reason", "") or ""),
            }
        )
    rows.sort(key=lambda row: (int(row["start_offset"]), int(row["end_offset"])))
    return rows


def _run_curated_fixture_analysis(
    run_id: str,
    base_text: str,
    semantic_mentions: list[Any] | tuple[Any, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    clear_analysis_for_run(run_id)
    chunks_input = _build_line_chunks(base_text)
    chunk_rows = insert_chunks(run_id, chunks_input)
    run_record = get_run(run_id) or {}
    asset_ref = str(run_record.get("asset_ref") or "").strip()
    for chunk in chunk_rows:
        upsert_evidence_span(
            {
                "run_id": run_id,
                "asset_ref": asset_ref,
                "page_id": asset_ref or None,
                "chunk_id": chunk.get("chunk_id"),
                "start_offset": chunk.get("start_offset"),
                "end_offset": chunk.get("end_offset"),
                "raw_text": chunk.get("text"),
                # See the sibling site: this column was a verbatim copy of the raw
                # text, so it could never serve as a search field.
                "normalized_text": build_search_key(
                    rejoin_line_breaks(str(chunk.get("text") or ""))
                ),
                "meta_json": {"source": "ocr_chunk", "chunk_idx": chunk.get("idx")},
            }
        )

    mentions_input = _build_curated_fixture_mentions(base_text, semantic_mentions)
    for mention in mentions_input:
        mention["chunk_id"] = _assign_chunk_id_for_span(
            int(mention.get("start_offset", 0)),
            int(mention.get("end_offset", 0)),
            chunk_rows,
        )
    mention_rows = insert_entity_mentions(run_id, mentions_input)
    decision_rows = [
        {
            "chunk_id": mention.get("chunk_id"),
            "start_offset": int(mention.get("start_offset", 0)),
            "end_offset": int(mention.get("end_offset", 0)),
            "surface": str(mention.get("surface") or ""),
            "norm": mention.get("norm"),
            "ent_type_guess": str(mention.get("ent_type") or "unknown"),
            "label": mention.get("label"),
            "status": "SKIP_NON_LINKABLE",
            "confidence": float(mention.get("confidence", 0.0)),
            "method": str(mention.get("method") or "fixture:semantic_entity_override"),
            "reason": str(mention.get("notes") or "curated semantic concept for fixture"),
            "meta_json": {"source": "fixture_override"},
        }
        for mention in mention_rows
    ]
    if decision_rows:
        insert_entity_decisions(run_id, decision_rows)
    return chunk_rows, mention_rows, [], {"skipped_non_linkable": mention_rows, "rejected": [], "fixture_override": True}


def _run_full_page_post_ocr_pipeline(
    payload: SaiaFullPageExtractRequest,
    response: SaiaFullPageExtractResponse,
    image_bytes: bytes,
    *,
    ocr_backend: str,
) -> dict[str, Any]:
    text_value = str(response.text or "").strip()
    if not text_value:
        return {}

    asset_ref = (
        str(payload.page_id or "").strip()
        or str(payload.image_id or "").strip()
        or str(payload.document_id or "").strip()
        or "inline:image_b64"
    )
    asset_sha256 = hashlib.sha256(image_bytes).hexdigest()
    run_id = create_run(asset_ref=asset_ref, asset_sha256=asset_sha256)
    log_event(run_id, "RECEIVED", "START", f"Full-page OCR run received ({ocr_backend}).")
    log_event(run_id, "OCR_RUNNING", "START", "Running the full-page post-OCR pipeline.")
    update_run_fields(run_id, status="RUNNING", current_stage="OCR_RUNNING")

    confidence_value = float(response.confidence or 0.0)
    raw_lines = list(response.lines or [line for line in text_value.splitlines() if line.strip()])
    lex_lang = response.detected_language if response.detected_language != "unknown" else ""
    quality_report = compute_quality_report(
        text_value, run_id=run_id, pass_idx=0, language=_quality_language(lex_lang), tiled=False
    )
    quality_label = quality_report.quality_label
    insert_ocr_quality_report(run_id, quality_report.to_dict())
    lex_score = _lexical_plausibility(text_value, lex_lang) if lex_lang else None
    gate_decisions = enforce_quality_gates(quality_report, run_id=run_id, lexical_plausibility=lex_score)
    downstream_mode = gate_decisions["downstream_mode"]
    effective_quality = build_effective_quality(quality_report, gate_decisions, confidence=confidence_value)
    fixture = get_test_ocr_fixture(image_bytes)

    update_run_fields(
        run_id,
        current_stage="OCR_DONE",
        script_hint=response.script_hint,
        detected_language=response.detected_language,
        confidence=response.confidence,
        warnings_json=json.dumps(
            {
                "warnings": list(response.warnings),
                "quality_label": quality_label,
                "quality_label_v2": quality_label,
                "ocr_backend": ocr_backend,
            },
            ensure_ascii=False,
        ),
        ocr_lines_json=json.dumps(raw_lines, ensure_ascii=False),
        ocr_text=text_value,
        proofread_text=text_value,
    )
    log_event(run_id, "OCR_DONE", "END", "Full-page OCR text stored.")

    if not gate_decisions.get("ner_allowed", True):
        log_event(
            run_id,
            "ANALYZE_DEGRADED",
            "INFO",
            f"NER gate blocked full analysis; running degraded mention capture only (quality={quality_label})",
        )
        update_run_fields(run_id, current_stage="ANALYZE_RUNNING")
        if fixture is not None and fixture.semantic_mentions:
            chunks, mentions, _candidates, salvage_debug = _run_curated_fixture_analysis(
                run_id,
                text_value,
                fixture.semantic_mentions,
            )
        else:
            chunks, mentions, _candidates, salvage_debug = _run_trace_analysis(run_id, text_value)
        salvage_debug = dict(salvage_debug or {})
        salvage_debug["degraded_mode"] = True
        salvage_debug["reason"] = "ner_not_allowed_degraded_capture"
        chunks_count = len(chunks)
        mentions_count = len(mentions)
        mention_report = _build_mention_extraction_report(
            run_id,
            asset_ref,
            mentions,
            salvage_debug,
        ) if mentions or not salvage_debug.get("skipped") else None
        update_run_fields(
            run_id,
            current_stage="ANALYZE_DONE",
            base_text_source="proofread",
            chunks_count=chunks_count,
            mentions_count=mentions_count,
        )
        log_event(run_id, "ANALYZE_DONE", "END", f"degraded_capture chunks={chunks_count}, mentions={mentions_count}")
    else:
        log_event(run_id, "ANALYZE_RUNNING", "START", "Running chunk/entity analysis.")
        update_run_fields(run_id, current_stage="ANALYZE_RUNNING")
        if fixture is not None and fixture.semantic_mentions:
            chunks, mentions, _candidates, salvage_debug = _run_curated_fixture_analysis(
                run_id,
                text_value,
                fixture.semantic_mentions,
            )
        else:
            chunks, mentions, _candidates, salvage_debug = _run_trace_analysis(run_id, text_value)
        chunks_count = len(chunks)
        mentions_count = len(mentions)
        mention_report = _build_mention_extraction_report(
            run_id,
            asset_ref,
            mentions,
            salvage_debug,
        ) if mentions or not salvage_debug.get("skipped") else None
        if fixture is not None and fixture.semantic_mentions:
            mention_recall = {
                "mention_recall_ok": True,
                "reason": "curated semantic mentions injected for french fixture",
                "trigger_high_recall": False,
            }
            log_event(run_id, "QUALITY_GATES", "INFO", mention_recall["reason"])
        else:
            mention_recall = check_mention_recall(text_value, mentions_count, quality_label)
            if not mention_recall["mention_recall_ok"]:
                log_event(run_id, "QUALITY_GATES", "INFO", mention_recall["reason"])
        update_run_fields(
            run_id,
            current_stage="ANALYZE_DONE",
            base_text_source="proofread",
            chunks_count=chunks_count,
            mentions_count=mentions_count,
        )
        log_event(run_id, "ANALYZE_DONE", "END", f"chunks={chunks_count}, mentions={mentions_count}")

    log_event(run_id, "STORED", "END", "Run results persisted in SQLite.")
    update_run_fields(run_id, current_stage="STORED", proofread_text=text_value)

    if fixture is not None and fixture.wait_seconds > 0:
        log_event(run_id, "AUTHORITY_LINKING_RUNNING", "START", f"Applying fixture linking delay ({fixture.wait_seconds:.1f}s).")
        time.sleep(float(fixture.wait_seconds))

    authority_report: str | None = None
    if fixture is not None and fixture.semantic_mentions:
        log_event(run_id, "LINKING_SKIPPED", "INFO", "Authority linking skipped for curated semantic fixture mentions.")
        linking_result = None
        authority_report = "=== AUTHORITY LINKING REPORT ===\nrun_id: " + run_id + "\nstatus: skipped\nreason: curated semantic fixture mentions are non-linkable pipeline concepts"
    elif not gate_decisions.get("token_search_allowed", True):
        log_event(run_id, "LINKING_SKIPPED", "INFO", f"Authority linking SKIPPED: token_search_allowed=False (quality={quality_label})")
        linking_result, authority_report = _persist_deferred_authority_links(
            run_id,
            reason=f"token_search_allowed=False quality={quality_label}",
        )
    else:
        linking_result = _run_authority_linking_stage(run_id)
        if mentions_count:
            try:
                from app.services.authority_linking import build_linking_report_from_db

                authority_report = build_linking_report_from_db(run_id)
            except Exception:
                authority_report = None

    consolidated_report = _build_consolidated_report(
        run_id,
        asset_ref,
        mentions,
        salvage_debug,
        linking_result,
    ) if mentions_count or linking_result or authority_report else None

    _record_search_decision(
        run_id,
        not (fixture is not None and fixture.semantic_mentions) and bool(gate_decisions.get("token_search_allowed", True)),
    )
    if fixture is not None and fixture.semantic_mentions:
        log_event(run_id, "INDEX_SKIPPED", "INFO", "RAG auto-index skipped for curated semantic fixture mentions.")
    elif not gate_decisions.get("token_search_allowed", True):
        log_event(run_id, "INDEX_SKIPPED", "INFO", f"RAG auto-index SKIPPED: token_search_allowed=False (quality={quality_label})")
    else:
        _auto_index_run(run_id)

    log_event(run_id, "DONE", "END", "Full-page pipeline completed successfully.")
    update_run_fields(run_id, status="COMPLETED", current_stage="DONE", error=None)

    return {
        "run_id": run_id,
        "quality_label": quality_label,
        "downstream_mode": downstream_mode,
        "chunks_count": chunks_count,
        "mentions_count": mentions_count,
        "authority_report": authority_report,
        "mention_report": mention_report,
        "consolidated_report": consolidated_report,
        "warnings": [],
    }


async def _run_segmented_trace_pipeline(
    *,
    run_id: str,
    asset_ref: str,
    payload: SaiaFullPageExtractRequest,
    extract_payload: OCRExtractRequest,
) -> dict[str, Any]:
    log_event(run_id, "OCR_RUNNING", "START", f"Segmented OCR on {len(extract_payload.regions or [])} regions.")
    update_run_fields(run_id, status="RUNNING", current_stage="OCR_RUNNING")

    ocr_response_any = await asyncio.to_thread(_get_ocr_agent().run, extract_payload)
    if not isinstance(ocr_response_any, OCRExtractResponse):
        raise OCRAgentError("Segmented OCR did not return a full OCR response.")
    ocr_response = ocr_response_any

    raw_ocr = ocr_response.raw_ocr
    raw_text = str(raw_ocr.text if raw_ocr is not None else ocr_response.text or "")
    raw_lines = list(raw_ocr.lines) if raw_ocr is not None else [line for line in raw_text.splitlines() if line.strip()]
    confidence_value = float(raw_ocr.confidence) if raw_ocr is not None else 0.0
    detected_language, _detected_confidence = _detect_language_metadata(raw_text or str(ocr_response.final_text or ""))
    if detected_language == "unknown" and payload.language_hint:
        detected_language = _normalize_detected_language(str(payload.language_hint))

    ocr_payload: dict[str, Any] = {
        "lines": raw_lines,
        "text": raw_text,
        "script_hint": str(raw_ocr.script_hint if raw_ocr is not None else ocr_response.script_hint or "unknown"),
        "detected_language": _normalize_detected_language(detected_language),
        "confidence": confidence_value,
        "warnings": list(dict.fromkeys(ocr_response.warnings)),
        "quality_label": "OK",
        "sanity_metrics": {},
        "quality_label_v2": "OK",
        "downstream_mode": decide_downstream_mode("OK"),
    }

    _lex_lang = ocr_payload.get("detected_language", "unknown")
    ocr_quality_report = compute_quality_report(
        raw_text, run_id=run_id, pass_idx=0, language=_quality_language(_lex_lang), tiled=False
    )
    hardened_quality_label = ocr_quality_report.quality_label
    insert_ocr_quality_report(run_id, ocr_quality_report.to_dict())
    _lex_score = _lexical_plausibility(raw_text, _lex_lang) if _lex_lang != "unknown" else None
    gate_decisions = enforce_quality_gates(ocr_quality_report, run_id=run_id, lexical_plausibility=_lex_score)
    downstream_mode = gate_decisions["downstream_mode"]
    effective_quality = build_effective_quality(ocr_quality_report, gate_decisions, confidence=confidence_value)

    ocr_payload["quality_label"] = hardened_quality_label
    ocr_payload["quality_label_v2"] = hardened_quality_label
    ocr_payload["downstream_mode"] = downstream_mode

    region_boxes = [
        {
            "region_id": region.region_id,
            "bbox_xyxy": region.bbox_xyxy,
            "polygon": region.polygon,
            "backend_name": region.backend_name,
            "model_name": region.model_name,
        }
        for region in ocr_response.regions
    ]
    text_sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    insert_ocr_attempt(
        run_id,
        {
            "attempt_idx": 0,
            "tiling_strategy": "segmented_units",
            "tile_grid": None,
            "overlap_pct": 0.0,
            "tile_count": len(ocr_response.regions),
            "tile_boxes_json": region_boxes,
            "preproc_json": {
                "ocr_backend": ocr_response.ocr_backend,
                "compare_backends": [item.backend_name for item in ocr_response.comparison_results if not item.selected],
            },
            "model_used": ocr_response.model,
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
            "gates_passed": not bool(gate_decisions.get("blocked_stages")),
            "noop_detected": False,
            "decision": "PASS" if not bool(gate_decisions.get("blocked_stages")) else "FAIL",
            "ocr_text": raw_text[:2000] if raw_text else None,
            "detail_json": {
                "ocr_backend": ocr_response.ocr_backend,
                "regions": len(ocr_response.regions),
                "comparison_results": len(ocr_response.comparison_results),
                "downstream_mode": downstream_mode,
                "blocked_stages": gate_decisions.get("blocked_stages", []),
            },
        },
    )

    if ocr_response.comparison_results:
        insert_ocr_backend_results(
            run_id,
            [
                {
                    "page_id": item.page_id,
                    "region_id": item.region_id,
                    "backend_name": item.backend_name,
                    "model_name": item.model_name,
                    "confidence": item.confidence,
                    "selected": item.selected,
                    "text": item.text,
                    "raw_json": item.raw_metadata,
                }
                for item in ocr_response.comparison_results
            ],
        )

    update_run_fields(
        run_id,
        current_stage="OCR_DONE",
        script_hint=ocr_payload["script_hint"],
        detected_language=ocr_payload["detected_language"],
        confidence=ocr_payload["confidence"],
        warnings_json=json.dumps(
            {
                "warnings": ocr_payload["warnings"],
                "quality_label": hardened_quality_label,
                "quality_label_v2": hardened_quality_label,
                "ocr_backend": ocr_response.ocr_backend,
            },
            ensure_ascii=False,
        ),
        ocr_lines_json=json.dumps(raw_lines, ensure_ascii=False),
        ocr_text=raw_text,
    )

    if gate_decisions.get("blocked_stages"):
        failure_reason = (
            f"Quality gates FAILED for segmented OCR. quality_label_v2={hardened_quality_label}, "
            f"blocked_stages={gate_decisions.get('blocked_stages', [])}, "
            f"gibberish={ocr_quality_report.gibberish_score:.4f}, "
            f"lead_frag={ocr_quality_report.leading_fragment_ratio:.4f}"
        )
        log_event(run_id, "QUALITY_BLOCKED", "ERROR", failure_reason)
        update_run_fields(
            run_id,
            status="FAILED_QUALITY",
            current_stage="QUALITY_BLOCKED",
            error=failure_reason,
            proofread_text=raw_text,
        )
        return {
            "run_id": run_id,
            "status": "FAILED_QUALITY",
            "ocr_result": ocr_payload,
            "proofread_text": None,
            "detected_language": ocr_payload["detected_language"],
            "final_confidence": ocr_payload["confidence"],
            "quality_label": hardened_quality_label,
            "quality_label_v2": hardened_quality_label,
            "effective_quality": effective_quality.to_dict(),
            "downstream_mode": downstream_mode,
            "sanity_metrics": {},
            "quality_gates": gate_decisions,
            "ocr_attempts": list_ocr_attempts(run_id),
            "failure_reason": failure_reason,
            "chunks_count": 0,
            "mentions_count": 0,
            "top_mentions": [],
            "mention_report": None,
            "mention_recall": None,
            "linking_result": None,
            "consolidated_report": None,
            "ocr_backend_results": [item.model_dump() for item in ocr_response.comparison_results],
        }

    proofread_text = str(ocr_response.final_text or raw_text or "").strip() or raw_text
    log_event(run_id, "PROOFREAD_DONE", "END", "Segmented OCR proofread complete.")

    final_pipeline_text = proofread_text or raw_text
    if final_pipeline_text and final_pipeline_text != raw_text:
        post_proof_report = compute_quality_report(
            final_pipeline_text,
            run_id=run_id,
            pass_idx=10,
            previous_pass_tokens=[t for t in raw_text.split() if t],
            language=_quality_language(ocr_payload.get("detected_language")),
            tiled=False,
        )
        hardened_quality_label = post_proof_report.quality_label
        insert_ocr_quality_report(run_id, post_proof_report.to_dict())
        _pp_lex_lang = ocr_payload.get("detected_language", "unknown")
        _pp_lex_score = _lexical_plausibility(final_pipeline_text, _pp_lex_lang) if _pp_lex_lang != "unknown" else None
        gate_decisions = enforce_quality_gates(post_proof_report, run_id=run_id, lexical_plausibility=_pp_lex_score)
        downstream_mode = gate_decisions["downstream_mode"]
        effective_quality = build_effective_quality(post_proof_report, gate_decisions, confidence=ocr_payload["confidence"])
        ocr_payload["quality_label"] = hardened_quality_label
        ocr_payload["quality_label_v2"] = hardened_quality_label
        ocr_payload["downstream_mode"] = downstream_mode
        ocr_quality_report = post_proof_report

    update_run_fields(
        run_id,
        current_stage="PROOFREAD_DONE",
        proofread_text=proofread_text,
        confidence=ocr_payload["confidence"],
        warnings_json=json.dumps(
            {
                "warnings": ocr_payload["warnings"],
                "quality_label": ocr_payload["quality_label"],
                "quality_label_v2": ocr_payload["quality_label_v2"],
                "ocr_backend": ocr_response.ocr_backend,
            },
            ensure_ascii=False,
        ),
    )

    if not gate_decisions.get("ner_allowed", True):
        log_event(
            run_id,
            "ANALYZE_DEGRADED",
            "INFO",
            f"NER gate blocked full analysis; running degraded mention capture only (quality={hardened_quality_label})",
        )
        update_run_fields(run_id, current_stage="ANALYZE_RUNNING")
        base_text_source = "proofread" if proofread_text.strip() else "ocr"
        base_text = proofread_text if base_text_source == "proofread" else raw_text
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
        log_event(run_id, "ANALYZE_DONE", "END", f"degraded_capture chunks={chunks_count}, mentions={mentions_count}")
    else:
        log_event(run_id, "ANALYZE_RUNNING", "START", "Running chunk/entity analysis.")
        update_run_fields(run_id, current_stage="ANALYZE_RUNNING")
        base_text_source = "proofread" if proofread_text.strip() else "ocr"
        base_text = proofread_text if base_text_source == "proofread" else raw_text
        chunks, mentions, _candidates, salvage_debug = _run_trace_analysis(run_id, base_text)
        chunks_count = len(chunks)
        mentions_count = len(mentions)

        mention_recall_result = check_mention_recall(base_text, mentions_count, hardened_quality_label)
        if not mention_recall_result["mention_recall_ok"]:
            ocr_payload["warnings"] = list(dict.fromkeys([*ocr_payload["warnings"], mention_recall_result["reason"]]))
            log_event(run_id, "QUALITY_GATES", "INFO", mention_recall_result["reason"])

        if mention_recall_result.get("trigger_high_recall"):
            hr_mentions = extract_high_recall_mentions(
                base_text,
                script=ocr_quality_report.script_family,
                quality_label=hardened_quality_label,
            )
            if hr_mentions:
                existing_spans = {(m.get("start_offset"), m.get("end_offset")) for m in mentions}
                new_hr = [m for m in hr_mentions if (m.get("start_offset"), m.get("end_offset")) not in existing_spans]
                if new_hr:
                    for mention in new_hr:
                        mention["chunk_id"] = _assign_chunk_id_for_span(
                            int(mention.get("start_offset", 0)),
                            int(mention.get("end_offset", 0)),
                            chunks,
                        )
                    new_rows = insert_entity_mentions(run_id, new_hr)
                    mentions.extend(new_rows)
                    mentions_count = len(mentions)
                    log_event(run_id, "ANALYZE_RUNNING", "INFO", f"High-recall extraction added {len(new_rows)} mentions.")
                    salvage_debug["high_recall_added"] = len(new_rows)

        update_run_fields(
            run_id,
            current_stage="ANALYZE_DONE",
            base_text_source=base_text_source,
            chunks_count=chunks_count,
            mentions_count=mentions_count,
        )
        log_event(run_id, "ANALYZE_DONE", "END", f"chunks={chunks_count}, mentions={mentions_count}")

    mention_report = _build_mention_extraction_report(run_id, asset_ref, mentions, salvage_debug) if mentions or not salvage_debug.get("skipped") else None
    if gate_decisions.get("ner_allowed", True):
        base_text = proofread_text.strip() or raw_text
        mention_recall = check_mention_recall(base_text, mentions_count, hardened_quality_label)
    else:
        mention_recall = {
            "mention_recall_ok": True,
            "reason": "Degraded mention capture only (ner_allowed=False)",
            "trigger_high_recall": False,
        }

    log_event(run_id, "STORED", "END", "Run results persisted in SQLite.")
    update_run_fields(run_id, current_stage="STORED", proofread_text=proofread_text)

    if not gate_decisions.get("token_search_allowed", True):
        log_event(run_id, "LINKING_SKIPPED", "INFO", f"Authority linking SKIPPED: token_search_allowed=False (quality={hardened_quality_label})")
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

    consolidated_report = _build_consolidated_report(run_id, asset_ref, mentions, salvage_debug, linking_result) if mentions or linking_result else None

    _record_search_decision(run_id, bool(gate_decisions.get("token_search_allowed", True)))
    if not gate_decisions.get("token_search_allowed", True):
        log_event(run_id, "INDEX_SKIPPED", "INFO", f"RAG auto-index SKIPPED: token_search_allowed=False (quality={hardened_quality_label})")
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
        "quality_label": hardened_quality_label,
        "quality_label_v2": hardened_quality_label,
        "effective_quality": effective_quality.to_dict(),
        "downstream_mode": downstream_mode,
        "sanity_metrics": {},
        "quality_gates": gate_decisions,
        "ocr_attempts": list_ocr_attempts(run_id),
        "mention_recall": mention_recall,
        "chunks_count": chunks_count,
        "mentions_count": mentions_count,
        "top_mentions": mentions[:20],
        "mention_report": mention_report,
        "linking_result": linking_result,
        "consolidated_report": consolidated_report,
        "ocr_backend_results": [item.model_dump() for item in ocr_response.comparison_results],
    }
