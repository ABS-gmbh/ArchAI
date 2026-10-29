"""How chat answers and their evidence are presented.

Stage metadata, the inspection payloads of chat and translation, the markdown
of retrieved evidence and its debug view, the linked-entity and
unresolved-mention evidence blocks, and the verification appendix. Moved out
of ``app/services/chat_ai.py``, which calls the models.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.services.model_router import chat_stage_from_context


def _build_stage_metadata(
    *,
    stage_name: str,
    model_used: str,
    mode_used: str | None = None,
    duration_ms: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stage_name": str(stage_name or "").strip() or "rag_chat",
        "model_used": str(model_used or "").strip(),
    }
    if mode_used:
        payload["mode_used"] = str(mode_used).strip()
    if duration_ms is not None:
        payload["duration_ms"] = round(float(duration_ms), 1)
    return payload


def _evidence_tag_counts(evidence_text: str) -> dict[str, int]:
    text = str(evidence_text or "")
    return {
        "ocr_chunk_evidence": len(re.findall(r"\[(?:OCR_CHUNK_EVIDENCE|EVIDENCE)\]", text)),
        "linked_entity_evidence": len(re.findall(r"\[LINKED_ENTITY_EVIDENCE\]", text)),
        "unresolved_mention_evidence": len(re.findall(r"\[UNRESOLVED_MENTION_EVIDENCE\]", text)),
    }


def _translation_inspection_payload(
    *,
    context: dict[str, Any] | None,
    output_text: str,
    stage_metadata: dict[str, Any],
) -> dict[str, Any]:
    context = context or {}
    metadata_hints = {
        key: value
        for key, value in {
            "language": context.get("document_language"),
            "year": context.get("document_year"),
            "origin": context.get("place_or_origin"),
            "script": context.get("script_family"),
            "document_type": context.get("document_type"),
            "notes": context.get("document_notes"),
        }.items()
        if str(value or "").strip()
    }
    transcript = str(context.get("transcript") or "")
    return {
        "input_source_summary": {
            "source_type": "extracted_transcript",
            "ocr_run_id": str(context.get("ocr_run_id") or context.get("run_id") or "") or None,
            "transcript_chars": len(transcript),
            "source_language": str(context.get("document_language") or "") or None,
            "target_language": "English",
            "metadata_hints": metadata_hints,
        },
        "model_used": stage_metadata.get("model_used"),
        "evidence_used": {
            "ocr_transcript_used": bool(transcript),
            "page_image_used": False,
            "metadata_hint_keys": list(metadata_hints.keys()),
        },
        "final_output": output_text,
        "confidence_or_assessment": None,
        "stage_metadata": stage_metadata,
    }


def _normalize_translation_output(text: str, *, source_text: str = "") -> str:
    lines = [str(line or "").strip().strip('"') for line in str(text or "").replace("\r", "\n").split("\n")]
    filtered: list[str] = []
    for line in lines:
        if not line:
            continue
        line = re.sub(r"(?is)```(?:json)?|```", " ", line).strip()
        line = re.sub(
            r"(?i)^\s*(here(?: is|'s)? (?:the )?translation|translation|tentative translation|english translation|modern english|english rendering|fluent translation|old french to english(?: translation)?)\s*:\s*",
            "",
            line,
        ).strip()
        lower = line.lower()
        if lower.startswith(
            (
                "user request:",
                "ocr-extracted source text:",
                "source text:",
                "source language:",
                "target language:",
                "translation note:",
                "translator's note:",
                "metadata:",
                "context:",
                "explanation:",
                "commentary:",
            )
        ):
            continue
        if re.match(r"(?i)^(the|this)\s+(page|image|manuscript)\b", line) and len(lines) > 1:
            continue
        filtered.append(line)

    body = re.sub(r"\s+", " ", " ".join(filtered)).strip()
    body = re.sub(
        r"(?i)^\s*(translation|tentative translation|english translation|modern english|english rendering)\s*:\s*",
        "",
        body,
    ).strip()
    body = re.sub(r"(?is)\s+\bnote:\s+.*$", "", body).strip()
    body = re.sub(r"(?is)\s+\btranslator'?s note:\s+.*$", "", body).strip()
    if source_text:
        source_norm = re.sub(r"\s+", " ", str(source_text or "")).strip().lower()
        if source_norm and body.lower() == source_norm:
            body = "[unclear]"
    if not body:
        body = "[unclear]"
    return body


def _chat_inspection_payload(
    *,
    context: dict[str, Any] | None,
    question: str,
    output_text: str,
    evidence_text: str,
    stage_metadata: dict[str, Any],
    verification: dict[str, Any] | None,
) -> dict[str, Any]:
    context = context or {}
    evidence_counts = _evidence_tag_counts(evidence_text)
    return {
        "input_source_summary": {
            "chat_stage": chat_stage_from_context(context),
            "question": _short_text(question, limit=180),
            "ocr_run_id": str(context.get("ocr_run_id") or context.get("run_id") or "") or None,
            "document_id": str(context.get("document_id") or "") or None,
        },
        "model_used": stage_metadata.get("model_used"),
        "evidence_used": {
            **evidence_counts,
            "authority_report_present": bool(str(context.get("authority_report") or "").strip()),
        },
        "final_output": output_text,
        "confidence_or_assessment": verification.get("assessment") if verification else None,
        "stage_metadata": stage_metadata,
    }


def _format_rag_presentation_markdown(payload: dict[str, Any]) -> str:
    lines = ["**RAG Evidence**", f"- Query: {payload.get('query', '')}", f"- Run ID: {payload.get('run_id', '')}"]
    document_metadata = dict(payload.get("document_metadata") or {})
    metadata_lines: list[str] = []
    for key in ("asset_ref", "detected_language"):
        value = str(document_metadata.get(key) or "").strip()
        if value:
            label = "Asset ref" if key == "asset_ref" else "Detected language"
            metadata_lines.append(f"- {label}: {value}")
    if metadata_lines:
        lines.extend(["", "**Document Metadata**", *metadata_lines])

    extraction_summary = dict(payload.get("extraction_summary") or {})
    extraction_lines = [
        f"- OCR chunk hits: {int(extraction_summary.get('ocr_chunk_hits', 0))}",
        f"- Linked entity hits: {int(extraction_summary.get('linked_entity_hits', 0))}",
        f"- Unresolved mention hits: {int(extraction_summary.get('unresolved_mention_hits', 0))}",
    ]
    lines.extend(["", "**Extraction Summary**", *extraction_lines])

    transcript_snippet = str(payload.get("transcript_snippet") or "").strip()
    lines.extend(["", "**Transcript Snippet**"])
    if transcript_snippet:
        lines.append(f"- {transcript_snippet}")
    else:
        lines.append("- None")

    lines.extend(["", "**Linked Entities**"])
    chunk_rows = list(payload.get("ocr_chunk_evidence", []) or [])
    linked_rows = list(payload.get("linked_entity_evidence", []) or [])
    if not linked_rows:
        lines.append("- None")
    for row in linked_rows[:4]:
        authority_bits = ", ".join(
            part
            for part in (
                f"Wikidata {row.get('wikidata_qid', '')}" if row.get("wikidata_qid") else "",
                f"VIAF {row.get('viaf_id', '')}" if row.get("viaf_id") else "",
                f"GeoNames {row.get('geonames_id', '')}" if row.get("geonames_id") else "",
            )
            if part
        )
        authority_suffix = f"; {authority_bits}" if authority_bits else ""
        lines.append(
            "- "
            f"{row.get('canonical_label', '') or row.get('mention_surface', '')} "
            f"({row.get('entity_type', '') or 'entity'}; confidence {_confidence_text(row.get('confidence', ''))}"
            f"{authority_suffix}) "
            f"[chunk {row.get('chunk_id', '')}, offsets {row.get('offsets', '')}]"
        )

    lines.extend(["", "**Unresolved Mentions**"])
    unresolved_rows = list(payload.get("unresolved_mention_evidence", []) or [])
    if not unresolved_rows:
        lines.append("- None")
    for row in unresolved_rows[:4]:
        lines.append(
            "- "
            f"{row.get('mention_surface', '')} "
            f"({row.get('probable_type', '') or 'unknown type'}; confidence {_confidence_text(row.get('confidence', ''))}) "
            f"unresolved because {_short_text(row.get('reason_unresolved', ''), limit=110)} "
            f"[chunk {row.get('chunk_id', '')}, offsets {row.get('offsets', '')}]"
        )

    uncertainty_notes = []
    if len(chunk_rows) > 4:
        uncertainty_notes.append(f"{len(chunk_rows) - 4} additional OCR chunk hit(s) omitted from presentation view.")
    if len(linked_rows) > 4:
        uncertainty_notes.append(f"{len(linked_rows) - 4} additional linked entity hit(s) omitted from presentation view.")
    if len(unresolved_rows) > 4:
        uncertainty_notes.append(f"{len(unresolved_rows) - 4} additional unresolved mention hit(s) omitted from presentation view.")
    if uncertainty_notes:
        lines.extend(["", "**Uncertainty Notes**"])
        for note in uncertainty_notes[:3]:
            lines.append(f"- {note}")
    return "\n".join(lines)


def _format_rag_debug_markdown(payload: dict[str, Any]) -> str:
    chunk_rows = list(payload.get("ocr_chunk_evidence", []) or [])
    linked_rows = list(payload.get("linked_entity_evidence", []) or [])
    unresolved_rows = list(payload.get("unresolved_mention_evidence", []) or [])
    debug_payload = {
        "query": payload.get("query", ""),
        "run_id": payload.get("run_id", ""),
        "document_metadata": payload.get("document_metadata", {}),
        "summary": {
            "ocr_chunk_hits": len(chunk_rows),
            "linked_entity_hits": len(linked_rows),
            "unresolved_mention_hits": len(unresolved_rows),
        },
        "authority_summary": payload.get("authority_summary", {}),
        "transcript_snippet": payload.get("transcript_snippet", ""),
        "ocr_chunk_evidence_sample": chunk_rows[:4],
        "linked_entity_evidence_sample": linked_rows[:4],
        "unresolved_mention_evidence_sample": unresolved_rows[:4],
        "omitted_counts": {
            "ocr_chunk_evidence": max(0, len(chunk_rows) - 4),
            "linked_entity_evidence": max(0, len(linked_rows) - 4),
            "unresolved_mention_evidence": max(0, len(unresolved_rows) - 4),
        },
    }
    lines = [
        "**RAG Debug View**",
        f"- Query: {payload.get('query', '')}",
        f"- Run ID: {payload.get('run_id', '')}",
        f"- OCR chunk hits: {len(chunk_rows)}",
        f"- Linked entity hits: {len(linked_rows)}",
        f"- Unresolved mention hits: {len(unresolved_rows)}",
        "",
        "```json",
        json.dumps(debug_payload, ensure_ascii=False, indent=2),
        "```",
    ]
    return "\n".join(lines)


def _short_text(value: Any, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _confidence_text(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return ""


def _format_linked_entity_evidence(section: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for row in section:
        authority_bits = [
            f"authority_source: {row.get('authority_source', '')}",
            f"authority_id: {row.get('authority_id', '')}",
            f"wikidata_qid: {row.get('wikidata_qid', '')}",
            f"viaf_id: {row.get('viaf_id', '')}",
            f"geonames_id: {row.get('geonames_id', '')}",
        ]
        bbox = row.get("bbox") or []
        blocks.append(
            "[LINKED_ENTITY_EVIDENCE]\n"
            f"run_id: {row.get('run_id', '')}\n"
            f"asset_ref: {row.get('asset_ref', '')}\n"
            f"entity_id: {row.get('entity_id', '')}\n"
            f"canonical_label: {row.get('canonical_label', '')}\n"
            f"mention_surfaces: {', '.join(row.get('mention_surfaces', [])[:4])}\n"
            f"entity_type: {row.get('entity_type', '')}\n"
            + "\n".join(authority_bits)
            + "\n"
            f"link_confidence: {_confidence_text(row.get('confidence', ''))}\n"
            f"retrieval_score: {_confidence_text(row.get('retrieval_score', ''))}\n"
            f"chunk_id: {row.get('chunk_id', '')}\n"
            f"offsets: {row.get('offsets', '')}\n"
            f"bbox: {','.join(str(item) for item in bbox) if bbox else ''}\n"
            f"evidence_excerpt: {row.get('evidence_excerpt', '')}\n"
            "[/LINKED_ENTITY_EVIDENCE]"
        )
    return "\n\n".join(blocks)


def _format_unresolved_mention_evidence(section: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for row in section:
        bbox = row.get("bbox") or []
        blocks.append(
            "[UNRESOLVED_MENTION_EVIDENCE]\n"
            f"run_id: {row.get('run_id', '')}\n"
            f"asset_ref: {row.get('asset_ref', '')}\n"
            f"mention_id: {row.get('mention_id', '')}\n"
            f"mention_surface: {row.get('mention_surface', '')}\n"
            f"probable_type: {row.get('probable_type', '')}\n"
            f"reason_unresolved: {row.get('reason_unresolved', '')}\n"
            f"link_confidence: {_confidence_text(row.get('confidence', ''))}\n"
            f"retrieval_score: {_confidence_text(row.get('retrieval_score', ''))}\n"
            f"chunk_id: {row.get('chunk_id', '')}\n"
            f"offsets: {row.get('offsets', '')}\n"
            f"bbox: {','.join(str(item) for item in bbox) if bbox else ''}\n"
            f"evidence_excerpt: {row.get('evidence_excerpt', '')}\n"
            "[/UNRESOLVED_MENTION_EVIDENCE]"
        )
    return "\n\n".join(blocks)


def _clean_verification_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"(?is)<\s*(think|analysis|scratchpad)\b[^>]*>.*?<\s*/\s*\1\s*>", " ", text)
    text = re.sub(
        r"(?im)^\s*(reasoning|analysis|scratchpad|chain[\s-]*of[\s-]*thought|internal reasoning|thought process)\s*:\s*.*$",
        " ",
        text,
    )
    text = re.sub(r"(?is)```(?:json)?|```", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -:\n\t")


def _format_verification_appendix(verification: dict[str, Any] | None) -> str:
    if not verification:
        return ""
    notes = [_clean_verification_value(item) for item in list(verification.get("notes") or [])]
    notes = [item for item in notes if item]
    citations = [
        _clean_verification_value(item) for item in list(verification.get("citations_checked") or [])
    ]
    citations = [item for item in citations if item]
    corrected_answer = _clean_verification_value(verification.get("corrected_answer", ""))
    lines = [
        "",
        "",
        "[Verification]",
        f"Assessment: {_clean_verification_value(verification.get('assessment', 'partially_supported')) or 'partially_supported'}",
        f"Verified answer: {corrected_answer}",
        f"Notes: {' | '.join(notes[:4]) if notes else 'none'}",
        f"Citations checked: {' | '.join(citations[:6]) if citations else 'none'}",
    ]
    lines.append(f"Verifier model: {_clean_verification_value(verification.get('model_used', ''))}")
    return "\n".join(lines)
