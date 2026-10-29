"""What chat's presentation helpers give on fixed payloads.

``tests/test_chat_presentation_characterisation.py`` compares them with
``tests/fixtures/chat_presentation_characterisation.json``, recorded before
they were moved out of ``app/services/chat_ai.py``. The payloads follow the
shapes ``build_rag_evidence_for_debug`` and ``_structured_entity_sections``
produce, seeded, with keys left out, lists emptied and boxes missing. To
re-record after an intended change::

    python tests/characterise_chat_presentation.py --write
"""

from __future__ import annotations

import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
FIXTURE = BACKEND / "tests" / "fixtures" / "chat_presentation_characterisation.json"
READINGS = BACKEND / "eval" / "quality_gate" / "readings.json"


def helper(name: str) -> Any:
    try:
        found = importlib.import_module("app.services.chat_presentation")
        if hasattr(found, name):
            return getattr(found, name)
    except ModuleNotFoundError:
        pass
    return getattr(importlib.import_module("app.services.chat_ai"), name)


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0])) if k != "duration_ms"}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def _maybe(rng: random.Random, row: dict[str, Any], keep: float = 0.8) -> dict[str, Any]:
    return {k: v for k, v in row.items() if rng.random() < keep}


def _payload(seed: int, text: str) -> dict[str, Any]:
    rng = random.Random(seed)
    lines = [line for line in text.splitlines() if line.strip()]
    chunks, linked, unresolved = [], [], []
    for i in range(rng.randrange(0, 4)):
        start = rng.randrange(0, max(1, len(text) - 60))
        chunks.append(_maybe(rng, {"run_id": f"run-{seed}", "asset_ref": f"page-{seed}.png", "chunk_id": f"c{i}", "chunk_idx": i, "offsets": f"{start}-{start + 60}", "text": text[start : start + 60]}))
    for i in range(rng.randrange(0, 3)):
        surface = rng.choice(lines).split()[0] if lines else "Arthur"
        bbox = rng.choice([None, [10, 20, 300, 40], {"x": 1, "y": 2, "w": 3, "h": 4}])
        linked.append(_maybe(rng, {
            "record_type": "linked", "run_id": f"run-{seed}", "asset_ref": f"page-{seed}.png", "chunk_id": f"c{i}", "entity_id": f"wikidata:Q{1000 + seed * 10 + i}",
            "canonical_label": surface.title(), "entity_type": rng.choice(["person", "place", "work"]), "authority_source": rng.choice(["wikidata", "viaf", "geonames"]),
            "authority_id": f"A{seed}{i}", "wikidata_qid": f"Q{1000 + i}", "viaf_id": rng.choice(["", f"{9000 + i}"]), "geonames_id": rng.choice(["", f"{2600 + i}"]),
            "confidence": round(rng.random(), 3), "retrieval_score": round(rng.random(), 3), "offsets": f"{i * 10}-{i * 10 + 8}",
            "mention_surfaces": [surface, surface.lower()][: rng.randrange(1, 3)], "evidence_excerpt": " ".join(lines[i : i + 2])[: rng.choice([40, 400])], "bbox": bbox,
        }))
    for i in range(rng.randrange(0, 3)):
        surface = rng.choice(lines).split()[-1] if lines else "uilain"
        unresolved.append(_maybe(rng, {
            "record_type": "unresolved", "run_id": f"run-{seed}", "asset_ref": f"page-{seed}.png", "chunk_id": f"c{i}", "mention_id": f"m{seed}{i}",
            "mention_surface": surface, "probable_type": rng.choice(["person", "place", "unknown"]), "reason_unresolved": rng.choice(["unresolved_low_quality", "no_candidates", "ambiguous"]),
            "confidence": round(rng.random(), 3), "retrieval_score": round(rng.random(), 3), "offsets": f"{i * 7}-{i * 7 + 5}",
            "evidence_excerpt": " ".join(lines[-2:])[:120], "bbox": rng.choice([None, [1, 2, 3, 4]]),
        }))
    payload = {
        "query": rng.choice(["Who is mentioned on this page?", "print the RAG", "Arthur", "Où est Lausanne ?"]),
        "run_id": f"run-{seed}",
        "extraction_summary": {"ocr_chunk_hits": len(chunks), "linked_entity_hits": len(linked), "unresolved_mention_hits": len(unresolved)},
        "authority_summary": {
            "linked_entities": [{k: row.get(k, "") for k in ("entity_id", "canonical_label", "authority_source", "authority_id", "wikidata_qid", "viaf_id", "geonames_id", "confidence")} for row in linked],
            "unresolved_mentions": [{k: row.get(k, "") for k in ("mention_id", "mention_surface", "probable_type", "reason_unresolved", "confidence")} for row in unresolved],
        },
        "document_metadata": _maybe(rng, {"asset_ref": f"page-{seed}.png", "detected_language": rng.choice(["latin", "old_french", ""])}),
        "transcript_snippet": " || ".join(lines[:2])[:280],
        "ocr_chunk_evidence": chunks,
        "linked_entity_evidence": linked,
        "unresolved_mention_evidence": unresolved,
    }
    return _maybe(rng, payload, keep=0.92)


def record() -> dict[str, Any]:
    texts = [p["reference"] for p in json.loads(READINGS.read_text(encoding="utf-8"))["pages"]]
    out: dict[str, Any] = {}
    for seed in range(24):
        text = texts[seed % len(texts)]
        rng = random.Random(1000 + seed)
        payload = _payload(seed, text)
        stage = helper("_build_stage_metadata")(stage_name=rng.choice(["chat", "translation", "rag_inspection"]), model_used="qwen3-30b-a3b-instruct-2507", mode_used=rng.choice([None, "presentation", "debug"]), duration_ms=12.5)
        context = rng.choice([None, {}, {"ocr_run_id": f"run-{seed}", "document_id": "doc-1", "document_language": "latin", "script_family": "gothic", "document_year": "1350", "transcript": text[:200], "place_or_origin": "Paris"}])
        verification = rng.choice([None, {}, {"assessment": "supported", "notes": ["  checked  ", "", None], "citations_checked": ["chunk-1", "  wikidata:Q45720 ", None], "corrected_answer": "Arthur is named."}])
        translation = rng.choice(["Translation:\\nThe king came.", "  « Le roi vint. »  ", "", text[:120]])
        out[f"{seed:02d}"] = {
            "presentation": helper("_format_rag_presentation_markdown")(payload),
            "debug": helper("_format_rag_debug_markdown")(payload),
            "linked": helper("_format_linked_entity_evidence")(payload.get("linked_entity_evidence") or []),
            "unresolved": helper("_format_unresolved_mention_evidence")(payload.get("unresolved_mention_evidence") or []),
            "stage": stage,
            "chat_inspection": helper("_chat_inspection_payload")(context=context, question=str(payload.get("query", "")), output_text="Arthur is mentioned.", evidence_text=helper("_format_linked_entity_evidence")(payload.get("linked_entity_evidence") or []), stage_metadata=stage, verification=verification),
            "translation_inspection": helper("_translation_inspection_payload")(context=context, output_text=translation, stage_metadata=stage),
            "translation_normalized": helper("_normalize_translation_output")(translation, source_text=text[:120]),
            "appendix": helper("_format_verification_appendix")(verification),
            "clean_value": [helper("_clean_verification_value")(v) for v in ("  x  ", None, 3, ["a", " b "], {"k": " v "})],
        }
    return _clean(out)


if __name__ == "__main__":
    if "--write" in sys.argv:
        FIXTURE.write_text(json.dumps(record(), ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {FIXTURE}")
    else:
        print(json.dumps(record(), ensure_ascii=False)[:1500])
