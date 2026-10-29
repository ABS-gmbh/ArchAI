"""What the OCR agent does with a model's answer: strip fences and junk,
normalise lines under a Latin lock, parse the JSON, score the text, merge
tiles' lines and confidence, pick the language and summarise what a
proofread changed. Moved out of ``app/agents/ocr_agent.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import re

from app.services.lexicon_trust import _TRIGRAM_PROFILES, lexical_plausibility
from app.services.ocr_quality import compute_quality_report


# Language profiles usable for a best-fit lexical score ("unknown" is a flat 0.50).
_LEXICAL_PROFILES: tuple[str, ...] = tuple(p for p in _TRIGRAM_PROFILES if p != "unknown")


ALLOWED_SCRIPT_HINTS = {"insular_old_english", "latin_medieval", "unknown"}


REQUIRED_JSON_KEYS = ("lines", "text", "script_hint", "confidence", "warnings")


REQUIRED_JSON_KEY_SET = set(REQUIRED_JSON_KEYS)


META_TEXT_PATTERNS = (
    r"\bi[' ]?m sorry\b",
    r"\bcannot transcribe\b",
    r"\bunclear image\b",
    r"\bnot clear enough\b",
    r"\bas an ai\b",
)


UNWANTED_SCRIPT_RE = re.compile(r"[\u0370-\u03FF\u0400-\u04FF\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")


NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


@dataclass
class OCRTileResult:
    tile_id: str
    bbox_xyxy: list[float]
    lines: list[str]
    text: str
    confidence: float
    script_hint: str
    warnings: list[str]
    flags: list[str]
    model: str


def sanitize_ocr_text(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    lower = text.lower()
    if any(marker in lower for marker in ("no readable text", "nothing readable", "return an empty string")):
        return ""

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, str):
                return parsed.strip()
            if isinstance(parsed, dict):
                for key in ("text", "transcription", "content", "output"):
                    value = parsed.get(key)
                    if isinstance(value, str):
                        return value.strip()
                return ""
            return ""
        except Exception:
            pass

    return text.strip()


def _has_meta_text(text: str) -> bool:
    value = (text or "").lower()
    return any(re.search(pattern, value) for pattern in META_TEXT_PATTERNS)


def _normalize_uncertainty_tokens(value: str) -> str:
    text = value.replace("[...]", "[…]").replace("...", "[…]")
    text = text.replace("…", "[…]")
    text = text.replace("[[…]]", "[…]")
    return text


def _looks_like_pattern_junk(value: str) -> bool:
    text = (value or "").strip().lower()
    if not text:
        return False
    if "ccooocoo" in text:
        return True
    if re.fullmatch(r"\?{6,}", text):
        return True
    if re.search(r"(.)\1{6,}", text):
        return True
    compact = re.sub(r"\s+", "", text)
    if len(compact) >= 10 and len(set(ch for ch in compact if ch.isalpha())) <= 2:
        return True
    return False


def _sanitize_line_for_latin(line: str, *, latin_lock: bool) -> tuple[str, int]:
    text = NON_PRINTABLE_RE.sub("", str(line or ""))
    text = _normalize_uncertainty_tokens(text)
    text = re.sub(r"\?{6,}", "[… ]", text).replace("[… ]", "[…]")
    replaced = 0

    if latin_lock:
        chars: list[str] = []
        for ch in text:
            if UNWANTED_SCRIPT_RE.match(ch):
                chars.append("?")
                replaced += 1
            else:
                chars.append(ch)
        text = "".join(chars)

    text = re.sub(r"\s+", " ", text).strip()
    return text, replaced


def _normalize_lines(raw_lines: list[str], *, latin_lock: bool) -> tuple[list[str], list[str], float]:
    lines: list[str] = []
    warnings: list[str] = []
    replaced_non_latin = 0
    total_chars = 0

    for raw_line in raw_lines:
        if _has_meta_text(raw_line):
            warnings.append("meta_text_removed")
            continue
        line, replaced = _sanitize_line_for_latin(raw_line, latin_lock=latin_lock)
        replaced_non_latin += replaced
        total_chars += max(1, len(line))
        if not line:
            continue
        if _looks_like_pattern_junk(line):
            warnings.append("pattern_junk_removed")
            continue
        lines.append(line)

    ratio = replaced_non_latin / max(1, total_chars)
    if replaced_non_latin > 0:
        warnings.append("non_latin_chars_replaced")
    return lines, warnings, ratio


def _extract_json_candidate(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return "{}"
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


def _detect_script_hint(text: str) -> str:
    value = text or ""
    if UNWANTED_SCRIPT_RE.search(value):
        return "unknown"
    if re.search(r"[þðƿÞÐǷ]", value) or re.search(r"(?<!\d)7(?!\d)", value):
        return "insular_old_english"
    if re.search(r"[A-Za-zÀ-ÿ]", value):
        return "latin_medieval"
    return "unknown"


def _contains_script_drift(text: str) -> bool:
    return bool(UNWANTED_SCRIPT_RE.search(text or ""))


def _normalize_script_hint_value(value: object) -> tuple[str, bool]:
    hint = str(value or "").strip().lower()
    if hint in ALLOWED_SCRIPT_HINTS:
        return hint, False
    return "unknown", True


def parse_ocr_json(
    raw: str,
    *,
    expected_region_id: str | None = None,
    latin_lock: bool = True,
    strict_keys: bool = False,
) -> dict[str, object]:
    candidate = _extract_json_candidate(raw)
    data: dict[str, object] | None = None

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            data = parsed
    except Exception:
        data = None

    if data is None:
        sanitized = sanitize_ocr_text(raw)
        base_lines = [line for line in sanitized.splitlines() if line.strip()] if sanitized else []
        lines, normalized_warnings, non_latin_ratio = _normalize_lines(base_lines, latin_lock=latin_lock)
        text = "\n".join(lines)
        script_hint = _detect_script_hint(text)
        warnings: list[str] = ["INVALID_JSON_RESPONSE", *normalized_warnings]
        if _contains_script_drift(text):
            warnings.append("SCRIPT_DRIFT")
        return {
            "lines": lines,
            "text": text,
            "script_hint": script_hint,
            "confidence": 0.0 if not text else 0.6,
            "warnings": list(dict.fromkeys(warnings)),
            "non_latin_ratio": non_latin_ratio,
        }

    lines: list[str]
    text: str
    script_hint: str
    confidence_value: object
    warnings_value: object

    warnings: list[str] = []
    if strict_keys:
        incoming_keys = set(data.keys())
        if incoming_keys != REQUIRED_JSON_KEY_SET:
            warnings.append("INVALID_SCHEMA_KEYS")

    if not strict_keys and isinstance(data.get("regions"), list):
        regions = [item for item in data.get("regions", []) if isinstance(item, dict)]
        target: dict[str, object] = {}
        if expected_region_id:
            for item in regions:
                if str(item.get("region_id") or "") == str(expected_region_id):
                    target = item
                    break
        if not target and regions:
            target = regions[0]

        maybe_lines = target.get("lines") if isinstance(target, dict) else None
        lines = [str(item) for item in maybe_lines] if isinstance(maybe_lines, list) else []
        maybe_text = target.get("text") if isinstance(target, dict) else None
        if isinstance(maybe_text, str):
            text = maybe_text
        else:
            full_text = data.get("full_text")
            text = str(full_text) if isinstance(full_text, str) else "\n".join(lines)
        script_hint = str(data.get("document_script_hint") or "unknown").strip().lower()
        confidence_value = target.get("confidence", 0.0) if isinstance(target, dict) else 0.0
        warnings_value = target.get("warnings", []) if isinstance(target, dict) else []
    else:
        maybe_lines = data.get("lines")
        lines = [str(item) for item in maybe_lines] if isinstance(maybe_lines, list) else []
        maybe_text = data.get("text")
        text = str(maybe_text) if isinstance(maybe_text, str) else "\n".join(lines)
        script_hint = str(data.get("script_hint") or "unknown").strip().lower()
        confidence_value = data.get("confidence", 0.0)
        warnings_value = data.get("warnings", [])

    if not lines and text:
        lines = [line for line in text.splitlines() if line.strip()]

    lines, normalized_warnings, non_latin_ratio = _normalize_lines(lines, latin_lock=latin_lock)
    recomputed_text = "\n".join(lines)
    if text != recomputed_text:
        warnings.append("TEXT_JOIN_FIXED")
    text = recomputed_text

    script_hint, script_hint_invalid = _normalize_script_hint_value(script_hint)
    if script_hint_invalid:
        warnings.append("SCRIPT_HINT_INVALID")
    if not script_hint:
        script_hint = "unknown"
    if script_hint == "unknown" and text and not script_hint_invalid:
        detected = _detect_script_hint(text)
        if detected in ALLOWED_SCRIPT_HINTS:
            script_hint = detected

    try:
        confidence = float(confidence_value)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    model_warnings = [str(item).strip() for item in warnings_value] if isinstance(warnings_value, list) else []
    model_warnings = [item for item in model_warnings if item]
    warnings.extend(model_warnings)
    warnings.extend(normalized_warnings)

    if _has_meta_text(text):
        text = ""
        lines = []
        confidence = 0.0
        warnings.append("meta_text_removed")

    if _contains_script_drift(text):
        warnings.append("SCRIPT_DRIFT")

    return {
        "lines": lines,
        "text": text,
        "script_hint": script_hint,
        "confidence": confidence,
        "warnings": list(dict.fromkeys(warnings)),
        "non_latin_ratio": non_latin_ratio,
    }


def score_text_quality(text: str) -> float:
    content = (text or "").strip()
    if not content:
        return 0.0

    chars = list(content)
    total = len(chars)
    letters = sum(1 for ch in chars if ch.isalpha())
    non_latin = sum(1 for ch in chars if UNWANTED_SCRIPT_RE.match(ch))
    printable = sum(1 for ch in chars if ch.isprintable())
    ellipsis = content.count("…")

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        lines = [content]
    short_line_ratio = sum(1 for line in lines if len(line) <= 2) / len(lines)
    token_count = len(re.findall(r"[A-Za-zÀ-ÿ]+", content))

    score = (
        0.40 * min(1.0, letters / max(total * 0.6, 1))
        + 0.30 * (printable / max(total, 1))
        + 0.20 * (1.0 - short_line_ratio)
        + 0.10 * min(1.0, token_count / 20.0)
    )
    score -= min(0.20, ellipsis / max(total, 1))
    score -= min(0.50, non_latin / max(total, 1))
    return max(0.0, min(1.0, score))


def _normalize_for_dedupe(line: str) -> str:
    value = _normalize_uncertainty_tokens(str(line or ""))
    value = value.lower().strip()
    value = re.sub(r"[\u2018\u2019\u201c\u201d`'\".,;:!(){}<>]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _uncertainty_score(line: str) -> int:
    return line.count("?") + (line.count("[…]") * 3)


def _prefer_line(candidate: str, candidate_conf: float, current: str, current_conf: float) -> bool:
    cand_uncertainty = _uncertainty_score(candidate)
    cur_uncertainty = _uncertainty_score(current)
    if cand_uncertainty != cur_uncertainty:
        return cand_uncertainty < cur_uncertainty
    if abs(candidate_conf - current_conf) > 1e-6:
        return candidate_conf > current_conf
    return len(candidate.strip()) > len(current.strip())


def _merge_tile_lines(tile_results: list[OCRTileResult]) -> tuple[list[str], int]:
    merged_lines: list[str] = []
    merged_meta: list[tuple[str, float]] = []
    deduped_count = 0

    for tile in tile_results:
        for raw_line in tile.lines:
            line = str(raw_line or "").strip()
            if not line:
                continue
            norm = _normalize_for_dedupe(line)
            if not norm:
                continue

            duplicate_idx: int | None = None
            best_ratio = 0.0
            for idx, (existing_norm, existing_conf) in enumerate(merged_meta):
                ratio = 1.0 if norm == existing_norm else SequenceMatcher(None, norm, existing_norm).ratio()
                threshold = 0.96 if min(len(norm), len(existing_norm)) < 12 else 0.92
                if ratio >= threshold and ratio > best_ratio:
                    duplicate_idx = idx
                    best_ratio = ratio

            if duplicate_idx is None:
                merged_lines.append(line)
                merged_meta.append((norm, tile.confidence))
                continue

            deduped_count += 1
            existing_norm, existing_conf = merged_meta[duplicate_idx]
            if _prefer_line(line, tile.confidence, merged_lines[duplicate_idx], existing_conf):
                merged_lines[duplicate_idx] = line
                merged_meta[duplicate_idx] = (existing_norm, max(existing_conf, tile.confidence))

    return merged_lines, deduped_count


def _merge_confidence(tile_results: list[OCRTileResult]) -> float:
    if not tile_results:
        return 0.0

    numerator = 0.0
    denominator = 0.0
    for tile in tile_results:
        weight = float(max(1, len(tile.lines)))
        numerator += float(tile.confidence) * weight
        denominator += weight

    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def best_fit_language(text: str) -> tuple[str, float]:
    """Best-fitting language profile for *text*, and how plausible it is.

    Scoring against every profile and keeping the best turns the lexical signal
    into something usable without a language hint, and makes a WRONG hint
    harmless. Both mattered: lexical_plausibility returns a flat 0.50 for
    "unknown" - which is the common detected value - so the only content-aware
    term in the old selector became a constant.
    """
    if not text or not text.strip():
        return "unknown", 0.0
    best_language = "unknown"
    best_score = 0.0
    for profile in _LEXICAL_PROFILES:
        score = lexical_plausibility(text, profile)
        if score > best_score:
            best_language, best_score = profile, score
    return best_language, best_score


def _region_quality_value(text: str, confidence: float | None, language_hint: str | None) -> float:
    """Score a region's transcription for choosing between OCR backends.

    This is the only signal used to pick which backend's text becomes the
    transcription, and it could not tell good text from bad. Measured on the old
    implementation with no usable language hint - the common case, since language
    detection frequently yields "unknown":

        clean Latin          0.8137
        random letters       0.8250   <- outscored clean text
        repetition loop x20  0.8236   <- the failure mode the prompt warns about

    A wrong hint was actively harmful too: clean Old French scored 0.8481 under
    hint "old_french" but 0.4626 under "latin".

    The lexical term is now the best fit across all profiles rather than a lookup
    on a possibly-absent hint, and repetition is penalised explicitly, reusing the
    calibrated detector in ocr_quality. Measured after, with no hint supplied:

        clean Latin          0.8973
        clean Old French     0.7930
        Latin reversed       0.5968
        random letters       0.5607
        repetition loop      0.5473
    """
    if not text or not text.strip():
        return 0.0

    confidence_value = max(0.0, min(1.0, float(confidence))) if confidence is not None else 0.5

    hinted = str(language_hint or "").strip().lower()
    if hinted in _LEXICAL_PROFILES and hinted != "unknown":
        # Trust an explicit, usable hint, but never let it score BELOW the best
        # fit - that is what punished correct text under a mistaken hint.
        lexical_score = max(lexical_plausibility(text, hinted), best_fit_language(text)[1])
    else:
        lexical_score = best_fit_language(text)[1]

    repetition = compute_quality_report(text).repetition_score
    combined = (
        0.20 * confidence_value
        + 0.45 * lexical_score
        + 0.35 * (1.0 - repetition)
    )
    return round(max(0.0, min(1.0, combined)), 4)


def _summarize_final_changes(raw_text: str, final_text: str) -> list[str] | None:
    if raw_text == final_text:
        return None

    raw_lines = raw_text.splitlines()
    final_lines = final_text.splitlines()
    changes: list[str] = []
    for idx in range(max(len(raw_lines), len(final_lines))):
        raw_line = raw_lines[idx] if idx < len(raw_lines) else ""
        final_line = final_lines[idx] if idx < len(final_lines) else ""
        if raw_line == final_line:
            continue
        changes.append(f"line_{idx + 1}")
        if len(changes) >= 50:
            break

    return changes or None
