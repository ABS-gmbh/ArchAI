from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Sequence

from app.agents.ocr_proofreader_agent import OcrProofreaderAgent, check_proofread_delta
from app.config import settings
from app.services.lexicon_trust import lexical_trust_adjustment
from app.services.multiview import generate_variants, pick_retry_variant
from app.schemas.agents_ocr import (
    DEFAULT_SAIA_OCR_MODEL_PREFS,
    SaiaOCRFallback,
    SaiaOCRLocationSuggestion,
    SaiaOCRRequest,
    SaiaOCRResponse,
)
from app.services.saia_client import SaiaClient, is_model_not_found_error
from app.agents.saia_ocr_errors import (
    SaiaOCRAgentError,
)
from app.services.ocr_image_prep import (
    _crop_and_maybe_upscale,
    _downscale_image_b64_for_retry,
    _encode_png_base64,
    _open_rgb_image_from_b64,
    _prepare_image_for_ocr,
)
from app.services.ocr_tiling import (
    _TILE_PAD_RATIO,
    _order_tiles_reading,
    _pad_rect,
    _rects_to_block_rects,
    _suggestions_to_rects,
)
from app.services.ocr_sanity import (
    _OCR_VOWELS,
    _garbage_ratio,
    _is_low_text_quality,
    _is_sane_ocr,
    _looks_hallucinated_passage,
    _ocr_sanity_metrics,
    _quality_label_from_sanity,
    _script_drift_ratio,
    compute_sanity,
    force_uncertainty_markers,
    quality_gate_enforce,
    sanitize_lines,
    sanity_adjust_confidence,
)

ALLOWED_SCRIPT_HINTS = {"latin", "greek", "cyrillic", "mixed", "unknown"}
ALLOWED_DETECTED_LANGUAGES = {
    "latin",
    "old_english",
    "middle_english",
    "french",
    "old_french",
    "middle_french",
    "anglo_norman",
    "occitan",
    "old_high_german",
    "middle_high_german",
    "german",
    "dutch",
    "italian",
    "spanish",
    "portuguese",
    "catalan",
    "church_slavonic",
    "greek",
    "hebrew",
    "arabic",
    "mixed",
    "unknown",
}
OLD_ENGLISH_MARKERS_RE = re.compile(r"[þðƿÞÐǷ]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
LATIN_TOKEN_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿĀ-žſƀ-ɏ]+$")
VISION_MODEL_HINTS = ("vl", "vision", "internvl", "gemma", "mistral", "omni")
INTERNVL_MODEL_HINT = "internvl"
REQUIRED_JSON_KEYS = ("lines", "text", "script_hint", "detected_language", "confidence", "warnings")
FRENCH_STRONG_ANCHORS = {
    "tres",
    "tresreuerend",
    "pere",
    "euesque",
    "eusque",
    "prince",
    "lausanne",
    "laufanne",
    "grace",
    "considerat",
    "desirs",
    "chapellain",
    "seruiteur",
    "approchant",
}
FRENCH_FUNCTION_WORDS = {"que", "et", "de", "la", "le", "des"}
FRENCH_OLD_MARKERS = {"pur", "ki", "mei", "sun", "uostre", "nostre", "cil", "cest", "dunc"}
FRENCH_MIDDLE_MARKERS = {
    "approchant",
    "considerat",
    "grace",
    "mille",
    "cens",
    "chapellain",
    "seruiteur",
    "desirs",
    "lausanne",
    "prince",
    "pere",
}
ANGLO_NORMAN_MARKERS = {"anglo", "norman", "normand", "engleterre"}
OCR_EVIDENCE_GUIDE = (
    "INTERNAL OCR METHOD\n"
    "- First inspect the visible letterforms, stroke shapes, spacing, abbreviation marks, and line structure.\n"
    "- Next infer the most likely script family, language family, and genre from what is visible on the page.\n"
    "- Only then use highly recognizable formulae as soft anchors to validate a reading.\n"
    "- Anchor phrases are CHECKS, not sources. Never complete a known passage from memory.\n"
    "- A proposed reading is acceptable only if it stays in the right script family, preserves plausible word boundaries, fits the visible morphology, and matches the visible strokes.\n"
    "- If a famous biblical or liturgical passage seems likely, you may test candidate phrases against the strokes, but you must reject them if the image does not support them.\n"
)

SAIA_OCR_PROMPT_VERSION = "saia_ocr_full_page_v1"
SAIA_OCR_SYSTEM_PROMPT = (
    "You are a professional paleographer producing diplomatic OCR for historical manuscripts.\n"
    "Transcribe ONLY what is visibly written in the provided image in natural reading order.\n"
    "NEVER invent, continue, normalize, or complete missing text from memory or external context.\n"
    "Use uncertainty markers exactly: ? for unclear characters and […] for unclear spans.\n"
    "Do not guess words.\n"
    "If a word is not clearly legible, you MUST NOT output a plausible substitute.\n"
    "For partially legible words: keep visible letters and replace unclear characters with ? (example: tresr?uerend).\n"
    "For wholly unclear words/spans: output […], not a made-up token.\n"
    "Never replace unclear text with a different familiar word.\n"
    "Treat decorated initials as letters when they clearly form the first letter(s) of a line/word.\n"
    "\n"
    "CRITICAL: GOTHIC BOOKHAND / TEXTURA READING GUIDE\n"
    "Medieval manuscripts are typically written in Gothic textura or bookhand. You MUST apply these rules:\n"
    "\n"
    "MINIM CONFUSION — the most common OCR error:\n"
    "  Gothic script writes i, u, n, m as sequences of identical vertical strokes (minims).\n"
    "  - Two minims = u or n (context decides).\n"
    "  - Three minims = m, in, ni, ui, iu.\n"
    "  - Do NOT guess: if a minim sequence is ambiguous, transcribe the most likely reading based on surrounding letters.\n"
    "  - Common minim-heavy words in Old French/Latin: uilain (vilain), honeur (honneur), enuie, dominus, anima.\n"
    "\n"
    "LONG-S (ſ) vs f:\n"
    "  - Long-s (ſ) looks almost identical to f but has NO crossbar or only a half-crossbar on the left.\n"
    "  - f has a FULL crossbar through the stem.\n"
    "  - Common misread: 'furent' read as 'surent', or 'si' read as 'fi'. Check the crossbar carefully.\n"
    "  - In diplomatic transcription, you may output either 's' or 'ſ' for long-s.\n"
    "\n"
    "TIRONIAN ET (⁊ / z-shaped):\n"
    "  - In many medieval manuscripts, 'et' (and) is written as a z-shaped or ⁊-shaped symbol.\n"
    "  - Transcribe it as 'z' or '⁊' or 'et' — but be consistent. The symbol is NOT the letter z.\n"
    "  - If you see a standalone z-like character between words, it almost certainly means 'et'.\n"
    "\n"
    "ABBREVIATION MARKS:\n"
    "  - A horizontal bar (macron/tilde) over a vowel usually indicates a missing nasal: ā = an/am, ō = on/om, ū = un/um.\n"
    "  - A bar over a consonant often indicates a missing following letter: q̄ = que, p̄ = per/par/pre.\n"
    "  - Superscript letters indicate omitted sequences.\n"
    "  - DO NOT expand abbreviations — keep the abbreviated form as written.\n"
    "  - If an abbreviation mark is visible, note it; do not silently drop it.\n"
    "\n"
    "LETTER CONFUSIONS TO WATCH:\n"
    "  - c/t: in Gothic, c and t look very similar. Context helps: 'ct' clusters are common in Latin.\n"
    "  - r: Gothic r has two forms — normal r and round-r (shaped like 2) used after o, p, b, d.\n"
    "  - d: Gothic d can be upright (like modern d) or uncial (rounded back, looks like ð without the stroke).\n"
    "  - v/b: Gothic v and b can be confused. v is usually used word-initially for 'u' sounds.\n"
    "  - a/o: in some hands these are very similar.\n"
    "\n"
    "DECORATED / RUBRICATED INITIALS:\n"
    "  - Large coloured letters (red, blue, gold) at line starts are decorated initials.\n"
    "  - They are real letters — transcribe them as the letter they represent.\n"
    "  - A large R, C, L, Q, etc. at the start of a stanza is the first letter of the first word.\n"
    "\n"
    "LAYOUT:\n"
    "  - Many manuscript pages have TWO COLUMNS. Read each column top-to-bottom, left column first.\n"
    "  - Marginal annotations may appear outside the main text block.\n"
    "  - Verse texts have one line of poetry per manuscript line.\n"
    "\n"
    "NUMERAL AND SPACING RULES\n"
    "- Do NOT output Arabic numerals (0-9) unless clearly written as numerals in the manuscript.\n"
    "- Do NOT insert spaces inside a single word.\n"
    "- If two letters appear close but you are unsure whether they belong to the same word, keep them together.\n"
    "\n"
    "Language selection rubric (detected_language)\n"
    "- detected_language must reflect the language of the visible text.\n"
    "- Choose latin for clearly Latin morphology (many -us/-um/-ae endings).\n"
    "- For French-family text, prefer old_french for pre-1300 texts, middle_french for c.1300-1500, french for modern.\n"
    "- anglo_norman only with explicit Anglo-Norman evidence; if unsure, use old_french or middle_french.\n"
    "- Confidence guidance:\n"
    "  - Long readable text with strong language cues: 0.80-0.95.\n"
    "  - Noisy or short text: 0.55-0.75.\n"
    "  - Truly ambiguous or mixed: choose mixed and keep confidence <= 0.60.\n"
    "\n"
    "Return exactly one JSON object with keys: lines, text, script_hint, detected_language, confidence, warnings.\n"
    "text MUST equal \"\\n\".join(lines).\n"
    "detected_language MUST be one of: latin, old_english, middle_english, french, old_french, middle_french, anglo_norman, occitan, old_high_german, middle_high_german, german, dutch, italian, spanish, portuguese, catalan, church_slavonic, greek, hebrew, arabic, mixed, unknown.\n"
    "If you feel tempted to output a known passage that is not clearly visible, STOP and output lines=[] and text=\"\" instead.\n"
    "If you are not sure a word is correct, you MUST output ? or […]; do not output a plausible-looking word without visible support."
    "\n\n"
    + OCR_EVIDENCE_GUIDE
)
SAIA_OCR_USER_PROMPT = (
    "Return JSON only.\n"
    "Schema: keys must be exactly: lines, text, script_hint, detected_language, confidence, warnings.\n"
    "- One manuscript line per entry in lines. Each entry MUST NOT contain '\\n'.\n"
    "- text must equal \"\\n\".join(lines).\n"
    "- Preserve original spelling, abbreviations, punctuation, capitalization, and line breaks.\n"
    "- Do not translate, normalize, modernize, or expand abbreviations.\n"
    "- Use uncertainty markers: ? for unclear characters, […] for unclear spans.\n"
    "- No markdown, no explanations, no extra keys.\n"
    "- Never transcribe prompt metadata. Region ids, list numbers, category names, coordinates, tile numbers, and labels are not manuscript text.\n"
    "- If a reading looks like a familiar biblical or liturgical formula, verify it against the visible strokes before using it.\n"
    "\n"
    "READING REMINDERS FOR GOTHIC SCRIPT:\n"
    "- This manuscript is likely in Gothic textura/bookhand. Read each letter carefully.\n"
    "- Watch for MINIM confusion: sequences of identical vertical strokes encode i, u, n, m.\n"
    "- The z-shaped symbol between words = Tironian et (meaning 'and'). Transcribe as z or et.\n"
    "- Long-s (ſ) has NO crossbar; f HAS a crossbar. Do not confuse them.\n"
    "- Bars/tildes over letters = abbreviation marks (nasal or missing letters). Keep them.\n"
    "- Two-column layouts: read left column completely first, then right column.\n"
    "- Decorated/colored initials are real letters — include them in the transcription.\n"
    "\n"
    "Language: use old_french for pre-1300 French, middle_french for c.1300-1500.\n"
    "If nothing readable is visible, return lines=[] and text=\"\"."
)
SAIA_OCR_JSON_REPAIR_PROMPT = (
    "Return VALID JSON only. Keys must be exactly: lines, text, script_hint, detected_language, confidence, warnings. "
    "No markdown. No extra keys."
)

DEFAULT_MODEL_PREFS_CSV = ",".join(DEFAULT_SAIA_OCR_MODEL_PREFS)
DEFAULT_OCR_TEMPERATURE = 0.0
DEFAULT_OCR_MAX_TOKENS = 4096
SAIA_OCR_IMAGE_SIZE_RETRY_LIMIT = max(
    0,
    int(settings.ocr_image_size_retry_limit or os.getenv("OCR_IMAGE_SIZE_RETRY_LIMIT", "2") or "2"),
)

def _extract_base64_payload(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise SaiaOCRAgentError("image_b64 is required.")
    if text.startswith("data:"):
        marker = text.find(",")
        if marker == -1:
            raise SaiaOCRAgentError("image_b64 data URL is malformed.")
        text = text[marker + 1 :]
    try:
        base64.b64decode(text, validate=True)
    except Exception as exc:  # pragma: no cover - validation guard
        raise SaiaOCRAgentError("image_b64 is not valid base64.") from exc
    return text


# ---------------------------------------------------------------------------
# Image enhancement for manuscript OCR  (language-agnostic, single-pass)
# ---------------------------------------------------------------------------
# These transforms improve the signal the vision model receives without
# adding extra SAIA API calls.  They target three common manuscript issues:
#   1. Low / uneven contrast (faded ink, stained parchment)  → CLAHE on L*
#   2. Soft focus / blurry strokes                           → unsharp-mask
#   3. Sensor / compression noise                            → bilateral filter
# All operations preserve the original RGB colour space so the model still
# sees colour cues (rubricated initials, ink colour, marginalia).
# ---------------------------------------------------------------------------

def _is_image_too_large_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        ("image size" in text and "exceeds limit" in text)
        or "decompression bomb" in text
        or ("too large" in text and "pixels" in text)
    )


def _compact_error_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text[:180] if len(text) > 180 else text


def _parse_csv_or_json_list(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(",") if item.strip()]


def _filter_internvl_models(models: Sequence[str]) -> list[str]:
    return [model for model in models if INTERNVL_MODEL_HINT in str(model).lower()]


def _strip_fences(raw: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    return text


def _extract_json_object(raw: str) -> str:
    text = _strip_fences(raw)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


def _detect_script_hint(text: str) -> str:
    value = str(text or "")
    has_greek = bool(re.search(r"[\u0370-\u03FF]", value))
    has_cyrillic = bool(re.search(r"[\u0400-\u04FF]", value))
    has_latin = bool(re.search(r"[A-Za-zÀ-ÿ]", value))
    if has_latin and not has_greek and not has_cyrillic:
        return "latin"
    if has_greek and not has_latin and not has_cyrillic:
        return "greek"
    if has_cyrillic and not has_latin and not has_greek:
        return "cyrillic"
    if sum([has_latin, has_greek, has_cyrillic]) >= 2:
        return "mixed"
    return "unknown"


def _normalize_detected_language(value: str | None) -> str:
    language = str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "la": "latin",
        "fr": "french",
        "fro": "old_french",
        "frm": "middle_french",
        "enm": "middle_english",
        "ang": "old_english",
        "de": "german",
        "it": "italian",
        "es": "spanish",
        "pt": "portuguese",
        "nl": "dutch",
        "ca": "catalan",
        "oc": "occitan",
        "he": "hebrew",
        "ar": "arabic",
        "el": "greek",
    }
    language = aliases.get(language, language)
    if language in ALLOWED_DETECTED_LANGUAGES:
        return language
    return "unknown"


def _normalize_language_token(token: str) -> str:
    return str(token or "").lower().replace("ſ", "s").replace("v", "u").replace("j", "i")


def _resolve_french_family_language(text: str) -> str | None:
    value = str(text or "")
    if not value.strip():
        return None

    tokens: list[str] = []
    for raw_token in re.split(r"\s+", value.lower()):
        if not raw_token:
            continue
        cleaned = "".join(ch for ch in raw_token if ch.isalpha())
        if not cleaned:
            continue
        if LATIN_TOKEN_RE.match(cleaned) is None:
            continue
        tokens.append(_normalize_language_token(cleaned))

    if not tokens:
        return None

    strong_hits = 0
    function_hits = 0
    old_hits = 0
    middle_hits = 0
    anglo_hits = 0
    for token in tokens:
        if any(token == anchor or token.startswith(anchor) for anchor in FRENCH_STRONG_ANCHORS):
            strong_hits += 1
        if token in FRENCH_FUNCTION_WORDS:
            function_hits += 1
        if token in FRENCH_OLD_MARKERS:
            old_hits += 1
        if token in FRENCH_MIDDLE_MARKERS:
            middle_hits += 1
        if token in ANGLO_NORMAN_MARKERS:
            anglo_hits += 1

    if not (strong_hits >= 2 or (strong_hits >= 1 and function_hits >= 2)):
        return None
    if anglo_hits >= 2:
        return "anglo_norman"
    if old_hits >= 2 and middle_hits == 0:
        return "old_french"
    return "middle_french"


def _fallback_detected_language(script_hint: str, text: str) -> str:
    value = str(text or "")
    script = str(script_hint or "unknown").lower()

    if not value.strip():
        return "unknown"

    if OLD_ENGLISH_MARKERS_RE.search(value):
        return "old_english"
    if ARABIC_RE.search(value):
        return "arabic"
    if HEBREW_RE.search(value):
        return "hebrew"
    french_family = _resolve_french_family_language(value)
    if french_family is not None:
        return french_family
    if script == "greek":
        return "greek"
    if script == "cyrillic":
        return "church_slavonic"
    if script == "mixed":
        return "mixed"
    return "latin"


def _normalize_lines(lines: Sequence[Any], text: str) -> list[str]:
    normalized: list[str] = []
    for item in lines:
        raw = str(item)
        if "\n" in raw or "\r" in raw:
            normalized.extend([fragment.strip() for fragment in raw.splitlines() if fragment.strip()])
            continue
        stripped = raw.strip()
        if stripped:
            normalized.append(stripped)
    if not normalized and text:
        normalized = [line.strip() for line in text.splitlines() if line.strip()]
    return normalized


def _repair_lines_and_text(lines: Sequence[Any], text: str) -> tuple[list[str], str, list[str]]:
    normalized_warnings: list[str] = []
    had_embedded_newline = any(("\n" in str(item) or "\r" in str(item)) for item in lines)
    normalized_lines = _normalize_lines(lines, text)

    if had_embedded_newline:
        normalized_warnings.append("repair:lines_split_embedded_newlines")
    if not any(str(item).strip() for item in lines) and str(text).strip() and normalized_lines:
        normalized_warnings.append("repair:rebuilt_lines_from_text")

    normalized_text = "\n".join(normalized_lines).strip()
    if str(text).strip() != normalized_text:
        normalized_warnings.append("TEXT_JOIN_FIXED")

    return normalized_lines, normalized_text, list(dict.fromkeys(normalized_warnings))


def _junk_ratio(text: str) -> float:
    tokens = [token for token in re.split(r"\s+", str(text or "").strip()) if token]
    considered = 0
    junk = 0
    for token in tokens:
        if len(token) < 3:
            continue
        considered += 1
        letters = [ch for ch in token if ch.isalpha()]
        letters_count = len(letters)
        if letters_count < 3:
            junk += 1
            continue
        vowel_count = sum(1 for ch in letters if ch.lower() in _OCR_VOWELS)
        vowel_ratio = vowel_count / max(1, letters_count)
        non_letters = len(token) - letters_count
        non_letter_ratio = non_letters / max(1, len(token))
        if vowel_ratio < 0.20 or non_letter_ratio > 0.30:
            junk += 1
    if considered == 0:
        return 0.0
    return junk / considered


def _letter_count(text: str) -> int:
    return sum(1 for ch in str(text or "") if ch.isalpha())


def _apply_ocr_confidence_caps(text: str, confidence: float, warnings: list[str]) -> tuple[float, list[str]]:
    junk_ratio = _junk_ratio(text)
    letters_count = _letter_count(text)
    next_warnings = list(warnings)
    if junk_ratio >= 0.35 or (letters_count < 120 and junk_ratio >= 0.25):
        capped = min(confidence, 0.75)
        next_warnings.extend(["ocr:confidence_capped", f"ocr:junk_ratio={junk_ratio:.2f}"])
        return capped, list(dict.fromkeys(next_warnings))
    if junk_ratio >= 0.20:
        capped = min(confidence, 0.85)
        next_warnings.append(f"ocr:junk_ratio={junk_ratio:.2f}")
        return capped, list(dict.fromkeys(next_warnings))
    return confidence, list(dict.fromkeys(next_warnings))


def _parse_ocr_payload(raw: str) -> dict[str, Any] | None:
    candidate = _extract_json_object(raw)
    try:
        payload = json.loads(candidate)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if set(payload.keys()) != set(REQUIRED_JSON_KEYS):
        return None

    lines = payload.get("lines")
    text = payload.get("text")
    script_hint = str(payload.get("script_hint") or "unknown").strip().lower()
    detected_language = _normalize_detected_language(str(payload.get("detected_language") or "unknown"))
    confidence = payload.get("confidence", 0.0)
    warnings = payload.get("warnings", [])

    if not isinstance(lines, list):
        return None
    if not isinstance(text, str):
        return None
    if not isinstance(warnings, list):
        return None

    normalized_lines, normalized_text, normalized_warnings = _repair_lines_and_text(lines, text)
    if script_hint not in ALLOWED_SCRIPT_HINTS:
        script_hint = _detect_script_hint(normalized_text)
    if script_hint not in ALLOWED_SCRIPT_HINTS:
        script_hint = "unknown"
    if detected_language == "unknown" and normalized_text:
        script_guess = _detect_script_hint(normalized_text)
        if script_guess == "greek":
            detected_language = "greek"
    if detected_language == "unknown":
        detected_language = _fallback_detected_language(script_hint, normalized_text)
    try:
        confidence_value = max(0.0, min(1.0, float(confidence)))
    except Exception:
        confidence_value = 0.0
    normalized_warnings.extend([str(item).strip() for item in warnings if str(item).strip()])

    return {
        "lines": normalized_lines,
        "text": normalized_text,
        "script_hint": script_hint,
        "detected_language": detected_language,
        "confidence": confidence_value,
        "warnings": list(dict.fromkeys(normalized_warnings)),
    }


def _format_location_suggestions(suggestions: Sequence[SaiaOCRLocationSuggestion]) -> str:
    if not suggestions:
        return ""
    filtered = []
    for suggestion in suggestions:
        category = str(suggestion.category or "").lower()
        if any(token in category for token in ("miniature", "illustration", "graphic")):
            continue
        filtered.append(suggestion)
    if not filtered:
        filtered = list(suggestions)

    lines = [
        "Location suggestions (x,y,w,h in full-page coordinates; hints only):",
        "These hints are metadata only. Never copy ids, numbering, labels, or coordinates into the transcription.",
        "",
    ]
    xmins: list[float] = []
    ymins: list[float] = []
    xmaxs: list[float] = []
    ymaxs: list[float] = []

    for idx, suggestion in enumerate(filtered, start=1):
        bbox = [float(v) for v in (suggestion.bbox_xywh or [])]
        if len(bbox) != 4:
            continue
        category = str(suggestion.category or "text")
        x, y, w, h = bbox
        xmins.append(x)
        ymins.append(y)
        xmaxs.append(x + w)
        ymaxs.append(y + h)
        lines.append(f"- {category}: ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")
    if xmins and ymins and xmaxs and ymaxs:
        tx = min(xmins)
        ty = min(ymins)
        tw = max(xmaxs) - tx
        th = max(ymaxs) - ty
        lines.insert(0, f"Suggested text block: ({tx:.1f}, {ty:.1f}, {tw:.1f}, {th:.1f})")
        lines.insert(1, "")
    if len(lines) <= 2:
        return ""
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tile-based OCR helpers: crop → OCR each tile → stitch in reading order
# ---------------------------------------------------------------------------

import logging as _logging

_tile_logger = _logging.getLogger("saia_ocr_agent.tiles")

_TILE_MIN_SIDE = 64  # minimum crop dimension (px) before upscaling
_TILE_COLUMN_SPLIT_X_RATIO = 0.45  # x-center threshold to distinguish columns

SAIA_TILE_SYSTEM_ADDENDUM = (
    "\nThis is a cropped region of a manuscript page. "
    "Transcribe ONLY what is visible in this crop. "
    "Do NOT repeat text that is not present in the crop. "
    "Preserve line breaks as they appear in the crop. "
    "Do NOT output Arabic numerals (0-9) unless clearly written as numerals. "
    "Do NOT insert spaces inside a single word.\n"
    "REMEMBER: Gothic minims (vertical strokes) encode i/u/n/m — read them carefully. "
    "The z-shaped mark between words = 'et' (Tironian nota). "
    "Long-s has no crossbar; f has a crossbar. "
    "Bars over letters = abbreviation marks — keep them."
)


# ---------------------------------------------------------------------------
# Morphological block merging via PIL mask dilation + BFS connected components
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Diplomatic sanity: compute_ocr_sanity + enforce_diplomatic_uncertainty
# ---------------------------------------------------------------------------

_CONSONANT_MERGE_SET = set("cdlnprstvxCDLNPRSTVX")
# Backward-compatible alias
compute_ocr_sanity = compute_sanity


def build_saia_ocr_messages(
    image_b64: str,
    *,
    repair_json: bool = False,
    location_suggestions: Sequence[SaiaOCRLocationSuggestion] | None = None,
) -> list[dict[str, Any]]:
    text_prompt = SAIA_OCR_USER_PROMPT
    location_block = _format_location_suggestions(location_suggestions or [])
    if location_block:
        text_prompt = (
            f"{text_prompt}\n\n"
            "Use the suggested text block only as a focus guide.\n"
            "Do not transcribe the suggestion metadata itself.\n"
            "Ignore decorative miniatures unless they contain letters.\n\n"
            f"{location_block}"
        )
    if repair_json:
        text_prompt = f"{text_prompt}\n\n{SAIA_OCR_JSON_REPAIR_PROMPT}"
    return [
        {"role": "system", "content": SAIA_OCR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        },
    ]


def build_tile_ocr_messages(
    tile_image_b64: str,
    *,
    repair_json: bool = False,
    tile_index: int = 0,
    total_tiles: int = 1,
) -> list[dict[str, Any]]:
    """Build messages for a single tile crop OCR. Uses a system prompt addendum to
    prevent duplication and keep output focused on visible content only."""
    system_prompt = SAIA_OCR_SYSTEM_PROMPT + SAIA_TILE_SYSTEM_ADDENDUM
    text_prompt = (
        "This is one cropped text-bearing region from the manuscript page. "
        "Transcribe ONLY what is visible in this crop. "
        "Do not output any tile number or prompt metadata.\n\n"
        + SAIA_OCR_USER_PROMPT
    )
    if repair_json:
        text_prompt = f"{text_prompt}\n\n{SAIA_OCR_JSON_REPAIR_PROMPT}"
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tile_image_b64}"}},
            ],
        },
    ]


def _pick_better_result(tile_result: "SaiaOCRResponse", fullpage_result: "SaiaOCRResponse") -> "SaiaOCRResponse":
    """Choose the better OCR result between tile-based and full-page based on sanity metrics."""
    tile_text = str(tile_result.text or "").strip()
    full_text = str(fullpage_result.text or "").strip()

    if not tile_text:
        return fullpage_result
    if not full_text:
        return tile_result

    tile_metrics = _ocr_sanity_metrics(tile_text)
    full_metrics = _ocr_sanity_metrics(full_text)

    # Compute a simple score: lower is better (less junk)
    def _score(m: dict[str, float], conf: float) -> float:
        penalty = m["single_char_ratio"] * 2.0 + m["digit_ratio"] * 1.5 + m["junk_ratio"] * 1.0
        return penalty - (conf * 0.5)

    tile_score = _score(tile_metrics, float(tile_result.confidence or 0.0))
    full_score = _score(full_metrics, float(fullpage_result.confidence or 0.0))

    _tile_logger.info(
        "Quality comparison: tile_score=%.3f full_score=%.3f (lower is better)",
        tile_score,
        full_score,
    )

    if tile_score <= full_score:
        return tile_result
    return fullpage_result


class SaiaOCRAgent:
    def __init__(
        self,
        *,
        client: SaiaClient | None = None,
        model_prefs: Sequence[str] | None = None,
    ) -> None:
        self.client = client or SaiaClient()
        self.model_prefs = (
            _filter_internvl_models(model_prefs) if model_prefs is not None else self._resolve_model_prefs()
        )
        self.temperature = self._resolve_temperature()
        self.max_tokens = self._resolve_max_tokens()
        # At temperature 0 the reading must not depend on the draw: ask for
        # greedy decoding too (n=1, no penalties, do_sample=False; the client
        # drops do_sample if the provider rejects it). A temperature set above
        # 0 is a choice to sample, and is left alone.
        self.deterministic = self.temperature == 0.0

    @staticmethod
    def _resolve_model_prefs() -> list[str]:
        prefs = _filter_internvl_models(
            _parse_csv_or_json_list(
            settings.saia_ocr_models
            or os.getenv("SAIA_OCR_MODELS", "")
            or settings.saia_ocr_model_prefs
            or os.getenv("SAIA_OCR_MODEL_PREFS", "")
            or settings.saia_ocr_model_preferences
            or DEFAULT_MODEL_PREFS_CSV
            )
        )
        return prefs or _filter_internvl_models(DEFAULT_SAIA_OCR_MODEL_PREFS)

    @staticmethod
    def _resolve_temperature() -> float:
        raw = str(
            settings.saia_ocr_temperature
            if settings.saia_ocr_temperature is not None
            else os.getenv("SAIA_OCR_TEMPERATURE", str(DEFAULT_OCR_TEMPERATURE))
        )
        try:
            return float(raw)
        except Exception:
            return DEFAULT_OCR_TEMPERATURE

    @staticmethod
    def _resolve_max_tokens() -> int:
        raw = str(
            settings.saia_ocr_max_tokens
            if settings.saia_ocr_max_tokens is not None
            else os.getenv("SAIA_OCR_MAX_TOKENS", str(DEFAULT_OCR_MAX_TOKENS))
        )
        try:
            return max(256, int(raw))
        except Exception:
            return DEFAULT_OCR_MAX_TOKENS

    def _select_candidate_models(self, available_models: Sequence[str]) -> list[str]:
        by_lower = {model.lower(): model for model in available_models}
        ordered: list[str] = []
        seen: set[str] = set()

        for preferred in self.model_prefs:
            resolved = by_lower.get(preferred.lower())
            if not resolved:
                continue
            key = resolved.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(resolved)

        if ordered:
            return ordered

        for model in available_models:
            key = model.lower()
            if key in seen:
                continue
            if INTERNVL_MODEL_HINT in key:
                seen.add(key)
                ordered.append(model)

        return ordered

    def _request_json_from_model(
        self,
        *,
        model: str,
        image_b64: str,
        location_suggestions: Sequence[SaiaOCRLocationSuggestion],
    ) -> dict[str, Any]:
        messages = build_saia_ocr_messages(
            image_b64,
            repair_json=False,
            location_suggestions=location_suggestions,
        )
        response = self._chat_completion_with_optional_json_object(model=model, messages=messages)
        parsed = _parse_ocr_payload(str(response.get("text") or ""))
        if parsed is not None:
            return parsed

        repair_messages = build_saia_ocr_messages(
            image_b64,
            repair_json=True,
            location_suggestions=location_suggestions,
        )
        repair_response = self._chat_completion_with_optional_json_object(model=model, messages=repair_messages)
        repaired = _parse_ocr_payload(str(repair_response.get("text") or ""))
        if repaired is None:
            return {
                "lines": [],
                "text": "",
                "script_hint": "unknown",
                "detected_language": "unknown",
                "confidence": 0.0,
                "warnings": ["INVALID_OCR_JSON"],
            }
        repaired["warnings"] = list(dict.fromkeys([*repaired["warnings"], "JSON_REPAIR_RETRY"]))
        return repaired

    def _chat_completion_with_optional_json_object(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            return self.client.chat_completion(
                model=model,
                temperature=self.temperature,
                top_p=1.0,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                messages=messages,
                deterministic=self.deterministic,
            )
        except Exception as exc:
            text = str(exc).lower()
            if "response_format" not in text and "json_object" not in text:
                raise
            return self.client.chat_completion(
                model=model,
                temperature=self.temperature,
                top_p=1.0,
                max_tokens=self.max_tokens,
                messages=messages,
                deterministic=self.deterministic,
            )

    @staticmethod
    def _apply_diplomatic_enforcement(
        lines: list[str],
        script_hint: str,
        detected_language: str,
    ) -> tuple[list[str], str, list[str]]:
        """Run sanitize_lines + force_uncertainty_markers, return (lines, text, extra_warnings)."""
        sanitized, count = sanitize_lines(lines, detected_language, script_hint)
        sanitized_text = "\n".join(sanitized).strip()
        extra_warnings: list[str] = []
        if count > 0:
            extra_warnings.append(f"SANITIZED_TOKENS:{count}")

        # Force uncertainty markers if text is noisy but has none
        post_san_sanity = compute_sanity(sanitized_text)
        sanitized, forced = force_uncertainty_markers(sanitized, post_san_sanity)
        if forced > 0:
            extra_warnings.append(f"FORCED_UNCERTAINTY:{forced}")
            sanitized_text = "\n".join(sanitized).strip()

        return sanitized, sanitized_text, extra_warnings

    @staticmethod
    def _enrich_raw_json(
        raw_json: Any,
        sanity_metrics: dict[str, float],
        quality_label: str,
    ) -> dict[str, Any]:
        """Pack sanity_metrics + quality_label into the raw_json blob."""
        blob: dict[str, Any] = {}
        if isinstance(raw_json, dict):
            blob.update(raw_json)
        blob["sanity_metrics"] = sanity_metrics
        blob["quality_label"] = quality_label
        return blob

    def _maybe_proofread(
        self,
        *,
        model: str,
        script_hint: str,
        detected_language: str,
        lines: list[str],
        text: str,
        enabled: bool,
    ) -> tuple[list[str], str, list[str]]:
        if not enabled:
            return lines, text, []
        if len(lines) < 3:
            return lines, text, []
        if _garbage_ratio(text) >= 0.35:
            return lines, text, ["PROOFREAD_SKIPPED_LOW_QUALITY"]

        proofreader = OcrProofreaderAgent(client=self.client, model_override=model)
        try:
            corrected = proofreader.proofread(text, script_hint, detected_language=detected_language)
        except Exception:
            return lines, text, ["PROOFREAD_FAILED"]

        corrected = corrected.strip()
        if not corrected:
            return lines, text, ["PROOFREAD_EMPTY_RESULT"]
        corrected_lines = [line.strip() for line in corrected.splitlines() if line.strip()]
        corrected_text = "\n".join(corrected_lines).strip()
        if not corrected_text:
            return lines, text, ["PROOFREAD_EMPTY_RESULT"]
        if corrected_text.strip() == text.strip():
            return corrected_lines, corrected_text, ["proofread:no_changes"]

        # ── Proofreader guard: reject hallucinated rewrites ──
        verdict = check_proofread_delta(text, corrected_text)
        if not verdict.accepted:
            return lines, text, [verdict.reason]

        return corrected_lines, corrected_text, []

    def _request_tile_json_from_model(
        self,
        *,
        model: str,
        tile_image_b64: str,
        tile_index: int,
        total_tiles: int,
    ) -> dict[str, Any]:
        """OCR a single tile crop using the tile-specific prompt."""
        messages = build_tile_ocr_messages(
            tile_image_b64,
            repair_json=False,
            tile_index=tile_index,
            total_tiles=total_tiles,
        )
        response = self._chat_completion_with_optional_json_object(model=model, messages=messages)
        parsed = _parse_ocr_payload(str(response.get("text") or ""))
        if parsed is not None:
            return parsed

        repair_messages = build_tile_ocr_messages(
            tile_image_b64,
            repair_json=True,
            tile_index=tile_index,
            total_tiles=total_tiles,
        )
        repair_response = self._chat_completion_with_optional_json_object(model=model, messages=repair_messages)
        repaired = _parse_ocr_payload(str(repair_response.get("text") or ""))
        if repaired is None:
            return {
                "lines": [],
                "text": "",
                "script_hint": "unknown",
                "detected_language": "unknown",
                "confidence": 0.0,
                "warnings": ["INVALID_OCR_JSON"],
            }
        repaired["warnings"] = list(dict.fromkeys([*repaired["warnings"], "JSON_REPAIR_RETRY"]))
        return repaired

    # ---- Column-split fallback ------------------------------------------------

    _COL_SPLIT_MIN_WIDTH = 1200   # only try column split on wide-enough images
    _COL_SPLIT_OVERLAP = 0.04     # 4% overlap between left/right halves

    def _try_column_split_ocr(
        self,
        *,
        image_b64: str,
        model: str,
    ) -> SaiaOCRResponse | None:
        """Split a wide page into left/right halves, OCR each, stitch.

        Returns None when the image is not wide enough to warrant splitting.
        """
        image = _open_rgb_image_from_b64(image_b64)
        img_w, img_h = image.size

        if img_w < self._COL_SPLIT_MIN_WIDTH:
            return None

        # Only try column split when the page is roughly portrait or square
        aspect = img_w / max(1, img_h)
        if aspect > 1.6:
            # Very landscape → unlikely two-column manuscript page
            return None

        overlap_px = max(8, int(img_w * self._COL_SPLIT_OVERLAP))
        mid = img_w // 2

        col_rects = [
            (0, 0, mid + overlap_px, img_h),          # left column
            (mid - overlap_px, 0, img_w, img_h),       # right column
        ]

        all_lines: list[str] = []
        all_warnings: list[str] = ["COLUMN_SPLIT_OCR:2_cols"]
        confidences: list[tuple[float, float]] = []
        languages: list[str] = []

        for idx, rect in enumerate(col_rects):
            crop = image.crop(rect)
            tile_b64 = _encode_png_base64(crop)
            tile_b64, _ = _prepare_image_for_ocr(tile_b64)

            try:
                parsed = self._request_tile_json_from_model(
                    model=model,
                    tile_image_b64=tile_b64,
                    tile_index=idx,
                    total_tiles=2,
                )
            except Exception as exc:
                all_warnings.append(f"col_{idx}: OCR failed: {exc}")
                continue

            tile_lines = [str(ln) for ln in parsed.get("lines", []) if str(ln).strip()]
            tile_lang = _normalize_detected_language(str(parsed.get("detected_language") or "unknown"))
            tile_conf = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
            tile_warnings = [str(w) for w in parsed.get("warnings", []) if str(w).strip()]
            for w in tile_warnings:
                all_warnings.append(f"col_{idx}: {w}")

            if tile_lines:
                all_lines.extend(tile_lines)
            if tile_lang and tile_lang != "unknown":
                languages.append(tile_lang)
            area = float((rect[2] - rect[0]) * (rect[3] - rect[1]))
            confidences.append((tile_conf, area))
            _tile_logger.info("Column %d: %d lines, lang=%s, conf=%.2f", idx, len(tile_lines), tile_lang, tile_conf)

        if not all_lines:
            return None

        stitched_text = "\n".join(all_lines)
        total_area = sum(a for _, a in confidences) or 1.0
        final_confidence = sum(c * a for c, a in confidences) / total_area
        final_script_hint = _detect_script_hint(stitched_text)
        if final_script_hint not in ALLOWED_SCRIPT_HINTS:
            final_script_hint = "unknown"

        if languages:
            from collections import Counter
            lc = Counter(languages)
            top_lang, _ = lc.most_common(1)[0]
            final_language = top_lang
        else:
            final_language = _fallback_detected_language(final_script_hint, stitched_text)

        final_confidence, all_warnings = _apply_ocr_confidence_caps(stitched_text, final_confidence, all_warnings)
        all_warnings = list(dict.fromkeys(all_warnings))

        return SaiaOCRResponse(
            status="FULL" if stitched_text.strip() else "FAIL",
            model_used=model,
            fallbacks=[],
            fallbacks_used=[],
            warnings=all_warnings,
            lines=all_lines,
            text=stitched_text,
            script_hint=final_script_hint,
            detected_language=final_language,
            confidence=final_confidence,
            raw_json={"column_split": True, "columns": 2},
        )

    def extract_tiles(
        self,
        *,
        image_b64: str,
        location_suggestions: Sequence[SaiaOCRLocationSuggestion],
        model: str,
        page_script_hint: str = "latin",
    ) -> SaiaOCRResponse | None:
        """Tile-based OCR: crop image by location suggestions, OCR each tile,
        stitch results in reading order. Returns None if tiles cannot be produced
        (e.g., no valid rects)."""
        if not location_suggestions:
            return None

        image = _open_rgb_image_from_b64(image_b64)
        img_w, img_h = image.size

        # 1. Normalize suggestions → rects
        rects = _suggestions_to_rects(location_suggestions, img_w, img_h)
        if not rects:
            _tile_logger.info("No valid rects from %d suggestions; skipping tile OCR.", len(location_suggestions))
            return None

        # 2. Morphological block merging (dilation + connected components)
        rects = _rects_to_block_rects(rects, img_w, img_h)
        _tile_logger.info("After block merge: %d blocks from %d suggestions.", len(rects), len(location_suggestions))

        # 3. Pad each tile
        rects = [_pad_rect(r, img_w, img_h) for r in rects]

        # 4. Reading order
        rects = _order_tiles_reading(rects, img_w)

        # 5. Crop and OCR each tile
        all_lines: list[str] = []
        all_warnings: list[str] = []
        languages: list[str] = []
        confidences: list[tuple[float, float]] = []  # (confidence, area_weight)
        tile_results: list[dict[str, Any]] = []

        for idx, rect in enumerate(rects):
            crop = _crop_and_maybe_upscale(image, rect)
            tile_b64 = _encode_png_base64(crop)
            tile_b64, tile_prep_warnings = _prepare_image_for_ocr(tile_b64)

            try:
                parsed = self._request_tile_json_from_model(
                    model=model,
                    tile_image_b64=tile_b64,
                    tile_index=idx,
                    total_tiles=len(rects),
                )
            except Exception as exc:
                all_warnings.append(f"tile_{idx}: OCR failed: {exc}")
                _tile_logger.warning("Tile %d OCR failed: %s", idx, exc)
                continue

            tile_lines = [str(line) for line in parsed.get("lines", []) if str(line).strip()]
            tile_text = str(parsed.get("text") or "").strip()
            tile_script = str(parsed.get("script_hint") or "unknown").lower()
            tile_lang = _normalize_detected_language(str(parsed.get("detected_language") or "unknown"))
            tile_conf = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
            tile_warnings = [str(w) for w in parsed.get("warnings", []) if str(w).strip()]

            tile_sanity = compute_sanity(tile_text) if tile_text else {
                "weird_ratio": 1.0,
                "uncertainty_marker_ratio": 0.0,
            }

            # --- Multi-view retry: if initial result is weak, try alt variant ---
            _mv_weak = (
                tile_text
                and tile_conf < 0.65
                and (
                    tile_sanity.get("weird_ratio", 0.0) >= 0.08
                    or tile_sanity.get("single_char_ratio", 0.0) >= 0.12
                )
            )
            if _mv_weak:
                try:
                    variants = generate_variants(crop)
                    alt = pick_retry_variant(variants, "enhanced_rgb")
                    if alt is not None:
                        alt_b64 = _encode_png_base64(alt.image)
                        alt_parsed = self._request_tile_json_from_model(
                            model=model,
                            tile_image_b64=alt_b64,
                            tile_index=idx,
                            total_tiles=len(rects),
                        )
                        alt_text = str(alt_parsed.get("text") or "").strip()
                        alt_conf = max(0.0, min(1.0, float(alt_parsed.get("confidence", 0.0) or 0.0)))
                        alt_sanity = compute_sanity(alt_text) if alt_text else {"weird_ratio": 1.0}
                        # Pick the variant with lower weird_ratio + higher conf
                        alt_score = alt_conf * (1.0 - alt_sanity.get("weird_ratio", 0.0))
                        orig_score = tile_conf * (1.0 - tile_sanity.get("weird_ratio", 0.0))
                        if alt_score > orig_score and alt_text:
                            _tile_logger.info(
                                "Tile %d: multiview %s beat enhanced_rgb (%.3f > %.3f)",
                                idx, alt.label, alt_score, orig_score,
                            )
                            parsed = alt_parsed
                            tile_lines = [str(ln) for ln in parsed.get("lines", []) if str(ln).strip()]
                            tile_text = alt_text
                            tile_script = str(parsed.get("script_hint") or "unknown").lower()
                            tile_lang = _normalize_detected_language(str(parsed.get("detected_language") or "unknown"))
                            tile_conf = alt_conf
                            tile_warnings = [str(w) for w in parsed.get("warnings", []) if str(w).strip()]
                            tile_sanity = alt_sanity
                            all_warnings.append(f"tile_{idx}: MULTIVIEW_RETRY:{alt.label}")
                except Exception as exc:
                    _tile_logger.debug("Tile %d multiview retry failed: %s", idx, exc)

            # --- Lexical trust adjustment ---
            if tile_text and tile_lang != "unknown":
                tile_conf, lex_warns = lexical_trust_adjustment(tile_conf, tile_text, tile_lang)
                tile_warnings.extend(lex_warns)

            # --- Non-Latin tile contamination check ---
            has_non_latin_warning = any(
                "non-latin" in str(w).lower() or "non_latin" in str(w).lower()
                for w in tile_warnings
            )
            script_mismatch = (
                page_script_hint == "latin"
                and tile_script not in ("latin", "unknown")
            )
            if tile_text and (
                script_mismatch or has_non_latin_warning
            ):
                # Try rerun with +10% padding
                padded = _pad_rect(rect, img_w, img_h, pad_ratio=_TILE_PAD_RATIO + 0.10)
                retry_ok = False
                if padded != rect:
                    try:
                        crop2 = _crop_and_maybe_upscale(image, padded)
                        tile_b64_2 = _encode_png_base64(crop2)
                        tile_b64_2, _ = _prepare_image_for_ocr(tile_b64_2)
                        parsed2 = self._request_tile_json_from_model(
                            model=model, tile_image_b64=tile_b64_2,
                            tile_index=idx, total_tiles=len(rects),
                        )
                        r2_script = str(parsed2.get("script_hint") or "unknown").lower()
                        r2_text = str(parsed2.get("text") or "").strip()
                        r2_warnings = [str(w) for w in parsed2.get("warnings", []) if str(w).strip()]
                        r2_sanity = compute_sanity(r2_text) if r2_text else {
                            "weird_ratio": 1.0,
                            "uncertainty_marker_ratio": 0.0,
                        }
                        r2_has_non_latin = any(
                            "non-latin" in w.lower() or "non_latin" in w.lower()
                            for w in r2_warnings
                        )
                        retry_bad = (
                            r2_has_non_latin
                            or (page_script_hint == "latin" and r2_script not in ("latin", "unknown"))
                            or r2_sanity.get("weird_ratio", 1.0) >= 0.15
                            or (
                                r2_sanity.get("uncertainty_marker_ratio", 0.0) == 0.0
                                and r2_sanity.get("weird_ratio", 0.0) > 0.05
                            )
                        )
                        if not retry_bad:
                            parsed = parsed2
                            tile_lines = [str(ln) for ln in parsed.get("lines", []) if str(ln).strip()]
                            tile_text = r2_text
                            tile_script = r2_script
                            tile_lang = _normalize_detected_language(str(parsed.get("detected_language") or "unknown"))
                            tile_conf = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
                            tile_warnings = r2_warnings
                            tile_sanity = r2_sanity
                            all_warnings.append(f"tile_{idx}: RETRIED_WITH_PADDING_OK")
                            retry_ok = True
                    except Exception:
                        pass
                if not retry_ok:
                    drop_reason = "non_latin_or_weird_after_retry"
                    all_warnings.append(f"DROPPED_TILE_{idx}:{drop_reason}")
                    _tile_logger.info("Dropped tile %d: %s", idx, drop_reason)
                    continue

            if tile_text and (
                tile_sanity.get("weird_ratio", 0.0) >= 0.18
                or (
                    tile_sanity.get("uncertainty_marker_ratio", 0.0) == 0.0
                    and tile_sanity.get("weird_ratio", 0.0) > 0.05
                )
            ):
                all_warnings.append(f"DROPPED_TILE_{idx}:sanity")
                _tile_logger.info("Dropped tile %d on sanity gate: %s", idx, tile_sanity)
                continue

            # Prefix tile-level warnings
            for w in tile_prep_warnings:
                all_warnings.append(f"tile_{idx}: {w}")
            for w in tile_warnings:
                all_warnings.append(f"tile_{idx}: {w}")

            if tile_lines:
                all_lines.extend(tile_lines)
            elif tile_text:
                all_lines.extend([ln.strip() for ln in tile_text.splitlines() if ln.strip()])

            if tile_lang and tile_lang != "unknown":
                languages.append(tile_lang)

            # Weight confidence by tile area
            x1, y1, x2, y2 = rect
            area = float((x2 - x1) * (y2 - y1))
            confidences.append((tile_conf, area))

            tile_results.append(parsed)
            _tile_logger.info(
                "Tile %d/%d: %d lines, lang=%s, conf=%.2f",
                idx + 1, len(rects), len(tile_lines), tile_lang, tile_conf,
            )

        if not all_lines:
            _tile_logger.info("Tile OCR produced no lines; returning None to trigger fallback.")
            return None

        # 6. Stitch results
        stitched_text = "\n".join(all_lines)
        stitched_lines = all_lines

        # Script hint from stitched text. Computed here because the language
        # vote below reads final_script_hint; it used to be assigned only after
        # that block, so the no-majority branch raised UnboundLocalError.
        final_script_hint = _detect_script_hint(stitched_text)
        if final_script_hint not in ALLOWED_SCRIPT_HINTS:
            final_script_hint = "unknown"

        # Final detected_language: most frequent non-unknown
        if languages:
            from collections import Counter
            lang_counts = Counter(languages)
            most_common_lang, most_common_count = lang_counts.most_common(1)[0]
            if len(lang_counts) == 1:
                final_language = most_common_lang
            elif most_common_count > len(languages) // 2:
                final_language = most_common_lang
            else:
                # All tiles are Latin-script: prefer majority vote over "mixed"
                # Only return "mixed" when there's strong multi-script evidence
                if final_script_hint in ("latin", "unknown"):
                    final_language = most_common_lang
                else:
                    final_language = "mixed"
        else:
            final_language = _fallback_detected_language(
                _detect_script_hint(stitched_text), stitched_text,
            )

        # Final confidence: area-weighted average
        total_area = sum(area for _, area in confidences) or 1.0
        final_confidence = sum(conf * area for conf, area in confidences) / total_area
        final_confidence = max(0.0, min(1.0, final_confidence))

        # Apply confidence caps
        final_confidence, all_warnings = _apply_ocr_confidence_caps(stitched_text, final_confidence, all_warnings)

        all_warnings.insert(0, f"TILE_OCR:{len(rects)}_tiles")
        all_warnings = list(dict.fromkeys(all_warnings))

        status = "FULL" if stitched_text.strip() else "FAIL"
        if all_warnings:
            status = "PARTIAL" if stitched_text.strip() else "FAIL"

        return SaiaOCRResponse(
            status=status,
            model_used=model,
            fallbacks=[],
            fallbacks_used=[],
            warnings=all_warnings,
            lines=stitched_lines,
            text=stitched_text,
            script_hint=final_script_hint,
            detected_language=final_language,
            confidence=final_confidence,
            raw_json={"tile_results": tile_results, "tile_count": len(rects)},
        )

    def extract(self, payload: SaiaOCRRequest) -> SaiaOCRResponse:
        image_b64 = _extract_base64_payload(payload.image_b64 or "")
        image_b64, preprocess_warnings = _prepare_image_for_ocr(image_b64)
        available_models = self.client.list_models()
        candidate_models = self._select_candidate_models(available_models)
        if not candidate_models:
            raise SaiaOCRAgentError("No InternVL model is available on SAIA /models.")

        # --- Tile-based OCR when location_suggestions are present ---
        if payload.location_suggestions and len(payload.location_suggestions) >= 1:
            primary_model = candidate_models[0]
            _tile_logger.info(
                "Attempting tile-based OCR with %d location suggestions, model=%s",
                len(payload.location_suggestions),
                primary_model,
            )
            try:
                tile_result = self.extract_tiles(
                    image_b64=image_b64,
                    location_suggestions=payload.location_suggestions,
                    model=primary_model,
                    page_script_hint=str(payload.script_hint_seed or "latin").lower(),
                )
            except Exception as exc:
                _tile_logger.warning("Tile OCR failed, falling back to full-page: %s", exc)
                tile_result = None

            if tile_result is not None and tile_result.text.strip():
                # Quality gate: check if tile-based result is sane
                if _is_sane_ocr(tile_result.text):
                    _tile_logger.info("Tile OCR passed sanity check; using tile result.")
                    # Diplomatic sanitization
                    enforced_lines, enforced_text, enforce_warns = self._apply_diplomatic_enforcement(
                        list(tile_result.lines),
                        str(tile_result.script_hint or "unknown"),
                        str(tile_result.detected_language or "latin"),
                    )
                    combined_warnings = [*preprocess_warnings, *tile_result.warnings, *enforce_warns]
                    sanity = compute_sanity(enforced_text)
                    # Quality gate
                    if sanity["single_char_ratio"] > 0.10 or sanity["weird_ratio"] > 0.15:
                        enforced_lines, mask_count = quality_gate_enforce(enforced_lines)
                        if mask_count > 0:
                            combined_warnings.append(f"QUALITY_GATE_MASKED:{mask_count}_lines")
                            enforced_text = "\n".join(enforced_lines).strip()
                            sanity = compute_sanity(enforced_text)
                    qlabel = _quality_label_from_sanity(sanity)
                    tile_conf = sanity_adjust_confidence(
                        float(tile_result.confidence or 0.0), sanity,
                    )
                    _tile_logger.info("Tile quality_label=%s sanity=%s conf=%.3f", qlabel, sanity, tile_conf)
                    tile_result = tile_result.model_copy(
                        update={
                            "lines": enforced_lines,
                            "text": enforced_text,
                            "confidence": tile_conf,
                            "warnings": list(dict.fromkeys(combined_warnings)),
                            "raw_json": self._enrich_raw_json(tile_result.raw_json, sanity, qlabel),
                        }
                    )
                    return tile_result
                else:
                    _tile_logger.info("Tile OCR failed sanity check; falling back to full-page OCR.")
                    # Keep tile result as a candidate for comparison with full-page
                    tile_fallback = tile_result
            else:
                tile_fallback = tile_result if tile_result else None

            # Fall through to full-page OCR; will compare results later
        else:
            # --- Column-split fallback when no location suggestions ---
            tile_fallback = None
            try:
                col_result = self._try_column_split_ocr(
                    image_b64=image_b64,
                    model=candidate_models[0],
                )
            except Exception as exc:
                _tile_logger.warning("Column-split OCR failed: %s", exc)
                col_result = None
            if col_result is not None and col_result.text.strip() and _is_sane_ocr(col_result.text):
                _tile_logger.info("Column-split OCR passed sanity; using as tile_fallback candidate.")
                tile_fallback = col_result
        # --- End tile-based / column-split OCR block ---

        fallbacks: list[SaiaOCRFallback] = []
        best_partial: SaiaOCRResponse | None = None

        for index, model in enumerate(candidate_models):
            current_image_b64 = image_b64
            size_retry_count = 0
            resized_after_limit = False
            parsed: dict[str, Any] | None = None
            try:
                while True:
                    try:
                        parsed = self._request_json_from_model(
                            model=model,
                            image_b64=current_image_b64,
                            location_suggestions=payload.location_suggestions,
                        )
                        break
                    except Exception as exc:
                        if _is_image_too_large_error(exc) and size_retry_count < SAIA_OCR_IMAGE_SIZE_RETRY_LIMIT:
                            shrunk = _downscale_image_b64_for_retry(current_image_b64)
                            if shrunk:
                                current_image_b64 = shrunk
                                size_retry_count += 1
                                resized_after_limit = True
                                continue
                        raise
            except Exception as exc:
                err = "MODEL_NOT_FOUND" if is_model_not_found_error(exc) else str(exc)
                fallbacks.append(SaiaOCRFallback(model=model, error=err))
                continue
            if parsed is None:
                fallbacks.append(SaiaOCRFallback(model=model, error="EMPTY_MODEL_RESPONSE"))
                continue

            lines = [str(line) for line in parsed["lines"] if str(line).strip()]
            text = str(parsed["text"] or "").strip()
            script_hint = str(parsed["script_hint"] or "unknown").lower()
            detected_language = _normalize_detected_language(str(parsed.get("detected_language") or "unknown"))
            if script_hint not in ALLOWED_SCRIPT_HINTS:
                script_hint = _detect_script_hint(text)
            if script_hint not in ALLOWED_SCRIPT_HINTS:
                script_hint = "unknown"
            if detected_language == "unknown":
                detected_language = _fallback_detected_language(script_hint, text)
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
            warnings = [*preprocess_warnings, *[str(item) for item in parsed.get("warnings", []) if str(item).strip()]]
            confidence, warnings = _apply_ocr_confidence_caps(text, confidence, warnings)
            if resized_after_limit:
                warnings.append("AUTO_RESIZED_FOR_LIMIT_RETRY")

            script_drift = _script_drift_ratio(text, script_hint)
            if script_drift >= 0.08:
                warnings.append("SCRIPT_DRIFT")

            hallucinated = _looks_hallucinated_passage(text, lines)
            if hallucinated:
                warnings.append("HALLUCINATION_SUSPECTED")

            low_quality = _is_low_text_quality(text, lines, script_hint)
            if low_quality:
                warnings.append("LOW_TEXT_QUALITY")

            # --- Lexical trust adjustment on full-page result ---
            if text and detected_language != "unknown":
                confidence, lex_warns = lexical_trust_adjustment(confidence, text, detected_language)
                warnings.extend(lex_warns)

            if (hallucinated or low_quality) and index + 1 < len(candidate_models):
                fallbacks.append(SaiaOCRFallback(model=model, error="LOW_TEXT_QUALITY"))
                partial_status = "PARTIAL" if text else "FAIL"
                partial_candidate = SaiaOCRResponse(
                    status=partial_status,
                    model_used=model,
                    fallbacks=list(fallbacks),
                    fallbacks_used=[item.model for item in fallbacks],
                    warnings=list(dict.fromkeys(warnings)),
                    lines=lines,
                    text=text,
                    script_hint=script_hint,
                    detected_language=detected_language,
                    confidence=confidence,
                    raw_json=parsed,
                )
                if best_partial is None or len(partial_candidate.text) > len(best_partial.text):
                    best_partial = partial_candidate
                continue

            # --- Diplomatic sanitization (proofreading deferred to route) ---
            enforced_lines, enforced_text, enforce_warns = self._apply_diplomatic_enforcement(
                lines, script_hint, detected_language,
            )
            if enforce_warns:
                warnings.extend(enforce_warns)
            final_lines = enforced_lines
            final_text = enforced_text
            confidence, warnings = _apply_ocr_confidence_caps(final_text, confidence, warnings)

            # Quality gate enforcement
            final_sanity = compute_sanity(final_text)
            if final_sanity["single_char_ratio"] > 0.10 or final_sanity["weird_ratio"] > 0.15:
                final_lines, mask_count = quality_gate_enforce(final_lines)
                if mask_count > 0:
                    warnings.append(f"QUALITY_GATE_MASKED:{mask_count}_lines")
                    final_text = "\n".join(final_lines).strip()
                    final_sanity = compute_sanity(final_text)
            final_qlabel = _quality_label_from_sanity(final_sanity)
            confidence = sanity_adjust_confidence(confidence, final_sanity)
            _tile_logger.info("Full-page quality_label=%s sanity=%s conf=%.3f", final_qlabel, final_sanity, confidence)

            status = "FULL"
            if not final_text:
                status = "FAIL"
            elif warnings or fallbacks:
                status = "PARTIAL"

            fullpage_result = SaiaOCRResponse(
                status=status,
                model_used=model,
                fallbacks=fallbacks,
                fallbacks_used=[item.model for item in fallbacks],
                warnings=list(dict.fromkeys(warnings)),
                lines=final_lines,
                text=final_text,
                script_hint=script_hint,
                detected_language=detected_language,
                confidence=confidence,
                raw_json=self._enrich_raw_json(parsed, final_sanity, final_qlabel),
            )

            # Compare with tile fallback if available
            if tile_fallback is not None and tile_fallback.text.strip():
                chosen = _pick_better_result(tile_fallback, fullpage_result)
                _tile_logger.info(
                    "Tile vs full-page comparison: chose %s",
                    "tile" if chosen is tile_fallback else "full-page",
                )
                return chosen
            return fullpage_result

        if best_partial is not None:
            # Compare best partial with tile fallback too
            if tile_fallback is not None and tile_fallback.text.strip():
                chosen = _pick_better_result(tile_fallback, best_partial)
                return chosen
            return best_partial

        warning_details = [*preprocess_warnings]
        for fallback in fallbacks:
            warning_details.append(f"MODEL_ERROR:{fallback.model}:{_compact_error_text(fallback.error)}")

        return SaiaOCRResponse(
            status="FAIL",
            model_used="",
            fallbacks=fallbacks,
            fallbacks_used=[item.model for item in fallbacks],
            warnings=list(dict.fromkeys(["OCR_FAILED_ALL_MODELS", *warning_details])),
            lines=[],
            text="",
            script_hint="unknown",
            detected_language="unknown",
            confidence=0.0,
            raw_json=None,
        )
