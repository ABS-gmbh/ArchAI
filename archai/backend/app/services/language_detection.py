"""Which language an OCR'd page is in.

langdetect has no model for Latin and none for the medieval forms of French and
English, so its answer is checked against anchor words and, for French, sorted
into old, middle and Anglo-Norman by their markers. Moved out of
``app/routers/ocr.py``; the router and the quality gate both use it.
"""

from __future__ import annotations

import re

from langdetect import detect_langs

from app.services.lexicon_trust import reads_as_latin


_ALLOWED_DETECTED_LANGUAGES = {
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


_OLD_ENGLISH_MARKERS_RE = re.compile(r"[þðƿÞÐǷ]")


_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


_HEBREW_RE = re.compile(r"[\u0590-\u05FF]")


_LATIN_TOKEN_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿĀ-žſƀ-ɏ]+$")


_LANGID_MIN_LETTERS = 50


_LATIN_VOWELS = set("aeiouyàâäæéèêëîïôöœùûüÿ")


_FRENCH_STRONG_ANCHORS = {
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


_FRENCH_FUNCTION_WORDS = {"que", "et", "de", "la", "le", "des"}


_FRENCH_OLD_MARKERS = {"pur", "ki", "mei", "sun", "uostre", "nostre", "cil", "cest", "dunc"}


_FRENCH_MIDDLE_MARKERS = {
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


_ANGLO_NORMAN_MARKERS = {"anglo", "norman", "normand", "engleterre"}


_LATIN_STRONG_ANCHORS = {
    "domini",
    "amen",
    "quod",
    "canonici",
    "capitulum",
    "instrumentum",
    "publicum",
    "consensu",
    "confirmamus",
    "perpetuo",
}


def _normalize_language_token(token: str) -> str:
    return str(token or "").lower().replace("ſ", "s").replace("v", "u").replace("j", "i")


def _token_vowel_ratio(token: str) -> float:
    value = str(token or "").lower()
    if not value:
        return 0.0
    vowels = sum(1 for char in value if char in _LATIN_VOWELS)
    return float(vowels) / float(len(value))


def _clean_text_for_langid(text: str, *, enforce_min_letters: bool) -> str:
    value = str(text or "")
    if not value.strip():
        return ""

    value = value.replace("&", " et ")
    value = re.sub(r"\[(?:…|\.\.\.)\]", " ", value)
    value = value.replace("?", " ")
    value = value.replace("ſ", "s")
    value = value.replace("\n", " ")

    tokens: list[str] = []
    for raw_token in re.split(r"\s+", value):
        token = str(raw_token or "").strip()
        if not token:
            continue
        letters_only = "".join(ch for ch in token if ch.isalpha())
        if not letters_only:
            continue
        non_letters = max(0, len(token) - len(letters_only))
        if len(token) > 0 and (float(non_letters) / float(len(token))) > 0.5:
            continue
        if len(letters_only) >= 4 and _LATIN_TOKEN_RE.match(letters_only):
            if _token_vowel_ratio(letters_only) < 0.20:
                continue
        tokens.append(letters_only.lower())

    cleaned = " ".join(tokens)
    if enforce_min_letters and len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žſƀ-ɏ]", cleaned)) < _LANGID_MIN_LETTERS:
        return ""
    return cleaned


def _langid_view(text: str) -> str:
    return _clean_text_for_langid(text, enforce_min_letters=True)


# What langdetect calls Latin text: it has a model for each of these, none for Latin.
_LATIN_LOOKALIKES = frozenset({
    "french", "old_french", "middle_french", "anglo_norman", "italian", "spanish", "portuguese", "catalan", "occitan",
})


def _latin_anchor_hits(cleaned_text: str) -> int:
    tokens = {_normalize_language_token(token) for token in re.split(r"\s+", cleaned_text) if token}
    return sum(1 for anchor in _LATIN_STRONG_ANCHORS if anchor in tokens)


def _resolve_french_family_language(text: str, cleaned_text: str | None = None) -> str | None:
    cleaned = cleaned_text if cleaned_text is not None else _clean_text_for_langid(text, enforce_min_letters=False)
    if not cleaned:
        return None

    tokens = [_normalize_language_token(token) for token in re.split(r"\s+", cleaned) if token]
    if not tokens:
        return None

    strong_hits = 0
    function_hits = 0
    old_hits = 0
    middle_hits = 0
    anglo_hits = 0
    for token in tokens:
        if any(token == anchor or token.startswith(anchor) for anchor in _FRENCH_STRONG_ANCHORS):
            strong_hits += 1
        if token in _FRENCH_FUNCTION_WORDS:
            function_hits += 1
        if token in _FRENCH_OLD_MARKERS:
            old_hits += 1
        if token in _FRENCH_MIDDLE_MARKERS:
            middle_hits += 1
        if token in _ANGLO_NORMAN_MARKERS:
            anglo_hits += 1

    if not (strong_hits >= 2 or (strong_hits >= 1 and function_hits >= 2)):
        return None
    if anglo_hits >= 2:
        return "anglo_norman"
    if old_hits >= 2 and middle_hits == 0:
        return "old_french"
    return "middle_french"


def _detect_language_metadata(text: str) -> tuple[str, float | None]:
    value = str(text or "")
    if not value.strip():
        return "unknown", None
    if _ARABIC_RE.search(value):
        return "arabic", 0.99
    if _HEBREW_RE.search(value):
        return "hebrew", 0.99

    cleaned = _langid_view(value)
    if not cleaned:
        return "unknown", None

    try:
        candidates = list(detect_langs(cleaned))
    except Exception:
        candidates = []

    best_language = "unknown"
    best_confidence: float | None = None
    for candidate in candidates:
        raw_lang = str(getattr(candidate, "lang", "unknown") or "unknown").strip().lower()
        normalized = _normalize_detected_language(raw_lang)
        if normalized == "unknown":
            continue
        try:
            probability = float(getattr(candidate, "prob", None))
        except Exception:
            probability = 0.0
        best_language = normalized
        best_confidence = max(0.0, min(1.0, probability))
        break

    french_family = _resolve_french_family_language(value, cleaned_text=cleaned)
    if french_family is not None:
        if best_language in {"unknown", "latin"}:
            promoted_confidence = max(0.60, best_confidence or 0.0)
            return french_family, min(0.75, promoted_confidence)
        if best_language == "french":
            promoted_confidence = max(0.62, best_confidence or 0.0)
            return french_family, min(0.75, promoted_confidence)
    elif best_language in {
        "french",
        "old_french",
        "middle_french",
        "anglo_norman",
        "italian",
        "spanish",
        "portuguese",
        "catalan",
        "occitan",
    } and _latin_anchor_hits(cleaned) >= 2:
        return "latin", max(0.55, best_confidence or 0.0)

    if best_language == "unknown":
        return "latin", 0.45
    # langdetect has no Latin model and reports Latin as one of the Romance
    # languages; the lexical contrast settles which it is.
    if best_language in _LATIN_LOOKALIKES and reads_as_latin(value):
        return "latin", max(0.55, best_confidence or 0.0)
    return best_language, best_confidence


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
    if language in _ALLOWED_DETECTED_LANGUAGES:
        return language
    return "unknown"


def _fallback_detected_language(script_hint: str | None, text: str | None) -> str:
    value = str(text or "")
    script = str(script_hint or "unknown").lower()

    if not value.strip():
        return "unknown"

    if _OLD_ENGLISH_MARKERS_RE.search(value):
        return "old_english"
    if _ARABIC_RE.search(value):
        return "arabic"
    if _HEBREW_RE.search(value):
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


def _quality_language(value: str | None) -> str | None:
    """Normalize a detected language for compute_quality_report.

    Returns None for unknown/absent values so the scorer falls back to its
    no-language weighting instead of scoring against an absent profile.
    """
    normalized = _normalize_detected_language(value)
    return None if normalized == "unknown" else normalized
