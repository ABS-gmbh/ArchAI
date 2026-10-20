"""Entity mentions - people, places, works, dates, titles - found in a page's text.

Anchor and trigger rules, canonical names, a salvage pass for noisy OCR and
name propagation, all on diplomatic text (combining marks and MUFI
abbreviation letters kept). Moved out of ``app/routers/ocr.py``;
``scripts/benchmark_entities.py`` scores it.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

from app.services.medieval_text import build_search_key


# Letters, the combining marks diplomatic transcriptions write over them (ẽ, the
# superscript a of "mͣrcadigas") and the MUFI abbreviation letters (ꝑ, ꝯ, ꝰ).
# Without the last two, a word was cut at its first abbreviation: "mͣrcadigas"
# was read as "m" and "rcadigas", "Caldꝰ" as "Cald".
_WORD_CHARS = "A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0100-\u024f\u1e00-\u1eff\ua720-\ua7ff\uab30-\uab6f"


_MARK_CHARS = "\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f"


_MENTION_TOKEN_RE = re.compile(rf"[{_WORD_CHARS}][{_WORD_CHARS}{_MARK_CHARS}'’\-]*")


def _word_key(token: str) -> str:
    """A word as the search key spells it: abbreviations expanded, u/v and i/j merged.

    Every lexicon the mention rules consult is compared on these keys, so the
    diplomatic "uilanie" meets the listed "vilanie" and "Caldꝰ" is "caldus".
    """
    return build_search_key(token)


def _keyed(words: Any) -> frozenset[str]:
    return frozenset(_word_key(word) for word in words)


_PERSON_PARTICLES = {"de", "d'", "d’", "du", "des"}


_PLACE_ANCHORS = {"lausanne", "paris", "lyon", "rome", "avignon", "geneve", "genève"}


_WEAK_PLACE_LEXICAL_BLACKLIST = {
    "enfant",
    "enfans",
    "enfants",
    "vilanie",
    "vilenie",
    "villanie",
    "amor",
    "amour",
    "courtoisie",
    "felonie",
    "honte",
    "honor",
    "honneur",
    "merci",
    "paor",
    "peor",
    "peur",
    "proece",
    "proesce",
    "chevalerie",
}


_TITLE_ANCHORS = {"euesque", "évesque", "evesque", "prince", "chapellain", "pere"}


_PLACE_ANCHOR_KEYS = _keyed(_PLACE_ANCHORS)


_WEAK_PLACE_LEXICAL_KEYS = _keyed(_WEAK_PLACE_LEXICAL_BLACKLIST)


_DATE_PATTERNS = (
    re.compile(r"\b(?:l['’]an|lan|an de grace)\b(?:\s+[A-Za-z0-9IVXLCDM]+){0,4}", re.IGNORECASE),
    re.compile(r"\b(?:mil|mille)\b(?:\s+[A-Za-z0-9IVXLCDM]+){0,4}", re.IGNORECASE),
    re.compile(r"\b1[0-9]{3}\b"),
    re.compile(r"\bM{1,4}[CDLXVI]{2,}\b"),
)


# ── Date anchor validation ────────────────────────────────────────────
# A span may be classified as DATE only if it contains at least one
# strong anchor.  "mil" or "cent" alone without anchors are rejected.
_DATE_STRONG_ANCHORS: set[str] = {
    "an", "l'an", "l’an", "lan", "en l'an", "en l’an",
    "ans", "apres", "après", "avant",
}


_MONTH_NAMES: set[str] = {
    "janvier", "fevrier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "aout", "septembre", "octobre", "novembre",
    "decembre", "décembre",
    "januarii", "februarii", "martii", "aprilis", "maii", "junii",
    "julii", "augusti", "septembris", "octobris", "novembris", "decembris",
}


_DATE_FORMULA_RE = re.compile(
    r"\b(?:l['’']an|lan|an\s+de\s+grace)\s+(?:mil|mille)\b",
    re.IGNORECASE,
)


_NUMERIC_YEAR_RE = re.compile(r"\b[0-9]{3,4}\b")


_ROMAN_YEAR_RE = re.compile(r"\bM{1,4}[CDLXVI]{2,}\b")


# ── Editorial / philology blacklist (non-entity markers) ──────────────
_EDITORIAL_BLACKLIST: set[str] = {
    "lacune", "lacuna", "lacunae",
    "[...]", "…", "illegible", "gap", "missing",
    "illisible", "effacé", "rature",
    "trou", "déchirure", "dechirure",
}


def _has_date_anchor(span: str) -> bool:
    """Return True if *span* contains a strong date anchor.

    Strong anchors:
      - One of _DATE_STRONG_ANCHORS tokens
      - A numeric year pattern (3-4 digits)
      - A roman-numeral year pattern (e.g. MCLXIIII)
      - A month name
      - A full date formula like "l'an mil ..."
    """
    low = span.lower()
    # Check full date formula ("l'an mil ...") first
    if _DATE_FORMULA_RE.search(low):
        return True
    # Numeric year
    if _NUMERIC_YEAR_RE.search(span):
        return True
    # Roman-numeral year
    if _ROMAN_YEAR_RE.search(span):
        return True
    # Tokenize and check anchors / month names
    tokens = re.findall(r"[A-Za-z\u00c0-\u00ff'’]+", low)
    for tok in tokens:
        if tok in _DATE_STRONG_ANCHORS:
            return True
        if tok in _MONTH_NAMES:
            return True
    # Check bigrams for multi-word anchors ("en l'an", "an de grace")
    text_joined = " ".join(tokens)
    for anchor in _DATE_STRONG_ANCHORS:
        if " " in anchor and anchor in text_joined:
            return True
    return False


def _norm_surface(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _is_roman_numeral_token(token: str) -> bool:
    value = str(token or "").upper()
    return bool(value) and re.fullmatch(r"[IVXLCDM]+", value) is not None


def _extract_person_mentions(base_text: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    lines = base_text.splitlines()
    cursor = 0
    for line in lines:
        token_matches = list(_MENTION_TOKEN_RE.finditer(line))
        i = 0
        while i < len(token_matches):
            first = token_matches[i]
            token = first.group(0)
            if not token[:1].isupper() or _is_roman_numeral_token(token):
                i += 1
                continue
            best_end_idx = -1
            best_non_particle = 0
            used_particle = False
            for j in range(i + 1, min(len(token_matches), i + 7)):
                prev_end = token_matches[j - 1].end()
                cur_start = token_matches[j].start()
                between = line[prev_end:cur_start]
                if not re.fullmatch(r"[\s,.;:()\-–—]*", between):
                    break
                token_j = token_matches[j].group(0)
                token_j_norm = token_j.lower()
                if token_j_norm in _PERSON_PARTICLES:
                    used_particle = True
                    continue
                if not token_j[:1].isupper() or _is_roman_numeral_token(token_j):
                    break
                non_particle_tokens = 0
                for k in range(i, j + 1):
                    tok_k = token_matches[k].group(0).lower()
                    if tok_k not in _PERSON_PARTICLES:
                        non_particle_tokens += 1
                if 2 <= non_particle_tokens <= 5:
                    best_end_idx = j
                    best_non_particle = non_particle_tokens
            if best_end_idx == -1:
                i += 1
                continue
            local_start = token_matches[i].start()
            local_end = token_matches[best_end_idx].end()
            start = cursor + local_start
            end = cursor + local_end
            surface = base_text[start:end]
            conf = 0.65 + min(0.15, 0.05 * max(0, best_non_particle - 2))
            if used_particle:
                conf = min(0.85, conf + 0.05)
            mentions.append(
                {
                    "start_offset": start,
                    "end_offset": end,
                    "surface": surface,
                    "norm": _norm_surface(surface),
                    "label": None,
                    "ent_type": "person",
                    "confidence": conf,
                    "method": "rule:proper_name_sequence",
                    "notes": None,
                }
            )
            i = best_end_idx + 1
        cursor += len(line) + 1
    return mentions


def _extract_anchor_mentions(base_text: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    lowered = base_text.lower()

    for place in sorted(_PLACE_ANCHORS):
        pattern = re.compile(rf"\b{re.escape(place)}\b", re.IGNORECASE)
        for match in pattern.finditer(lowered):
            start = match.start()
            end = match.end()
            surface = base_text[start:end]
            mentions.append(
                {
                    "start_offset": start,
                    "end_offset": end,
                    "surface": surface,
                    "norm": _norm_surface(surface),
                    "label": None,
                    "ent_type": "place",
                    "confidence": 0.84,
                    "method": "rule:toponym_anchor",
                    "notes": None,
                }
            )

    for title in sorted(_TITLE_ANCHORS):
        pattern = re.compile(rf"\b{re.escape(title)}\b", re.IGNORECASE)
        for match in pattern.finditer(lowered):
            start = match.start()
            end = match.end()
            surface = base_text[start:end]
            mentions.append(
                {
                    "start_offset": start,
                    "end_offset": end,
                    "surface": surface,
                    "norm": _norm_surface(surface),
                    "label": None,
                    "ent_type": "title",
                    "confidence": 0.64,
                    "method": "rule:title_anchor",
                    "notes": None,
                }
            )

    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(base_text):
            start = match.start()
            end = match.end()
            surface = base_text[start:end].strip()
            if not surface:
                continue
            # ── Req 1: Validate date anchor presence ──────────────
            if not _has_date_anchor(surface):
                continue
            # Reject spans that cross a newline unless an anchor
            # appears on BOTH sides or the full phrase matches a
            # known date formula.
            if "\n" in base_text[start:end]:
                parts = base_text[start:end].split("\n")
                if not _DATE_FORMULA_RE.search(base_text[start:end].lower()):
                    left_ok = _has_date_anchor(parts[0])
                    right_ok = _has_date_anchor(parts[-1])
                    if not (left_ok and right_ok):
                        continue
            mentions.append(
                {
                    "start_offset": start,
                    "end_offset": end,
                    "surface": base_text[start:end],
                    "norm": _norm_surface(surface),
                    "label": None,
                    "ent_type": "date",
                    "confidence": 0.72,
                    "method": "rule:date_pattern",
                    "notes": None,
                }
            )

    first_non_empty = next((line.strip() for line in base_text.splitlines() if line.strip()), "")
    if first_non_empty and len(first_non_empty) <= 80:
        lowered_line = first_non_empty.lower()
        if lowered_line.startswith(("la ", "le ", "l'", "l’")) and len(first_non_empty.split()) >= 2:
            start = base_text.find(first_non_empty)
            if start >= 0:
                end = start + len(first_non_empty)
                mentions.append(
                    {
                        "start_offset": start,
                        "end_offset": end,
                        "surface": base_text[start:end],
                        "norm": _norm_surface(first_non_empty),
                        "label": None,
                        "ent_type": "work",
                        "confidence": 0.56,
                        "method": "rule:first_line_title_heuristic",
                        "notes": None,
                    }
                )
    return mentions


def _dedupe_mentions(base_text: str, mentions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    for item in mentions:
        start = int(item.get("start_offset", 0))
        end = int(item.get("end_offset", 0))
        ent_type = str(item.get("ent_type") or "unknown")
        if start < 0 or end <= start or end > len(base_text):
            continue
        key = (start, end, ent_type)
        prev = by_key.get(key)
        if prev is None or float(item.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
            by_key[key] = item

    ordered = sorted(by_key.values(), key=lambda row: (-float(row.get("confidence", 0.0)), -(int(row["end_offset"]) - int(row["start_offset"]))))
    selected: list[dict[str, Any]] = []
    for item in ordered:
        start = int(item["start_offset"])
        end = int(item["end_offset"])
        ent_type = str(item.get("ent_type") or "unknown")
        overlap = False
        for existing in selected:
            if str(existing.get("ent_type") or "unknown") != ent_type:
                continue
            ex_start = int(existing["start_offset"])
            ex_end = int(existing["end_offset"])
            if start < ex_end and end > ex_start:
                overlap = True
                break
        if overlap:
            continue
        selected.append(item)

    selected.sort(key=lambda row: (int(row["start_offset"]), -float(row.get("confidence", 0.0))))
    return selected


# ── Salvage extractor constants ─────────────────────────────────────────
_SALVAGE_TRIGGERS: dict[str, str] = {
    "roi": "PERSON_OR_ROLE", "roy": "PERSON_OR_ROLE", "roya": "PERSON_OR_ROLE",
    # Old French nominative singular: "li rois Artus", "Caldus li roys".
    "rois": "PERSON_OR_ROLE", "roys": "PERSON_OR_ROLE",
    # Latin: "euangelistae iohannis", "sancti benedicti", "rex karolus".
    "sanctus": "PERSON", "sancti": "PERSON", "sancta": "PERSON", "sancte": "PERSON",
    "beatus": "PERSON", "beati": "PERSON", "beata": "PERSON", "beate": "PERSON",
    "apostolus": "PERSON", "apostoli": "PERSON", "euangelista": "PERSON", "euangelistae": "PERSON",
    "rex": "PERSON_OR_ROLE", "regis": "PERSON_OR_ROLE", "episcopus": "PERSON_OR_ROLE", "episcopi": "PERSON_OR_ROLE",
    "evesque": "PERSON_OR_ROLE", "évesque": "PERSON_OR_ROLE", "euesque": "PERSON_OR_ROLE",
    "saint": "PERSON", "sainz": "PERSON",
    "pape": "PERSON_OR_ROLE", "duc": "PERSON_OR_ROLE",
    "conte": "PERSON_OR_ROLE", "dame": "PERSON_OR_ROLE", "sire": "PERSON_OR_ROLE",
}


_CANONICAL_WORKS: dict[str, str] = {
    "lancelot": "lancelot", "arthur": "arthur", "guenievre": "guenievre",
    "tristan": "tristan", "perceval": "perceval", "graal": "graal",
    "merlin": "merlin", "galahad": "galahad",
}


# Canonical person names used by the name-heuristic check in salvage Rule 1.
# Keys are normalised forms; values are the preferred canonical spelling.
_CANONICAL_PERSON_NAMES: dict[str, str] = {
    "arthur": "arthur", "artus": "arthur", "artur": "arthur",
    "lancelot": "lancelot", "lancelo": "lancelot",
    "lancelote": "lancelot", "leantlote": "lancelot",
    "lanselot": "lancelot", "lanceloc": "lancelot",
    "guenievre": "guenievre", "guenievr": "guenievre", "guenevere": "guenievre",
    "perceval": "perceval", "parsival": "perceval", "percevall": "perceval",
    "merlin": "merlin", "merlim": "merlin",
    "galahad": "galahad", "galaad": "galahad",
    "tristan": "tristan", "tristam": "tristan",
    "gauvain": "gauvain", "gawain": "gauvain",
    "yvain": "yvain", "ywain": "yvain", "ivain": "yvain",
    "erec": "erec", "enide": "enide",
    "bohort": "bohort", "bors": "bohort",
}


_SALVAGE_STOPWORDS = {
    "et", "de", "la", "le", "les", "des", "du", "en", "a", "au", "aux", "que",
    "qui", "ne", "pas", "par", "pour", "sur", "son", "ses", "sa", "ce",
    "est", "fut", "sont", "une", "uns", "si", "il", "li", "lor", "mais",
    "car", "con", "com", "tout", "molt", "bien", "fait", "dit",
    "del", "d'", "d\u2019", "mes", "ot", "ont", "vint", "fist", "dist",
    "estoit", "avoit", "ala", "alerent", "vont", "font", "firent",
    "puis", "lors", "tant", "moult", "aussi", "donc", "dont",
}


_SALVAGE_TRIGGER_KEYS: dict[str, str] = {_word_key(word): label for word, label in _SALVAGE_TRIGGERS.items()}


_SALVAGE_STOPWORD_KEYS = _keyed(_SALVAGE_STOPWORDS)


_APPOSITION_ARTICLE_KEYS = _keyed({"li", "le", "lo"})


_SALVAGE_TOKEN_RE = re.compile(rf"[{_WORD_CHARS}][{_WORD_CHARS}{_MARK_CHARS}]*")


# Only prepositions that reliably signal a place name in medieval French.
# Deliberately excludes "a" (too common as a verb/article) to avoid noise.
_PLACE_PREP_RE = re.compile(
    r"\b(?:en|au|aux|de\s+la|de\s+l['\u2019]|de|du|des|del|d['\u2019])\s+",
    re.IGNORECASE,
)


_VOWELS = set("aeiouyàâäéèêëïîôùûüœæ")


def _looks_like_medieval_name(token: str) -> bool:
    """Heuristic: does *token* look like a plausible medieval proper name?

    Used by salvage Rule 1 to decide whether the tokens following a role
    trigger (roi, dame, …) constitute a person name or are OCR garbage.

    Checks:
      0. Reject if token is in the common-word blacklist.
      1. Length-aware canonical matching: normalized edit distance <= 0.25
         AND bigram overlap >= 0.40 for tokens >= 5 chars.
      2. Length 3–18 characters.
      3. Vowel ratio between 0.20 and 0.80.
      4. No run of ≥ 4 consecutive consonants (unusual in French names).

    Callers pass the search key (:func:`_word_key`), so abbreviation marks do
    not count as consonants.

    Returns ``True`` if the token passes all applicable checks.
    """
    from app.services.text_normalization import (
        is_blacklisted_token,
        normalized_edit_distance,
        bigram_overlap,
    )
    if len(token) < 3 or len(token) > 18:
        return False
    low = token.lower()

    # Gate 0: reject common words — no common word is ever a name
    if is_blacklisted_token(low):
        return False

    # Gate 1: canonical match using normalized distance + bigram overlap
    for canon in _CANONICAL_PERSON_NAMES:
        nd = normalized_edit_distance(low, canon)
        # Short tokens (< 5 chars): require exact or 1-char-off match
        if len(low) < 5:
            if nd == 0.0:  # exact match only for very short tokens
                return True
            continue
        # Tokens >= 5 chars: normalized distance + bigram overlap
        bo = bigram_overlap(low, canon)
        if nd <= 0.25 and bo >= 0.40:
            return True

    # Vowel ratio
    vowels = sum(1 for c in low if c in _VOWELS)
    ratio = vowels / len(low) if low else 0
    if ratio < 0.20 or ratio > 0.80:
        return False

    # No long consonant runs (≥ 4 consecutive non-vowels)
    consonant_run = 0
    for c in low:
        if c in _VOWELS:
            consonant_run = 0
        else:
            consonant_run += 1
            if consonant_run >= 4:
                return False

    # No closeness to the canonical names is required: they are the
    # Arthurian cast, and requiring it made every other name - Marcadigas,
    # Iohannes - a role. The trigger before the token is the evidence.
    return True


def _is_canonical_person_name(key: str) -> bool:
    from app.services.text_normalization import bigram_overlap, normalized_edit_distance

    for canon in _CANONICAL_PERSON_NAMES:
        if len(key) < 5:
            if key == canon:
                return True
        elif normalized_edit_distance(key, canon) <= 0.25 and bigram_overlap(key, canon) >= 0.40:
            return True
    return False


def _is_trigger_name(token: str, key_counts: Counter[str], *, saint: bool = False) -> bool:
    """Is the word beside a title ("li rois", "beati") the name of a person?

    A name from the canonical list is. Any other must look like a word and
    carry a second sign of being a name: a capital, or a second occurrence on
    the page. Misread words rarely recur exactly, and on 72 transcriptions held
    out from the pipeline database the title rule otherwise took "qu'uncz",
    "enguit" and "nirendignal" for kings. After a saint's or an apostle's title
    (*saint*, *beatus*, *euangelista*) the word that looks like one is a name:
    "euangelistae iohannis".
    """
    key = _word_key(token)
    if _is_canonical_person_name(key):
        return True
    if not _looks_like_medieval_name(key):
        return False
    return saint or token[:1].isupper() or key_counts[key] >= 2


def _extract_salvage_mentions(base_text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Salvage extractor for medieval lowercase text.

    Returns ``(mentions, debug)`` where *debug* contains counts per rule
    and rejected candidates.
    """
    from app.services.text_normalization import (
        is_blacklisted_token,
        normalized_edit_distance as _ned,
        bigram_overlap as _bo,
    )
    value = str(base_text or "")
    if not value.strip():
        return [], {"trigger": 0, "work_fuzzy": 0, "place_candidate": 0, "rejected": []}

    tokens_matches = list(_SALVAGE_TOKEN_RE.finditer(value))
    tokens_lower = [(m.group(0).lower(), m.start(), m.end()) for m in tokens_matches]
    keys = [_word_key(m.group(0)) for m in tokens_matches]
    key_counts = Counter(keys)

    mentions: list[dict[str, Any]] = []
    debug: dict[str, Any] = {
        "trigger": 0,
        "work_fuzzy": 0,
        "place_candidate": 0,
        "rejected": [],
    }
    seen_spans: set[tuple[int, int]] = set()

    # ── Rule 1: Trigger words ─────────────────────────────────────────
    for idx, (tok_low, tok_start, tok_end) in enumerate(tokens_lower):
        label = _SALVAGE_TRIGGER_KEYS.get(keys[idx])
        if label is None:
            continue
        # Old French also puts the name first: "Caldus li roys".
        if idx >= 2 and keys[idx - 1] in _APPOSITION_ARTICLE_KEYS:
            name_start, name_end_back = tokens_lower[idx - 2][1], tokens_lower[idx - 2][2]
            name_token = value[name_start:name_end_back]
            gap = value[name_end_back:tokens_lower[idx - 1][1]]
            if (
                not any(ch in gap for ch in ".;:!?")
                and keys[idx - 2] not in _SALVAGE_STOPWORD_KEYS
                and keys[idx - 2] not in _SALVAGE_TRIGGER_KEYS
                and not is_blacklisted_token(name_token)
                and _is_trigger_name(name_token, key_counts)
                and (name_start, tok_end) not in seen_spans
            ):
                seen_spans.add((name_start, tok_end))
                surface = value[name_start:tok_end]
                mentions.append({
                    "start_offset": name_start,
                    "end_offset": tok_end,
                    "surface": surface,
                    "norm": _norm_surface(surface),
                    "label": label,
                    "ent_type": "person",
                    "confidence": 0.55,
                    "method": "rule:salvage_apposition",
                    "notes": f"name before article+trigger ({tok_low})",
                    "name_keys": [keys[idx - 2]],
                })
                debug["trigger"] += 1
        # Capture next 1–4 non-stopword tokens as the candidate name
        name_parts: list[str] = []
        name_end = tok_end
        for j in range(1, 5):
            nxt = idx + j
            if nxt >= len(tokens_lower):
                break
            nxt_tok, nxt_start, nxt_end = tokens_lower[nxt]
            # Stop on another trigger or punctuation gap > 3 chars
            gap = value[name_end:nxt_start]
            # The Tironian et (⁊) and & are "et", which ends a name as the word
            # does; so does a line end - in verse the next line opens with a
            # capital whether or not it is a name.
            if len(gap) > 3 or any(ch in gap for ch in '.;:!?⁊&\n'):
                break
            if keys[nxt] in _SALVAGE_STOPWORD_KEYS or len(nxt_tok) < 2:
                break
            if keys[nxt] in _SALVAGE_TRIGGER_KEYS:
                break
            name_parts.append(value[nxt_start:nxt_end])
            name_end = nxt_end
        if not name_parts:
            continue
        # Filter out blacklisted common words from name_parts
        non_blacklisted = [p for p in name_parts if not is_blacklisted_token(p)]
        if not non_blacklisted:
            debug["rejected"].append({
                "surface": " ".join(name_parts),
                "start_offset": tok_start,
                "end_offset": name_end,
                "canonical": "(blacklisted)",
                "dist": 0,
                "max_dist": 0,
                "reason": f"all name_parts blacklisted: {name_parts}",
            })
            continue
        surf_start = tok_start
        surf_end = name_end
        if (surf_start, surf_end) in seen_spans:
            continue
        seen_spans.add((surf_start, surf_end))
        surface = value[surf_start:surf_end]

        # ── Name-quality heuristic: decide person vs role ─────────
        # If *any* non-blacklisted name_part looks like a real medieval
        # name → person.  Otherwise → role (not sent for person linking).
        any_name_like = _is_trigger_name(non_blacklisted[0], key_counts, saint=label == "PERSON")
        ent_type = "person" if any_name_like else "role"
        confidence = 0.55 if any_name_like else 0.25

        if not any_name_like:
            # SKIP_NON_LINKABLE: goes into skipped (not rejected) because
            # it IS emitted as a role mention — it's not a failed candidate.
            debug.setdefault("skipped", []).append({
                "surface": surface,
                "canonical": "(person heuristic)",
                "dist": 0,
                "max_dist": 0,
                "reason": f"name_parts={name_parts} failed medieval-name heuristic → role (SKIP_NON_LINKABLE)",
            })

        mentions.append({
            "start_offset": surf_start,
            "end_offset": surf_end,
            "surface": surface,
            "norm": _norm_surface(surface),
            "label": label,
            "ent_type": ent_type,
            "confidence": confidence,
            "method": "rule:salvage_trigger",
            "notes": f"trigger={tok_low} name_quality={'name' if any_name_like else 'role'}",
            "name_keys": [_word_key(non_blacklisted[0])] if any_name_like else [],
        })
        debug["trigger"] += 1

    # ── Rule 2: Fuzzy work/legend matching ────────────────────────────
    for tok_low, tok_start, tok_end in tokens_lower:
        if len(tok_low) < 4:
            continue
        if is_blacklisted_token(tok_low):
            continue
        for canon, canon_norm in _CANONICAL_WORKS.items():
            # Use normalized edit distance + bigram overlap
            nd = _ned(tok_low, canon)
            # Short tokens (< 5 chars): exact match only
            if len(tok_low) < 5:
                if nd > 0.0:
                    continue
            else:
                bo = _bo(tok_low, canon)
                if nd > 0.25 or bo < 0.40:
                    # Track close misses for debug
                    if nd <= 0.40:
                        debug["rejected"].append({
                            "surface": value[tok_start:tok_end],
                            "start_offset": tok_start,
                            "end_offset": tok_end,
                            "canonical": canon,
                            "dist": round(nd, 3),
                            "max_dist": 0.25,
                            "reason": f"nd={nd:.3f}>0.25 or bo={bo:.3f}<0.40",
                        })
                    continue
            if (tok_start, tok_end) in seen_spans:
                break
            seen_spans.add((tok_start, tok_end))
            surface = value[tok_start:tok_end]
            # Compute bo for provenance (may already have it from the gate above)
            if len(tok_low) >= 5:
                bo_val = _bo(tok_low, canon)
            else:
                bo_val = 1.0  # exact match for short tokens
            mentions.append({
                "start_offset": tok_start,
                "end_offset": tok_end,
                "surface": surface,
                "norm": canon_norm,
                "label": "WORK",
                "ent_type": "work",
                "confidence": max(0.35, 0.65 - nd),
                "method": "rule:salvage_work_fuzzy",
                "notes": f"canonical={canon} nd={nd:.3f} bo={bo_val:.3f}",
            })
            debug["work_fuzzy"] += 1
            break  # matched one canonical – stop checking others

    # ── Rule 3: Place candidate (preposition + token) ─────────────────
    # Uses place_likeness gate: token must have plausible phonotactics
    # AND meet stricter length/case requirements to avoid false positives
    # from random short OCR fragments like "ament".
    from app.services.text_normalization import token_quality_score as _tqs
    for m in _PLACE_PREP_RE.finditer(value):
        prep_end = m.end()
        # Get the next token after the preposition
        next_tok_m = _SALVAGE_TOKEN_RE.search(value, prep_end)
        if next_tok_m is None or next_tok_m.start() > prep_end + 2:
            continue
        tok_text = next_tok_m.group(0)
        tok_low_val = tok_text.lower()
        tok_key = _word_key(tok_text)
        # ── Req 2: editorial/philology blacklist ─────────────────
        if tok_low_val in _EDITORIAL_BLACKLIST:
            debug["rejected"].append({
                "surface": tok_text,
                "start_offset": next_tok_m.start(),
                "end_offset": next_tok_m.end(),
                "canonical": "(place)",
                "dist": 0,
                "max_dist": 0,
                "reason": f"editorial_blacklist: '{tok_low_val}' is not an entity",
            })
            continue
        if tok_key in _WEAK_PLACE_LEXICAL_KEYS:
            debug["rejected"].append({
                "surface": tok_text,
                "start_offset": next_tok_m.start(),
                "end_offset": next_tok_m.end(),
                "canonical": "(place)",
                "dist": 0,
                "max_dist": 0,
                "reason": f"weak_place_lexical_blacklist: '{tok_low_val}' is lexical, not a place candidate",
            })
            continue
        # Require length >= 5, alpha-only after norm, and not a stopword
        norm_tok = _norm_surface(tok_text)
        if len(tok_text) < 5 or tok_key in _SALVAGE_STOPWORD_KEYS:
            debug["rejected"].append({
                "surface": tok_text,
                "start_offset": next_tok_m.start(),
                "end_offset": next_tok_m.end(),
                "canonical": "(place)",
                "dist": 0,
                "max_dist": 0,
                "reason": f"too short ({len(tok_text)}<5) or stopword",
            })
            continue
        if not norm_tok.replace(" ", "").isalpha():
            debug["rejected"].append({
                "surface": tok_text,
                "start_offset": next_tok_m.start(),
                "end_offset": next_tok_m.end(),
                "canonical": "(place)",
                "dist": 0,
                "max_dist": 0,
                "reason": "non-alpha after normalisation",
            })
            continue
        # ── place_likeness gate ──────────────────────────────────────
        # A place candidate must satisfy EITHER:
        # (a) token starts with uppercase → likely intentional name, OR
        # (b) token is a known place anchor, OR
        # (c) token length >= 6 AND good phonotactic quality
        # This prevents short lowercase OCR garbage ("ament") from
        # slipping through.
        is_known_place = tok_key in _PLACE_ANCHOR_KEYS
        is_capitalized = tok_text[:1].isupper()
        place_quality = _tqs(tok_key)

        if not is_known_place and not is_capitalized:
            # A lowercase word after "de" or "en" is nearly always a common
            # noun: on the five reference pages this branch produced five
            # places - oeures, lordena, uilanie, lignage, vilain - and no
            # real one. Lowercase places are found through the gazetteer.
            debug["rejected"].append({
                "surface": tok_text,
                "start_offset": next_tok_m.start(),
                "end_offset": next_tok_m.end(),
                "canonical": "(place)",
                "dist": 0,
                "max_dist": 0,
                "reason": "place_likeness: lowercase and not a known place",
            })
            continue
        elif not is_known_place:
            # Capitalized but not a known place: basic quality check
            if place_quality < 0.40:
                debug["rejected"].append({
                    "surface": tok_text,
                    "start_offset": next_tok_m.start(),
                    "end_offset": next_tok_m.end(),
                    "canonical": "(place)",
                    "dist": 0,
                    "max_dist": 0,
                    "reason": f"place_likeness={place_quality:.3f} < 0.40 (OCR garbage)",
                })
                continue
        span = (next_tok_m.start(), next_tok_m.end())
        if span in seen_spans:
            continue
        seen_spans.add(span)
        mentions.append({
            "start_offset": next_tok_m.start(),
            "end_offset": next_tok_m.end(),
            "surface": tok_text,
            "norm": norm_tok,
            "label": "PLACE_CANDIDATE",
            "ent_type": "place",
            "confidence": 0.35,
            "method": "rule:salvage_place_candidate",
            "notes": f"prep='{m.group(0).strip()}' place_quality={place_quality:.3f} is_cap={is_capitalized} is_anchor={is_known_place}",
        })
        debug["place_candidate"] += 1

    # Trim rejected list
    debug["rejected"] = debug["rejected"][:20]
    return mentions, debug


def _propagate_names(base_text: str, mentions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every other occurrence of a name a rule found, as a person mention.

    The rules need context - a title before the name, "li rois" after it -
    that most occurrences lack: one verse page names Marcadigas four times, once
    after "Roys". Only names a rule accepted as a person's are propagated, to
    tokens with the same search key, so "Marcadigas", "marcadigas" and
    "mͣrcadigas" are one name.
    """
    names = {
        key
        for mention in mentions
        if mention.get("ent_type") == "person"
        for key in mention.get("name_keys") or []
        if len(key) >= 4
    }
    if not names:
        return []
    covered = [
        (int(mention["start_offset"]), int(mention["end_offset"]))
        for mention in mentions
        if mention.get("ent_type") == "person"
    ]
    out: list[dict[str, Any]] = []
    for match in _SALVAGE_TOKEN_RE.finditer(base_text):
        start, end = match.start(), match.end()
        if _word_key(match.group(0)) not in names:
            continue
        if any(c_start <= start and end <= c_end for c_start, c_end in covered):
            continue
        surface = match.group(0)
        out.append(
            {
                "start_offset": start,
                "end_offset": end,
                "surface": surface,
                "norm": _norm_surface(surface),
                "label": "PERSON",
                "ent_type": "person",
                "confidence": 0.5,
                "method": "rule:name_propagation",
                "notes": "same name as a person mention elsewhere on the page",
            }
        )
    return out


def _extract_mentions_from_text(
    base_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Extract mentions using quality-aware pipeline.

    Strategy:
      1. Salvage extractor (trigger/work/place rules — works on ALL text)
      2. Person mentions from proper-name sequences (capitalization heuristic)
      3. Anchor mentions (place/title/date — case-insensitive)
      4. N-gram canonical scan (case-insensitive, catches matches that
         the capitalization-based extractor misses in lowercase OCR)

    Returns ``(mentions, heuristic_candidates, salvage_debug)``.
    """
    value = str(base_text or "")
    if not value.strip():
        return [], [], {"trigger": 0, "work_fuzzy": 0, "place_candidate": 0, "rejected": []}

    salvage_mentions, salvage_debug = _extract_salvage_mentions(value)
    candidates = [
        *_extract_person_mentions(value),
        *_extract_anchor_mentions(value),
        *salvage_mentions,
    ]

    # ── N-gram canonical scan (case-insensitive fallback) ─────────────
    # Catches canonical person/work names in fully-lowercase OCR output
    # that the capitalization-dependent _extract_person_mentions misses.
    ngram_mentions = _extract_ngram_canonical_mentions(value)
    candidates.extend(ngram_mentions)
    candidates.extend(_propagate_names(value, candidates))
    for candidate in candidates:
        candidate.pop("name_keys", None)

    mentions = _dedupe_mentions(value, candidates)

    # ── Post-dedup reconciliation: single decision per span ────────
    # A surface that was rejected by one strategy (e.g. salvage Rule 2
    # fuzzy match against _CANONICAL_WORKS) may have been accepted by
    # another (e.g. ngram_canonical against _CANONICAL_PERSON_NAMES).
    # After dedup selects the winners, purge stale rejections so the
    # report cannot list the same surface as both accepted AND rejected.
    #
    # Linkable types (ACCEPT) vs non-linkable types (SKIP_NON_LINKABLE):
    _LINKABLE_TYPES = {"person", "place", "work", "date"}

    accepted_surfaces_lower: set[str] = {
        str(m.get("surface", "")).lower() for m in mentions
    }
    truly_rejected: list[dict[str, Any]] = []
    attempts_for_accepted: list[dict[str, Any]] = []
    for r in salvage_debug.get("rejected", []):
        surface_low = str(r.get("surface", "")).lower()
        if surface_low in accepted_surfaces_lower:
            attempts_for_accepted.append(r)
        else:
            truly_rejected.append(r)
    salvage_debug["rejected"] = truly_rejected
    salvage_debug["attempts_for_accepted"] = attempts_for_accepted

    # Partition accepted mentions into linkable (ACCEPT) and non-linkable
    # (SKIP_NON_LINKABLE, e.g. role/title with no entity to link).
    skipped_non_linkable: list[dict[str, Any]] = []
    for m in mentions:
        ent_type = str(m.get("ent_type") or "unknown")
        if ent_type not in _LINKABLE_TYPES:
            skipped_non_linkable.append(m)
    salvage_debug["skipped_non_linkable"] = (
        salvage_debug.get("skipped_non_linkable", []) + skipped_non_linkable
    )
    # Also migrate any "skipped" entries from Rule 1 into the list
    for s in salvage_debug.pop("skipped", []):
        salvage_debug["skipped_non_linkable"].append(s)

    # ── Assertion: accepted ∩ rejected = ∅ ────────────────────────
    rejected_surfaces_lower: set[str] = {
        str(r.get("surface", "")).lower() for r in salvage_debug["rejected"]
    }
    overlap = accepted_surfaces_lower & rejected_surfaces_lower
    assert not overlap, (
        f"BUG: surfaces in BOTH accepted and rejected: {overlap}"
    )

    heuristic_candidates: list[dict[str, Any]] = []
    for mention in mentions:
        ent_type = str(mention.get("ent_type") or "unknown")
        if ent_type not in {"person", "place", "work"}:
            continue
        heuristic_candidates.append(
            {
                "mention_id": mention.get("mention_id"),
                "start_offset": int(mention.get("start_offset", 0)),
                "end_offset": int(mention.get("end_offset", 0)),
                "ent_type": ent_type,
                "surface": str(mention.get("surface") or ""),
                "source": "heuristic",
                "candidate": str(mention.get("surface") or "").strip(),
                "score": float(mention.get("confidence", 0.0)),
                "meta_json": {"ent_type": ent_type, "method": mention.get("method")},
            }
        )
    return mentions, heuristic_candidates, salvage_debug


def _extract_ngram_canonical_mentions(base_text: str) -> list[dict[str, Any]]:
    """Case-insensitive scan for canonical entity names via fuzzy n-gram matching.

    This extracts mentions that the capitalization-dependent extractor would
    miss in fully lowercase OCR output.  It uses length-aware, normalized
    edit distance + bigram overlap to avoid false positives like
    "main" → Yvain or "port" → Bohort.

    Rules:
      - Tokens < 5 chars: NEVER fuzzy-match (too many false positives).
        Only exact canonical matches are accepted.
      - Tokens >= 5 chars: normalized_edit_distance <= 0.25 AND
        bigram_overlap >= 0.40.
      - Tokens in COMMON_WORD_BLACKLIST: always rejected.
    """
    from app.services.text_normalization import (
        is_blacklisted_token,
        normalized_edit_distance,
        bigram_overlap as _bigram_overlap,
    )
    mentions: list[dict[str, Any]] = []
    value = str(base_text or "")
    if not value.strip():
        return mentions

    token_matches = list(_SALVAGE_TOKEN_RE.finditer(value))
    for m in token_matches:
        tok = m.group(0)
        tok_low = tok.lower()
        if len(tok_low) < 4:
            continue
        # Blacklist gate: common words never match
        if is_blacklisted_token(tok_low):
            continue

        # Check against canonical person names
        matched_person = False
        for canon, canon_norm in _CANONICAL_PERSON_NAMES.items():
            nd = normalized_edit_distance(tok_low, canon)
            # Short tokens (< 5 chars): exact match only
            if len(tok_low) < 5:
                if nd == 0.0:
                    mentions.append({
                        "start_offset": m.start(),
                        "end_offset": m.end(),
                        "surface": tok,
                        "norm": canon_norm,
                        "label": None,
                        "ent_type": "person",
                        "confidence": 0.65,
                        "method": "rule:ngram_canonical_person",
                        "notes": f"canonical={canon} nd=0.000 bo=1.000",
                    })
                    matched_person = True
                    break
                continue
            # Tokens >= 5 chars: normalized distance + bigram overlap
            bo = _bigram_overlap(tok_low, canon)
            if nd <= 0.25 and bo >= 0.40:
                conf = max(0.45, 0.70 - nd)
                mentions.append({
                    "start_offset": m.start(),
                    "end_offset": m.end(),
                    "surface": tok,
                    "norm": canon_norm,
                    "label": None,
                    "ent_type": "person",
                    "confidence": round(conf, 3),
                    "method": "rule:ngram_canonical_person",
                    "notes": f"canonical={canon} nd={nd:.3f} bo={bo:.3f}",
                })
                matched_person = True
                break
        if matched_person:
            continue

        # Check against canonical works
        for canon, canon_norm in _CANONICAL_WORKS.items():
            nd = normalized_edit_distance(tok_low, canon)
            if len(tok_low) < 5:
                if nd == 0.0:
                    mentions.append({
                        "start_offset": m.start(),
                        "end_offset": m.end(),
                        "surface": tok,
                        "norm": canon_norm,
                        "label": "WORK",
                        "ent_type": "work",
                        "confidence": 0.60,
                        "method": "rule:ngram_canonical_work",
                        "notes": f"canonical={canon} nd=0.000 bo=1.000",
                    })
                    break
                continue
            bo = _bigram_overlap(tok_low, canon)
            if nd <= 0.25 and bo >= 0.40:
                conf = max(0.40, 0.65 - nd)
                mentions.append({
                    "start_offset": m.start(),
                    "end_offset": m.end(),
                    "surface": tok,
                    "norm": canon_norm,
                    "label": "WORK",
                    "ent_type": "work",
                    "confidence": round(conf, 3),
                    "method": "rule:ngram_canonical_work",
                    "notes": f"canonical={canon} nd={nd:.3f} bo={bo:.3f}",
                })
                break
    return mentions
