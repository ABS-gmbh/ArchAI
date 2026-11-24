"""How authority candidates are judged against a mention.

Name likeness, canonical names, place context, query variants, prefiltering
and the candidate score, with the lexicons they use. Moved out of
``app/services/authority_linking.py``, which searches the sources and
persists the links.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.services.entity_scoring import context_coverage, compute_score, string_similarity
from app.services.text_normalization import normalize_for_search, token_quality_score


log = logging.getLogger("archai.authority_linking")


def _title_texts(titles: Any) -> list[str]:
    """The title strings of a candidate's ``titles``, in order.

    A VIAF cluster gives ``{"work": [{"title": ...}, ...]}`` (a single work is
    a dict, not a list). Read as ``list(titles)``, that dict gave only its key,
    so the context text held the word "work" and never a title.
    """
    if titles is None:
        return []
    if isinstance(titles, str):
        return [titles] if titles.strip() else []
    if isinstance(titles, dict):
        if "title" in titles or "text" in titles:
            return _title_texts(titles.get("title", titles.get("text")))
        return [text for value in titles.values() for text in _title_texts(value)]
    if isinstance(titles, (list, tuple)):
        return [text for item in titles for text in _title_texts(item)]
    return [str(titles)]


def _score_candidate(
    *,
    mention: dict[str, Any],
    candidate: dict[str, Any],
    context: str,
    canonical_norm: str,
    domain_bonus: float,
    type_ok: bool,
    resolved_tokens: set[str],
) -> tuple[float, dict[str, Any]]:
    surface = str(mention.get("surface") or "")
    candidate_label = str(candidate.get("label") or "")
    compare_surface = canonical_norm or surface
    label_similarity = string_similarity(compare_surface, candidate_label)

    aliases_raw = list(candidate.get("aliases", []) or [])
    alias_values = [str(item.get("value") or "").strip() for item in aliases_raw if isinstance(item, dict)]
    alias_values = [value for value in alias_values if value]
    # A candidate without aliases is judged on its label. Scoring its missing
    # alias as a 0.0 match capped it near 0.75, under every auto-select
    # threshold, while a candidate whose only alias repeats its label got the
    # full 0.25: the same evidence scored differently depending on whether
    # Wikidata happens to list an alias.
    alias_match_quality = max(
        (string_similarity(compare_surface, alias) for alias in alias_values), default=label_similarity
    )

    description_parts = [
        str(candidate.get("description") or ""),
        str(candidate.get("canonical_description") or ""),
        str(candidate.get("parent_location") or ""),
        " ".join(_title_texts(candidate.get("titles"))[:5]),
    ]
    description_text = " ".join(part for part in description_parts if part).strip()
    # Reported for auditing. It is already folded into base_score below via
    # compute_score's context term; adding it again here double-counted a signal
    # that measures ~0.014 on real text, while the genuine alias match got 0.12.
    document_context_compatibility = context_coverage(context, description_text)

    source = str(candidate.get("source") or "wikidata").strip() or "wikidata"
    source_confidence = float(candidate.get("source_confidence") or {"wikidata": 1.0, "viaf": 0.92, "geonames": 0.95}.get(source, 0.8))

    mention_label = str(mention.get("label") or "").lower()
    segmentation_label_prior = 0.0
    if source == "geonames" and mention.get("ent_type") == "place":
        segmentation_label_prior = 0.04
    elif source == "viaf" and mention.get("ent_type") in {"person", "work"}:
        segmentation_label_prior = 0.03
    elif mention_label.startswith("person") or mention_label.startswith("place") or mention_label.startswith("work"):
        segmentation_label_prior = 0.02

    cooccurrence_bonus = 0.0
    if resolved_tokens:
        haystack = " ".join(
            [
                candidate_label.lower(),
                description_text.lower(),
                str(candidate.get("country_name") or "").lower(),
                str(candidate.get("admin1_name") or "").lower(),
            ]
        )
        if any(token and token in haystack for token in resolved_tokens):
            cooccurrence_bonus = 0.05

    chronology_bonus = 0.0
    if any(token in description_text.lower() for token in ("medieval", "arthurian", "saint", "abbey", "monastery", "diocese")):
        chronology_bonus = 0.03

    base_score = compute_score(
        surface,
        candidate_label,
        context,
        description_text,
        type_compatible=type_ok,
        canonical_norm=canonical_norm,
        domain_bonus=domain_bonus,
        # The real alias signal, instead of leaving compute_score to substitute a
        # context measurement into the slot named alias_sim.
        alias_sim=alias_match_quality,
    )

    # base_score already carries label, alias, type, context and domain and can
    # reach 1.0. Only source confidence is added proportionally; the priors stay
    # small additive nudges. The previous form could not exceed 0.7936 even for a
    # perfect candidate, which is below every AUTO_SELECT threshold.
    final_score = (
        base_score * 0.88
        + source_confidence * 0.12
        + segmentation_label_prior
        + cooccurrence_bonus
        + chronology_bonus
    )
    final_score = max(0.0, min(1.0, round(final_score, 4)))

    return final_score, {
        "label_similarity": round(label_similarity, 4),
        "alias_match_quality": round(alias_match_quality, 4),
        "document_context_compatibility": round(document_context_compatibility, 4),
        "source_confidence": round(source_confidence, 4),
        "segmentation_label_prior": round(segmentation_label_prior, 4),
        "cooccurrence_bonus": round(cooccurrence_bonus, 4),
        "chronology_bonus": round(chronology_bonus, 4),
        "domain_bonus": round(domain_bonus, 4),
        "base_score": round(base_score, 4),
        "final_score": final_score,
    }


# ── Modern-occupation keywords to reject from medieval-entity linking ──
_MODERN_DESCRIPTION_REJECTS: set[str] = {
    "singer", "footballer", "basketball", "baseball", "hockey",
    "rapper", "actress", "actor", "politician", "president",
    "musician", "composer", "band", "album", "song",
    "software", "programming", "computer", "video game",
    "film", "television", "tv series", "anime",
    "bacterium", "bacteria", "virus", "protein", "enzyme",
    "chemical", "compound", "mineral", "drug",
    "asteroid", "galaxy", "planet", "star system",
    "erectile", "dysfunction",
    "operating system", "linux",
    "port", "seaport", "airport",
    "brand", "company", "corporation",
    "football club", "sports team",
}
# Whole words only, plural allowed: as substrings "band" rejected "husband of
# Guinevere", "actor" "benefactor" and "risk factor", and "port " (its trailing
# space a guard against "important") still matched "to support life".
_MODERN_DESCRIPTION_REJECT_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_MODERN_DESCRIPTION_REJECTS, key=len, reverse=True)) + r")s?\b"
)


# ── Domain keywords that boost medieval entity confidence ──────────────
_MEDIEVAL_DOMAIN_KEYWORDS: set[str] = {
    "arthurian", "round table", "table ronde", "chevalier",
    "knight", "grail", "graal", "camelot", "legendary",
    "chrétien", "chretien", "troyes", "medieval", "médiéval",
    "fictional character", "literary character", "mythological",
    "legendary king", "queen consort",
    "literature", "literary work", "roman de",
    "chanson de geste", "matière de bretagne",
}


# ── Known-place gazetteer for conditional linking ─────────────────────
_KNOWN_PLACE_GAZETTEER: set[str] = {
    "lausanne", "paris", "lyon", "rome", "avignon", "geneve", "genève",
    "camelot", "logres", "bretagne", "cornouailles", "gaules",
    "tintagel", "carduel", "carlion", "winchester", "londres",
    "jérusalem", "jerusalem", "constantinople", "acre",
    "norgales", "gorre", "sarras", "benoic", "gaule",
}


_WEAK_PLACE_LEXICAL_BLACKLIST: set[str] = {
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


_PLACE_CONTEXT_PATTERNS: tuple[str, ...] = (
    " a {surface} ",
    " a la {surface} ",
    " au {surface} ",
    " aux {surface} ",
    " as {surface} ",
    " en {surface} ",
    " vers {surface} ",
    " sur {surface} ",
    " sor {surface} ",
    " de la cite de {surface} ",
    " de la ville de {surface} ",
    " del pais de {surface} ",
    " du pais de {surface} ",
    " en la cite de {surface} ",
    " en la ville de {surface} ",
)


_PLACE_LOCATIVE_PREPOSITIONS: tuple[str, ...] = (
    "a",
    "au",
    "aux",
    "en",
    "vers",
    "sur",
    "sor",
    "devers",
    "lez",
    "pres",
    "près",
)


_PLACE_HEADWORDS: tuple[str, ...] = (
    "cite",
    "cité",
    "ville",
    "castel",
    "chastel",
    "pais",
    "pays",
    "terre",
    "roiaume",
    "reaume",
    "contree",
    "contrée",
    "abbaye",
    "monastere",
    "monastère",
    "eglise",
    "église",
    "diocese",
    "diocèse",
)


_CANONICAL_ENTITIES: dict[str, dict[str, Any]] = {
    "arthur":     {"canon": "Arthur",     "queries": ["King Arthur", "Arthur Arthurian legend"], "type": "person"},
    "artus":      {"canon": "Arthur",     "queries": ["King Arthur", "Arthur Arthurian legend"], "type": "person"},
    "lancelot":   {"canon": "Lancelot",   "queries": ["Lancelot", "Lancelot du Lac"], "type": "person"},
    "leantlote":  {"canon": "Lancelot",   "queries": ["Lancelot", "Lancelot du Lac"], "type": "person"},
    "leantilote": {"canon": "Lancelot",   "queries": ["Lancelot", "Lancelot du Lac"], "type": "person"},
    "lancelote":  {"canon": "Lancelot",   "queries": ["Lancelot", "Lancelot du Lac"], "type": "person"},
    "lanselot":   {"canon": "Lancelot",   "queries": ["Lancelot", "Lancelot du Lac"], "type": "person"},
    "lanceloc":   {"canon": "Lancelot",   "queries": ["Lancelot", "Lancelot du Lac"], "type": "person"},
    "guenievre":  {"canon": "Guinevere",  "queries": ["Guinevere", "Guenièvre"], "type": "person"},
    "guenievr":   {"canon": "Guinevere",  "queries": ["Guinevere", "Guenièvre"], "type": "person"},
    "perceval":   {"canon": "Perceval",   "queries": ["Perceval", "Percival Arthurian"], "type": "person"},
    "parsival":   {"canon": "Perceval",   "queries": ["Perceval", "Percival Arthurian"], "type": "person"},
    "merlin":     {"canon": "Merlin",     "queries": ["Merlin", "Merlin enchanteur"], "type": "person"},
    "galahad":    {"canon": "Galahad",    "queries": ["Galahad", "Galaad chevalier"], "type": "person"},
    "galaad":     {"canon": "Galahad",    "queries": ["Galahad", "Galaad chevalier"], "type": "person"},
    "tristan":    {"canon": "Tristan",    "queries": ["Tristan", "Tristan Iseut"], "type": "person"},
    "gauvain":    {"canon": "Gauvain",    "queries": ["Gauvain", "Gawain chevalier"], "type": "person"},
    "gawain":     {"canon": "Gauvain",    "queries": ["Gauvain", "Gawain chevalier"], "type": "person"},
    "yvain":      {"canon": "Yvain",      "queries": ["Yvain", "Yvain chevalier au lion"], "type": "person"},
    "ivain":      {"canon": "Yvain",      "queries": ["Yvain", "Yvain chevalier au lion"], "type": "person"},
    "graal":      {"canon": "Graal",      "queries": ["Saint Graal", "Holy Grail"], "type": "work"},
    "bohort":     {"canon": "Bohort",     "queries": ["Bohort", "Bors chevalier"], "type": "person"},
    "erec":       {"canon": "Érec",       "queries": ["Érec", "Erec et Enide"], "type": "person"},
}


_VOWELS_SET = set("aeiouyàâäéèêëïîôùûüœæ")


def _levenshtein_quick(s: str, t: str) -> int:
    """Levenshtein edit distance (pure-Python, small strings)."""
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


def _compute_name_likeness(surface: str, ent_type: str) -> float:
    """Return a 0-1 quality score for a mention surface.

    A high score means the surface looks like a plausible entity name.
    A low score (< ``_NAME_QUALITY_THRESHOLD``) means the surface is
    likely OCR garbage and should NOT be sent to Wikidata.

    Factors:
      * Blacklisted common word → 0.0
      * Canonical match (normalized distance + bigram overlap) → 1.0 / 0.90
      * Vowel ratio within [0.20, 0.80]
      * No runs of ≥ 4 consecutive consonants
      * Token length within [3, 20]
      * Not purely stopwords
    """
    import re as _re
    from app.services.text_normalization import (
        is_blacklisted_token,
        normalized_edit_distance as _ned,
        bigram_overlap as _bo,
    )
    norm = _re.sub(r"[^\w\s]", " ", surface, flags=_re.UNICODE)
    norm = " ".join(norm.split()).strip().lower()
    if not norm:
        return 0.0

    tokens = norm.split()

    # Fast path: canonical match on any token → excellent quality
    # Uses normalized edit distance + bigram overlap (not raw dist)
    for tok in tokens:
        # Reject blacklisted tokens from contributing to quality
        if is_blacklisted_token(tok):
            continue
        if tok in _CANONICAL_ENTITIES:
            return 1.0
        for canon in _CANONICAL_ENTITIES:
            if len(tok) < 5:
                continue  # short tokens: exact match only (above)
            nd = _ned(tok, canon)
            bo = _bo(tok, canon)
            if nd <= 0.25 and bo >= 0.40:
                return 0.90

    # Remove role triggers / stopwords for scoring
    content = [
        t for t in tokens
        if t not in _QUERY_STOPWORDS and len(t) >= 2
        and not is_blacklisted_token(t)
    ]
    if not content:
        return 0.05  # pure stopwords / blacklisted → low

    score = 0.0
    n_good = 0
    for tok in content:
        tok_score = 0.5  # base
        # Length
        if len(tok) < 3 or len(tok) > 20:
            tok_score -= 0.30
        # Vowel ratio
        vowels = sum(1 for c in tok if c in _VOWELS_SET)
        ratio = vowels / len(tok)
        if ratio < 0.15 or ratio > 0.85:
            tok_score -= 0.30
        elif 0.25 <= ratio <= 0.65:
            tok_score += 0.15  # ideal range
        # Consonant clusters
        max_run = 0
        run = 0
        for c in tok:
            if c in _VOWELS_SET:
                run = 0
            else:
                run += 1
                max_run = max(max_run, run)
        if max_run >= 4:
            tok_score -= 0.35  # heavy penalty

        if tok_score > 0:
            n_good += 1
        score += max(0.0, tok_score)

    avg = score / len(content) if content else 0.0
    # Bonus for having multiple good tokens (reinforces signal)
    if n_good >= 2:
        avg = min(1.0, avg + 0.10)
    return round(min(1.0, max(0.0, avg)), 3)


def _check_canonical_match(surface: str) -> dict[str, Any] | None:
    """If any token in *surface* fuzzy-matches a canonical entity, return it.

    Uses length-aware matching:
      - Tokens < 5 chars: exact match only
      - Tokens >= 5 chars: normalized_edit_distance <= 0.25 AND
        bigram_overlap >= 0.40
      - Blacklisted common words are never matched.

    Returns a dict ``{"canon": …, "queries": …, "type": …, "token": …,
    "dist": …, "nd": …, "bo": …}`` or ``None``.
    """
    import re as _re
    from app.services.text_normalization import (
        is_blacklisted_token,
        normalized_edit_distance as _ned,
        bigram_overlap as _bo,
    )
    norm = _re.sub(r"[^\w\s]", " ", surface, flags=_re.UNICODE)
    tokens = norm.split()

    best: dict[str, Any] | None = None
    best_nd: float = 1.0

    for tok in tokens:
        tok_low = tok.lower()
        if len(tok_low) < 3:
            continue
        if is_blacklisted_token(tok_low):
            continue
        for canon_key, info in _CANONICAL_ENTITIES.items():
            nd = _ned(tok_low, canon_key)
            # Short tokens (< 5 chars): exact match only
            if len(tok_low) < 5:
                if nd == 0.0 and nd < best_nd:
                    best = {**info, "token": tok_low, "dist": 0, "nd": 0.0, "bo": 1.0}
                    best_nd = nd
                continue
            # Tokens >= 5 chars: normalized distance + bigram overlap
            bo = _bo(tok_low, canon_key)
            if nd <= 0.25 and bo >= 0.40 and nd < best_nd:
                raw_dist = _levenshtein_quick(tok_low, canon_key)
                best = {**info, "token": tok_low, "dist": raw_dist, "nd": nd, "bo": bo}
                best_nd = nd
    return best


def _has_strong_place_context(surface: str, evidence_text: str) -> bool:
    import re as _re

    surface_norm = normalize_for_search(surface)
    evidence_norm = f" {normalize_for_search(evidence_text)} "
    if not surface_norm or not evidence_norm.strip():
        return False
    for pattern in _PLACE_CONTEXT_PATTERNS:
        if pattern.format(surface=surface_norm) in evidence_norm:
            return True
    escaped_surface = _re.escape(surface_norm)
    locatives = "|".join(_re.escape(token) for token in _PLACE_LOCATIVE_PREPOSITIONS)
    headwords = "|".join(_re.escape(normalize_for_search(token)) for token in _PLACE_HEADWORDS)
    if _re.search(
        rf"\b(?:{locatives})\b(?:\s+\w+){{0,3}}\s+{escaped_surface}\b",
        evidence_norm,
    ):
        return True
    if _re.search(
        rf"\b(?:{headwords})\b(?:\s+\w+){{0,2}}\s+(?:de|del|du|des)\s+{escaped_surface}\b",
        evidence_norm,
    ):
        return True
    return False


def _deferred_reason_for_mention(surface: str, ent_type: str, evidence_text: str, base_reason: str) -> str:
    ent_low = str(ent_type or "").strip().lower()
    surface_text = str(surface or "").strip()
    surface_low = normalize_for_search(surface_text)
    reason_text = str(base_reason or "").strip()
    if ent_low != "place" or not surface_low:
        return reason_text
    is_known_place = surface_low in _KNOWN_PLACE_GAZETTEER
    is_capitalized = surface_text[:1].isupper()
    lexical = surface_low in _WEAK_PLACE_LEXICAL_BLACKLIST
    context_ok = _has_strong_place_context(surface_text, evidence_text)
    minimal_shape = len(surface_low) >= 4 and token_quality_score(surface_low) >= 0.45
    if is_known_place or ((is_capitalized or context_ok) and not lexical and minimal_shape):
        return reason_text
    suffix = (
        f"low_evidence_place: cap={is_capitalized} context={context_ok} "
        f"lexical={lexical} shape={minimal_shape}"
    )
    if not reason_text:
        return suffix
    if suffix in reason_text:
        return reason_text
    return f"{reason_text}; {suffix}"


def _summarize_authority_reason(reason: str, surface: str, ent_type: str) -> str:
    reason_text = str(reason or "").strip()
    surface_low = str(surface or "").strip().lower()
    ent_low = str(ent_type or "").strip().lower()
    if not reason_text:
        return "no reliable authority match"
    if "low_evidence_place" in reason_text:
        if surface_low in _WEAK_PLACE_LEXICAL_BLACKLIST or "lexical=True" in reason_text:
            return "insufficient evidence to treat this lexical item as a place"
        return "insufficient contextual evidence to support place linking"
    if "token_search_allowed=False" in reason_text:
        return "linking deferred because the passage quality was too risky for reliable authority lookup"
    if "name_likeness=" in reason_text:
        return "surface form is too noisy for reliable authority lookup"
    if reason_text == "no wikidata candidates":
        return "no reliable authority candidate found"
    if reason_text == "no candidate marked as selected":
        return "no candidate met the selection threshold"
    if reason_text == "marked ambiguous":
        return "multiple candidates remained ambiguous"
    if ent_low == "date" and "date" in reason_text:
        return "date detected but not authority-linked"
    return reason_text[:120]


# Trigger tokens that denote a title/role preceding a personal name.
_ROLE_TRIGGERS: set[str] = {
    "roi", "roy", "roya", "rex", "evesque", "évesque", "euesque",
    "duc", "conte", "sire", "pape", "dame", "monsieur",
}


_QUERY_STOPWORDS: set[str] = {
    "et", "de", "la", "le", "les", "des", "du", "en", "a", "au",
    "que", "qui", "ne", "pas", "par", "son", "ses", "sa",
    "li", "lo", "el", "al", "ou", "un", "une",
    # Also include role triggers so they get stripped in query simplification
    "roi", "roy", "roya", "rex", "sire", "monsieur", "dame",
    "evesque", "évesque", "euesque", "duc", "conte", "pape",
}


# Words that stand beside a name in a mention span but are not part of the
# name: titles and roles, liturgical rubric words, and function words a run
# of capitals picks up at a sentence start. Compared on the search key, so
# "Sainct", "scĩ" (sancti) and "BEATI" match. The span shown to the user keeps
# them; only the authority query drops them.
_NAME_QUERY_DROP_KEYS: frozenset[str] = frozenset(
    {
        # saints and the blessed
        "saint", "sainct", "sainz", "sains", "sainte", "saincte", "seint", "seinte",
        "san", "sant", "santa", "santo", "sanctus", "sancti", "sancta", "sancte",
        "sancto", "sanctum", "sanctam", "sanctorum", "beatus", "beati", "beata",
        "beate", "beato", "beatum",
        # rulers, clergy, nobility and their Latin cases
        "roi", "rois", "roy", "roys", "roya", "rey", "re", "rex", "regis", "regem",
        "regina", "reine", "royne", "dux", "ducis", "ducem", "duci", "duc", "dus",
        "comes", "comitis", "conte", "comte", "cuens", "sire", "messire",
        "monseigneur", "monsieur", "dominus", "domini", "dominum", "domino",
        "dominos", "domine", "dom", "dame", "dompna", "episcopus", "episcopi",
        "episcopum", "euesque", "euesques", "papa", "pape", "abbas", "abbatis",
        "frater", "fratris", "fratres", "magister", "magistri", "maistre", "mestre",
        "apostolus", "apostoli", "apostle", "euangelista", "euangeliste",
        "euangelistae", "martyr", "martir", "martiris", "martyris", "confessor",
        "confessoris", "presbiter", "presbyter", "clericus", "clerico", "cl",
        # rubric words
        "capitulum", "oratio", "lectio", "antiphona", "ant", "ps", "psalmus",
        "uigilia", "feria", "alius", "alia", "item", "incipit", "explicit",
        # function words at a sentence start
        "aussi", "et", "est", "in", "ad", "de", "que", "qui", "le", "la", "li", "lo",
        "les", "el", "al", "del", "a", "o",
    }
)


# A point after an abbreviated word abbreviates; any other sentence mark ends
# the name ("Sancti Marci. In").
_NAME_QUERY_SENTENCE_RE = re.compile(r"[.;:!?¶]")
_COMBINING_RE = re.compile("[\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f]")


def _strip_marks(text: str) -> str:
    """Drop the combining abbreviation marks left after NFC: "Rog̾o" -> "Rogo", "Frãcois" kept, "mͣrcadigas" -> "marcadigas".

    _build_query_variants turned them into spaces ("Rog o"). A letter that has
    a precomposed form (ã, é) keeps it.
    """
    import unicodedata

    text = unicodedata.normalize("NFC", text).translate(_SUPERSCRIPT_LETTERS)
    return _COMBINING_RE.sub("", text)


# Combining superscript letters stand for letters ("mͣrcadigas" is marcadigas).
_SUPERSCRIPT_LETTERS = {0x0363 + i: letter for i, letter in enumerate("aeioucdhmrtvx")}


_NAME_PARTICLE_KEYS = frozenset({"de", "d", "du", "des"})
_ARTICLE_AFTER_DE_KEYS = frozenset({"la", "le", "les"})
# Dropped words that are not titles: a name never starts right after them.
_NAME_QUERY_FUNCTION_KEYS = frozenset(
    {"aussi", "et", "est", "in", "ad", "de", "que", "qui", "le", "la", "li", "lo", "les", "el", "al", "del", "a", "o", "item"}
)


def _name_query(surface: str) -> str:
    """The name in a person mention's span, as the authority query.

    Titles, roles, rubric words and sentence-start function words are dropped
    and the span is cut at a sentence mark (a point after an abbreviated word
    excepted): "Sainct Pierre" -> "Pierre", "Rey Rodrigo" -> "Rodrigo", "Aussi
    de Bar" -> "Bar", "Sancti Marci. In" -> "Marci". After a title only its
    name is kept, since the words after it are the sentence ("beati iohannis
    obtutibus presentatus" -> "iohannis"); so it is in a span written all in
    capitals, where capitals tell nothing. A particle keeps the word after it,
    whatever its case ("Thomas de hereforde", "Rommant de la Rose"). Returns
    the surface itself when nothing would be left.
    """
    from app.services.medieval_text import build_search_key

    kept: list[str] = []
    for word in str(surface or "").split():
        core = word.rstrip(".;:!?¶,")
        mark = word[len(core):]
        if core:
            kept.append(core)
        if _NAME_QUERY_SENTENCE_RE.search(mark):
            abbreviated = mark == "." and (
                _COMBINING_RE.search(core[-1:]) is not None or any(c in "ꝑꝓꝗꝙꝯꝰꝭꝫłđħ" for c in core)
            )
            if abbreviated:
                continue
            if any(build_search_key(w) not in _NAME_QUERY_DROP_KEYS for w in kept):
                break
            kept = []  # only dropped words so far: the name is after the mark
    tokens = [w.strip(",()") for w in kept if w.strip(",()")]
    keys = [build_search_key(w) for w in tokens]
    if not tokens:
        return str(surface or "").strip()
    all_caps = all(w.upper() == w for w in tokens)
    dropped = [k in _NAME_QUERY_DROP_KEYS or not k for k in keys]
    # The name starts right after the first title that has a word after it
    # ("DISCIPLINIS BEATI COLUMBANI" -> COLUMBANI), else at the first word
    # not dropped.
    start = next(
        (
            i + 1
            for i, d in enumerate(dropped)
            if d and keys[i] not in _NAME_QUERY_FUNCTION_KEYS and i + 1 < len(tokens) and not dropped[i + 1]
        ),
        None,
    )
    after_title = start is not None
    if start is None:
        start = next((i for i, d in enumerate(dropped) if not d), None)
    if start is None:
        return str(surface or "").strip()
    # After a title the following words are the sentence ("beati iohannis
    # obtutibus"), unless capitals mark them as the name ("Domini Martini Iusti").
    capitals_tell = not all_caps and tokens[start][:1].isupper()
    name = [tokens[start]]
    i = start + 1
    while i < len(tokens):
        key, word = keys[i], tokens[i]
        is_particle = key in _NAME_PARTICLE_KEYS or (key in _ARTICLE_AFTER_DE_KEYS and keys[i - 1] == "de")
        if is_particle and not all_caps:
            name.append(word)
            i += 1
            continue
        if name and build_search_key(name[-1]) in (_NAME_PARTICLE_KEYS | _ARTICLE_AFTER_DE_KEYS):
            if key in _NAME_QUERY_DROP_KEYS and key not in _ARTICLE_AFTER_DE_KEYS:
                break
            name.append(word)  # the word after a particle, whatever its case
            i += 1
            continue
        if key in _NAME_QUERY_DROP_KEYS or all_caps or not word[:1].isupper() or (after_title and not capitals_tell):
            break
        name.append(word)
        i += 1
    while name and build_search_key(name[-1]) in (_NAME_PARTICLE_KEYS | _ARTICLE_AFTER_DE_KEYS):
        name.pop()
    query = _strip_marks(" ".join(name)).strip()
    return query or str(surface or "").strip()


def _prefilter_candidates(
    candidates: list[dict[str, Any]],
    surface: str,
    ent_type: str,
) -> list[dict[str, Any]]:
    """Pre-filter Wikidata candidates BEFORE calling wbgetentities.

    Removes candidates whose description contains modern-occupation
    keywords that are obviously irrelevant to medieval entity linking.
    Also sorts by relevance (domain keyword bonus) so the enrichment
    cap processes the best candidates first.
    """
    from app.services.text_normalization import normalize_for_search

    surface_norm = normalize_for_search(surface)
    kept: list[tuple[float, dict[str, Any]]] = []

    for wd in candidates:
        desc = str(wd.get("description", "")).lower()
        # Reject modern-occupation descriptions
        reject = _MODERN_DESCRIPTION_REJECT_RE.search(desc)
        if reject:
            log.debug(
                "pre-filter: rejecting %s (%s) — description contains '%s'",
                wd.get("qid", "?"), wd.get("label", "?"), reject.group(1),
            )
            continue

        # Domain keyword bonus for sorting
        bonus = 0.0
        for kw in _MEDIEVAL_DOMAIN_KEYWORDS:
            if kw in desc:
                bonus += 0.10
        # String proximity bonus
        label_norm = normalize_for_search(str(wd.get("label", "")))
        if label_norm and surface_norm and label_norm == surface_norm:
            bonus += 0.30
        elif label_norm and surface_norm and surface_norm in label_norm:
            bonus += 0.15
        kept.append((bonus, wd))

    # Sort by bonus descending, return just the candidates
    kept.sort(key=lambda x: -x[0])
    return [wd for _, wd in kept]


def _build_query_variants(surface: str, method: str) -> list[str]:
    """Generate progressively simplified queries for noisy medieval OCR.

    Strategy (always applied, regardless of method):
      a) normalised full surface (whitespace normalised, punct removed)
      b) drop stopwords/roles
      c) longest token(s) with len >= 5

    For ``rule:salvage_trigger`` mentions that start with a role trigger,
    the role-stripped form is always the *first* query so that Wikidata
    searches the personal name directly.

    Returns 1-3 deduplicated query strings.
    """
    # Normalise: collapse newlines/whitespace, strip punctuation
    import re as _re
    norm = _re.sub(r"[^\w\s]", " ", surface, flags=_re.UNICODE)
    norm = " ".join(norm.split()).strip()
    if not norm:
        return [surface]

    tokens = norm.split()

    queries: list[str] = []

    # ── (a) normalised full surface ───────────────────────────────────
    queries.append(norm)

    # ── (b) drop stopwords / roles ────────────────────────────────────
    content_tokens = [
        t for t in tokens
        if t.lower() not in _QUERY_STOPWORDS and len(t) >= 2
    ]
    if content_tokens and " ".join(content_tokens) != norm:
        queries.append(" ".join(content_tokens))

    # ── (c) longest tokens with len >= 5 ──────────────────────────────
    long_tokens = sorted(
        [t for t in tokens if len(t) >= 5],
        key=len, reverse=True,
    )[:2]
    if long_tokens:
        lt_str = " ".join(long_tokens)
        queries.append(lt_str)

    # ── For salvage_trigger role mentions: inject name-only first ─────
    if method == "rule:salvage_trigger" and len(tokens) >= 2:
        first_low = tokens[0].lower()
        if first_low in _ROLE_TRIGGERS:
            name_only = [
                t for t in tokens[1:]
                if t.lower() not in _QUERY_STOPWORDS and len(t) >= 2
            ]
            if name_only:
                # Insert name-only query at the front (highest priority)
                queries.insert(0, " ".join(name_only))

    # Deduplicate preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        q_low = q.strip().lower()
        if q_low and q_low not in seen:
            seen.add(q_low)
            deduped.append(q.strip())
    return deduped or [surface]
