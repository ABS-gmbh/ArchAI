"""Language-aware lexical trust scoring for OCR output.

Provides a *lexical plausibility* signal that complements the existing
structural sanity checks (single-char ratio, consonant runs, etc.).

The scorer uses **character-trigram frequency** profiles for common
manuscript languages.  Rather than a full dictionary lookup (which would
require large word-lists and is brittle for medieval orthography), trigram
profiles capture the *statistical texture* of a language and degrade
gracefully on variant spellings.

Usage::

    score = lexical_plausibility("furent les noces", "old_french")
    # → 0.82  (high: trigrams match Old French profile)

    score = lexical_plausibility("qjxvvbbx cccnnn", "old_french")
    # → 0.15  (low: trigrams are implausible)

The module is language-agnostic in structure — adding a new language
only requires appending to ``_TRIGRAM_PROFILES``.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from app.services.medieval_text import build_search_key, fold_orthography, rejoin_line_breaks

# ── Trigram frequency profiles ───────────────────────────────────────
# Each profile is a set of the ~120 most frequent character trigrams for
# the language, derived from representative medieval corpora.  Membership
# in this set is the cheapest useful signal; full frequency weights are
# not needed for the trust-gate use-case.

_TRIGRAM_PROFILES: dict[str, frozenset[str]] = {
    "latin": frozenset({
        "ent", "ter", "ion", "tio", "ati", "nis", "est", "unt", "tis",
        "que", "per", "tur", "ita", "oru", "rum", "ere", "iss", "sti",
        "con", "ant", "tur", "mus", "qui", "ius", "ine", "eri", "min",
        "men", "ali", "tes", "pro", "tra", "str", "bus", "ita", "ris",
        "dem", "cum", "nos", "ter", "dom", "ndo", "tus", "ium", "omi",
        "ect", "unt", "tat", "atu", "sec", "nes", "ver", "tan", "pot",
        "ens", "mit", "pra", "end", "ost", "pon", "aut", "quo", "tem",
        "und", "int", "unt", "rat", "cit", "ect", "dit", "ere", "ibu",
        "tri", "ati", "non", "nti", "ere", "nte", "sed", "tis", "ras",
        "ris", "ica", "rea", "ore", "mis", "nos", "mni", "lla", "pri",
        "lib", "nim", "eri", "uri", "eni", "san", "ern", "ist", "acc",
        "pre", "nat", "cre", "gen", "gra", "mul", "ple", "sem", "spi",
    }),
    "old_french": frozenset({
        "ent", "les", "est", "ant", "oit", "ois", "oit", "que", "ous",
        "ure", "des", "ont", "aut", "par", "ain", "ien", "ter", "our",
        "ort", "com", "con", "ere", "ier", "ion", "ais", "ame", "oie",
        "roi", "ors", "ois", "ver", "ali", "oit", "nes", "mes", "lor",
        "ent", "art", "ard", "ort", "ure", "ust", "ren", "out", "tot",
        "poi", "hom", "gra", "ran", "ble", "res", "nce", "ele", "ien",
        "rou", "uss", "eur", "ais", "ort", "der", "sen", "ten", "gen",
        "ave", "oir", "oir", "vil", "ail", "ner", "rei", "erm", "ard",
        "sem", "pre", "pro", "soi", "ign", "age", "ail", "onn", "rie",
        "anc", "ari", "noi", "moi", "doi", "voi", "nui", "tos", "uel",
        "cor", "don", "mon", "hon", "bon", "fon", "pon", "lon", "son",
        "enc", "emp", "ens", "erl", "ble", "ain", "ein", "oin", "uin",
    }),
    "middle_french": frozenset({
        "ent", "les", "est", "que", "ant", "des", "ous", "our", "ion",
        "con", "par", "ois", "com", "ait", "ure", "ont", "ter", "aut",
        "ain", "ien", "ere", "res", "ier", "ois", "nce", "ble", "eur",
        "ais", "ort", "ren", "mes", "ver", "ali", "ois", "art", "ard",
        "pre", "pro", "sen", "ten", "gen", "age", "onn", "rie", "ave",
        "enc", "emp", "ens", "erl", "ain", "ein", "oin", "uin", "oir",
        "deu", "ieu", "ieu", "roy", "mon", "don", "hon", "bon", "pon",
        "tre", "ell", "omm", "app", "aue", "lle", "iss", "ans", "oir",
        "pri", "che", "cha", "chi", "cho", "chu", "ran", "san", "man",
        "dis", "mis", "fai", "mai", "lai", "pai", "rai", "sai", "tai",
        "ign", "gne", "ner", "mer", "per", "der", "ser", "ler", "rer",
        "nte", "ntr", "str", "cti", "iqu", "eme", "ess", "ass", "oss",
    }),
    "french": frozenset({
        "les", "ent", "des", "que", "ion", "est", "ait", "ons", "ant",
        "ous", "our", "con", "par", "com", "eur", "ois", "ure", "ier",
        "ter", "res", "nce", "ble", "men", "ais", "oir", "ort", "pre",
        "pro", "age", "ell", "omm", "app", "lle", "iss", "ans", "oir",
        "tre", "ain", "ein", "oin", "enc", "emp", "ens", "che", "cha",
        "ran", "san", "man", "dis", "mis", "fai", "mai", "rai", "tai",
        "nte", "ntr", "str", "iqu", "eme", "ess", "ass", "iti", "ali",
        "ens", "ell", "lle", "eau", "aux", "eux", "ieu", "oeu", "ail",
        "eil", "ouv", "ouv", "cha", "chi", "cho", "chu", "cha", "gne",
    }),
    "unknown": frozenset(),   # no profile → neutral score
}

# Aliases
_TRIGRAM_PROFILES["anglo_norman"] = _TRIGRAM_PROFILES["old_french"]
_TRIGRAM_PROFILES["occitan"] = _TRIGRAM_PROFILES["middle_french"]
_TRIGRAM_PROFILES["italian"] = _TRIGRAM_PROFILES["latin"]
_TRIGRAM_PROFILES["spanish"] = _TRIGRAM_PROFILES["latin"]
_TRIGRAM_PROFILES["portuguese"] = _TRIGRAM_PROFILES["latin"]
_TRIGRAM_PROFILES["catalan"] = _TRIGRAM_PROFILES["middle_french"]

# Text is scored through its search key (abbreviations expanded, u/v and i/j
# merged), so the profiles are folded the same way: "ver" is matched as "uer".
_TRIGRAM_PROFILES = {
    language: frozenset(folded for folded in map(fold_orthography, profile) if len(folded) == 3)
    for language, profile in _TRIGRAM_PROFILES.items()
}


def lexical_plausibility(text: str, detected_language: str) -> float:
    """Score the lexical plausibility of *text* for *detected_language*.

    Returns a float 0-1 where:
      - >= 0.60 → text is lexically consistent with the language
      - 0.35-0.60 → uncertain
      - < 0.35 → likely OCR garbage or wrong language

    If no trigram profile exists for the language, returns 0.50 (neutral).
    """
    profile = _TRIGRAM_PROFILES.get(detected_language)
    if profile is None or not profile:
        return 0.50

    trigrams = _extract_trigrams(text)
    if not trigrams:
        return 0.50

    hits = sum(1 for tri in trigrams if tri in profile)
    ratio = hits / len(trigrams)
    # Scale to 0-1 range (a perfect hit-rate is unlikely ~0.6 for real text)
    return min(1.0, ratio / 0.55)


@dataclass(frozen=True)
class LexicalContrast:
    """How language-like the order of a text's letters is.

    ``ratio`` is the share of the text's trigrams found in a language profile,
    divided by the same share for the text with the letters of each word
    shuffled. Shuffling keeps each word's letters, and so their frequencies,
    and removes only their order: real words are made of common letter
    sequences, and reversed or scrambled words are not. The hit rate
    (:func:`lexical_plausibility`) falls with wrong letters and much less with
    misordered ones; the contrast is the other way round. At thresholds that
    passed faithful page windows and light noise alike, the hit rate caught
    80% of windows at 37% CER and 57% of reversed ones, the contrast 30% and
    95%, so the quality gate uses both. The reference pages long enough to
    measure read 3.1 to 4.7, the same pages with each word reversed 1.3 to 1.9,
    in random letters 0.9 to 1.2.
    """

    ratio: float
    language: str
    trigrams: int


# Languages the contrast is measured in when the hint is absent or unusable.
CONTRAST_LANGUAGES: tuple[str, ...] = ("latin", "old_french", "middle_french", "french")
# Below this many trigrams - eight to fifteen manuscript lines, depending on the
# hand - the contrast is not measured and the hit rate alone judges the text.
# Windows of the faithful reference pages fell under the UNRELIABLE threshold 3%
# of the time at 50-100 trigrams, and never from 125 on.
CONTRAST_MIN_TRIGRAMS = 150
_CONTRAST_SHUFFLES = 8
# Added to both hit counts. Random letters rarely contain a profile's trigrams,
# and a few lucky hits against one or two expected make a large ratio: with none
# added, 29% of random-letter texts of 150 trigrams or more reached 2.0 in their
# best-fitting profile, and one reached 10. With four, 0.7% reached 2.0 and none
# 2.2; of reversed-word texts, 2.7% reached 2.2 against 17%.
_CONTRAST_PSEUDO_HITS = 4.0


def lexical_contrasts(text: str, languages: Sequence[str] = CONTRAST_LANGUAGES) -> dict[str, LexicalContrast] | None:
    """The :class:`LexicalContrast` of *text* in each of *languages* that has a profile.

    None when *text* has fewer than ``CONTRAST_MIN_TRIGRAMS`` trigrams. The
    shuffles are seeded, so the result is a function of the text.
    """
    tokens = _key_tokens(text)
    real = _token_trigrams(tokens)
    if len(real) < CONTRAST_MIN_TRIGRAMS:
        return None
    shuffled = [_token_trigrams(_shuffle_letters(tokens, seed)) for seed in range(_CONTRAST_SHUFFLES)]
    shuffled_total = sum(len(trigrams) for trigrams in shuffled)

    contrasts: dict[str, LexicalContrast] = {}
    for language in dict.fromkeys(languages):
        profile = _TRIGRAM_PROFILES.get(language)
        if not profile:
            continue
        hits = sum(1 for trigram in real if trigram in profile)
        # Hits expected of the same number of trigrams at the shuffled rate.
        expected = len(real) * sum(1 for trigrams in shuffled for trigram in trigrams if trigram in profile) / shuffled_total
        ratio = (hits + _CONTRAST_PSEUDO_HITS) / (expected + _CONTRAST_PSEUDO_HITS)
        contrasts[language] = LexicalContrast(round(ratio, 4), language, len(real))
    return contrasts


def lexical_contrast(text: str, language: str | None = None) -> LexicalContrast | None:
    """The best :class:`LexicalContrast` of *text* over the profiled languages.

    *language*, when it has a profile, is measured too, so a correct hint can
    only raise the result - a mistaken one cannot lower it. None when *text* is
    too short to measure.
    """
    languages = [*CONTRAST_LANGUAGES, *([language] if language else [])]
    contrasts = lexical_contrasts(text, languages)
    if not contrasts:
        return None
    return max(contrasts.values(), key=lambda contrast: contrast.ratio)


# Latin is chosen over a guessed Romance language only when its contrast is
# this many times the best French one. On the reference pages and their CATMuS
# and CREMMA readings it was 1.28-1.88 times for the Latin pages and 0.43-0.64
# for the French ones; no reading of a French page reached 0.96.
LATIN_OVER_FRENCH_MARGIN = 1.25


def reads_as_latin(text: str) -> bool:
    """True when *text* is clearly more Latin than French, by lexical contrast.

    langdetect has no Latin model: it reported the Latin reference pages and
    their readings as Catalan or French.
    """
    contrasts = lexical_contrasts(text)
    if not contrasts or "latin" not in contrasts:
        return False
    french = max(contrast.ratio for language, contrast in contrasts.items() if language != "latin")
    return contrasts["latin"].ratio >= LATIN_OVER_FRENCH_MARGIN * french


def best_fit_plausibility(text: str, language: str) -> float:
    """:func:`lexical_plausibility` in *language* or in a better-fitting profile.

    Neutral 0.50 for a language with no profile, as before: no profile here
    can judge German or Greek text.
    """
    if not _TRIGRAM_PROFILES.get(language):
        return 0.50
    return max(lexical_plausibility(text, candidate) for candidate in dict.fromkeys((language, *CONTRAST_LANGUAGES)))


def lexical_trust_adjustment(
    confidence: float,
    text: str,
    detected_language: str,
) -> tuple[float, list[str]]:
    """Adjust OCR confidence using lexical plausibility.

    Returns (adjusted_confidence, extra_warnings).
    """
    plausibility = lexical_plausibility(text, detected_language)
    warnings: list[str] = []

    if plausibility < 0.25:
        adjusted = confidence * 0.55
        warnings.append(f"LEXICAL_IMPLAUSIBLE:{plausibility:.2f}")
    elif plausibility < 0.40:
        adjusted = confidence * 0.75
        warnings.append(f"LEXICAL_WEAK:{plausibility:.2f}")
    elif plausibility >= 0.70:
        # Slight boost for strong lexical match
        adjusted = min(0.95, confidence * 1.05)
    else:
        adjusted = confidence

    return max(0.05, min(0.95, adjusted)), warnings


def agreement_score(texts: Sequence[str]) -> float:
    """Compute pairwise agreement among multiple OCR outputs.

    Returns 0-1 where 1.0 = all texts identical.
    Uses character-level Jaccard similarity on trigram bags.
    """
    if len(texts) < 2:
        return 1.0

    trigram_sets = [set(_extract_trigrams(t)) for t in texts]
    scores: list[float] = []
    for i in range(len(trigram_sets)):
        for j in range(i + 1, len(trigram_sets)):
            a, b = trigram_sets[i], trigram_sets[j]
            if not a and not b:
                scores.append(1.0)
                continue
            inter = len(a & b)
            union = len(a | b)
            scores.append(inter / max(union, 1))

    return sum(scores) / max(len(scores), 1)


def line_length_mismatch_ratio(
    lines: Sequence[str],
    expected_line_count: int | None = None,
) -> float:
    """Detect mismatch between segmentation geometry and recognized lines.

    If expected_line_count is provided, returns abs(actual - expected) / max(actual, expected).
    Otherwise returns 0.0 (no signal).
    """
    if expected_line_count is None or expected_line_count <= 0:
        return 0.0
    actual = len([ln for ln in lines if ln.strip()])
    if actual == 0 and expected_line_count == 0:
        return 0.0
    diff = abs(actual - expected_line_count)
    return diff / max(actual, expected_line_count, 1)


# ── Internals ────────────────────────────────────────────────────────


_LATIN_ALPHABET_TOKEN = re.compile(r"[a-z]+")


def _key_tokens(text: str) -> list[str]:
    """The Latin-alphabet words of *text*'s search key, three letters or longer.

    Diplomatic transcriptions write ẽ, ũ, ꝑ and ⁊. The earlier filter
    kept only a-z and Latin-1 letters, so it deleted those from the middle of
    their words - "cõciẽce" was scored as "ccice" - and a faithful transcription
    looked less like its language than a normalised one. The search key expands
    the abbreviations instead. Words in other scripts are left out, as before:
    no profile describes them.
    """
    return [
        token
        for token in build_search_key(rejoin_line_breaks(text or "")).split()
        if len(token) >= 3 and _LATIN_ALPHABET_TOKEN.fullmatch(token)
    ]


def _token_trigrams(tokens: Sequence[str]) -> list[str]:
    return [token[i : i + 3] for token in tokens for i in range(len(token) - 2)]


def _shuffle_letters(tokens: Sequence[str], seed: int) -> list[str]:
    rng = random.Random(seed)
    shuffled: list[str] = []
    for token in tokens:
        letters = list(token)
        rng.shuffle(letters)
        shuffled.append("".join(letters))
    return shuffled


def _extract_trigrams(text: str) -> list[str]:
    """Trigrams of *text*'s search key, as the folded profiles expect them."""
    return _token_trigrams(_key_tokens(text))
