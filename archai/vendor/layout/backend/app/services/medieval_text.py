"""Scribal abbreviation expansion, line-break rejoining and search-key building.

The Kraken models this pipeline ships transcribe *diplomatically*: they emit the
scribe's own abbreviation signs rather than expanding them. Measured against the
recognisers' own codecs (kraken.lib.models.load_any):

    mccatmus        116 symbols / 11 combining marks / 0 precomposed letters
    catmus_medieval 224 symbols / 32 combining marks / 0 precomposed letters
    cremma_medieval  98 symbols / 16 combining marks / 0 precomposed letters

catmus_medieval alone can emit 14 of 19 tested abbreviation signs. Nothing in the
pipeline expanded them, and normalize_for_search's NFKD pass *deleted* combining
marks outright - so a macron, which carries the whole suspension, silently
vanished. The result was that 0 of 21 medieval/modern spelling pairs collapsed to
a common search key, and a query could never match the indexed text.

This module produces a SEARCH form only. The diplomatic transcription is never
mutated: a thesis needs the reading as the scribe wrote it, so expansion output
is stored alongside the original, never in place of it.
"""

from __future__ import annotations

import re
import unicodedata

# ── Standalone abbreviation letters (MUFI / Unicode Latin Extended-D) ──
#
# Each maps to its most frequent expansion. Where a sign is genuinely ambiguous
# (per vs par) the commoner form is used; the diplomatic text keeps the sign, so
# nothing is lost, and the search key only needs to be *consistent*.
ABBREVIATION_LETTERS: dict[str, str] = {
    "ꝑ": "per",   # ꝑ  p with stroke through descender
    "ꝓ": "pro",   # ꝓ  p with flourish
    "ꝕ": "pro",   # ꝕ  p with squirrel tail
    "ꝗ": "qui",   # ꝗ  q with stroke through descender
    "ꝙ": "quod",  # ꝙ  q with diagonal stroke
    "ꝛ": "r",     # ꝛ  r rotunda
    "ꝝ": "rum",   # ꝝ  rum rotunda
    "ꝯ": "con",   # ꝯ  con
    "ꝰ": "us",    # ꝰ  us / os
    "ꝫ": "et",    # ꝫ  et / us
    "ꝭ": "is",    # ꝭ  is
    "⁊": "et",    # ⁊  Tironian et
    "ſ": "s",     # ſ  long s
    "ß": "ss",    # ß
    "ʒ": "z",     # ʒ  ezh, used for -et / -us in some hands
}

# ── Superscript vowels and consonants (contraction markers) ──
SUPERSCRIPT_LETTERS: dict[str, str] = {
    "ͣ": "a",  # ◌ͣ
    "ͤ": "e",  # ◌ͤ
    "ͥ": "i",  # ◌ͥ
    "ͦ": "o",  # ◌ͦ
    "ͧ": "u",  # ◌ͧ
    "ͨ": "c",  # ◌ͨ
    "ͩ": "d",  # ◌ͩ
    "ͪ": "h",  # ◌ͪ
    "ͫ": "m",  # ◌ͫ
    "ͬ": "r",  # ◌ͬ
    "ͭ": "t",  # ◌ͭ
    "ͮ": "v",  # ◌ͮ
    "ͯ": "x",  # ◌ͯ
}

# ── Suspension marks: a bar over a letter standing for a nasal ──
NASAL_MARKS = ("̄", "̅", "̃")  # combining macron, overline, tilde

# Whole-word suspensions that a bar cannot resolve positionally.
#
# A key must NOT also be an ordinary word in the languages handled here, or a
# genuine word gets silently rewritten. "aut" (Latin for "or") was removed for
# exactly that reason; the gain on the abbreviation never justifies corrupting
# every real occurrence.
#
# Keys must be written in the form they take AFTER folding (lowercase, no
# diacritics, digraphs collapsed, i/j and u/v merged) or the lookup can never
# fire. "noe" for nomine, for example, folds to "ne" because oe -> e, so it is
# unreachable; test_suspension_keys_survive_folding enforces this.
WORD_SUSPENSIONS: dict[str, str] = {
    "dns": "dominus",
    "dni": "domini",
    "dno": "domino",
    "dnm": "dominum",
    "ihs": "iesus",
    "ihu": "iesu",
    "xps": "christus",
    "xpi": "christi",
    "xpo": "christo",
    "scs": "sanctus",
    "sci": "sancti",
    "sco": "sancto",
    "sca": "sancta",
    "eps": "episcopus",
    "epi": "episcopi",
    "nra": "nostra",
    "nri": "nostri",
    "nro": "nostro",
    "spu": "spiritu",
    "sps": "spiritus",
    "aplus": "apostolus",
    # Positional nasal expansion produces these intermediate forms, because a
    # single bar encodes only one nasal: "oīs" yields "oins" but the word is
    # "omnis". Resolving that needs the word, not the mark.
    "oins": "omnis",
    "oim": "omnium",
    "oina": "omnia",
    "tnc": "tunc",
    "qnd": "quando",
    "qm": "quoniam",
    "qa": "quia",
    "ee": "esse",
    "ht": "habet",
    "hns": "habens",
}

# Orthographic variation that is not abbreviation but still blocks matching:
# medieval scribes do not distinguish i/j or u/v, and spell ae/oe as e.
_ORTHOGRAPHIC_FOLD = str.maketrans({"j": "i", "v": "u", "w": "uu"})

_DIGRAPHS = (("æ", "e"), ("œ", "e"), ("ae", "e"), ("oe", "e"))

_HYPHEN_CHARS = "-‐‑‒–­¬"
_TRAILING_BREAK_RE = re.compile(rf"[{re.escape(_HYPHEN_CHARS)}]\s*$")
_MULTI_SPACE = re.compile(r"\s+")


def expand_abbreviations(text: str) -> str:
    """Expand scribal abbreviation signs into letters.

    Operates on a decomposed form so that a base letter and its combining mark
    are visible as separate code points, then reassembles. A nasal bar becomes
    "n" (the commoner expansion; "m" before a labial is handled by context).
    """
    if not text:
        return ""

    # NFD so combining marks detach from their base letters.
    decomposed = unicodedata.normalize("NFD", text)
    out: list[str] = []

    for char in decomposed:
        if char in ABBREVIATION_LETTERS:
            out.append(ABBREVIATION_LETTERS[char])
            continue
        if char in SUPERSCRIPT_LETTERS:
            out.append(SUPERSCRIPT_LETTERS[char])
            continue
        if char in NASAL_MARKS:
            # A bar over a vowel stands for a following nasal. "m" is correct
            # before a labial, so decide from the letter that follows.
            out.append("~NASAL~")
            continue
        out.append(char)

    joined = "".join(out)
    joined = _resolve_nasals(joined)
    return unicodedata.normalize("NFC", joined)


def expand_abbreviations_without_nasals(text: str) -> str:
    """expand_abbreviations, but nasal bars are dropped rather than resolved.

    Produces the classic scholarly abbreviation spelling ("dñs" -> "dns"), which
    is how WORD_SUSPENSIONS is keyed. Resolving the bar positionally first would
    give "dnns" and miss the entry.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text)
    out: list[str] = []
    for char in decomposed:
        if char in ABBREVIATION_LETTERS:
            out.append(ABBREVIATION_LETTERS[char])
        elif char in SUPERSCRIPT_LETTERS:
            out.append(SUPERSCRIPT_LETTERS[char])
        elif char in NASAL_MARKS:
            continue
        else:
            out.append(char)
    return unicodedata.normalize("NFC", "".join(out))


def _resolve_nasals(text: str) -> str:
    """Turn each nasal placeholder into m before a labial, else n."""
    result: list[str] = []
    index = 0
    marker = "~NASAL~"
    while index < len(text):
        if text.startswith(marker, index):
            following = text[index + len(marker) : index + len(marker) + 1].lower()
            result.append("m" if following in {"b", "p", "m"} else "n")
            index += len(marker)
            continue
        result.append(text[index])
        index += 1
    return "".join(result)


def rejoin_line_breaks(text: str, *, lexicon: frozenset[str] | None = None) -> str:
    """Rejoin words split across manuscript line breaks.

    A trailing hyphen (including ¬ and the soft hyphen) is consumed and the two
    fragments joined. Pre-1300 hands frequently break with no mark at all; when
    a lexicon is supplied, an unmarked break is joined only if the concatenation
    is a known word and neither fragment is.

    Without this, one chunk per line strands 12-26% of content words as
    fragments that match no query token.
    """
    if not text:
        return ""

    lines = text.splitlines()
    if len(lines) < 2:
        return text

    out: list[str] = []
    buffer = ""

    for line in lines:
        current = line.rstrip()
        if buffer:
            current = buffer + current.lstrip()
            buffer = ""

        if _TRAILING_BREAK_RE.search(current):
            # Explicit break mark: drop it and hold the fragment for the next line.
            buffer = _TRAILING_BREAK_RE.sub("", current)
            continue

        if lexicon is not None:
            joined = _try_unmarked_join(current, lines, out, lexicon)
            if joined is not None:
                buffer = joined
                continue

        out.append(current)

    if buffer:
        out.append(buffer)
    return "\n".join(out)


def _try_unmarked_join(
    current: str, lines: list[str], emitted: list[str], lexicon: frozenset[str]
) -> str | None:
    """Return the held fragment when an unmarked break looks like a split word."""
    words = current.split()
    if not words:
        return None
    tail = words[-1].strip(".,;:!?")
    if len(tail) < 2 or tail.lower() in lexicon:
        return None

    index = len(emitted)
    if index + 1 >= len(lines):
        return None
    next_words = lines[index + 1].split()
    if not next_words:
        return None
    head = next_words[0].strip(".,;:!?")
    if not head or head.lower() in lexicon:
        return None

    if (tail + head).lower() in lexicon:
        return current
    return None


def build_search_key(text: str, *, expand: bool = True) -> str:
    """Build the normalised, matchable form used for indexing and querying.

    Diacritics are still folded, but only AFTER abbreviation signs have been
    expanded, so the mark that carried the suspension contributes letters instead
    of being deleted. Orthographic variation that medieval scribes did not
    observe (i/j, u/v, ae/oe) is folded so a modern query can reach the text.

    Expansion is deliberately approximate. A single nasal bar encodes one nasal,
    so "oīs" resolves positionally to "oins" rather than philologically to
    "omnis"; WORD_SUSPENSIONS catches the frequent cases. What retrieval needs is
    that documents and queries pass through THIS SAME function, so the two agree
    - not that either is philologically ideal. The diplomatic text is untouched
    and remains the citable reading.
    """
    if not text:
        return ""

    if not expand:
        return _MULTI_SPACE.sub(" ", " ".join(_fold(text))).strip()

    # Two readings of the same text: nasal bars resolved positionally, and nasal
    # bars dropped. WORD_SUSPENSIONS is keyed on the latter.
    resolved = _fold(expand_abbreviations(text))
    classic = _fold(expand_abbreviations_without_nasals(text))

    tokens: list[str] = []
    for index, token in enumerate(resolved):
        alt = classic[index] if index < len(classic) else token
        if alt in WORD_SUSPENSIONS:
            tokens.append(WORD_SUSPENSIONS[alt])
        elif token in WORD_SUSPENSIONS:
            tokens.append(WORD_SUSPENSIONS[token])
        else:
            tokens.append(token)
    return _MULTI_SPACE.sub(" ", " ".join(tokens)).strip()


def _fold(text: str) -> list[str]:
    """Lowercase, fold digraphs and medieval orthography, and tokenize."""
    working = text.lower()
    for source, target in _DIGRAPHS:
        working = working.replace(source, target)
    decomposed = unicodedata.normalize("NFD", working)
    working = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    working = working.translate(_ORTHOGRAPHIC_FOLD)
    return [tok for tok in re.split(r"[^0-9a-z]+", working) if tok]
