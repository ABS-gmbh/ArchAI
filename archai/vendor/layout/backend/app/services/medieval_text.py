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
# Unicode word tokens: keeps Greek, Hebrew, Arabic and Cyrillic, which an
# ASCII-only pattern silently deleted.
#
# Combining marks must be allowed INSIDE a token. They are not word characters,
# so a bare \w+ pattern splits decomposed text: NFD "dñs" became ["dn", "s"] and
# the suspension lookup could never fire. That is the common case rather than an
# edge case, because the Kraken models here contain zero precomposed letters and
# emit base letter plus combining mark.
_COMBINING = r"\u0300-\u036F\u1AB0-\u1AFF\u1DC0-\u1DFF\u20D0-\u20F0"

# Some abbreviation signs are not word characters at all - the Tironian et U+204A
# is punctuation - so they must be named explicitly or the tokenizer discards
# them and the expansion never runs. Derived from the table so the two cannot
# drift apart.
_SIGN_CHARS = "".join(sorted(ch for ch in ABBREVIATION_LETTERS if not ch.isalnum()))
_SIGNS = re.escape(_SIGN_CHARS)
_TOKEN_RE = re.compile(
    rf"[^\W_{_SIGNS}](?:[^\W_]|[{_COMBINING}{_SIGNS}])*|[{_SIGNS}]+", re.UNICODE
)


def _is_latin_letter(char: str) -> bool:
    return char.isalpha() and ("LATIN" in unicodedata.name(char, ""))


_LATIN_VOWELS = frozenset("aeiouyAEIOUY")


def _is_suspension_bar(base: str | None, marks_since_base: int) -> bool:
    """Whether a macron/tilde here is a scribal nasal suspension.

    Two conditions, both needed to avoid mangling ordinary modern orthography:

    * The base must be a LATIN VOWEL. A medieval suspension bar stands for a
      nasal following a vowel. Spanish and Portuguese put a tilde on a
      consonant, so "España" was becoming "espanna".
    * The mark must sit DIRECTLY on the base, with no other combining mark
      between. Vietnamese stacks a tone mark on top of a vowel diacritic, so
      "Nguyễn" (e + circumflex + tilde) was becoming "nguyenn".

    A tilde directly on a vowel stays ambiguous - Portuguese "irmã" against a
    medieval suspension - and is read as a suspension, which is the right default
    for a medieval manuscript pipeline.
    """
    if base is None or marks_since_base != 0:
        return False
    return base in _LATIN_VOWELS and _is_latin_letter(base)


def _expand_chars(text: str, *, resolve_nasals: bool) -> str:
    """Shared expansion pass over a decomposed string.

    A nasal bar is only honoured when the character it sits on is a LATIN letter.
    U+0303/0304/0305 are generic combining marks that also occur in Greek
    long vowels, Spanish/Portuguese tildes and Vietnamese tone marks; treating
    them as suspensions there turned "España" into "espanna" and Greek long
    alpha into a Latin "n".
    """
    if not text:
        return ""

    decomposed = unicodedata.normalize("NFD", text)
    out: list[str] = []
    # Base letter the pending combining marks belong to, and how many marks have
    # been seen since it.
    base: str | None = None
    marks_since_base = 0

    for char in decomposed:
        if char in ABBREVIATION_LETTERS:
            out.append(ABBREVIATION_LETTERS[char])
            base = None
            continue
        if char in SUPERSCRIPT_LETTERS:
            out.append(SUPERSCRIPT_LETTERS[char])
            base = None
            continue
        if char in NASAL_MARKS:
            if _is_suspension_bar(base, marks_since_base):
                if resolve_nasals:
                    # A placeholder object, not in-band text, so source text that
                    # literally contains a marker cannot be mistaken for one.
                    out.append(_NASAL)
                # else: drop it, producing the classic abbreviation spelling.
            else:
                out.append(char)
            marks_since_base += 1
            continue
        if unicodedata.combining(char):
            marks_since_base += 1
        else:
            base = char
            marks_since_base = 0
        out.append(char)

    return _assemble(out)


# Sentinel object for a pending nasal. A distinct object cannot collide with
# input text, unlike the in-band string marker this replaced.
class _Nasal:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<nasal>"


_NASAL = _Nasal()


def _assemble(parts: list[object]) -> str:
    """Join expansion output, resolving each nasal from the letter that follows."""
    result: list[str] = []
    for index, part in enumerate(parts):
        if not isinstance(part, _Nasal):
            result.append(str(part))
            continue
        # Look ahead past any further combining marks to the next real letter,
        # so a second, unrelated diacritic cannot flip the m/n decision.
        following = ""
        for candidate in parts[index + 1 :]:
            if isinstance(candidate, _Nasal):
                continue
            text = str(candidate)
            stripped = "".join(ch for ch in text if not unicodedata.combining(ch))
            if stripped:
                following = stripped[0].lower()
                break
        result.append("m" if following in {"b", "p", "m"} else "n")
    return unicodedata.normalize("NFC", "".join(result))


def expand_abbreviations(text: str) -> str:
    """Expand scribal abbreviation signs into letters.

    A nasal bar becomes "m" before a labial and "n" elsewhere.
    """
    return _expand_chars(text, resolve_nasals=True)


def expand_abbreviations_without_nasals(text: str) -> str:
    """expand_abbreviations, but nasal bars are dropped rather than resolved.

    Produces the classic scholarly abbreviation spelling ("dñs" -> "dns"), which
    is how WORD_SUSPENSIONS is keyed. Resolving the bar positionally first would
    give "dnns" and miss the entry.
    """
    return _expand_chars(text, resolve_nasals=False)


def rejoin_line_breaks(text: str, *, lexicon: frozenset[str] | None = None) -> str:
    """Rejoin words split across manuscript line breaks.

    A trailing hyphen (including ¬ and the soft hyphen) is consumed and the two
    fragments joined. Pre-1300 hands frequently break with no mark at all; when a
    lexicon is supplied, an unmarked break is joined only if the concatenation is
    a known word and neither fragment is.

    Without this, one chunk per line strands 12-26% of content words as fragments
    that match no query token.
    """
    if not text:
        return ""

    lines = text.splitlines()
    if len(lines) < 2:
        return text

    out: list[str] = []
    buffer = ""
    # Iterate the INPUT positions explicitly. Deriving the position from the
    # number of emitted lines drifted backwards after every join, because a join
    # consumes two input lines and emits one - so look-ahead inspected the wrong
    # line, missing real joins and fabricating false ones.
    for index, raw in enumerate(lines):
        current = raw.rstrip()
        if buffer:
            current = buffer + current.lstrip()
            buffer = ""

        if _TRAILING_BREAK_RE.search(current):
            # Explicit break mark: drop it and hold the fragment for the next line.
            buffer = _TRAILING_BREAK_RE.sub("", current)
            continue

        if lexicon is not None and _looks_like_split_word(current, lines, index, lexicon):
            buffer = current
            continue

        out.append(current)

    if buffer:
        out.append(buffer)
    return "\n".join(out)


def _looks_like_split_word(
    current: str, lines: list[str], index: int, lexicon: frozenset[str]
) -> bool:
    """True when an unmarked break looks like a word split across two lines.

    Requires positive evidence: the concatenation must be a known word and
    neither fragment may be one on its own. Guessing without that corrupts
    ordinary line-final words.
    """
    if index + 1 >= len(lines):
        return False

    words = current.split()
    if not words:
        return False
    tail = words[-1].strip(".,;:!?")
    if len(tail) < 2 or tail.lower() in lexicon:
        return False

    next_words = lines[index + 1].split()
    if not next_words:
        return False
    head = next_words[0].strip(".,;:!?")
    if not head or head.lower() in lexicon:
        return False

    return (tail + head).lower() in lexicon


def build_search_key(text: str, *, expand: bool = True) -> str:
    """Build the normalised, matchable form used for indexing and querying.

    Each token is normalised independently. An earlier version derived two whole-
    string readings and paired them by token index; those two tokenizations could
    differ in length, which shifted every subsequent lookup and substituted wrong
    words into the key ("Iesus ᾱ dns ht" produced "iesus dominus habet habet").
    Deriving both readings per token removes that coupling entirely.

    Non-Latin scripts pass through case- and diacritic-folded rather than being
    dropped: the pipeline transcribes Greek and Hebrew pages too, and the previous
    ASCII-only tokenizer deleted them completely.

    Expansion is deliberately approximate. A single nasal bar encodes one nasal,
    so "oīs" resolves positionally to "oins" rather than philologically to
    "omnis"; WORD_SUSPENSIONS catches the frequent cases. The diplomatic text is
    never mutated and remains the citable reading.

    NOTE: matching only benefits where BOTH sides use this function. Authority
    linking still normalises with text_normalization.normalize_for_search, so the
    expanded form is currently written but not yet read by that matcher. Routing
    it through is a separate change.
    """
    if not text:
        return ""

    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        token = _normalise_token(raw, expand=expand)
        if token:
            tokens.append(token)
    return " ".join(tokens)


def _normalise_token(raw: str, *, expand: bool) -> str:
    """Normalise one token, applying Latin-specific rules only to Latin text."""
    if not _has_latin(raw):
        # Fold case and diacritics but keep the script's own letters.
        decomposed = unicodedata.normalize("NFD", raw.lower())
        return "".join(ch for ch in decomposed if not unicodedata.combining(ch))

    if not expand:
        return _fold_latin(raw)

    resolved = _fold_latin(expand_abbreviations(raw))
    classic = _fold_latin(expand_abbreviations_without_nasals(raw))

    # Both readings describe THIS token, so no cross-token alignment is involved.
    if classic in WORD_SUSPENSIONS:
        return WORD_SUSPENSIONS[classic]
    if resolved in WORD_SUSPENSIONS:
        return WORD_SUSPENSIONS[resolved]
    return resolved


def _fold_latin(text: str) -> str:
    """Lowercase, collapse digraphs, drop combining marks, fold i/j and u/v."""
    working = text.lower()
    for source, target in _DIGRAPHS:
        working = working.replace(source, target)
    decomposed = unicodedata.normalize("NFD", working)
    working = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    working = working.translate(_ORTHOGRAPHIC_FOLD)
    return "".join(ch for ch in working if ch.isalnum())


def _has_latin(text: str) -> bool:
    """True when a token should take the Latin-specific normalisation path.

    Abbreviation signs count as Latin regardless of their Unicode name: the
    Tironian et U+204A is punctuation and U+A770 is "MODIFIER LETTER US" with no
    "LATIN" in its name, so a name test alone sent them down the passthrough
    branch and they survived verbatim into the search key.
    """
    normalized = unicodedata.normalize("NFD", text)
    if any(ch in ABBREVIATION_LETTERS or ch in SUPERSCRIPT_LETTERS for ch in normalized):
        return True
    return any(_is_latin_letter(ch) for ch in normalized)


def _fold(text: str) -> list[str]:
    """Tokenize and fold, for the WORD_SUSPENSIONS key invariant test."""
    return [tok for tok in (_fold_latin(t) for t in _TOKEN_RE.findall(text)) if tok]
