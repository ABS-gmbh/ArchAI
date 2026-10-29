"""Measures of how much an OCR reading looks like text.

Script family, character entropy, token statistics, vowel and bigram
anomalies, fragment ratios at line starts, seams and line ends, cross-pass
stability, repetition, lexical contrast and the gibberish score built from
them. Moved out of ``app/services/ocr_quality.py``, which turns them into a
quality report and a label.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from app.services.ocr_quality_config import (
    GIBBERISH_WEIGHTS_NO_LANGUAGE,
    GIBBERISH_WEIGHTS_WITH_LANGUAGE,
    LEXICAL_CONTRAST_CLEAN,
    REPETITION_NGRAM,
    ENTROPY_HIGH_LIMIT,
    ENTROPY_LOW_LIMIT,
)


_SCRIPT_RANGES: list[tuple[str, int, int]] = [
    ("latin",     0x0000, 0x024F),
    ("latin",     0x1E00, 0x1EFF),   # Latin Extended Additional
    ("greek",     0x0370, 0x03FF),
    ("cyrillic",  0x0400, 0x04FF),
    ("arabic",    0x0600, 0x06FF),
    ("arabic",    0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
    ("hebrew",    0x0590, 0x05FF),
    ("cjk",       0x4E00, 0x9FFF),   # CJK Unified
    ("cjk",       0x3400, 0x4DBF),   # CJK Extension A
    ("cjk",       0x3040, 0x309F),   # Hiragana
    ("cjk",       0x30A0, 0x30FF),   # Katakana
    ("cjk",       0xAC00, 0xD7AF),   # Hangul
    ("devanagari", 0x0900, 0x097F),
    ("thai",      0x0E00, 0x0E7F),
    ("georgian",  0x10A0, 0x10FF),
    ("armenian",  0x0530, 0x058F),
    ("ethiopic",  0x1200, 0x137F),
]


def detect_script_family(text: str) -> str:
    """Classify the dominant script family in *text*.

    Returns one of: latin, greek, cyrillic, arabic, hebrew, cjk,
    devanagari, thai, georgian, armenian, ethiopic, mixed, unknown.
    """
    if not text or not text.strip():
        return "unknown"

    counts: Counter[str] = Counter()
    total_letters = 0
    for ch in text:
        if not ch.isalpha():
            continue
        total_letters += 1
        cp = ord(ch)
        matched = False
        for script, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[script] += 1
                matched = True
                break
        if not matched:
            counts["other"] += 1

    if total_letters == 0:
        return "unknown"

    # Dominant script = highest count
    top_script, top_count = counts.most_common(1)[0]
    ratio = top_count / total_letters
    if ratio >= 0.80:
        return top_script
    if len(counts) >= 2:
        return "mixed"
    return top_script


def char_entropy(text: str) -> float:
    """Shannon entropy (bits) of character distribution.

    Low entropy (< 2.5) → repetitive / few unique chars → suspicious.
    Very high entropy (> 5.5) → random-looking → suspicious.
    Normal text: 3.0–5.0.
    """
    if not text:
        return 0.0
    counts = Counter(text.lower())
    total = len(text)
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
        if c > 0
    )


def token_length_stats(tokens: Sequence[str]) -> dict[str, float]:
    """Compute token-length distribution statistics.

    Returns mean, stddev, fraction of very short (<3) and very long (>15).
    """
    if not tokens:
        return {"mean": 0.0, "stddev": 0.0, "very_short_frac": 1.0, "very_long_frac": 0.0}

    lengths = [len(t) for t in tokens]
    n = len(lengths)
    mean = sum(lengths) / n
    variance = sum((l - mean) ** 2 for l in lengths) / max(n, 1)
    stddev = variance ** 0.5

    very_short = sum(1 for l in lengths if l < 3)
    very_long = sum(1 for l in lengths if l > 15)

    return {
        "mean": round(mean, 3),
        "stddev": round(stddev, 3),
        "very_short_frac": round(very_short / n, 3),
        "very_long_frac": round(very_long / n, 3),
    }


# Vowel sets per script family  (configurable)
_VOWELS_LATIN = set("aeiouyàáâãäåæèéêëìíîïòóôõöùúûüýÿœ")


_VOWELS_GREEK = set("αεηιουωάέήίόύώ")


_VOWELS_CYRILLIC = set("аеёиоуыэюя")


_VOWEL_SETS: dict[str, set[str]] = {
    "latin": _VOWELS_LATIN,
    "greek": _VOWELS_GREEK,
    "cyrillic": _VOWELS_CYRILLIC,
}


def vowel_ratio(token: str, script: str = "latin") -> float:
    """Fraction of alphabetic chars that are vowels (0..1).

    Uses script-specific vowel sets.  For unknown/CJK/Arabic scripts,
    returns -1.0 (not applicable).
    """
    vowels = _VOWEL_SETS.get(script)
    if vowels is None:
        return -1.0  # not applicable for this script
    letters = [ch for ch in token.lower() if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ch in vowels) / len(letters)


def token_vowel_anomaly(token: str, script: str = "latin") -> bool:
    """True if token has anomalous vowel ratio for its script.

    Latin: normal 0.20–0.70.  Outside → anomaly.
    """
    vr = vowel_ratio(token, script)
    if vr < 0:
        return False  # not applicable
    if script in ("latin", "greek", "cyrillic"):
        return vr < 0.15 or vr > 0.75
    return False


# For Latin script: bigrams that almost never occur in real words
_RARE_LATIN_BIGRAMS = frozenset({
    "zx", "xz", "qx", "xq", "jx", "xj", "zq", "qz",
    "bx", "xb", "vx", "xv", "kx", "xk", "wx", "xw",
    "jq", "qj", "jz", "zj", "bk", "kb", "fq", "qf",
    "vq", "qv", "zz", "xx", "qq",
})


def rare_bigram_ratio(text: str, script: str = "latin") -> float:
    """Fraction of character bigrams that are "rare" for the script.

    Only implemented for Latin script.  Returns 0.0 for other scripts.
    """
    if script != "latin":
        return 0.0
    low = text.lower()
    # Only consider alpha bigrams
    bigrams = [low[i:i+2] for i in range(len(low) - 1) if low[i].isalpha() and low[i+1].isalpha()]
    if not bigrams:
        return 0.0
    rare = sum(1 for bg in bigrams if bg in _RARE_LATIN_BIGRAMS)
    return rare / len(bigrams)


def non_wordlike_score(token: str, script: str = "latin") -> float:
    """Score how "non-wordlike" a token is (0 = normal, 1 = gibberish).

    Combines:
      - vowel anomaly
      - max consonant run (for Latin-like scripts)
      - digit/symbol mixing
      - rare bigram presence
    """
    if not token or len(token) < 2:
        return 0.5

    score = 0.0
    low = token.lower()
    alpha_chars = [ch for ch in low if ch.isalpha()]
    alpha_count = len(alpha_chars)

    # Non-alpha mixing penalty
    if alpha_count > 0:
        non_alpha_ratio = (len(low) - alpha_count) / len(low)
        score += min(0.3, non_alpha_ratio)

    # Vowel anomaly
    if script in _VOWEL_SETS:
        vr = vowel_ratio(low, script)
        if vr >= 0:
            if vr < 0.12:
                score += 0.30
            elif vr < 0.18:
                score += 0.15
            elif vr > 0.80:
                score += 0.20

    # Consonant cluster (Latin/Greek/Cyrillic only)
    if script in ("latin", "greek", "cyrillic"):
        vowels = _VOWEL_SETS.get(script, set())
        max_run = 0
        run = 0
        for ch in low:
            if ch.isalpha() and ch not in vowels:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        if max_run >= 5:
            score += 0.35
        elif max_run >= 4:
            score += 0.15

    # Rare bigrams
    if script == "latin" and alpha_count >= 4:
        rbr = rare_bigram_ratio(low, script)
        score += rbr * 0.5

    return min(1.0, score)


# Common 1-2 letter function words per script that should NOT be counted
# as seam fragments.  Lowercase only.
_FUNCTION_WORDS_LATIN: frozenset[str] = frozenset({
    # French / Old French
    "a", "à", "au", "de", "du", "en", "et", "il", "je", "la", "le", "li",
    "ne", "ni", "nu", "on", "ou", "où", "si", "tu", "un",
    # Latin
    "ab", "ad", "an", "at", "ex", "in", "is", "it", "id", "ob", "se", "ut",
    # Spanish / Italian / Portuguese
    "al", "da", "di", "do", "el", "eu", "ha", "ho", "lo", "mi", "no", "os",
    "su", "te", "ti", "un", "va", "vi", "yo",
    # English
    "am", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is",
    "it", "me", "my", "no", "of", "ok", "on", "or", "so", "to", "up", "us",
    "we",
    # German
    "ob", "um", "zu",
    # Single vowels that are legitimate words
    "a", "e", "i", "o", "u", "y",
})


_FUNCTION_WORDS_GREEK: frozenset[str] = frozenset({
    "αν", "ας", "γη", "δε", "εν", "θα", "κι", "με", "να", "ου", "σε", "ωσ",
})


_FUNCTION_WORDS_CYRILLIC: frozenset[str] = frozenset({
    "а", "в", "и", "к", "о", "с", "у", "я", "да", "до", "за", "из", "на",
    "не", "ни", "но", "ну", "об", "от", "по", "то",
})


_FUNCTION_WORDS: dict[str, frozenset[str]] = {
    "latin": _FUNCTION_WORDS_LATIN,
    "greek": _FUNCTION_WORDS_GREEK,
    "cyrillic": _FUNCTION_WORDS_CYRILLIC,
}


def _is_set_apart_initial(tokens: Sequence[str]) -> bool:
    """True when a line opens with an initial set apart from the rest of its word.

    Verse and lists often put each line's capital in a column of its own, and a
    line recogniser then reads "C aldꝰ li roys". A lone capital followed by a
    lowercase word is that initial, not a word cut at a tile seam: counted as a
    fragment, it marked 47% of the lines of a faithful Old French verse page.
    """
    if len(tokens) < 2 or len(tokens[0]) != 1:
        return False
    initial, following = tokens[0], tokens[1]
    return initial.isalpha() and initial.isupper() and following[:1].isalpha() and following[:1].islower()


def leading_fragment_ratio(lines: Sequence[str], script: str = "latin") -> float:
    """Fraction of lines starting with a partial-word fragment.

    Improved heuristic that avoids false-positives on legitimate short
    function words that commonly appear at line starts in medieval verse
    and prose (e.g. "de", "et", "li", "si", "en").

    True fragment indicators:
      - Single non-letter character at line start
      - Single consonant (not a known function word) at line start
      - 1-2 char lowercase token that is NOT in the script's function-word list
      - Line starts with a continuation marker (e.g. hyphen leftover)
    """
    if not lines:
        return 0.0

    func_words = _FUNCTION_WORDS.get(script, frozenset())
    fragments = 0
    total = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        total += 1
        tokens = stripped.split()
        first_word = tokens[0] if tokens else ""
        if not first_word:
            continue

        is_fragment = False

        # Continuation marker: line starts with hyphen or dash
        if first_word[0] in "-–—":
            is_fragment = True
        elif script in ("latin", "greek", "cyrillic"):
            lw = first_word.lower()
            # Skip known function words — they are NOT fragments
            if lw in func_words:
                continue
            if _is_set_apart_initial(tokens):
                continue
            # Single non-vowel letter that isn't a function word
            if len(first_word) == 1 and first_word.isalpha():
                vowels = _VOWEL_SETS.get(script, set())
                if first_word.lower() not in vowels:
                    is_fragment = True
            # 2-char lowercase token not in function words → likely fragment
            elif len(first_word) == 2 and first_word.isalpha() and not first_word[0].isupper():
                # Already checked it's not in func_words above
                is_fragment = True
        else:
            # For non-Latin scripts: only single non-alpha chars
            if len(first_word) == 1 and not first_word.isalpha():
                is_fragment = True

        if is_fragment:
            fragments += 1

    return fragments / max(total, 1)


def seam_fragment_ratio(
    lines: Sequence[str],
    seam_line_indices: Sequence[int] | None = None,
    script: str = "latin",
) -> float:
    """Fraction of seam-adjacent lines that show fragmentation.

    If *seam_line_indices* are provided (line numbers where tile seams fall),
    only those lines are checked — giving a geometry-aware detection.
    If not provided, all lines are checked (falling back to heuristic).
    A fragment at a seam boundary is a much stronger signal than a generic
    short token at an arbitrary line start.
    """
    if not lines:
        return 0.0

    func_words = _FUNCTION_WORDS.get(script, frozenset())

    # Determine which lines to check
    if seam_line_indices is not None and seam_line_indices:
        check_indices = set()
        for si in seam_line_indices:
            # Check the seam line and its immediate neighbours
            for offset in (-1, 0, 1):
                idx = si + offset
                if 0 <= idx < len(lines):
                    check_indices.add(idx)
    else:
        check_indices = set(range(len(lines)))

    if not check_indices:
        return 0.0

    fragments = 0
    checked = 0
    for idx in sorted(check_indices):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        checked += 1
        tokens = stripped.split()
        first = tokens[0] if tokens else ""
        last = tokens[-1] if tokens else ""

        has_leading = False
        has_trailing = False

        # Leading fragment check
        if first and first[0] in "-–—":
            has_leading = True
        elif first and script in ("latin", "greek", "cyrillic"):
            lw = first.lower()
            if lw not in func_words and not _is_set_apart_initial(tokens):
                if (len(first) == 1 and first.isalpha()
                        and first.lower() not in _VOWEL_SETS.get(script, set())):
                    has_leading = True
                elif len(first) == 2 and first.isalpha() and not first[0].isupper():
                    has_leading = True

        # Trailing fragment check (broken word at end)
        if last and last[-1] == "-":
            has_trailing = True
        elif last and len(last) == 1 and last.isalpha():
            has_trailing = True

        if has_leading or has_trailing:
            fragments += 1

    return fragments / max(checked, 1)


def trailing_fragment_ratio(lines: Sequence[str]) -> float:
    """Fraction of lines ending with a likely broken-off word.

    Indicators: line ends with a hyphen, or a single consonant.
    """
    if not lines:
        return 0.0

    fragments = 0
    total = 0
    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            continue
        total += 1
        if stripped.endswith("-"):
            fragments += 1
        elif len(stripped) >= 2:
            last_word = stripped.split()[-1] if stripped.split() else ""
            if len(last_word) == 1 and last_word.isalpha() and last_word.lower() not in _VOWELS_LATIN:
                fragments += 1

    return fragments / max(total, 1)


def token_stability_score(
    pass1_tokens: Sequence[str],
    pass2_tokens: Sequence[str],
) -> float:
    """Measure how stable tokens are across two OCR passes (0..1).

    1.0 = identical token sets.
    0.0 = completely different.

    Uses Jaccard similarity of token sets (case-insensitive).
    """
    if not pass1_tokens and not pass2_tokens:
        return 1.0
    s1 = {t.lower() for t in pass1_tokens if t.strip()}
    s2 = {t.lower() for t in pass2_tokens if t.strip()}
    if not s1 and not s2:
        return 1.0
    union = s1 | s2
    if not union:
        return 1.0
    return len(s1 & s2) / len(union)


def normalized_levenshtein_similarity(a: str, b: str) -> float:
    """Normalized Levenshtein similarity between two strings (0..1).

    Uses standard DP edit-distance.  1.0 = identical, 0.0 = completely
    different.  Normalised by the length of the longer string.
    """
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    # Standard two-row DP
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    dist = prev[lb]
    return 1.0 - dist / max(la, lb)


def compute_cross_pass_stability(text_a: str, text_b: str) -> float:
    """Robust cross-pass stability combining Jaccard + normalised Levenshtein.

    - Normalize whitespace before comparison.
    - Average of line-sequence Levenshtein similarity and token-level Jaccard.
    - Returns 0..1 (1.0 = identical).
    """
    def _norm(t: str) -> str:
        # r"\\s+" (escaped backslash + literal "s") matched a backslash followed
        # by s, never whitespace, so this normalizer was a no-op and two passes
        # differing only in spacing scored as unstable.
        return re.sub(r"\s+", " ", t.strip().lower())

    na, nb = _norm(text_a), _norm(text_b)
    if na == nb:
        return 1.0
    if not na or not nb:
        return 0.0

    # Line-level normalised Levenshtein
    lev_sim = normalized_levenshtein_similarity(na, nb)

    # Token-level Jaccard
    tok_a = na.split()
    tok_b = nb.split()
    jaccard = token_stability_score(tok_a, tok_b)

    return round((lev_sim + jaccard) / 2.0, 4)


def uncertainty_density(text: str) -> float:
    """Fraction of characters that are uncertainty markers (?, …, [...]).

    High values (> 0.10) indicate very low confidence OCR.
    """
    if not text:
        return 0.0
    markers = text.count("?") + text.count("…") + text.count("[…]") * 3
    return markers / max(len(text), 1)


def repetition_score(text: str, ngram: int = REPETITION_NGRAM) -> float:
    """Share of the page occupied by its single most repeated line or n-gram.

    0 = no repetition, 1 = the page is one phrase repeated. Detects the
    degenerate decoding loop that vision-language models fall into, where the
    same line is emitted dozens of times. Character-level signals are blind to
    this: repeated *real* text has entirely normal entropy, bigrams and
    word-likeness, so it scores identically to a clean transcription.
    """
    if not text or not text.strip():
        return 0.0

    # Line-level repetition: the share of lines that repeat an earlier line.
    # Measuring only the single most repeated line missed block loops: GLM-OCR
    # re-emitted an 8-line page sixteen times (112 lines, each line 1/8 of them),
    # and that scored 0.14 - under the RISKY limit - while 93% of it was repeats.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    line_share = 0.0
    if len(lines) >= 3:
        line_share = (len(lines) - len(set(lines))) / len(lines)

    # Token n-gram repetition: catches loops that are not line-aligned.
    #
    # n-grams are collected WITHIN each line, never across a line break. Spanning
    # breaks made structurally similar but distinct lines look like a loop: for
    # "line number 1 of the folio" / "line number 2 of the folio" / ... the
    # boundary-spanning 5-gram "of the folio line number" recurs on every line,
    # which scored a templated ledger at 0.59 as if it were degenerate output.
    all_tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    ngram_share = 0.0
    if len(all_tokens) >= ngram * 3:
        counts: Counter[tuple[str, ...]] = Counter()
        for line in (lines or [text]):
            line_tokens = [t for t in re.split(r"\s+", line.strip()) if t]
            for i in range(len(line_tokens) - ngram + 1):
                counts[tuple(line_tokens[i : i + ngram])] += 1
        if counts:
            _, top_count = counts.most_common(1)[0]
            if top_count > 1:
                # Occurrences of the dominant n-gram, weighted by tokens covered.
                ngram_share = min(1.0, (top_count * ngram) / len(all_tokens))

    return round(min(1.0, max(line_share, ngram_share)), 4)


def lexical_contrast(text: str, language: str | None) -> LexicalContrast | None:
    """The lexical contrast of *text*, or None when it cannot be measured.

    None when no language is supplied, when the language has no trigram profile
    (so the scorer redistributes the weight rather than assume a neutral 0.5),
    and when the text is too short. A profiled language is measured together
    with the other profiled ones and the best fit kept: langdetect has no Latin
    and calls it Catalan or French, and the faithful Latin reference pages scored
    as Catalan were graded UNRELIABLE.
    """
    if not _has_profile(language) or not text or not text.strip():
        return None
    from app.services.lexicon_trust import lexical_contrast as measure

    return measure(text, language)


def _has_profile(language: str | None) -> bool:
    if not language or language == "unknown":
        return False
    from app.services.lexicon_trust import _TRIGRAM_PROFILES

    return bool(_TRIGRAM_PROFILES.get(language))


def implausibility_from_contrast(ratio: float) -> float:
    """Map a contrast ratio onto 0 (as language-like as clean text) to 1 (none).

    1.0 - no more common trigrams than shuffled letters - maps to 1, and
    LEXICAL_CONTRAST_CLEAN or above to 0.
    """
    span = LEXICAL_CONTRAST_CLEAN - 1.0
    return round(min(1.0, max(0.0, (LEXICAL_CONTRAST_CLEAN - ratio) / span)), 4)


def lexical_implausibility(text: str, language: str | None) -> float | None:
    """How unlike real language *text* is, 0-1, or None when unmeasured.

    See :func:`lexical_contrast` for when it is measured.
    """
    contrast = lexical_contrast(text, language)
    return None if contrast is None else implausibility_from_contrast(contrast.ratio)


def gibberish_score(text: str, script: str = "latin", language: str | None = None) -> float:
    """Composite gibberish detector (0 = clean, 1 = total gibberish).

    When *language* names a language with a trigram profile, the lexical signal
    carries most of the weight. The character-level signals this function used
    to rely on exclusively cannot tell real words from transposed ones: a
    sentence and the same sentence with every word reversed have identical
    entropy, bigram load and word-likeness, and both scored 0.0.
    """
    if not text or not text.strip():
        return 0.0

    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    if not tokens:
        return 0.0

    # 1. Non-wordlike token fraction
    nwl_scores = [non_wordlike_score(t, script) for t in tokens if len(t) >= 3]
    nwl_frac = (
        sum(1 for s in nwl_scores if s > 0.45) / max(len(nwl_scores), 1)
        if nwl_scores else 0.0
    )

    # 2. Entropy anomaly
    ent = char_entropy(text)
    ent_penalty = 0.0
    if ent < ENTROPY_LOW_LIMIT:
        ent_penalty = 0.3
    elif ent > ENTROPY_HIGH_LIMIT:
        ent_penalty = 0.2

    # 3. Rare bigram load
    rbr = rare_bigram_ratio(text, script)

    # 4. Uncertainty marker density
    unc = uncertainty_density(text)

    # 5. Repetition (decoding loops)
    rep = repetition_score(text)

    # 6. Lexical implausibility, when it can be measured. Clean text contributes
    #    nothing: the contrast is scaled so that clean-text contrast maps to 0.
    lex = lexical_implausibility(text, language)

    components = {
        "non_wordlike": nwl_frac,
        "entropy": ent_penalty,
        "rare_bigram": min(1.0, rbr * 5),
        "uncertainty": min(1.0, unc * 5),
        "repetition": rep,
    }
    if lex is None:
        weights = GIBBERISH_WEIGHTS_NO_LANGUAGE
    else:
        weights = GIBBERISH_WEIGHTS_WITH_LANGUAGE
        components["lexical"] = lex

    score = sum(weights[name] * value for name, value in components.items() if name in weights)
    return round(min(1.0, score), 4)
