"""Sanity checks on a model's transcription.

Junk and script-drift ratios, suspicious tokens, sanitising lines, the
line-level quality gate, uncertainty markers and the confidence they allow.
The OCR router and the SAIA agent both use them. Moved out of
``app/agents/saia_ocr_agent.py``.
"""

from __future__ import annotations

import re


_OCR_VOWELS = set("aeiouyàáâãäåæèéêëìíîïòóôõöùúûüýÿœ")


def _ocr_sanity_metrics(text: str) -> dict[str, float]:
    """Compute simple sanity metrics on OCR output."""
    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    if not tokens:
        return {"single_char_ratio": 0.0, "digit_ratio": 0.0, "junk_ratio": 0.0}

    single_char = sum(1 for t in tokens if len(t) == 1)
    digit_tokens = sum(1 for t in tokens if any(c.isdigit() for c in t))
    non_letter = sum(1 for t in tokens if not any(c.isalpha() for c in t))

    n = float(len(tokens))
    return {
        "single_char_ratio": single_char / n,
        "digit_ratio": digit_tokens / n,
        "junk_ratio": non_letter / n,
    }


_SANITY_SINGLE_CHAR_THRESHOLD = 0.30


_SANITY_DIGIT_THRESHOLD = 0.25


_SANITY_JUNK_THRESHOLD = 0.35


def _is_sane_ocr(text: str) -> bool:
    """Return True if OCR text passes basic sanity checks."""
    metrics = _ocr_sanity_metrics(text)
    if metrics["single_char_ratio"] > _SANITY_SINGLE_CHAR_THRESHOLD:
        return False
    if metrics["digit_ratio"] > _SANITY_DIGIT_THRESHOLD:
        return False
    if metrics["junk_ratio"] > _SANITY_JUNK_THRESHOLD:
        return False
    return True


_IMPROBABLE_BIGRAMS = {"qj", "vvv", "iii", "kk", "iiit", "xq", "jq", "zx", "qx"}


_ROMAN_NUMERAL_RE = re.compile(r"^[MDCLXVI]{2,}$", re.IGNORECASE)


_YEAR_LIKE_RE = re.compile(r"^(anno|an|m[cdxlvi]+)$", re.IGNORECASE)


_STANDALONE_FUNCTION_WORDS: dict[str, set[str]] = {
    "latin": {"a", "e", "i", "o", "u", "y", "et"},
    "old_french": {"a", "e", "i", "o", "u", "y", "et", "à"},
    "middle_french": {"a", "e", "i", "o", "u", "y", "et", "à"},
    "french": {"a", "e", "i", "o", "u", "y", "et", "à"},
    "old_english": {"a", "i"},
    "middle_english": {"a", "i", "o"},
}


_CONSONANT_LEADING_SET = set("cdlnprstvx")


_ZERO_WIDTH_RE = re.compile(r"[\ufeff\u200b\u200c\u200d\u2060\ufffe]")


_STRAY_K_ENDING = re.compile(r"^([a-zA-Z\u00C0-\u00FF]{3,}[bdfghjlmnpqrstvwxz])k$", re.IGNORECASE)


# Minimum consecutive single-char tokens to treat as a fragment run and collapse
_FRAGMENT_RUN_MIN = 3


# Languages where 'w' and 'k' are rare OCR artifacts (Old French, Latin, etc.)
_WK_SUSPICIOUS_LANGUAGES = {"old_french", "middle_french", "french", "latin"}


# Tokens that are entirely w/k junk → replace with '[…]'
_WK_JUNK_TOKENS = {"w", "k", "wr", "wk", "kw", "ww", "kk"}


def compute_sanity(text: str) -> dict[str, float]:
    """Compute archival sanity metrics for quality gate.

    Returns dict with:
      - single_char_ratio
      - digit_ratio
      - weird_ratio  (token len>=10 with vowel_ratio<0.20 OR improbable bigrams)
      - junk_ratio   (tokens where >40% of chars are non-letter / non-'?')
      - leading_fragment_ratio
      - uncertainty_marker_ratio
    """
    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    lines = [ln for ln in text.splitlines() if ln.strip()]
    n = float(len(tokens)) if tokens else 1.0
    n_lines = float(len(lines)) if lines else 1.0

    single_char = sum(1 for t in tokens if len(t) == 1)
    digit_tokens = sum(1 for t in tokens if any(ch.isdigit() for ch in t))

    weird = 0
    junk = 0
    for t in tokens:
        letters = [ch for ch in t if ch.isalpha()]
        lcount = len(letters)
        # Junk: tokens where >40% of chars are non-letter / non-'?' and len>=2
        if len(t) >= 2 and t not in ("[…]", "?"):
            non_letter = sum(1 for ch in t if not ch.isalpha() and ch != "?" and ch != "…" and ch not in "[]'''-–—.,;:")
            if non_letter / max(1, len(t)) > 0.40:
                junk += 1
        if lcount == 0:
            continue
        lower_t = t.lower()
        # Token len >= 10 with low vowels
        if lcount >= 10:
            vowel_ratio = sum(1 for ch in letters if ch.lower() in _OCR_VOWELS) / max(1, lcount)
            if vowel_ratio < 0.20:
                weird += 1
                continue
        # Consonant runs >= 5 (unlikely in Latin-script)
        if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", lower_t):
            weird += 1
            continue
        # Improbable bigrams
        if any(bg in lower_t for bg in _IMPROBABLE_BIGRAMS):
            weird += 1
            continue
        # 3+ repeated same char
        if re.search(r"(.)\1{2,}", lower_t) and not _ROMAN_NUMERAL_RE.match(t):
            weird += 1
            continue

    uncertainty_markers = sum(1 for t in tokens if t == "?" or t == "[…]" or "?" in t)

    leading_frag = 0
    for ln in lines:
        parts = ln.strip().split()
        if not parts:
            continue
        first = parts[0]
        if len(first) == 1 or (first.isdigit() and len(first) == 1):
            leading_frag += 1

    return {
        "single_char_ratio": single_char / n,
        "digit_ratio": digit_tokens / n,
        "weird_ratio": weird / n,
        "junk_ratio": junk / n,
        "leading_fragment_ratio": leading_frag / n_lines,
        "uncertainty_marker_ratio": uncertainty_markers / n,
    }


def _quality_label_from_sanity(metrics: dict[str, float]) -> str:
    """Compute HIGH / MEDIUM / LOW quality label from sanity metrics."""
    scr = metrics.get("single_char_ratio", metrics.get("single_char_token_ratio", 0.0))
    dr = metrics.get("digit_ratio", metrics.get("digit_token_ratio", 0.0))
    wr = metrics.get("weird_ratio", metrics.get("weird_token_ratio", 0.0))
    jr = metrics.get("junk_ratio", 0.0)
    if scr < 0.05 and dr < 0.01 and wr < 0.08 and jr < 0.03:
        return "HIGH"
    if scr < 0.10 and dr < 0.01 and wr < 0.15 and jr < 0.06:
        return "MEDIUM"
    return "LOW"


def _token_is_suspicious(t: str) -> bool:
    """Return True if a single token should be masked by sanitize rules."""
    letters = [ch for ch in t if ch.isalpha()]
    lcount = len(letters)
    if lcount == 0:
        return False
    lower_t = t.lower()
    # Digit tokens (not roman numerals / year-like)
    if any(ch.isdigit() for ch in t):
        stripped_alpha = "".join(ch for ch in t if ch.isalpha())
        if (
            not _ROMAN_NUMERAL_RE.match(t)
            and not _YEAR_LIKE_RE.match(t)
            and not _ROMAN_NUMERAL_RE.match(stripped_alpha)
        ):
            return True
    # Long token with low vowels
    if lcount >= 12:
        vowel_ratio = sum(1 for ch in letters if ch.lower() in _OCR_VOWELS) / max(1, lcount)
        if vowel_ratio < 0.20:
            return True
    # Improbable bigrams or repeated chars
    if lcount >= 3:
        if any(bg in lower_t for bg in _IMPROBABLE_BIGRAMS):
            return True
        if re.search(r"(.)\1{2,}", lower_t) and not _ROMAN_NUMERAL_RE.match(t):
            return True
    # Consonant run >= 5 (exempt Roman numerals)
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", lower_t) and not _ROMAN_NUMERAL_RE.match(t):
        return True
    return False


def _collapse_single_char_runs(
    tokens: list[str],
    func_words: set[str],
) -> tuple[list[str], int]:
    """Collapse runs of >= _FRAGMENT_RUN_MIN consecutive single-char tokens.

    Runs of single-char tokens like "a a mno y a nis" indicate severe
    segmentation failure. We collapse each such run into '[…]'.

    Single-char tokens that are legitimate function words are still counted
    in the run because the pattern "a a … a" is fragment noise, not real
    function words appearing consecutively.

    Returns (new_tokens, replacement_count).
    """
    out: list[str] = []
    replacements = 0
    i = 0
    n = len(tokens)
    while i < n:
        # Check if we're starting a run of single-char tokens
        if len(tokens[i]) == 1 and tokens[i].isalpha():
            # Count the run
            run_end = i + 1
            while run_end < n and len(tokens[run_end]) == 1 and tokens[run_end].isalpha():
                run_end += 1
            run_len = run_end - i
            if run_len >= _FRAGMENT_RUN_MIN:
                # Collapse the entire run to '[…]'
                out.append("[…]")
                replacements += run_len
                i = run_end
                continue
        out.append(tokens[i])
        i += 1
    return out, replacements


def _collapse_fragment_sequences(
    tokens: list[str],
    func_words: set[str],
) -> tuple[list[str], int]:
    """Collapse sequences where >=60% of tokens in a window of 5+ are single-char.

    Handles patterns like "a a mno y a nis" where short non-single-char
    tokens are interspersed among single-char fragments.
    """
    n = len(tokens)
    if n < 5:
        return tokens, 0

    # Mark positions that are single-char alphabetic
    is_single = [len(t) == 1 and t.isalpha() for t in tokens]

    # Use a sliding window to find stretches dominated by single-char tokens
    window = 5
    mask = [False] * n
    for start in range(n - window + 1):
        single_count = sum(1 for j in range(start, start + window) if is_single[j])
        if single_count / window >= 0.60:
            for j in range(start, start + window):
                mask[j] = True

    # Apply mask: consecutive masked regions become a single '[…]'
    out: list[str] = []
    replacements = 0
    i = 0
    while i < n:
        if mask[i]:
            # Consume entire masked run
            while i < n and mask[i]:
                replacements += 1
                i += 1
            out.append("[…]")
        else:
            out.append(tokens[i])
            i += 1
    return out, replacements


def sanitize_lines(
    lines: list[str],
    detected_language: str,
    script_hint: str,
) -> tuple[list[str], int]:
    """Deterministic diplomatic sanitizer.

    Returns (sanitized_lines, replacement_count).
    Rules:
      - Remove BOM / zero-width chars, drop empty lines.
      - Merge leading consonant initials {c,d,l,n,p,r,s,t,v,w,x} with next token.
      - Collapse runs of >=3 consecutive single-char tokens to '[…]'.
      - Collapse fragment-dominated sequences (>=60% single-char in windows of 5+).
      - Digit tokens (not roman numerals) → '[…]'.
      - Long low-vowel tokens (>=12 letters, <20% vowels) → '[…]'.
      - Improbable bigrams / 3+ repeated char → '[…]'.
      - Consonant run >= 5 in a token → '[…]'.
      - Trailing stray 'k' (OCR noise): replace trailing k with '?' if base
        is plausible, else '[…]'.
    """
    func_words = _STANDALONE_FUNCTION_WORDS.get(
        detected_language, _STANDALONE_FUNCTION_WORDS.get("latin", set())
    )
    replacements = 0
    result_lines: list[str] = []

    for line in lines:
        # Remove BOM / zero-width characters
        line = _ZERO_WIDTH_RE.sub("", line)
        stripped = line.strip()
        if not stripped:
            continue  # drop empty lines

        tokens = stripped.split()
        out: list[str] = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            lower_t = t.lower()

            # --- w/k masking (language-aware) ---
            # Skip tokens with digits (let digit rule handle those)
            if detected_language in _WK_SUSPICIOUS_LANGUAGES and not any(ch.isdigit() for ch in t):
                if lower_t in _WK_JUNK_TOKENS:
                    out.append("[…]")
                    replacements += 1
                    i += 1
                    continue
                if len(t) >= 2 and ("w" in lower_t or "k" in lower_t):
                    masked = ""
                    did_mask = False
                    for ch in t:
                        if ch.lower() in ("w", "k"):
                            masked += "?"
                            did_mask = True
                        else:
                            masked += ch
                    if did_mask:
                        out.append(masked)
                        replacements += 1
                        i += 1
                        continue

            # --- Leading consonant merge ---
            if (
                len(t) == 1
                and lower_t in _CONSONANT_LEADING_SET
                and lower_t not in func_words
                and i + 1 < len(tokens)
                and any(ch.isalpha() for ch in tokens[i + 1])
            ):
                merged = t + tokens[i + 1]
                out.append(merged)
                replacements += 1
                i += 2
                continue

            # --- Digit token → '[…]' ---
            if any(ch.isdigit() for ch in t):
                stripped_alpha = "".join(ch for ch in t if ch.isalpha())
                if (
                    not _ROMAN_NUMERAL_RE.match(t)
                    and not _YEAR_LIKE_RE.match(t)
                    and not _ROMAN_NUMERAL_RE.match(stripped_alpha)
                ):
                    out.append("[…]")
                    replacements += 1
                    i += 1
                    continue

            # --- Long low-vowel token → '[…]' ---
            letters = [ch for ch in t if ch.isalpha()]
            lcount = len(letters)
            if lcount >= 12:
                vowel_count = sum(1 for ch in letters if ch.lower() in _OCR_VOWELS)
                if vowel_count / max(1, lcount) < 0.20:
                    out.append("[…]")
                    replacements += 1
                    i += 1
                    continue

            # --- Improbable bigrams / repeated chars → '[…]' ---
            if lcount >= 3:
                if any(bg in lower_t for bg in _IMPROBABLE_BIGRAMS):
                    out.append("[…]")
                    replacements += 1
                    i += 1
                    continue
                # 3+ repeated char
                if re.search(r"(.)\1{2,}", lower_t) and not _ROMAN_NUMERAL_RE.match(t):
                    out.append("[…]")
                    replacements += 1
                    i += 1
                    continue

            # --- Consonant run >= 5 → '[…]' (exempt Roman numerals) ---
            if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", lower_t) and not _ROMAN_NUMERAL_RE.match(t):
                out.append("[…]")
                replacements += 1
                i += 1
                continue

            # --- Trailing stray 'k' (OCR noise) ---
            m = _STRAY_K_ENDING.match(t)
            if m:
                base = m.group(1)
                base_vowels = sum(1 for ch in base if ch.lower() in _OCR_VOWELS)
                if base_vowels > 0:
                    out.append(base + "?")
                else:
                    out.append("[…]")
                replacements += 1
                i += 1
                continue

            out.append(t)
            i += 1

        # --- Pass 2: Collapse consecutive single-char fragment runs ---
        out, run_repl = _collapse_single_char_runs(out, func_words)
        replacements += run_repl

        # --- Pass 3: Collapse fragment-dominated sequences ---
        out, seq_repl = _collapse_fragment_sequences(out, func_words)
        replacements += seq_repl

        # If >30% of line tokens were fully replaced with '[…]', 
        # the line is not archival-safe. Diplomatic '?' marks are fine.
        full_mask_count = sum(1 for tok in out if tok == "[…]")
        if out and (full_mask_count / max(1, len(out))) > 0.30:
            result_lines.append("[…]")
            replacements += len(out)
            continue

        # Deduplicate consecutive '[…]' markers
        deduped: list[str] = []
        for tok in out:
            if tok == "[…]" and deduped and deduped[-1] == "[…]":
                continue
            deduped.append(tok)

        result_lines.append(" ".join(deduped))

    return result_lines, replacements


def enforce_diplomatic_uncertainty(
    lines: list[str],
    script_hint: str,
    detected_language: str = "latin",
) -> tuple[list[str], bool]:
    """Backward-compatible wrapper around sanitize_lines."""
    sanitized, count = sanitize_lines(lines, detected_language, script_hint)
    return sanitized, count > 0


def quality_gate_enforce(
    lines: list[str],
    max_line_suspicious_ratio: float = 0.30,
) -> tuple[list[str], int]:
    """Second-pass masking: replace entire suspicious lines with '[…]'.

    A line is suspicious when > max_line_suspicious_ratio of its tokens
    are flagged by _token_is_suspicious.

    Returns (gated_lines, masked_line_count).
    """
    gated: list[str] = []
    masked_count = 0
    for line in lines:
        tokens = line.split()
        if not tokens:
            gated.append(line)
            continue
        suspicious_count = sum(1 for t in tokens if _token_is_suspicious(t))
        ratio = suspicious_count / len(tokens)
        # Mask lines with >30% suspicious tokens, or lines that are just '[…]' already
        if ratio > max_line_suspicious_ratio:
            gated.append("[…]")
            masked_count += 1
        else:
            gated.append(line)
    return gated, masked_count


def force_uncertainty_markers(
    lines: list[str],
    sanity: dict[str, float],
    weird_threshold: float = 0.05,
    junk_threshold: float = 0.03,
) -> tuple[list[str], int]:
    """Force uncertainty markers when text is noisy but has none.

    If after sanitization uncertainty_marker_ratio == 0 AND
    (weird_ratio > weird_threshold OR junk_ratio > junk_threshold),
    force-mask the top-N weirdest/most-suspicious tokens until at least
    one uncertainty marker is present.

    Returns (updated_lines, forced_mask_count).
    """
    unc = sanity.get("uncertainty_marker_ratio", 0.0)
    wr = sanity.get("weird_ratio", 0.0)
    jr = sanity.get("junk_ratio", 0.0)

    # Only force if no uncertainty but metrics indicate noise
    if unc > 0.0:
        return lines, 0
    if wr <= weird_threshold and jr <= junk_threshold:
        return lines, 0

    # Collect all tokens with suspicion scores, ranked by suspiciousness
    scored: list[tuple[int, int, float]] = []  # (line_idx, tok_idx, score)
    for li, line in enumerate(lines):
        tokens = line.split()
        for ti, t in enumerate(tokens):
            if t in ("[…]", "?"):
                continue
            score = 0.0
            if _token_is_suspicious(t):
                score += 1.0
            letters = [ch for ch in t if ch.isalpha()]
            lcount = len(letters)
            if lcount >= 6:
                vr = sum(1 for ch in letters if ch.lower() in _OCR_VOWELS) / max(1, lcount)
                if vr < 0.25:
                    score += 0.5
            if lcount >= 3 and re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", t.lower()):
                score += 0.3
            if score > 0:
                scored.append((li, ti, score))

    if not scored:
        # Nothing suspicious at token level; add a trailing uncertainty marker
        if lines:
            lines[-1] = lines[-1].rstrip() + " [?]"
            return lines, 1
        return lines, 0

    # Sort by suspicion score descending, mask top-N (at least 1, at most 3)
    scored.sort(key=lambda x: -x[2])
    n_to_mask = min(max(1, int(len(scored) * 0.10)), 3)
    forced = 0
    # Group by line for efficiency
    masks_by_line: dict[int, set[int]] = {}
    for li, ti, _score in scored[:n_to_mask]:
        masks_by_line.setdefault(li, set()).add(ti)
        forced += 1

    result: list[str] = []
    for li, line in enumerate(lines):
        if li in masks_by_line:
            tokens = line.split()
            for ti in masks_by_line[li]:
                if ti < len(tokens):
                    tokens[ti] = tokens[ti] + "?"
            result.append(" ".join(tokens))
        else:
            result.append(line)

    return result, forced


def format_sanity_summary(sanity: dict[str, float]) -> str:
    return (
        "SANITY "
        f"single_char={sanity.get('single_char_ratio', 0.0):.3f} "
        f"weird={sanity.get('weird_ratio', 0.0):.3f} "
        f"junk={sanity.get('junk_ratio', 0.0):.3f} "
        f"uncert={sanity.get('uncertainty_marker_ratio', 0.0):.3f}"
    )


def sanity_adjust_confidence(confidence_raw: float, sanity: dict[str, float]) -> float:
    """Sanity-adjust confidence: cannot be high when metrics are bad.

    Uses a multiplicative penalty for single_char_ratio, weird_ratio,
    and junk_ratio. Adds an extra penalty when uncertainty_marker_ratio == 0
    but weird_ratio is non-trivial (the "confident garbage" scenario).
    """
    scr = sanity.get("single_char_ratio", 0.0)
    wr = sanity.get("weird_ratio", 0.0)
    jr = sanity.get("junk_ratio", 0.0)
    umr = sanity.get("uncertainty_marker_ratio", 0.0)

    adjusted = confidence_raw * (1.0 - scr) * (1.0 - wr) * (1.0 - jr)

    # Extra penalty: "confident garbage" = no uncertainty markers but noisy output
    if umr == 0.0 and (wr > 0.05 or jr > 0.03):
        adjusted *= 0.70

    # Extra penalty for very high single_char_ratio
    if scr > 0.15:
        adjusted *= 0.60

    return max(0.05, min(0.95, adjusted))


def _line_dup_ratio(lines: list[str]) -> float:
    normalized = [re.sub(r"\s+", " ", line.lower()).strip() for line in lines if line.strip()]
    if len(normalized) < 2:
        return 0.0
    return 1.0 - (len(set(normalized)) / len(normalized))


def _garbage_ratio(text: str) -> float:
    value = str(text or "")
    if not value:
        return 1.0
    repeated_runs = sum(len(match.group(0)) - 4 for match in re.finditer(r"(.)\1{4,}", value))
    non_printable = sum(1 for ch in value if not ch.isprintable())
    weird = repeated_runs + non_printable
    return max(0.0, min(1.0, weird / max(1, len(value))))


def _script_drift_ratio(text: str, script_hint: str) -> float:
    if script_hint != "latin":
        return 0.0
    letters = re.findall(r"[A-Za-zÀ-ÿ\u0370-\u03FF\u0400-\u04FF]", text or "")
    if not letters:
        return 0.0
    cross = re.findall(r"[\u0370-\u03FF\u0400-\u04FF]", text or "")
    return len(cross) / len(letters)


def _looks_hallucinated_passage(text: str, lines: list[str]) -> bool:
    if len(text) < 900 or len(lines) < 12:
        return False
    duplicate_ratio = _line_dup_ratio(lines)
    uncertainty_ratio = (text.count("?") + text.count("[…]")) / max(1, len(text))
    words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", text.lower())
    if len(words) < 140:
        return False

    chunk_counts: dict[str, int] = {}
    for idx in range(len(words) - 5):
        chunk = " ".join(words[idx : idx + 6])
        chunk_counts[chunk] = chunk_counts.get(chunk, 0) + 1
    max_repeat = max(chunk_counts.values()) if chunk_counts else 0

    return duplicate_ratio >= 0.28 and max_repeat >= 4 and uncertainty_ratio <= 0.01


def _is_low_text_quality(text: str, lines: list[str], script_hint: str) -> bool:
    if not text.strip():
        return True
    if _garbage_ratio(text) >= 0.22:
        return True
    if _script_drift_ratio(text, script_hint) >= 0.20:
        return True
    if _line_dup_ratio(lines) >= 0.40 and len(lines) >= 8:
        return True
    return False
