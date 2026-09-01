"""Single source-of-truth for OCR quality thresholds and retry parameters.

Every gate, label-derivation, and logging line MUST import from here.
No threshold literals should appear elsewhere in the codebase.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════
# Retry / attempt budget
# ═══════════════════════════════════════════════════════════════════════

MAX_OCR_ATTEMPTS: int = 3  # attempt 0 = default, 1-2 = seam strategies


# ═══════════════════════════════════════════════════════════════════════
# Fragment thresholds (seam + leading)
# ═══════════════════════════════════════════════════════════════════════

LEADING_FRAG_HARD_LIMIT: float = 0.15
"""Above this leading-fragment ratio, force seam retry."""

SEAM_FRAG_HARD_LIMIT: float = 0.10
"""Above this seam-fragment ratio, force seam retry."""


def frag_gate_value(leading_frag: float, seam_frag: float) -> float:
    """Single number used by the LEADING_FRAGMENT gate.

    Always ``max(leading_fragment_ratio, seam_fragment_ratio)`` so the
    stronger signal wins regardless of which metric detected the problem.
    """
    return max(leading_frag, seam_frag)


# ═══════════════════════════════════════════════════════════════════════
# Other quality thresholds
# ═══════════════════════════════════════════════════════════════════════

GIBBERISH_HARD_LIMIT: float = 0.40   # above -> UNRELIABLE
GIBBERISH_SOFT_LIMIT: float = 0.25   # above -> RISKY
NWL_TOKEN_HARD_LIMIT: float = 0.35   # non-wordlike token fraction
NON_WORDLIKE_GATE_LIMIT: float = 0.55  # gate fails above
CROSS_PASS_STABILITY_MIN: float = 0.55  # below -> UNRELIABLE/gate fail
ENTROPY_LOW_LIMIT: float = 2.0
ENTROPY_HIGH_LIMIT: float = 5.5
UNCERTAINTY_HARD_LIMIT: float = 0.15
UNCERTAINTY_RISKY_LIMIT: float = 0.08  # above -> RISKY

# ═══════════════════════════════════════════════════════════════════════
# Repetition (degenerate VLM decoding loops)
# ═══════════════════════════════════════════════════════════════════════

REPETITION_HARD_LIMIT: float = 0.35
"""Share of the page occupied by one repeated line/n-gram, above which the
transcription is UNRELIABLE. A decoding loop that emits the same line dozens of
times is otherwise indistinguishable from clean text to character-level signals."""

REPETITION_SOFT_LIMIT: float = 0.20
"""Above this repetition share the transcription is RISKY."""

REPETITION_NGRAM: int = 5
"""Token n-gram width used by the repetition detector."""


# ═══════════════════════════════════════════════════════════════════════
# Lexical implausibility (language-aware garbage detection)
# ═══════════════════════════════════════════════════════════════════════

LEXICON_CLEAN_FLOOR: float = 0.45
"""Implausibility that genuinely clean text is expected to reach anyway.

lexical_plausibility scores a trigram hit-rate against a hand-built profile, and
profile coverage varies by language: clean Latin scores ~0.82 plausible but clean
Old French only ~0.57, purely because the Old French profile is sparser. Feeding
raw implausibility into the gibberish score therefore penalises clean text in
under-profiled languages. Only implausibility *above* this floor is treated as
evidence of garbage, and it is rescaled across the remaining range."""


LEXICON_UNRELIABLE_LIMIT: float = 0.72
"""Implausibility (1 - lexical_plausibility) above which text is UNRELIABLE.
Only applied when a trigram profile exists for the detected language."""

LEXICON_RISKY_LIMIT: float = 0.58
"""Implausibility above which text is RISKY."""


# ═══════════════════════════════════════════════════════════════════════
# gibberish_score component weights
# ═══════════════════════════════════════════════════════════════════════

# With a known language the lexical signal dominates, because the character
# heuristics alone cannot separate real words from transposed ones.
GIBBERISH_WEIGHTS_WITH_LANGUAGE: dict[str, float] = {
    "lexical": 0.45,
    "repetition": 0.20,
    "non_wordlike": 0.15,
    "entropy": 0.07,
    "rare_bigram": 0.07,
    "uncertainty": 0.06,
}

# Without a language profile the lexical term is unavailable; its weight is
# redistributed onto the remaining signals.
GIBBERISH_WEIGHTS_NO_LANGUAGE: dict[str, float] = {
    "repetition": 0.30,
    "non_wordlike": 0.35,
    "entropy": 0.13,
    "rare_bigram": 0.12,
    "uncertainty": 0.10,
}


# Mention recall
MENTION_MIN_PER_1K_CHARS: int = 2
MENTION_ABSOLUTE_MIN: int = 1


# ═══════════════════════════════════════════════════════════════════════
# Cross-pass stability helper thresholds
# ═══════════════════════════════════════════════════════════════════════

CROSS_PASS_PERTURBATION_OVERLAP_EXTRA: float = 0.10
"""Extra overlap fraction added for the stability verification pass."""

CROSS_PASS_GRID_SHIFT_FRAC: float = 0.08
"""Y-offset fraction used in the stability verification grid-shift."""


# ═══════════════════════════════════════════════════════════════════════
# Uncertainty enforcement
# ═══════════════════════════════════════════════════════════════════════

UNCERTAINTY_ENFORCEMENT_STABILITY_THRESHOLD: float = 0.70
"""If cross_pass_stability is below this, run uncertainty enforcement."""

UNCERTAINTY_ENFORCEMENT_FRAG_THRESHOLD: float = 0.10
"""If frag_gate_value is above this, run uncertainty enforcement."""

UNCERTAINTY_ENFORCEMENT_DENSITY_THRESHOLD: float = 0.08
"""If uncertainty_density is above this, run uncertainty enforcement."""

HALLUCINATED_HYPHEN_MIN_INSTABILITY: float = 0.40
"""Minimum instability for a hyphenated token to be replaced."""
