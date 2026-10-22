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
"""Share of the page's lines that repeat an earlier line, or of its tokens taken
up by its most repeated n-gram, above which the transcription is UNRELIABLE. A
decoding loop that emits the same lines dozens of times is otherwise
indistinguishable from clean text to character-level signals."""

REPETITION_SOFT_LIMIT: float = 0.20
"""Above this repetition share the transcription is RISKY."""

REPETITION_NGRAM: int = 5
"""Token n-gram width used by the repetition detector."""


# ═══════════════════════════════════════════════════════════════════════
# Lexical implausibility (language-aware garbage detection)
# ═══════════════════════════════════════════════════════════════════════

LEXICAL_CONTRAST_UNRELIABLE: float = 2.0
"""Contrast below which a transcription is UNRELIABLE.

lexical_contrast is the share of a text's trigrams that are common trigrams of
its best-fitting language, over the same share for the text with each word's
letters shuffled. It replaced a hit rate against the profile of the detected
language, with fixed thresholds, which blocked three of the five reference
transcriptions, the CATMuS readings of four of their pages, and 71 of the 72
held-out transcriptions below. Calibrated on two sets
(scripts/calibrate_quality_gate.py):

- the five reference transcriptions, every recogniser's reading of them, and
  synthetic corruptions of each, with their CER;
- 72 distinct transcriptions from a pipeline database, held out, with
  corrupted versions of each.

The references long enough to measure read 3.1-4.7 and their CATMuS readings
2.9-4.5. The same pages with each word reversed read 1.3-1.9, in random
letters 0.9-1.2; the held-out texts reversed 1.0-1.7 and in random letters
0.8-1.8, every one of them below this threshold."""

LEXICAL_CONTRAST_RISKY: float = 2.2
"""Contrast below which a transcription is RISKY.

Synthetic noise up to 23% CER stayed above 2.6 on every page long enough to
measure. At 35-38% CER two of those four pages fell below 2.2, and the
plausibility floor caught the other two. Shorter text varies more; see
LEXICAL_PLAUSIBILITY_FLOOR for what the two together block by length."""

LEXICAL_CONTRAST_CLEAN: float = 3.0
"""Contrast at which a transcription counts as fully language-like: a HIGH label
needs it, and lexical implausibility is 0 from here up."""

LEXICAL_PLAUSIBILITY_FLOOR: float = 0.25
"""Best-fit lexical plausibility below which a transcription is RISKY, and token
search and NER are blocked.

lexical_plausibility is the trigram hit rate, scaled: the floor is a hit rate
of 14%, where faithful pages hit 20% (Latin) to 33% (French). It catches wrong
letters, which the contrast is slow to notice, and it is all there is for text
too short for the contrast. With the contrast thresholds, the gate blocked
these shares of 360 windows of the references and their CATMuS readings, and
of corrupted copies of the same windows:

    trigrams  windows  faithful  7% CER  21% CER  37% CER  reversed  random
    30-149      259      1.2%     5.4%     38%      89%      87%       99%
    150-299      91      1.1%     5.5%     44%      93%      93%       99%
    300+         10      0%       0%       50%     100%     100%      100%
"""


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
