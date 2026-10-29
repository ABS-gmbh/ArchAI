"""Language-agnostic OCR quality engine.

Computes per-token and per-line quality signals that work across scripts
(Latin, Arabic, CJK, mixed, unknown).  No dictionary lookups — only
statistical / character-level heuristics.

Quality labels:
  HIGH        — reliable for token-based downstream (NER, word search)
  OK          — usable with caveats; may miss some entities
  RISKY       — token search unreliable; use with fallback
  UNRELIABLE  — block token-based downstream; use vision/shape fallback

Hard gates:
  If quality_label in {RISKY, UNRELIABLE}:
    - Token-based NER/word-search MUST be disabled or run in fallback mode
    - Ligature candidate mining switches to shape/layout-driven path
  If leading_fragment_ratio >= LEADING_FRAG_HARD_LIMIT:
    - Force seam-aware retry before accepting result

This module is stateless.  All functions take strings / lists and return
dicts / labels.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from app.services.ocr_quality_config import (
    LEXICAL_CONTRAST_CLEAN,
    LEXICAL_CONTRAST_RISKY,
    LEXICAL_CONTRAST_UNRELIABLE,
    LEXICAL_PLAUSIBILITY_FLOOR,
    REPETITION_HARD_LIMIT,
    REPETITION_SOFT_LIMIT,
    CROSS_PASS_STABILITY_MIN,
    ENTROPY_HIGH_LIMIT,
    ENTROPY_LOW_LIMIT,
    GIBBERISH_HARD_LIMIT,
    GIBBERISH_SOFT_LIMIT,
    LEADING_FRAG_HARD_LIMIT,
    MENTION_ABSOLUTE_MIN,
    MENTION_MIN_PER_1K_CHARS,
    NWL_TOKEN_HARD_LIMIT,
    NON_WORDLIKE_GATE_LIMIT,
    SEAM_FRAG_HARD_LIMIT,
    UNCERTAINTY_HARD_LIMIT,
    UNCERTAINTY_RISKY_LIMIT,
)
from app.services.ocr_text_metrics import (
    _has_profile,
    char_entropy,
    detect_script_family,
    gibberish_score,
    implausibility_from_contrast,
    leading_fragment_ratio,
    lexical_contrast,
    lexical_implausibility,
    non_wordlike_score,
    rare_bigram_ratio,
    repetition_score,
    seam_fragment_ratio,
    token_length_stats,
    token_stability_score,
    trailing_fragment_ratio,
    uncertainty_density,
    vowel_ratio,
)

if TYPE_CHECKING:
    from app.services.lexicon_trust import LexicalContrast


# ═══════════════════════════════════════════════════════════════════════
# Script detection
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Character-level entropy
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Token-length distribution
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Vowel / consonant heuristics (script-aware)
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Rare bigram / improbable char transitions
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Non-wordlike token scoring (character-level)
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Leading / trailing fragment ratios
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Cross-pass stability scoring
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Uncertainty marker density
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Gibberish detection (composite score)
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Per-token quality report
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TokenQuality:
    """Quality assessment for a single token."""
    token: str
    index: int
    non_wordlike: float = 0.0
    vowel_ratio: float = 0.0
    rare_bigram: bool = False
    is_anomalous: bool = False


@dataclass
class LineQuality:
    """Quality assessment for a single line."""
    line_idx: int
    text: str
    token_count: int = 0
    gibberish_score: float = 0.0
    non_wordlike_frac: float = 0.0
    is_fragment_start: bool = False
    is_fragment_end: bool = False


# ═══════════════════════════════════════════════════════════════════════
# OCR Quality Report (main output)
# ═══════════════════════════════════════════════════════════════════════

# ── backward-compat aliases (thresholds now live in ocr_quality_config) ──
# Re-exported so that ``from ocr_quality import LEADING_FRAG_HARD_LIMIT``
# still works without touching every importer.
# The canonical values live in ``ocr_quality_config``.
CROSS_PASS_INSTABILITY_LIMIT = CROSS_PASS_STABILITY_MIN  # alias


@dataclass
class OCRQualityReport:
    """Structured OCR quality assessment for a single run/pass.

    Stored in DB (as JSON) for auditing and gating decisions.
    """
    # Identification
    run_id: str = ""
    pass_idx: int = 0    # 0=baseline, 1=overlap, 2=high-recall

    # Script detection
    script_family: str = "unknown"

    # Global signals
    char_entropy: float = 0.0
    gibberish_score: float = 0.0
    uncertainty_density: float = 0.0
    rare_bigram_ratio: float = 0.0
    repetition_score: float = 0.0
    # How unlike real language the text is, 0-1, from lexical_contrast; -1.0
    # when not measured, so "not measured" is distinguishable from "plausible".
    lexical_implausibility: float = -1.0
    # Hit rate of the text's trigrams over that of its shuffled letters, in the
    # best-fitting profiled language; -1.0 when not measured.
    lexical_contrast: float = -1.0
    lexical_language: str = ""
    # lexicon_trust.best_fit_plausibility, the hit rate token search and NER are
    # gated on; -1.0 when no profile can judge the language.
    lexical_plausibility: float = -1.0
    detected_language: str = ""
    # Whether the text was assembled from tiles. Only then can a line start or
    # end with a word cut at a seam, so only then do the fragment ratios count.
    tiled: bool = True

    # Token-level stats
    token_count: int = 0
    non_wordlike_frac: float = 0.0
    token_length_mean: float = 0.0
    token_length_stddev: float = 0.0
    very_short_token_frac: float = 0.0
    very_long_token_frac: float = 0.0

    # Fragment stats
    leading_fragment_ratio: float = 0.0
    seam_fragment_ratio: float = 0.0
    trailing_fragment_ratio: float = 0.0
    line_count: int = 0

    # Cross-pass stability (populated only after multi-pass)
    cross_pass_stability: float = -1.0   # -1 = not computed yet

    # Sanitization
    sanitized_token_count: int = 0

    # Derived label
    quality_label: str = "OK"           # HIGH | OK | RISKY | UNRELIABLE

    # Gate decisions
    token_search_allowed: bool = True
    ner_allowed: bool = True
    seam_retry_required: bool = False

    # Per-line detail (optional, for DB storage)
    line_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Don't serialize massive line_details to top-level
        if len(d.get("line_details", [])) > 100:
            d["line_details"] = d["line_details"][:100]
            d["line_details_truncated"] = True
        return d


def compute_quality_report(
    text: str,
    *,
    run_id: str = "",
    pass_idx: int = 0,
    previous_pass_tokens: Sequence[str] | None = None,
    language: str | None = None,
    tiled: bool = True,
) -> OCRQualityReport:
    """Compute a full OCR quality report for a text.

    This is the main entry point for quality assessment.

    Args:
        text: The OCR output text.
        run_id: Pipeline run identifier.
        pass_idx: Which OCR pass (0=baseline, 1=overlap, 2=high-recall).
        previous_pass_tokens: Tokens from a previous pass for stability scoring.
        language: Normalized detected language (e.g. 'latin', 'old_french').
            When it names a language with a trigram profile, the lexical
            signal is computed and dominates the gibberish score.
        tiled: Whether the text was assembled from tiles. A whole-page or
            line-by-line transcription has no seams, and its line starts and
            ends - separated initials, abbreviated words, words broken across
            lines without a hyphen - are not fragments.

    Returns:
        OCRQualityReport with all signals computed and quality_label set.
    """
    report = OCRQualityReport(run_id=run_id, pass_idx=pass_idx, tiled=tiled)

    if not text or not text.strip():
        report.quality_label = "UNRELIABLE"
        report.token_search_allowed = False
        report.ner_allowed = False
        return report

    # ── Script detection ───────────────────────────────────────────
    report.script_family = detect_script_family(text)

    # ── Tokenize ───────────────────────────────────────────────────
    lines = [ln for ln in text.splitlines() if ln.strip()]
    report.line_count = len(lines)

    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    report.token_count = len(tokens)

    # ── Character entropy ──────────────────────────────────────────
    report.char_entropy = round(char_entropy(text), 4)

    # ── Token length distribution ──────────────────────────────────
    tl_stats = token_length_stats(tokens)
    report.token_length_mean = tl_stats["mean"]
    report.token_length_stddev = tl_stats["stddev"]
    report.very_short_token_frac = tl_stats["very_short_frac"]
    report.very_long_token_frac = tl_stats["very_long_frac"]

    # ── Non-wordlike token fraction ────────────────────────────────
    if tokens:
        nwl_scores = [non_wordlike_score(t, report.script_family) for t in tokens if len(t) >= 3]
        if nwl_scores:
            report.non_wordlike_frac = round(
                sum(1 for s in nwl_scores if s > 0.45) / len(nwl_scores), 4
            )

    # ── Rare bigram ratio ──────────────────────────────────────────
    report.rare_bigram_ratio = round(rare_bigram_ratio(text, report.script_family), 4)

    # ── Repetition (decoding loops) ────────────────────────────────
    report.repetition_score = repetition_score(text)

    # ── Lexical implausibility for the detected language ───────────
    report.detected_language = str(language or "")
    contrast = lexical_contrast(text, language)
    if contrast is not None:
        report.lexical_contrast = contrast.ratio
        report.lexical_language = contrast.language
        report.lexical_implausibility = implausibility_from_contrast(contrast.ratio)
    if _has_profile(language):
        from app.services.lexicon_trust import best_fit_plausibility

        report.lexical_plausibility = round(best_fit_plausibility(text, str(language)), 4)

    # ── Gibberish score ────────────────────────────────────────────
    report.gibberish_score = gibberish_score(text, report.script_family, language)

    # ── Uncertainty density ────────────────────────────────────────
    report.uncertainty_density = round(uncertainty_density(text), 4)

    # ── Fragment ratios ────────────────────────────────────────────
    report.leading_fragment_ratio = round(
        leading_fragment_ratio(lines, report.script_family), 4
    )
    report.seam_fragment_ratio = round(
        seam_fragment_ratio(lines, script=report.script_family), 4
    )
    report.trailing_fragment_ratio = round(trailing_fragment_ratio(lines), 4)

    # ── Cross-pass stability ───────────────────────────────────────
    if previous_pass_tokens is not None:
        report.cross_pass_stability = round(
            token_stability_score(previous_pass_tokens, tokens), 4
        )

    # ── Sanitized token count ──────────────────────────────────────
    report.sanitized_token_count = sum(
        1 for t in tokens
        if "?" in t or "…" in t or "[" in t
    )

    # ── Per-line details ───────────────────────────────────────────
    lfr_lines = leading_fragment_ratio(lines, report.script_family)
    for idx, line in enumerate(lines):
        line_tokens = [t for t in re.split(r"\s+", line.strip()) if t]
        line_gs = gibberish_score(line, report.script_family) if len(line.strip()) >= 5 else 0.0
        nwl_line = 0.0
        if line_tokens:
            nwl_line_scores = [non_wordlike_score(t, report.script_family) for t in line_tokens if len(t) >= 3]
            if nwl_line_scores:
                nwl_line = sum(1 for s in nwl_line_scores if s > 0.45) / len(nwl_line_scores)

        report.line_details.append({
            "line_idx": idx,
            "text": line.strip()[:120],
            "token_count": len(line_tokens),
            "gibberish_score": round(line_gs, 4),
            "non_wordlike_frac": round(nwl_line, 4),
        })

    # ── Derive quality label ───────────────────────────────────────
    report.quality_label = _derive_quality_label(report)

    # ── Gate decisions ─────────────────────────────────────────────
    report.token_search_allowed = report.quality_label in ("HIGH", "OK")
    report.ner_allowed = report.quality_label in ("HIGH", "OK")
    # Seam retry: prefer geometry-aware seam_fragment_ratio; fall back to
    # leading_fragment_ratio only when seam_fragment_ratio is unavailable.
    report.seam_retry_required = report.tiled and (
        report.seam_fragment_ratio >= SEAM_FRAG_HARD_LIMIT
        or report.leading_fragment_ratio >= LEADING_FRAG_HARD_LIMIT
    )

    return report


def _derive_quality_label(r: OCRQualityReport) -> str:
    """Derive quality label from computed signals.

    Hierarchy:
      UNRELIABLE → RISKY → OK → HIGH
    """
    # UNRELIABLE conditions (any one triggers)
    if r.gibberish_score >= GIBBERISH_HARD_LIMIT:
        return "UNRELIABLE"
    if r.uncertainty_density >= UNCERTAINTY_HARD_LIMIT:
        return "UNRELIABLE"
    if r.cross_pass_stability >= 0 and r.cross_pass_stability < CROSS_PASS_STABILITY_MIN:
        return "UNRELIABLE"
    if r.non_wordlike_frac >= NON_WORDLIKE_GATE_LIMIT:
        return "UNRELIABLE"
    if r.char_entropy < 1.5:
        return "UNRELIABLE"
    # A decoding loop repeating one line is lexically and statistically normal;
    # only the repetition share exposes it.
    if r.repetition_score >= REPETITION_HARD_LIMIT:
        return "UNRELIABLE"
    # Only meaningful when the contrast was measured (-1.0 = not measured).
    if 0 <= r.lexical_contrast < LEXICAL_CONTRAST_UNRELIABLE:
        return "UNRELIABLE"

    # RISKY conditions
    if r.gibberish_score >= GIBBERISH_SOFT_LIMIT:
        return "RISKY"
    if r.non_wordlike_frac >= NWL_TOKEN_HARD_LIMIT:
        return "RISKY"
    # Use seam_fragment_ratio (geometry-aware) for seam-related RISKY.
    # leading_fragment_ratio alone does NOT trigger RISKY — it may be
    # caused by legitimate short function words in medieval verse. Neither
    # applies to text that was never cut into tiles.
    if r.tiled and r.seam_fragment_ratio >= SEAM_FRAG_HARD_LIMIT:
        return "RISKY"
    if r.tiled and r.leading_fragment_ratio >= LEADING_FRAG_HARD_LIMIT:
        return "RISKY"
    if r.char_entropy < ENTROPY_LOW_LIMIT or r.char_entropy > ENTROPY_HIGH_LIMIT:
        return "RISKY"
    if r.uncertainty_density >= UNCERTAINTY_RISKY_LIMIT:
        return "RISKY"
    if r.repetition_score >= REPETITION_SOFT_LIMIT:
        return "RISKY"
    if 0 <= r.lexical_contrast < LEXICAL_CONTRAST_RISKY:
        return "RISKY"
    # The hit rate token search and NER are gated on. It catches wrong letters,
    # which the contrast is slow to notice, and it is all there is for text too
    # short for the contrast.
    if 0 <= r.lexical_plausibility < LEXICAL_PLAUSIBILITY_FLOOR:
        return "RISKY"

    # HIGH conditions
    if (r.gibberish_score < 0.10
        and r.non_wordlike_frac < 0.15
        and (not r.tiled or (r.leading_fragment_ratio < 0.06 and r.seam_fragment_ratio < 0.05))
        and r.uncertainty_density < 0.03
        and 2.5 <= r.char_entropy <= 5.0
        and r.repetition_score < 0.10
        and (r.lexical_contrast < 0 or r.lexical_contrast >= LEXICAL_CONTRAST_CLEAN)):
        return "HIGH"

    return "OK"


# ═══════════════════════════════════════════════════════════════════════
# Mention recall gating
# ═══════════════════════════════════════════════════════════════════════

def check_mention_recall(
    text: str,
    mentions_total: int,
    quality_label: str,
) -> dict[str, Any]:
    """Check if mention extraction meets minimum expectations.

    Returns:
        {
            "mention_recall_ok": bool,
            "reason": str,
            "trigger_high_recall": bool,  # should we re-run with relaxed params?
        }
    """
    if quality_label not in ("HIGH", "OK"):
        return {
            "mention_recall_ok": True,
            "reason": f"quality_label={quality_label}, mention recall check skipped",
            "trigger_high_recall": False,
        }

    text_len = len(text.strip())
    if text_len < 200:
        return {
            "mention_recall_ok": True,
            "reason": "text too short for recall check",
            "trigger_high_recall": False,
        }

    expected_min = max(
        MENTION_ABSOLUTE_MIN,
        int(text_len / 1000 * MENTION_MIN_PER_1K_CHARS),
    )

    if mentions_total >= expected_min:
        return {
            "mention_recall_ok": True,
            "reason": f"mentions={mentions_total} >= expected_min={expected_min}",
            "trigger_high_recall": False,
        }

    return {
        "mention_recall_ok": False,
        "reason": (
            f"MENTION_RECALL_LOW: mentions={mentions_total} < expected_min={expected_min} "
            f"(text_len={text_len}, quality={quality_label})"
        ),
        "trigger_high_recall": True,
    }


# ═══════════════════════════════════════════════════════════════════════
# Format for report / DB
# ═══════════════════════════════════════════════════════════════════════

def format_quality_report_summary(report: OCRQualityReport) -> str:
    """Format a quality report as a human-readable summary string."""
    L = [
        "=== OCR QUALITY REPORT ===",
        f"run_id: {report.run_id}",
        f"pass_idx: {report.pass_idx}",
        f"script_family: {report.script_family}",
        f"quality_label: {report.quality_label}",
        f"",
        f"char_entropy: {report.char_entropy:.4f}",
        f"gibberish_score: {report.gibberish_score:.4f}",
        f"non_wordlike_frac: {report.non_wordlike_frac:.4f}",
        f"rare_bigram_ratio: {report.rare_bigram_ratio:.4f}",
        f"uncertainty_density: {report.uncertainty_density:.4f}",
        f"",
        f"token_count: {report.token_count}",
        f"token_length_mean: {report.token_length_mean:.2f}",
        f"token_length_stddev: {report.token_length_stddev:.2f}",
        f"very_short_token_frac: {report.very_short_token_frac:.3f}",
        f"very_long_token_frac: {report.very_long_token_frac:.3f}",
        f"",
        f"leading_fragment_ratio: {report.leading_fragment_ratio:.4f}",
        f"seam_fragment_ratio: {report.seam_fragment_ratio:.4f}",
        f"trailing_fragment_ratio: {report.trailing_fragment_ratio:.4f}",
        f"line_count: {report.line_count}",
        f"",
        f"cross_pass_stability: {report.cross_pass_stability:.4f}",
        f"sanitized_token_count: {report.sanitized_token_count}",
        f"",
        f"=== GATE DECISIONS ===",
        f"token_search_allowed: {report.token_search_allowed}",
        f"ner_allowed: {report.ner_allowed}",
        f"seam_retry_required: {report.seam_retry_required}",
    ]
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════
# Unified EffectiveQuality — single source of truth
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EffectiveQuality:
    """Single unified quality object used by UI, DB, and gate enforcement.

    This eliminates the mismatch between v1 ``quality_label`` (sanity-based)
    and v2 ``quality_label_v2`` (hardened) by providing ONE authoritative
    source of truth.

    Fields:
      label         — "HIGH" | "OK" | "RISKY" | "UNRELIABLE"
      downstream    — "token_based" | "vision_fallback"
      confidence    — 0..1 overall OCR confidence
      ner_allowed   — can downstream NER/chunking run?
      token_search_allowed — can downstream word-search/linking run?
      seam_retry_required — should we retry with shifted tiles?
      metrics       — dict of numeric quality signals
    """
    label: str = "OK"
    downstream: str = "token_based"
    confidence: float = 0.0
    ner_allowed: bool = True
    token_search_allowed: bool = True
    seam_retry_required: bool = False
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_effective_quality(
    report: OCRQualityReport,
    gate_decisions: dict[str, Any],
    confidence: float = 0.0,
) -> EffectiveQuality:
    """Build a single EffectiveQuality from a quality report + gate decisions.

    This is the canonical way to produce the quality object that flows
    through UI, DB, and gating.
    """
    return EffectiveQuality(
        label=report.quality_label,
        downstream=gate_decisions.get("downstream_mode", "vision_fallback"),
        confidence=confidence,
        ner_allowed=gate_decisions.get("ner_allowed", report.ner_allowed),
        token_search_allowed=gate_decisions.get(
            "token_search_allowed", report.token_search_allowed,
        ),
        seam_retry_required=gate_decisions.get(
            "seam_retry_required", report.seam_retry_required,
        ),
        metrics={
            "gibberish_score": report.gibberish_score,
            "non_wordlike_frac": report.non_wordlike_frac,
            "leading_fragment_ratio": report.leading_fragment_ratio,
            "seam_fragment_ratio": report.seam_fragment_ratio,
            "char_entropy": report.char_entropy,
            "uncertainty_density": report.uncertainty_density,
            "cross_pass_stability": report.cross_pass_stability,
            "trailing_fragment_ratio": report.trailing_fragment_ratio,
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# Conservative uncertainty enforcement
# ═══════════════════════════════════════════════════════════════════════

_SUSPICIOUS_HYPHEN_RE = re.compile(
    r"(?<=[a-z])(?=[A-Z])-|"          # mixed-case around hyphen
    r"-(?=[A-Z][a-z]{0,2}$)|"          # hyphen before very short suffix
    r"(?<=[A-Z])-(?=[a-z])",           # upper-lower boundary
    re.UNICODE,
)

_UNLIKELY_BIGRAMS_SET = frozenset({
    "zx", "xz", "qx", "xq", "jx", "xj", "zq", "qz",
    "bx", "xb", "vx", "xv", "kx", "xk", "wx", "xw",
    "jq", "qj", "jz", "zj", "bk", "kb", "fq", "qf",
    "vq", "qv",
})


def _token_has_unlikely_bigrams(token: str) -> bool:
    low = token.lower()
    for i in range(len(low) - 1):
        if low[i:i + 2] in _UNLIKELY_BIGRAMS_SET:
            return True
    return False


def _looks_like_hallucinated_hyphen(token: str, is_unstable: bool) -> bool:
    """Conservative: only flag when multiple hallucination signals coincide."""
    if "-" not in token:
        return False
    parts = token.split("-")
    if len(parts) != 2:
        return False
    left, right = parts
    if not left or not right:
        return False
    # Mixed case + hyphen + short suffix + instability
    has_mixed_case = (left[-1:].islower() and right[0:1].isupper()) or \
                     (left[-1:].isupper() and right[0:1].islower())
    has_unlikely = _token_has_unlikely_bigrams(left + right)
    has_short_part = len(left) <= 2 or len(right) <= 2
    enough_signals = sum([has_mixed_case, has_unlikely, has_short_part, is_unstable])
    return enough_signals >= 3


def apply_uncertainty_markers(
    text: str,
    cross_pass_text: str | None = None,
    cross_pass_stability: float = -1.0,
    frag_gate_val: float = 0.0,
    uncertainty_dens: float = 0.0,
) -> tuple[str, int]:
    """Apply conservative uncertainty markers to unstable/suspicious tokens.

    This function NEVER corrects text into plausible words — it only
    downgrades uncertain tokens to ``?`` (character) or ``[…]`` (span).

    Returns:
        (processed_text, markers_inserted_count)
    """
    from app.services.ocr_quality_config import (
        UNCERTAINTY_ENFORCEMENT_STABILITY_THRESHOLD,
        UNCERTAINTY_ENFORCEMENT_FRAG_THRESHOLD,
        UNCERTAINTY_ENFORCEMENT_DENSITY_THRESHOLD,
    )

    # Decide whether enforcement is needed
    should_enforce = False
    if 0 <= cross_pass_stability < UNCERTAINTY_ENFORCEMENT_STABILITY_THRESHOLD:
        should_enforce = True
    if frag_gate_val >= UNCERTAINTY_ENFORCEMENT_FRAG_THRESHOLD:
        should_enforce = True
    if uncertainty_dens >= UNCERTAINTY_ENFORCEMENT_DENSITY_THRESHOLD:
        should_enforce = True

    if not should_enforce or not text.strip():
        return text, 0
    if cross_pass_text is None:
        return text, 0

    # Build set of unstable tokens by comparing the two passes line-by-line
    lines_a = text.splitlines()
    lines_b = cross_pass_text.splitlines()
    # Build unstable-token set from differing lines
    unstable_tokens: set[str] = set()
    max_lines = max(len(lines_a), len(lines_b))
    for i in range(max_lines):
        la = lines_a[i].strip() if i < len(lines_a) else ""
        lb = lines_b[i].strip() if i < len(lines_b) else ""
        if la == lb:
            continue
        toks_a = set(la.split())
        toks_b = set(lb.split())
        # Tokens that appear in one pass but not the other are unstable
        unstable_tokens |= toks_a.symmetric_difference(toks_b)

    if not unstable_tokens:
        return text, 0

    # Apply markers - replace unstable tokens in-place
    markers_count = 0
    result_lines: list[str] = []
    for line in lines_a:
        tokens = line.split()
        new_tokens: list[str] = []
        for tok in tokens:
            if tok in unstable_tokens:
                # Check for hallucinated hyphen
                is_unstable = True
                if _looks_like_hallucinated_hyphen(tok, is_unstable):
                    new_tokens.append("[…]")
                    markers_count += 1
                elif _token_has_unlikely_bigrams(tok) and non_wordlike_score(tok) > 0.45:
                    # Non-wordlike + unlikely bigrams → replace chars
                    replaced = "".join(
                        "?" if c.isalpha() and non_wordlike_score(c, "latin") > 0.3 else c
                        for c in tok
                    )
                    if replaced != tok:
                        new_tokens.append(replaced)
                        markers_count += 1
                    else:
                        new_tokens.append("[…]")
                        markers_count += 1
                elif non_wordlike_score(tok) > 0.55:
                    new_tokens.append("[…]")
                    markers_count += 1
                else:
                    # Token differs across passes but isn't clearly garbage →
                    # leave it, don't hallucinate a correction
                    new_tokens.append(tok)
            else:
                new_tokens.append(tok)
        result_lines.append(" ".join(new_tokens))

    return "\n".join(result_lines), markers_count
