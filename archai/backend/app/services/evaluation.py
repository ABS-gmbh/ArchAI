"""Accuracy evaluation for OCR transcriptions and evidence retrieval.

Nothing in this repository could previously measure accuracy: there was no
ground truth, no CER or WER implementation and no retrieval metric. Every
accuracy-affecting choice - which recogniser, which preprocessing, which chunk
geometry - was therefore unfalsifiable.

This module is deliberately dependency-free beyond numpy, and deliberately
explicit about the choices that silently change reported numbers:

* **Normalisation.** CER moves by several points depending on whether Unicode
  forms, case and whitespace are folded before comparison. The policy is a value
  the caller passes and the report records, never an implicit default.
* **Micro versus macro aggregation.** Summing errors over the corpus weights long
  pages more; averaging per-page rates weights every page equally. They answer
  different questions and are frequently conflated, so both are reported.
* **Uncertainty.** With the handful of gold pages a project like this can afford,
  a point estimate is close to meaningless. Rates come with bootstrap confidence
  intervals, resampled over pages - the unit that was actually sampled.

CER is edits divided by REFERENCE length, and so exceeds 1.0 when the hypothesis
inserts more than the reference contains. That is correct, not a bug: clamping it
hides the hallucinated-text failure mode.
"""

from __future__ import annotations

import math
import random
import re
import unicodedata
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Normalisation
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class NormalizationPolicy:
    """How reference and hypothesis are folded before they are compared.

    The default is the conservative *diplomatic* policy: canonical Unicode
    composition and whitespace collapsing only. Case, punctuation and scribal
    abbreviation marks are all meaningful in a diplomatic transcription and are
    kept. Relax them explicitly when evaluating a normalised edition.
    """

    unicode_form: str | None = "NFC"
    casefold: bool = False
    strip_punctuation: bool = False
    collapse_whitespace: bool = True

    def describe(self) -> str:
        parts = [self.unicode_form or "no-unicode-normalisation"]
        if self.casefold:
            parts.append("casefold")
        if self.strip_punctuation:
            parts.append("strip-punctuation")
        if self.collapse_whitespace:
            parts.append("collapse-whitespace")
        return "+".join(parts)


DIPLOMATIC = NormalizationPolicy()
"""Compare what the scribe wrote: composition and whitespace only."""

LENIENT = NormalizationPolicy(casefold=True, strip_punctuation=True)
"""Compare content words: also fold case and drop punctuation."""

_WHITESPACE_RE = re.compile(r"\s+")

# Unicode files these under punctuation (Po), but in a medieval transcription each
# one stands for the word "et". Stripping them would delete words from both sides
# and make the comparison blind to errors in them.
_WORD_SIGNS = frozenset({"\u204a", "\u2e52", "&"})  # ⁊  ⹒  &


def _strip_punctuation(value: str) -> str:
    # By Unicode category rather than ``[^\w\s]``: that pattern also matches
    # combining marks (Mn), so the macron in an abbreviation such as "dn̄s" would
    # become a space and split one word into two.
    return "".join(
        " " if unicodedata.category(ch).startswith("P") and ch not in _WORD_SIGNS else ch
        for ch in value
    )


def normalize_for_eval(text: str, policy: NormalizationPolicy = DIPLOMATIC) -> str:
    value = str(text or "")
    if policy.unicode_form:
        value = unicodedata.normalize(policy.unicode_form, value)
    if policy.casefold:
        value = value.casefold()
    if policy.strip_punctuation:
        value = _strip_punctuation(value)
    if policy.collapse_whitespace:
        value = _WHITESPACE_RE.sub(" ", value).strip()
    return value


# ═══════════════════════════════════════════════════════════════════════
# Edit distance
# ═══════════════════════════════════════════════════════════════════════


def _encode(ref: Sequence[Hashable], hyp: Sequence[Hashable]) -> tuple[np.ndarray, np.ndarray]:
    """Map both sequences onto one shared integer vocabulary."""
    vocab: dict[Hashable, int] = {}
    ref_ids = np.fromiter((vocab.setdefault(t, len(vocab)) for t in ref), dtype=np.int64, count=len(ref))
    hyp_ids = np.fromiter((vocab.setdefault(t, len(vocab)) for t in hyp), dtype=np.int64, count=len(hyp))
    return ref_ids, hyp_ids


def _next_row(prev: np.ndarray, token: int, hyp_ids: np.ndarray, row_index: int, idx: np.ndarray) -> np.ndarray:
    """One row of the Levenshtein recurrence, vectorised.

    D[i][j] = min(D[i-1][j] + 1, D[i][j-1] + 1, D[i-1][j-1] + cost)

    The insertion term D[i][j-1] + 1 depends on the row being built, which is
    what stops a naive vectorisation. Unrolling it gives
    D[i][j] = min over l <= j of (T[l] + (j - l)), where T holds the other two
    terms; that is a running minimum of T - j, shifted back by j.
    """
    cost = (hyp_ids != token).astype(np.int64)
    partial = np.empty_like(prev)
    partial[0] = row_index
    partial[1:] = np.minimum(prev[1:] + 1, prev[:-1] + cost)
    return np.minimum.accumulate(partial - idx) + idx


def edit_distance(ref: Sequence[Hashable], hyp: Sequence[Hashable]) -> int:
    """Levenshtein distance between two token sequences, in O(len(hyp)) memory."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    ref_ids, hyp_ids = _encode(ref, hyp)
    idx = np.arange(len(hyp_ids) + 1, dtype=np.int64)
    row = idx.copy()
    for i, token in enumerate(ref_ids, start=1):
        row = _next_row(row, int(token), hyp_ids, i, idx)
    return int(row[-1])


# Above this many DP cells a full alignment matrix is not materialised; distance
# is still exact, but the substitution/deletion/insertion split is omitted.
_MAX_ALIGNMENT_CELLS = 25_000_000


@dataclass(frozen=True)
class ErrorCounts:
    """Edits needed to turn a hypothesis into its reference."""

    distance: int
    reference_length: int
    hypothesis_length: int
    substitutions: int | None = None
    deletions: int | None = None
    insertions: int | None = None

    @property
    def rate(self) -> float:
        """Edits per reference token. May exceed 1.0; see the module docstring."""
        if self.reference_length == 0:
            return 0.0 if self.hypothesis_length == 0 else math.inf
        return self.distance / self.reference_length

    @property
    def has_breakdown(self) -> bool:
        return self.substitutions is not None


def align(ref: Sequence[Hashable], hyp: Sequence[Hashable]) -> ErrorCounts:
    """Edit distance plus its substitution / deletion / insertion breakdown.

    Ties between equally short edit paths are broken towards substitution, then
    deletion, then insertion, the convention used by standard scoring tools, so
    the split is deterministic.
    """
    n, m = len(ref), len(hyp)
    if n == 0 or m == 0:
        return ErrorCounts(max(n, m), n, m, 0, n, m)
    if (n + 1) * (m + 1) > _MAX_ALIGNMENT_CELLS:
        return ErrorCounts(edit_distance(ref, hyp), n, m)

    ref_ids, hyp_ids = _encode(ref, hyp)
    idx = np.arange(m + 1, dtype=np.int64)
    # int32 halves the footprint of the one allocation that grows as n * m.
    matrix = np.empty((n + 1, m + 1), dtype=np.int32)
    matrix[0] = idx
    for i in range(1, n + 1):
        matrix[i] = _next_row(matrix[i - 1], int(ref_ids[i - 1]), hyp_ids, i, idx)

    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        here = matrix[i, j]
        if i > 0 and j > 0:
            same = ref_ids[i - 1] == hyp_ids[j - 1]
            diagonal = matrix[i - 1, j - 1] + (0 if same else 1)
            if here == diagonal:
                if not same:
                    subs += 1
                i, j = i - 1, j - 1
                continue
        if i > 0 and here == matrix[i - 1, j] + 1:
            dels += 1
            i -= 1
            continue
        ins += 1
        j -= 1

    return ErrorCounts(int(matrix[n, m]), n, m, subs, dels, ins)


def alignment_offsets(ref: Sequence[Hashable], hyp: Sequence[Hashable]) -> np.ndarray:
    """Map every offset of *ref*, 0 through len(ref), to its aligned offset in *hyp*.

    Follows the same minimum-edit alignment as :func:`align`, so a span [s, e)
    marked on one transcription can be found in another as [out[s], out[e]):
    how a passage is located in a second OCR reading of the same page. Offsets
    never decrease; where *hyp* inserts text between two offsets, the later
    offset maps past the insertion.
    """
    n, m = len(ref), len(hyp)
    if (n + 1) * (m + 1) > _MAX_ALIGNMENT_CELLS:
        raise ValueError(f"sequences of length {n} and {m} are too long to align in memory")
    ref_ids, hyp_ids = _encode(ref, hyp)
    idx = np.arange(m + 1, dtype=np.int64)
    matrix = np.empty((n + 1, m + 1), dtype=np.int32)
    matrix[0] = idx
    for i in range(1, n + 1):
        matrix[i] = _next_row(matrix[i - 1], int(ref_ids[i - 1]), hyp_ids, i, idx)

    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[n] = m
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and matrix[i, j] == matrix[i - 1, j - 1] + (0 if ref_ids[i - 1] == hyp_ids[j - 1] else 1):
            i, j = i - 1, j - 1
        elif i > 0 and matrix[i, j] == matrix[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
            continue
        offsets[i] = j
    return offsets


def _characters(text: str, policy: NormalizationPolicy) -> list[str]:
    return list(normalize_for_eval(text, policy))


def _words(text: str, policy: NormalizationPolicy) -> list[str]:
    return normalize_for_eval(text, policy).split()


def cer(reference: str, hypothesis: str, policy: NormalizationPolicy = DIPLOMATIC) -> ErrorCounts:
    """Character error counts for one transcription."""
    return align(_characters(reference, policy), _characters(hypothesis, policy))


def wer(reference: str, hypothesis: str, policy: NormalizationPolicy = DIPLOMATIC) -> ErrorCounts:
    """Word error counts for one transcription."""
    return align(_words(reference, policy), _words(hypothesis, policy))


# ═══════════════════════════════════════════════════════════════════════
# Uncertainty
# ═══════════════════════════════════════════════════════════════════════


def bootstrap_ci(
    samples: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for *statistic* over *samples*.

    Resampling is over whatever the caller passes, which should be the unit that
    was actually sampled - pages, for OCR. Resampling characters instead would
    treat thousands of highly correlated errors on one page as independent
    evidence and produce an interval that is far too narrow.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    # One sample has no sampling variance to estimate. Returning (value, value)
    # printed as a zero-width interval, implying perfect certainty from a single
    # page - the false precision this function exists to prevent. Say "unknown".
    if len(samples) < 2:
        return (math.nan, math.nan)

    rng = random.Random(seed)
    population = list(samples)
    estimates = sorted(
        statistic([population[rng.randrange(len(population))] for _ in population])
        for _ in range(max(1, n_resamples))
    )
    tail = (1.0 - confidence) / 2.0
    low = estimates[int(math.floor(tail * (len(estimates) - 1)))]
    high = estimates[int(math.ceil((1.0 - tail) * (len(estimates) - 1)))]
    return (low, high)


# ═══════════════════════════════════════════════════════════════════════
# Corpus-level OCR accuracy
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PageResult:
    page_id: str
    char: ErrorCounts
    word: ErrorCounts


@dataclass(frozen=True)
class RateSummary:
    micro: float
    macro: float
    micro_ci: tuple[float, float]
    macro_ci: tuple[float, float]


@dataclass
class CorpusReport:
    pages: list[PageResult]
    policy: NormalizationPolicy
    cer: RateSummary
    wer: RateSummary
    confidence: float
    char_breakdown: dict[str, int | None] = field(default_factory=dict)

    def worst_pages(self, n: int = 5) -> list[PageResult]:
        return sorted(self.pages, key=lambda p: p.char.rate, reverse=True)[:n]

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary: non-finite values become ``None``, since NaN and
        Infinity are not valid JSON and break strict parsers such as jq."""

        def summary(rate: RateSummary) -> dict[str, Any]:
            return {
                "micro": _finite(rate.micro),
                "macro": _finite(rate.macro),
                "micro_ci": [_finite(v) for v in rate.micro_ci],
                "macro_ci": [_finite(v) for v in rate.macro_ci],
            }

        return {
            "pages": len(self.pages),
            "normalization": self.policy.describe(),
            "confidence": self.confidence,
            "cer": summary(self.cer),
            "wer": summary(self.wer),
            "char_breakdown": self.char_breakdown,
            "per_page": [
                {
                    "page_id": p.page_id,
                    "cer": _finite(p.char.rate),
                    "wer": _finite(p.word.rate),
                    "char_distance": p.char.distance,
                    "reference_chars": p.char.reference_length,
                }
                for p in self.pages
            ],
        }


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _micro(counts: Sequence[ErrorCounts]) -> float:
    reference = sum(c.reference_length for c in counts)
    if reference == 0:
        return 0.0
    return sum(c.distance for c in counts) / reference


def _macro(counts: Sequence[ErrorCounts]) -> float:
    # A page with an empty reference has no finite rate, so it cannot enter a
    # mean of rates. Its insertions still count towards the micro rate.
    rates = [c.rate for c in counts if math.isfinite(c.rate)]
    return sum(rates) / len(rates) if rates else 0.0


def _summarise(counts: Sequence[ErrorCounts], *, confidence: float, seed: int) -> RateSummary:
    return RateSummary(
        micro=_micro(counts),
        macro=_macro(counts),
        micro_ci=bootstrap_ci(counts, _micro, confidence=confidence, seed=seed),
        macro_ci=bootstrap_ci(counts, _macro, confidence=confidence, seed=seed),
    )


def evaluate_corpus(
    pairs: Iterable[tuple[str, str, str]],
    *,
    policy: NormalizationPolicy = DIPLOMATIC,
    confidence: float = 0.95,
    seed: int = 0,
) -> CorpusReport:
    """Score (page_id, reference, hypothesis) triples at page and corpus level."""
    pages = [
        PageResult(page_id, cer(reference, hypothesis, policy), wer(reference, hypothesis, policy))
        for page_id, reference, hypothesis in pairs
    ]
    chars = [p.char for p in pages]
    words = [p.word for p in pages]

    breakdown: dict[str, int | None]
    if pages and all(c.has_breakdown for c in chars):
        breakdown = {
            "substitutions": sum(c.substitutions or 0 for c in chars),
            "deletions": sum(c.deletions or 0 for c in chars),
            "insertions": sum(c.insertions or 0 for c in chars),
        }
    else:
        breakdown = {"substitutions": None, "deletions": None, "insertions": None}

    return CorpusReport(
        pages=pages,
        policy=policy,
        cer=_summarise(chars, confidence=confidence, seed=seed),
        wer=_summarise(words, confidence=confidence, seed=seed),
        confidence=confidence,
        char_breakdown=breakdown,
    )


# ═══════════════════════════════════════════════════════════════════════
# Retrieval accuracy
# ═══════════════════════════════════════════════════════════════════════


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")


def recall_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """Share of the relevant items that appear in the first *k* results."""
    _validate_k(k)
    wanted = set(relevant)
    if not wanted:
        return 0.0
    return len(wanted.intersection(ranked[:k])) / len(wanted)


def precision_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """Share of the first *k* results that are relevant.

    Divides by k, not by however many results came back: a system that returns
    two perfect results when asked for five has not achieved precision 1.0.
    """
    _validate_k(k)
    wanted = set(relevant)
    return sum(1 for item in ranked[:k] if item in wanted) / k


def reciprocal_rank(ranked: Sequence[Hashable], relevant: Iterable[Hashable]) -> float:
    """1 / rank of the first relevant result, or 0 when none is returned."""
    wanted = set(relevant)
    for position, item in enumerate(ranked, start=1):
        if item in wanted:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[Hashable], gains: Mapping[Hashable, float], k: int) -> float:
    """Normalised discounted cumulative gain with graded relevance.

    *gains* maps an item to its relevance grade; unlisted items score zero. The
    ideal ordering is taken over every judged item, not just the retrieved ones,
    so a system cannot score well by retrieving few items.
    """
    _validate_k(k)

    def dcg(grades: Iterable[float]) -> float:
        return sum(g / math.log2(rank + 1) for rank, g in enumerate(grades, start=1))

    actual = dcg(float(gains.get(item, 0.0)) for item in ranked[:k])
    ideal = dcg(sorted((float(g) for g in gains.values() if g > 0), reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


@dataclass(frozen=True)
class RetrievalQuery:
    query_id: str
    ranked: tuple[Hashable, ...]
    relevant: frozenset[Hashable]
    gains: Mapping[Hashable, float] | None = None


def evaluate_retrieval(
    queries: Sequence[RetrievalQuery],
    *,
    ks: Sequence[int] = (1, 5, 10),
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Mean recall@k, precision@k, nDCG@k and MRR over a query set, with CIs.

    Queries with no relevant items are excluded rather than scored as zero:
    they measure the judgement set, not the system.
    """
    judged = [q for q in queries if q.relevant]

    def with_ci(values: Sequence[float]) -> dict[str, Any]:
        # NaN, not 0.0, when nothing was judged: zero would read as a system that
        # retrieved nothing relevant.
        mean = sum(values) / len(values) if values else math.nan
        low, high = bootstrap_ci(values, lambda xs: sum(xs) / len(xs), confidence=confidence, seed=seed)
        return {"mean": mean, "ci": [low, high], "n": len(values)}

    report: dict[str, dict[str, Any]] = {
        "mrr": with_ci([reciprocal_rank(q.ranked, q.relevant) for q in judged]),
    }
    for k in ks:
        report[f"recall@{k}"] = with_ci([recall_at_k(q.ranked, q.relevant, k) for q in judged])
        report[f"precision@{k}"] = with_ci([precision_at_k(q.ranked, q.relevant, k) for q in judged])
        report[f"ndcg@{k}"] = with_ci(
            [ndcg_at_k(q.ranked, q.gains or {item: 1.0 for item in q.relevant}, k) for q in judged]
        )
    report["_meta"] = {"queries": len(queries), "judged": len(judged), "confidence": confidence}
    return report
