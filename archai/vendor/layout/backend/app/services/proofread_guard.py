"""Keep a proofreader's corrections and revert what it invented, line by line.

A language-model proofreader that recognises a passage can write it out from
memory: the result is fluent, grades better than the OCR it replaces, and is
not what the page says. The page-level check this module complements
(``check_proofread_delta``) compares whole texts against fixed ratios, and on
real readings those ratios sat inside the recognisers' own disagreement: half a
page of invented lines passed.

Here the proofread text is aligned to the OCR line by line - a line may be
joined to the next, as a set-apart initial is, or split - and each line is
judged on its own:

* a line changed by more than ``LINE_REWRITE_LIMIT`` of its characters was
  rewritten, not corrected;
* a word unlike the word it replaces (``WORD_INVENTION_LIMIT``) was invented;
* an uncertainty mark ([…], [?], ?) that disappears means a gap was filled.

Such lines revert to the OCR; every other change is kept. Lines the
proofreader added are dropped and lines it deleted are restored.

Calibrated with scripts/calibrate_proofread_guard.py on the reference pages
of eval/quality_gate/readings.json: each recogniser's line was "proofread"
into its reference line (a perfect correction), into the reference with one
word replaced by a word from another page, and into a line of another page.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

LINE_REWRITE_LIMIT = 0.5
"""Share of a line's characters a correction may change.

Turning each recogniser's line into its reference line - a perfect correction -
changed more than half of it on none of the CATMuS lines, 3% of the CREMMA
lines and 10% of the McCATMuS lines. A line of another page differed from the
reading it replaced by at least 0.64."""

WORD_INVENTION_LIMIT = 0.6
"""Edit distance, as a share of the longer word, above which a changed word is
unlike the word it replaces.

With both limits, ``judge_line`` kept 94.8% of the perfect corrections of
CATMuS lines and 84.0% of CREMMA lines, and reverted 96.6-98.9% of lines with
one word replaced by a word of another page, every swapped line and every
filled gap. It kept 37.1% of McCATMuS corrections: at 35-40% CER a correction
is a rewrite, however right, and nothing here can tell it from an invention.
On 25 real proofreads from a pipeline database it kept 40 of the 44 lines the
model changed; the other four dropped an uncertainty mark."""

_JOIN_LIMIT = 0.15
_UNCERTAIN = re.compile(r"\[…\]|\[\.\.\.\]|\[\?\]|\?|…")


@dataclass(frozen=True)
class GuardedProofread:
    """The text to keep, and what the guard did to get it."""

    text: str
    lines_kept: int = 0
    lines_reverted: int = 0
    lines_dropped: int = 0
    lines_restored: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.lines_reverted or self.lines_dropped or self.lines_restored)


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def _distance(a: str, b: str) -> float:
    """Levenshtein distance over the longer length, 0-1."""
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1] / max(len(a), len(b))


def _cheap(a: str, b: str) -> float:
    """Alignment cost: 1 - difflib ratio, a fast stand-in for edit distance."""
    if a == b:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _align_lines(raw: list[str], proof: list[str]) -> list[tuple[list[str], list[str]]]:
    """Groups of (raw lines, proofread lines): 1-1, 2-1 (joined), 1-2 (split), 1-0, 0-1."""
    n, m = len(raw), len(proof)
    nr = [_norm(line) for line in raw]
    np_ = [_norm(line) for line in proof]
    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    back: dict[tuple[int, int], tuple[int, int]] = {}
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            here = cost[i][j]
            if here == inf:
                continue
            steps = []
            if i < n:
                steps.append((i + 1, j, 1.0))
            if j < m:
                steps.append((i, j + 1, 1.0))
            if i < n and j < m:
                steps.append((i + 1, j + 1, _cheap(nr[i], np_[j])))
            # Joining or splitting lines is for a line break moved, as when an
            # initial set apart is joined to its word: the text must nearly match.
            if i + 1 < n and j < m:
                joined = min(_cheap(nr[i] + nr[i + 1], np_[j]), _cheap(f"{nr[i]} {nr[i + 1]}", np_[j]))
                if joined <= _JOIN_LIMIT:
                    steps.append((i + 2, j + 1, joined + 0.05))
            if i < n and j + 1 < m:
                joined = min(_cheap(nr[i], np_[j] + np_[j + 1]), _cheap(nr[i], f"{np_[j]} {np_[j + 1]}"))
                if joined <= _JOIN_LIMIT:
                    steps.append((i + 1, j + 2, joined + 0.05))
            for ni, nj, step in steps:
                if here + step < cost[ni][nj]:
                    cost[ni][nj] = here + step
                    back[(ni, nj)] = (i, j)
    groups = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj = back[(i, j)]
        groups.append((raw[pi:i], proof[pj:j]))
        i, j = pi, pj
    return groups[::-1]


def _word_pairs(raw: list[str], proof: list[str]) -> list[tuple[str | None, str]]:
    """Each proofread word with the raw word it replaces (None for an inserted word)."""
    n, m = len(raw), len(proof)
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = float(i)
    for j in range(1, m + 1):
        cost[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1, cost[i - 1][j - 1] + _distance(raw[i - 1], proof[j - 1]))
    pairs: list[tuple[str | None, str]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if cost[i][j] == cost[i - 1][j - 1] + _distance(raw[i - 1], proof[j - 1]):
            pairs.append((raw[i - 1], proof[j - 1]))
            i, j = i - 1, j - 1
        elif cost[i][j] == cost[i][j - 1] + 1:
            pairs.append((None, proof[j - 1]))
            j -= 1
        else:
            i -= 1
    pairs.extend((None, proof[k]) for k in range(j - 1, -1, -1))
    return pairs[::-1]


def judge_line(raw: str, proof: str) -> str | None:
    """Why the proofread line must revert to *raw*, or None when the change is a correction."""
    a, b = _norm(raw), _norm(proof)
    if a == b:
        return None
    if len(_UNCERTAIN.findall(proof)) < len(_UNCERTAIN.findall(raw)):
        return "an uncertain reading was filled in"
    rewrite = _distance(a, b)
    if rewrite > LINE_REWRITE_LIMIT:
        return f"line rewritten ({rewrite:.2f} of it changed)"
    squashed = re.sub(r"\W", "", a)
    for old, new in _word_pairs(a.split(), b.split()):
        letters = re.sub(r"\W", "", new)
        if len(letters) < 3 or old == new:
            continue
        # A word split or joined differently is still the same letters.
        if letters in squashed:
            continue
        if old is None or _distance(old, new) > WORD_INVENTION_LIMIT:
            return f"word invented: {new!r}" + ("" if old is None else f" for {old!r}")
    return None


def guard_proofread(raw: str, proofread: str) -> GuardedProofread:
    """The proofread text with every rewritten, invented or gap-filling line put back as OCR'd."""
    raw_lines = [line.strip() for line in (raw or "").splitlines() if line.strip()]
    proof_lines = [line.strip() for line in (proofread or "").splitlines() if line.strip()]
    if not proof_lines:
        return GuardedProofread(text="\n".join(raw_lines), lines_restored=len(raw_lines), reasons=["empty proofread output"] if raw_lines else [])
    out: list[str] = []
    kept = reverted = dropped = restored = 0
    reasons: list[str] = []
    for raw_group, proof_group in _align_lines(raw_lines, proof_lines):
        if not proof_group:
            out.extend(raw_group)
            restored += len(raw_group)
            reasons.append(f"line deleted by the proofreader, restored: {raw_group[0]!r}")
            continue
        if not raw_group:
            dropped += len(proof_group)
            reasons.append(f"line added by the proofreader, dropped: {proof_group[0]!r}")
            continue
        verdict = judge_line(" ".join(raw_group), " ".join(proof_group))
        if verdict is None:
            out.extend(proof_group)
            kept += 1
        else:
            out.extend(raw_group)
            reverted += 1
            reasons.append(f"{verdict}: {proof_group[0]!r}")
    return GuardedProofread(
        text="\n".join(out),
        lines_kept=kept,
        lines_reverted=reverted,
        lines_dropped=dropped,
        lines_restored=restored,
        reasons=reasons,
    )
