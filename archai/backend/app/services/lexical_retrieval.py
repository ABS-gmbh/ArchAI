"""Lexical retrieval for noisy medieval OCR, and fusion with the dense ranking.

Dense retrieval alone misses passage lookups on this material. The benchmark in
``scripts/benchmark_retrieval.py`` takes a passage from one OCR reading of a
page and looks it up among the chunks of a *different* reading of the same page,
so every query carries the noise a user's quotation would: 1,457 lookups against
58 readings, with a median CER of 0.42 between the two readings. The dense ranking
put the right chunk among the five sent to the model for 78.0% of them.

Words are the wrong unit for this text. OCR errors and unstable medieval
spelling mean the same word seldom has the same form in two readings ("confel" /
"confeil", "uilain" / "vilam"), so word-level BM25 did no better than dense (MRR
0.60 against 0.59). Overlapping character 4-grams survive most of those
differences and reached MRR 0.84 on their own. n=4 follows McNamee and Mayfield
(2004) on European-language retrieval; on the benchmark 3-grams were
indistinguishable once fused and 5-grams were worse.

Terms are taken from :func:`app.services.medieval_text.build_search_key`, so a
query in expanded or modern spelling meets the abbreviated diplomatic form on the
same terms: "dominus" matches "dn̄s", "iohannes" matches "Johannes", "uir" "vir".
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence

from app.services.medieval_text import build_search_key, rejoin_line_breaks

NGRAM = 4

# Robertson/Walker defaults as used by Lucene; not tuned on the benchmark.
_K1 = 1.2
_B = 0.75


def search_terms(text: str, *, n: int = NGRAM) -> list[str]:
    """Character n-grams of each search-key token.

    Tokens are padded with a space on each side, so a word's first and last
    letters form their own grams and whole-word matches outscore matches inside
    a longer word. A token shorter than *n* is kept whole rather than dropped.
    """
    key = build_search_key(rejoin_line_breaks(text or ""))
    terms: list[str] = []
    for token in key.split():
        padded = f" {token} "
        if len(padded) <= n:
            terms.append(padded)
        else:
            terms.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return terms


class LexicalIndex:
    """Okapi BM25 over :func:`search_terms`, built once per candidate set.

    *terms* turns text into index terms; the same function is applied to the
    query. It exists so the benchmark can compare units, and defaults to the
    character 4-grams retrieval uses.
    """

    def __init__(
        self,
        documents: Iterable[tuple[str, str]],
        *,
        terms: Callable[[str], list[str]] = search_terms,
    ) -> None:
        self._terms = terms
        self.ids: list[str] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths: list[int] = []
        for position, (doc_id, text) in enumerate(documents):
            counts = Counter(terms(text))
            self.ids.append(doc_id)
            lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self._postings[term].append((position, tf))
        average = sum(lengths) / len(lengths) if lengths else 0.0
        # The length-normalised part of the BM25 denominator depends only on
        # the document, so it is computed once here rather than per query term.
        self._norms = [_K1 * (1 - _B + _B * length / average) if average else _K1 for length in lengths]

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: str, *, limit: int | None = None) -> list[tuple[str, float]]:
        """``(id, score)`` for every document sharing a term with *query*, best first.

        Documents that share no term are omitted rather than scored zero, so a
        query with no lexical overlap produces an empty ranking and leaves any
        fusion with the dense ranking unchanged.
        """
        total = len(self.ids)
        scores: dict[int, float] = defaultdict(float)
        for term in set(self._terms(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            # Lucene's IDF: never negative, unlike the original Robertson form,
            # which on a single page would reject any term in most chunks.
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            for position, tf in postings:
                scores[position] += idf * tf * (_K1 + 1) / (tf + self._norms[position])
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if limit is not None:
            ranked = ranked[:limit]
        return [(self.ids[position], score) for position, score in ranked]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], *, k: int = 60) -> list[tuple[str, float]]:
    """Fuse rankings with RRF: score(d) = sum over rankings of 1 / (k + rank of d).

    Cormack, Clarke and Buettcher (SIGIR 2009); k = 60 is their constant. Only
    ranks are used, so a cosine distance and a BM25 score never have to be put
    on a common scale. Ties keep the order in which items were first ranked.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(item, len(first_seen))
    return sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
