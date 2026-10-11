"""Benchmark chunk retrieval on real OCR output: dense, lexical and fused.

Usage::

    python scripts/benchmark_retrieval.py
    python scripts/benchmark_retrieval.py --db path/to/archai.sqlite --json
    python scripts/benchmark_retrieval.py --embedder provider

Relevance judgments are the expensive part of a retrieval benchmark, and this
project has none. They are derived from structure instead: the pipeline DB holds
many independent OCR readings of the same page, and a passage from one reading
should retrieve the chunk of another reading that holds the same text.

1. Distinct transcriptions (``ocr_text`` and ``proofread_text`` of every run)
   are grouped into pages by character 4-gram Jaccard similarity. Readings of
   one page measured >= 0.14 against each other and different pages <= 0.07, so
   the 0.12 threshold separates them with a margin.
2. Each reading is chunked exactly as the pipeline chunks it.
3. For a pair of readings A and B of one page, passages of 3, 5 and 8 words are
   drawn from A. The pair must differ (CER >= 0.10: re-runs that agree verbatim
   would only test string equality) yet stay alignable (CER <= 0.60). A and B are
   aligned character by character and each passage's span is projected into B;
   the relevant chunks are those of B containing the projected span. Relevance
   therefore never depends on any retriever's scores.
4. Every retriever ranks B's chunks - the one-page scope chat retrieval uses.
   MRR, Hit@1 and Hit@5 (top_k is 5) are reported with bootstrap intervals
   resampled over target readings, since queries against one reading are not
   independent.

This measures passage lookup under OCR noise: a user quoting or searching for
words they have read. It does not measure questions that share no vocabulary
with the page; no judgments exist for those.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.chunking import build_window_chunks  # noqa: E402
from app.services.evaluation import (  # noqa: E402
    alignment_offsets,
    bootstrap_ci,
    cer,
    recall_at_k,
    reciprocal_rank,
)
from app.services.lexical_retrieval import LexicalIndex, reciprocal_rank_fusion  # noqa: E402
from app.services.medieval_text import build_search_key, rejoin_line_breaks  # noqa: E402

SAME_PAGE_JACCARD = 0.12
MIN_PAIR_CER, MAX_PAIR_CER = 0.10, 0.60
PASSAGE_WORDS = (3, 5, 8)
PASSAGES_PER_LENGTH = 2
MIN_TARGET_CHUNKS = 4  # below this, ranking is close to trivial


@dataclass(frozen=True)
class Query:
    qid: str
    source: str
    target: str
    text: str
    relevant: frozenset[str]


def load_transcriptions(db_path: Path) -> dict[str, str]:
    """Distinct transcriptions of 400+ characters, keyed ``run_id:field``."""
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute("SELECT run_id, ocr_text, proofread_text FROM pipeline_runs ORDER BY run_id").fetchall()
    seen: dict[str, tuple[str, str]] = {}
    for run_id, ocr_text, proofread in rows:
        for field, text in (("ocr", ocr_text), ("proofread", proofread)):
            text = (text or "").strip()
            if len(text) >= 400:
                seen.setdefault(" ".join(text.split()), (f"{run_id}:{field}", text))
    return dict(seen.values())


def _grams(text: str) -> set[str]:
    folded = " ".join(re.findall(r"\w+", text.lower()))
    return {folded[i : i + 4] for i in range(len(folded) - 3)}


def group_pages(texts: dict[str, str]) -> list[list[str]]:
    """Single-linkage groups of readings of the same page (two or more readings)."""
    ids = sorted(texts)
    grams = {key: _grams(texts[key]) for key in ids}
    parent = {key: key for key in ids}

    def root(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            union = grams[a] | grams[b]
            if union and len(grams[a] & grams[b]) / len(union) >= SAME_PAGE_JACCARD:
                parent[root(a)] = root(b)
    groups: dict[str, list[str]] = {}
    for key in ids:
        groups.setdefault(root(key), []).append(key)
    return [group for group in groups.values() if len(group) > 1]


def build_benchmark(texts: dict[str, str], *, pairs: int, seed: int):
    rng = random.Random(seed)
    groups = group_pages(texts)
    page_of = {key: n for n, group in enumerate(groups) for key in group}
    chunks = {
        key: build_window_chunks(texts[key], window_lines=6, overlap_lines=2)
        for group in groups
        for key in group
    }
    candidates = [
        (a, b)
        for a in sorted(page_of)
        for b in sorted(page_of)
        if a != b and page_of[a] == page_of[b] and len(chunks[b]) >= MIN_TARGET_CHUNKS
    ]
    rng.shuffle(candidates)

    queries: list[Query] = []
    pair_cers: list[float] = []
    for a, b in candidates:
        if len(pair_cers) >= pairs:
            break
        pair_cer = cer(texts[a], texts[b]).rate
        if not MIN_PAIR_CER <= pair_cer <= MAX_PAIR_CER:
            continue
        pair_cers.append(pair_cer)
        offsets = alignment_offsets(texts[a], texts[b])
        words = [(m.start(), m.end()) for m in re.finditer(r"\S+", texts[a])]
        for length in PASSAGE_WORDS:
            for _ in range(PASSAGES_PER_LENGTH):
                first = rng.randrange(0, max(1, len(words) - length))
                start, end = words[first][0], words[first + length - 1][1]
                b_start, b_end = int(offsets[start]), int(offsets[end])
                span = b_end - b_start
                # A projection far shorter or longer than the passage means the
                # alignment lost its way here; its label cannot be trusted.
                if not 0.5 * (end - start) <= span <= 2.0 * (end - start):
                    continue
                relevant = _relevant_chunks(chunks[b], b, b_start, b_end)
                if relevant:
                    text = " ".join(texts[a][start:end].split())
                    queries.append(Query(f"q{len(queries)}", a, b, text, relevant))
    return queries, chunks, groups, pair_cers


def _relevant_chunks(chunks: list[dict], target: str, start: int, end: int) -> frozenset[str]:
    containing = {_cid(target, c) for c in chunks if c["start_offset"] <= start and end <= c["end_offset"]}
    if containing:
        return frozenset(containing)
    overlap = {_cid(target, c): max(0, min(end, c["end_offset"]) - max(start, c["start_offset"])) for c in chunks}
    best = max(overlap.values(), default=0)
    return frozenset(cid for cid, size in overlap.items() if size > 0 and (size >= 0.5 * (end - start) or size == best))


def _cid(target: str, chunk: dict) -> str:
    return f"{target}#{chunk['idx']}"


# ── retrievers ─────────────────────────────────────────────────────────


def _embedder(kind: str) -> Callable[[list[str]], list[list[float]]]:
    if kind == "local":
        from chromadb.utils import embedding_functions

        function = embedding_functions.DefaultEmbeddingFunction()
        return lambda batch: [list(v) for v in function(batch)]

    from app.services import rag_store

    def provider(batch: list[str]) -> list[list[float]]:
        try:
            vectors = rag_store._embed(batch)
        except rag_store.EmbeddingUnavailable as exc:
            raise SystemExit(f"error: the configured embedding provider is unavailable: {exc}") from exc
        if vectors is None:
            raise SystemExit("error: no embedding provider is configured (RAG_EMBEDDING_MODEL is empty)")
        return vectors

    return provider


def dense_rankings(queries, chunks, embed) -> dict[str, list[str]]:
    texts = list(dict.fromkeys([c["text"] for cs in chunks.values() for c in cs] + [q.text for q in queries]))
    vectors: dict[str, np.ndarray] = {}
    for i in range(0, len(texts), 64):
        batch = texts[i : i + 64]
        for text, vector in zip(batch, embed(batch), strict=True):
            array = np.asarray(vector, dtype=np.float64)
            vectors[text] = array / (np.linalg.norm(array) or 1.0)
    out = {}
    for q in queries:
        cands = chunks[q.target]
        # Exact cosine ranking: what Chroma's HNSW index approximates.
        sims = np.stack([vectors[c["text"]] for c in cands]) @ vectors[q.text]
        out[q.qid] = [_cid(q.target, cands[i]) for i in np.argsort(-sims, kind="stable")]
    return out


def word_terms(text: str) -> list[str]:
    """Whole search-key words: the ablation against character n-grams."""
    return build_search_key(rejoin_line_breaks(text)).split()


def lexical_rankings(queries, chunks, **index_options) -> dict[str, list[str]]:
    indexes = {
        target: LexicalIndex(((_cid(target, c), c["text"]) for c in cands), **index_options)
        for target, cands in chunks.items()
    }
    return {q.qid: [cid for cid, _ in indexes[q.target].search(q.text)] for q in queries}


def fused_rankings(*runs: dict[str, list[str]]) -> dict[str, list[str]]:
    return {qid: [cid for cid, _ in reciprocal_rank_fusion([run[qid] for run in runs])] for qid in runs[0]}


# ── scoring ────────────────────────────────────────────────────────────


def score(queries: Sequence[Query], run: dict[str, list[str]]) -> dict[str, tuple[float, float, float]]:
    return {
        q.qid: (
            reciprocal_rank(run[q.qid], q.relevant),
            float(recall_at_k(run[q.qid], q.relevant, 1) > 0),
            float(recall_at_k(run[q.qid], q.relevant, 5) > 0),
        )
        for q in queries
    }


def summarise(queries, scores, baseline=None, *, seed: int) -> dict[str, dict[str, float]]:
    groups: dict[str, list[str]] = {}
    for q in queries:
        groups.setdefault(q.target, []).append(q.qid)
    clusters = list(groups.values())
    out = {}
    for column, name in enumerate(("mrr", "hit@1", "hit@5")):

        def mean(sample, column=column, source=scores):
            qids = [qid for cluster in sample for qid in cluster]
            return sum(source[qid][column] for qid in qids) / len(qids)

        low, high = bootstrap_ci(clusters, mean, n_resamples=1000, seed=seed)
        entry = {"mean": mean(clusters), "ci": [low, high]}
        if baseline is not None:

            def delta(sample, column=column):
                qids = [qid for cluster in sample for qid in cluster]
                return sum(scores[qid][column] - baseline[qid][column] for qid in qids) / len(qids)

            d_low, d_high = bootstrap_ci(clusters, delta, n_resamples=1000, seed=seed)
            entry["delta_vs_dense"] = {"mean": delta(clusters), "ci": [d_low, d_high]}
        out[name] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    default_db = os.getenv("ARCHAI_DB_PATH") or str(Path(__file__).resolve().parents[1] / "app" / "archai.sqlite")
    parser.add_argument("--db", type=Path, default=Path(default_db))
    parser.add_argument(
        "--embedder",
        choices=("local", "provider"),
        default="local",
        help="local: Chroma's default model, the pipeline's fallback; provider: settings.rag_embedding_model",
    )
    parser.add_argument("--pairs", type=int, default=260, help="reading pairs to draw passages from")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.db.is_file():
        print(f"error: pipeline DB not found: {args.db}", file=sys.stderr)
        return 2
    texts = load_transcriptions(args.db)
    queries, chunks, groups, pair_cers = build_benchmark(texts, pairs=args.pairs, seed=args.seed)
    if not queries:
        print("error: no page has two alignable readings; nothing to benchmark", file=sys.stderr)
        return 2

    dense = dense_rankings(queries, chunks, _embedder(args.embedder))
    lexical = lexical_rankings(queries, chunks)
    runs = {
        "dense": dense,
        "BM25 words": lexical_rankings(queries, chunks, terms=word_terms),
        "BM25 4-grams": lexical,
        "hybrid (RRF)": fused_rankings(dense, lexical),
    }
    base = score(queries, dense)
    report = {
        name: summarise(queries, score(queries, run), None if name == "dense" else base, seed=args.seed)
        for name, run in runs.items()
    }
    meta = {
        "queries": len(queries),
        "targets": len({q.target for q in queries}),
        "reading_pairs": len(pair_cers),
        "median_pair_cer": float(np.median(pair_cers)),
        "pages": len(groups),
        "embedder": args.embedder,
    }
    if args.json:
        print(json.dumps({"meta": meta, "results": report}, indent=2))
        return 0

    print(
        f"{meta['queries']} passage lookups over {meta['targets']} readings "
        f"({meta['reading_pairs']} reading pairs, median CER between readings {meta['median_pair_cer']:.2f}, "
        f"{meta['pages']} pages), dense embedder: {args.embedder}"
    )
    print(f"{'':<14}{'MRR':>22}{'Hit@1':>22}{'Hit@5':>22}")
    for name, metrics in report.items():
        cells = [
            f"{m['mean']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]"
            for m in (metrics["mrr"], metrics["hit@1"], metrics["hit@5"])
        ]
        print(f"{name:<14}" + "".join(f"{cell:>22}" for cell in cells))
    print("\nchange against dense (95% CI):")
    print(f"{'':<14}{'MRR':>26}{'Hit@1':>26}{'Hit@5':>26}")
    for name, metrics in report.items():
        if name == "dense":
            continue
        deltas = [metrics[metric]["delta_vs_dense"] for metric in ("mrr", "hit@1", "hit@5")]
        cells = [f"{d['mean']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]" for d in deltas]
        print(f"{name:<14}" + "".join(f"{cell:>26}" for cell in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
