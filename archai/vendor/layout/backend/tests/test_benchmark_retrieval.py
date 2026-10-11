"""The retrieval benchmark derives relevance from alignment; check those labels."""

from __future__ import annotations

import importlib.util
import random
import re
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_retrieval.py"
LETTERS = "abcdefghilmnopqrstu"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("benchmark_retrieval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _page(seed: int, lines: int = 30) -> str:
    rng = random.Random(seed)
    words = ["".join(rng.choice(LETTERS) for _ in range(rng.randrange(3, 9))) for _ in range(400)]
    return "\n".join(" ".join(rng.choice(words) for _ in range(5)) for _ in range(lines))


def _reread(text: str, rate: float, seed: int) -> str:
    """Another OCR reading: letters substituted in place, layout untouched, so
    every offset of the original is the same offset in the reading."""
    rng = random.Random(seed)
    return "".join(rng.choice(LETTERS) if ch.isalpha() and rng.random() < rate else ch for ch in text)


def _write_db(path: Path, readings: dict[str, str]) -> Path:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE pipeline_runs (run_id TEXT, ocr_text TEXT, proofread_text TEXT)")
        conn.executemany("INSERT INTO pipeline_runs VALUES (?, ?, NULL)", readings.items())
    return path


@pytest.fixture
def readings() -> dict[str, str]:
    page = _page(1)
    return {
        "clean": page,
        "noisy-1": _reread(page, 0.15, seed=2),
        "noisy-2": _reread(page, 0.15, seed=3),
        "unrelated": _page(99),
    }


def test_readings_of_one_page_are_grouped_apart_from_other_pages(bench, readings) -> None:
    groups = bench.group_pages(readings)
    assert groups == [["clean", "noisy-1", "noisy-2"]]


def test_labels_are_the_chunks_holding_the_passage(bench, readings) -> None:
    queries, chunks, _groups, pair_cers = bench.build_benchmark(readings, pairs=6, seed=0)
    assert queries and all(0.10 <= c <= 0.60 for c in pair_cers)
    checked = 0
    for q in queries:
        source = readings[q.source]
        match = re.search(r"\s+".join(map(re.escape, q.text.split())), source)
        assert match, q.text
        start, end = match.span()
        # Substitution-only readings keep offsets, so the truth is known exactly.
        expected = {
            f"{q.target}#{c['idx']}"
            for c in chunks[q.target]
            if c["start_offset"] <= start and end <= c["end_offset"]
        }
        if expected:
            assert q.relevant == expected, q
            checked += 1
    assert checked >= 0.9 * len(queries)


def test_lexical_retrieval_finds_most_passages(bench, readings) -> None:
    queries, chunks, _groups, _pair_cers = bench.build_benchmark(readings, pairs=6, seed=0)
    lexical = bench.lexical_rankings(queries, chunks)
    hits = [bench.score(queries, lexical)[q.qid][2] for q in queries]
    assert sum(hits) / len(hits) > 0.8


def test_fusion_combines_rankings_per_query(bench, readings) -> None:
    queries, chunks, _groups, _pair_cers = bench.build_benchmark(readings, pairs=2, seed=0)
    lexical = bench.lexical_rankings(queries, chunks)
    reversed_dense = {q.qid: [f"{q.target}#{c['idx']}" for c in reversed(chunks[q.target])] for q in queries}
    fused = bench.fused_rankings(reversed_dense, lexical)
    assert set(fused) == {q.qid for q in queries}
    assert all(set(fused[q.qid]) == set(reversed_dense[q.qid]) for q in queries)


def test_cli_rejects_a_missing_database(bench, tmp_path: Path, capsys) -> None:
    assert bench.main(["--db", str(tmp_path / "absent.sqlite")]) == 2
    assert "not found" in capsys.readouterr().err


def test_cli_needs_two_readings_of_a_page(bench, tmp_path: Path, capsys) -> None:
    db = _write_db(tmp_path / "one.sqlite", {"only": _page(5)})
    assert bench.main(["--db", str(db)]) == 2
    assert "nothing to benchmark" in capsys.readouterr().err
