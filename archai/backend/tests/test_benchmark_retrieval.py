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


def _readings_file(path: Path, pages: dict[str, dict[str, str]]) -> Path:
    import json

    payload = {"pages": [{"page_id": pid, "reference": r.pop("reference"), "readings": r} for pid, r in pages.items()]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_readings_file_is_grouped_by_page_id(bench, tmp_path: Path) -> None:
    page_a, page_b = _page(1), _page(2)
    path = _readings_file(
        tmp_path / "readings.json",
        {
            "a": {"reference": page_a, "kraken": _reread(page_a, 0.15, seed=4), "empty": ""},
            "b": {"reference": page_b, "kraken": _reread(page_b, 0.15, seed=5)},
        },
    )
    texts, groups = bench.load_readings(path)
    assert sorted(texts) == ["a:kraken", "a:reference", "b:kraken", "b:reference"]
    assert groups == [["a:reference", "a:kraken"], ["b:reference", "b:kraken"]]


def test_the_report_counts_lookups_per_page(bench, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    """A database can hold one page read many times; the report must say so."""
    import json

    page_a, page_b = _page(1), _page(2)
    path = _readings_file(
        tmp_path / "readings.json",
        {
            "a": {"reference": page_a, "k1": _reread(page_a, 0.15, seed=4), "k2": _reread(page_a, 0.15, seed=6)},
            "b": {"reference": page_b, "k1": _reread(page_b, 0.15, seed=5)},
        },
    )
    # No model: the offline suite cannot download one, and the counts do not depend on it.
    monkeypatch.setattr(bench, "_embedder", lambda kind: lambda batch: [[float(len(t)), 1.0] for t in batch])
    assert bench.main(["--readings", str(path), "--json"]) == 0
    meta = json.loads(capsys.readouterr().out)["meta"]
    assert set(meta["lookups_per_page"]) == {"a", "b"}
    assert meta["pages"] == 2
    assert sum(meta["lookups_per_page"].values()) == meta["queries"]


BACKEND = Path(__file__).resolve().parents[1]


def test_every_question_answer_is_in_its_reference_once() -> None:
    """The labels come from these spans: one that drifts from the text labels nothing."""
    import json

    pages = {p["page_id"]: p["reference"] for p in json.loads((BACKEND / "eval/quality_gate/readings.json").read_text())["pages"]}
    questions = json.loads((BACKEND / "eval/retrieval/questions.json").read_text(encoding="utf-8"))["questions"]
    assert len(questions) >= 30
    for q in questions:
        assert pages[q["page_id"]].count(q["answer"]) == 1, q


def test_a_question_is_relevant_where_its_answer_is_in_each_reading(bench, tmp_path: Path) -> None:
    page = _page(1, lines=30)
    reread = _reread(page, 0.15, seed=4)
    path = _readings_file(tmp_path / "readings.json", {"a": {"reference": page, "kraken": reread}})
    texts, groups = bench.load_readings(path)
    lines = page.splitlines()
    answer = "\n".join(lines[20:22])
    queries, chunks = bench.build_question_benchmark(texts, groups, [{"page_id": "a", "question": "where?", "answer": answer}])
    assert {q.target for q in queries} == {"a:reference", "a:kraken"}
    start = page.index(answer)
    for q in queries:
        holding = {f"{q.target}#{c['idx']}" for c in chunks[q.target] if c["start_offset"] <= start and start + len(answer) <= c["end_offset"]}
        # Rereading keeps every offset, so the same chunks hold the answer in both.
        assert q.relevant == frozenset(holding)
        assert q.text == "where?"
