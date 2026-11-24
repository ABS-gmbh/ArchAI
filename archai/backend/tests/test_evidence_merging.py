"""Overlapping retrieval hits share one evidence block: no line twice, every chunk still citable."""

from __future__ import annotations

import itertools
import json
import random
import re
from pathlib import Path

from app.services import rag_store
from app.services.chunking import build_window_chunks

BACKEND = Path(__file__).resolve().parents[1]
PAGE = next(p["reference"] for p in json.loads((BACKEND / "eval/quality_gate/readings.json").read_text())["pages"] if p["page_id"] == "old-french")


def _hits(indices: list[int], run_id: str = "run-1") -> list[dict]:
    windows = build_window_chunks(PAGE, window_lines=6, overlap_lines=2)
    return [
        {**windows[i], "chunk_id": f"{run_id}:{windows[i]['idx']}", "run_id": run_id, "asset_ref": "page", "distance": 0.3 + i / 100}
        for i in indices
    ]


def _blocks(evidence: str) -> list[dict[str, str]]:
    out = []
    for raw in re.findall(r"\[OCR_CHUNK_EVIDENCE\]\n(.*?)\n\[/OCR_CHUNK_EVIDENCE\]", evidence, flags=re.S):
        fields, text = raw.split("\ntext: ", 1)
        out.append({**dict(line.split(": ", 1) for line in fields.splitlines()), "text": text})
    return out


def _check(hits: list[dict]) -> list[dict[str, str]]:
    blocks = _blocks(rag_store.format_evidence_blocks(hits))
    # Every chunk id is cited, exactly once.
    cited = [cid.strip() for block in blocks for cid in block["chunk_id"].split(",")]
    assert sorted(cited) == sorted(hit["chunk_id"] for hit in hits)
    # Every line of every retrieved chunk appears exactly once across the blocks.
    covered = {(hit["run_id"], hit["start_offset"] + m.start()) for hit in hits for m in re.finditer(r"[^\n]+", hit["text"])}
    shown = []
    for block in blocks:
        start = int(block["offsets"].split("-")[0])
        assert PAGE[start : int(block["offsets"].split("-")[1])] == block["text"]  # the run's own text
        shown += [(block["run_id"], start + m.start()) for m in re.finditer(r"[^\n]+", block["text"])]
    assert sorted(shown) == sorted(covered)
    return blocks


def test_neighbouring_windows_merge_and_lines_appear_once() -> None:
    blocks = _check(_hits([1, 0, 2]))
    assert len(blocks) == 1
    assert blocks[0]["chunk_id"] == "run-1:0, run-1:1, run-1:2"
    assert blocks[0]["chunk_offsets"].count("=") == 3


def test_separate_windows_stay_separate_and_keep_the_single_format() -> None:
    blocks = _check(_hits([3, 0]))
    assert [b["chunk_id"] for b in blocks] == ["run-1:3", "run-1:0"]  # rank order kept
    assert all("chunk_offsets" not in b for b in blocks)


def test_groups_follow_their_best_ranked_hit() -> None:
    blocks = _check(_hits([5, 0, 1, 6]))
    assert [b["chunk_id"] for b in blocks] == ["run-1:5, run-1:6", "run-1:0, run-1:1"]


def test_hits_of_different_runs_never_merge() -> None:
    blocks = _check([*_hits([0]), *_hits([0], run_id="run-2")])
    assert [b["run_id"] for b in blocks] == ["run-1", "run-2"]


def test_any_selection_and_order_of_hits() -> None:
    rng = random.Random(7)
    windows = len(build_window_chunks(PAGE, window_lines=6, overlap_lines=2))
    for size in range(1, 7):
        for _ in range(20):
            _check(_hits(rng.sample(range(windows), size)))
    for order in itertools.permutations(range(4)):
        _check(_hits(list(order)))


def test_merging_does_not_change_which_chunks_are_retrieved() -> None:
    hits = _hits([2, 0, 1])
    before = [dict(hit) for hit in hits]
    rag_store.format_evidence_blocks(hits)
    assert hits == before


def test_the_default_top_k_is_eight() -> None:
    """Measured on eval/chat: 56/62 answers in evidence at 8 merged, 1,620 chars, under 5 unmerged (1,739)."""
    from app.config import Settings

    assert Settings.model_fields["rag_top_k"].default == 8
