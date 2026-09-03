"""Tests for windowed retrieval chunking.

One chunk per OCR line gave a measured 24.1-char median on the live store, so
top_k=5 delivered ~136 characters of evidence — about 11% of a page. Any question
needing more than one manuscript line was unanswerable from retrieved text.
"""

from __future__ import annotations

import pytest

from app.services.chunking import build_window_chunks, coverage_at_k

PAGE = "\n".join(f"linea numero {i} de folio scripta" for i in range(24))


def test_windows_are_produced_instead_of_single_lines() -> None:
    chunks = build_window_chunks(PAGE, window_lines=6, overlap_lines=2)
    assert len(chunks) < 24
    assert all(len(c["text"]) > 100 for c in chunks)


def test_offsets_index_the_original_text() -> None:
    """Citations and evidence spans depend on this exactly."""
    chunks = build_window_chunks(PAGE, window_lines=6, overlap_lines=2)
    for c in chunks:
        assert PAGE[c["start_offset"] : c["end_offset"]] == c["text"]


def test_windows_overlap_by_the_requested_amount() -> None:
    chunks = build_window_chunks(PAGE, window_lines=6, overlap_lines=2)
    for earlier, later in zip(chunks, chunks[1:]):
        assert later["line_start"] == earlier["line_start"] + 4
        assert later["start_offset"] < earlier["end_offset"], "windows must overlap"


def test_every_line_appears_in_some_window() -> None:
    chunks = build_window_chunks(PAGE, window_lines=6, overlap_lines=2)
    covered: set[int] = set()
    for c in chunks:
        covered.update(range(c["line_start"], c["line_end"] + 1))
    assert covered == set(range(24))


def test_zero_overlap_is_allowed() -> None:
    chunks = build_window_chunks(PAGE, window_lines=6, overlap_lines=0)
    for earlier, later in zip(chunks, chunks[1:]):
        assert later["line_start"] == earlier["line_start"] + 6


def test_overlap_equal_to_window_is_rejected() -> None:
    """Otherwise the window never advances and the loop would not terminate."""
    with pytest.raises(ValueError, match="never advances"):
        build_window_chunks(PAGE, window_lines=4, overlap_lines=4)


def test_overlap_larger_than_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="never advances"):
        build_window_chunks(PAGE, window_lines=4, overlap_lines=9)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_window_is_rejected(bad: int) -> None:
    with pytest.raises(ValueError, match="window_lines must be"):
        build_window_chunks(PAGE, window_lines=bad)


def test_negative_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap_lines must be"):
        build_window_chunks(PAGE, window_lines=6, overlap_lines=-1)


def test_blank_input_yields_nothing() -> None:
    assert build_window_chunks("") == []
    assert build_window_chunks("   \n\n  ") == []


def test_blank_lines_are_skipped_without_breaking_offsets() -> None:
    text = "prima linea de folio\n\n\nsecunda linea de folio"
    chunks = build_window_chunks(text, window_lines=6, overlap_lines=2)
    assert len(chunks) == 1
    assert text[chunks[0]["start_offset"] : chunks[0]["end_offset"]] == chunks[0]["text"]


def test_page_shorter_than_one_window_is_a_single_chunk() -> None:
    text = "prima linea\nsecunda linea"
    chunks = build_window_chunks(text, window_lines=6, overlap_lines=2)
    assert len(chunks) == 1
    assert chunks[0]["line_start"] == 0 and chunks[0]["line_end"] == 1


def test_short_final_window_is_folded_into_its_predecessor() -> None:
    """A runt tail chunk would reintroduce the starved-context problem."""
    text = "\n".join(f"linea numero {i} de folio scripta" for i in range(9))
    chunks = build_window_chunks(text, window_lines=4, overlap_lines=0, min_chars=200)
    assert len(chunks) >= 1
    assert chunks[-1]["line_end"] == 8, "the last line must still be covered"


def test_merged_tail_keeps_text_consistent_with_its_offsets() -> None:
    """Regression: the merge widened the offsets but left the old text behind, so
    the final line was missing from the chunk that claimed to contain it."""
    text = "\n".join(f"linea numero {i} de folio scripta" for i in range(9))
    chunks = build_window_chunks(text, window_lines=4, overlap_lines=0, min_chars=200)
    for c in chunks:
        assert text[c["start_offset"] : c["end_offset"]] == c["text"]
    assert "numero 8" in chunks[-1]["text"], "merged content must actually be present"


@pytest.mark.parametrize("line_count", list(range(1, 41)))
@pytest.mark.parametrize(("window", "overlap"), [(6, 2), (4, 0), (3, 1), (8, 3)])
def test_offsets_and_coverage_hold_for_every_shape(line_count, window, overlap) -> None:
    """Sweep line counts and window shapes: no dropped lines, no offset drift."""
    text = "\n".join(f"linea numero {i} de folio scripta est" for i in range(line_count))
    chunks = build_window_chunks(text, window_lines=window, overlap_lines=overlap)
    covered: set[int] = set()
    for c in chunks:
        assert text[c["start_offset"] : c["end_offset"]] == c["text"]
        covered.update(range(c["line_start"], c["line_end"] + 1))
    assert covered == set(range(line_count)), f"lines dropped: {set(range(line_count)) - covered}"


def test_search_key_is_derived_and_normalised() -> None:
    chunks = build_window_chunks("ꝓpter dñs\n⁊ ſanctus est", window_lines=6)
    key = chunks[0]["search_key"]
    assert "propter" in key and "dominus" in key and " et " in f" {key} "
    assert "ꝓ" not in key and "ſ" not in key


def test_diplomatic_text_is_kept_alongside_the_search_key() -> None:
    chunks = build_window_chunks("ꝓpter dñs\n⁊ ſanctus est", window_lines=6)
    assert "ꝓpter" in chunks[0]["text"], "the citable reading must be preserved"


def test_chunk_indices_are_contiguous_from_zero() -> None:
    chunks = build_window_chunks(PAGE, window_lines=6, overlap_lines=2)
    assert [c["idx"] for c in chunks] == list(range(len(chunks)))


# ────────────────────────────────────────────────── coverage metric ──


def test_coverage_at_k_rewards_windows_over_lines() -> None:
    """The measurement that motivates this module."""
    lines = PAGE.split("\n")
    spans = []
    for i in range(0, 21, 3):
        start = PAGE.index(lines[i])
        end = PAGE.index(lines[i + 2]) + len(lines[i + 2])
        spans.append((start, end))

    per_line = build_window_chunks(PAGE, window_lines=1, overlap_lines=0)
    windowed = build_window_chunks(PAGE, window_lines=6, overlap_lines=2)
    assert coverage_at_k(windowed, spans, 5) > coverage_at_k(per_line, spans, 5)


def test_coverage_at_k_edge_cases() -> None:
    chunks = build_window_chunks(PAGE, window_lines=6, overlap_lines=2)
    assert coverage_at_k(chunks, [], 5) == 0.0
    assert coverage_at_k(chunks, [(0, 10)], 0) == 0.0
    assert coverage_at_k([], [(0, 10)], 5) == 0.0
