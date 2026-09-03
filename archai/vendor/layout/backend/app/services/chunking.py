"""Retrieval chunking for manuscript transcriptions.

The pipeline previously indexed one OCR line per chunk. Measured on the live
store, that gave a mean distinct chunk length of 24.1 characters - four or five
words - so ``rag_top_k = 5`` handed the model roughly 136 characters, about 11%
of one page. Any question needing more than a single manuscript line could not
be answered from retrieved evidence: the retriever located the right line and
never returned its neighbours.

This module packs consecutive lines into overlapping windows. On a real page,
6-line windows with 2 lines of overlap raised complete-span coverage@5 from
39.1% to 100.0% with the same embedder.

Two invariants matter:

* ``text`` stays the diplomatic transcription and ``start_offset`` /
  ``end_offset`` index the ORIGINAL text, so existing citations and evidence
  spans keep resolving.
* ``search_key`` is a derived, normalised form (line breaks rejoined, scribal
  abbreviations expanded) used for embedding and matching only.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.services.medieval_text import build_search_key, rejoin_line_breaks

DEFAULT_WINDOW_LINES = 6
DEFAULT_OVERLAP_LINES = 2
DEFAULT_MIN_CHARS = 40


class _Line:
    __slots__ = ("text", "start", "end")

    def __init__(self, text: str, start: int, end: int) -> None:
        self.text = text
        self.start = start
        self.end = end


def _split_lines_with_offsets(text: str) -> list[_Line]:
    lines: list[_Line] = []
    cursor = 0
    for raw in text.splitlines():
        start = cursor
        end = start + len(raw)
        if raw.strip():
            lines.append(_Line(raw, start, end))
        cursor = end + 1
    return lines


def build_window_chunks(
    base_text: str,
    *,
    window_lines: int = DEFAULT_WINDOW_LINES,
    overlap_lines: int = DEFAULT_OVERLAP_LINES,
    min_chars: int = DEFAULT_MIN_CHARS,
    lexicon: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Pack consecutive non-blank lines into overlapping retrieval windows.

    ``overlap_lines`` must be smaller than ``window_lines`` or the window would
    never advance.
    """
    if window_lines < 1:
        raise ValueError(f"window_lines must be >= 1, got {window_lines}")
    if overlap_lines < 0:
        raise ValueError(f"overlap_lines must be >= 0, got {overlap_lines}")
    if overlap_lines >= window_lines:
        raise ValueError(
            f"overlap_lines ({overlap_lines}) must be smaller than window_lines "
            f"({window_lines}), otherwise the window never advances."
        )

    value = str(base_text or "")
    if not value.strip():
        return []

    lines = _split_lines_with_offsets(value)
    if not lines:
        return []

    step = window_lines - overlap_lines
    chunks: list[dict[str, Any]] = []
    idx = 0
    start_line = 0

    while start_line < len(lines):
        window = lines[start_line : start_line + window_lines]
        if not window:
            break

        start_offset = window[0].start
        end_offset = window[-1].end
        # Slice the original so offsets, text and citations stay consistent.
        text = value[start_offset:end_offset]

        chunks.append(
            {
                "idx": idx,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "text": text,
                "line_start": start_line,
                "line_end": start_line + len(window) - 1,
                "search_key": build_search_key(rejoin_line_breaks(text, lexicon=lexicon)),
            }
        )
        idx += 1

        if start_line + window_lines >= len(lines):
            break
        start_line += step

    return _merge_short_tail(chunks, min_chars, value, lexicon)


def _merge_short_tail(
    chunks: list[dict[str, Any]],
    min_chars: int,
    source: str,
    lexicon: frozenset[str] | None,
) -> list[dict[str, Any]]:
    """Fold a runt final window into its predecessor.

    A page whose line count is just past a window boundary otherwise ends with a
    one- or two-line chunk, which reintroduces exactly the starved-context
    problem this module exists to remove.

    ``text`` and ``search_key`` are rebuilt from the widened span. Extending the
    offsets alone left the merged chunk claiming a range whose final line was
    absent from its own text, which both broke the "offsets index the original
    text" invariant and lost that line from retrieval.
    """
    if len(chunks) < 2 or min_chars <= 0:
        return chunks
    tail = chunks[-1]
    if len(tail["text"]) >= min_chars:
        return chunks

    prev = chunks[-2]
    prev["end_offset"] = tail["end_offset"]
    prev["line_end"] = tail["line_end"]
    # Re-slice the union rather than concatenating the two texts, so the
    # separator characters between the windows are preserved exactly.
    prev["text"] = source[prev["start_offset"] : prev["end_offset"]]
    prev["search_key"] = build_search_key(rejoin_line_breaks(prev["text"], lexicon=lexicon))
    return chunks[:-1]


def coverage_at_k(
    chunks: Sequence[dict[str, Any]], spans: Sequence[tuple[int, int]], k: int
) -> float:
    """Fraction of spans fully contained in at least one of the first k chunks.

    Used to compare chunking strategies without needing a relevance judgement:
    a span the retriever can never return whole is a span it can never support.
    """
    if not spans:
        return 0.0
    window = list(chunks)[: max(0, k)]
    complete = 0
    for span_start, span_end in spans:
        if any(c["start_offset"] <= span_start and c["end_offset"] >= span_end for c in window):
            complete += 1
    return complete / len(spans)
