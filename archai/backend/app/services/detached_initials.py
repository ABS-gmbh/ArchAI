"""Join a versal initial that Kraken wrote apart from its word ("C aldꝰ" -> "Caldꝰ").

Pages that set their initials in a column of their own leave a gap between the
initial and the rest of the word, and Kraken reads that gap as a space. The
reference transcriptions join them (eval/recognizers/README.md), and so do the
search keys and mention rules downstream: "C aldꝰ li roys" yields no name.

The decision rests on the page's geometry, not on its letters:

1. ``leading_glyph`` measures, from Kraken's own per-character cuts, whether a
   line begins with one glyph and a space, and the gap after that glyph as a
   share of the line's median word gap.
2. ``lines_to_join`` groups the lines of a request into columns by the left
   edge of their boxes. In a column where enough lines begin that way (the page
   sets its initials apart), a line is joined when its first glyph is a letter
   (word signs such as ⁊, & and ꝯ stay apart, as the references keep them) and
   its gap is at least a word gap (an abbreviated one-glyph word such as Qͥ at
   word spacing stays apart).

The three thresholds are settings (``ocr_detached_initial_*``). They were set
on the two development pages with versals (latin-abaton, old-french-verse),
with one annotator.
"""

from __future__ import annotations

import statistics
import unicodedata
from collections.abc import Sequence
from typing import Any


def _x_extent(cut: Any) -> tuple[float, float] | None:
    """A Kraken cut's horizontal extent: a polygon of points, or a bbox (x0, y0, x1, y1)."""
    try:
        if cut and not isinstance(cut[0], (list, tuple)):
            return float(cut[0]), float(cut[2])
        xs = [float(point[0]) for point in cut]
        return min(xs), max(xs)
    except (TypeError, ValueError, IndexError):
        return None


def _first_glyph_end(text: str) -> int:
    """Index just past the first character and its combining marks."""
    end = 1
    while end < len(text) and unicodedata.combining(text[end]):
        end += 1
    return end


def starts_detached(text: str) -> bool:
    """The line begins with one glyph (combining marks allowed), a single space and a word."""
    text = text or ""
    end = _first_glyph_end(text)
    return bool(text[:1].strip()) and text[end : end + 1] == " " and bool(text[end + 1 : end + 2].strip())


def leading_glyph(prediction: str, cuts: Sequence[Any]) -> dict[str, Any] | None:
    """Whether *prediction* starts with a detached glyph, and its gap as a share of the median word gap.

    *prediction* and *cuts* are a Kraken record's, one cut per character (Kraken
    predicts NFD, so a combining mark has its own cut). ``gap_ratio`` is None
    when the line has no other word gap to compare with. Returns None when the
    cuts do not match the prediction.
    """
    extents = [_x_extent(cut) for cut in cuts]
    if not prediction or len(extents) != len(prediction) or any(e is None for e in extents):
        return None
    if not starts_detached(prediction):
        return {"detached": False, "gap_ratio": None}
    end = _first_glyph_end(prediction)
    first_gap = extents[end + 1][0] - max(e[1] for e in extents[:end])  # type: ignore[index]
    gaps = []
    for idx, char in enumerate(prediction):
        if char != " " or idx == end or idx == 0 or idx + 1 >= len(prediction):
            continue
        if prediction[idx - 1] == " " or prediction[idx + 1] == " ":
            continue
        left = idx - 1
        while left > 0 and unicodedata.combining(prediction[left]):
            left -= 1
        gaps.append(extents[idx + 1][0] - max(e[1] for e in extents[left:idx]))  # type: ignore[index]
    median = statistics.median(gaps) if gaps else None
    ratio = round(first_gap / median, 3) if median and median > 0 else None
    return {"detached": True, "gap_ratio": ratio}


def _columns(boxes: Sequence[Sequence[float]]) -> list[int]:
    """A column index per box, clustering the boxes' left edges.

    A new column starts where the next left edge, in sorted order, is more than
    half a median line width to the right of the previous one.
    """
    widths = sorted(float(b[2]) - float(b[0]) for b in boxes)
    split = 0.5 * widths[len(widths) // 2] if widths else 0.0
    order = sorted(range(len(boxes)), key=lambda i: float(boxes[i][0]))
    columns = [0] * len(boxes)
    column, previous = 0, None
    for i in order:
        x0 = float(boxes[i][0])
        if previous is not None and x0 - previous > split:
            column += 1
        columns[i] = column
        previous = x0
    return columns


def lines_to_join(
    lines: Sequence[dict[str, Any]],
    *,
    min_column_lines: int,
    min_share: float,
    min_gap_ratio: float,
) -> set[int]:
    """Indexes of *lines* whose detached first glyph should be joined to the next word.

    Each line is ``{"text", "bbox_xyxy", "leading_glyph"}``; ``leading_glyph`` is
    what ``leading_glyph`` returned for it (None when unknown). Lines without a
    box or text take no part.
    """
    usable = [i for i, line in enumerate(lines) if str(line.get("text") or "").strip() and line.get("bbox_xyxy")]
    if not usable:
        return set()
    columns = _columns([lines[i]["bbox_xyxy"] for i in usable])
    by_column: dict[int, list[int]] = {}
    for i, column in zip(usable, columns):
        by_column.setdefault(column, []).append(i)
    chosen: set[int] = set()
    for members in by_column.values():
        if len(members) < min_column_lines:
            continue
        detached = [i for i in members if (lines[i].get("leading_glyph") or {}).get("detached")]
        if len(detached) < min_share * len(members):
            continue
        for i in detached:
            text = str(lines[i]["text"])
            ratio = (lines[i].get("leading_glyph") or {}).get("gap_ratio")
            # An initial is a letter: word signs (⁊, &, ꝯ) stay apart.
            if not unicodedata.category(text[0]).startswith("L"):
                continue
            # A one-glyph word at word spacing (Qͥ, Q̾) stays apart.
            if ratio is not None and ratio < min_gap_ratio:
                continue
            if starts_detached(text):
                chosen.add(i)
    return chosen


def join_first_glyph(text: str) -> str:
    """'C aldꝰ li roys' -> 'Caldꝰ li roys'; text that does not start detached is returned as is."""
    if not starts_detached(text):
        return text
    end = _first_glyph_end(text)
    return text[:end] + text[end + 1 :]
