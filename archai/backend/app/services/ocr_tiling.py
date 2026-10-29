"""Tiles for reading a page region by region.

Layout suggestions become pixel rectangles, which are merged, grouped into
blocks, padded and put in reading order. Moved out of
``app/agents/saia_ocr_agent.py``.
"""

from __future__ import annotations

from typing import Sequence

from PIL import Image, ImageFilter

from app.schemas.agents_ocr import SaiaOCRLocationSuggestion


_TILE_MIN_AREA_RATIO = 0.0003  # tiles < 0.03% of image area are noise


_TILE_PAD_RATIO = 0.06  # 6% padding around each tile (avoids clipping ascenders/descenders)


def _suggestions_to_rects(
    suggestions: Sequence[SaiaOCRLocationSuggestion],
    img_w: int,
    img_h: int,
) -> list[tuple[int, int, int, int]]:
    """Convert location suggestions (xywh) → pixel rects (x1,y1,x2,y2), filtering tiny boxes."""
    img_area = float(img_w * img_h)
    rects: list[tuple[int, int, int, int]] = []
    for s in suggestions:
        bbox = list(s.bbox_xywh or [])
        if len(bbox) != 4:
            continue
        x, y, w, h = [float(v) for v in bbox]
        if w <= 0 or h <= 0:
            continue
        # Filter tiny boxes
        if (w * h) / max(img_area, 1.0) < _TILE_MIN_AREA_RATIO:
            continue
        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))
        x2 = min(img_w, int(round(x + w)))
        y2 = min(img_h, int(round(y + h)))
        if (x2 - x1) < 8 or (y2 - y1) < 8:
            continue
        rects.append((x1, y1, x2, y2))
    return rects


def _merge_overlapping_rects(
    rects: list[tuple[int, int, int, int]],
    merge_distance: int = 10,
) -> list[tuple[int, int, int, int]]:
    """Merge rectangles that overlap or are within `merge_distance` pixels."""
    if not rects:
        return []
    merged = list(rects)
    changed = True
    while changed:
        changed = False
        new_merged: list[tuple[int, int, int, int]] = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            ax1, ay1, ax2, ay2 = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = merged[j]
                # Check if they overlap or are within merge_distance
                if (
                    ax1 - merge_distance <= bx2
                    and ax2 + merge_distance >= bx1
                    and ay1 - merge_distance <= by2
                    and ay2 + merge_distance >= by1
                ):
                    ax1 = min(ax1, bx1)
                    ay1 = min(ay1, by1)
                    ax2 = max(ax2, bx2)
                    ay2 = max(ay2, by2)
                    used[j] = True
                    changed = True
            new_merged.append((ax1, ay1, ax2, ay2))
            used[i] = True
        merged = new_merged
    return merged


_MORPH_DOWNSCALE = 4          # work on 1/4-resolution mask for speed


_MORPH_DILATE_RADIUS = 18     # dilation radius in downscaled px (~72 original px)


_MORPH_MIN_BLOCK_AREA = 400   # ignore tiny connected components (in downscaled px²)


def _rects_to_block_rects(
    rects: list[tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
) -> list[tuple[int, int, int, int]]:
    """Merge nearby suggestion rects into block-level rects using morphological dilation.

    Algorithm:
      1. Paint all suggestion rects onto a binary mask (downscaled for speed).
      2. Dilate the mask with a circular kernel to bridge nearby regions.
      3. Find connected components via BFS.
      4. For each component return the bounding box (upscaled back to original coords).

    This replaces the naive rect-overlap merge and correctly merges text lines
    that belong to the same column/block even when there are small gaps.
    """
    if not rects:
        return []

    ds = _MORPH_DOWNSCALE
    mw = max(1, img_w // ds)
    mh = max(1, img_h // ds)

    # --- 1. Create binary mask ---
    mask = Image.new("L", (mw, mh), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in rects:
        draw.rectangle(
            [x1 // ds, y1 // ds, max(x1 // ds + 1, x2 // ds), max(y1 // ds + 1, y2 // ds)],
            fill=255,
        )

    # --- 2. Dilate using PIL's MaxFilter (square approximation of dilation) ---
    from PIL import ImageFilter
    # Apply MaxFilter repeatedly; kernel size 3 means 1-px dilation per pass.
    # Number of passes ≈ radius to reach desired dilation radius.
    passes = max(1, _MORPH_DILATE_RADIUS)
    for _ in range(passes):
        mask = mask.filter(ImageFilter.MaxFilter(size=3))

    # --- 3. BFS connected components on the dilated mask ---
    pixels = list(mask.getdata())
    labels = [0] * (mw * mh)
    label_id = 0

    def _bfs(start_idx: int, lid: int) -> tuple[int, int, int, int]:
        """BFS flood-fill returning bbox (x1, y1, x2, y2) in mask coords."""
        queue = [start_idx]
        labels[start_idx] = lid
        min_x = start_idx % mw
        max_x = min_x
        min_y = start_idx // mw
        max_y = min_y
        head = 0
        while head < len(queue):
            idx = queue[head]
            head += 1
            cx = idx % mw
            cy = idx // mw
            if cx < min_x:
                min_x = cx
            if cx > max_x:
                max_x = cx
            if cy < min_y:
                min_y = cy
            if cy > max_y:
                max_y = cy
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < mw and 0 <= ny < mh:
                    ni = ny * mw + nx
                    if labels[ni] == 0 and pixels[ni] > 0:
                        labels[ni] = lid
                        queue.append(ni)
        return min_x, min_y, max_x, max_y

    component_bboxes: list[tuple[int, int, int, int]] = []
    for i in range(mw * mh):
        if pixels[i] > 0 and labels[i] == 0:
            label_id += 1
            bx1, by1, bx2, by2 = _bfs(i, label_id)
            area = (bx2 - bx1 + 1) * (by2 - by1 + 1)
            if area >= _MORPH_MIN_BLOCK_AREA:
                component_bboxes.append((bx1, by1, bx2, by2))

    # --- 4. Upscale bboxes back to original coordinates ---
    block_rects: list[tuple[int, int, int, int]] = []
    for bx1, by1, bx2, by2 in component_bboxes:
        ox1 = max(0, bx1 * ds)
        oy1 = max(0, by1 * ds)
        ox2 = min(img_w, (bx2 + 1) * ds)
        oy2 = min(img_h, (by2 + 1) * ds)
        if (ox2 - ox1) >= 16 and (oy2 - oy1) >= 16:
            block_rects.append((ox1, oy1, ox2, oy2))

    if not block_rects:
        # Fallback: if morphological merge yields nothing, use simple rect merge
        return _merge_overlapping_rects_simple(rects, merge_distance=max(10, min(img_w, img_h) // 40))

    return block_rects


def _merge_overlapping_rects_simple(
    rects: list[tuple[int, int, int, int]],
    merge_distance: int = 10,
) -> list[tuple[int, int, int, int]]:
    """Simple fallback: merge rects that overlap or are within merge_distance px."""
    if not rects:
        return []
    merged = list(rects)
    changed = True
    while changed:
        changed = False
        new_merged: list[tuple[int, int, int, int]] = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            ax1, ay1, ax2, ay2 = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = merged[j]
                if (
                    ax1 - merge_distance <= bx2
                    and ax2 + merge_distance >= bx1
                    and ay1 - merge_distance <= by2
                    and ay2 + merge_distance >= by1
                ):
                    ax1 = min(ax1, bx1)
                    ay1 = min(ay1, by1)
                    ax2 = max(ax2, bx2)
                    ay2 = max(ay2, by2)
                    used[j] = True
                    changed = True
            new_merged.append((ax1, ay1, ax2, ay2))
            used[i] = True
        merged = new_merged
    return merged


def _pad_rect(
    rect: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    pad_ratio: float = _TILE_PAD_RATIO,
) -> tuple[int, int, int, int]:
    """Add *dynamic* padding around a rectangle, clamped to image bounds.

    Padding adapts to region size:
      - Small regions (<300px short side) get extra padding to avoid
        clipping ascenders, descenders, and decorated initials.
      - The vertical pad is always >= horizontal pad because manuscript
        text lines have ascenders/descenders that extend above/below.
    """
    x1, y1, x2, y2 = rect
    rw = x2 - x1
    rh = y2 - y1
    short_side = min(rw, rh)

    # Base pad from ratio
    base_px = max(4, int(round(rw * pad_ratio)))
    base_py = max(4, int(round(rh * pad_ratio)))

    # Dynamic boost for small regions (ascender/descender protection)
    if short_side < 300:
        boost = max(8, int(round(short_side * 0.12)))
        base_px += boost
        base_py += boost

    # Vertical pad should be >= horizontal pad (text has vertical extenders)
    py = max(base_py, base_px)
    px = base_px

    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(img_w, x2 + px),
        min(img_h, y2 + py),
    )


def _order_tiles_reading(
    rects: list[tuple[int, int, int, int]],
    img_w: int,
) -> list[tuple[int, int, int, int]]:
    """Sort tiles in reading order with adaptive column detection.

    Instead of a fixed 45% x-threshold, uses gap-based clustering:
      1. Sort x-centers, find the largest horizontal gap.
      2. If the gap is > 8% of page width AND both sides have tiles,
         treat as two columns.
      3. For each column group: sort top→bottom.
      4. Marginal regions (very narrow, at page edges) are appended last
         to avoid polluting the main text flow.
    """
    if len(rects) <= 1:
        return list(rects)

    # --- Separate marginal regions from main text ---
    main_rects: list[tuple[int, int, int, int]] = []
    marginal_rects: list[tuple[int, int, int, int]] = []
    for r in rects:
        rw = r[2] - r[0]
        # Marginal: very narrow (<12% of page width) AND at page edges
        is_marginal = (
            rw < float(img_w) * 0.12
            and (r[0] < float(img_w) * 0.08 or r[2] > float(img_w) * 0.92)
        )
        if is_marginal:
            marginal_rects.append(r)
        else:
            main_rects.append(r)

    if not main_rects:
        # All marginal → just sort top-to-bottom
        return sorted(rects, key=lambda r: (r[1], r[0]))

    # --- Adaptive column detection via gap analysis ---
    x_centers = sorted([(r[0] + r[2]) / 2.0 for r in main_rects])

    # Find largest gap between consecutive x-centers
    best_gap = 0.0
    best_gap_pos = 0.0
    for i in range(len(x_centers) - 1):
        gap = x_centers[i + 1] - x_centers[i]
        if gap > best_gap:
            best_gap = gap
            best_gap_pos = (x_centers[i] + x_centers[i + 1]) / 2.0

    # Two columns if gap > 8% of page width
    is_two_col = best_gap > float(img_w) * 0.08

    if is_two_col:
        left = [(r, (r[0] + r[2]) / 2.0) for r in main_rects if (r[0] + r[2]) / 2.0 < best_gap_pos]
        right = [(r, (r[0] + r[2]) / 2.0) for r in main_rects if (r[0] + r[2]) / 2.0 >= best_gap_pos]
        if left and right:
            left_sorted = sorted(left, key=lambda t: t[0][1])
            right_sorted = sorted(right, key=lambda t: t[0][1])
            ordered = [r for r, _ in left_sorted] + [r for r, _ in right_sorted]
        else:
            ordered = sorted(main_rects, key=lambda r: (r[1], r[0]))
    else:
        ordered = sorted(main_rects, key=lambda r: (r[1], r[0]))

    # Marginal regions last, sorted top→bottom
    if marginal_rects:
        ordered.extend(sorted(marginal_rects, key=lambda r: (r[1], r[0])))

    return ordered
