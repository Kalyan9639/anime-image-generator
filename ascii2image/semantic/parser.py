"""
Semantic ASCII Layout Parser.

Parses boxed ASCII diagrams (including nested containers, counts like `x4` or `16x16`,
inline `key=value` attributes like `view=iso`, and connecting arrows) into a
normalized spatial scene hierarchy that locks entity positions and proportions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..diagram import _UNICODE_DIAGRAM_TRANS
from ..normalize import sanitize_ascii_text


_CORNER_CHARS = set("+.#'`")
_HORIZ_CHARS = set("-=_~")
_VERT_CHARS = set("|!")

_KV_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)=([^\s,;]+)")
_GRID_RE = re.compile(r"\b(\d+)\s*[xX×]\s*(\d+)\b")
_COUNT_RE = re.compile(r"\b[xX×](\d+)\b")


@dataclass
class AsciiBoxNode:
    id: str
    label: str
    raw_text: str
    grid_rect: Tuple[int, int, int, int]          # (r0, c0, r1, c1) inclusive
    norm_bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in [0, 1]
    attributes: Dict[str, str] = field(default_factory=dict)
    count: int = 1
    grid_repeat: Optional[Tuple[int, int]] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)


@dataclass
class AsciiLayout:
    width_chars: int
    height_chars: int
    boxes: List[AsciiBoxNode] = field(default_factory=list)
    global_attributes: Dict[str, str] = field(default_factory=dict)
    free_text: List[str] = field(default_factory=list)

    def to_summary_dict(self) -> Dict[str, object]:
        return {
            "width_chars": self.width_chars,
            "height_chars": self.height_chars,
            "global_attributes": self.global_attributes,
            "boxes": [
                {
                    "id": b.id,
                    "label": b.label,
                    "norm_bbox": [round(v, 3) for v in b.norm_bbox],
                    "count": b.count,
                    "grid_repeat": list(b.grid_repeat) if b.grid_repeat else None,
                    "attributes": b.attributes,
                    "parent_id": b.parent_id,
                    "children_ids": b.children_ids,
                }
                for b in self.boxes
            ],
            "free_text": self.free_text,
        }


def _is_horiz_edge(grid: List[str], r: int, c0: int, c1: int) -> bool:
    if c1 - c0 < 2:
        return False
    for c in range(c0 + 1, c1):
        ch = grid[r][c]
        if ch not in _HORIZ_CHARS and ch not in _CORNER_CHARS and ch not in ("v", "^"):
            return False
    return True


def _is_vert_edge(grid: List[str], c: int, r0: int, r1: int) -> bool:
    if r1 - r0 < 2:
        return False
    for r in range(r0 + 1, r1):
        ch = grid[r][c]
        if ch not in _VERT_CHARS and ch not in _CORNER_CHARS and ch not in ("<", ">"):
            return False
    return True


def _parse_label_and_attrs(raw_lines: List[str]) -> Tuple[str, Dict[str, str], int, Optional[Tuple[int, int]]]:
    joined = " ".join(ln.strip() for ln in raw_lines if ln.strip()).strip()
    attrs: Dict[str, str] = {}
    for m in _KV_RE.finditer(joined):
        attrs[m.group(1).lower()] = m.group(2)
    cleaned = _KV_RE.sub("", joined)

    grid_repeat: Optional[Tuple[int, int]] = None
    gm = _GRID_RE.search(cleaned)
    if gm:
        grid_repeat = (int(gm.group(1)), int(gm.group(2)))
        cleaned = _GRID_RE.sub("", cleaned, count=1)

    count = 1
    cm = _COUNT_RE.search(cleaned)
    if cm:
        count = max(1, int(cm.group(1)))
        cleaned = _COUNT_RE.sub("", cleaned, count=1)

    label = re.sub(r"\s+", " ", cleaned).strip(" |:-")
    return label, attrs, count, grid_repeat


def parse_semantic_ascii(text: str) -> AsciiLayout:
    """
    Extracts rectangular boxes (including nested sub-boxes), labels, counts,
    and attributes from ASCII/Unicode diagrams.
    """
    cleaned, _ = sanitize_ascii_text(text, repair_alignment=True, strip_blank_edges=True)
    translated = cleaned.translate(_UNICODE_DIAGRAM_TRANS)
    lines = translated.split("\n") if translated else []
    if not lines:
        return AsciiLayout(width_chars=0, height_chars=0)

    max_w = max(len(ln) for ln in lines)
    grid = [ln.ljust(max_w) for ln in lines]
    h = len(grid)

    # 1. Detect all rectangular box candidates (r0, c0, r1, c1)
    candidates: List[Tuple[int, int, int, int]] = []
    for r0 in range(h - 2):
        for c0 in range(max_w - 2):
            if grid[r0][c0] not in _CORNER_CHARS:
                continue
            # Scan right along row r0 for matching top-right corner
            for c1 in range(c0 + 2, max_w):
                if grid[r0][c1] not in _CORNER_CHARS:
                    continue
                if not _is_horiz_edge(grid, r0, c0, c1):
                    continue
                # Scan down from r0 for bottom edge
                for r1 in range(r0 + 2, h):
                    if grid[r1][c0] not in _CORNER_CHARS or grid[r1][c1] not in _CORNER_CHARS:
                        continue
                    if (
                        _is_horiz_edge(grid, r1, c0, c1)
                        and _is_vert_edge(grid, c0, r0, r1)
                        and _is_vert_edge(grid, c1, r0, r1)
                    ):
                        candidates.append((r0, c0, r1, c1))
                        break

    # Deduplicate identical rects and sort largest-area first (outer containers first)
    unique_rects = sorted(
        set(candidates),
        key=lambda rc: ((rc[2] - rc[0]) * (rc[3] - rc[1]), -rc[0], -rc[1]),
        reverse=True,
    )

    boxes: List[AsciiBoxNode] = []
    for idx, (r0, c0, r1, c1) in enumerate(unique_rects):
        norm_bbox = (
            c0 / max(1, max_w - 1),
            r0 / max(1, h - 1),
            c1 / max(1, max_w - 1),
            r1 / max(1, h - 1),
        )
        boxes.append(
            AsciiBoxNode(
                id=f"box_{idx}",
                label="",
                raw_text="",
                grid_rect=(r0, c0, r1, c1),
                norm_bbox=norm_bbox,
            )
        )

    # 2. Compute parent-child nesting (smallest strictly enclosing box is parent)
    for child in boxes:
        cr0, cc0, cr1, cc1 = child.grid_rect
        best_parent: Optional[AsciiBoxNode] = None
        best_area = 10**9
        for cand in boxes:
            if cand is child:
                continue
            pr0, pc0, pr1, pc1 = cand.grid_rect
            if pr0 < cr0 and pc0 < cc0 and pr1 > cr1 and pc1 > cc1:
                area = (pr1 - pr0) * (pc1 - pc0)
                if area < best_area:
                    best_area = area
                    best_parent = cand
        if best_parent is not None:
            child.parent_id = best_parent.id
            best_parent.children_ids.append(child.id)

    # 3. Extract interior text belonging strictly to each box (excluding child box interiors)
    box_by_id = {b.id: b for b in boxes}
    used_mask = [[False] * max_w for _ in range(h)]

    for box in boxes:
        r0, c0, r1, c1 = box.grid_rect
        child_rects = [box_by_id[cid].grid_rect for cid in box.children_ids]
        own_lines: List[str] = []
        for r in range(r0 + 1, r1):
            row_chars = []
            for c in range(c0 + 1, c1):
                in_child = any(cr0 <= r <= cr1 and cc0 <= c <= cc1 for cr0, cc0, cr1, cc1 in child_rects)
                if not in_child:
                    row_chars.append(grid[r][c])
                    used_mask[r][c] = True
                else:
                    row_chars.append(" ")
            seg = "".join(row_chars).strip()
            if seg:
                own_lines.append(seg)
        for r in range(r0, r1 + 1):
            used_mask[r][c0] = True
            used_mask[r][c1] = True
        for c in range(c0, c1 + 1):
            used_mask[r0][c] = True
            used_mask[r1][c] = True

        label, attrs, count, grid_repeat = _parse_label_and_attrs(own_lines)
        box.raw_text = "\n".join(own_lines)
        box.label = label or box.id
        box.attributes = attrs
        box.count = count
        box.grid_repeat = grid_repeat
        if label:
            slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:24]
            if slug:
                # Keep unique IDs while preserving parent/child links if we rename later; keep box_idx stable
                pass

    # 4. Collect global attributes & free text outside all boxes
    free_text: List[str] = []
    global_attrs: Dict[str, str] = {}
    for r in range(h):
        chars = [grid[r][c] if not used_mask[r][c] else " " for c in range(max_w)]
        seg = "".join(chars).strip()
        if seg and not all(ch in "-=|><^v+ " for ch in seg):
            free_text.append(seg)
            for m in _KV_RE.finditer(seg):
                global_attrs[m.group(1).lower()] = m.group(2)

    for b in boxes:
        if "view" in b.attributes and "view" not in global_attrs:
            global_attrs["view"] = b.attributes["view"]

    return AsciiLayout(
        width_chars=max_w,
        height_chars=h,
        boxes=boxes,
        global_attributes=global_attrs,
        free_text=free_text,
    )
