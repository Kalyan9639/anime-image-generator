"""
Robust 6-Feature Topological and Statistical ASCII Classifier.

Distinguishes between structured engineering diagrams (flowcharts, sequence diagrams,
network topologies, box tables) and aesthetic ASCII art pictures (tonal portraits,
line-art sketches, unicode block/braille art).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple


CategoryType = Literal["diagram", "art"]
SubtypeType = Literal["diagram:box", "diagram:unicode", "art:tone", "art:line", "art:block"]


@dataclass
class DetectionResult:
    """Detailed result of ASCII classification with confidence and diagnostic scores."""
    category: CategoryType
    subtype: SubtypeType
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


# Standard character sets
UNICODE_BOX_CORNERS = set("┌┐└┘├┤┬┴┼╔╗╚╝╠╣╦╩╬")
UNICODE_BOX_LINES = set("─│═║┄┆┈┊")
ASCII_BOX_CORNERS = set("+.#'*")
ASCII_HORIZ_LINES = set("-=_~")
ASCII_VERT_LINES = set("|!")
HEAVY_SHADING_CHARS = set("@#%&$8BMW0Q")
TONAL_RAMP_CHARS = set("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")
BRAILLE_START = 0x2800
BRAILLE_END = 0x28FF
BLOCK_START = 0x2580
BLOCK_END = 0x259F


def _pad_lines(lines: List[str]) -> Tuple[List[str], int, int]:
    """Pads all lines with spaces to a rectangular matrix."""
    if not lines:
        return [], 0, 0
    max_w = max(len(l) for l in lines)
    padded = [l.ljust(max_w) for l in lines]
    return padded, len(padded), max_w


def _calculate_box_corner_score(grid: List[str], h: int, w: int) -> Tuple[float, int]:
    """
    Scans for true geometric corners where a horizontal line segment meets
    a vertical line segment at a 90-degree intersection.
    """
    if h < 2 or w < 2:
        return 0.0, 0

    verified_corners = 0

    for r in range(h):
        for c in range(w):
            ch = grid[r][c]
            is_ascii_corner = ch in ASCII_BOX_CORNERS
            is_unicode_corner = ch in UNICODE_BOX_CORNERS

            if not (is_ascii_corner or is_unicode_corner):
                continue

            has_horiz = False
            has_vert = False

            # Check horizontal connectivity
            for dc in (-1, 1):
                nc = c + dc
                if 0 <= nc < w:
                    neighbor = grid[r][nc]
                    if neighbor in ASCII_HORIZ_LINES or neighbor in UNICODE_BOX_LINES or neighbor in ASCII_BOX_CORNERS:
                        has_horiz = True
                        break

            # Check vertical connectivity
            for dr in (-1, 1):
                nr = r + dr
                if 0 <= nr < h:
                    neighbor = grid[nr][c]
                    if neighbor in ASCII_VERT_LINES or neighbor in UNICODE_BOX_LINES or neighbor in ASCII_BOX_CORNERS:
                        has_vert = True
                        break

            if has_horiz and has_vert:
                verified_corners += 1

    # Normalize by estimated minimum corners for a diagram (at least 4 corners for a single box)
    score = min(1.0, verified_corners / 4.0)
    return score, verified_corners


def _calculate_arrow_score(text: str) -> Tuple[float, int]:
    """Detects directional relational arrows connecting entities."""
    patterns = [
        r"-->", r"<--", r"==>", r"<==", r"<->", r"->>", r"<<-",
        r"(?<=[-|\s])->(?=[a-zA-Z0-9_\s])",
        r"(?<=[a-zA-Z0-9_\s])<-(?=[-|\s])",
        r"\^[\s]*\|", r"\|[\s]*v"
    ]
    total_arrows = 0
    for pat in patterns:
        matches = re.findall(pat, text)
        total_arrows += len(matches)

    score = min(1.0, total_arrows / 2.0)
    return score, total_arrows


def _calculate_orthogonal_runs(grid: List[str], h: int, w: int) -> float:
    """Measures the proportion of characters participating in continuous orthogonal lines."""
    if h == 0 or w == 0:
        return 0.0

    horiz_chars = 0
    for r in range(h):
        line = grid[r]
        for match in re.finditer(r"([-=_~─═]{3,})", line):
            horiz_chars += len(match.group(1))

    vert_chars = 0
    for c in range(w):
        col_str = "".join(grid[r][c] for r in range(h))
        for match in re.finditer(r"([|!│║]{2,})", col_str):
            vert_chars += len(match.group(1))

    total_non_space = sum(1 for r in range(h) for c in range(w) if grid[r][c] != " ")
    if total_non_space == 0:
        return 0.0

    return min(1.0, (horiz_chars + vert_chars) / total_non_space)


def _calculate_enclosed_labels(text: str) -> Tuple[float, int]:
    """Detects isolated human-readable words enclosed inside box boundaries or brackets."""
    # Must contain at least two alphanumeric characters (prevents single-letter doorknobs 'o' or '___' lines)
    patterns = [
        r"\|\s*([A-Za-z0-9_]*[A-Za-z0-9]{2,}[A-Za-z0-9_]*(?:\s+[A-Za-z0-9_]+)*)\s*\|",
        r"\[\s*([A-Za-z0-9_]*[A-Za-z0-9]{2,}[A-Za-z0-9_]*(?:\s+[A-Za-z0-9_]+)*)\s*\]",
        r"\(\s*([A-Za-z0-9_]*[A-Za-z0-9]{2,}[A-Za-z0-9_]*(?:\s+[A-Za-z0-9_]+)*)\s*\)",
        r"│\s*([A-Za-z0-9_]*[A-Za-z0-9]{2,}[A-Za-z0-9_]*(?:\s+[A-Za-z0-9_]+)*)\s*│",
    ]
    total_labels = 0
    for pat in patterns:
        total_labels += len(re.findall(pat, text))

    score = min(1.0, total_labels / 2.0)
    return score, total_labels


def _calculate_tonal_shading(text: str, total_non_space: int) -> float:
    """Measures the density of heavy typographic shading characters used for halftoning."""
    if total_non_space == 0:
        return 0.0
    heavy_count = sum(1 for ch in text if ch in HEAVY_SHADING_CHARS)
    return heavy_count / total_non_space


def _calculate_block_and_braille(text: str, total_non_space: int) -> float:
    """Measures the presence of Unicode block elements and Braille glyphs."""
    if total_non_space == 0:
        return 0.0
    special_count = 0
    for ch in text:
        cp = ord(ch)
        if (BLOCK_START <= cp <= BLOCK_END) or (BRAILLE_START <= cp <= BRAILLE_END):
            special_count += 1
    return special_count / total_non_space


def _calculate_line_sketch_score(text: str, total_non_space: int) -> float:
    """Detects sparse line art / ASCII cartoons using slashes, backslashes, eyes, carets."""
    if total_non_space == 0:
        return 0.0
    sketch_chars = set("/\\()^><_.,oO'\"")
    sketch_count = sum(1 for ch in text if ch in sketch_chars)
    return sketch_count / total_non_space


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def classify_ascii(text: str) -> DetectionResult:
    """
    Evaluates 6 topological and statistical features to accurately classify
    ASCII input as an engineering diagram or aesthetic art.
    """
    cleaned = _ANSI_RE.sub("", text)
    raw_lines = [l.rstrip("\r\n").expandtabs(4) for l in cleaned.splitlines()]
    text = "\n".join(raw_lines)
    # Remove leading and trailing completely blank lines
    while raw_lines and not raw_lines[0].strip():
        raw_lines.pop(0)
    while raw_lines and not raw_lines[-1].strip():
        raw_lines.pop()

    if not raw_lines:
        return DetectionResult(
            category="art",
            subtype="art:line",
            confidence=0.5,
            scores={},
            explanation="Empty or whitespace-only input.",
        )

    grid, h, w = _pad_lines(raw_lines)
    total_non_space = sum(1 for r in range(h) for c in range(w) if grid[r][c] != " ")

    if total_non_space == 0:
        return DetectionResult(
            category="art",
            subtype="art:line",
            confidence=0.5,
            scores={},
            explanation="Input contains only spaces.",
        )

    # 1. Box Corner Score (F1)
    f1_corner_score, n_corners = _calculate_box_corner_score(grid, h, w)

    # 2. Arrow Score (F2)
    f2_arrow_score, n_arrows = _calculate_arrow_score(text)

    # 3. Orthogonal Run Score (F3)
    f3_run_score = _calculate_orthogonal_runs(grid, h, w)

    # 4. Enclosed Label Score (F4)
    f4_label_score, n_labels = _calculate_enclosed_labels(text)

    # 5. Tonal Shading Score (F5)
    f5_shading_score = _calculate_tonal_shading(text, total_non_space)

    # 6. Unicode Block & Braille Score (F6)
    f6_block_score = _calculate_block_and_braille(text, total_non_space)

    # Secondary: Line sketch score
    line_sketch_score = _calculate_line_sketch_score(text, total_non_space)

    scores = {
        "box_corners": float(f1_corner_score),
        "corner_count": float(n_corners),
        "arrow_connectivity": float(f2_arrow_score),
        "arrow_count": float(n_arrows),
        "orthogonal_runs": float(f3_run_score),
        "enclosed_labels": float(f4_label_score),
        "tonal_shading": float(f5_shading_score),
        "block_braille": float(f6_block_score),
        "line_sketch": float(line_sketch_score),
    }

    # Diagram Evidence vs Art Evidence
    has_unicode_box = any(ch in UNICODE_BOX_CORNERS or ch in UNICODE_BOX_LINES for ch in text)
    diagram_weight = (
        0.35 * f1_corner_score +
        0.25 * f2_arrow_score +
        0.25 * f3_run_score +
        0.15 * f4_label_score
    )

    # Check for Block / Braille Art first
    if f6_block_score > 0.20:
        conf = min(0.99, 0.70 + f6_block_score * 0.30)
        return DetectionResult(
            category="art",
            subtype="art:block",
            confidence=float(conf),
            scores=scores,
            explanation=f"High density of Unicode block/braille elements ({f6_block_score*100:.1f}%).",
        )

    # Check for ASCII Scene / Landscape Art (e.g. house with roof, windows, tree foliage, sun)
    # Even if drawn windows/doors have corners, if there are NO arrows and NO text labels,
    # and there is high tonal shading or line sketches, it is an art illustration scene.
    if n_arrows == 0 and n_labels == 0 and (f5_shading_score > 0.08 or line_sketch_score > 0.18):
        subtype = "art:tone" if f5_shading_score > 0.08 else "art:line"
        conf = min(0.95, 0.70 + (f5_shading_score + line_sketch_score) * 0.40)
        return DetectionResult(
            category="art",
            subtype=subtype,
            confidence=float(conf),
            scores=scores,
            explanation=f"ASCII art illustration/scene: contains drawn geometry (roof/windows) with {f5_shading_score*100:.1f}% shading (e.g. foliage) and no relational arrows.",
        )

    # Directory trees (├── / └──) without closed right-side box borders render cleanest in crisp mode
    if ("├──" in text or "└──" in text) and n_arrows == 0 and n_labels == 0 and "┐" not in text and "┘" not in text:
        return DetectionResult(
            category="art",
            subtype="art:line",
            confidence=0.90,
            scores=scores,
            explanation="Directory tree structure detected (preserves crisp tree glyphs cleanly).",
        )

    # Strong Diagram Indicators: verified corners, arrows, or enclosed box labels
    if (n_corners >= 4 and f3_run_score > 0.15) or (n_corners >= 2 and n_arrows >= 1) or (has_unicode_box and f3_run_score > 0.2):
        conf = min(0.99, 0.65 + diagram_weight * 0.35)
        subtype = "diagram:unicode" if has_unicode_box else "diagram:box"
        explanation_parts = []
        if n_corners > 0:
            explanation_parts.append(f"{n_corners} verified 90° box corner(s)")
        if n_arrows > 0:
            explanation_parts.append(f"{n_arrows} relational arrow vector(s)")
        if f3_run_score > 0.1:
            explanation_parts.append(f"{f3_run_score*100:.1f}% orthogonal grid lines")
        if n_labels > 0:
            explanation_parts.append(f"{n_labels} enclosed label(s)")

        return DetectionResult(
            category="diagram",
            subtype=subtype,
            confidence=float(conf),
            scores=scores,
            explanation="Detected diagram structure: " + ", ".join(explanation_parts) + ".",
        )

    # Strong Tonal Art Indicators
    if f5_shading_score > 0.15:
        conf = min(0.99, 0.60 + f5_shading_score * 0.40)
        return DetectionResult(
            category="art",
            subtype="art:tone",
            confidence=float(conf),
            scores=scores,
            explanation=f"Tonal shading characters detected ({f5_shading_score*100:.1f}% density) with gradient variation.",
        )

    # Sparser sketches (line art, cartoons, faces)
    if line_sketch_score > 0.35 and n_corners == 0 and n_arrows == 0:
        conf = min(0.95, 0.60 + line_sketch_score * 0.35)
        return DetectionResult(
            category="art",
            subtype="art:line",
            confidence=float(conf),
            scores=scores,
            explanation=f"Line-art sketch using structural typography ({line_sketch_score*100:.1f}% strokes) without box grammar.",
        )

    # Fallback comparison between diagram and art weights
    if diagram_weight > 0.25:
        conf = min(0.90, 0.50 + diagram_weight * 0.40)
        return DetectionResult(
            category="diagram",
            subtype="diagram:box",
            confidence=float(conf),
            scores=scores,
            explanation=f"Weighted heuristic favors diagram ({diagram_weight*100:.1f}% alignment score).",
        )
    else:
        conf = min(0.90, 0.55 + max(f5_shading_score, line_sketch_score) * 0.35)
        subtype = "art:tone" if f5_shading_score > 0.05 else "art:line"
        return DetectionResult(
            category="art",
            subtype=subtype,
            confidence=float(conf),
            scores=scores,
            explanation="Classified as ASCII art picture (lacks closed orthogonal box geometry).",
        )
