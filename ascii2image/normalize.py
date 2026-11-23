"""
Shared ASCII Text Normalization & Alignment Repair Utilities.

Sanitizes copy-pasted ASCII diagrams and ASCII art without disturbing intended
2D column geometry:
- Strips ANSI escape codes, BOM, zero-width characters, and illegal XML control chars.
- Normalizes Unicode spaces (NBSP, thin/en/em spaces) to ASCII spaces.
- Strips outer Markdown code fences (``` ... ```) when pasted from chat/docs.
- Expands tabs consistently.
- Safely repairs a missing first-line indentation caused by drag-selecting an
  indented diagram from its first visible character (e.g. `+---+` shifted to col 0
  while `| ... |` starts at col 20).
"""

from __future__ import annotations

import re
from typing import List, Tuple

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Illegal XML 1.0 control characters (keep \t=0x09, \n=0x0A, \r=0x0D)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Zero-width / BOM characters that silently shift visual column counts
_ZERO_WIDTH_RE = re.compile(r"[\ufeff\u200b\u200c\u200d\u2060]")
# Unicode non-ASCII horizontal space characters -> standard space
_UNICODE_SPACES_RE = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f]")

# Box top border at column 0: e.g. +-------+, .-------., ┌───────┐, ╔═══════╗, ┏━━━━━━━┓, ╭───────╮
# Allows embedded junction/arrow markers like v, ^, +, ┬, ┴, ╦, ╩, ┳, ┻ inside the top bar.
_BOX_TOP_RE = re.compile(
    r"^([+.┌╔┏╭][-=─═━+v^┬┴╦╩┳┻.]{1,}[+.┐╗┓╮])"
)
_VERT_WALL_CHARS = set("|│║┃")

# Pure ASCII art peak/cap on line 0 (no alphanumerics), e.g. /\_____/\ or .-"""-.
_ART_CAP_RE = re.compile(r"^([/\\._\-^~`'\"(),*=+@#]{2,})\s*$")


def strip_markdown_fences(text: str) -> str:
    """
    If the text is wrapped in a single Markdown fenced code block (``` ... ```),
    unwrap it while preserving the exact inner indentation.
    """
    lines = text.split("\n")
    # Trim leading/trailing completely empty lines for fence inspection
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    end = len(lines) - 1
    while end >= start and not lines[end].strip():
        end -= 1

    if end - start >= 2:
        first_stripped = lines[start].strip()
        last_stripped = lines[end].strip()
        if (
            (first_stripped.startswith("```") and last_stripped == "```")
            or (first_stripped.startswith("~~~") and last_stripped == "~~~")
        ):
            return "\n".join(lines[start + 1 : end])
    return text


def repair_first_line_indent(lines: List[str]) -> Tuple[List[str], int]:
    """
    Detects and repairs the common copy-paste bug where the first non-empty line
    loses its leading spaces (col 0) while the lines immediately below remain
    indented at column k > 0.

    Returns (repaired_lines, spaces_added). Only modifies line 0 when exact
    geometric alignment with line 1 is verified, preventing false positives.
    """
    if len(lines) < 2:
        return lines, 0

    first_idx = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if first_idx is None or first_idx + 1 >= len(lines):
        return lines, 0

    line0 = lines[first_idx]
    line1 = lines[first_idx + 1]
    if not line1.strip():
        return lines, 0

    indent0 = len(line0) - len(line0.lstrip(" "))
    indent1 = len(line1) - len(line1.lstrip(" "))

    if indent0 != 0 or indent1 < 2:
        return lines, 0

    # Case 1: Box top border at col 0 whose exact width matches vertical walls on line 1
    m = _BOX_TOP_RE.match(line0)
    if m:
        box_w = len(m.group(1))
        right_col = indent1 + box_w - 1
        if (
            indent1 < len(line1)
            and line1[indent1] in _VERT_WALL_CHARS
            and right_col < len(line1)
            and line1[right_col] in _VERT_WALL_CHARS
        ):
            repaired = list(lines)
            repaired[first_idx] = (" " * indent1) + line0
            return repaired, indent1

    # Case 2: Symmetrical ASCII art top cap (e.g. /\ or /\_____/\ or .-"""-.)
    # where line 1 opens with '/' at indent1 and closes its shape with '\' matching width
    m_cap = _ART_CAP_RE.match(line0)
    if m_cap:
        cap = m_cap.group(1)
        cap_w = len(cap)
        if indent1 < len(line1) and line1[indent1] == "/":
            # Check for matching '\' on line 1 at indent1 + cap_w + 1 (or +2 for wider slope)
            for delta in (1, 2):
                close_col = indent1 + cap_w + delta
                if close_col < len(line1) and line1[close_col] == "\\":
                    shift = indent1 + 1
                    repaired = list(lines)
                    repaired[first_idx] = (" " * shift) + line0
                    return repaired, shift

    return lines, 0


def sanitize_ascii_text(
    text: str,
    *,
    tab_size: int = 4,
    repair_alignment: bool = True,
    strip_blank_edges: bool = True,
) -> Tuple[str, List[str]]:
    """
    Normalizes raw ASCII input text for reliable classification and rendering.

    Returns (cleaned_text, notes) where `notes` lists any automatic repairs
    applied (e.g. unwrapped Markdown fences, repaired first-line indentation).
    """
    notes: List[str] = []
    cleaned = _ANSI_RE.sub("", text or "")
    cleaned = _ZERO_WIDTH_RE.sub("", cleaned)
    cleaned = _CTRL_RE.sub("", cleaned)
    cleaned = cleaned.replace("\u3000", "  ")
    cleaned = _UNICODE_SPACES_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    unwrapped = strip_markdown_fences(cleaned)
    if unwrapped != cleaned:
        notes.append("Unwrapped outer Markdown code fences.")
        cleaned = unwrapped

    lines = [ln.expandtabs(tab_size).rstrip(" ") for ln in cleaned.split("\n")]

    if strip_blank_edges:
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

    if repair_alignment and lines:
        lines, shift = repair_first_line_indent(lines)
        if shift > 0:
            notes.append(f"Auto-repaired misaligned first line (+{shift} leading spaces).")

    return "\n".join(lines), notes
