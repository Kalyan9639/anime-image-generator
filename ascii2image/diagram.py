"""
Diagram Rendering & Styling Pipeline.

Converts structured ASCII and Unicode box-drawing diagrams (flowcharts,
sequence diagrams, architecture blocks, tables) into clean, scalable SVG
vector graphics with modern themes and safe cross-platform fallbacks.
Supports svgbob (bundled binary) and aafigure (100% pure-Python fallback).
"""

from __future__ import annotations

import io
import os
import re
import shutil
import stat
import subprocess
from typing import List, Literal, Optional

import aafigure
import markdown_svgbob

from .normalize import sanitize_ascii_text


EngineType = Literal["svgbob", "aafigure", "aasvg", "auto"]
ThemeType = Literal["light", "dark", "blueprint", "tinted"]

# Map Unicode box-drawing, dash, diagonal & arrow symbols to svgbob/aafigure-compatible ASCII
_UNICODE_DIAGRAM_TRANS = str.maketrans({
    # Light, double, heavy, and mixed corners/junctions
    "┌": "+", "┐": "+", "└": "+", "┘": "+",
    "├": "+", "┤": "+", "┬": "+", "┴": "+", "┼": "+",
    "╔": "+", "╗": "+", "╚": "+", "╝": "+",
    "╠": "+", "╣": "+", "╦": "+", "╩": "+", "╬": "+",
    "┏": "+", "┓": "+", "┗": "+", "┛": "+",
    "┣": "+", "┫": "+", "┳": "+", "┻": "+", "╋": "+",
    "╒": "+", "╓": "+", "╕": "+", "╖": "+",
    "╘": "+", "╙": "+", "╛": "+", "╜": "+",
    "╞": "+", "╟": "+", "╡": "+", "╢": "+",
    "╤": "+", "╥": "+", "╧": "+", "╨": "+", "╪": "+", "╫": "+",
    "┍": "+", "┎": "+", "┑": "+", "┒": "+",
    "┕": "+", "┖": "+", "┙": "+", "┚": "+",
    "┝": "+", "┞": "+", "┟": "+", "┠": "+", "┡": "+", "┢": "+",
    "┥": "+", "┦": "+", "┧": "+", "┨": "+", "┩": "+", "┪": "+",
    "┭": "+", "┮": "+", "┯": "+", "┰": "+", "┱": "+", "┲": "+",
    "┵": "+", "┶": "+", "┷": "+", "┸": "+", "┹": "+", "┺": "+",
    "┽": "+", "┾": "+", "┿": "+", "╀": "+", "╁": "+", "╂": "+",
    "╃": "+", "╄": "+", "╅": "+", "╆": "+", "╇": "+", "╈": "+", "╉": "+", "╊": "+",
    # Rounded corners
    "╭": ".", "╮": ".", "╰": "'", "╯": "'",
    # Horizontal lines & dashes
    "─": "-", "━": "-", "┄": "-", "┅": "-", "┈": "-", "┉": "-",
    "╴": "-", "╶": "-", "╸": "-", "╺": "-",
    "—": "-", "–": "-", "−": "-",
    "═": "=",
    # Vertical lines
    "│": "|", "┃": "|", "║": "|", "┆": "|", "┇": "|", "┊": "|", "┋": "|",
    "╵": "|", "╷": "|", "╹": "|", "╻": "|",
    # Diagonals
    "╱": "/", "╲": "\\", "╳": "X",
    # Arrows
    "▶": ">", "►": ">", "→": ">", "⟶": ">", "⇒": ">", "➔": ">", "➜": ">", "➝": ">", "➞": ">", "➡": ">", "▸": ">", "▷": ">",
    "◀": "<", "◄": "<", "←": "<", "⟵": "<", "⇐": "<", "⬅": "<", "◂": "<", "◁": "<",
    "▲": "^", "↑": "^", "⇑": "^", "⬆": "^", "▴": "^", "△": "^",
    "▼": "v", "↓": "v", "⇓": "v", "⬇": "v", "▾": "v", "▽": "v",
})


THEME_CONFIGS = {
    "light": {
        "bg_color": "#ffffff",
        "stroke_color": "#1e293b",       # Slate 800
        "fill_color": "#f8fafc",         # Slate 50
        "text_color": "#0f172a",         # Slate 900
        "accent_color": "#3b82f6",       # Blue 500
        "font_family": "Consolas, 'DejaVu Sans Mono', 'Liberation Mono', 'Courier New', monospace",
    },
    "dark": {
        "bg_color": "#0f172a",           # Slate 900
        "stroke_color": "#38bdf8",       # Sky 400
        "fill_color": "#1e293b",         # Slate 800
        "text_color": "#f8fafc",         # Slate 50
        "accent_color": "#818cf8",       # Indigo 400
        "font_family": "Consolas, 'DejaVu Sans Mono', 'Liberation Mono', 'Courier New', monospace",
    },
    "blueprint": {
        "bg_color": "#1e3a8a",           # Deep Blueprint Blue
        "stroke_color": "#93c5fd",       # Light CAD Blue
        "fill_color": "#172554",         # Midnight Blue
        "text_color": "#ffffff",         # Pure White
        "accent_color": "#67e8f9",       # Cyan 300
        "font_family": "Consolas, 'DejaVu Sans Mono', 'Liberation Mono', 'Courier New', monospace",
    },
    "tinted": {
        "bg_color": "#fafafa",           # Warm Gray
        "stroke_color": "#334155",       # Slate 700
        "fill_color": "#e0f2fe",         # Soft Sky Tint
        "text_color": "#1e293b",         # Slate 800
        "accent_color": "#0284c7",       # Sky 600
        "font_family": "Consolas, 'DejaVu Sans Mono', 'Liberation Mono', 'Courier New', monospace",
    },
}


def get_available_engines() -> List[str]:
    """Returns the list of available diagram rendering engines."""
    engines = ["svgbob", "aafigure"]
    try:
        if shutil.which("aasvg"):
            engines.append("aasvg")
    except Exception:
        pass
    return engines


def _normalize_diagram_text(text: str, repair_alignment: bool = True) -> str:
    """
    Sanitizes ANSI escape codes, Unicode spaces, Markdown fences, tabs,
    repairs missing first-line indentation, and translates Unicode box-drawing
    and arrow characters into standard ASCII diagram grammar.
    """
    cleaned, _ = sanitize_ascii_text(
        text,
        tab_size=4,
        repair_alignment=repair_alignment,
        strip_blank_edges=True,
    )
    return cleaned.translate(_UNICODE_DIAGRAM_TRANS)


def _convert_rounded_corners_for_aafigure(text: str) -> str:
    """
    aafigure does not recognize svgbob's `.---.` / `'---'` rounded-box syntax
    and renders those corners as disconnected dots and literal dashes.
    Converts `.` and `'` that sit at verified 90-degree box corners into `+`.
    """
    lines = text.split("\n")
    if len(lines) < 2:
        return text
    max_w = max(len(ln) for ln in lines)
    grid = [list(ln.ljust(max_w)) for ln in lines]
    h = len(grid)
    horiz = set("-=")
    vert = set("|")

    for r in range(h):
        for c in range(max_w):
            ch = grid[r][c]
            if ch not in (".", "'", "`"):
                continue
            has_h = (c > 0 and grid[r][c - 1] in horiz) or (c + 1 < max_w and grid[r][c + 1] in horiz)
            if not has_h:
                continue
            if ch == "." and r + 1 < h and grid[r + 1][c] in vert:
                grid[r][c] = "+"
            elif ch in ("'", "`") and r > 0 and grid[r - 1][c] in vert:
                grid[r][c] = "+"

    return "\n".join("".join(row).rstrip() for row in grid)


def _find_svgbob_binary() -> Optional[str]:
    """
    Safely locates the svgbob binary without crashing on restricted PATH entries
    (a known pathlib2 PermissionError bug inside markdown_svgbob._get_usr_bin_path).
    """
    try:
        usr_bin = shutil.which("svgbob")
        if usr_bin and os.path.isfile(usr_bin):
            return usr_bin
    except Exception:
        pass

    try:
        from markdown_svgbob.wrapper import _get_pkg_bin_path

        pkg_bin = str(_get_pkg_bin_path())
        if os.path.isfile(pkg_bin):
            if os.name == "posix" and not os.access(pkg_bin, os.X_OK):
                try:
                    current_mode = os.stat(pkg_bin).st_mode
                    os.chmod(pkg_bin, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                except Exception:
                    pass
            return pkg_bin
    except Exception:
        pass

    return None


def _render_with_svgbob(text: str) -> str:
    """Renders ASCII diagram using svgbob directly via stdin/stdout."""
    bin_path = _find_svgbob_binary()
    if bin_path:
        proc = subprocess.run(
            [bin_path],
            input=text.encode("utf-8"),
            capture_output=True,
            check=True,
            timeout=15,
        )
        return proc.stdout.decode("utf-8", errors="replace")

    svg_bytes = markdown_svgbob.text2svg(text)
    if isinstance(svg_bytes, bytes):
        return svg_bytes.decode("utf-8")
    return str(svg_bytes)


def _render_with_aafigure(text: str) -> str:
    """Renders ASCII diagram using aafigure (100% pure Python)."""
    aa_text = _convert_rounded_corners_for_aafigure(text)
    bio = io.BytesIO()
    options = {
        "format": "svg",
        "scale": 1.0,
        "line_width": 2.0,
        "textual": True,
    }
    aafigure.render(aa_text, bio, options=options)
    return bio.getvalue().decode("utf-8")


def _render_with_aasvg(text: str) -> str:
    """Renders ASCII diagram using aasvg (Node.js CLI if available)."""
    proc = subprocess.run(
        ["aasvg"],
        input=text.encode("utf-8"),
        capture_output=True,
        check=True,
        timeout=15,
    )
    return proc.stdout.decode("utf-8")


def _add_svg_padding(svg_content: str, pad: int = 16) -> str:
    """
    Expands the root <svg> width/height/viewBox by `pad` pixels on each side
    so strokes along the outer edges are never clipped, and guarantees
    non-zero dimensions.
    """
    svg_match = re.search(r"<svg\b[^>]*>", svg_content, flags=re.IGNORECASE | re.DOTALL)
    if not svg_match:
        return svg_content

    open_tag = svg_match.group(0)
    w_match = re.search(r'\bwidth="([0-9.]+)(?:px)?"', open_tag)
    h_match = re.search(r'\bheight="([0-9.]+)(?:px)?"', open_tag)
    if not (w_match and h_match):
        return svg_content

    w = max(1.0, float(w_match.group(1)))
    h = max(1.0, float(h_match.group(1)))
    new_w = w + 2 * pad
    new_h = h + 2 * pad

    vb_match = re.search(r'\bviewBox="([^"]+)"', open_tag)
    if vb_match:
        parts = vb_match.group(1).replace(",", " ").split()
        if len(parts) == 4:
            vx, vy, vw, vh = [float(p) for p in parts]
            vw = max(1.0, vw)
            vh = max(1.0, vh)
            new_vb = f"{vx - pad:.1f} {vy - pad:.1f} {vw + 2 * pad:.1f} {vh + 2 * pad:.1f}"
        else:
            new_vb = f"{-pad} {-pad} {new_w:.1f} {new_h:.1f}"
        new_open = re.sub(r'\bviewBox="[^"]+"', f'viewBox="{new_vb}"', open_tag, count=1)
    else:
        new_vb = f"{-pad} {-pad} {new_w:.1f} {new_h:.1f}"
        new_open = open_tag[:-1] + f' viewBox="{new_vb}">'

    new_open = re.sub(r'\bwidth="[0-9.]+(?:px)?"', f'width="{new_w:.1f}"', new_open, count=1)
    new_open = re.sub(r'\bheight="[0-9.]+(?:px)?"', f'height="{new_h:.1f}"', new_open, count=1)

    return svg_content[: svg_match.start()] + new_open + svg_content[svg_match.end() :]


def style_diagram_svg(
    svg_content: str,
    theme: ThemeType = "light",
    stroke_width: float = 2.0,
    font_size: int = 14,
    font_family: Optional[str] = None,
    padding: int = 16,
) -> str:
    """
    Applies modern typography, theme palettes, edge padding, and valid XML
    background fills to raw diagram SVG from any engine (svgbob, aafigure, aasvg).
    """
    conf = THEME_CONFIGS.get(theme, THEME_CONFIGS["light"])
    font = font_family or conf["font_family"]

    # Strip XML declaration and DOCTYPE so downstream SVG parsers (resvg / browsers)
    # receive clean, self-contained SVG root markup without prolog conflicts.
    cleaned = re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", svg_content, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*<!DOCTYPE[^>]*>\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    # Remove svgbob's inner <rect class="backdrop"...> so it never renders a double border
    # inside our padded canvas.
    cleaned = re.sub(
        r'<rect\b[^>]*\bclass="backdrop"[^>]*(?:/>|>\s*</rect>)',
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Expand canvas with outer padding so border strokes are never clipped
    if padding > 0:
        cleaned = _add_svg_padding(cleaned, pad=padding)

    custom_css = f"""
    <style>
        line, path, circle, rect, polygon {{
            stroke: {conf["stroke_color"]} !important;
            stroke-width: {stroke_width}px !important;
            stroke-linecap: round !important;
            stroke-linejoin: round !important;
        }}
        path {{
            fill: none;
        }}
        .filled {{
            fill: {conf["stroke_color"]} !important;
        }}
        .bg_filled, .nofill {{
            fill: {conf["fill_color"]} !important;
        }}
        polygon.arrowhead, marker path, marker polygon {{
            fill: {conf["stroke_color"]} !important;
            stroke: none !important;
        }}
        rect.bg-canvas, rect.backdrop {{
            fill: {conf["bg_color"]} !important;
            stroke: none !important;
            stroke-width: 0 !important;
        }}
        text {{
            fill: {conf["text_color"]} !important;
            stroke: none !important;
            font-family: {font} !important;
            font-size: {font_size}px !important;
            font-weight: 500 !important;
        }}
    </style>
    """

    # Compute background rect spanning the entire viewBox (including negative padded origin)
    vb_match = re.search(r'\bviewBox="([^"]+)"', cleaned)
    if vb_match:
        parts = vb_match.group(1).replace(",", " ").split()
        if len(parts) == 4:
            bg_rect = (
                f'<rect x="{parts[0]}" y="{parts[1]}" '
                f'width="{parts[2]}" height="{parts[3]}" '
                f'fill="{conf["bg_color"]}" stroke="none" class="bg-canvas"/>'
            )
        else:
            bg_rect = f'<rect width="100%" height="100%" fill="{conf["bg_color"]}" stroke="none" class="bg-canvas"/>'
    else:
        bg_rect = f'<rect width="100%" height="100%" fill="{conf["bg_color"]}" stroke="none" class="bg-canvas"/>'

    # Replace existing <style> block if present (svgbob), otherwise insert cleanly
    # immediately AFTER the opening <svg ...> tag (fixes aafigure XML error).
    if "<style>" in cleaned:
        styled_svg = re.sub(r"<style>.*?</style>", custom_css, cleaned, count=1, flags=re.DOTALL)
        pos = styled_svg.find("</style>")
        if pos != -1 and 'class="bg-canvas"' not in styled_svg:
            styled_svg = styled_svg[: pos + 8] + "\n    " + bg_rect + styled_svg[pos + 8 :]
    else:
        svg_match = re.search(r"<svg\b[^>]*>", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if svg_match:
            insert_at = svg_match.end()
            styled_svg = (
                cleaned[:insert_at]
                + custom_css
                + "\n    "
                + bg_rect
                + cleaned[insert_at:]
            )
        else:
            styled_svg = custom_css + cleaned

    return styled_svg


def diagram_to_svg(
    text: str,
    engine: EngineType = "auto",
    theme: ThemeType = "light",
    stroke_width: float = 2.0,
    font_size: int = 14,
    font_family: Optional[str] = None,
    repair_alignment: bool = True,
) -> str:
    """
    High-level entry point to convert an ASCII or Unicode diagram into a fully styled SVG.
    """
    normalized_text = _normalize_diagram_text(text, repair_alignment=repair_alignment)
    if not normalized_text.strip():
        normalized_text = " "

    selected_engine = engine
    if selected_engine == "auto":
        selected_engine = "svgbob"

    raw_svg = ""
    if selected_engine == "svgbob":
        try:
            raw_svg = _render_with_svgbob(normalized_text)
        except Exception:
            # Pure-Python fallback (e.g. ARM Linux hosts without svgbob binary)
            raw_svg = _render_with_aafigure(normalized_text)
    elif selected_engine == "aasvg":
        if shutil.which("aasvg"):
            raw_svg = _render_with_aasvg(normalized_text)
        else:
            try:
                raw_svg = _render_with_svgbob(normalized_text)
            except Exception:
                raw_svg = _render_with_aafigure(normalized_text)
    elif selected_engine == "aafigure":
        raw_svg = _render_with_aafigure(normalized_text)
    else:
        try:
            raw_svg = _render_with_svgbob(normalized_text)
        except Exception:
            raw_svg = _render_with_aafigure(normalized_text)

    return style_diagram_svg(
        raw_svg,
        theme=theme,
        stroke_width=stroke_width,
        font_size=font_size,
        font_family=font_family,
    )
