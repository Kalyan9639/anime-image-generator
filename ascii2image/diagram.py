"""
Diagram Rendering & Styling Pipeline.

Converts structured ASCII diagrams (flowcharts, sequence diagrams, architecture blocks)
into clean, scalable SVG vector graphics and styled typography.
Supports svgbob (bundled Windows 64-bit binary) and aafigure (pure-Python fallback).
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
from typing import List, Literal, Optional

import markdown_svgbob
import aafigure


EngineType = Literal["svgbob", "aafigure", "auto"]
ThemeType = Literal["light", "dark", "blueprint", "tinted"]


THEME_CONFIGS = {
    "light": {
        "bg_color": "#ffffff",
        "stroke_color": "#1e293b",       # Slate 800
        "fill_color": "#f8fafc",         # Slate 50
        "text_color": "#0f172a",         # Slate 900
        "accent_color": "#3b82f6",       # Blue 500
        "font_family": "Consolas, 'Courier New', monospace",
    },
    "dark": {
        "bg_color": "#0f172a",           # Slate 900
        "stroke_color": "#38bdf8",       # Sky 400
        "fill_color": "#1e293b",         # Slate 800
        "text_color": "#f8fafc",         # Slate 50
        "accent_color": "#818cf8",       # Indigo 400
        "font_family": "Consolas, 'Courier New', monospace",
    },
    "blueprint": {
        "bg_color": "#1e3a8a",           # Deep Blueprint Blue
        "stroke_color": "#93c5fd",       # Light CAD Blue
        "fill_color": "#172554",         # Midnight Blue
        "text_color": "#ffffff",         # Pure White
        "accent_color": "#67e8f9",       # Cyan
        "font_family": "Consolas, 'Courier New', monospace",
    },
    "tinted": {
        "bg_color": "#fafafa",           # Warm Gray
        "stroke_color": "#334155",       # Slate 700
        "fill_color": "#e0f2fe",         # Soft Sky Tint
        "text_color": "#1e293b",         # Slate 800
        "accent_color": "#0284c7",       # Sky 600
        "font_family": "Consolas, 'Courier New', monospace",
    },
}


def get_available_engines() -> List[str]:
    """Returns the list of available diagram rendering engines."""
    engines = ["svgbob", "aafigure"]
    if shutil.which("aasvg"):
        engines.append("aasvg")
    return engines


def _render_with_svgbob(text: str) -> str:
    """Renders ASCII diagram using svgbob."""
    svg_bytes = markdown_svgbob.text2svg(text)
    if isinstance(svg_bytes, bytes):
        return svg_bytes.decode("utf-8")
    return str(svg_bytes)


def _render_with_aafigure(text: str) -> str:
    """Renders ASCII diagram using aafigure (pure Python)."""
    bio = io.BytesIO()
    options = {"format": "svg"}
    aafigure.render(text, bio, options=options)
    return bio.getvalue().decode("utf-8")


def _render_with_aasvg(text: str) -> str:
    """Renders ASCII diagram using aasvg (Node.js CLI if available)."""
    proc = subprocess.run(
        ["aasvg"],
        input=text.encode("utf-8"),
        capture_output=True,
        check=True
    )
    return proc.stdout.decode("utf-8")


def style_diagram_svg(
    svg_content: str,
    theme: ThemeType = "light",
    stroke_width: float = 2.0,
    font_size: int = 14,
    font_family: Optional[str] = None
) -> str:
    """
    Applies modern typography, theme palettes, and background fill to raw diagram SVG.
    """
    conf = THEME_CONFIGS.get(theme, THEME_CONFIGS["light"])
    font = font_family or conf["font_family"]

    custom_css = f"""
    <style>
        rect.bg-canvas {{ fill: {conf["bg_color"]}; }}
        line, path, circle, rect, polygon {{
            stroke: {conf["stroke_color"]} !important;
            stroke-width: {stroke_width}px !important;
            stroke-linecap: round !important;
            stroke-linejoin: round !important;
        }}
        polygon.arrowhead, marker path {{
            fill: {conf["stroke_color"]} !important;
            stroke: none !important;
        }}
        rect.backdrop {{
            fill: {conf["bg_color"]} !important;
            stroke: none !important;
        }}
        .bg_filled, .filled {{
            fill: {conf["fill_color"]} !important;
        }}
        .nofill {{
            fill: {conf["fill_color"]} !important;
        }}
        text {{
            fill: {conf["text_color"]} !important;
            font-family: {font} !important;
            font-size: {font_size}px !important;
            font-weight: 500 !important;
        }}
    </style>
    """

    # If svgbob output, replace or prepend style
    if "<style>" in svg_content:
        # Replace the default style block with our themed style block
        styled_svg = re.sub(r"<style>.*?</style>", custom_css, svg_content, count=1, flags=re.DOTALL)
    else:
        # Insert style block right after <svg ...>
        svg_tag_end = svg_content.find(">")
        if svg_tag_end != -1:
            styled_svg = svg_content[: svg_tag_end + 1] + custom_css + svg_content[svg_tag_end + 1 :]
        else:
            styled_svg = custom_css + svg_content

    # Inject background rect if not present to ensure solid color export
    if 'class="bg-canvas"' not in styled_svg:
        bg_rect = '<rect width="100%" height="100%" class="bg-canvas"/>'
        pos = styled_svg.find("</style>")
        if pos != -1:
            styled_svg = styled_svg[: pos + 8] + "\n    " + bg_rect + styled_svg[pos + 8 :]

    return styled_svg


def diagram_to_svg(
    text: str,
    engine: EngineType = "auto",
    theme: ThemeType = "light",
    stroke_width: float = 2.0,
    font_size: int = 14,
    font_family: Optional[str] = None
) -> str:
    """
    High-level entry point to convert an ASCII diagram into a fully styled SVG.
    """
    selected_engine = engine
    if selected_engine == "auto":
        selected_engine = "svgbob"

    raw_svg = ""
    if selected_engine == "svgbob":
        try:
            raw_svg = _render_with_svgbob(text)
        except Exception:
            # Fallback to pure Python aafigure
            raw_svg = _render_with_aafigure(text)
    elif selected_engine == "aasvg":
        if shutil.which("aasvg"):
            raw_svg = _render_with_aasvg(text)
        else:
            raw_svg = _render_with_svgbob(text)
    elif selected_engine == "aafigure":
        raw_svg = _render_with_aafigure(text)
    else:
        raw_svg = _render_with_svgbob(text)

    return style_diagram_svg(
        raw_svg,
        theme=theme,
        stroke_width=stroke_width,
        font_size=font_size,
        font_family=font_family
    )
