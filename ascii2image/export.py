"""
High-DPI SVG to PNG Export Module.

Converts SVG documents into clean, anti-aliased PNG images with arbitrary
supersampling (scale factors) using resvg_py (Rust-powered, standalone wheel).
"""

from __future__ import annotations

import re
from typing import Optional

import resvg_py

from .ink import resolve_monospace_font


def extract_svg_background(svg_string: str) -> Optional[str]:
    """Attempts to discover background color from <rect class="bg-canvas" fill="..."> or style."""
    match = re.search(r'rect\.(?:bg-canvas|bg)[^{]*\{\s*fill:\s*([^;!]+)', svg_string)
    if match:
        return match.group(1).strip()
    match_attr = re.search(r'<rect[^>]+fill="([^"]+)"[^>]*class="(?:bg-canvas|bg)"', svg_string)
    if match_attr:
        return match_attr.group(1).strip()
    return None


def svg_to_png(
    svg_string: str,
    scale: float = 2.0,
    background: Optional[str] = None
) -> bytes:
    """
    Renders SVG markup into PNG image bytes with high-DPI scaling.

    :param svg_string: Complete SVG document markup as a string.
    :param scale: Zoom factor (e.g. 2.0 for 2x Retina / supersampled output).
    :param background: Optional CSS background color (e.g. "#ffffff" or "#0f172a").
    :return: PNG-encoded binary bytes.
    """
    bg = background or extract_svg_background(svg_string)
    mono_font = resolve_monospace_font()

    kwargs = {
        "svg_string": svg_string,
        "zoom": float(scale),
        "shape_rendering": "geometric_precision",
        "text_rendering": "geometric_precision",
        "image_rendering": "optimize_quality",
    }
    if bg:
        kwargs["background"] = bg
    if mono_font:
        kwargs["font_files"] = [mono_font]

    try:
        return resvg_py.svg_to_bytes(**kwargs)
    except Exception:
        # Retry without optional font_files / background overrides
        kwargs.pop("font_files", None)
        try:
            return resvg_py.svg_to_bytes(**kwargs)
        except Exception:
            if "background" in kwargs:
                kwargs.pop("background")
                return resvg_py.svg_to_bytes(**kwargs)
            raise
