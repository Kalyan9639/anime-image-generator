"""
High-DPI SVG to PNG Export Module.

Converts SVG documents into clean, anti-aliased PNG images with arbitrary
supersampling (scale factors) using resvg_py (Rust-powered, standalone wheel).
"""

from __future__ import annotations

import re
from typing import Optional

import resvg_py


def extract_svg_background(svg_string: str) -> Optional[str]:
    """Attempts to discover background color from <rect class="bg-canvas" fill="..."> or style."""
    match = re.search(r'rect\.(?:bg-canvas|bg)\s*\{\s*fill:\s*([^;]+);', svg_string)
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

    kwargs = {
        "svg_string": svg_string,
        "zoom": float(scale),
        "shape_rendering": "geometric_precision",
        "text_rendering": "geometric_precision",
        "image_rendering": "optimize_quality",
    }
    if bg:
        kwargs["background"] = bg

    try:
        png_bytes = resvg_py.svg_to_bytes(**kwargs)
    except Exception as e:
        # Retry without background keyword if color string was rejected
        if "background" in kwargs:
            kwargs.pop("background")
            png_bytes = resvg_py.svg_to_bytes(**kwargs)
        else:
            raise e

    return png_bytes
