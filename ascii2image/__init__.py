"""
ASCII to Image Generator (ascii2image)
Convert ASCII diagrams and ASCII art pictures into high-quality SVG and PNG images.
Pure mathematical signal processing and rendering frameworks - no AI generators.
"""

__version__ = "1.0.0"
__author__ = "Antigravity"

from .detect import classify_ascii, DetectionResult
from .normalize import sanitize_ascii_text, repair_first_line_indent, strip_markdown_fences
from .ink import get_photometric_atlas, PhotometricAtlas, GlyphMetrics
from .diagram import diagram_to_svg, get_available_engines
from .art import (
    render_art_crisp_svg,
    render_art_crisp_png,
    reconstruct_art_photo,
    vectorize_line_art_svg,
)
from .export import svg_to_png

__all__ = [
    "classify_ascii",
    "DetectionResult",
    "sanitize_ascii_text",
    "repair_first_line_indent",
    "strip_markdown_fences",
    "get_photometric_atlas",
    "PhotometricAtlas",
    "GlyphMetrics",
    "diagram_to_svg",
    "get_available_engines",
    "render_art_crisp_svg",
    "render_art_crisp_png",
    "reconstruct_art_photo",
    "vectorize_line_art_svg",
    "svg_to_png",
]
