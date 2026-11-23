"""
Unit tests for the Diagram Rendering Pipeline (ascii2image.diagram & export).
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import pytest
from ascii2image.diagram import diagram_to_svg, get_available_engines
from ascii2image.export import svg_to_png


SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


SAMPLE_DIAGRAM = """
+-------+       +-------+
| Start | ----> |  End  |
+-------+       +-------+
"""


def test_engines_available():
    engines = get_available_engines()
    assert "svgbob" in engines
    assert "aafigure" in engines


def test_diagram_to_svg_svgbob():
    svg = diagram_to_svg(SAMPLE_DIAGRAM, engine="svgbob", theme="light")
    assert "<svg" in svg
    assert "</svg>" in svg
    # Must parse as valid XML
    root = ET.fromstring(svg)
    assert "svg" in root.tag


def test_diagram_to_svg_aafigure():
    svg = diagram_to_svg(SAMPLE_DIAGRAM, engine="aafigure", theme="dark")
    assert "<svg" in svg
    assert "</svg>" in svg


def test_diagram_themes():
    for theme in ["light", "dark", "blueprint", "tinted"]:
        svg = diagram_to_svg(SAMPLE_DIAGRAM, theme=theme)
        assert "<svg" in svg
        assert f'class="bg-canvas"' in svg


def test_svg_to_png_export():
    svg = diagram_to_svg(SAMPLE_DIAGRAM, theme="light")
    png_bytes = svg_to_png(svg, scale=2.0)
    assert isinstance(png_bytes, bytes)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png_bytes) > 200


def test_aafigure_to_png_no_xml_error():
    """
    Regression test for Streamlit Cloud error 'unknown token at 31:5':
    aafigure SVGs across all themes must parse as valid XML and export cleanly via resvg_py.
    """
    for sample_name in ("flowchart.txt", "network.txt"):
        text = (SAMPLES_DIR / sample_name).read_text(encoding="utf-8")
        for theme in ("light", "dark", "blueprint", "tinted"):
            svg = diagram_to_svg(text, engine="aafigure", theme=theme)
            root = ET.fromstring(svg)
            assert "svg" in root.tag
            png_bytes = svg_to_png(svg, scale=1.5)
            assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
            assert len(png_bytes) > 500


def test_unicode_diagram_to_svg_and_png():
    unicode_diag = """
┌──────────┐     ┌──────────┐
│  Client  │ ──▶ │  Server  │
└──────────┘     └──────────┘
"""
    for engine in ("svgbob", "aafigure"):
        svg = diagram_to_svg(unicode_diag, engine=engine, theme="light")
        root = ET.fromstring(svg)
        assert "svg" in root.tag
        png_bytes = svg_to_png(svg, scale=2.0)
        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(png_bytes) > 300


def test_svgbob_no_backdrop_border_and_solid_arrows():
    """svgbob output must strip the inner backdrop rect so no outer black border appears."""
    svg = diagram_to_svg(SAMPLE_DIAGRAM, engine="svgbob", theme="light")
    assert 'class="backdrop"' not in svg
    assert 'class="bg-canvas"' in svg
    # Ensure svgbob actually ran (produces <rect ... rx="0"> or <defs>) rather than silently failing
    assert "<defs>" in svg or "svgbob" in svg or "rx=" in svg
    ET.fromstring(svg)


def test_aafigure_preserves_double_letters_and_rounded_boxes():
    """aafigure must keep words like Process, Haskell, Passing intact and handle .---. boxes."""
    cc_text = (SAMPLES_DIR / "cc_diag.txt").read_text(encoding="utf-8")
    svg_cc = diagram_to_svg(cc_text, engine="aafigure", theme="light")
    assert "Process 1" in svg_cc
    assert "Haskell" in svg_cc
    assert "Passing" in svg_cc
    ET.fromstring(svg_cc)

    net_text = (SAMPLES_DIR / "network.txt").read_text(encoding="utf-8")
    svg_net = diagram_to_svg(net_text, engine="aafigure", theme="light")
    assert "Gateway Router" in svg_net
    assert "Core Switch" in svg_net
    # Rounded box borders must be converted to vector lines, not literal ASCII dashes
    assert "------------" not in svg_net
    ET.fromstring(svg_net)


def test_misaligned_top_border_auto_repaired_in_svg():
    """Missing Line-1 spaces from drag-select copy-paste must align with the box walls below."""
    misaligned = (
        "+---------------+\n"
        "                | Core Switches |\n"
        "                +---------------+"
    )
    svg = diagram_to_svg(misaligned, engine="svgbob", theme="light")
    root = ET.fromstring(svg)
    assert "svg" in root.tag
    # When properly aligned, svgbob merges the 4 sides into a single <rect> (no orphan line at x1=4)
    assert 'x1="4"' not in svg
    png_bytes = svg_to_png(svg, scale=2.0)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_heavy_and_rounded_unicode_boxes():
    heavy = "┏━━━━━━━━━┓     ┏━━━━━━━━━┓\n┃ Alpha   ┃ ──▶ ┃ Beta    ┃\n┗━━━━━━━━━┛     ┗━━━━━━━━━┛"
    for engine in ("svgbob", "aafigure"):
        svg = diagram_to_svg(heavy, engine=engine, theme="dark")
        ET.fromstring(svg)
        png = svg_to_png(svg, scale=2.0)
        assert png.startswith(b"\x89PNG\r\n\x1a\n")

