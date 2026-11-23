"""
Unit tests for the Art Rendering & 5-Stage Mathematical Reconstruction Pipeline (ascii2image.art).
"""

import xml.etree.ElementTree as ET
import numpy as np
import pytest
from ascii2image.art import (
    render_art_crisp_svg,
    render_art_crisp_png,
    reconstruct_art_photo,
    vectorize_line_art_svg,
)


SAMPLE_ART_TONE = """
   @@@@@@@@@@@@   
 @@@@########@@@@ 
@@@##++++++++##@@@
@@#++::::::::++#@@
@@#++::::::::++#@@
@@@##++++++++##@@@
 @@@@########@@@@ 
   @@@@@@@@@@@@   
"""

SAMPLE_ART_LINE = """
  /\\_/\\  
 ( o.o ) 
  > ^ <  
"""


def test_render_art_crisp_svg():
    svg = render_art_crisp_svg(SAMPLE_ART_TONE, theme="dark")
    assert "<svg" in svg
    assert "<text" in svg
    assert 'xml:space="preserve"' in svg
    assert "xml-space:" not in svg
    root = ET.fromstring(svg)
    assert "svg" in root.tag


def test_render_art_crisp_png():
    png_1x = render_art_crisp_png(SAMPLE_ART_TONE, theme="terminal_green", scale=1.0)
    png_2x = render_art_crisp_png(SAMPLE_ART_TONE, theme="terminal_green", scale=2.0)
    assert isinstance(png_1x, bytes)
    assert png_1x.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png_1x) > 200
    assert len(png_2x) > len(png_1x)


def test_art_handles_whitespace_only_safely():
    mat = reconstruct_art_photo("   \n   ", target_width=200)
    assert mat.dtype == np.uint8
    assert mat.shape[0] >= 100 and mat.shape[1] >= 100
    svg = vectorize_line_art_svg("   \n   ", theme="dark")
    ET.fromstring(svg)


def test_reconstruct_art_photo_output():
    reconstructed = reconstruct_art_photo(
        SAMPLE_ART_TONE,
        target_width=400,
        denoise_strength=1.0,
        clahe_clip=2.0
    )
    assert isinstance(reconstructed, np.ndarray)
    assert reconstructed.dtype == np.uint8
    assert reconstructed.shape[1] == 400
    assert reconstructed.shape[0] > 100
    # Must have non-trivial contrast / dynamic range
    assert reconstructed.std() > 15.0
    assert reconstructed.min() < 50
    assert reconstructed.max() > 200


def test_vectorize_line_art_svg():
    svg = vectorize_line_art_svg(SAMPLE_ART_LINE, theme="light")
    assert "<svg" in svg
    root = ET.fromstring(svg)
    assert "svg" in root.tag
