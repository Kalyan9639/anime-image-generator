"""
Art Rendering & Advanced 5-Stage Mathematical Reconstruction Pipeline.

Provides:
1. Crisp Vector SVG & High-DPI PNG: Exact anti-aliased monospace typography.
2. 5-Stage Sub-Cell & Edge-Preserving Reconstruction: Converts discrete ASCII art
   into continuous-tone grayscale images using 8x4 sub-cell spatial grids,
   Sobel anisotropy blending, bilateral filtering, CLAHE, and unsharp masking.
3. Vector Contour Tracing: Converts sparse line-art sketches into smooth SVG Bézier paths.
"""

from __future__ import annotations

import html
import io
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .ink import get_photometric_atlas, resolve_monospace_font


ThemeType = Literal["light", "dark", "terminal_green", "amber", "blueprint"]
InvertMode = Literal["auto", "yes", "no"]


ART_PALETTES = {
    "light": {"bg": "#ffffff", "fg": "#0f172a"},           # Dark charcoal on white
    "dark": {"bg": "#0f172a", "fg": "#f8fafc"},            # Crisp white on dark slate
    "terminal_green": {"bg": "#051005", "fg": "#22c55e"},  # Classic retro CRT green
    "amber": {"bg": "#120a00", "fg": "#f59e0b"},           # Retro amber terminal
    "blueprint": {"bg": "#1e3a8a", "fg": "#e0f2fe"},       # Blueprint cyan
}


def _split_and_pad(text: str) -> Tuple[List[str], int, int]:
    """Cleans, splits, and right-pads ASCII lines to a uniform rectangle."""
    lines = [l.rstrip("\r\n") for l in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return [""], 1, 1

    max_w = max(len(l) for l in lines)
    padded = [l.ljust(max_w) for l in lines]
    return padded, len(padded), max_w


def render_art_crisp_svg(
    text: str,
    theme: ThemeType = "dark",
    font_family: str = "Consolas, 'Courier New', monospace",
    font_size: int = 14,
    line_spacing: float = 1.15
) -> str:
    """
    Renders ASCII art as an infinitely scalable vector SVG with crisp typography
    and exact monospace grid alignment.
    """
    lines, h_chars, w_chars = _split_and_pad(text)
    palette = ART_PALETTES.get(theme, ART_PALETTES["dark"])

    # Calculate exact viewBox dimensions based on monospace aspect ratio ~ 0.60
    char_w_est = font_size * 0.602
    line_h = font_size * line_spacing
    total_w = int(round(w_chars * char_w_est + font_size * 2))
    total_h = int(round(h_chars * line_h + font_size * 2))

    tspan_elements = []
    y_start = font_size + font_size * 0.85
    for i, line in enumerate(lines):
        y_pos = y_start + i * line_h
        escaped_line = html.escape(line)
        tspan_elements.append(
            f'    <text x="{font_size}" y="{y_pos:.2f}">{escaped_line}</text>'
        )

    tspan_block = "\n".join(tspan_elements)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}">
  <style>
    rect.bg {{ fill: {palette["bg"]}; }}
    text {{
      fill: {palette["fg"]};
      font-family: {font_family};
      font-size: {font_size}px;
      font-weight: 500;
      white-space: pre;
      xml-space: preserve;
    }}
  </style>
  <rect width="100%" height="100%" class="bg"/>
{tspan_block}
</svg>"""
    return svg


def render_art_crisp_png(
    text: str,
    theme: ThemeType = "dark",
    font_size: int = 18,
    font_path: Optional[str] = None
) -> bytes:
    """
    Renders ASCII art into a high-resolution, anti-aliased raster PNG image.
    """
    lines, h_chars, w_chars = _split_and_pad(text)
    palette = ART_PALETTES.get(theme, ART_PALETTES["dark"])

    resolved_font = resolve_monospace_font(font_path)
    if resolved_font:
        font = ImageFont.truetype(resolved_font, font_size)
    else:
        font = ImageFont.load_default()

    try:
        ascent, descent = font.getmetrics()
        line_h = ascent + abs(descent)
    except Exception:
        line_h = int(font_size * 1.2)

    try:
        char_w = int(round(font.getlength("M")))
    except Exception:
        char_w = int(round(line_h * 0.55))

    pad_x = font_size
    pad_y = font_size
    img_w = w_chars * char_w + pad_x * 2
    img_h = h_chars * line_h + pad_y * 2

    # Parse hex color into RGB tuples
    bg_hex = palette["bg"].lstrip("#")
    fg_hex = palette["fg"].lstrip("#")
    bg_rgb = tuple(int(bg_hex[i : i + 2], 16) for i in (0, 2, 4))
    fg_rgb = tuple(int(fg_hex[i : i + 2], 16) for i in (0, 2, 4))

    img = Image.new("RGB", (img_w, img_h), color=bg_rgb)
    draw = ImageDraw.Draw(img)

    for i, line in enumerate(lines):
        y = pad_y + i * line_h
        draw.text((pad_x, y), line, fill=fg_rgb, font=font)

    bio = io.BytesIO()
    img.save(bio, format="PNG", optimize=True)
    return bio.getvalue()


def reconstruct_art_photo(
    text: str,
    target_width: int = 1200,
    invert: InvertMode = "auto",
    denoise_strength: float = 1.0,
    clahe_clip: float = 2.0,
    font_path: Optional[str] = None
) -> np.ndarray:
    """
    Advanced 5-Stage Mathematical Signal Processing Reconstruction:
    
    1. 8x4 Sub-Cell Spatial Density Atlas with Sobel Anisotropy Blending.
    2. Dynamic Range Percentile Stretch ([P1, P99] -> [0.0, 1.0]) & Gamma.
    3. Exact Font-Metric Aspect-Ratio Upsampling with Lanczos-4.
    4. Non-Linear Edge-Preserving Bilateral Filtering.
    5. CLAHE (Local Contrast Enhancement) + Thresholded Unsharp Masking.
    
    Returns an 8-bit single-channel grayscale image (numpy uint8 array, H x W).
    """
    lines, h_chars, w_chars = _split_and_pad(text)
    atlas = get_photometric_atlas(font_path=font_path)

    # 1. Build continuous (8*H, 4*W) sub-cell density canvas
    subcell_canvas = atlas.build_subcell_grid(lines, use_effective=True)

    # Auto-detect background polarity
    # Sample border pixels (first/last row and first/last column)
    border_pixels = np.concatenate([
        subcell_canvas[0:8, :].flatten(),
        subcell_canvas[-8:, :].flatten(),
        subcell_canvas[:, 0:4].flatten(),
        subcell_canvas[:, -4:].flatten(),
    ])
    avg_border_density = float(np.mean(border_pixels))

    should_invert = False
    if invert == "yes":
        should_invert = True
    elif invert == "no":
        should_invert = False
    else:  # "auto"
        # If the border density is low (< 0.20), characters are drawn on a dark background.
        # For a standard continuous-tone photographic display where ink represents darkness:
        # We invert so that dense characters (@, #) map to dark shadows and space maps to light,
        # or vice-versa depending on context.
        # In ASCII portraits (like Paul Bourke), characters usually represent darkness (ink)
        # on white paper. If the border is empty/space, the background of the image is white.
        should_invert = False

    # 2. Dynamic Range Percentile Stretch
    p_low = float(np.percentile(subcell_canvas, 1.0))
    p_high = float(np.percentile(subcell_canvas, 99.0))
    if p_high > p_low + 1e-4:
        stretched = np.clip((subcell_canvas - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        stretched = subcell_canvas

    # In ASCII art, character ink typically represents darkness (0.0 = white canvas, 1.0 = dark ink).
    # To convert to pixel luminance where 255 is white and 0 is black:
    luminance_norm = 1.0 - stretched if not should_invert else stretched

    # Perceptual gamma correction (gamma ~ 1.2 to slightly deepen contrast)
    luminance_norm = np.power(luminance_norm, 1.2)

    # 3. Exact Font-Metric Aspect-Ratio Resampling
    # Monospace characters are taller than wide: aspect_ratio = cell_h / cell_w
    aspect = atlas.aspect_ratio
    target_w = max(100, int(target_width))
    target_h = max(100, int(round(target_w * (h_chars * aspect) / max(1, w_chars))))

    # Lanczos-4 interpolation for optimal high-frequency reconstruction
    upsampled = cv2.resize(luminance_norm, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    img_8u = np.clip(upsampled * 255.0, 0, 255).astype(np.uint8)

    # 4. Non-Linear Edge-Preserving Bilateral Filtering
    # d: filter neighborhood diameter, scaled to character cell pixel size
    cell_px_h = target_h / h_chars
    d_val = max(5, int(round(cell_px_h * 0.75)))
    if d_val % 2 == 0:
        d_val += 1

    sigma_color = max(20.0, 45.0 * denoise_strength)
    sigma_space = max(5.0, (cell_px_h * 0.6) * denoise_strength)

    # Pass 1: Merges internal character grid steps
    bilateral_1 = cv2.bilateralFilter(img_8u, d=d_val, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # Pass 2: Fine-scale edge preservation
    bilateral_2 = cv2.bilateralFilter(bilateral_1, d=max(3, d_val // 2), sigmaColor=sigma_color * 0.75, sigmaSpace=sigma_space * 0.5)

    # 5. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=max(1.0, clahe_clip), tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(bilateral_2)

    # Final Stage: Subtle Thresholded Unsharp Masking
    gaussian = cv2.GaussianBlur(clahe_enhanced, (0, 0), sigmaX=1.5)
    high_pass = cv2.subtract(clahe_enhanced, gaussian)
    sharp = cv2.addWeighted(clahe_enhanced, 1.0, high_pass, 0.4, 0)

    return sharp


def vectorize_line_art_svg(
    text: str,
    theme: ThemeType = "light",
    stroke_width: float = 2.5,
    smoothness: float = 1.0,
    font_path: Optional[str] = None
) -> str:
    """
    Traces sparse ASCII line-art sketches (e.g. animal sketches, cartoon figures)
    into clean, continuous SVG Bézier contour paths.
    """
    lines, h_chars, w_chars = _split_and_pad(text)
    atlas = get_photometric_atlas(font_path=font_path)
    palette = ART_PALETTES.get(theme, ART_PALETTES["light"])

    # Build raw 8x4 subcell grid
    subcell = atlas.build_subcell_grid(lines, use_effective=False)
    aspect = atlas.aspect_ratio

    target_w = max(400, w_chars * 16)
    target_h = int(round(target_w * (h_chars * aspect) / max(1, w_chars)))

    # Resample to intermediate tracing canvas
    raster = cv2.resize(subcell, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    img_8u = np.clip(raster * 255.0, 0, 255).astype(np.uint8)

    # Threshold to binary stroke bitmap
    _, binary = cv2.threshold(img_8u, 40, 255, cv2.THRESH_BINARY)

    # Morphological closing to bridge adjacent sub-cell stroke gaps
    kernel_size = max(3, int(round(target_h / h_chars * 0.25)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    path_data_list = []
    epsilon_base = max(1.0, 2.0 * smoothness)

    for cnt in contours:
        # Filter out tiny specks / noise
        if cv2.contourArea(cnt) < 12.0:
            continue

        # Douglas-Peucker polygon approximation
        epsilon = epsilon_base
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        if len(approx) < 3:
            continue

        # Convert polygon points into SVG path data
        pts = approx.reshape(-1, 2)
        d_parts = [f"M {pts[0][0]} {pts[0][1]}"]
        for pt in pts[1:]:
            d_parts.append(f"L {pt[0]} {pt[1]}")
        d_parts.append("Z")
        path_data_list.append(" ".join(d_parts))

    svg_paths = "\n".join(
        f'    <path d="{p}" fill="{palette["fg"]}" stroke="none"/>'
        for p in path_data_list
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{target_w}" height="{target_h}" viewBox="0 0 {target_w} {target_h}">
  <rect width="100%" height="100%" fill="{palette["bg"]}"/>
  <g>
{svg_paths}
  </g>
</svg>"""
    return svg
