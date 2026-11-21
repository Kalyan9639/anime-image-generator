"""
Photometric Ink & Sub-Cell Spatial Calibration Module.

Calculates exact optical ink density and 8x4 sub-cell spatial distributions
using supersampled fixed-cell font rendering, structure tensor analysis,
and eigenvalue decomposition for directional anisotropy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\consola.ttf",   # Consolas (Windows default monospace)
    r"C:\Windows\Fonts\cour.ttf",      # Courier New
    r"C:\Windows\Fonts\lucon.ttf",     # Lucida Console
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]


def resolve_monospace_font(preferred_path: Optional[str] = None) -> str:
    """Find a verified monospace TrueType font path on the operating system."""
    if preferred_path and os.path.isfile(preferred_path):
        return preferred_path

    for candidate in DEFAULT_FONT_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate

    # Return empty string if no system font file found (will trigger fallback)
    return ""


@dataclass(frozen=True)
class GlyphMetrics:
    """Accurate physical and spatial metrics of a rendered character glyph."""
    char: str
    raw_density: float          # [0.0, 1.0] true fraction of monospace cell filled with ink
    norm_density: float         # [0.0, 1.0] scaled relative to darkest character in font
    anisotropy: float           # [0.0, 1.0] directional stroke coherence from structure tensor
    subcell_8x4: np.ndarray     # Shape (8, 4), float32: exact sub-cell spatial ink distribution
    effective_8x4: np.ndarray   # Shape (8, 4), float32: anisotropy-blended sub-cell tile


class PhotometricAtlas:
    """
    Supersampled photometric and topological atlas of character glyphs.
    
    Renders every printable character onto an identical baseline-locked monospace
    canvas of dimensions (cell_w, cell_h) to prevent tight-bounding-box bias.
    """

    def __init__(self, font_path: Optional[str] = None, font_size: int = 64):
        self.font_path = resolve_monospace_font(font_path)
        self.font_size = font_size

        if self.font_path:
            self._font = ImageFont.truetype(self.font_path, font_size)
        else:
            try:
                self._font = ImageFont.load_default(size=font_size)
            except TypeError:
                self._font = ImageFont.load_default()

        # Compute uniform monospace cell dimensions at this scale
        try:
            ascent, descent = self._font.getmetrics()
            self.cell_h = max(1, ascent + abs(descent))
            self.ascent = ascent
        except Exception:
            self.cell_h = max(1, font_size)
            self.ascent = int(font_size * 0.8)

        try:
            self.cell_w = max(1, int(round(self._font.getlength("M"))))
        except Exception:
            self.cell_w = max(1, int(round(self.cell_h * 0.55)))

        self.aspect_ratio = self.cell_h / self.cell_w

        self._cache: Dict[str, GlyphMetrics] = {}
        self._max_raw_density = 0.001

        # Pre-calibrate standard ASCII printable range
        self._precalibrate()

    def _precalibrate(self) -> None:
        """Calibrate ASCII printable characters (32..126) and common Unicode symbols."""
        standard_chars = [chr(i) for i in range(32, 127)]
        # Add common Unicode shading and box/block symbols if available
        extended_chars = [
            "█", "▓", "▒", "░", "▀", "▄", "▌", "▐", "■", "□",
            "─", "│", "┌", "┐", "└", "┘", "├", "┤", "┬", "┴", "┼"
        ]

        temp_metrics = {}
        for ch in standard_chars + extended_chars:
            raw_dens, aniso, sub_8x4, eff_8x4 = self._compute_glyph_tensor(ch)
            temp_metrics[ch] = (raw_dens, aniso, sub_8x4, eff_8x4)
            if raw_dens > self._max_raw_density:
                self._max_raw_density = raw_dens

        # Re-normalize relative to maximum observed raw density (e.g. '█' or '@')
        for ch, (raw_dens, aniso, sub_8x4, eff_8x4) in temp_metrics.items():
            norm_dens = float(np.clip(raw_dens / self._max_raw_density, 0.0, 1.0))
            self._cache[ch] = GlyphMetrics(
                char=ch,
                raw_density=raw_dens,
                norm_density=norm_dens,
                anisotropy=aniso,
                subcell_8x4=sub_8x4,
                effective_8x4=eff_8x4,
            )

    def _compute_glyph_tensor(self, ch: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Renders a single character on a baseline-locked canvas, downsamples to 8x4 subcells,
        and computes the Sobel structure tensor for directional anisotropy.
        """
        # Canvas initialized to 0 (black background / no ink)
        img = Image.new("L", (self.cell_w, self.cell_h), color=0)
        draw = ImageDraw.Draw(img)
        try:
            draw.text((0, self.ascent), ch, fill=255, font=self._font, anchor="ls")
        except Exception:
            draw.text((0, 0), ch, fill=255, font=self._font)

        arr = np.array(img, dtype=np.float32)

        # 1. True Cell Optical Density
        raw_density = float(np.mean(arr) / 255.0)

        # 2. 8x4 Sub-Cell Area Integration
        # cv2.INTER_AREA performs exact pixel area resampling (perfect for downsampling)
        subcell_8x4 = cv2.resize(arr, (4, 8), interpolation=cv2.INTER_AREA) / 255.0

        if raw_density < 1e-4:
            # Completely empty / whitespace
            empty_tile = np.zeros((8, 4), dtype=np.float32)
            return 0.0, 0.0, empty_tile, empty_tile

        # 3. Structure Tensor Analysis via Sobel Gradients
        sobel_x = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)

        j_xx = float(np.mean(sobel_x * sobel_x))
        j_yy = float(np.mean(sobel_y * sobel_y))
        j_xy = float(np.mean(sobel_x * sobel_y))

        # Eigenvalues of 2x2 symmetric matrix [[j_xx, j_xy], [j_xy, j_yy]]
        trace = j_xx + j_yy
        det = j_xx * j_yy - j_xy * j_xy
        discriminant = max(0.0, trace * trace - 4.0 * det)
        sqrt_disc = float(np.sqrt(discriminant))

        lambda_1 = (trace + sqrt_disc) / 2.0
        lambda_2 = max(0.0, (trace - sqrt_disc) / 2.0)

        # Directional Coherence / Anisotropy Score [0.0, 1.0]
        denom = lambda_1 + lambda_2 + 1e-6
        anisotropy = float(((lambda_1 - lambda_2) / denom) ** 2)
        anisotropy = float(np.clip(anisotropy, 0.0, 1.0))

        # 4. Effective Anisotropy-Blended Tile (8x4)
        # Preserves directional strokes (lines, diagonals, boundaries)
        # while smoothing out character-interior holes inside dense shading characters.
        uniform_tile = np.full((8, 4), raw_density, dtype=np.float32)
        effective_8x4 = anisotropy * subcell_8x4 + (1.0 - anisotropy) * uniform_tile

        return raw_density, anisotropy, subcell_8x4.astype(np.float32), effective_8x4.astype(np.float32)

    def get_metrics(self, ch: str) -> GlyphMetrics:
        """Retrieve or dynamically calibrate the metrics for any character."""
        if ch in self._cache:
            return self._cache[ch]

        raw_dens, aniso, sub_8x4, eff_8x4 = self._compute_glyph_tensor(ch)
        if raw_dens > self._max_raw_density:
            self._max_raw_density = raw_dens

        norm_dens = float(np.clip(raw_dens / self._max_raw_density, 0.0, 1.0))
        metrics = GlyphMetrics(
            char=ch,
            raw_density=raw_dens,
            norm_density=norm_dens,
            anisotropy=aniso,
            subcell_8x4=sub_8x4,
            effective_8x4=eff_8x4,
        )
        self._cache[ch] = metrics
        return metrics

    def build_subcell_grid(self, lines: List[str], use_effective: bool = True) -> np.ndarray:
        """
        Transforms a 2D text matrix of shape (H, W) characters into an exact
        continuous sub-cell density canvas of shape (8*H, 4*W) float32 in [0.0, 1.0].
        """
        if not lines:
            return np.zeros((8, 4), dtype=np.float32)

        max_len = max(len(l) for l in lines)
        padded_lines = [l.ljust(max_len) for l in lines]
        h_chars = len(padded_lines)
        w_chars = max_len

        canvas = np.zeros((h_chars * 8, w_chars * 4), dtype=np.float32)

        for row_idx, line in enumerate(padded_lines):
            for col_idx, ch in enumerate(line):
                metrics = self.get_metrics(ch)
                tile = metrics.effective_8x4 if use_effective else metrics.subcell_8x4
                r0 = row_idx * 8
                c0 = col_idx * 4
                canvas[r0 : r0 + 8, c0 : c0 + 4] = tile

        return canvas

    def build_scalar_density_grid(self, lines: List[str]) -> np.ndarray:
        """
        Transforms a 2D text matrix of shape (H, W) characters into an
        exact (H, W) normalized density matrix in [0.0, 1.0].
        """
        if not lines:
            return np.zeros((1, 1), dtype=np.float32)

        max_len = max(len(l) for l in lines)
        padded_lines = [l.ljust(max_len) for l in lines]
        h_chars = len(padded_lines)
        w_chars = max_len

        grid = np.zeros((h_chars, w_chars), dtype=np.float32)

        for row_idx, line in enumerate(padded_lines):
            for col_idx, ch in enumerate(line):
                metrics = self.get_metrics(ch)
                grid[row_idx, col_idx] = metrics.norm_density

        return grid


_DEFAULT_ATLAS: Optional[PhotometricAtlas] = None


def get_photometric_atlas(font_path: Optional[str] = None, font_size: int = 64) -> PhotometricAtlas:
    """Retrieve or initialize the global PhotometricAtlas singleton."""
    global _DEFAULT_ATLAS
    if _DEFAULT_ATLAS is None or (font_path and font_path != _DEFAULT_ATLAS.font_path):
        _DEFAULT_ATLAS = PhotometricAtlas(font_path=font_path, font_size=font_size)
    return _DEFAULT_ATLAS
