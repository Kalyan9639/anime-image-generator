"""
2.5D Optical Compositor & Post-Processing Engine (Track 1).

Analytically synthesizes rich atmospheric backgrounds & optical effects purely from
math (zero stored images):
- Multi-stop sky / twilight / studio / mountain-ridge gradients
- Depth-layered warm/cool Bokeh light discs with soft optical iris rings
- Johan Gielis Superformula snowflake / star / spark particles
- Z-Buffer Depth-of-Field (DOF) blur, additive glow/bloom, ACES filmic tone mapping,
  and subtle lens vignette.
"""

from __future__ import annotations

import io
import math
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from .math_core import gielis_radius
from .patterns import fbm_2d, hex_to_rgb01
from .renderer_3d import GBuffer
from .schema import BackgroundSpec, LightingSpec


def _render_background_layer(
    bg: BackgroundSpec,
    width: int,
    height: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Renders the analytical procedural background canvas + emissive glow buffer
    in linear HDR float32 [0, ~2.0].
    """
    rng = np.random.default_rng(seed + 701)
    top_c = hex_to_rgb01(bg.top_color)
    bot_c = hex_to_rgb01(bg.bottom_color)
    glow_c = hex_to_rgb01(bg.glow_color)

    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    # Base vertical gradient
    canvas = top_c[None, None, :] * (1.0 - yy[:, :, None]) + bot_c[None, None, :] * yy[:, :, None]
    canvas = np.broadcast_to(canvas, (height, width, 3)).copy()
    glow = np.zeros((height, width, 3), dtype=np.float32)

    if bg.style in ("twilight_glow", "bokeh_night", "studio"):
        # Soft radial studio / horizon glow
        gx = 0.52 if bg.style != "bokeh_night" else 0.65
        gy = 0.58 if bg.style == "twilight_glow" else 0.42
        dist2 = ((xx - gx) * 1.15) ** 2 + ((yy - gy) * 1.15) ** 2
        rad_glow = np.exp(-dist2 * (3.2 if bg.style == "studio" else 2.2))
        strength = 0.35 if bg.style == "studio" else 0.55
        canvas += glow_c[None, None, :] * rad_glow[:, :, None] * strength

    if bg.style == "mountain_horizon":
        # Multi-layer fBm procedural mountain silhouettes
        u_line = np.linspace(0.0, 5.5, width, dtype=np.float32)
        for layer_i, (base_y, amp, freq, shade) in enumerate(
            [(0.56, 0.14, 1.0, 0.55), (0.64, 0.11, 1.8, 0.38), (0.73, 0.08, 2.9, 0.24)]
        ):
            ridge = base_y - amp * fbm_2d(u_line * freq + layer_i * 7.3, np.full_like(u_line, layer_i * 3.1), octaves=3)
            mask = yy >= ridge[None, :]
            col = bot_c * shade + top_c * (1.0 - shade) * 0.6
            canvas = np.where(mask[:, :, None], col[None, None, :], canvas)

    if bg.style == "starfield":
        n_stars = max(80, bg.particle_count or 140)
        sx = rng.integers(0, width, size=n_stars)
        sy = rng.integers(0, height, size=n_stars)
        sb = rng.uniform(0.4, 1.6, size=n_stars).astype(np.float32)
        for x, y, b in zip(sx, sy, sb):
            canvas[y, x] += np.array([b, b * 0.98, b * 1.05], dtype=np.float32)
            glow[y, x] += b * 0.8

    # Procedural Optical Bokeh Light Discs (depth-sorted soft glowing circles!)
    bokeh_n = bg.bokeh_count if bg.bokeh_count > 0 else (95 if bg.style == "bokeh_night" else 0)
    if bokeh_n > 0:
        palette = [hex_to_rgb01(c) for c in (bg.bokeh_colors or ["#fde68a", "#fbbf24", "#fff7ed"])]
        min_dim = float(min(width, height))
        # Generate depth layers (far large soft discs -> closer bright discs)
        depths = np.sort(rng.uniform(0.25, 1.0, size=bokeh_n))[::-1]
        for idx in range(bokeh_n):
            d_val = float(depths[idx])
            cx = float(rng.uniform(-0.05, 1.05) * width)
            cy = float(rng.uniform(-0.05, 0.92) * height)
            rad = max(4.0, float(rng.uniform(0.028, 0.088) * min_dim * bg.bokeh_size * (0.65 + 0.6 * d_val)))
            col = palette[idx % len(palette)]
            intensity = float(rng.uniform(0.22, 0.68) * (0.55 + 0.55 * (1.0 - d_val)))

            x0 = max(0, int(math.floor(cx - rad * 1.35)))
            x1 = min(width, int(math.ceil(cx + rad * 1.35)))
            y0 = max(0, int(math.floor(cy - rad * 1.35)))
            y1 = min(height, int(math.ceil(cy + rad * 1.35)))
            if x1 <= x0 or y1 <= y0:
                continue

            lx = np.arange(x0, x1, dtype=np.float32)[None, :] - cx
            ly = np.arange(y0, y1, dtype=np.float32)[:, None] - cy
            r_norm = np.hypot(lx, ly) / rad
            # Soft optical bokeh disc profile (slight bright iris rim + smooth falloff)
            disc = np.clip(1.0 - np.power(np.clip(r_norm, 0.0, 2.0), 4.5), 0.0, 1.0)
            iris_ring = np.exp(-((r_norm - 0.82) ** 2) * 28.0) * 0.25
            halo = np.exp(-(r_norm * r_norm) * 1.6) * 0.25
            alpha = np.clip((disc * 0.85 + iris_ring + halo) * intensity, 0.0, 1.2)

            contrib = col[None, None, :] * alpha[:, :, None]
            canvas[y0:y1, x0:x1] += contrib
            glow[y0:y1, x0:x1] += contrib * 0.85

    return canvas, glow


def _draw_foreground_particles(
    canvas: np.ndarray,
    glow: np.ndarray,
    bg: BackgroundSpec,
    seed: int,
) -> None:
    """Draws pure-math Gielis Superformula snowflakes / sparks / star sparkles."""
    if bg.particle_type == "none" or bg.particle_count <= 0:
        return

    h, w, _ = canvas.shape
    rng = np.random.default_rng(seed + 1337)
    p_col = hex_to_rgb01(bg.particle_color)
    min_dim = float(min(w, h))

    for _ in range(bg.particle_count):
        cx = float(rng.uniform(0.03, 0.97) * w)
        cy = float(rng.uniform(0.03, 0.95) * h)
        rad = max(3.0, float(rng.uniform(0.008, 0.028) * min_dim))
        rot = float(rng.uniform(0.0, math.pi))
        bright = float(rng.uniform(0.55, 1.25))

        x0 = max(0, int(math.floor(cx - rad * 1.6)))
        x1 = min(w, int(math.ceil(cx + rad * 1.6)))
        y0 = max(0, int(math.floor(cy - rad * 1.6)))
        y1 = min(h, int(math.ceil(cy + rad * 1.6)))
        if x1 <= x0 or y1 <= y0:
            continue

        lx = np.arange(x0, x1, dtype=np.float32)[None, :] - cx
        ly = np.arange(y0, y1, dtype=np.float32)[:, None] - cy
        r = np.hypot(lx, ly) / rad
        ang = np.arctan2(ly, lx) + rot

        if bg.particle_type == "snowflakes":
            # 6-fold Johan Gielis Superformula crystalline star
            target_r = gielis_radius(ang, m=6.0, n1=0.38, n2=1.0, n3=1.0)
            target_r = target_r / max(1e-4, float(np.max(target_r)))
            shape = np.clip((target_r - r) * 5.0 + 0.5, 0.0, 1.0)
            core = np.exp(-(r * r) * 4.0) * 0.7
            mask = np.clip(shape * 0.95 + core, 0.0, 1.0) * bright
        elif bg.particle_type == "stars":
            target_r = gielis_radius(ang, m=4.0, n1=0.25, n2=1.0, n3=1.0)
            target_r = target_r / max(1e-4, float(np.max(target_r)))
            shape = np.clip((target_r - r) * 4.5 + 0.4, 0.0, 1.0)
            mask = np.clip(shape + np.exp(-(r * r) * 5.0) * 0.6, 0.0, 1.0) * bright
        else:  # sparks
            mask = np.exp(-(r * r) * 3.5) * bright

        contrib = p_col[None, None, :] * mask[:, :, None]
        canvas[y0:y1, x0:x1] += contrib
        glow[y0:y1, x0:x1] += contrib * 0.9


def _aces_filmic_tonemap(hdr: np.ndarray, exposure: float = 1.05) -> np.ndarray:
    """ACES filmic curve for pleasing highlight roll-off without washed-out bloom."""
    x = np.maximum(hdr * float(exposure), 0.0)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    mapped = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
    # sRGB gamma encode
    return np.power(mapped, 1.0 / 2.2)


def composite_scene_to_rgb(
    gbuf: GBuffer,
    bg: BackgroundSpec,
    lit: LightingSpec,
    dof_strength: float = 0.25,
    seed: int = 42,
) -> np.ndarray:
    """
    Composites the 3D GBuffer over the analytical bokeh/gradient background,
    applies Z-buffer Depth-of-Field blur, additive bloom, foreground snowflakes,
    ACES filmic tone mapping, and subtle lens vignette.

    Returns uint8 RGB image array of shape (H, W, 3).
    """
    h, w = gbuf.height, gbuf.width
    bg_rgb, bg_glow = _render_background_layer(bg, w, h, seed=seed)

    # Anti-alias entity silhouette edges subtly against background
    mask_f = gbuf.hit_mask.astype(np.float32)
    mask_aa = cv2.GaussianBlur(mask_f, (3, 3), 0.65)
    # Keep solid interior opaque
    alpha = np.where(gbuf.hit_mask, np.maximum(mask_aa, 0.85), mask_aa * 0.35)[:, :, None]

    hdr = bg_rgb * (1.0 - alpha) + gbuf.color_rgb * alpha
    glow_buf = bg_glow * (1.0 - alpha * 0.75) + gbuf.emissive_rgb * alpha

    # Z-Buffer Depth-of-Field (DOF) Blur on far/near out-of-focus regions
    dof = float(np.clip(dof_strength, 0.0, 1.0))
    if dof > 0.03:
        k_rad = max(3, int(round(min(w, h) * 0.014 * dof)) * 2 + 1)
        blurred = cv2.GaussianBlur(hdr, (k_rad, k_rad), 0)
        finite_z = np.where(np.isfinite(gbuf.depth_z), gbuf.depth_z, gbuf.focus_depth * 2.4)
        coc = np.clip(np.abs(finite_z - gbuf.focus_depth) / max(1.0, gbuf.focus_depth * 0.65), 0.0, 1.0) * dof
        # Keep in-focus subject crisp while softening far background/foreground
        hdr = hdr * (1.0 - coc[:, :, None]) + blurred * coc[:, :, None]

    # Foreground optical particles (snowflakes / sparks / star glints)
    _draw_foreground_particles(hdr, glow_buf, bg, seed=seed)

    # Multi-scale Additive Optical Bloom
    if lit.bloom_intensity > 0.01:
        bright_pass = np.maximum(hdr - 0.82, 0.0) + glow_buf * 0.65
        b1 = cv2.GaussianBlur(bright_pass, (0, 0), sigmaX=max(3.0, min(w, h) * 0.012))
        b2 = cv2.GaussianBlur(bright_pass, (0, 0), sigmaX=max(9.0, min(w, h) * 0.035))
        hdr = hdr + (b1 * 0.65 + b2 * 0.55) * float(lit.bloom_intensity)

    # Subtle Photographic Lens Vignette
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    vig = np.clip(1.0 - 0.18 * (xx * xx + yy * yy), 0.78, 1.0)
    hdr = hdr * vig[:, :, None]

    ldr = _aces_filmic_tonemap(hdr, exposure=lit.exposure)
    return np.clip(ldr * 255.0, 0, 255).astype(np.uint8)


def rgb_to_png_bytes(rgb_u8: np.ndarray) -> bytes:
    img = Image.fromarray(rgb_u8, mode="RGB")
    bio = io.BytesIO()
    img.save(bio, format="PNG", optimize=True)
    return bio.getvalue()
