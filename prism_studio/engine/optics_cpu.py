"""
6-Stage Screen-Space HDR Optical Compositor (CPU Fallback Path) for PrismStudio.

Implements fast vectorized NumPy + OpenCV parity with `glsl_Pass2_optics.glsl`:
1. Light-modulated Sobel G-buffer ink outlines (anime_npr)
2. Foreground-protected 12-tap Vogel disk DOF bokeh (speed-first CPU profile)
3. 16-tap radial volumetric spotlight/sun god-rays
4. Background bokeh orbs & floating sunlit dust sparkles
5. Multi-scale HDR bloom + radial chromatic aberration
6. ACES filmic tone mapping + 1% luma film grain + sRGB encoding
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..schema.scene_dna import SceneDNA
from .cpu_kernel import CPUGBuffer


def composite_cpu_optics(gbuf: CPUGBuffer, dna: SceneDNA) -> np.ndarray:
    """Executes all 6 HDR post-FX stages on a CPUGBuffer and returns uint8 RGB (H, W, 3)."""
    h, w = gbuf.height, gbuf.width
    lit = dna.lighting
    atmo = dna.zone3_atmosphere
    hdr = gbuf.hdr_rim[:, :, :3].copy()
    rim = gbuf.hdr_rim[:, :, 3]
    nrm = gbuf.nrm_mat[:, :, :3]
    obj_id = gbuf.nrm_mat[:, :, 3]
    depth = gbuf.depth_hit[:, :, 0]
    hit_f = gbuf.depth_hit[:, :, 1]

    # Stage 1: Light-Modulated Sobel Ink Outlines
    if dna.style_mode == "anime_npr" and lit.ink_strength > 0.01:
        lz = np.log(np.clip(depth, 0.1, 40.0))
        gx_z = cv2.Sobel(lz, cv2.CV_32F, 1, 0, ksize=3)
        gy_z = cv2.Sobel(lz, cv2.CV_32F, 0, 1, ksize=3)
        edge_z = np.clip((np.hypot(gx_z, gy_z) - 0.25) / 0.65, 0.0, 1.0)
        gx_n = cv2.Sobel(nrm, cv2.CV_32F, 1, 0, ksize=3)
        gy_n = cv2.Sobel(nrm, cv2.CV_32F, 0, 1, ksize=3)
        edge_n = np.clip((np.linalg.norm(gx_n, axis=2) + np.linalg.norm(gy_n, axis=2) - 1.2) / 1.8, 0.0, 1.0)
        edge = np.clip(np.maximum(edge_z, edge_n * 0.65), 0.0, 1.0) * (hit_f > 0.7)
        ink_w = float(lit.ink_strength) * np.clip(1.0 - 0.92 * rim, 0.08, 1.0)
        ink_col = np.array([0.055, 0.035, 0.075], dtype=np.float32)[None, None, :]
        w_m = (edge * ink_w)[:, :, None]
        hdr = hdr * (1.0 - w_m) + ink_col * w_m

    # Stage 2: Foreground-Protected Vogel Disk Bokeh (12 taps fast-CPU profile)
    dof = float(dna.camera.dof_strength)
    if dof > 0.02:
        rel_z = np.where(depth > 500.0, 1.0, np.clip(np.abs(depth - gbuf.focus_depth) / np.maximum(depth, 0.35), 0.0, 1.0))
        coc_px = rel_z * dof * (min(w, h) * 0.012)
        if float(np.max(coc_px)) > 0.65:
            acc = hdr.copy()
            w_sum = np.ones((h, w), dtype=np.float32)
            golden = math.pi * (3.0 - math.sqrt(5.0))
            yy, xx = np.mgrid[0:h, 0:w]
            xx = xx.astype(np.float32)
            yy = yy.astype(np.float32)
            for i in range(12):
                r_f = math.sqrt((i + 0.5) / 12.0)
                ang = i * golden
                sx = np.clip(np.round(xx + coc_px * r_f * math.cos(ang)).astype(np.int32), 0, w - 1)
                sy = np.clip(np.round(yy + coc_px * r_f * math.sin(ang)).astype(np.int32), 0, h - 1)
                sz = depth[sy, sx]
                s_coc = coc_px[sy, sx]
                wt = np.where((sz >= depth - 0.18) | (s_coc > coc_px * 0.65), 1.0, 0.08).astype(np.float32)
                acc += hdr[sy, sx] * wt[:, :, None]
                w_sum += wt
            hdr = np.where((coc_px > 0.65)[:, :, None], acc / w_sum[:, :, None], hdr)

    # Stage 3: Screen-Space 16-Tap Radial Volumetric God-Rays
    if gbuf.godray_strength > 0.01:
        yy, xx = np.mgrid[0:h, 0:w]
        u = (xx.astype(np.float32) + 0.5) / w
        v = (yy.astype(np.float32) + 0.5) / h
        su, sv = gbuf.sun_uv
        du = (su - u) / 16.0
        dv = (sv - v) / 16.0
        occ_mask = (hit_f < 0.75).astype(np.float32)
        ray_acc = np.zeros((h, w), dtype=np.float32)
        decay = 1.0
        for k in range(1, 17):
            px = np.clip(np.round((u + du * k) * w - 0.5).astype(np.int32), 0, w - 1)
            py = np.clip(np.round((v + dv * k) * h - 0.5).astype(np.int32), 0, h - 1)
            ray_acc += occ_mask[py, px] * decay * 0.045
            decay *= 0.91
        aspect = w / max(1.0, float(h))
        sun_dist = np.hypot((u - su) * aspect, v - sv)
        phase = np.exp(-sun_dist * 1.45)
        hdr += gbuf.sun_col[None, None, :] * (ray_acc * phase * gbuf.godray_strength)[:, :, None]

    # Stage 4: Background Bokeh Orbs & Floating Dust Sparkles
    rng = np.random.default_rng(dna.seed + 505)
    min_d = float(min(w, h))
    if atmo.bokeh_count > 0:
        for _ in range(min(32, atmo.bokeh_count)):
            cx = float(rng.uniform(0.02, 0.98) * w)
            cy = float(rng.uniform(0.02, 0.82) * h)
            rad = float(rng.uniform(0.025, 0.072) * min_d * atmo.bokeh_size)
            x0, x1 = max(0, int(cx - rad * 1.3)), min(w, int(cx + rad * 1.3) + 1)
            y0, y1 = max(0, int(cy - rad * 1.3)), min(h, int(cy + rad * 1.3) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            lx = np.arange(x0, x1, dtype=np.float32)[None, :] - cx
            ly = np.arange(y0, y1, dtype=np.float32)[:, None] - cy
            rn = np.hypot(lx, ly) / rad
            disc = np.clip(1.0 - np.power(np.clip(rn, 0.0, 2.0), 4.0), 0.0, 1.0)
            ring = np.exp(-((rn - 0.82) ** 2) * 24.0) * 0.25
            bg_mask = (hit_f[y0:y1, x0:x1] < 0.75).astype(np.float32)
            hdr[y0:y1, x0:x1] += gbuf.sun_col[None, None, :] * ((disc * 0.25 + ring * 0.2) * bg_mask)[:, :, None]

    if atmo.dust_motes_count > 0:
        for _ in range(min(28, atmo.dust_motes_count)):
            cx = float(rng.uniform(0.03, 0.97) * w)
            cy = float(rng.uniform(0.03, 0.97) * h)
            rad = float(rng.uniform(0.004, 0.011) * min_d)
            x0, x1 = max(0, int(cx - rad * 1.6)), min(w, int(cx + rad * 1.6) + 1)
            y0, y1 = max(0, int(cy - rad * 1.6)), min(h, int(cy + rad * 1.6) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            lx = np.arange(x0, x1, dtype=np.float32)[None, :] - cx
            ly = np.arange(y0, y1, dtype=np.float32)[:, None] - cy
            rn = np.hypot(lx, ly) / rad
            hdr[y0:y1, x0:x1] += gbuf.sun_col[None, None, :] * (np.exp(-(rn * rn) * 2.5) * 0.8)[:, :, None]

    # Stage 5: HDR Bloom & Vignette
    if lit.bloom_intensity > 0.01:
        bright = np.maximum(hdr - 0.95, 0.0)
        b1 = cv2.GaussianBlur(bright, (0, 0), sigmaX=max(2.5, min_d * 0.012))
        b2 = cv2.GaussianBlur(bright, (0, 0), sigmaX=max(7.0, min_d * 0.032))
        hdr = hdr + (b1 * 0.55 + b2 * 0.45) * float(lit.bloom_intensity)

    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    hdr *= np.clip(1.0 - 0.16 * (xx * xx + yy * yy), 0.78, 1.0)[:, :, None]

    # Stage 6: ACES Filmic Tone Mapping + 1% Film Grain
    x = np.maximum(hdr * float(lit.exposure), 0.0)
    mapped = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)
    ldr = np.power(mapped, 1.0 / 2.2)
    if lit.film_grain > 1e-4:
        grain = rng.uniform(-0.5, 0.5, size=(h, w, 1)).astype(np.float32) * float(lit.film_grain)
        ldr = np.clip(ldr + grain, 0.0, 1.0)
    return np.clip(ldr * 255.0, 0, 255).astype(np.uint8)
