"""
Pure Mathematical Procedural Pattern & Surface Shaders.

Evaluates rich procedural materials (tartan plaid, cable-knit braids, stone tiles,
brushed metal, wood grain, Voronoi stone, etched circuit traces, fBm noise) directly
from 3D surface coordinates using vectorized trigonometric & hash-noise math.
Zero image textures stored on disk.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def hex_to_rgb01(hex_color: str, linear: bool = True) -> np.ndarray:
    """Converts #rrggbb hex string to float32 RGB in [0, 1] (linearized by default for PBR shading)."""
    s = hex_color.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        srgb = np.array([0.8, 0.82, 0.86], dtype=np.float32)
    else:
        srgb = np.array([int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)], dtype=np.float32) / 255.0
    if linear:
        return np.power(np.clip(srgb, 0.0, 1.0), 2.2).astype(np.float32)
    return srgb


def _hash2(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    h = np.sin(ix * 127.1 + iy * 311.7) * 43758.5453123
    return h - np.floor(h)


def value_noise_2d(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    ix = np.floor(u)
    iy = np.floor(v)
    fx = u - ix
    fy = v - iy
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)

    a = _hash2(ix, iy)
    b = _hash2(ix + 1.0, iy)
    c = _hash2(ix, iy + 1.0)
    d = _hash2(ix + 1.0, iy + 1.0)

    return (a * (1.0 - ux) + b * ux) * (1.0 - uy) + (c * (1.0 - ux) + d * ux) * uy


def fbm_2d(u: np.ndarray, v: np.ndarray, octaves: int = 3) -> np.ndarray:
    total = np.zeros_like(u, dtype=np.float32)
    amp = 0.5
    freq = 1.0
    norm = 0.0
    for _ in range(octaves):
        total += amp * value_noise_2d(u * freq, v * freq).astype(np.float32)
        norm += amp
        freq *= 2.03
        amp *= 0.5
    return total / max(1e-6, norm)


def _triplanar_uv(pos: np.ndarray, normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Selects dominant projection UV per point to prevent stretching on steep sides."""
    an = np.abs(normals)
    use_xz = (an[:, 1] >= an[:, 0]) & (an[:, 1] >= an[:, 2])
    use_yz = (~use_xz) & (an[:, 0] >= an[:, 2])

    u = np.where(use_xz, pos[:, 0], np.where(use_yz, pos[:, 2], pos[:, 0]))
    v = np.where(use_xz, pos[:, 2], np.where(use_yz, pos[:, 1], pos[:, 1]))
    return u.astype(np.float32), v.astype(np.float32)


def evaluate_pattern(
    pattern: str,
    pos: np.ndarray,
    normals: np.ndarray,
    primary_hex: str,
    secondary_hex: str = "#334155",
    accent_hex: str = "#f59e0b",
    scale: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates a procedural pattern at N surface points.

    :param pos: (N, 3) float32 coordinates (world or local).
    :param normals: (N, 3) float32 unit normals.
    :return: (albedo_rgb (N, 3) in [0, 1], bump (N,) in [-1, 1]).
    """
    n = pos.shape[0]
    c1 = hex_to_rgb01(primary_hex)
    c2 = hex_to_rgb01(secondary_hex)
    c3 = hex_to_rgb01(accent_hex)

    if n == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    if pattern == "solid":
        return np.broadcast_to(c1, (n, 3)).copy(), np.zeros((n,), dtype=np.float32)

    u, v = _triplanar_uv(pos, normals)
    u = u * float(scale)
    v = v * float(scale)

    if pattern == "grid_tiles":
        fu = u - np.floor(u)
        fv = v - np.floor(v)
        grout = 0.065
        is_grout = (fu < grout) | (fu > 1.0 - grout) | (fv < grout) | (fv > 1.0 - grout)
        tile_id = _hash2(np.floor(u), np.floor(v)).astype(np.float32)
        grain = fbm_2d(u * 2.5, v * 2.5, octaves=2)
        tile_col = c1[None, :] * (0.88 + 0.18 * tile_id[:, None] + 0.08 * (grain[:, None] - 0.5))
        albedo = np.where(is_grout[:, None], c2[None, :], tile_col)
        bump = np.where(is_grout, -0.7, 0.15 * (grain - 0.5)).astype(np.float32)
        return np.clip(albedo, 0.0, 1.0), bump

    if pattern == "plaid_tartan":
        # Classic multi-frequency Scottish tartan weave via orthogonal periodic bands
        # Diagonal twill micro-modulation
        twill = 0.5 + 0.5 * np.sin((u + v) * 28.0)
        pu = u - np.floor(u)
        pv = v - np.floor(v)
        broad_u = (pu > 0.18) & (pu < 0.44)
        broad_v = (pv > 0.18) & (pv < 0.44)
        thin_u = (np.abs(pu - 0.31) < 0.025) | (np.abs(pu - 0.78) < 0.022)
        thin_v = (np.abs(pv - 0.31) < 0.025) | (np.abs(pv - 0.78) < 0.022)

        albedo = np.broadcast_to(c1, (n, 3)).copy()
        # Broad dark bands blend at 55% each, deeper at intersections
        albedo = np.where(broad_u[:, None], albedo * 0.45 + c2[None, :] * 0.55, albedo)
        albedo = np.where(broad_v[:, None], albedo * 0.45 + c2[None, :] * 0.55, albedo)
        # Fine accent overcheck stripes
        accent_mask = thin_u | thin_v
        albedo = np.where(accent_mask[:, None], albedo * 0.25 + c3[None, :] * 0.75, albedo)
        albedo *= (0.93 + 0.09 * twill[:, None])
        bump = (0.18 * (twill - 0.5)).astype(np.float32)
        return np.clip(albedo, 0.0, 1.0), bump

    if pattern == "cable_knit":
        # Braided sinusoidal cable-knit sweater waves + fine rib stitches
        col = u * 1.4
        row = v * 2.2
        col_frac = col - np.floor(col)
        # Intertwining sine strands inside each cable column
        s1 = np.sin(row * np.pi) * 0.22 + 0.5
        s2 = -np.sin(row * np.pi) * 0.22 + 0.5
        d1 = np.abs(col_frac - s1)
        d2 = np.abs(col_frac - s2)
        cable = np.maximum(np.clip(1.0 - d1 * 5.2, 0.0, 1.0), np.clip(1.0 - d2 * 5.2, 0.0, 1.0))
        stitch = 0.5 + 0.5 * np.cos(row * 14.0 + np.abs(col_frac - 0.5) * 10.0)
        h = np.clip(cable * 0.75 + stitch * 0.25, 0.0, 1.0)
        albedo = c2[None, :] * (1.0 - h[:, None]) * 0.35 + c1[None, :] * (0.65 + 0.35 * h[:, None])
        bump = ((h - 0.5) * 1.1).astype(np.float32)
        return np.clip(albedo, 0.0, 1.0), bump

    if pattern == "brushed_metal":
        streaks = value_noise_2d(u * 0.8, v * 24.0).astype(np.float32)
        fine = value_noise_2d(u * 2.0, v * 52.0).astype(np.float32)
        m = 0.65 * streaks + 0.35 * fine
        albedo = c1[None, :] * (0.86 + 0.24 * m[:, None]) + c2[None, :] * (0.08 * (1.0 - m[:, None]))
        bump = ((m - 0.5) * 0.25).astype(np.float32)
        return np.clip(albedo, 0.0, 1.0), bump

    if pattern == "wood_grain":
        warp = fbm_2d(u * 0.9, v * 0.35, octaves=3) * 2.2
        rings = 0.5 + 0.5 * np.sin((u * 4.5 + warp * 3.0) * np.pi)
        rings = np.power(rings, 0.7)
        albedo = c2[None, :] * (1.0 - rings[:, None]) + c1[None, :] * rings[:, None]
        bump = ((rings - 0.5) * 0.35).astype(np.float32)
        return np.clip(albedo, 0.0, 1.0), bump

    if pattern == "stone_voronoi":
        ix = np.floor(u)
        iy = np.floor(v)
        fx = u - ix
        fy = v - iy
        min_d1 = np.full(n, 9.0, dtype=np.float32)
        min_d2 = np.full(n, 9.0, dtype=np.float32)
        cell_val = np.zeros(n, dtype=np.float32)
        for dx in (-1.0, 0.0, 1.0):
            for dy in (-1.0, 0.0, 1.0):
                cx = ix + dx
                cy = iy + dy
                rx = _hash2(cx, cy)
                ry = _hash2(cx + 19.3, cy + 7.1)
                dist = np.hypot(dx + rx - fx, dy + ry - fy).astype(np.float32)
                closer = dist < min_d1
                min_d2 = np.where(closer, min_d1, np.minimum(min_d2, dist))
                min_d1 = np.where(closer, dist, min_d1)
                cell_val = np.where(closer, rx, cell_val)
        edge = np.clip((min_d2 - min_d1) * 6.0, 0.0, 1.0)
        albedo = c2[None, :] * (1.0 - edge[:, None]) + c1[None, :] * edge[:, None] * (0.82 + 0.3 * cell_val[:, None])
        bump = ((edge - 0.5) * 0.65).astype(np.float32)
        return np.clip(albedo, 0.0, 1.0), bump

    if pattern == "circuit_traces":
        fu = u - np.floor(u)
        fv = v - np.floor(v)
        cell_h = _hash2(np.floor(u), np.floor(v))
        trace_h = (np.abs(fv - 0.5) < 0.07) & (cell_h > 0.32)
        trace_v = (np.abs(fu - 0.5) < 0.07) & (cell_h < 0.68)
        pad = np.hypot(fu - 0.5, fv - 0.5) < 0.16
        copper = trace_h | trace_v | pad
        subgrid = (np.abs((u * 4.0) - np.floor(u * 4.0) - 0.5) > 0.45) | (
            np.abs((v * 4.0) - np.floor(v * 4.0) - 0.5) > 0.45
        )
        base = np.where(subgrid[:, None], c1[None, :] * 0.85, c1[None, :])
        albedo = np.where(copper[:, None], c3[None, :], base)
        bump = np.where(copper, 0.45, -0.05).astype(np.float32)
        return np.clip(albedo, 0.0, 1.0), bump

    if pattern == "stripes":
        band = 0.5 + 0.5 * np.sin(u * np.pi * 2.0)
        t = (band > 0.5).astype(np.float32)
        albedo = c1[None, :] * t[:, None] + c2[None, :] * (1.0 - t[:, None])
        return np.clip(albedo, 0.0, 1.0), ((t - 0.5) * 0.15).astype(np.float32)

    if pattern == "dots":
        fu = u - np.floor(u) - 0.5
        fv = v - np.floor(v) - 0.5
        is_dot = np.hypot(fu, fv) < 0.24
        albedo = np.where(is_dot[:, None], c2[None, :], c1[None, :])
        return np.clip(albedo, 0.0, 1.0), np.where(is_dot, 0.25, 0.0).astype(np.float32)

    # Default: fbm_noise
    f = fbm_2d(u, v, octaves=3)
    albedo = c2[None, :] * (1.0 - f[:, None]) + c1[None, :] * f[:, None]
    return np.clip(albedo, 0.0, 1.0), ((f - 0.5) * 0.45).astype(np.float32)
