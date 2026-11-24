"""
Procedural 3D & Body-UV Surface Material Evaluators for PrismStudio.

Synthesizes organic skin, plush/tabby fur, brick masonry, Voronoi stone,
wood grain, brushed metal, fabric twill/silk/velvet, and foliage leaf patterns
purely from math—zero bitmap textures required.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def hex_to_rgb_linear(hex_color: str) -> np.ndarray:
    """Converts #rrggbb hex string to linear float32 RGB in [0, 1]."""
    s = hex_color.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        srgb = np.array([0.8, 0.82, 0.86], dtype=np.float32)
    else:
        srgb = np.array([int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)], dtype=np.float32) / 255.0
    return np.power(np.clip(srgb, 0.0, 1.0), 2.2).astype(np.float32)


def _hash2(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    h = np.sin(ix * 127.1 + iy * 311.7) * 43758.5453123
    return h - np.floor(h)


def value_noise_2d(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    ix, iy = np.floor(u), np.floor(v)
    fx, fy = u - ix, v - iy
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    a = _hash2(ix, iy)
    b = _hash2(ix + 1.0, iy)
    c = _hash2(ix, iy + 1.0)
    d = _hash2(ix + 1.0, iy + 1.0)
    return ((a * (1.0 - ux) + b * ux) * (1.0 - uy) + (c * (1.0 - ux) + d * ux) * uy).astype(np.float32)


def fbm_2d(u: np.ndarray, v: np.ndarray, octaves: int = 3) -> np.ndarray:
    total = np.zeros_like(u, dtype=np.float32)
    amp = 0.5
    freq = 1.0
    norm = 0.0
    for _ in range(octaves):
        total += amp * value_noise_2d(u * freq, v * freq)
        norm += amp
        freq *= 2.03
        amp *= 0.5
    return total / max(1e-6, norm)


def _triplanar_uv(pos: np.ndarray, normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    an = np.abs(normals)
    use_xz = (an[:, 1] >= an[:, 0]) & (an[:, 1] >= an[:, 2])
    use_yz = (~use_xz) & (an[:, 0] >= an[:, 2])
    u = np.where(use_xz, pos[:, 0], np.where(use_yz, pos[:, 2], pos[:, 0]))
    v = np.where(use_xz, pos[:, 2], np.where(use_yz, pos[:, 1], pos[:, 1]))
    return u.astype(np.float32), v.astype(np.float32)


def evaluate_procedural_material(
    shader: str,
    pos: np.ndarray,
    normals: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    scale: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (albedo_rgb (N, 3) linear, bump_height (N,) in [-1, 1]).
    """
    n = pos.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    if shader in ("solid", "solid_cel", "skin_sss"):
        return np.broadcast_to(c1, (n, 3)).copy(), np.zeros((n,), dtype=np.float32)

    u, v = _triplanar_uv(pos, normals)
    u *= float(scale)
    v *= float(scale)

    if shader == "plush_fur":
        streak = value_noise_2d(u * 6.0, v * 26.0)
        fine = value_noise_2d(u * 14.0, v * 52.0)
        f = 0.65 * streak + 0.35 * fine
        albedo = c1[None, :] * (0.9 + 0.18 * f[:, None]) + c2[None, :] * (0.08 * (1.0 - f[:, None]))
        return np.clip(albedo, 0.0, 1.0), ((f - 0.5) * 0.45).astype(np.float32)

    if shader == "tabby_stripes":
        warp = fbm_2d(u * 1.4, v * 1.4, octaves=2) * 0.85
        band = np.sin((u * 4.5 + warp * 4.0) * 3.14159)
        stripe = np.clip((band - 0.15) * 2.5, 0.0, 1.0)
        fur = value_noise_2d(u * 8.0, v * 28.0)
        albedo = c1[None, :] * (1.0 - 0.42 * stripe[:, None]) + c2[None, :] * (0.22 * (1.0 - fur[:, None]))
        return np.clip(albedo, 0.0, 1.0), ((fur - 0.5) * 0.35).astype(np.float32)

    if shader == "brick_masonry":
        course = np.floor(v)
        u_shift = u + 0.5 * (course % 2.0)
        fu = u_shift - np.floor(u_shift)
        fv = v - course
        mortar = (fu < 0.055) | (fu > 0.945) | (fv < 0.09) | (fv > 0.91)
        bid = _hash2(np.floor(u_shift), course)
        grain = fbm_2d(u_shift * 2.8, v * 3.5, octaves=2)
        brick_col = c1[None, :] * (0.78 + 0.34 * bid[:, None]) * (0.9 + 0.16 * (grain[:, None] - 0.5))
        albedo = np.where(mortar[:, None], c2[None, :], brick_col)
        bump = np.where(mortar, -0.75, 0.28 * (grain - 0.5)).astype(np.float32)
        return np.clip(albedo, 0.0, 1.0), bump

    if shader == "stone_voronoi":
        ix, iy = np.floor(u), np.floor(v)
        fx, fy = u - ix, v - iy
        d1 = np.full(n, 9.0, dtype=np.float32)
        d2 = np.full(n, 9.0, dtype=np.float32)
        cid = np.zeros(n, dtype=np.float32)
        for dx in (-1.0, 0.0, 1.0):
            for dy in (-1.0, 0.0, 1.0):
                rx = _hash2(ix + dx, iy + dy)
                ry = _hash2(ix + dx + 17.3, iy + dy + 9.1)
                dist = np.hypot(dx + rx - fx, dy + ry - fy).astype(np.float32)
                closer = dist < d1
                d2 = np.where(closer, d1, np.minimum(d2, dist))
                d1 = np.where(closer, dist, d1)
                cid = np.where(closer, rx, cid)
        edge = np.clip((d2 - d1) * 6.0, 0.0, 1.0)
        albedo = c2[None, :] * (1.0 - edge[:, None]) + c1[None, :] * edge[:, None] * (0.85 + 0.25 * cid[:, None])
        return np.clip(albedo, 0.0, 1.0), ((edge - 0.5) * 0.55).astype(np.float32)

    if shader == "foliage_leaf":
        n_val = fbm_2d(u * 2.2, v * 2.2, octaves=2)
        albedo = c1[None, :] * (0.78 + 0.38 * n_val[:, None]) + c3[None, :] * (0.15 * n_val[:, None])
        return np.clip(albedo, 0.0, 1.0), ((n_val - 0.5) * 0.35).astype(np.float32)

    if shader == "fabric_twill":
        twill = 0.5 + 0.5 * np.sin((u + v) * 22.0)
        albedo = c1[None, :] * (0.93 + 0.1 * twill[:, None])
        return np.clip(albedo, 0.0, 1.0), ((twill - 0.5) * 0.2).astype(np.float32)

    if shader == "fabric_velvet":
        weave = fbm_2d(u * 6.0, v * 6.0, octaves=2)
        albedo = c1[None, :] * (0.88 + 0.2 * weave[:, None])
        return np.clip(albedo, 0.0, 1.0), ((weave - 0.5) * 0.15).astype(np.float32)

    if shader == "plaid_tartan":
        twill = 0.5 + 0.5 * np.sin((u + v) * 26.0)
        pu, pv = u - np.floor(u), v - np.floor(v)
        bu = (pu > 0.18) & (pu < 0.44)
        bv = (pv > 0.18) & (pv < 0.44)
        tu = (np.abs(pu - 0.31) < 0.025) | (np.abs(pu - 0.78) < 0.022)
        tv = (np.abs(pv - 0.31) < 0.025) | (np.abs(pv - 0.78) < 0.022)
        albedo = np.broadcast_to(c1, (n, 3)).copy()
        albedo = np.where(bu[:, None], albedo * 0.45 + c2[None, :] * 0.55, albedo)
        albedo = np.where(bv[:, None], albedo * 0.45 + c2[None, :] * 0.55, albedo)
        albedo = np.where((tu | tv)[:, None], albedo * 0.25 + c3[None, :] * 0.75, albedo)
        return np.clip(albedo * (0.93 + 0.09 * twill[:, None]), 0.0, 1.0), ((twill - 0.5) * 0.18).astype(np.float32)

    if shader == "wood_grain":
        warp = fbm_2d(u * 0.9, v * 0.35, octaves=3) * 2.2
        rings = np.power(0.5 + 0.5 * np.sin((u * 4.5 + warp * 3.0) * 3.14159), 0.7)
        albedo = c2[None, :] * (1.0 - rings[:, None]) + c1[None, :] * rings[:, None]
        return np.clip(albedo, 0.0, 1.0), ((rings - 0.5) * 0.3).astype(np.float32)

    if shader == "brushed_metal":
        s1 = value_noise_2d(u * 0.8, v * 22.0)
        albedo = c1[None, :] * (0.88 + 0.22 * s1[:, None])
        return np.clip(albedo, 0.0, 1.0), ((s1 - 0.5) * 0.2).astype(np.float32)

    # Default subtle fBm variation
    f = fbm_2d(u, v, octaves=2)
    albedo = c2[None, :] * (0.12 * (1.0 - f[:, None])) + c1[None, :] * (0.88 + 0.18 * f[:, None])
    return np.clip(albedo, 0.0, 1.0), ((f - 0.5) * 0.25).astype(np.float32)
