"""
Analytical 2D UV-Projected Eyes, Mouths & Garment Decals for PrismStudio.

Evaluated directly in cranial spherical UV coordinates or garment body-UV space
with zero protruding 3D eyeball spheres or blocky overlays.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _sd_ellipse_2d(px: np.ndarray, py: np.ndarray, rx: float, ry: float) -> np.ndarray:
    ax = max(1e-4, float(rx))
    ay = max(1e-4, float(ry))
    nx = px / ax
    ny = py / ay
    r = np.hypot(nx, ny)
    return (r - 1.0) * min(ax, ay)


def _sd_quadratic_bezier_2d(
    px: np.ndarray,
    py: np.ndarray,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    segments: int = 8,
) -> np.ndarray:
    """Fast piecewise quadratic Bezier curve 2D distance for anime upper/lower lashes."""
    ts = np.linspace(0.0, 1.0, segments + 1, dtype=np.float32)
    u = 1.0 - ts
    cx = (u * u * p0[0] + 2.0 * u * ts * p1[0] + ts * ts * p2[0]).astype(np.float32)
    cy = (u * u * p0[1] + 2.0 * u * ts * p1[1] + ts * ts * p2[1]).astype(np.float32)

    min_d2 = np.full(px.shape[0], 1e9, dtype=np.float32)
    for i in range(segments):
        ax, ay = cx[i], cy[i]
        bx, by = cx[i + 1], cy[i + 1]
        dx, dy = bx - ax, by - ay
        l2 = max(1e-8, dx * dx + dy * dy)
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / l2, 0.0, 1.0)
        qx = px - (ax + t * dx)
        qy = py - (ay + t * dy)
        min_d2 = np.minimum(min_d2, qx * qx + qy * qy)
    return np.sqrt(min_d2).astype(np.float32)


def evaluate_2d_eye_uv(
    u: np.ndarray,
    v: np.ndarray,
    *,
    eye_style: str = "anime_round",
    iris_color: np.ndarray,
    iris_secondary: np.ndarray,
    lash_color: np.ndarray,
    gaze_offset: Tuple[float, float] = (0.0, 0.0),
    eye_scale: float = 1.0,
    key_light_2d: Tuple[float, float] = (-0.5, 0.65),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates a crisp analytical 2D anime or feline eye in normalized eye-patch UV
    coordinates (u, v) centered at (0, 0) per eye, where |u| ~ 1 is eye half-width.
    Returns (rgb (N, 3), alpha (N,) in [0, 1]).
    """
    n = u.shape[0]
    s = max(0.4, float(eye_scale))
    ux = u / s
    vy = v / s

    rgb = np.zeros((n, 3), dtype=np.float32)
    alpha = np.zeros(n, dtype=np.float32)

    if eye_style == "closed_happy":
        d_arc = _sd_quadratic_bezier_2d(ux, vy, (-0.75, -0.05), (0.0, 0.38), (0.75, -0.05))
        lash_mask = np.clip(1.0 - (d_arc - 0.075) * 28.0, 0.0, 1.0)
        return np.broadcast_to(lash_color, (n, 3)).copy(), lash_mask.astype(np.float32)

    # Sclera boundary
    ry_eye = 0.68 if eye_style == "anime_round" else (0.58 if eye_style == "feline_slit" else 0.52)
    d_sclera = _sd_ellipse_2d(ux, vy, 0.78, ry_eye)
    sclera_mask = np.clip(1.0 - d_sclera * 32.0, 0.0, 1.0)

    # Gaze-shifted iris/pupil coordinates
    gx = ux - float(gaze_offset[0])
    gy = vy - float(gaze_offset[1])
    iris_rx = 0.42 if eye_style != "feline_slit" else 0.48
    iris_ry = 0.50 if eye_style != "feline_slit" else 0.50
    d_iris = _sd_ellipse_2d(gx, gy, iris_rx, iris_ry)
    iris_mask = np.clip(1.0 - d_iris * 38.0, 0.0, 1.0) * sclera_mask

    # Radial iris gradient + fibrous star rays
    ang = np.arctan2(gy, gx)
    rad_n = np.clip(np.hypot(gx / iris_rx, gy / iris_ry), 0.0, 1.2)
    fibers = 0.5 + 0.5 * np.sin(ang * 22.0 + rad_n * 4.0)
    grad_t = np.clip(gy / (2.0 * iris_ry) + 0.5, 0.0, 1.0)
    iris_rgb = (
        iris_secondary[None, :] * (1.0 - grad_t[:, None])
        + iris_color[None, :] * grad_t[:, None]
    ) * (0.86 + 0.24 * fibers[:, None])
    # Dark outer limbal ring
    limbal = np.clip(1.0 - np.abs(rad_n - 0.94) * 14.0, 0.0, 1.0)
    iris_rgb = iris_rgb * (1.0 - 0.55 * limbal[:, None])

    # Pupil (round anime vs sharp vertical feline slit!)
    if eye_style == "feline_slit":
        d_pupil = _sd_ellipse_2d(gx, gy, 0.11, 0.36)
    elif eye_style == "sharp_mystic":
        d_pupil = _sd_ellipse_2d(gx, gy, 0.16, 0.26)
    else:
        d_pupil = _sd_ellipse_2d(gx, gy, 0.18, 0.23)
    pupil_mask = np.clip(1.0 - d_pupil * 48.0, 0.0, 1.0) * iris_mask

    # Primary key-light catchlight + secondary lower-rim sparkle
    kx, ky = float(key_light_2d[0]) * 0.22, float(key_light_2d[1]) * 0.24
    d_catch1 = _sd_ellipse_2d(gx - kx, gy - ky, 0.13, 0.11)
    d_catch2 = _sd_ellipse_2d(gx + kx * 0.75, gy + 0.24, 0.065, 0.055)
    catch_mask = np.maximum(
        np.clip(1.0 - d_catch1 * 55.0, 0.0, 1.0),
        np.clip(1.0 - d_catch2 * 60.0, 0.0, 1.0) * 0.85,
    ) * sclera_mask

    # Upper & lower asymmetric quadratic Bezier eyelashes
    d_upper = _sd_quadratic_bezier_2d(ux, vy, (-0.86, 0.05), (-0.08, ry_eye * 1.18), (0.88, 0.12))
    upper_thick = np.where(ux > 0.2, 0.095, 0.072)
    upper_lash = np.clip(1.0 - (d_upper - upper_thick) * 34.0, 0.0, 1.0)

    d_lower = _sd_quadratic_bezier_2d(ux, vy, (-0.70, -ry_eye * 0.78), (0.0, -ry_eye * 0.98), (0.72, -ry_eye * 0.75))
    lower_lash = np.clip(1.0 - (d_lower - 0.032) * 36.0, 0.0, 1.0)
    lash_mask = np.maximum(upper_lash, lower_lash)

    # Composite sclera -> iris -> pupil -> catchlights -> lashes
    sclera_col = np.array([0.96, 0.97, 0.99], dtype=np.float32)[None, :] * (0.85 + 0.15 * np.clip(1.0 - gy, 0.0, 1.0))[:, None]
    pupil_col = np.array([0.04, 0.03, 0.07], dtype=np.float32)[None, :]
    catch_col = np.array([1.8, 1.85, 1.9], dtype=np.float32)[None, :]  # subtle HDR glint!

    rgb = sclera_col * (1.0 - iris_mask[:, None]) + iris_rgb * iris_mask[:, None]
    rgb = rgb * (1.0 - pupil_mask[:, None]) + pupil_col * pupil_mask[:, None]
    rgb = rgb * (1.0 - catch_mask[:, None]) + catch_col * catch_mask[:, None]
    rgb = rgb * (1.0 - lash_mask[:, None]) + lash_color[None, :] * lash_mask[:, None]

    alpha = np.clip(np.maximum(sclera_mask, lash_mask), 0.0, 1.0).astype(np.float32)
    return rgb.astype(np.float32), alpha


def evaluate_2d_garment_decal(
    kind: str,
    du: np.ndarray,
    dv: np.ndarray,
    size: float,
) -> np.ndarray:
    """
    Evaluates analytical 2D SDF garment emblems (star badge, medical cross,
    safety stripes, button seam, emblem circle) in local decal UV coordinates.
    Returns coverage mask in [0, 1].
    """
    s = max(0.02, float(size))
    x = du / s
    y = dv / s

    if kind == "star_badge":
        r = np.hypot(x, y)
        ang = np.arctan2(y, x) + math.pi * 0.5
        k = 5.0
        sector = 2.0 * math.pi / k
        a_local = ((ang % sector) + sector) % sector - 0.5 * sector
        star_r = 0.55 / np.maximum(np.cos(a_local) * 0.55 + np.abs(np.sin(a_local)) * 1.15, 0.25)
        return np.clip(1.0 - (r - star_r) * 24.0, 0.0, 1.0).astype(np.float32)

    if kind == "medical_cross":
        bar_h = np.maximum(np.abs(x) - 0.82, np.abs(y) - 0.28)
        bar_v = np.maximum(np.abs(x) - 0.28, np.abs(y) - 0.82)
        d = np.minimum(bar_h, bar_v)
        return np.clip(1.0 - d * 28.0, 0.0, 1.0).astype(np.float32)

    if kind == "safety_stripes":
        d = np.abs(y) - 0.22
        return np.clip(1.0 - d * 32.0, 0.0, 1.0).astype(np.float32)

    if kind == "button_seam":
        y_rep = ((y + 0.5) % 1.0) - 0.5
        d = np.hypot(x, y_rep) - 0.24
        return np.clip(1.0 - d * 32.0, 0.0, 1.0).astype(np.float32)

    if kind == "lace_border":
        wave = 0.5 + 0.5 * np.sin(x * 16.0) * np.cos(y * 16.0)
        in_band = (np.abs(y) < 0.7) & (wave > 0.35)
        return in_band.astype(np.float32)

    # emblem_circle
    d = np.hypot(x, y) - 0.75
    return np.clip(1.0 - d * 28.0, 0.0, 1.0).astype(np.float32)
