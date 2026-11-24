"""
Universal Parametric Mathematical Part-Generators for PrismStudio.

Implements the 5 core pure-math operators (all vectorized float32 NumPy with
exact Lipschitz safety bounds, axial endcaps, and continuous material blending):
1. Universal Swept Super-Spline with Parallel-Transport Bishop Frames & Axial Endcaps
2. Semantic-Basis Real Spherical Harmonics Shell (L <= 3)
3. Single-Pass Body-UV Garment Shell & Joint-Driven Fold Operator
4. C2 Cubic Smooth Blending (smin / smax) + Continuous Material Weight Interpolation
5. Bounded & Half-Bond Staggered Masonry Domain Repetition
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


# ── Geometry Helpers & Rotations ─────────────────────────────────────────────


def rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Returns 3x3 float32 rotation matrix R = Rz @ Ry @ Rx."""
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def _normalize_vec3(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def _rodrigues_rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    ca = math.cos(angle)
    sa = math.sin(angle)
    return (v * ca + np.cross(axis, v) * sa + axis * (float(np.dot(axis, v)) * (1.0 - ca))).astype(np.float32)


# ── 1. Swept Super-Spline with Bishop Frames & Axial Endcaps ─────────────────


@dataclass
class BishopSplineStations:
    """Pre-sampled arc-length-uniform spline stations and tight bounding sphere."""

    points: np.ndarray          # (K, 3) float32
    tangents: np.ndarray        # (K, 3) float32 unit T
    normals: np.ndarray         # (K, 3) float32 unit N
    binormals: np.ndarray       # (K, 3) float32 unit B
    radii_a: np.ndarray         # (K,) float32 width along N
    radii_b: np.ndarray         # (K,) float32 thickness along B
    squareness: np.ndarray      # (K,) float32 superellipse n in [0.4, 4.0]
    t_vals: np.ndarray          # (K,) float32 normalized arc parameter in [0, 1]
    total_length: float
    cap_roundness: float        # 1.0 = organic dome cap, 0.1 = beveled flat cap
    bvh_center: np.ndarray      # (3,) float32
    bvh_radius: float


def _eval_curve_samples(ctrl: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Evaluates piecewise linear, quadratic, or cubic Bezier curve at parameter array ts."""
    n_ctrl = ctrl.shape[0]
    if n_ctrl == 2:
        return ((1.0 - ts[:, None]) * ctrl[0][None, :] + ts[:, None] * ctrl[1][None, :]).astype(np.float32)
    if n_ctrl == 3:
        u = (1.0 - ts)[:, None]
        t = ts[:, None]
        return (u * u * ctrl[0][None, :] + 2.0 * u * t * ctrl[1][None, :] + t * t * ctrl[2][None, :]).astype(np.float32)
    if n_ctrl == 4:
        u = (1.0 - ts)[:, None]
        t = ts[:, None]
        return (
            (u**3) * ctrl[0][None, :]
            + 3.0 * (u**2) * t * ctrl[1][None, :]
            + 3.0 * u * (t**2) * ctrl[2][None, :]
            + (t**3) * ctrl[3][None, :]
        ).astype(np.float32)
    # Catmull-Rom / piecewise polyline interpolation for >= 5 control points
    idx_f = ts * (n_ctrl - 1)
    i0 = np.clip(np.floor(idx_f).astype(np.int32), 0, n_ctrl - 1)
    i1 = np.minimum(i0 + 1, n_ctrl - 1)
    frac = (idx_f - i0)[:, None]
    return (ctrl[i0] * (1.0 - frac) + ctrl[i1] * frac).astype(np.float32)


def compute_bishop_frames(
    control_points: Sequence[Sequence[float]],
    radii_a: Sequence[float],
    radii_b: Optional[Sequence[float]] = None,
    squareness_n: Sequence[float] = (2.0, 2.0),
    *,
    n_stations: int = 10,
    cap_roundness: float = 1.0,
) -> BishopSplineStations:
    """
    Resamples a 3D curve into arc-length-uniform stations equipped with
    twist-free parallel-transport Bishop frames (T, N, B) and a tight BVH sphere.
    """
    ctrl = np.asarray(control_points, dtype=np.float32)
    if ctrl.ndim != 2 or ctrl.shape[0] < 2:
        ctrl = np.array([[0.0, 0.0, 0.0], [0.0, 0.4, 0.0]], dtype=np.float32)

    k = max(3, int(n_stations))
    dense_t = np.linspace(0.0, 1.0, 65, dtype=np.float32)
    dense_pts = _eval_curve_samples(ctrl, dense_t)
    diffs = np.linalg.norm(dense_pts[1:] - dense_pts[:-1], axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(diffs)]).astype(np.float32)
    total_len = float(cum_len[-1])

    if total_len < 1e-6:
        pts = np.repeat(ctrl[:1], k, axis=0)
        pts[:, 1] += np.linspace(-0.01, 0.01, k, dtype=np.float32)
        total_len = 0.02
        t_arc = np.linspace(0.0, 1.0, k, dtype=np.float32)
    else:
        target_s = np.linspace(0.0, total_len, k, dtype=np.float32)
        t_uniform = np.interp(target_s, cum_len, dense_t).astype(np.float32)
        pts = _eval_curve_samples(ctrl, t_uniform)
        t_arc = target_s / total_len

    # Tangents via central difference
    tangents = np.zeros((k, 3), dtype=np.float32)
    for i in range(k):
        if i == 0:
            tangents[i] = _normalize_vec3(pts[1] - pts[0])
        elif i == k - 1:
            tangents[i] = _normalize_vec3(pts[-1] - pts[-2])
        else:
            tangents[i] = _normalize_vec3(pts[i + 1] - pts[i - 1])

    # Parallel transport Bishop frame initialization
    t0 = tangents[0]
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(float(t0[2])) < 0.85 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    n0 = _normalize_vec3(ref - float(np.dot(ref, t0)) * t0)
    b0 = _normalize_vec3(np.cross(t0, n0))

    normals = np.zeros((k, 3), dtype=np.float32)
    binormals = np.zeros((k, 3), dtype=np.float32)
    normals[0] = n0
    binormals[0] = b0

    for i in range(1, k):
        t_prev = tangents[i - 1]
        t_cur = tangents[i]
        axis = np.cross(t_prev, t_cur)
        ax_len = float(np.linalg.norm(axis))
        n_cur = normals[i - 1].copy()
        if ax_len > 1e-6:
            axis_u = axis / ax_len
            angle = math.atan2(ax_len, float(np.clip(np.dot(t_prev, t_cur), -1.0, 1.0)))
            n_cur = _rodrigues_rotate(n_cur, axis_u, angle)
        # Re-orthogonalize cleanly
        n_cur = _normalize_vec3(n_cur - float(np.dot(n_cur, t_cur)) * t_cur)
        b_cur = _normalize_vec3(np.cross(t_cur, n_cur))
        normals[i] = n_cur
        binormals[i] = b_cur

    def _interp_profile(seq: Sequence[float], default_v: float, lo: float, hi: float) -> np.ndarray:
        arr = np.asarray(list(seq) if seq else [default_v, default_v], dtype=np.float32)
        if arr.size == 1:
            arr = np.array([arr[0], arr[0]], dtype=np.float32)
        src_t = np.linspace(0.0, 1.0, arr.size, dtype=np.float32)
        return np.clip(np.interp(t_arc, src_t, arr), lo, hi).astype(np.float32)

    ra = _interp_profile(radii_a, 0.06, 0.004, 2.0)
    rb = _interp_profile(radii_b if radii_b is not None else radii_a, 0.06, 0.004, 2.0)
    sq = _interp_profile(squareness_n, 2.0, 0.45, 4.0)

    bmin = np.min(pts, axis=0)
    bmax = np.max(pts, axis=0)
    center = 0.5 * (bmin + bmax)
    max_r_cross = float(np.max(np.maximum(ra, rb)))
    bvh_r = float(np.max(np.linalg.norm(pts - center[None, :], axis=1))) + max_r_cross * 1.35 + 0.04

    return BishopSplineStations(
        points=pts,
        tangents=tangents,
        normals=normals,
        binormals=binormals,
        radii_a=ra,
        radii_b=rb,
        squareness=sq,
        t_vals=t_arc,
        total_length=total_len,
        cap_roundness=float(np.clip(cap_roundness, 0.05, 1.0)),
        bvh_center=center.astype(np.float32),
        bvh_radius=bvh_r,
    )


def sd_swept_superspline(
    p: np.ndarray,
    stations: BishopSplineStations,
    *,
    return_uv: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates the Universal Swept Super-Spline SDF with closed rounded/beveled axial endcaps.
    Optionally returns cylindrical coordinates (t_star, theta) in [0,1] x [-pi, pi].
    """
    n_pts = p.shape[0]
    pts = stations.points
    k = pts.shape[0]

    # Segment projection across all K-1 linear spans -> (N, K-1)
    p0 = pts[:-1]                         # (M, 3)
    seg = pts[1:] - p0                    # (M, 3)
    seg_len2 = np.maximum(np.sum(seg * seg, axis=1), 1e-10)  # (M,)

    diff = p[:, None, :] - p0[None, :, :]  # (N, M, 3)
    s_all = np.clip(np.sum(diff * seg[None, :, :], axis=2) / seg_len2[None, :], 0.0, 1.0)
    closest_all = p0[None, :, :] + s_all[:, :, None] * seg[None, :, :]
    dist2_all = np.sum((p[:, None, :] - closest_all) ** 2, axis=2)

    best_seg = np.argmin(dist2_all, axis=1)
    row_idx = np.arange(n_pts)
    s_best = s_all[row_idx, best_seg]
    c_best = closest_all[row_idx, best_seg]

    t_star = ((best_seg.astype(np.float32) + s_best) / max(1, k - 1)).astype(np.float32)
    i0 = best_seg
    i1 = best_seg + 1
    w1 = s_best[:, None]
    w0 = 1.0 - w1

    T = stations.tangents[i0] * w0 + stations.tangents[i1] * w1
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-7)
    N = stations.normals[i0] * w0 + stations.normals[i1] * w1
    N -= np.sum(N * T, axis=1, keepdims=True) * T
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-7)
    B = np.cross(T, N)

    a = stations.radii_a[i0] * w0[:, 0] + stations.radii_a[i1] * w1[:, 0]
    b = stations.radii_b[i0] * w0[:, 0] + stations.radii_b[i1] * w1[:, 0]
    n_exp = stations.squareness[i0] * w0[:, 0] + stations.squareness[i1] * w1[:, 0]

    delta = p - c_best
    u = np.sum(delta * N, axis=1)
    v = np.sum(delta * B, axis=1)

    # Axial overhang beyond endpoint tangents (prevents infinite cylinders!)
    at_start = (best_seg == 0) & (s_best <= 1e-4)
    at_end = (best_seg == k - 2) & (s_best >= 1.0 - 1e-4)
    w_ax = np.where(
        at_start,
        np.maximum(-np.sum(delta * stations.tangents[0][None, :], axis=1), 0.0),
        np.where(at_end, np.maximum(np.sum(delta * stations.tangents[-1][None, :], axis=1), 0.0), 0.0),
    )

    # Overflow-safe 2D superellipse distance
    ru = np.abs(u) / a
    rv = np.abs(v) / b
    min_ab = np.minimum(a, b)
    max_r = np.maximum(ru, rv)
    safe_m = np.maximum(max_r, 1e-6)
    norm_cross = safe_m * np.power(
        np.power(ru / safe_m, n_exp) + np.power(rv / safe_m, n_exp),
        1.0 / n_exp,
    )
    d_cross = np.where(max_r < 1e-6, -min_ab, min_ab * (norm_cross - 1.0))

    if stations.cap_roundness >= 0.55:
        # C1-continuous 3D dome endcap centered at spline endpoint (zero interior jump)
        rw = w_ax / (min_ab * stations.cap_roundness)
        max_3 = np.maximum(max_r, rw)
        safe_3 = np.maximum(max_3, 1e-6)
        norm_3 = safe_3 * np.power(
            np.power(ru / safe_3, n_exp) + np.power(rv / safe_3, n_exp) + np.power(rw / safe_3, n_exp),
            1.0 / n_exp,
        )
        d_dome = np.where(max_3 < 1e-6, -min_ab, min_ab * (norm_3 - 1.0))
        d_raw = np.where(w_ax > 0.0, d_dome, d_cross)
    else:
        # Beveled flat-cut cap using signed distance to nearest spline endpoint
        s_arc = t_star * stations.total_length
        d_end_signed = np.where(
            w_ax > 0.0,
            w_ax,
            -np.minimum(s_arc, np.maximum(stations.total_length - s_arc, 0.0)),
        )
        bev = min_ab * stations.cap_roundness * 0.45
        d_raw = (
            np.hypot(np.maximum(d_cross + bev, 0.0), np.maximum(d_end_signed + bev, 0.0))
            + np.minimum(np.maximum(d_cross + bev, d_end_signed + bev), 0.0)
            - bev
        )

    # Conservative Lipschitz factor prevents sphere-tracing overshoot on curved/pinched cross-sections
    d_out = (d_raw * 0.72).astype(np.float32)

    if return_uv:
        theta = np.arctan2(v, u).astype(np.float32)
        return d_out, t_star, theta
    return d_out


# ── 2. Real Spherical Harmonics Cranial Shell (L <= 3, 16 coeffs) ────────────


def evaluate_sh_basis_l3(dirs: np.ndarray) -> np.ndarray:
    """
    Evaluates real Spherical Harmonics basis Y_l^m (l = 0..3 -> 16 orthonormal functions)
    directly in Cartesian unit direction (x, y, z) without trig singularities at poles.
    """
    x = dirs[:, 0]
    y = dirs[:, 1]
    z = dirs[:, 2]
    xx, yy, zz = x * x, y * y, z * z

    basis = np.empty((dirs.shape[0], 16), dtype=np.float32)
    # l = 0
    basis[:, 0] = 0.282095
    # l = 1 (y, z, x)
    basis[:, 1] = 0.488603 * y
    basis[:, 2] = 0.488603 * z
    basis[:, 3] = 0.488603 * x
    # l = 2
    basis[:, 4] = 1.092548 * x * y
    basis[:, 5] = 1.092548 * y * z
    basis[:, 6] = 0.315392 * (3.0 * zz - 1.0)
    basis[:, 7] = 1.092548 * x * z
    basis[:, 8] = 0.546274 * (xx - yy)
    # l = 3
    basis[:, 9] = 0.590044 * y * (3.0 * xx - yy)
    basis[:, 10] = 2.890611 * x * y * z
    basis[:, 11] = 0.457046 * y * (5.0 * zz - 1.0)
    basis[:, 12] = 0.373176 * z * (5.0 * zz - 3.0)
    basis[:, 13] = 0.457046 * x * (5.0 * zz - 1.0)
    basis[:, 14] = 1.445306 * z * (xx - yy)
    basis[:, 15] = 0.590044 * x * (xx - 3.0 * yy)
    return basis


def sd_sh_shell(
    p: np.ndarray,
    radii_ellipsoid: Tuple[float, float, float],
    sh_coeffs: np.ndarray,
    lipschitz: float = 0.55,
) -> np.ndarray:
    """
    Evaluates an asymmetric Spherical Harmonics radial cranial/torso shell around an
    ellipsoidal base metric with origin regularization to prevent r=0 NaNs.
    """
    scale = np.maximum(np.array(radii_ellipsoid, dtype=np.float32), 1e-4)
    q = p / scale[None, :]
    r_reg = np.sqrt(np.sum(q * q, axis=1) + 1e-8)
    dirs = q / r_reg[:, None]
    basis = evaluate_sh_basis_l3(dirs)
    r_target = np.clip(1.0 + basis @ sh_coeffs.astype(np.float32), 0.42, 1.95)
    min_s = float(np.min(scale))
    return ((r_reg - r_target) * min_s * float(lipschitz)).astype(np.float32)


# ── 3. Body-UV Garment Shell & Joint-Fold Operator ───────────────────────────


def sd_cloth_shell(
    d_body: np.ndarray,
    t_uv: np.ndarray,
    theta_uv: np.ndarray,
    *,
    fit_offset: float = 0.018,
    thickness: float = 0.012,
    flare: float = 0.0,
    flare_start_t: float = 0.35,
    pleat_count: int = 0,
    pleat_amp: float = 0.0,
    ruffle_amp: float = 0.0,
    fold_amp: float = 0.018,
    joint_bend_rad: float = 0.0,
    open_front_rad: float = 0.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    w_ax: np.ndarray | float = 0.0,
) -> np.ndarray:
    """
    Single-pass Body-UV cloth shell offset from pre-evaluated body SDF and local (t, theta).
    Uses C1-smooth cosine pleats, soft cuff tapering, and C2 smooth trim boundary.
    """
    t_past = np.maximum(t_uv - float(flare_start_t), 0.0)
    pleat_d = min(0.009, float(pleat_amp)) * np.sin(theta_uv * float(pleat_count)) * t_past if pleat_count > 0 else 0.0
    hem_near = np.maximum(1.0 - np.abs(t_uv - t_max) * 4.5, 0.0)
    ruffle_d = min(0.009, float(ruffle_amp)) * np.sin(12.0 * theta_uv) * hem_near if ruffle_amp > 0.0 else 0.0
    flare_d = float(flare) * np.power(t_past, 1.35)

    # Gentle joint-bend wrinkles localized around mid-bone (t ~ 0.5)
    joint_dist = np.abs(t_uv - 0.5)
    fold_d = min(0.005, float(fold_amp)) * float(joint_bend_rad) * np.exp(-4.0 * joint_dist) * np.sin(3.0 * theta_uv + 4.0 * t_uv)

    lo_t = np.clip((t_uv - (float(t_min) - 0.03)) / 0.15, 0.0, 1.0) if t_min > 0.04 else 1.0
    hi_t = np.clip(((float(t_max) + 0.03) - t_uv) / 0.15, 0.0, 1.0)
    hem_taper = (lo_t * lo_t * (3.0 - 2.0 * lo_t) if t_min > 0.04 else 1.0) * (hi_t * hi_t * (3.0 - 2.0 * hi_t))

    delta_eff = np.maximum(0.004, float(fit_offset) * (0.45 + 0.55 * hem_taper) + flare_d + pleat_d + ruffle_d + fold_d)
    d_shell = d_body - delta_eff

    # Trim mask in body-UV space including axial overhang at open distal/cut boundaries
    w_eff = np.where((t_uv > 0.5) | (float(t_min) > 0.02), np.asarray(w_ax, dtype=np.float32) * 0.78, 0.0)
    d_trim_t = w_eff + np.maximum(float(t_min) - t_uv, t_uv - float(t_max)) * 0.45
    if open_front_rad > 0.02:
        # Angle 0 faces +Z front
        d_open = float(open_front_rad) - np.abs(theta_uv)
        d_trim = np.maximum(d_trim_t, d_open * 0.25)
    else:
        d_trim = d_trim_t

    return (smax_cubic(d_shell, d_trim, 0.014) * 0.68).astype(np.float32)


# ── 4. C2 Cubic Smooth Blending & Continuous Material Interpolation ──────────


def smin_cubic_mat(d_a: np.ndarray, d_b: np.ndarray, k: float = 0.06) -> Tuple[np.ndarray, np.ndarray]:
    """
    C2-continuous cubic polynomial smooth minimum returning (d_blended, blend_weight_eta).
    eta == 0 selects material A, eta == 1 selects material B, smoothly transitioning across joints.
    """
    if k <= 1e-5:
        eta = (d_b < d_a).astype(np.float32)
        return np.minimum(d_a, d_b).astype(np.float32), eta

    h = np.maximum(float(k) - np.abs(d_a - d_b), 0.0) / float(k)
    d_blend = np.minimum(d_a, d_b) - (1.0 / 6.0) * float(k) * (h * h * h)
    m = 0.5 * h * h
    eta = np.where(d_a < d_b, m, 1.0 - m)
    return d_blend.astype(np.float32), eta.astype(np.float32)


def smax_cubic(d_a: np.ndarray, d_b: np.ndarray, k: float = 0.04) -> np.ndarray:
    """C2 smooth maximum (used with -d_cut for smooth carving of mouths/sockets)."""
    d, _ = smin_cubic_mat(-d_a, -d_b, k)
    return (-d).astype(np.float32)


# ── 5. Bounded & Half-Bond Staggered Masonry Repetition ──────────────────────


def domain_repeat_bounded(
    p: np.ndarray,
    spacing: Tuple[float, float, float],
    bounds: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Folds 3D space onto a bounded integer grid [-L, L] in O(1) time.
    Returns (q_local, cell_ids) so hashes can vary per-cell height/color/tilt.
    """
    s = np.maximum(np.array(spacing, dtype=np.float32), 1e-4)
    L = np.maximum(np.array(bounds, dtype=np.float32), 0.0)
    cell_id = np.clip(np.round(p / s[None, :]), -L[None, :], L[None, :])
    q = p - s[None, :] * cell_id
    return q.astype(np.float32), cell_id.astype(np.int32)


def domain_repeat_masonry_staggered(
    p: np.ndarray,
    brick_pitch: Tuple[float, float, float],
    bounds: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Half-bond staggered masonry domain repetition: alternate Y courses shift in X by 0.5 * pitch_x.
    Produces authentic brick/stone wall courses in O(1) evaluation time.
    """
    sx, sy, sz = max(1e-4, float(brick_pitch[0])), max(1e-4, float(brick_pitch[1])), max(1e-4, float(brick_pitch[2]))
    lx, ly, lz = max(0, int(bounds[0])), max(0, int(bounds[1])), max(0, int(bounds[2]))

    iy = np.clip(np.round(p[:, 1] / sy), -ly, ly)
    stagger_x = 0.5 * sx * (np.abs(iy) % 2.0)
    ix = np.clip(np.round((p[:, 0] - stagger_x) / sx), -lx, lx)
    iz = np.clip(np.round(p[:, 2] / sz), -lz, lz)

    qx = p[:, 0] - (ix * sx + stagger_x)
    qy = p[:, 1] - iy * sy
    qz = p[:, 2] - iz * sz
    q = np.stack([qx, qy, qz], axis=1).astype(np.float32)
    cell_id = np.stack([ix, iy, iz], axis=1).astype(np.int32)
    return q, cell_id


def sd_rounded_box(p: np.ndarray, half_extents: Tuple[float, float, float], radius: float = 0.025) -> np.ndarray:
    b = np.maximum(np.array(half_extents, dtype=np.float32), 1e-4)
    rad = float(min(radius, float(np.min(b)) * 0.48))
    q = np.abs(p) - (b - rad)[None, :]
    return (np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0) - rad).astype(np.float32)


def sd_torus(p: np.ndarray, major_r: float, minor_r: float) -> np.ndarray:
    qx = np.hypot(p[:, 0], p[:, 2]) - max(1e-4, float(major_r))
    return (np.hypot(qx, p[:, 1]) - max(1e-4, float(minor_r))).astype(np.float32)
