"""
Dynamic Vectorized Math Core (Track 1).

Pre-written analytic geometry, trigonometric shape synthesis, implicit surfaces,
CSG operators, domain warps, and BVH ray-AABB slab culling—all vectorized in NumPy
float32 with zero hardcoded object names.

Frameworks implemented:
1. Signed Distance Functions (Inigo Quilez exact formulas)
2. Superquadrics / Superellipsoids (Barr, 1981)
3. Johan Gielis 2D/3D Superformula (pure trigonometric morphing, 2003)
4. Skeletal Convolution / Tapered Capsule Limbs (Bloomenthal & Shoemake)
5. Surfaces of Revolution (Lathe profiles) & Swept Quadratic Bézier Tubes
6. Parametric L-System Branching Trees -> Tapered Capsule Skeletons
7. CSG Booleans (union, smooth_union, subtract, intersect) & Domain Modifiers
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Rotation & Coordinate Helpers ────────────────────────────────────────────


def rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Returns 3x3 rotation matrix R = Rz @ Ry @ Rx (float32)."""
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def ray_aabb_intersect(
    ro: np.ndarray,
    rd: np.ndarray,
    bmin: np.ndarray,
    bmax: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized slab ray-AABB intersection for N rays.
    :param ro: (N, 3) ray origins
    :param rd: (N, 3) normalized ray directions
    :param bmin: (3,) box min corner
    :param bmax: (3,) box max corner
    :return: (hit_mask (N,) bool, t_enter (N,) float32, t_exit (N,) float32)
    """
    inv_rd = np.where(np.abs(rd) > 1e-7, 1.0 / np.where(np.abs(rd) > 1e-7, rd, 1.0), 1e7)
    t0 = (bmin[None, :] - ro) * inv_rd
    t1 = (bmax[None, :] - ro) * inv_rd
    tmin = np.minimum(t0, t1)
    tmax = np.maximum(t0, t1)
    t_enter = np.max(tmin, axis=1)
    t_exit = np.min(tmax, axis=1)
    hit = (t_exit >= np.maximum(t_enter, 0.0)) & (t_exit > 0.0)
    return hit, np.maximum(t_enter, 0.0).astype(np.float32), t_exit.astype(np.float32)


# ── Exact SDF Primitives (Vectorized over p: (N, 3) float32) ─────────────────


def sd_sphere(p: np.ndarray, radius: float) -> np.ndarray:
    return np.linalg.norm(p, axis=1) - float(radius)


def sd_ellipsoid(p: np.ndarray, radii: Tuple[float, float, float]) -> np.ndarray:
    r = np.maximum(np.array(radii, dtype=np.float32), 1e-4)
    k0 = np.linalg.norm(p / r[None, :], axis=1)
    k1 = np.linalg.norm(p / (r * r)[None, :], axis=1)
    return np.where(k1 > 1e-7, k0 * (k0 - 1.0) / np.maximum(k1, 1e-7), -float(np.min(r)))


def sd_rounded_box(p: np.ndarray, half_extents: Tuple[float, float, float], radius: float = 0.04) -> np.ndarray:
    b = np.maximum(np.array(half_extents, dtype=np.float32), 1e-4)
    rad = float(min(radius, float(np.min(b)) * 0.48))
    q = np.abs(p) - (b - rad)[None, :]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside - rad


def sd_superquadric(
    p: np.ndarray,
    half_extents: Tuple[float, float, float],
    e1: float = 0.5,
    e2: float = 0.5,
) -> np.ndarray:
    """
    Barr (1981) Superellipsoid implicit SDF approximation:
    F(x,y,z) = ((|x/a1|^(2/e2) + |y/a2|^(2/e2))^(e2/e1) + |z/a3|^(2/e1))^(e1/2) - 1
    Smoothly morphs between box (e->0.1), cylinder (e1=0.1, e2=1.0), sphere (1,1), octahedron (2,2).
    """
    a = np.maximum(np.array(half_extents, dtype=np.float32), 1e-4)
    e1 = float(np.clip(e1, 0.12, 2.2))
    e2 = float(np.clip(e2, 0.12, 2.2))
    q = np.abs(p) / a[None, :]
    # Normalize to avoid overflow when far from origin
    scale_q = np.maximum(np.max(q, axis=1), 1e-6)
    qn = q / scale_q[:, None]

    inv_e2 = 2.0 / e2
    inv_e1 = 2.0 / e1
    xy = np.power(np.maximum(qn[:, 0], 1e-6), inv_e2) + np.power(np.maximum(qn[:, 2], 1e-6), inv_e2)
    xy_e = np.power(np.maximum(xy, 1e-12), e2 / e1)
    xyz = xy_e + np.power(np.maximum(qn[:, 1], 1e-6), inv_e1)
    f_val = np.power(np.maximum(xyz, 1e-12), e1 * 0.5) * scale_q

    # Scale into Lipschitz-safe distance using minimum semi-axis
    min_a = float(np.min(a))
    return (f_val - 1.0) * min_a * 0.72


def sd_cylinder(p: np.ndarray, radius: float, half_height: float, bevel: float = 0.02) -> np.ndarray:
    r = max(1e-4, float(radius))
    h = max(1e-4, float(half_height))
    bev = float(min(bevel, r * 0.35, h * 0.35))
    d_xz = np.hypot(p[:, 0], p[:, 2]) - (r - bev)
    d_y = np.abs(p[:, 1]) - (h - bev)
    outside = np.hypot(np.maximum(d_xz, 0.0), np.maximum(d_y, 0.0))
    inside = np.minimum(np.maximum(d_xz, d_y), 0.0)
    return outside + inside - bev


def sd_cone_frustum(p: np.ndarray, r_bottom: float, r_top: float, half_height: float) -> np.ndarray:
    """Exact capped cone / frustum along Y axis from -half_height to +half_height."""
    r1 = max(1e-4, float(r_bottom))
    r2 = max(1e-4, float(r_top))
    h = max(1e-4, float(half_height))
    qx = np.hypot(p[:, 0], p[:, 2])
    qy = p[:, 1]
    k1x, k1y = r2, h
    k2x, k2y = r2 - r1, 2.0 * h
    cax = qx - np.minimum(qx, np.where(qy < 0.0, r1, r2))
    cay = np.abs(qy) - h
    denom = max(1e-7, k2x * k2x + k2y * k2y)
    t = np.clip(((k1x - qx) * k2x + (k1y - qy) * k2y) / denom, 0.0, 1.0)
    cbx = qx - k1x + k2x * t
    cby = qy - k1y + k2y * t
    s = np.where((cbx < 0.0) & (cay < 0.0), -1.0, 1.0)
    d2 = np.minimum(cax * cax + cay * cay, cbx * cbx + cby * cby)
    return (s * np.sqrt(np.maximum(d2, 0.0))).astype(np.float32)


def sd_capsule_segment(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    r_a: float,
    r_b: float,
) -> np.ndarray:
    """Tapered capsule / round-cone bone segment from point a (radius r_a) to b (radius r_b)."""
    pa = p - a[None, :]
    ba = b - a
    ba_dot = float(np.dot(ba, ba))
    if ba_dot < 1e-8:
        return np.linalg.norm(pa, axis=1) - float(max(r_a, r_b))
    h = np.clip((pa @ ba) / ba_dot, 0.0, 1.0)
    closest = pa - h[:, None] * ba[None, :]
    r = r_a + (r_b - r_a) * h
    return (np.linalg.norm(closest, axis=1) - r).astype(np.float32)


def sd_torus(p: np.ndarray, major_r: float, minor_r: float) -> np.ndarray:
    R = max(1e-4, float(major_r))
    r = max(1e-4, float(minor_r))
    qx = np.hypot(p[:, 0], p[:, 2]) - R
    return (np.hypot(qx, p[:, 1]) - r).astype(np.float32)


def gielis_radius(phi: np.ndarray, m: float, n1: float, n2: float, n3: float, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    """Johan Gielis (2003) pure trigonometric Superformula r(phi)."""
    t1 = np.abs(np.cos(m * phi * 0.25) / max(1e-4, a))
    t2 = np.abs(np.sin(m * phi * 0.25) / max(1e-4, b))
    val = np.power(np.maximum(t1, 1e-6), n2) + np.power(np.maximum(t2, 1e-6), n3)
    r = np.power(np.maximum(val, 1e-6), -1.0 / max(0.15, n1))
    return np.clip(r, 0.05, 4.0).astype(np.float32)


def sd_gielis_supershape(
    p: np.ndarray,
    half_extents: Tuple[float, float, float],
    m: float = 6.0,
    n1: float = 1.0,
    n2: float = 1.7,
    n3: float = 1.7,
    m2: Optional[float] = None,
) -> np.ndarray:
    """
    3D Gielis Supershape via spherical product of trigonometric superformulas.
    """
    scale = np.maximum(np.array(half_extents, dtype=np.float32), 1e-4)
    q = p / scale[None, :]
    r_actual = np.linalg.norm(q, axis=1)
    theta = np.arctan2(q[:, 2], q[:, 0])
    phi = np.arcsin(np.clip(q[:, 1] / np.maximum(r_actual, 1e-6), -1.0, 1.0))

    r1 = gielis_radius(theta, m=m, n1=n1, n2=n2, n3=n3)
    r2 = gielis_radius(phi, m=m if m2 is None else m2, n1=max(0.4, n1), n2=n2, n3=n3)
    target_r = r1 * r2
    # Conservative Lipschitz factor (0.45) prevents sphere-tracing overshoot on spiky lobes
    return ((r_actual - target_r) * float(np.min(scale)) * 0.45).astype(np.float32)


def sd_lathe_profile(
    p: np.ndarray,
    half_height: float,
    max_radius: float,
    profile_radii: List[float],
    hollow: bool = False,
    wall_thickness: float = 0.06,
) -> np.ndarray:
    """
    Surface of revolution: revolves a 1D radius curve r(t) (t in [0, 1] bottom-to-top)
    360 degrees around the Y axis. Ideal for mugs, vases, bottles, lampshades, nozzles.
    """
    h = max(1e-4, float(half_height))
    mr = max(1e-4, float(max_radius))
    prof = np.array(profile_radii if profile_radii and len(profile_radii) >= 2 else [0.8, 1.0, 0.7], dtype=np.float32)
    prof = np.clip(prof, 0.04, 1.5) * mr

    # Normalized height t in [0, 1]
    t = np.clip((p[:, 1] + h) / (2.0 * h), 0.0, 1.0)
    idx_f = t * (len(prof) - 1)
    i0 = np.floor(idx_f).astype(np.int32)
    i1 = np.minimum(i0 + 1, len(prof) - 1)
    frac = idx_f - i0
    rad_at_y = prof[i0] * (1.0 - frac) + prof[i1] * frac

    r_xz = np.hypot(p[:, 0], p[:, 2])
    d_radial = r_xz - rad_at_y
    d_y = np.abs(p[:, 1]) - h

    if hollow:
        # Hollow vessel open at the top (y = +h)
        wall = max(0.015, float(wall_thickness) * mr)
        shell_side = np.abs(d_radial) - wall
        bottom_cap = np.maximum(r_xz - prof[0], np.abs(p[:, 1] + h) - wall)
        d_body = np.maximum(shell_side, d_y)
        return (np.minimum(d_body, bottom_cap) * 0.75).astype(np.float32)

    outside = np.hypot(np.maximum(d_radial, 0.0), np.maximum(d_y, 0.0))
    inside = np.minimum(np.maximum(d_radial, d_y), 0.0)
    return ((outside + inside) * 0.75).astype(np.float32)


def sd_convolution_limb(
    p: np.ndarray,
    points: List[Tuple[float, float, float]],
    radii: List[float],
    smooth_k: float = 0.08,
) -> np.ndarray:
    """
    Skeletal implicit surface / smooth-blended tapered capsule chain along 3D keypoints.
    Prevents lumpy metaballs while smoothly joining articulated limbs, branches, and bodies.
    """
    if len(points) < 2:
        pts = [np.array([0.0, -0.5, 0.0], dtype=np.float32), np.array([0.0, 0.5, 0.0], dtype=np.float32)]
        rads = [0.2, 0.15]
    else:
        pts = [np.array(pt, dtype=np.float32) for pt in points]
        rads = [max(0.015, float(r)) for r in radii]
        while len(rads) < len(pts):
            rads.append(rads[-1])

    d = sd_capsule_segment(p, pts[0], pts[1], rads[0], rads[1])
    for i in range(1, len(pts) - 1):
        seg_d = sd_capsule_segment(p, pts[i], pts[i + 1], rads[i], rads[i + 1])
        d = op_smooth_union(d, seg_d, smooth_k)
    return d


def sd_bezier_tube(
    p: np.ndarray,
    control_points: List[Tuple[float, float, float]],
    r_start: float = 0.08,
    r_end: float = 0.08,
    segments: int = 10,
) -> np.ndarray:
    """Sweeps a smooth tapered circular cross-section along a quadratic or cubic Bézier spline."""
    if len(control_points) < 3:
        pts = [
            np.array([-0.5, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.5, 0.0], dtype=np.float32),
            np.array([0.5, 0.0, 0.0], dtype=np.float32),
        ]
    else:
        pts = [np.array(pt, dtype=np.float32) for pt in control_points[:4]]

    ts = np.linspace(0.0, 1.0, max(3, segments + 1), dtype=np.float32)
    sampled = []
    for t in ts:
        u = 1.0 - t
        if len(pts) == 3:
            pt = u * u * pts[0] + 2.0 * u * t * pts[1] + t * t * pts[2]
        else:
            pt = (u**3) * pts[0] + 3.0 * (u**2) * t * pts[1] + 3.0 * u * (t**2) * pts[2] + (t**3) * pts[3]
        sampled.append(tuple(pt.tolist()))

    radii = [float(r_start + (r_end - r_start) * t) for t in ts]
    return sd_convolution_limb(p, sampled, radii, smooth_k=max(0.02, min(r_start, r_end) * 0.4))


def generate_lsystem_segments(
    height: float = 1.0,
    base_radius: float = 0.1,
    depth: int = 3,
    angle_deg: float = 30.0,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
    """
    Deterministic parametric 3D branching tree skeleton (L-system style).
    Returns a list of (start_pt, end_pt, r0, r1) tapered bone segments.
    """
    depth = max(1, min(4, int(depth)))
    rng = np.random.default_rng(seed)
    segments: List[Tuple[np.ndarray, np.ndarray, float, float]] = []

    def _branch(origin: np.ndarray, direction: np.ndarray, length: float, rad: float, lvl: int) -> None:
        d_norm = direction / max(1e-6, float(np.linalg.norm(direction)))
        end = origin + d_norm * length
        r_next = max(0.012, rad * 0.65)
        segments.append((origin.astype(np.float32), end.astype(np.float32), float(rad), float(r_next)))
        if lvl <= 1:
            return

        # Build orthonormal frame around d_norm
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(d_norm[0]) < 0.8 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        u_axis = np.cross(d_norm, ref)
        u_axis /= max(1e-6, float(np.linalg.norm(u_axis)))
        v_axis = np.cross(d_norm, u_axis)

        n_children = 3 if lvl >= 3 else 2
        tilt = math.radians(max(12.0, min(65.0, float(angle_deg))))
        base_az = float(rng.uniform(0.0, 2.0 * math.pi))
        for k in range(n_children):
            az = base_az + (2.0 * math.pi * k) / n_children + float(rng.uniform(-0.3, 0.3))
            radial = math.cos(az) * u_axis + math.sin(az) * v_axis
            child_dir = math.cos(tilt) * d_norm + math.sin(tilt) * radial
            _branch(end, child_dir, length * 0.68, r_next, lvl - 1)

    root = np.array([0.0, -height * 0.5, 0.0], dtype=np.float32)
    _branch(root, np.array([0.0, 1.0, 0.0], dtype=np.float32), height * 0.42, base_radius, depth)
    if segments:
        all_pts = np.vstack([s[0] for s in segments] + [s[1] for s in segments])
        mid = 0.5 * (np.min(all_pts, axis=0) + np.max(all_pts, axis=0))
        segments = [(a - mid, b - mid, r0, r1) for a, b, r0, r1 in segments]
    return segments


def sd_lsystem_segments(
    p: np.ndarray,
    segments: List[Tuple[np.ndarray, np.ndarray, float, float]],
    smooth_k: float = 0.03,
) -> np.ndarray:
    if not segments:
        return sd_sphere(p, 0.5)
    a, b, r0, r1 = segments[0]
    d = sd_capsule_segment(p, a, b, r0, r1)
    for a, b, r0, r1 in segments[1:]:
        seg_d = sd_capsule_segment(p, a, b, r0, r1)
        d = op_smooth_union(d, seg_d, smooth_k)
    return d


# ── CSG Boolean & Smooth Operators ───────────────────────────────────────────


def op_union(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    return np.minimum(d1, d2)


def op_smooth_union(d1: np.ndarray, d2: np.ndarray, k: float = 0.08) -> np.ndarray:
    if k <= 1e-5:
        return np.minimum(d1, d2)
    h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return d2 * (1.0 - h) + d1 * h - k * h * (1.0 - h)


def op_subtract(d_base: np.ndarray, d_cut: np.ndarray) -> np.ndarray:
    return np.maximum(d_base, -d_cut)


def op_intersect(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    return np.maximum(d1, d2)


# ── Dynamic Domain Modifiers (Warps, Symmetry & Repetition) ──────────────────


def apply_domain_modifiers(p: np.ndarray, modifiers: List[Dict[str, Any]]) -> Tuple[np.ndarray, float]:
    """
    Applies a dynamic chain of domain transformations (mirror, finite linear/grid/radial
    arrays, bend, twist, taper) to point batch p: (N, 3).
    Returns (transformed_p, distance_lipschitz_scale).
    """
    q = p.copy()
    lip_scale = 1.0

    for mod in modifiers:
        kind = mod.get("kind", "")
        params = mod.get("params", {}) or {}

        if kind == "mirror_x":
            offset = float(params.get("offset", 0.0))
            q[:, 0] = np.abs(q[:, 0]) - offset
        elif kind == "mirror_z":
            offset = float(params.get("offset", 0.0))
            q[:, 2] = np.abs(q[:, 2]) - offset
        elif kind == "array_linear":
            axis = str(params.get("axis", "x")).lower()
            ax_i = {"x": 0, "y": 1, "z": 2}.get(axis, 0)
            count = max(1, min(32, int(params.get("count", 4))))
            spacing = max(0.02, float(params.get("spacing", 0.3)))
            if count > 1:
                half_span = (count - 1) * 0.5
                idx = np.clip(np.round(q[:, ax_i] / spacing + half_span), 0.0, float(count - 1)) - half_span
                q[:, ax_i] = q[:, ax_i] - idx * spacing
        elif kind == "array_grid":
            nx = max(1, min(24, int(params.get("nx", 4))))
            nz = max(1, min(24, int(params.get("nz", 4))))
            sx = max(0.02, float(params.get("spacing_x", 0.2)))
            sz = max(0.02, float(params.get("spacing_z", 0.2)))
            if nx > 1:
                hx = (nx - 1) * 0.5
                ix = np.clip(np.round(q[:, 0] / sx + hx), 0.0, float(nx - 1)) - hx
                q[:, 0] = q[:, 0] - ix * sx
            if nz > 1:
                hz = (nz - 1) * 0.5
                iz = np.clip(np.round(q[:, 2] / sz + hz), 0.0, float(nz - 1)) - hz
                q[:, 2] = q[:, 2] - iz * sz
        elif kind == "array_radial":
            count = max(2, min(32, int(params.get("count", 6))))
            ring_r = float(params.get("radius", 0.0))
            sector = 2.0 * math.pi / count
            ang = np.arctan2(q[:, 2], q[:, 0])
            snapped = np.round(ang / sector) * sector
            ca = np.cos(-snapped)
            sa = np.sin(-snapped)
            nx_v = q[:, 0] * ca - q[:, 2] * sa - ring_r
            nz_v = q[:, 0] * sa + q[:, 2] * ca
            q[:, 0] = nx_v
            q[:, 2] = nz_v
        elif kind == "twist":
            rate = float(np.clip(params.get("rate", 1.2), -6.0, 6.0))
            ang = rate * q[:, 1]
            c, s = np.cos(ang), np.sin(ang)
            nx_v = q[:, 0] * c - q[:, 2] * s
            nz_v = q[:, 0] * s + q[:, 2] * c
            q[:, 0] = nx_v
            q[:, 2] = nz_v
            lip_scale = min(lip_scale, 0.72)
        elif kind == "bend":
            k = float(np.clip(params.get("angle", 0.4), -2.5, 2.5))
            ang = k * q[:, 0]
            c, s = np.cos(ang), np.sin(ang)
            nx_v = c * q[:, 0] - s * q[:, 1]
            ny_v = s * q[:, 0] + c * q[:, 1]
            q[:, 0] = nx_v
            q[:, 1] = ny_v
            lip_scale = min(lip_scale, 0.72)
        elif kind == "taper":
            factor = float(np.clip(params.get("factor", 0.35), -0.7, 0.7))
            scale_xz = np.clip(1.0 + factor * q[:, 1], 0.25, 2.5)
            q[:, 0] = q[:, 0] / scale_xz
            q[:, 2] = q[:, 2] / scale_xz
            lip_scale = min(lip_scale, 0.75)

    return q, lip_scale


def evaluate_surface_displacement(p: np.ndarray, modifiers: List[Dict[str, Any]]) -> np.ndarray:
    """Evaluates optional additive trigonometric/noise displacement modifiers on SDF distance."""
    disp = np.zeros((p.shape[0],), dtype=np.float32)
    for mod in modifiers:
        if mod.get("kind") != "surface_displace":
            continue
        params = mod.get("params", {}) or {}
        amp = float(np.clip(params.get("amplitude", 0.025), 0.0, 0.15))
        freq = float(np.clip(params.get("frequency", 10.0), 1.0, 40.0))
        style = str(params.get("style", "ribbed")).lower()
        if style == "knit":
            disp += amp * (np.sin(p[:, 0] * freq) * np.cos(p[:, 1] * freq * 1.4) + 0.5 * np.sin(p[:, 2] * freq))
        elif style == "bumpy":
            disp += amp * (np.sin(p[:, 0] * freq) * np.sin(p[:, 1] * freq) * np.sin(p[:, 2] * freq))
        else:  # ribbed / fluted
            disp += amp * np.sin(p[:, 0] * freq + p[:, 2] * freq * 0.5)
    return disp.astype(np.float32)
