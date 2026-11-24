"""
Procedural Hair-Clump & Scalp Cap Generator for PrismStudio.

Generates a sculpted cranial hair cap so scalps never look bald between locks,
short side-swept bangs that leave eyes/faces clearly uncovered, and flowing
back/side ribbons via damped catenary curves.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from ..math_ops.materials import hex_to_rgb_linear
from ..math_ops.primitives import compute_bishop_frames
from ..schema.scene_dna import ActorSpec
from .cranial_sh import CompiledCranialShell
from .kinematics import CompiledSplineBone


def compile_hair_clumps(
    actor: ActorSpec,
    cranial: CompiledCranialShell,
) -> List[CompiledSplineBone]:
    """Generates cranial hair cap + sculpted hair-clump ribbons leaving the face clear."""
    hair = actor.hair_or_mane
    if hair.style == "none" or hair.clump_count <= 0:
        return []

    s = float(actor.scale)
    rx, ry, rz = cranial.radii
    c_w = cranial.center
    rot = cranial.rot_fwd
    col1 = hex_to_rgb_linear(hair.color)
    col2 = hex_to_rgb_linear(hair.secondary_color)
    wind = np.array(hair.wind, dtype=np.float32)
    grav = np.array([0.0, -0.58, 0.0], dtype=np.float32)

    out: List[CompiledSplineBone] = []
    n_total = max(4, min(28, int(hair.clump_count)))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    length = float(hair.length) * s

    def _add_hair_spline(
        name: str,
        pts_w: List[tuple],
        ra: List[float],
        rb: List[float],
        smooth_k: float = 0.025,
    ) -> None:
        st = compute_bishop_frames(pts_w, radii_a=ra, radii_b=rb, squareness_n=[1.7, 1.5], n_stations=7, cap_roundness=0.95)
        out.append(
            CompiledSplineBone(
                name=name,
                actor_id=actor.id,
                stations=st,
                smooth_k=smooth_k * s,
                csg_op=0,
                bone_role=1,
                color_rgb=col1,
                secondary_rgb=col2,
                shader="anisotropic_hair",
                metallic=0.0,
                roughness=0.30,
                emissive=0.0,
            )
        )

    def _make_strand(name: str, p0_loc: np.ndarray, v0_loc: np.ndarray, len_scale: float, width: float, thick: float) -> None:
        p0_w = c_w + rot @ p0_loc
        v0_w = rot @ v0_loc
        L = length * len_scale
        pts = []
        for t in (0.0, 0.35, 0.72, 1.0):
            disp = v0_w * (t * L) + 0.5 * (grav + wind) * (t * t * L * 1.1)
            pts.append(tuple((p0_w + disp).tolist()))
        _add_hair_spline(
            name,
            pts,
            ra=[width, width * 0.90, width * 0.56, width * 0.16],
            rb=[thick, thick * 0.88, thick * 0.52, thick * 0.16],
            smooth_k=0.026,
        )

    if hair.style == "braided_beard":
        for i in range(min(n_total, 9)):
            u = (i / max(1, min(n_total, 9) - 1)) - 0.5
            p0 = np.array([u * rx * 0.95, -ry * 0.72, rz * 0.70], dtype=np.float32)
            v0 = np.array([u * 0.18, -0.88, 0.22], dtype=np.float32)
            _make_strand(f"beard_{i}", p0, v0, 0.65, 0.026 * s, 0.018 * s)
        return out

    # 1. Sculpted Cranial Hair Cap (wraps symmetrically over crown & back of skull, never floating offset!)
    cap_pts = [
        tuple((c_w + rot @ np.array([0.0, ry * 0.72, rz * 0.12], dtype=np.float32)).tolist()),
        tuple((c_w + rot @ np.array([0.0, ry * 0.65, -rz * 0.22], dtype=np.float32)).tolist()),
        tuple((c_w + rot @ np.array([0.0, ry * 0.16, -rz * 0.45], dtype=np.float32)).tolist()),
    ]
    _add_hair_spline(
        "hair_scalp_cap",
        cap_pts,
        ra=[rx * 0.96, rx * 1.03, rx * 0.96],
        rb=[rz * 0.60, rz * 0.74, rz * 0.68],
        smooth_k=0.038,
    )

    # 2. Curved Above-Brow Forehead Bangs & Cheek Side-Locks (keeps eyes & eyebrows open)
    for b_idx, b_frac in enumerate((-0.42, 0.0, 0.42)):
        p0_f = np.array([b_frac * rx * 0.75, ry * 0.94, rz * 0.48], dtype=np.float32)
        p1_f = np.array([b_frac * rx * 0.88, ry * 0.72, rz * 0.88], dtype=np.float32)
        p2_f = np.array([b_frac * rx * 0.94 + (0.04 * s if b_idx == 1 else 0.0), ry * 0.34, rz * 0.86], dtype=np.float32)
        _add_hair_spline(
            f"bang_center_{b_idx}",
            [tuple((c_w + rot @ p0_f).tolist()), tuple((c_w + rot @ p1_f).tolist()), tuple((c_w + rot @ p2_f).tolist())],
            ra=[0.052 * s, 0.048 * s, 0.018 * s],
            rb=[0.024 * s, 0.022 * s, 0.010 * s],
            smooth_k=0.036,
        )

    for sx, tag in ((1.0, "L"), (-1.0, "R")):
        p0_s = np.array([sx * rx * 0.98, ry * 0.56, rz * 0.42], dtype=np.float32)
        v0_s = np.array([sx * 0.22, -0.82, 0.10], dtype=np.float32)
        _make_strand(f"side_frame_{tag}", p0_s, v0_s, 0.50, 0.038 * s, 0.022 * s)

    # 3. Back / Crown Body Hair (visible silhouette sweep without occluding front face)
    n_body = max(4, n_total - 5)
    if hair.style == "high_ponytail":
        tie_loc = np.array([0.0, ry * 0.58, -rz * 0.95], dtype=np.float32)
        n_tail = min(10, max(5, n_body))
        for i in range(n_tail):
            a = i * golden_angle
            r_off = 0.028 * s * math.sqrt((i + 0.5) / n_tail)
            p0 = tie_loc + np.array([math.cos(a) * r_off, math.sin(a) * r_off, 0.0], dtype=np.float32)
            # Sweeps sideways & back cleanly so ponytail silhouette pops beside head in 3/4 view!
            v0 = np.array([0.36 + math.cos(a) * 0.22, 0.24 + math.sin(a) * 0.14, -0.78], dtype=np.float32)
            _make_strand(f"ponytail_{i}", p0, v0, 1.05, 0.048 * s, 0.024 * s)
    elif hair.style == "twin_tails":
        for sx, lbl in ((1.0, "L"), (-1.0, "R")):
            tie = np.array([sx * rx * 1.05, ry * 0.48, -rz * 0.45], dtype=np.float32)
            for i in range(max(3, n_body // 3)):
                v0 = np.array([sx * (0.48 + 0.14 * (i % 2)), 0.08, -0.68], dtype=np.float32)
                _make_strand(f"twintail_{lbl}_{i}", tie, v0, 0.88, 0.042 * s, 0.020 * s)
    elif hair.style == "short_spiky":
        for i in range(n_body):
            ang = math.pi * (0.55 + 0.9 * (i / max(1, n_body - 1)))
            dir_v = np.array([math.sin(ang) * 0.85, 0.45, math.cos(ang)], dtype=np.float32)
            p0 = dir_v * np.array([rx * 0.95, ry * 0.95, rz * 0.95], dtype=np.float32)
            _make_strand(f"spike_{i}", p0, dir_v + np.array([0.0, 0.22, -0.25], dtype=np.float32), 0.34, 0.036 * s, 0.020 * s)
    else:
        len_f = 0.46 if hair.style == "bob_cut" else 0.92
        n_back = min(9, max(5, n_body))
        for i in range(n_back):
            ang = math.pi * (0.56 + 0.88 * (i / max(1, n_back - 1)))
            sx_v = math.sin(ang)
            sz_v = math.cos(ang)
            p0 = np.array([sx_v * rx * 0.98, ry * 0.58, sz_v * rz * 0.95], dtype=np.float32)
            v0 = np.array([sx_v * 0.42, -0.62, sz_v * 0.55], dtype=np.float32)
            _make_strand(f"back_lock_{i}", p0, v0, len_f, 0.048 * s, 0.024 * s)

    return out
