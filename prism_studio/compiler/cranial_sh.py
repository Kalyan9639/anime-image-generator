"""
Semantic-Basis Spherical Harmonics Cranial Compiler for PrismStudio.

Translates bounded semantic morph weights (jaw taper, chin projection, cheek
fullness, brow ridge, snout length, muzzle width, cranium dome, asymmetry)
into 16 real Spherical Harmonics coefficients (L <= 3), proxy-normal centers
for clean anime shading, and exact cranial socket coordinates for ears, horns,
whiskers, and 2D UV eyes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ..math_ops.materials import hex_to_rgb_linear
from ..schema.scene_dna import ActorSpec, MorphProportions


@dataclass
class CompiledCranialShell:
    actor_id: str
    center: np.ndarray              # (3,) world-space head center
    rot_fwd: np.ndarray             # (3, 3) local->world (+Z forward, +Y up, +X left)
    rot_inv: np.ndarray             # (3, 3) world->local
    radii: np.ndarray               # (3,) ellipsoidal base radii (rx, ry, rz)
    sh_coeffs: np.ndarray           # (16,) real SH weights for L <= 3
    proxy_center_local: np.ndarray  # (3,) interior proxy normal origin
    proxy_weight: float             # [0.0, 0.9] anime proxy normal blend
    bvh_radius: float
    smooth_k: float
    skin_rgb: np.ndarray            # (3,) linear RGB
    shadow_rgb: np.ndarray          # (3,) linear RGB
    secondary_rgb: np.ndarray       # (3,) linear RGB (muzzle/belly/accent)
    coat_pattern: str
    fur_plushness: float
    eye_style: str
    eye_scale: float
    gaze_xy: Tuple[float, float]
    iris_rgb: np.ndarray
    iris2_rgb: np.ndarray
    lash_rgb: np.ndarray
    mouth_open: float
    fangs: bool
    has_cat_nose: bool
    socket_anchors_local: Dict[str, Tuple[np.ndarray, np.ndarray]]  # name -> (pos_local, dir_local)


def build_cranial_sh_coeffs(morph: MorphProportions, archetype: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Maps semantic morphology weights onto L <= 3 real Spherical Harmonics bases
    (order: 0:1, 1:y, 2:z, 3:x, 4:xy, 5:yz, 6:3z^2-1, 7:xz, 8:x^2-y^2,
     9:y(3x^2-y^2), 10:xyz, 11:y(5z^2-1), 12:z(5z^2-3), 13:x(5z^2-1), 14:z(x^2-y^2), 15:x(x^2-3y^2)).
    Returns (base_radii (3,), sh_coeffs (16,)).
    """
    c = np.zeros(16, dtype=np.float32)

    if archetype == "quadruped":
        # Soft round feline/canine skull with plump cheeks and subtle forward muzzle
        radii = np.array(
            [
                0.185 * morph.head_scale,
                0.160 * morph.head_scale,
                0.178 * morph.head_scale * (1.0 + 0.25 * morph.snout_length),
            ],
            dtype=np.float32,
        )
        # Wider lower cheeks (negative y, x^2-y^2 lobe) and gentle snout protrusion along +z
        c[1] = -0.06 * morph.cheek_fullness
        c[2] = 0.08 + 0.22 * morph.snout_length
        c[5] = -0.14 * (0.5 + morph.cheek_fullness * 0.5)   # lower front cheeks
        c[6] = 0.06 * morph.snout_length                    # forward-back elongation
        c[8] = 0.14 * morph.cheek_fullness                  # lateral cheek puff
        c[11] = -0.11 * (0.4 + morph.snout_length * 0.6)    # muzzle fullness below eyes
        c[12] = 0.09 * morph.snout_length                   # pointed nose bridge along +z
        c[14] = -0.05 * morph.muzzle_width
    else:
        # Anime / stylized humanoid cranium: rounded crown + tapered jaw/chin + gentle brow
        radii = np.array(
            [
                0.118 * morph.head_scale,
                0.138 * morph.head_scale,
                0.124 * morph.head_scale,
            ],
            dtype=np.float32,
        )
        c[1] = 0.06 * morph.cranium_dome - 0.04 * morph.jaw_taper
        c[2] = 0.06 * morph.chin_projection
        c[5] = 0.05 * morph.brow_ridge - 0.08 * morph.jaw_taper
        c[8] = 0.09 * morph.cheek_fullness - 0.06 * morph.jaw_taper
        c[9] = 0.07 * morph.jaw_taper                       # narrows lower chin while widening temples
        c[11] = -0.08 * morph.chin_projection               # crisp pointed anime chin forward-down
        c[12] = 0.05 * morph.snout_length                   # subtle nose projection

    # Asymmetric smirk / head expression term (x*z / x)
    c[3] = 0.03 * morph.asymmetry_smirk
    c[7] = 0.04 * morph.asymmetry_smirk
    return radii, np.clip(c, -0.36, 0.36).astype(np.float32)


def compile_cranial_shell(
    actor: ActorSpec,
    head_center_world: np.ndarray,
    head_rot_fwd: np.ndarray,
    *,
    style_mode: str = "anime_npr",
) -> CompiledCranialShell:
    """Compiles an ActorSpec's head morphology, 2D UV eye settings, and socket anchors."""
    m = actor.proportions
    radii, sh = build_cranial_sh_coeffs(m, actor.archetype)
    s = float(actor.scale)
    radii = (radii * s).astype(np.float32)

    # Gaze offset in eye-patch UV coordinates
    gaze_map = {
        "camera": (0.0, 0.0),
        "left": (-0.16, 0.02),
        "right": (0.16, 0.02),
        "up": (0.0, 0.16),
        "down": (0.0, -0.14),
        "support": (0.04, -0.12),
    }
    gaze_xy = gaze_map.get(actor.expression.gaze_target, (0.0, 0.0))

    max_r = float(np.max(radii)) * 1.45
    eye_style = actor.expression.eye_style
    if actor.archetype == "quadruped" and eye_style == "anime_round":
        eye_style = "feline_slit"

    # Proxy normal center shifted slightly inward along -Z so front face catches clean anime light bands
    proxy_local = np.array([0.0, -0.08 * radii[1], -0.18 * radii[2]], dtype=np.float32)
    proxy_w = 0.72 if style_mode == "anime_npr" else 0.28

    # Local sockets on the skull for ears, horns, hair crown, muzzle/nose, jaw
    rx, ry, rz = float(radii[0]), float(radii[1]), float(radii[2])
    tilt_rad = math.radians(m.ear_tilt_deg)
    if actor.archetype == "quadruped":
        ear_l = np.array([rx * 0.62, ry * 0.72, -rz * 0.08], dtype=np.float32)
        ear_r = np.array([-rx * 0.62, ry * 0.72, -rz * 0.08], dtype=np.float32)
        dir_l = np.array([math.sin(tilt_rad) * 0.72 + 0.32, math.cos(tilt_rad), 0.12], dtype=np.float32)
        dir_r = np.array([-math.sin(tilt_rad) * 0.72 - 0.32, math.cos(tilt_rad), 0.12], dtype=np.float32)
        muzzle_tip = np.array([0.0, -ry * 0.16, rz * (1.06 + 0.18 * m.snout_length)], dtype=np.float32)
    else:
        # Humanoid / Elf side-skull ears
        ear_l = np.array([rx * 0.94, ry * 0.05, -rz * 0.10], dtype=np.float32)
        ear_r = np.array([-rx * 0.94, ry * 0.05, -rz * 0.10], dtype=np.float32)
        dir_l = np.array([0.85, math.sin(tilt_rad) * 0.65 + 0.25, -0.35], dtype=np.float32)
        dir_r = np.array([-0.85, math.sin(tilt_rad) * 0.65 + 0.25, -0.35], dtype=np.float32)
        muzzle_tip = np.array([0.0, -ry * 0.05, rz * 1.02], dtype=np.float32)

    dir_l /= max(1e-6, float(np.linalg.norm(dir_l)))
    dir_r /= max(1e-6, float(np.linalg.norm(dir_r)))

    sockets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "ear_left": (ear_l, dir_l),
        "ear_right": (ear_r, dir_r),
        "horn_left": (np.array([rx * 0.45, ry * 0.85, rz * 0.18], dtype=np.float32), np.array([0.35, 0.88, 0.25], dtype=np.float32)),
        "horn_right": (np.array([-rx * 0.45, ry * 0.85, rz * 0.18], dtype=np.float32), np.array([-0.35, 0.88, 0.25], dtype=np.float32)),
        "crown": (np.array([0.0, ry * 0.95, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        "back_crown": (np.array([0.0, ry * 0.55, -rz * 0.92], dtype=np.float32), np.array([0.0, 0.2, -1.0], dtype=np.float32)),
        "muzzle_tip": (muzzle_tip, np.array([0.0, 0.0, 1.0], dtype=np.float32)),
    }

    return CompiledCranialShell(
        actor_id=actor.id,
        center=head_center_world.astype(np.float32),
        rot_fwd=head_rot_fwd.astype(np.float32),
        rot_inv=head_rot_fwd.T.astype(np.float32),
        radii=radii,
        sh_coeffs=sh,
        proxy_center_local=proxy_local,
        proxy_weight=proxy_w,
        bvh_radius=max_r + 0.05,
        smooth_k=0.045 * s,
        skin_rgb=hex_to_rgb_linear(actor.skin_color),
        shadow_rgb=hex_to_rgb_linear(actor.skin_shadow_color),
        secondary_rgb=hex_to_rgb_linear(actor.secondary_fur_color),
        coat_pattern=actor.coat_pattern,
        fur_plushness=float(m.fur_plushness),
        eye_style=eye_style,
        eye_scale=float(actor.expression.eye_scale),
        gaze_xy=gaze_xy,
        iris_rgb=hex_to_rgb_linear(actor.expression.iris_color),
        iris2_rgb=hex_to_rgb_linear(actor.expression.iris_secondary),
        lash_rgb=hex_to_rgb_linear(actor.expression.lash_color),
        mouth_open=float(actor.expression.mouth_open),
        fangs=bool(actor.expression.fangs),
        has_cat_nose=(actor.archetype == "quadruped"),
        socket_anchors_local=sockets,
    )
