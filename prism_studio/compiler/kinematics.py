"""
Analytical 2-Bone Inverse Kinematics & Compositional Bone-Graph DAG Compiler.

Builds anatomical rigs for humanoid, quadruped, avian, serpentine, and
open-ended creative creatures (elves, dwarfs, 4-armed mystics, wheel-legged
XLR8-style speedsters, winged/tailed hybrids) purely by composing splines,
omitting bones, adding extra IK chains, and attaching mechanical socket gear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..math_ops.materials import hex_to_rgb_linear
from ..math_ops.primitives import BishopSplineStations, compute_bishop_frames, rotation_matrix_xyz
from ..schema.scene_dna import ActorSpec, LimbChainSpec, MechanicalSocketSpec
from .cranial_sh import CompiledCranialShell, compile_cranial_shell


@dataclass
class CompiledSplineBone:
    name: str
    actor_id: str
    stations: BishopSplineStations
    smooth_k: float
    csg_op: int                  # 0=smin, 1=union, 2=smooth_sub
    bone_role: int               # 0=body, 1=hair, 2=prop, 3=ivy, 4=garment_skirt
    color_rgb: np.ndarray        # (3,) linear
    secondary_rgb: np.ndarray    # (3,) linear
    shader: str
    metallic: float
    roughness: float
    emissive: float
    garment_id: int = -1
    joint_bend_rad: float = 0.0
    t_uv_range: Tuple[float, float] = (0.0, 1.0)


@dataclass
class CompiledSolidPrimitive:
    name: str
    kind: int                    # 0=rounded_box, 1=staggered_masonry, 2=torus_wheel, 3=superquadric
    center: np.ndarray           # (3,)
    rot_fwd: np.ndarray          # (3, 3)
    rot_inv: np.ndarray          # (3, 3)
    half_extents: np.ndarray     # (3,)
    param_a: float               # bevel | major_r | e1
    param_b: float               # spoke_count | minor_r | e2
    repeat_bounds: Tuple[int, int, int] = (0, 0, 0)
    repeat_pitch: Tuple[float, float, float] = (0.4, 0.2, 0.4)
    csg_op: int = 1              # 0=smin, 1=union
    smooth_k: float = 0.02
    color_rgb: np.ndarray = field(default_factory=lambda: np.array([0.6, 0.6, 0.65], dtype=np.float32))
    secondary_rgb: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.3, 0.35], dtype=np.float32))
    accent_rgb: np.ndarray = field(default_factory=lambda: np.array([0.9, 0.7, 0.2], dtype=np.float32))
    shader: str = "solid_cel"
    metallic: float = 0.1
    roughness: float = 0.5
    emissive: float = 0.0
    bvh_radius: float = 1.0


@dataclass
class CompiledActorRig:
    actor: ActorSpec
    origin_world: np.ndarray
    yaw_rad: float
    cranial_shell: CompiledCranialShell
    bones: List[CompiledSplineBone]
    solid_parts: List[CompiledSolidPrimitive]
    sockets_world: Dict[str, Tuple[np.ndarray, np.ndarray]]  # socket_name -> (pos_world, dir_world)
    bvh_center: np.ndarray
    bvh_radius: float


def solve_two_bone_ik(
    root: np.ndarray,
    target: np.ndarray,
    l1: float,
    l2: float,
    pole_dir: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Exact Law-of-Cosines 2-Bone Analytical IK solver.
    Clamps reach strictly inside [|L1-L2|+0.01, 0.995*(L1+L2)] so limbs never detach.
    Returns (mid_joint_pos, clamped_end_pos, bend_angle_rad).
    """
    root_v = np.asarray(root, dtype=np.float32)
    tgt_v = np.asarray(target, dtype=np.float32)
    delta = tgt_v - root_v
    raw_d = float(np.linalg.norm(delta))

    min_d = abs(l1 - l2) + 0.012
    max_d = 0.995 * (l1 + l2)
    d = max(min_d, min(max_d, raw_d))

    if raw_d < 1e-6:
        fwd = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    else:
        fwd = (delta / raw_d).astype(np.float32)
    end_pos = root_v + fwd * d

    # Law of cosines at root shoulder/hip and mid elbow/knee
    cos_alpha = float(np.clip((l1 * l1 + d * d - l2 * l2) / max(1e-6, 2.0 * l1 * d), -1.0, 1.0))
    sin_alpha = math.sqrt(max(0.0, 1.0 - cos_alpha * cos_alpha))

    cos_elbow = float(np.clip((l1 * l1 + l2 * l2 - d * d) / max(1e-6, 2.0 * l1 * l2), -1.0, 1.0))
    bend_angle = math.pi - math.acos(cos_elbow)

    # Orthonormalize pole vector against root->target axis
    pole = np.asarray(pole_dir, dtype=np.float32)
    ortho = pole - float(np.dot(pole, fwd)) * fwd
    o_norm = float(np.linalg.norm(ortho))
    if o_norm < 1e-5:
        fallback = np.array([0.0, 0.0, -1.0], dtype=np.float32) if abs(float(fwd[2])) < 0.85 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ortho = fallback - float(np.dot(fallback, fwd)) * fwd
        o_norm = max(1e-6, float(np.linalg.norm(ortho)))
    ortho /= o_norm

    mid_joint = root_v + fwd * (l1 * cos_alpha) + ortho * (l1 * sin_alpha)
    return mid_joint.astype(np.float32), end_pos.astype(np.float32), float(bend_angle)


def _world_transform_pts(pts_local: List[Tuple[float, float, float]], origin: np.ndarray, yaw_mat: np.ndarray) -> List[Tuple[float, float, float]]:
    arr = np.asarray(pts_local, dtype=np.float32) @ yaw_mat.T + origin[None, :]
    return [tuple(row.tolist()) for row in arr]


def _build_spline_bone(
    name: str,
    actor: ActorSpec,
    pts_world: List[Tuple[float, float, float]],
    ra: List[float],
    rb: Optional[List[float]] = None,
    sq: List[float] = (2.0, 2.0),
    *,
    smooth_k: float = 0.045,
    csg_op: int = 0,
    bone_role: int = 0,
    cap_roundness: float = 1.0,
    n_stations: int = 8,
    color_rgb: Optional[np.ndarray] = None,
    joint_bend_rad: float = 0.0,
) -> CompiledSplineBone:
    st = compute_bishop_frames(pts_world, ra, rb, sq, n_stations=n_stations, cap_roundness=cap_roundness)
    s = float(actor.scale)
    skin = color_rgb if color_rgb is not None else hex_to_rgb_linear(actor.skin_color)
    sec = hex_to_rgb_linear(actor.secondary_fur_color)
    shader = "tabby_stripes" if actor.coat_pattern == "tabby_stripes" else ("plush_fur" if actor.proportions.fur_plushness > 0.25 else "skin_sss")
    return CompiledSplineBone(
        name=name,
        actor_id=actor.id,
        stations=st,
        smooth_k=float(smooth_k * s),
        csg_op=csg_op,
        bone_role=bone_role,
        color_rgb=skin.astype(np.float32),
        secondary_rgb=sec.astype(np.float32),
        shader=shader,
        metallic=0.0,
        roughness=0.72 if actor.proportions.fur_plushness > 0.2 else 0.55,
        emissive=0.0,
        joint_bend_rad=joint_bend_rad,
    )


def _compile_quadruped_cat_rig(actor: ActorSpec, origin: np.ndarray, yaw_mat: np.ndarray, style_mode: str) -> CompiledActorRig:
    """
    Compiles a smooth, cohesive quadruped rig (cat, fox, wolf, lion) with
    tucked haunches or standing stance, forelegs resting on surface, arched S-tail,
    soft cheek tufts, pointed ears, and SH head.
    """
    m = actor.proportions
    s = float(actor.scale)
    tw = float(m.torso_width)
    ll = float(m.limb_length)
    omitted = set(actor.omit_bones)
    bones: List[CompiledSplineBone] = []
    solids: List[CompiledSolidPrimitive] = []
    sockets_world: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    is_sitting = actor.posture in ("sitting_haunches", "sitting_edge", "crouching")
    is_edge = actor.posture == "sitting_edge"

    def wpt(x: float, y: float, z: float) -> np.ndarray:
        return (yaw_mat @ np.array([x * s, y * s, z * s], dtype=np.float32)) + origin

    if is_sitting:
        # Plump sitting feline spine: haunch base -> round belly -> narrow chest -> neck -> head
        spine_pts = [
            tuple(wpt(0.0, 0.13, -0.11).tolist()),
            tuple(wpt(0.0, 0.26, -0.05).tolist()),
            tuple(wpt(0.0, 0.41, 0.04).tolist()),
            tuple(wpt(0.0, 0.51, 0.09).tolist()),
        ]
        bones.append(
            _build_spline_bone(
                "torso_spine",
                actor,
                spine_pts,
                ra=[0.165 * tw * s, 0.175 * tw * s, 0.140 * tw * s, 0.095 * s],
                rb=[0.175 * tw * s, 0.185 * tw * s, 0.145 * tw * s, 0.095 * s],
                sq=[2.1, 2.0, 2.0],
                smooth_k=0.065,
                n_stations=10,
            )
        )
        # Neck smoothly linking chest to skull
        head_c = wpt(0.0, 0.62, 0.12)
        bones.append(
            _build_spline_bone(
                "neck",
                actor,
                [tuple(wpt(0.0, 0.48, 0.08).tolist()), tuple(head_c.tolist())],
                ra=[0.092 * s, 0.085 * s],
                smooth_k=0.065,
                n_stations=5,
            )
        )
        # Folded sitting haunches resting cleanly on y = 0.045..0.06
        for side_name, sx in (("left", 1.0), ("right", -1.0)):
            if f"{side_name}_leg" in omitted:
                continue
            haunch_pts = [
                tuple(wpt(sx * 0.12 * tw, 0.18, -0.09).tolist()),
                tuple(wpt(sx * 0.175 * tw, 0.12, -0.02).tolist()),
                tuple(wpt(sx * 0.145 * tw, 0.048, 0.10).tolist()),
            ]
            bones.append(
                _build_spline_bone(
                    f"{side_name}_haunch",
                    actor,
                    haunch_pts,
                    ra=[0.095 * s, 0.082 * s, 0.050 * s],
                    rb=[0.115 * s, 0.090 * s, 0.052 * s],
                    smooth_k=0.055,
                    n_stations=8,
                )
            )
            # Slender front foreleg + soft paw resting at y=0.036
            sh_p = wpt(sx * 0.092 * tw, 0.40, 0.08)
            paw_p = wpt(sx * 0.072 * tw, 0.038, 0.175)
            el_p, paw_end, bend_r = solve_two_bone_ik(
                sh_p,
                paw_p,
                0.20 * ll * s,
                0.21 * ll * s,
                yaw_mat @ np.array([0.0, 0.0, -1.0], dtype=np.float32),
            )
            bones.append(
                _build_spline_bone(
                    f"{side_name}_foreleg",
                    actor,
                    [tuple(sh_p.tolist()), tuple(el_p.tolist()), tuple(paw_end.tolist())],
                    ra=[0.058 * s, 0.046 * s, 0.042 * s],
                    smooth_k=0.045,
                    n_stations=8,
                    joint_bend_rad=bend_r,
                )
            )
            # Plush front paw toe pad
            paw_tip = wpt(sx * 0.072 * tw, 0.036, 0.225)
            bones.append(
                _build_spline_bone(
                    f"{side_name}_paw",
                    actor,
                    [tuple(paw_end.tolist()), tuple(paw_tip.tolist())],
                    ra=[0.044 * s, 0.046 * s],
                    rb=[0.036 * s, 0.034 * s],
                    smooth_k=0.03,
                    n_stations=5,
                )
            )
            sockets_world[f"{side_name}_paw"] = (paw_tip, yaw_mat @ np.array([0.0, 0.0, 1.0], dtype=np.float32))
    else:
        # Standing / walking quadruped body
        spine_pts = [
            tuple(wpt(0.0, 0.34, -0.24).tolist()),
            tuple(wpt(0.0, 0.37, 0.0).tolist()),
            tuple(wpt(0.0, 0.39, 0.24).tolist()),
        ]
        bones.append(
            _build_spline_bone(
                "torso_spine",
                actor,
                spine_pts,
                ra=[0.13 * tw * s, 0.145 * tw * s, 0.13 * tw * s],
                smooth_k=0.06,
                n_stations=9,
            )
        )
        head_c = wpt(0.0, 0.56, 0.34)
        bones.append(
            _build_spline_bone(
                "neck",
                actor,
                [tuple(wpt(0.0, 0.40, 0.22).tolist()), tuple(head_c.tolist())],
                ra=[0.095 * s, 0.08 * s],
                smooth_k=0.055,
                n_stations=5,
            )
        )
        for side_name, sx in (("left", 1.0), ("right", -1.0)):
            for limb_tag, z_hip, z_foot in (("fore", 0.18, 0.20), ("hind", -0.20, -0.22)):
                if f"{side_name}_{limb_tag}" in omitted:
                    continue
                hip_p = wpt(sx * 0.09 * tw, 0.34, z_hip)
                ft_p = wpt(sx * 0.09 * tw, 0.038, z_foot)
                mid_p, ft_end, bend_r = solve_two_bone_ik(
                    hip_p,
                    ft_p,
                    0.18 * ll * s,
                    0.18 * ll * s,
                    yaw_mat @ np.array([0.0, 0.0, -1.0 if limb_tag == "fore" else 1.0], dtype=np.float32),
                )
                bones.append(
                    _build_spline_bone(
                        f"{side_name}_{limb_tag}",
                        actor,
                        [tuple(hip_p.tolist()), tuple(mid_p.tolist()), tuple(ft_end.tolist())],
                        ra=[0.056 * s, 0.044 * s, 0.042 * s],
                        smooth_k=0.04,
                        n_stations=7,
                        joint_bend_rad=bend_r,
                    )
                )

    # Head pitch slightly tilted up/forward toward camera
    head_rot = yaw_mat @ rotation_matrix_xyz(-6.0, 0.0, 0.0)
    cranial = compile_cranial_shell(actor, head_c, head_rot, style_mode=style_mode)
    sockets_world["head"] = (head_c, head_rot @ np.array([0.0, 1.0, 0.0], dtype=np.float32))
    sockets_world["pelvis"] = (wpt(0.0, 0.14, -0.10), yaw_mat @ np.array([0.0, 0.0, -1.0], dtype=np.float32))
    sockets_world["upper_torso"] = (wpt(0.0, 0.41, 0.04), yaw_mat @ np.array([0.0, 1.0, 0.0], dtype=np.float32))

    # Pointed cat ears blended smoothly at skull crown sockets
    ear_len = max(0.08, m.ear_length) * s
    for side_key in ("ear_left", "ear_right"):
        p_loc, d_loc = cranial.socket_anchors_local[side_key]
        p0_w = head_c + head_rot @ (p_loc * 0.85)
        p1_w = p0_w + head_rot @ (d_loc * (ear_len * 0.55))
        p2_w = p0_w + head_rot @ (d_loc * ear_len)
        bones.append(
            _build_spline_bone(
                side_key,
                actor,
                [tuple(p0_w.tolist()), tuple(p1_w.tolist()), tuple(p2_w.tolist())],
                ra=[0.072 * s, 0.045 * s, 0.012 * s],
                rb=[0.042 * s, 0.026 * s, 0.010 * s],
                sq=[1.35, 1.35],
                smooth_k=0.04,
                n_stations=6,
            )
        )

    # Graceful S-curved cat tail (curls up by haunch or drapes down over wall edge!)
    t_len = max(0.35, m.tail_length if m.tail_length > 0.05 else 0.72) * s
    curl = float(m.tail_curl)
    if is_edge:
        # Drapes naturally over front/side wall ledge!
        tail_ctrl = [
            tuple(wpt(0.0, 0.10, -0.14).tolist()),
            tuple(wpt(0.24, 0.08, -0.02).tolist()),
            tuple(wpt(0.34, -0.18 * t_len, 0.18).tolist()),
            tuple(wpt(0.28 + 0.08 * curl, -0.46 * t_len, 0.28).tolist()),
        ]
    else:
        tail_ctrl = [
            tuple(wpt(0.0, 0.11, -0.14).tolist()),
            tuple(wpt(0.22, 0.10, -0.22).tolist()),
            tuple(wpt(0.32, 0.34 * t_len, -0.10).tolist()),
            tuple(wpt(0.24 + 0.06 * curl, 0.56 * t_len, 0.04).tolist()),
        ]
    bones.append(
        _build_spline_bone(
            "tail",
            actor,
            tail_ctrl,
            ra=[0.044 * s, 0.040 * s, 0.036 * s, 0.026 * s],
            smooth_k=0.045,
            n_stations=12,
        )
    )

    # Smooth cat fur body (omit harsh separate chest pill)
    if m.fur_plushness > 0.60:
        pass

    return CompiledActorRig(
        actor=actor,
        origin_world=origin,
        yaw_rad=math.radians(actor.facing_deg),
        cranial_shell=cranial,
        bones=bones,
        solid_parts=solids,
        sockets_world=sockets_world,
        bvh_center=wpt(0.0, 0.36, 0.04),
        bvh_radius=0.95 * s,
    )


def _resolve_humanoid_hand_target(
    target_name: str,
    side_sign: float,
    sh_pos: np.ndarray,
    head_pos: np.ndarray,
    origin: np.ndarray,
    yaw_mat: np.ndarray,
    s: float,
    posture: str,
) -> np.ndarray:
    def wpt(x: float, y: float, z: float) -> np.ndarray:
        return (yaw_mat @ np.array([x * s, y * s, z * s], dtype=np.float32)) + origin

    if target_name == "near_mouth":
        # Offset slightly right/below chin so microphone never hides eyes or nose!
        return head_pos + yaw_mat @ np.array([side_sign * 0.14 * s, -0.17 * s, 0.18 * s], dtype=np.float32)
    if target_name == "forward_reach":
        return wpt(side_sign * 0.24, 1.06, 0.38)
    if target_name == "piano_keys":
        return wpt(side_sign * 0.20, 0.84, 0.36)
    if target_name == "held_staff":
        return wpt(side_sign * 0.34, 1.08, 0.22)
    if target_name == "raised_wave":
        return wpt(side_sign * 0.34, 1.50, 0.14)
    if target_name == "skating_back" or posture == "skating_lean":
        return wpt(side_sign * 0.28, 0.88, -0.28)
    if target_name == "hip_side":
        return wpt(side_sign * 0.24, 0.82, 0.06)
    # Default relaxed natural hang or singing gesture
    if posture == "singing_pose" and side_sign > 0:
        return head_pos + yaw_mat @ np.array([0.08 * s, -0.13 * s, 0.19 * s], dtype=np.float32)
    return wpt(side_sign * 0.25, 0.78, 0.08)


def _attach_extra_chains_and_mech(
    rig: CompiledActorRig,
    actor: ActorSpec,
    origin: np.ndarray,
    yaw_mat: np.ndarray,
) -> None:
    """Compiles arbitrary extra IK limb chains (e.g. 4 arms, wings, tentacles) & mechanical sockets (wheels, visors, staffs)."""
    s = float(actor.scale)
    bones = rig.bones
    solids = rig.solid_parts

    # 1. Extra articulated limb chains (4 arms, tentacles, wings, tails)
    for chain in actor.extra_limb_chains:
        sides = [("left", 1.0), ("right", -1.0)] if chain.side == "pair" else (
            [("left", 1.0)] if chain.side == "left" else ([("right", -1.0)] if chain.side == "right" else [("center", 0.0)])
        )
        base_pos, _ = rig.sockets_world.get(chain.parent_socket, rig.sockets_world["mid_torso"])
        col = hex_to_rgb_linear(chain.color) if chain.color else None

        for s_lbl, sx in sides:
            sign = 1.0 if s_lbl == "center" else sx
            off_loc = np.array([sign * abs(chain.offset_local[0]) if chain.side == "pair" else chain.offset_local[0], chain.offset_local[1], chain.offset_local[2]], dtype=np.float32) * s
            root_w = base_pos + yaw_mat @ off_loc

            tgt_loc = np.array([sign * abs(chain.ik_target[0]) if chain.side == "pair" else chain.ik_target[0], chain.ik_target[1], chain.ik_target[2]], dtype=np.float32) * s
            tgt_w = base_pos + yaw_mat @ tgt_loc
            pole_w = yaw_mat @ np.array([sign * chain.pole_dir[0], chain.pole_dir[1], chain.pole_dir[2]], dtype=np.float32)

            mid_w, end_w, bend_r = solve_two_bone_ik(root_w, tgt_w, chain.l1 * s, chain.l2 * s, pole_w)

            if chain.kind == "tentacle":
                ctrl = [
                    tuple(root_w.tolist()),
                    tuple((0.5 * (root_w + mid_w) + yaw_mat @ np.array([sign * 0.08 * s, 0.08 * s, 0.05 * s], dtype=np.float32)).tolist()),
                    tuple(mid_w.tolist()),
                    tuple(end_w.tolist()),
                ]
                bones.append(
                    _build_spline_bone(
                        f"{chain.id}_{s_lbl}",
                        actor,
                        ctrl,
                        ra=[chain.radius_base * s, chain.radius_base * 0.8 * s, chain.radius_tip * 1.1 * s, chain.radius_tip * 0.45 * s],
                        smooth_k=0.05,
                        n_stations=12,
                        color_rgb=col,
                    )
                )
            elif chain.kind == "wing":
                tip_w = end_w + yaw_mat @ np.array([sign * 0.35 * s, 0.25 * s, -0.18 * s], dtype=np.float32)
                bones.append(
                    _build_spline_bone(
                        f"{chain.id}_{s_lbl}_wing",
                        actor,
                        [tuple(root_w.tolist()), tuple(mid_w.tolist()), tuple(tip_w.tolist())],
                        ra=[0.06 * s, 0.22 * s, 0.04 * s],
                        rb=[0.025 * s, 0.018 * s, 0.012 * s],
                        sq=[1.6, 1.6],
                        smooth_k=0.04,
                        n_stations=10,
                        color_rgb=col,
                    )
                )
            else:
                # Full 2-bone IK arm (upper arm + forearm + optional hand)
                bones.append(
                    _build_spline_bone(
                        f"{chain.id}_{s_lbl}_upper",
                        actor,
                        [tuple(root_w.tolist()), tuple(mid_w.tolist())],
                        ra=[chain.radius_base * s, chain.radius_base * 0.82 * s],
                        smooth_k=0.048,
                        n_stations=6,
                        color_rgb=col,
                        joint_bend_rad=bend_r,
                    )
                )
                bones.append(
                    _build_spline_bone(
                        f"{chain.id}_{s_lbl}_fore",
                        actor,
                        [tuple(mid_w.tolist()), tuple(end_w.tolist())],
                        ra=[chain.radius_base * 0.80 * s, chain.radius_tip * s],
                        smooth_k=0.04,
                        n_stations=6,
                        color_rgb=col,
                        joint_bend_rad=bend_r,
                    )
                )
                if chain.has_hand:
                    h_tip = end_w + (end_w - mid_w) / max(1e-5, float(np.linalg.norm(end_w - mid_w))) * (0.095 * s)
                    bones.append(
                        _build_spline_bone(
                            f"{chain.id}_{s_lbl}_hand",
                            actor,
                            [tuple(end_w.tolist()), tuple(h_tip.tolist())],
                            ra=[chain.radius_tip * 1.15 * s, chain.radius_tip * 0.65 * s],
                            rb=[chain.radius_tip * 0.68 * s, chain.radius_tip * 0.4 * s],
                            smooth_k=0.025,
                            n_stations=5,
                            color_rgb=col,
                        )
                    )
                rig.sockets_world[f"{chain.id}_{s_lbl}_end"] = (end_w, yaw_mat @ np.array([0.0, 0.0, 1.0], dtype=np.float32))

    # 2. Mechanical / Prop Socket Attachments (Wheel-legs, Visor helmets, Magic staffs, Mics)
    for mech in actor.mechanical_attachments:
        targets: List[Tuple[str, np.ndarray, np.ndarray]] = []
        sb = mech.socket_bone
        if sb == "both_ankles":
            for k in ("left_ankle", "right_ankle"):
                if k in rig.sockets_world:
                    targets.append((k, *rig.sockets_world[k]))
        elif sb == "both_wrists":
            for k in ("left_wrist", "right_wrist"):
                if k in rig.sockets_world:
                    targets.append((k, *rig.sockets_world[k]))
        elif sb in rig.sockets_world:
            targets.append((sb, *rig.sockets_world[sb]))
        else:
            targets.append(("pelvis", *rig.sockets_world["pelvis"]))

        ms = float(mech.scale) * s
        c1 = hex_to_rgb_linear(mech.primary_color)
        c2 = hex_to_rgb_linear(mech.secondary_color)
        c3 = hex_to_rgb_linear(mech.accent_color)
        m_rot = yaw_mat @ rotation_matrix_xyz(*mech.rotation_deg)

        for idx_t, (t_name, t_pos, _) in enumerate(targets):
            pos_w = t_pos + m_rot @ (np.array(mech.offset_local, dtype=np.float32) * s)
            if mech.kind == "wheel_assembly":
                # Torus tire oriented vertically in the actor's sagittal plane (rolling along Z!)
                wheel_rot = m_rot @ rotation_matrix_xyz(0.0, 0.0, 90.0)
                maj_r = 0.145 * ms
                min_r = 0.048 * ms
                w_center = pos_w + m_rot @ np.array([0.0, 0.015 * ms, 0.0], dtype=np.float32)
                solids.append(
                    CompiledSolidPrimitive(
                        name=f"{mech.name}_{t_name}_tire",
                        kind=2,  # torus_wheel
                        center=w_center,
                        rot_fwd=wheel_rot,
                        rot_inv=wheel_rot.T,
                        half_extents=np.array([maj_r + min_r, min_r, maj_r + min_r], dtype=np.float32),
                        param_a=maj_r,
                        param_b=min_r,
                        csg_op=1,
                        smooth_k=0.015,
                        color_rgb=c1,
                        secondary_rgb=c2,
                        accent_rgb=c3,
                        shader="brushed_metal",
                        metallic=0.35,
                        roughness=0.32,
                        emissive=float(mech.emissive),
                        bvh_radius=maj_r + min_r + 0.05,
                    )
                )
                # Glowing neon core hub inside wheel
                solids.append(
                    CompiledSolidPrimitive(
                        name=f"{mech.name}_{t_name}_hub",
                        kind=3,  # superquadric hub
                        center=w_center,
                        rot_fwd=wheel_rot,
                        rot_inv=wheel_rot.T,
                        half_extents=np.array([maj_r * 0.52, min_r * 1.15, maj_r * 0.52], dtype=np.float32),
                        param_a=0.35,
                        param_b=0.85,
                        csg_op=1,
                        color_rgb=c2,
                        secondary_rgb=c3,
                        metallic=0.75,
                        roughness=0.2,
                        emissive=max(0.65, float(mech.emissive)),
                        bvh_radius=maj_r * 0.7 + 0.04,
                    )
                )
            elif mech.kind == "visor_helmet":
                # Sleek swept aerodynamic helmet crest seated tightly on cranial shell + visor band
                h_c = rig.cranial_shell.center
                h_r = rig.cranial_shell.rot_fwd
                crest_pts = [
                    tuple((h_c + h_r @ np.array([0.0, 0.08 * ms, 0.05 * ms], dtype=np.float32)).tolist()),
                    tuple((h_c + h_r @ np.array([0.0, 0.095 * ms, -0.07 * ms], dtype=np.float32)).tolist()),
                    tuple((h_c + h_r @ np.array([0.0, 0.04 * ms, -0.24 * ms], dtype=np.float32)).tolist()),
                ]
                bones.append(
                    _build_spline_bone(
                        f"{mech.name}_crest",
                        actor,
                        crest_pts,
                        ra=[0.118 * ms, 0.122 * ms, 0.024 * ms],
                        rb=[0.058 * ms, 0.072 * ms, 0.020 * ms],
                        sq=[1.65, 1.4],
                        smooth_k=0.035,
                        n_stations=7,
                        color_rgb=c1,
                    )
                )
                visor_pts = [
                    tuple((h_c + h_r @ np.array([-0.095 * ms, 0.052 * ms, 0.032 * ms], dtype=np.float32)).tolist()),
                    tuple((h_c + h_r @ np.array([0.0, 0.064 * ms, 0.125 * ms], dtype=np.float32)).tolist()),
                    tuple((h_c + h_r @ np.array([0.095 * ms, 0.052 * ms, 0.032 * ms], dtype=np.float32)).tolist()),
                ]
                vb = _build_spline_bone(
                    f"{mech.name}_visor",
                    actor,
                    visor_pts,
                    ra=[0.018 * ms, 0.024 * ms, 0.018 * ms],
                    rb=[0.012 * ms, 0.014 * ms, 0.012 * ms],
                    smooth_k=0.022,
                    csg_op=1,
                    n_stations=6,
                    color_rgb=c2,
                )
                vb.emissive = max(0.85, float(mech.emissive))
                vb.roughness = 0.15
                bones.append(vb)
            elif mech.kind == "magic_staff":
                p_bot = pos_w + m_rot @ np.array([0.0, -0.78 * ms, 0.0], dtype=np.float32)
                p_top = pos_w + m_rot @ np.array([0.0, 0.72 * ms, 0.0], dtype=np.float32)
                shaft = _build_spline_bone(
                    f"{mech.name}_shaft",
                    actor,
                    [tuple(p_bot.tolist()), tuple(p_top.tolist())],
                    ra=[0.022 * ms, 0.028 * ms],
                    smooth_k=0.01,
                    csg_op=1,
                    cap_roundness=0.25,
                    n_stations=5,
                    color_rgb=c1,
                )
                shaft.metallic = 0.55
                shaft.roughness = 0.35
                bones.append(shaft)
                solids.append(
                    CompiledSolidPrimitive(
                        name=f"{mech.name}_orb",
                        kind=3,
                        center=p_top + m_rot @ np.array([0.0, 0.08 * ms, 0.0], dtype=np.float32),
                        rot_fwd=m_rot,
                        rot_inv=m_rot.T,
                        half_extents=np.array([0.085 * ms, 0.11 * ms, 0.085 * ms], dtype=np.float32),
                        param_a=0.7,
                        param_b=0.7,
                        csg_op=1,
                        color_rgb=c2,
                        emissive=max(1.8, float(mech.emissive)),
                        roughness=0.12,
                        bvh_radius=0.16 * ms,
                    )
                )
            elif mech.kind == "handheld_mic":
                p_bot = pos_w + m_rot @ np.array([0.0, -0.09 * ms, -0.04 * ms], dtype=np.float32)
                p_top = pos_w + m_rot @ np.array([0.0, 0.08 * ms, 0.05 * ms], dtype=np.float32)
                mic_b = _build_spline_bone(
                    f"{mech.name}_handle",
                    actor,
                    [tuple(p_bot.tolist()), tuple(p_top.tolist())],
                    ra=[0.018 * ms, 0.024 * ms],
                    smooth_k=0.01,
                    csg_op=1,
                    cap_roundness=0.3,
                    n_stations=4,
                    color_rgb=c1,
                )
                mic_b.metallic = 0.85
                mic_b.roughness = 0.22
                bones.append(mic_b)
                solids.append(
                    CompiledSolidPrimitive(
                        name=f"{mech.name}_grille",
                        kind=3,
                        center=p_top + m_rot @ np.array([0.0, 0.028 * ms, 0.018 * ms], dtype=np.float32),
                        rot_fwd=m_rot,
                        rot_inv=m_rot.T,
                        half_extents=np.array([0.038 * ms, 0.044 * ms, 0.038 * ms], dtype=np.float32),
                        param_a=1.0,
                        param_b=1.0,
                        csg_op=1,
                        color_rgb=c2,
                        metallic=0.9,
                        roughness=0.25,
                        bvh_radius=0.08 * ms,
                    )
                )
            elif mech.kind == "horn_pair":
                for sk in ("horn_left", "horn_right"):
                    pl, dl = rig.cranial_shell.socket_anchors_local[sk]
                    hc = rig.cranial_shell.center
                    hr = rig.cranial_shell.rot_fwd
                    p0 = hc + hr @ (pl * 0.9)
                    p1 = p0 + hr @ (dl * 0.14 * ms + np.array([0.0, 0.06 * ms, -0.04 * ms], dtype=np.float32))
                    p2 = p1 + hr @ (dl * 0.14 * ms + np.array([0.0, 0.08 * ms, 0.05 * ms], dtype=np.float32))
                    hb = _build_spline_bone(
                        f"{mech.name}_{sk}",
                        actor,
                        [tuple(p0.tolist()), tuple(p1.tolist()), tuple(p2.tolist())],
                        ra=[0.032 * ms, 0.022 * ms, 0.008 * ms],
                        smooth_k=0.025,
                        n_stations=7,
                        color_rgb=c1,
                    )
                    hb.roughness = 0.35
                    bones.append(hb)


def _compile_humanoid_rig(actor: ActorSpec, origin: np.ndarray, yaw_mat: np.ndarray, style_mode: str) -> CompiledActorRig:
    """
    Compiles a full articulated humanoid (or elf, dwarf, alien, speedster, mystic)
    rig with 2-bone analytical IK limbs, smooth C2 torso/neck/skull joints,
    and support for arbitrary bone omissions, extra arms, and mechanical sockets.
    """
    m = actor.proportions
    s = float(actor.scale) * float(m.stature)
    tw = float(m.torso_width)
    ll = float(m.limb_length)
    omitted = set(actor.omit_bones)
    bones: List[CompiledSplineBone] = []
    solids: List[CompiledSolidPrimitive] = []
    sockets_world: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    is_skating = actor.posture == "skating_lean"
    is_sitting = actor.posture in ("sitting_haunches", "sitting_edge")

    def wpt(x: float, y: float, z: float) -> np.ndarray:
        return (yaw_mat @ np.array([x * s, y * s, z * s], dtype=np.float32)) + origin

    # Torso lean angle & pelvis elevation
    lean_z = 0.14 if is_skating else (0.04 if actor.posture == "singing_pose" else 0.0)
    pelvis_y = 0.42 if is_sitting else (0.72 * ll if is_skating else 0.78 * ll)
    waist_y = pelvis_y + 0.18
    chest_y = pelvis_y + 0.38
    neck_base_y = pelvis_y + 0.47
    head_y = pelvis_y + 0.64

    pelvis_w = wpt(0.0, pelvis_y, -0.02 * lean_z)
    waist_w = wpt(0.0, waist_y, 0.25 * lean_z)
    chest_w = wpt(0.0, chest_y, 0.65 * lean_z)
    neck_base_w = wpt(0.0, neck_base_y, 0.75 * lean_z)
    head_c = wpt(0.0, head_y, 0.90 * lean_z)

    # Torso spine spline (pelvis -> waist -> upper chest)
    bones.append(
        _build_spline_bone(
            "torso",
            actor,
            [tuple(pelvis_w.tolist()), tuple(waist_w.tolist()), tuple(chest_w.tolist()), tuple(neck_base_w.tolist())],
            ra=[0.135 * tw * s, 0.110 * tw * s, 0.142 * tw * s, 0.068 * s],
            rb=[0.105 * tw * s, 0.088 * tw * s, 0.102 * tw * s, 0.062 * s],
            sq=[1.85, 1.85, 1.75],
            smooth_k=0.055,
            n_stations=10,
        )
    )

    # Neck smoothly joining chest to cranial shell
    bones.append(
        _build_spline_bone(
            "neck",
            actor,
            [tuple((neck_base_w - yaw_mat @ np.array([0.0, 0.03 * s, 0.0], dtype=np.float32)).tolist()), tuple(head_c.tolist())],
            ra=[0.054 * s, 0.048 * s],
            smooth_k=0.045,
            n_stations=5,
        )
    )

    head_rot = yaw_mat @ rotation_matrix_xyz(-8.0 if actor.posture == "singing_pose" else 0.0, 0.0, 0.0)
    cranial = compile_cranial_shell(actor, head_c, head_rot, style_mode=style_mode)

    sockets_world["head"] = (head_c, head_rot @ np.array([0.0, 1.0, 0.0], dtype=np.float32))
    sockets_world["upper_torso"] = (chest_w, yaw_mat @ np.array([0.0, 1.0, 0.0], dtype=np.float32))
    sockets_world["mid_torso"] = (0.5 * (waist_w + chest_w), yaw_mat @ np.array([0.0, 1.0, 0.0], dtype=np.float32))
    sockets_world["pelvis"] = (pelvis_w, yaw_mat @ np.array([0.0, 0.0, -1.0], dtype=np.float32))

    # Optional pointed Elf / Goblin ears on humanoid skull
    if m.ear_length > 0.16 or abs(m.ear_tilt_deg) > 22.0:
        ear_len = max(0.08, m.ear_length) * s * 0.55
        for ek in ("ear_left", "ear_right"):
            pl, dl = cranial.socket_anchors_local[ek]
            p0 = head_c + head_rot @ (pl * 0.88)
            p1 = p0 + head_rot @ (dl * ear_len * 0.55)
            p2 = p0 + head_rot @ (dl * ear_len)
            bones.append(
                _build_spline_bone(
                    ek,
                    actor,
                    [tuple(p0.tolist()), tuple(p1.tolist()), tuple(p2.tolist())],
                    ra=[0.026 * s, 0.018 * s, 0.005 * s],
                    rb=[0.016 * s, 0.012 * s, 0.004 * s],
                    smooth_k=0.022,
                    n_stations=5,
                )
            )

    # Primary 2-Bone IK Arms
    for side_name, sx, tgt_key in (("left", 1.0, actor.left_hand_target), ("right", -1.0, actor.right_hand_target)):
        if f"{side_name}_arm" in omitted:
            continue
        sh_p = chest_w + yaw_mat @ np.array([sx * 0.145 * tw * s, 0.012 * s, 0.0], dtype=np.float32)
        hand_tgt = _resolve_humanoid_hand_target(tgt_key, sx, sh_p, head_c, origin, yaw_mat, s, actor.posture)
        el_p, wr_p, bend_r = solve_two_bone_ik(
            sh_p,
            hand_tgt,
            0.26 * ll * s,
            0.24 * ll * s,
            yaw_mat @ np.array([sx * 0.25, -0.3, -1.0], dtype=np.float32),
        )
        bones.append(
            _build_spline_bone(
                f"{side_name}_upper_arm",
                actor,
                [tuple(sh_p.tolist()), tuple(el_p.tolist())],
                ra=[0.052 * tw * s, 0.044 * s],
                smooth_k=0.048,
                n_stations=6,
                joint_bend_rad=bend_r,
            )
        )
        bones.append(
            _build_spline_bone(
                f"{side_name}_forearm",
                actor,
                [tuple(el_p.tolist()), tuple(wr_p.tolist())],
                ra=[0.043 * s, 0.030 * s],
                smooth_k=0.038,
                n_stations=6,
                joint_bend_rad=bend_r,
            )
        )
        if f"{side_name}_hand" not in omitted:
            f_dir = wr_p - el_p
            f_dir /= max(1e-5, float(np.linalg.norm(f_dir)))
            h_tip = wr_p + f_dir * (0.095 * s)
            bones.append(
                _build_spline_bone(
                    f"{side_name}_hand",
                    actor,
                    [tuple(wr_p.tolist()), tuple(h_tip.tolist())],
                    ra=[0.034 * s, 0.018 * s],
                    rb=[0.018 * s, 0.011 * s],
                    smooth_k=0.02,
                    n_stations=5,
                )
            )
        sockets_world[f"{side_name}_wrist"] = (wr_p, yaw_mat @ np.array([0.0, 0.0, 1.0], dtype=np.float32))
        sockets_world[f"{side_name}_hand"] = sockets_world[f"{side_name}_wrist"]

    # Primary 2-Bone IK Legs (respecting omit_bones for wheel-legs or naga/merfolk!)
    for side_name, sx in (("left", 1.0), ("right", -1.0)):
        if f"{side_name}_leg" in omitted or "legs" in omitted:
            continue
        hip_p = pelvis_w + yaw_mat @ np.array([sx * 0.074 * tw * s, -0.005 * s, 0.0], dtype=np.float32)
        if is_sitting:
            ank_tgt = wpt(sx * 0.10 * tw, 0.04, 0.34)
        elif is_skating:
            z_stride = 0.24 if sx > 0 else -0.22
            ank_tgt = wpt(sx * 0.12 * tw, 0.16, z_stride)
        else:
            ank_tgt = wpt(sx * 0.095 * tw, 0.055, 0.02)

        kn_p, ank_p, bend_r = solve_two_bone_ik(
            hip_p,
            ank_tgt,
            0.39 * ll * s,
            0.38 * ll * s,
            yaw_mat @ np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        bones.append(
            _build_spline_bone(
                f"{side_name}_thigh",
                actor,
                [tuple(hip_p.tolist()), tuple(kn_p.tolist())],
                ra=[0.092 * tw * s, 0.056 * s],
                smooth_k=0.052,
                n_stations=7,
                joint_bend_rad=bend_r,
            )
        )
        if f"{side_name}_lower_leg" not in omitted:
            bones.append(
                _build_spline_bone(
                    f"{side_name}_calf",
                    actor,
                    [tuple(kn_p.tolist()), tuple(ank_p.tolist())],
                    ra=[0.056 * s, 0.036 * s],
                    smooth_k=0.038,
                    n_stations=7,
                    joint_bend_rad=bend_r,
                )
            )
        sockets_world[f"{side_name}_ankle"] = (ank_p, yaw_mat @ np.array([0.0, -1.0, 0.0], dtype=np.float32))
        if f"{side_name}_foot" not in omitted and "feet" not in omitted:
            toe_p = ank_p + yaw_mat @ np.array([0.0, -0.035 * s, 0.13 * s], dtype=np.float32)
            bones.append(
                _build_spline_bone(
                    f"{side_name}_foot",
                    actor,
                    [tuple(ank_p.tolist()), tuple(toe_p.tolist())],
                    ra=[0.040 * s, 0.036 * s],
                    rb=[0.034 * s, 0.022 * s],
                    smooth_k=0.025,
                    n_stations=5,
                )
            )

    # Optional humanoid tail (e.g. XLR8 speedster tail, demon/succubus tail, monkey tail)
    if m.tail_length > 0.05:
        tl = m.tail_length * s
        tc = m.tail_curl
        tail_ctrl = [
            tuple(pelvis_w.tolist()),
            tuple((pelvis_w + yaw_mat @ np.array([0.0, -0.05 * s, -0.35 * tl], dtype=np.float32)).tolist()),
            tuple((pelvis_w + yaw_mat @ np.array([0.16 * tc * s, 0.08 * tc * s, -0.75 * tl], dtype=np.float32)).tolist()),
            tuple((pelvis_w + yaw_mat @ np.array([0.28 * tc * s, 0.25 * tc * s, -1.05 * tl], dtype=np.float32)).tolist()),
        ]
        bones.append(
            _build_spline_bone(
                "tail",
                actor,
                tail_ctrl,
                ra=[0.068 * s, 0.052 * s, 0.036 * s, 0.018 * s],
                smooth_k=0.045,
                n_stations=12,
            )
        )

    rig = CompiledActorRig(
        actor=actor,
        origin_world=origin,
        yaw_rad=math.radians(actor.facing_deg),
        cranial_shell=cranial,
        bones=bones,
        solid_parts=solids,
        sockets_world=sockets_world,
        bvh_center=wpt(0.0, 0.85, 0.0),
        bvh_radius=1.35 * s,
    )
    _attach_extra_chains_and_mech(rig, actor, origin, yaw_mat)
    return rig


def compile_actor_rig(
    actor: ActorSpec,
    origin_world: Optional[np.ndarray] = None,
    *,
    style_mode: str = "anime_npr",
) -> CompiledActorRig:
    """
    Compiles any ActorSpec into a connected Bone-Graph DAG, cranial SH shell,
    extra IK limb chains, and socketed mechanical attachments.
    """
    origin = np.zeros(3, dtype=np.float32) if origin_world is None else np.asarray(origin_world, dtype=np.float32)
    yaw_mat = rotation_matrix_xyz(0.0, float(actor.facing_deg), 0.0)

    if actor.archetype == "quadruped":
        rig = _compile_quadruped_cat_rig(actor, origin, yaw_mat, style_mode)
        _attach_extra_chains_and_mech(rig, actor, origin, yaw_mat)
    else:
        rig = _compile_humanoid_rig(actor, origin, yaw_mat, style_mode)

    # Compute exact tight bounding sphere around all bones, solids, and skull
    pts_all = [rig.cranial_shell.center[None, :]]
    for b in rig.bones:
        pts_all.append(b.stations.points)
    for sp in rig.solid_parts:
        pts_all.append(sp.center[None, :])
    merged = np.vstack(pts_all)
    bmin = np.min(merged, axis=0)
    bmax = np.max(merged, axis=0)
    bvh_c = (0.5 * (bmin + bmax)).astype(np.float32)
    max_d = float(np.max(np.linalg.norm(merged - bvh_c[None, :], axis=1)))
    rig.bvh_center = bvh_c
    rig.bvh_radius = max_d + 0.35 * float(actor.scale)
    return rig
