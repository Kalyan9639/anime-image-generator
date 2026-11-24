"""
Single-Pass Body-UV Garment Pattern Cutter & Decal Compiler for PrismStudio.

Translates open-ended GarmentLayerSpec objects (hoodies, police uniforms,
nurse dresses, firefighter coats, pleated skirts, flared elf cloaks) into
compact shader/CPU records linked directly to underlying skeleton bones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..math_ops.materials import hex_to_rgb_linear
from ..math_ops.primitives import compute_bishop_frames
from ..schema.scene_dna import ActorSpec, GarmentLayerSpec
from .kinematics import CompiledActorRig, CompiledSplineBone


@dataclass
class CompiledDecalRecord:
    kind: int           # 0=star_badge, 1=medical_cross, 2=safety_stripes, 3=button_seam, 4=lace_border, 5=emblem_circle
    uv_t: float
    uv_theta: float
    size: float
    color_rgb: np.ndarray
    secondary_rgb: np.ndarray


@dataclass
class CompiledGarmentRecord:
    name: str
    fit_offset: float
    thickness: float
    flare: float
    flare_start_t: float
    pleat_count: int
    pleat_amp: float
    ruffle_amp: float
    fold_amp: float
    open_front_rad: float
    t_min: float
    t_max: float
    color_rgb: np.ndarray
    secondary_rgb: np.ndarray
    accent_rgb: np.ndarray
    shader: str
    roughness: float
    metallic: float
    decals: List[CompiledDecalRecord] = field(default_factory=list)


DECAL_KIND_MAP = {
    "star_badge": 0,
    "medical_cross": 1,
    "safety_stripes": 2,
    "button_seam": 3,
    "lace_border": 4,
    "emblem_circle": 5,
}


def compile_actor_garments(
    actor: ActorSpec,
    rig: CompiledActorRig,
    garments_table: List[CompiledGarmentRecord],
) -> None:
    """
    Links Body-UV garment shell records onto matching actor bones and emits
    independent flared skirt/cloak shells where flare > 0.1.
    """
    if not actor.garments or actor.archetype == "quadruped":
        return

    s = float(actor.scale) * float(actor.proportions.stature)
    tw = float(actor.proportions.torso_width)
    bone_by_name: Dict[str, CompiledSplineBone] = {b.name: b for b in rig.bones}

    for g in actor.garments:
        neck_cut = {"crew": 0.93, "v_neck": 0.80, "scoop": 0.84, "off_shoulder": 0.72, "mandarin": 0.97, "hoodie_cowl": 0.96, "open_lapel": 0.88}.get(g.neckline, 0.90)
        open_rad = math.radians(max(0.0, g.open_front_deg))
        if g.neckline == "open_lapel" and open_rad < 0.15:
            open_rad = 0.28

        decal_recs = [
            CompiledDecalRecord(
                kind=DECAL_KIND_MAP.get(d.kind, 0),
                uv_t=float(d.uv_t),
                uv_theta=math.radians(float(d.uv_theta_deg)),
                size=float(d.size),
                color_rgb=hex_to_rgb_linear(d.color),
                secondary_rgb=hex_to_rgb_linear(d.secondary_color),
            )
            for d in g.uv_decals
        ]

        g_id = len(garments_table)
        garments_table.append(
            CompiledGarmentRecord(
                name=g.name,
                fit_offset=max(0.004, float(g.fit_offset) * s),
                thickness=max(0.004, float(g.thickness) * s),
                flare=float(g.flare) * s,
                flare_start_t=0.35,
                pleat_count=int(g.pleat_count),
                pleat_amp=float(g.pleat_amp) * s,
                ruffle_amp=float(g.ruffle_amp) * s,
                fold_amp=float(g.fold_amp) * s,
                open_front_rad=open_rad,
                t_min=0.02,
                t_max=min(0.99, neck_cut),
                color_rgb=hex_to_rgb_linear(g.primary_color),
                secondary_rgb=hex_to_rgb_linear(g.secondary_color),
                accent_rgb=hex_to_rgb_linear(g.accent_color),
                shader=g.fabric_shader,
                roughness=float(g.roughness),
                metallic=float(g.metallic),
                decals=decal_recs,
            )
        )

        # Bind torso garment shell (never let back cloaks overwrite inner torso tunic!)
        if g.region in ("upper_torso", "full_dress", "outer_coat"):
            if "torso" in bone_by_name:
                bone_by_name["torso"].garment_id = g_id
            # Bind tailored sleeves proportionally to sleeve_length (no bulky stovepipe tubes!)
            if g.sleeve_length > 0.08:
                arm_gid = len(garments_table)
                sleeve_fit = min(0.018 * s, max(0.005 * s, float(g.fit_offset) * 0.72 * s))
                arm_rec = CompiledGarmentRecord(
                    name=f"{g.name}_sleeves",
                    fit_offset=sleeve_fit,
                    thickness=max(0.005, float(g.thickness) * s),
                    flare=0.0,
                    flare_start_t=0.5,
                    pleat_count=0,
                    pleat_amp=0.0,
                    ruffle_amp=0.0,
                    fold_amp=min(0.004, float(g.fold_amp)) * s,
                    open_front_rad=0.0,
                    t_min=-0.05,
                    t_max=min(0.98, float(g.sleeve_length) * 1.85),
                    color_rgb=hex_to_rgb_linear(g.primary_color),
                    secondary_rgb=hex_to_rgb_linear(g.secondary_color),
                    accent_rgb=hex_to_rgb_linear(g.accent_color),
                    shader=g.fabric_shader,
                    roughness=float(g.roughness),
                    metallic=float(g.metallic),
                    decals=[],
                )
                garments_table.append(arm_rec)
                for b_key in ("left_upper_arm", "right_upper_arm"):
                    if b_key in bone_by_name:
                        bone_by_name[b_key].garment_id = arm_gid
                if g.sleeve_length > 0.55:
                    fore_gid = len(garments_table)
                    fore_rec = CompiledGarmentRecord(
                        name=f"{g.name}_fore_sleeves",
                        fit_offset=min(0.015 * s, arm_rec.fit_offset * 0.85),
                        thickness=arm_rec.thickness,
                        flare=0.0,
                        flare_start_t=0.2,
                        pleat_count=0,
                        pleat_amp=0.0,
                        ruffle_amp=0.0,
                        fold_amp=arm_rec.fold_amp,
                        open_front_rad=0.0,
                        t_min=-0.02,
                        t_max=min(0.96, (float(g.sleeve_length) - 0.5) * 1.9),
                        color_rgb=arm_rec.color_rgb,
                        secondary_rgb=arm_rec.secondary_rgb,
                        accent_rgb=arm_rec.accent_rgb,
                        shader=g.fabric_shader,
                        roughness=float(g.roughness),
                        metallic=float(g.metallic),
                    )
                    garments_table.append(fore_rec)
                    for b_key in ("left_forearm", "right_forearm"):
                        if b_key in bone_by_name:
                            bone_by_name[b_key].garment_id = fore_gid

        # Flared skirt / dress hem / back cape shell
        if g.region in ("full_dress", "lower_skirt", "cloak"):
            pelvis_w, _ = rig.sockets_world["pelvis"]
            upper_w, _ = rig.sockets_world["upper_torso"]
            if g.region == "cloak":
                # Smooth back cape hanging behind shoulders & torso (never wraps chest front!)
                skirt_top = upper_w + np.array([0.0, -0.04 * s, -0.11 * s], dtype=np.float32)
                skirt_len = 0.85 * float(g.hem_length) * s
                skirt_bot = skirt_top + np.array([0.0, -skirt_len, -0.22 * s], dtype=np.float32)
                base_r = (0.14 * tw) * s
                flare_r = base_r + max(0.08, g.flare * 0.24) * s
                rad_b = [0.025 * s, 0.032 * s]
                open_cape = 0.0
            else:
                # Tuck top of skirt cleanly under waist/hoodie hem
                skirt_top = pelvis_w + np.array([0.0, 0.04 * s, 0.01 * s], dtype=np.float32)
                skirt_len = 0.42 * float(g.hem_length) * s
                skirt_bot = skirt_top + np.array([0.0, -skirt_len, 0.01 * s], dtype=np.float32)
                base_r = max(0.192 * tw, 0.182) * s
                flare_r = base_r + max(0.10, g.flare * 0.22) * s
                rad_b = [base_r * 0.86, flare_r * 0.90]
                open_cape = 0.0
            st = compute_bishop_frames(
                [tuple(skirt_top.tolist()), tuple(skirt_bot.tolist())],
                radii_a=[base_r, flare_r],
                radii_b=rad_b,
                squareness_n=[1.9, 1.9],
                n_stations=7,
                cap_roundness=0.35,
            )
            skirt_gid = len(garments_table)
            garments_table.append(
                CompiledGarmentRecord(
                    name=f"{g.name}_skirt_shell",
                    fit_offset=0.008 * s,
                    thickness=max(0.018, float(g.thickness) * 1.4 * s),
                    flare=0.0,
                    flare_start_t=0.0,
                    pleat_count=int(g.pleat_count),
                    pleat_amp=min(0.012, float(g.pleat_amp)) * s,
                    ruffle_amp=min(0.010, float(g.ruffle_amp)) * s,
                    fold_amp=0.003 * s,
                    open_front_rad=open_cape,
                    t_min=-0.02,
                    t_max=0.98,
                    color_rgb=hex_to_rgb_linear(g.primary_color),
                    secondary_rgb=hex_to_rgb_linear(g.secondary_color),
                    accent_rgb=hex_to_rgb_linear(g.accent_color),
                    shader=g.fabric_shader,
                    roughness=float(g.roughness),
                    metallic=float(g.metallic),
                    decals=decal_recs if g.region == "lower_skirt" else [],
                )
            )
            rig.bones.append(
                CompiledSplineBone(
                    name=f"{g.name}_skirt_bell",
                    actor_id=actor.id,
                    stations=st,
                    smooth_k=0.025 * s,
                    csg_op=1,
                    bone_role=4,
                    color_rgb=hex_to_rgb_linear(g.primary_color),
                    secondary_rgb=hex_to_rgb_linear(g.secondary_color),
                    shader=g.fabric_shader,
                    metallic=float(g.metallic),
                    roughness=float(g.roughness),
                    emissive=0.0,
                    garment_id=skirt_gid,
                )
            )

        if g.region == "trousers":
            # Full hip-to-ankle trousers + matching dark boots/shoes on feet
            pant_gid = len(garments_table)
            garments_table.append(
                CompiledGarmentRecord(
                    name=f"{g.name}_legs",
                    fit_offset=max(0.007, float(g.fit_offset) * s),
                    thickness=max(0.007, float(g.thickness) * s),
                    flare=0.0,
                    flare_start_t=0.5,
                    pleat_count=0,
                    pleat_amp=0.0,
                    ruffle_amp=0.0,
                    fold_amp=0.003 * s,
                    open_front_rad=0.0,
                    t_min=-0.04,
                    t_max=1.06,
                    color_rgb=hex_to_rgb_linear(g.primary_color),
                    secondary_rgb=hex_to_rgb_linear(g.secondary_color),
                    accent_rgb=hex_to_rgb_linear(g.accent_color),
                    shader=g.fabric_shader,
                    roughness=float(g.roughness),
                    metallic=float(g.metallic),
                )
            )
            for b_key in ("left_thigh", "right_thigh", "left_calf", "right_calf"):
                if b_key in bone_by_name:
                    bone_by_name[b_key].garment_id = pant_gid
            # Snug pelvis hip-brief spline under trousers so hip crotch never shows bare skin
            pelvis_w, _ = rig.sockets_world["pelvis"]
            hip_pts = [
                tuple((pelvis_w + np.array([0.0, -0.055 * s, 0.0], dtype=np.float32)).tolist()),
                tuple((pelvis_w + np.array([0.0, 0.055 * s, 0.0], dtype=np.float32)).tolist()),
            ]
            st_hip = compute_bishop_frames(hip_pts, radii_a=[0.135 * tw * s, 0.142 * tw * s], radii_b=[0.098 * tw * s, 0.105 * tw * s], n_stations=4, cap_roundness=0.8)
            rig.bones.append(
                CompiledSplineBone(
                    name=f"{g.name}_pelvis_brief",
                    actor_id=actor.id,
                    stations=st_hip,
                    smooth_k=0.035 * s,
                    csg_op=1,
                    bone_role=2,
                    color_rgb=hex_to_rgb_linear(g.primary_color),
                    secondary_rgb=hex_to_rgb_linear(g.secondary_color),
                    shader=g.fabric_shader,
                    metallic=float(g.metallic),
                    roughness=float(g.roughness),
                    emissive=0.0,
                    garment_id=pant_gid,
                )
            )
            for f_key in ("left_foot", "right_foot"):
                if f_key in bone_by_name:
                    bone_by_name[f_key].color_rgb = hex_to_rgb_linear(g.primary_color) * 0.65
                    bone_by_name[f_key].shader = g.fabric_shader
                    bone_by_name[f_key].roughness = 0.52
