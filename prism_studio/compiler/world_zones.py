"""
3-Zone World & Architecture Compiler for PrismStudio.

Compiles:
- Zone 1: Hero foreground architecture & props (staggered masonry brick walls
  with beveled stone coping and climbing ivy vines, concert stages, mic stands,
  piano keyboards, trees, boulders, street lamps).
- Zone 2: Midground depth silhouettes (rooftops, city skyline, forest canopy,
  stage crowd, mountains) positioned with realistic elevation relative to Zone 1.
- Zone 3: Sky gradients, analytical ground plane, and HDR emitters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from ..math_ops.materials import hex_to_rgb_linear
from ..math_ops.primitives import compute_bishop_frames, rotation_matrix_xyz
from ..schema.scene_dna import MidgroundZoneSpec, PropSpec, SceneDNA
from .kinematics import CompiledSolidPrimitive, CompiledSplineBone


@dataclass
class CompiledProp:
    id: str
    label: str
    origin_world: np.ndarray
    solids: List[CompiledSolidPrimitive]
    splines: List[CompiledSplineBone]
    top_surface_y: float
    bmin: np.ndarray
    bmax: np.ndarray
    bvh_center: np.ndarray
    bvh_radius: float


@dataclass
class CompiledMidgroundItem:
    center: np.ndarray          # (3,)
    half_extents: np.ndarray    # (3,)
    roof_pitch: float           # 0.0 = flat box, >0 = pitched gable roof height
    color_rgb: np.ndarray
    window_glow_rgb: np.ndarray
    has_windows: bool


def compile_zone1_prop(
    prop: PropSpec,
    origin_world: np.ndarray,
    *,
    seed: int = 42,
) -> CompiledProp:
    """Compiles a Zone 1 architectural element or prop into smooth solids + splines."""
    sx, sy, sz = (
        max(0.2, float(prop.size[0])),
        max(0.15, float(prop.size[1])),
        max(0.18, float(prop.size[2])),
    )
    rot_fwd = rotation_matrix_xyz(*prop.rotation_deg)
    rot_inv = rot_fwd.T
    c1 = hex_to_rgb_linear(prop.primary_color)
    c2 = hex_to_rgb_linear(prop.secondary_color)
    c3 = hex_to_rgb_linear(prop.accent_color)

    solids: List[CompiledSolidPrimitive] = []
    splines: List[CompiledSplineBone] = []

    if prop.archetype in ("masonry_wall", "rooftop_ledge"):
        coping_h = 0.095 if prop.has_top_coping else 0.0
        body_h = max(0.25, sy - coping_h)
        # Main staggered brick / ashlar body
        brick_w, brick_h, brick_d = 0.34, 0.145, sz * 0.96
        nx = max(1, min(14, int(round(sx / brick_w)) // 2))
        ny = max(1, min(12, int(round(body_h / brick_h)) // 2))
        body_center = origin_world + rot_fwd @ np.array([0.0, body_h * 0.5, 0.0], dtype=np.float32)

        # Solid recessed mortar core behind bricks so cracks never leak light
        solids.append(
            CompiledSolidPrimitive(
                name=f"{prop.id}_mortar_core",
                kind=0,
                center=body_center,
                rot_fwd=rot_fwd,
                rot_inv=rot_inv,
                half_extents=np.array([sx * 0.49, body_h * 0.49, sz * 0.45], dtype=np.float32),
                param_a=0.015,
                param_b=0.0,
                color_rgb=c2 * 0.72,
                secondary_rgb=c2 * 0.55,
                shader="stone_voronoi",
                roughness=0.88,
                bvh_radius=float(np.linalg.norm([sx * 0.5, body_h * 0.5, sz * 0.5])) + 0.05,
            )
        )
        # Staggered half-bond masonry courses
        solids.append(
            CompiledSolidPrimitive(
                name=f"{prop.id}_bricks",
                kind=1,  # staggered_masonry
                center=body_center,
                rot_fwd=rot_fwd,
                rot_inv=rot_inv,
                half_extents=np.array([brick_w * 0.46, brick_h * 0.44, brick_d * 0.5], dtype=np.float32),
                param_a=0.014,
                param_b=0.0,
                repeat_bounds=(nx, ny, 0),
                repeat_pitch=(brick_w, brick_h, brick_d),
                color_rgb=c1,
                secondary_rgb=c2,
                accent_rgb=c3,
                shader="brick_masonry",
                roughness=prop.roughness,
                metallic=prop.metallic,
                bvh_radius=float(np.linalg.norm([sx * 0.55, body_h * 0.55, sz * 0.55])) + 0.08,
            )
        )
        top_y = float(origin_world[1]) + body_h
        if prop.has_top_coping:
            coping_c = origin_world + rot_fwd @ np.array([0.0, body_h + coping_h * 0.5, 0.0], dtype=np.float32)
            solids.append(
                CompiledSolidPrimitive(
                    name=f"{prop.id}_coping",
                    kind=0,
                    center=coping_c,
                    rot_fwd=rot_fwd,
                    rot_inv=rot_inv,
                    half_extents=np.array([sx * 0.52, coping_h * 0.5, sz * 0.56], dtype=np.float32),
                    param_a=0.022,
                    param_b=0.0,
                    color_rgb=c2,
                    secondary_rgb=c2 * 0.82,
                    shader="stone_voronoi",
                    roughness=0.68,
                    bvh_radius=float(np.linalg.norm([sx * 0.55, coping_h, sz * 0.6])) + 0.05,
                )
            )
            top_y += coping_h

        if prop.has_foliage_ivy:
            rng = np.random.default_rng(seed + 41)
            for vi in range(4):
                x_anchor = -sx * 0.34 + (vi / 3.0) * sx * 0.68 + float(rng.uniform(-0.08, 0.08))
                z_front = sz * 0.515
                s_dir = 1.0 if (vi % 2 == 0) else -1.0
                # Graceful S-curved creeping ivy tendrils reaching up and across mortar joints
                pts = [
                    tuple((origin_world + rot_fwd @ np.array([x_anchor - 0.14 * s_dir, 0.08, z_front], dtype=np.float32)).tolist()),
                    tuple((origin_world + rot_fwd @ np.array([x_anchor + 0.16 * s_dir, body_h * 0.36, z_front + 0.022], dtype=np.float32)).tolist()),
                    tuple((origin_world + rot_fwd @ np.array([x_anchor - 0.12 * s_dir, body_h * 0.72, z_front + 0.022], dtype=np.float32)).tolist()),
                    tuple((origin_world + rot_fwd @ np.array([x_anchor + 0.08 * s_dir, body_h + 0.02, z_front - 0.01], dtype=np.float32)).tolist()),
                ]
                st = compute_bishop_frames(
                    pts,
                    [0.011, 0.024, 0.013, 0.026, 0.012, 0.022, 0.008],
                    [0.008, 0.011, 0.008, 0.012, 0.008, 0.010, 0.006],
                    [1.5, 1.5],
                    n_stations=9,
                )
                splines.append(
                    CompiledSplineBone(
                        name=f"{prop.id}_ivy_{vi}",
                        actor_id=prop.id,
                        stations=st,
                        smooth_k=0.015,
                        csg_op=0,
                        bone_role=3,
                        color_rgb=c3 * 0.85,
                        secondary_rgb=c3 * 0.55,
                        shader="foliage_leaf",
                        metallic=0.0,
                        roughness=0.58,
                        emissive=0.0,
                    )
                )

    elif prop.archetype == "stage_deck":
        deck_c = origin_world + rot_fwd @ np.array([0.0, sy * 0.5, 0.0], dtype=np.float32)
        solids.append(
            CompiledSolidPrimitive(
                name=f"{prop.id}_deck",
                kind=0,
                center=deck_c,
                rot_fwd=rot_fwd,
                rot_inv=rot_inv,
                half_extents=np.array([sx * 0.5, sy * 0.5, sz * 0.5], dtype=np.float32),
                param_a=0.03,
                param_b=0.0,
                color_rgb=c1,
                secondary_rgb=c2,
                shader=prop.shader if prop.shader else "wood_grain",
                metallic=0.12,
                roughness=0.28,
                bvh_radius=float(np.linalg.norm([sx * 0.5, sy * 0.5, sz * 0.5])) + 0.05,
            )
        )
        top_y = float(origin_world[1]) + sy

    elif prop.archetype == "microphone_stand":
        base_c = origin_world + rot_fwd @ np.array([0.0, 0.035, 0.0], dtype=np.float32)
        solids.append(
            CompiledSolidPrimitive(
                name=f"{prop.id}_base",
                kind=3,
                center=base_c,
                rot_fwd=rot_fwd,
                rot_inv=rot_inv,
                half_extents=np.array([0.19, 0.035, 0.19], dtype=np.float32),
                param_a=0.25,
                param_b=1.0,
                color_rgb=c1,
                metallic=0.85,
                roughness=0.25,
                bvh_radius=0.25,
            )
        )
        pole_pts = [
            tuple((origin_world + rot_fwd @ np.array([0.0, 0.04, 0.0], dtype=np.float32)).tolist()),
            tuple((origin_world + rot_fwd @ np.array([0.0, sy * 0.85, 0.0], dtype=np.float32)).tolist()),
            tuple((origin_world + rot_fwd @ np.array([0.0, sy, 0.16], dtype=np.float32)).tolist()),
        ]
        st = compute_bishop_frames(pole_pts, [0.018, 0.016, 0.014], n_stations=7, cap_roundness=0.4)
        splines.append(
            CompiledSplineBone(
                name=f"{prop.id}_pole",
                actor_id=prop.id,
                stations=st,
                smooth_k=0.01,
                csg_op=1,
                bone_role=2,
                color_rgb=c1,
                secondary_rgb=c2,
                shader="brushed_metal",
                metallic=0.88,
                roughness=0.22,
                emissive=0.0,
            )
        )
        mic_head_c = origin_world + rot_fwd @ np.array([0.0, sy + 0.01, 0.19], dtype=np.float32)
        solids.append(
            CompiledSolidPrimitive(
                name=f"{prop.id}_mic_grille",
                kind=3,
                center=mic_head_c,
                rot_fwd=rot_fwd,
                rot_inv=rot_inv,
                half_extents=np.array([0.042, 0.048, 0.042], dtype=np.float32),
                param_a=1.0,
                param_b=1.0,
                color_rgb=c2,
                metallic=0.92,
                roughness=0.22,
                bvh_radius=0.09,
            )
        )
        top_y = float(origin_world[1]) + sy

    elif prop.archetype == "piano_keyboard":
        body_c = origin_world + rot_fwd @ np.array([0.0, sy * 0.82, 0.0], dtype=np.float32)
        solids.append(
            CompiledSolidPrimitive(
                name=f"{prop.id}_cabinet",
                kind=0,
                center=body_c,
                rot_fwd=rot_fwd,
                rot_inv=rot_inv,
                half_extents=np.array([sx * 0.5, sy * 0.18, sz * 0.5], dtype=np.float32),
                param_a=0.025,
                param_b=0.0,
                color_rgb=c1,
                metallic=0.2,
                roughness=0.14,  # glossy lacquered piano!
                bvh_radius=float(np.linalg.norm([sx * 0.5, sy * 0.25, sz * 0.5])) + 0.05,
            )
        )
        # Bounded domain repetition for piano keys!
        keys_c = origin_world + rot_fwd @ np.array([0.0, sy * 0.99, sz * 0.18], dtype=np.float32)
        solids.append(
            CompiledSolidPrimitive(
                name=f"{prop.id}_keys",
                kind=1,
                center=keys_c,
                rot_fwd=rot_fwd,
                rot_inv=rot_inv,
                half_extents=np.array([0.022, 0.015, sz * 0.22], dtype=np.float32),
                param_a=0.004,
                param_b=0.0,
                repeat_bounds=(12, 0, 0),
                repeat_pitch=(0.052, 0.04, sz * 0.5),
                color_rgb=np.array([0.96, 0.96, 0.98], dtype=np.float32),
                secondary_rgb=np.array([0.08, 0.08, 0.10], dtype=np.float32),
                roughness=0.18,
                bvh_radius=sx * 0.55 + 0.05,
            )
        )
        # Legs
        for lx in (-sx * 0.42, sx * 0.42):
            leg_pts = [
                tuple((origin_world + rot_fwd @ np.array([lx, 0.0, 0.0], dtype=np.float32)).tolist()),
                tuple((origin_world + rot_fwd @ np.array([lx, sy * 0.78, 0.0], dtype=np.float32)).tolist()),
            ]
            st = compute_bishop_frames(leg_pts, [0.032, 0.042], n_stations=4, cap_roundness=0.3)
            splines.append(
                CompiledSplineBone(
                    name=f"{prop.id}_leg_{lx:.1f}",
                    actor_id=prop.id,
                    stations=st,
                    smooth_k=0.01,
                    csg_op=1,
                    bone_role=2,
                    color_rgb=c1,
                    secondary_rgb=c2,
                    shader="solid_cel",
                    metallic=0.15,
                    roughness=0.22,
                    emissive=0.0,
                )
            )
        top_y = float(origin_world[1]) + sy

    elif prop.archetype == "tree_canopy":
        trunk_pts = [
            tuple((origin_world + rot_fwd @ np.array([0.0, 0.0, 0.0], dtype=np.float32)).tolist()),
            tuple((origin_world + rot_fwd @ np.array([0.06, sy * 0.35, 0.02], dtype=np.float32)).tolist()),
            tuple((origin_world + rot_fwd @ np.array([0.0, sy * 0.65, 0.0], dtype=np.float32)).tolist()),
        ]
        st_tr = compute_bishop_frames(trunk_pts, [sx * 0.14, sx * 0.10, sx * 0.06], n_stations=7)
        splines.append(
            CompiledSplineBone(
                name=f"{prop.id}_trunk",
                actor_id=prop.id,
                stations=st_tr,
                smooth_k=0.03,
                csg_op=0,
                bone_role=2,
                color_rgb=c2,
                secondary_rgb=c2 * 0.7,
                shader="wood_grain",
                metallic=0.0,
                roughness=0.85,
                emissive=0.0,
            )
        )
        for bi, (bx, by, bz, br) in enumerate([(0.0, sy * 0.75, 0.0, sx * 0.45), (-sx * 0.25, sy * 0.62, sz * 0.18, sx * 0.32), (sx * 0.26, sy * 0.66, -sz * 0.14, sx * 0.34)]):
            solids.append(
                CompiledSolidPrimitive(
                    name=f"{prop.id}_canopy_{bi}",
                    kind=3,
                    center=origin_world + rot_fwd @ np.array([bx, by, bz], dtype=np.float32),
                    rot_fwd=rot_fwd,
                    rot_inv=rot_inv,
                    half_extents=np.array([br, br * 0.75, br], dtype=np.float32),
                    param_a=0.85,
                    param_b=0.85,
                    csg_op=0,
                    smooth_k=0.14,
                    color_rgb=c1,
                    secondary_rgb=c3,
                    shader="foliage_leaf",
                    roughness=0.75,
                    bvh_radius=br + 0.12,
                )
            )
        top_y = float(origin_world[1]) + sy

    else:
        # Generic table, boulder, street lamp, or custom_parts
        if prop.custom_parts:
            for cp in prop.custom_parts:
                pc = origin_world + rot_fwd @ np.array(cp.offset, dtype=np.float32)
                pr = rot_fwd @ rotation_matrix_xyz(*cp.rotation_deg)
                he = np.array([max(0.03, cp.size[0] * 0.5), max(0.03, cp.size[1] * 0.5), max(0.03, cp.size[2] * 0.5)], dtype=np.float32)
                solids.append(
                    CompiledSolidPrimitive(
                        name=f"{prop.id}_{cp.name}",
                        kind=1 if cp.staggered_masonry else (3 if cp.primitive == "superquadric" else 0),
                        center=pc,
                        rot_fwd=pr,
                        rot_inv=pr.T,
                        half_extents=he,
                        param_a=0.025 if cp.primitive != "superquadric" else 0.4,
                        param_b=0.4,
                        repeat_bounds=(max(0, cp.repeat_grid[0] // 2), max(0, cp.repeat_grid[1] // 2), max(0, cp.repeat_grid[2] // 2)),
                        repeat_pitch=cp.repeat_spacing,
                        color_rgb=hex_to_rgb_linear(cp.color),
                        secondary_rgb=hex_to_rgb_linear(cp.secondary_color),
                        shader=cp.shader,
                        metallic=cp.metallic,
                        roughness=cp.roughness,
                        emissive=cp.emissive,
                        bvh_radius=float(np.linalg.norm(he)) + 0.25,
                    )
                )
            top_y = float(origin_world[1]) + sy
        else:
            is_boulder = prop.archetype == "moss_boulder"
            kind_i = 3 if is_boulder else 0
            box_c = origin_world + rot_fwd @ np.array([0.0, sy * (0.42 if is_boulder else 0.5), 0.0], dtype=np.float32)
            solids.append(
                CompiledSolidPrimitive(
                    name=f"{prop.id}_body",
                    kind=kind_i,
                    center=box_c,
                    rot_fwd=rot_fwd,
                    rot_inv=rot_inv,
                    half_extents=np.array([sx * 0.5, sy * (0.42 if is_boulder else 0.5), sz * 0.5], dtype=np.float32),
                    param_a=0.65 if is_boulder else 0.025,
                    param_b=0.82 if is_boulder else 0.0,
                    color_rgb=c1,
                    secondary_rgb=c2,
                    accent_rgb=c3,
                    shader=prop.shader,
                    metallic=prop.metallic,
                    roughness=max(0.88, float(prop.roughness)) if is_boulder else prop.roughness,
                    emissive=prop.emissive,
                    bvh_radius=float(np.linalg.norm([sx * 0.5, sy * 0.5, sz * 0.5])) + 0.14,
                )
            )
            if is_boulder:
                sub_c = origin_world + rot_fwd @ np.array([sx * 0.28, sy * 0.55, sz * 0.12], dtype=np.float32)
                solids.append(
                    CompiledSolidPrimitive(
                        name=f"{prop.id}_outcrop",
                        kind=3,
                        center=sub_c,
                        rot_fwd=rot_fwd,
                        rot_inv=rot_inv,
                        half_extents=np.array([sx * 0.30, sy * 0.38, sz * 0.34], dtype=np.float32),
                        param_a=0.68,
                        param_b=0.85,
                        csg_op=0,
                        smooth_k=0.14,
                        color_rgb=c1 * 0.92,
                        secondary_rgb=c2,
                        accent_rgb=c3,
                        shader=prop.shader,
                        metallic=0.0,
                        roughness=0.92,
                        emissive=0.0,
                        bvh_radius=float(np.linalg.norm([sx * 0.35, sy * 0.42, sz * 0.38])) + 0.12,
                    )
                )
            top_y = float(origin_world[1]) + sy

    # Overall prop BVH
    pts_all = [sp.center[None, :] for sp in solids] + [bn.stations.points for bn in splines]
    if pts_all:
        merged = np.vstack(pts_all)
        bmin = np.min(merged, axis=0) - np.array([sx * 0.5, 0.1, sz * 0.5], dtype=np.float32)
        bmax = np.max(merged, axis=0) + np.array([sx * 0.5, 0.1, sz * 0.5], dtype=np.float32)
    else:
        bmin = origin_world - np.array([0.5, 0.0, 0.5], dtype=np.float32)
        bmax = origin_world + np.array([0.5, sy, 0.5], dtype=np.float32)
    bvh_c = (0.5 * (bmin + bmax)).astype(np.float32)
    bvh_r = float(np.linalg.norm(bmax - bmin) * 0.5) + 0.25

    return CompiledProp(
        id=prop.id,
        label=prop.label,
        origin_world=origin_world,
        solids=solids,
        splines=splines,
        top_surface_y=top_y,
        bmin=bmin.astype(np.float32),
        bmax=bmax.astype(np.float32),
        bvh_center=bvh_c,
        bvh_radius=bvh_r,
    )


def compile_zone2_midground(
    mid: MidgroundZoneSpec,
    zone1_top_y: float,
    *,
    seed: int = 42,
) -> List[CompiledMidgroundItem]:
    """
    Compiles coarse out-of-focus Zone 2 depth silhouettes (rooftops, city skyline,
    forest trees, concert stage crowd, mountains) positioned relative to Zone 1 ledge height.
    """
    if mid.kind == "none" or mid.density <= 0:
        return []

    rng = np.random.default_rng(seed + 911)
    n = max(1, min(28, int(mid.density)))
    sil_col = hex_to_rgb_linear(mid.silhouette_color)
    win_col = hex_to_rgb_linear(mid.window_glow_color)

    # Horizon ground elevation for midground sits below the high wall / ledge!
    base_y = zone1_top_y - float(mid.below_ledge_m) - 1.25
    items: List[CompiledMidgroundItem] = []

    for i in range(n):
        frac = (i + 0.5) / n
        z = -float(rng.uniform(mid.depth_start_m, mid.depth_end_m))
        spread_x = 4.5 + abs(z) * 0.68
        x = (frac - 0.5) * 2.0 * spread_x + float(rng.uniform(-0.6, 0.6))

        if mid.kind == "rooftop_village":
            w = float(rng.uniform(0.8, 1.55))
            d = float(rng.uniform(0.75, 1.35))
            h = float(rng.uniform(0.95, 1.75))
            roof = float(rng.uniform(0.32, 0.58))
            has_win = bool(rng.random() > 0.35)
        elif mid.kind == "city_skyline":
            w = float(rng.uniform(0.75, 1.4))
            d = float(rng.uniform(0.75, 1.3))
            h = float(rng.uniform(1.8, 4.2))
            roof = 0.0
            has_win = True
        elif mid.kind == "concert_stage_crowd":
            # Out-of-focus standing silhouette figures (rounded shoulders/heads -> roof_pitch=-1.0 sentinel)
            w = float(rng.uniform(0.32, 0.44))
            d = 0.26
            h = float(rng.uniform(0.95, 1.25))
            roof = -1.0
            has_win = False
        elif mid.kind == "forest_canopy":
            # Rounded tree crowns + trunks (roof_pitch=-2.0 sentinel so they look like trees, NOT houses!)
            w = float(rng.uniform(0.95, 1.65))
            d = w
            h = float(rng.uniform(1.8, 3.0))
            roof = -2.0
            has_win = False
        else:  # mountain_valley / interior_room_walls
            w = float(rng.uniform(1.8, 3.4))
            d = float(rng.uniform(1.2, 2.2))
            h = float(rng.uniform(1.6, 3.2))
            roof = 0.85 if mid.kind == "mountain_valley" else 0.0
            has_win = False

        shade = float(rng.uniform(0.78, 1.15))
        items.append(
            CompiledMidgroundItem(
                center=np.array([x, base_y + h * 0.5, z], dtype=np.float32),
                half_extents=np.array([w * 0.5, h * 0.5, d * 0.5], dtype=np.float32),
                roof_pitch=roof,
                color_rgb=np.clip(sil_col * shade, 0.0, 1.0).astype(np.float32),
                window_glow_rgb=win_col,
                has_windows=has_win,
            )
        )
    return items
