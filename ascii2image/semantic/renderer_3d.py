"""
Vectorized 3D BVH-Accelerated Sphere Tracer (Track 1).

Renders multi-object SceneGraphs into full G-Buffers (color, depth, normals,
emissive, hit mask) using:
- Ray-AABB slab culling per entity (only evaluates entities a ray enters)
- Active-ray compaction inside the sphere-tracing loop
- Central-difference SDF surface normals + procedural bump perturbation
- SDF Ambient Occlusion (AO), soft directional shadows, Fresnel rim light,
  Blinn-Phong metallic specular, and optional anime/cel-shading quantization
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .math_core import ray_aabb_intersect
from .patterns import evaluate_pattern, hex_to_rgb01
from .solver import CompiledEntity, ResolvedScene


@dataclass
class GBuffer:
    width: int
    height: int
    color_rgb: np.ndarray          # (H, W, 3) float32 linear/HDR
    emissive_rgb: np.ndarray       # (H, W, 3) float32 HDR glow buffer
    depth_z: np.ndarray            # (H, W) float32 ray distance (inf if miss)
    hit_mask: np.ndarray           # (H, W) bool
    focus_depth: float             # camera distance to scene focal point
    cam_pos: np.ndarray            # (3,)
    cam_target: np.ndarray         # (3,)
    cam_right: np.ndarray          # (3,)
    cam_up: np.ndarray             # (3,)
    cam_fwd: np.ndarray            # (3,)


def _build_camera_rays(
    resolved: ResolvedScene,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    cam = resolved.scene.camera
    bmin, bmax = resolved.scene_bmin, resolved.scene_bmax
    center = 0.5 * (bmin + bmax)
    ext = np.maximum(bmax - bmin, 0.6)
    radius = float(np.linalg.norm(ext) * 0.5)

    az = math.radians(cam.azimuth_deg)
    el = math.radians(cam.elevation_deg)
    fov = math.radians(cam.fov_deg)

    # Auto-frame camera distance so entire multi-object scene fills ~75% of frame cleanly
    dist = max(2.2, (radius / max(0.18, math.sin(fov * 0.5))) * 1.18 * float(cam.distance_scale))
    cam_dir = np.array(
        [math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)],
        dtype=np.float32,
    )
    cam_pos = (center + cam_dir * dist).astype(np.float32)
    cam_fwd = -cam_dir
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    cam_right = np.cross(cam_fwd, world_up)
    r_norm = float(np.linalg.norm(cam_right))
    if r_norm < 1e-5:
        cam_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        cam_right /= r_norm
    cam_up = np.cross(cam_right, cam_fwd)

    # Apply camera Dutch roll if specified
    if abs(cam.roll_deg) > 1e-3:
        rr = math.radians(cam.roll_deg)
        cr, sr = math.cos(rr), math.sin(rr)
        new_r = cr * cam_right + sr * cam_up
        new_u = -sr * cam_right + cr * cam_up
        cam_right, cam_up = new_r.astype(np.float32), new_u.astype(np.float32)

    aspect = width / max(1.0, float(height))
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32) * aspect
    ys = np.linspace(1.0, -1.0, height, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    gx_f = gx.reshape(-1)
    gy_f = gy.reshape(-1)
    n_rays = gx_f.shape[0]

    if cam.projection in ("isometric", "orthographic"):
        ortho_scale = radius * 1.28 * float(cam.distance_scale)
        ro = (
            cam_pos[None, :]
            + gx_f[:, None] * (cam_right[None, :] * ortho_scale)
            + gy_f[:, None] * (cam_up[None, :] * ortho_scale)
        ).astype(np.float32)
        rd = np.broadcast_to(cam_fwd[None, :], (n_rays, 3)).copy()
    else:
        tan_half = math.tan(fov * 0.5)
        rd = (
            cam_fwd[None, :]
            + (gx_f * tan_half)[:, None] * cam_right[None, :]
            + (gy_f * tan_half)[:, None] * cam_up[None, :]
        )
        rd /= np.maximum(np.linalg.norm(rd, axis=1, keepdims=True), 1e-7)
        rd = rd.astype(np.float32)
        ro = np.broadcast_to(cam_pos[None, :], (n_rays, 3)).copy()

    focus_depth = max(0.5, dist + float(cam.focus_offset))
    return ro, rd, cam_pos, center, cam_right, cam_up, cam_fwd, focus_depth


def _eval_entities_sdf(
    entities: list[CompiledEntity],
    p: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluates scene-wide SDF using AABB bounding checks to skip distant entities."""
    n = p.shape[0]
    d_min = np.full(n, 1e4, dtype=np.float32)
    m_min = np.zeros(n, dtype=np.int32)
    if n == 0:
        return d_min, m_min

    for ent in entities:
        # Distance to entity AABB (0 inside box, >0 outside)
        q_min = np.maximum(ent.bmin[None, :] - p, 0.0)
        q_max = np.maximum(p - ent.bmax[None, :], 0.0)
        box_dist = np.linalg.norm(q_min + q_max, axis=1)
        # Only evaluate exact SDF where box_dist could beat current d_min
        active = box_dist < (d_min + 0.02)
        if not np.any(active):
            # Still tighten upper-bound conservative step with box_dist + small margin
            d_min = np.minimum(d_min, box_dist + 0.02)
            continue
        sub_d, sub_m = ent.eval_sdf_and_mat(p[active])
        act_idx = np.nonzero(active)[0]
        better = sub_d < d_min[act_idx]
        win_idx = act_idx[better]
        d_min[win_idx] = sub_d[better]
        m_min[win_idx] = sub_m[better]
        # For non-active rays, box_dist is a safe conservative lower bound
        d_min = np.where(~active, np.minimum(d_min, box_dist + 0.02), d_min)

    return d_min, m_min


def _eval_scene_sdf_only(entities: list[CompiledEntity], p: np.ndarray) -> np.ndarray:
    d, _ = _eval_entities_sdf(entities, p)
    return d


def render_scene_gbuffer(
    resolved: ResolvedScene,
    width: int = 640,
    height: int = 480,
    max_steps: int = 72,
) -> GBuffer:
    """
    Renders all 3D entities + perspective environment surface into a GBuffer.
    """
    ro, rd, cam_pos, cam_target, cam_right, cam_up, cam_fwd, focus_depth = _build_camera_rays(
        resolved, width, height
    )
    n_rays = width * height
    entities = resolved.entities
    env = resolved.scene.environment
    lit = resolved.scene.lighting

    hit_mask = np.zeros(n_rays, dtype=bool)
    depth_z = np.full(n_rays, np.inf, dtype=np.float32)
    hit_pos = np.zeros((n_rays, 3), dtype=np.float32)
    hit_normal = np.zeros((n_rays, 3), dtype=np.float32)
    hit_mat = np.zeros(n_rays, dtype=np.int32)
    is_env_hit = np.zeros(n_rays, dtype=bool)

    # 1. Analytical Ray-Plane Intersection for Environment Surface (fast & exact!)
    if env.enabled and env.kind != "none":
        tilt = math.radians(env.tilt_deg)
        plane_n = np.array([0.0, math.cos(tilt), math.sin(tilt)], dtype=np.float32)
        plane_p0 = np.array([0.0, float(env.y_level), 0.0], dtype=np.float32)
        denom = rd @ plane_n
        valid_p = denom < -1e-4
        if np.any(valid_p):
            t_plane = ((plane_p0[None, :] - ro[valid_p]) @ plane_n) / denom[valid_p]
            p_world = ro[valid_p] + rd[valid_p] * t_plane[:, None]

            # Bound ledge/tabletop footprint if requested
            if env.kind in ("tiled_ledge", "tabletop"):
                ext = resolved.scene_bmax - resolved.scene_bmin
                cx = 0.5 * (resolved.scene_bmin[0] + resolved.scene_bmax[0])
                cz = 0.5 * (resolved.scene_bmin[2] + resolved.scene_bmax[2])
                span_x = max(3.6, float(ext[0]) * 1.45)
                span_z = max(2.2, float(ext[2]) * 1.25) if env.kind == "tiled_ledge" else max(3.2, float(ext[2]) * 1.45)
                in_bounds = (
                    (t_plane > 0.1)
                    & (t_plane < 45.0)
                    & (np.abs(p_world[:, 0] - cx) <= span_x)
                    & (np.abs(p_world[:, 2] - cz) <= span_z)
                )
            else:
                in_bounds = (t_plane > 0.1) & (t_plane < 55.0)

            idx_p = np.nonzero(valid_p)[0][in_bounds]
            depth_z[idx_p] = t_plane[in_bounds]
            hit_pos[idx_p] = p_world[in_bounds]
            hit_normal[idx_p] = plane_n
            hit_mask[idx_p] = True
            is_env_hit[idx_p] = True

    # 2. BVH Slab Culling for 3D Entities (skip rays that miss the entire entity scene!)
    if entities:
        scene_hit, t_enter, t_exit = ray_aabb_intersect(
            ro, rd, resolved.scene_bmin - 0.15, resolved.scene_bmax + 0.15
        )
        # Active ray compaction: only sphere-trace rays entering the scene AABB
        active_idx = np.nonzero(scene_hit)[0]
        if active_idx.size > 0:
            t_cur = t_enter[active_idx].copy()
            t_max = np.minimum(t_exit[active_idx], depth_z[active_idx] + 0.25)
            eps_hit = 0.0045

            for _ in range(max_steps):
                if active_idx.size == 0:
                    break
                pts = ro[active_idx] + rd[active_idx] * t_cur[:, None]
                d, m = _eval_entities_sdf(entities, pts)

                hit_now = d < eps_hit
                if np.any(hit_now):
                    win_rays = active_idx[hit_now]
                    win_t = t_cur[hit_now]
                    closer = win_t < depth_z[win_rays]
                    upd = win_rays[closer]
                    depth_z[upd] = win_t[closer]
                    hit_pos[upd] = pts[hit_now][closer]
                    hit_mat[upd] = m[hit_now][closer]
                    hit_mask[upd] = True
                    is_env_hit[upd] = False

                t_cur = t_cur + np.maximum(d * 0.92, 0.004)
                keep = (~hit_now) & (t_cur < t_max)
                active_idx = active_idx[keep]
                t_cur = t_cur[keep]
                t_max = t_max[keep]

        # 3. Compute Central-Difference Surface Normals ONLY on Entity Hit Pixels
        ent_hit_idx = np.nonzero(hit_mask & (~is_env_hit))[0]
        if ent_hit_idx.size > 0:
            p_h = hit_pos[ent_hit_idx]
            eps = 0.0035
            ex = np.array([eps, 0.0, 0.0], dtype=np.float32)
            ey = np.array([0.0, eps, 0.0], dtype=np.float32)
            ez = np.array([0.0, 0.0, eps], dtype=np.float32)
            nx = _eval_scene_sdf_only(entities, p_h + ex) - _eval_scene_sdf_only(entities, p_h - ex)
            ny = _eval_scene_sdf_only(entities, p_h + ey) - _eval_scene_sdf_only(entities, p_h - ey)
            nz = _eval_scene_sdf_only(entities, p_h + ez) - _eval_scene_sdf_only(entities, p_h - ez)
            nrm = np.stack([nx, ny, nz], axis=1)
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-7)
            hit_normal[ent_hit_idx] = nrm.astype(np.float32)

    # 4. Physically Inspired Shading + Procedural Patterns + Soft Shadows + AO + Rim Light
    color_rgb = np.zeros((n_rays, 3), dtype=np.float32)
    emissive_rgb = np.zeros((n_rays, 3), dtype=np.float32)

    all_hits = np.nonzero(hit_mask)[0]
    if all_hits.size > 0:
        p_all = hit_pos[all_hits]
        n_all = hit_normal[all_hits]
        v_all = -rd[all_hits]
        m_all = hit_mat[all_hits]
        env_flags = is_env_hit[all_hits]

        albedo = np.zeros((all_hits.size, 3), dtype=np.float32)
        bump = np.zeros(all_hits.size, dtype=np.float32)
        metallic = np.zeros(all_hits.size, dtype=np.float32)
        roughness = np.full(all_hits.size, 0.55, dtype=np.float32)
        emissive = np.zeros(all_hits.size, dtype=np.float32)

        # Environment pixels
        if np.any(env_flags):
            e_sub = np.nonzero(env_flags)[0]
            e_alb, e_bmp = evaluate_pattern(
                env.pattern,
                p_all[e_sub],
                n_all[e_sub],
                env.primary_color,
                env.secondary_color,
                "#334155",
                scale=env.pattern_scale,
            )
            albedo[e_sub] = e_alb
            bump[e_sub] = e_bmp
            metallic[e_sub] = env.metallic
            roughness[e_sub] = env.roughness

        # Entity pixels grouped by material_id (vectorized per material!)
        unique_mats = np.unique(m_all[~env_flags])
        for mid in unique_mats:
            sub = np.nonzero((~env_flags) & (m_all == mid))[0]
            mat = resolved.materials[int(mid)] if int(mid) < len(resolved.materials) else resolved.materials[0]
            m_alb, m_bmp = evaluate_pattern(
                mat.pattern,
                p_all[sub],
                n_all[sub],
                mat.color,
                mat.secondary_color,
                mat.accent_color,
                scale=mat.pattern_scale,
            )
            albedo[sub] = m_alb
            bump[sub] = m_bmp
            metallic[sub] = mat.metallic
            roughness[sub] = mat.roughness
            emissive[sub] = mat.emissive

        # Light vectors
        key_dir = np.array(lit.key_dir, dtype=np.float32)
        key_dir /= max(1e-6, float(np.linalg.norm(key_dir)))
        key_col = hex_to_rgb01(lit.key_color)
        amb_col = hex_to_rgb01(lit.ambient_color)
        rim_col = hex_to_rgb01(lit.rim_color)

        # Perturb diffuse term smoothly with procedural bump height
        ndotl = np.clip(n_all @ key_dir + bump * 0.18, 0.0, 1.0)

        # Soft Shadow March (8 fast steps toward key light)
        shadow = np.ones(all_hits.size, dtype=np.float32)
        if entities:
            sh_o = p_all + n_all * 0.025
            t_sh = np.full(all_hits.size, 0.05, dtype=np.float32)
            for _ in range(9):
                sp = sh_o + key_dir[None, :] * t_sh[:, None]
                sd = _eval_scene_sdf_only(entities, sp)
                shadow = np.minimum(shadow, np.clip(6.0 * sd / np.maximum(t_sh, 0.05), 0.15, 1.0))
                t_sh = t_sh + np.clip(sd, 0.04, 0.45)

        # Fast 3-tap SDF Ambient Occlusion
        ao = np.ones(all_hits.size, dtype=np.float32)
        if entities:
            occ = np.zeros(all_hits.size, dtype=np.float32)
            for step_i, h_ao in enumerate((0.08, 0.22, 0.45)):
                d_ao = _eval_scene_sdf_only(entities, p_all + n_all * h_ao)
                occ += (h_ao - d_ao) * (0.55 ** step_i)
            ao = np.clip(1.0 - occ * 1.45, 0.25, 1.0)

        if lit.cel_shading:
            # Crisp anime/illustration cel quantization with smooth penumbra band
            lit_factor = ndotl * shadow
            diffuse_band = np.where(lit_factor > 0.55, 1.0, np.where(lit_factor > 0.20, 0.62, 0.30)).astype(np.float32)
        else:
            diffuse_band = ndotl * shadow

        # Half-vector Blinn-Phong specular scaled by roughness & metallic
        half_v = key_dir[None, :] + v_all
        half_v /= np.maximum(np.linalg.norm(half_v, axis=1, keepdims=True), 1e-7)
        ndoth = np.clip(np.sum(n_all * half_v, axis=1), 0.0, 1.0)
        shininess = np.clip(2.0 / np.maximum(roughness * roughness, 0.02), 4.0, 140.0)
        spec_pow = np.power(ndoth, shininess) * shadow * (0.25 + 0.85 * metallic)
        spec_tint = albedo * metallic[:, None] + np.ones_like(albedo) * (1.0 - metallic[:, None])

        # Fresnel Rim Lighting (critical for glowing evening/studio separation!)
        ndotv = np.clip(np.sum(n_all * v_all, axis=1), 0.0, 1.0)
        fresnel = np.power(1.0 - ndotv, 3.0) * float(lit.rim_strength) * (0.35 + 0.65 * ao)

        # Secondary cool fill from opposite azimuth
        fill_dir = np.array([-key_dir[0], 0.35, -key_dir[2]], dtype=np.float32)
        fill_dir /= max(1e-6, float(np.linalg.norm(fill_dir)))
        fill_diff = np.clip(n_all @ fill_dir, 0.0, 1.0) * 0.28 * ao

        # Distance fade on infinite ground plane so horizon blends softly into background
        env_fade = np.ones(all_hits.size, dtype=np.float32)
        if env.enabled and env.kind == "ground_plane":
            d_env = depth_z[all_hits]
            env_fade = np.where(env_flags, np.clip(1.0 - (d_env - 8.0) / 28.0, 0.0, 1.0), 1.0)

        shaded = (
            albedo * (amb_col[None, :] * 0.45 * ao[:, None] + key_col[None, :] * diffuse_band[:, None] * 1.45 + fill_diff[:, None] * amb_col[None, :])
            + spec_tint * key_col[None, :] * spec_pow[:, None] * 1.35
            + rim_col[None, :] * fresnel[:, None] * 0.95
            + albedo * emissive[:, None] * 1.85
        )
        # Apply subtle fade into miss for far ground edges
        if np.any(env_fade < 0.02):
            drop = all_hits[env_fade < 0.02]
            hit_mask[drop] = False

        color_rgb[all_hits] = shaded * env_fade[:, None]
        emissive_rgb[all_hits] = albedo * emissive[:, None] * 1.35 + rim_col[None, :] * np.maximum(fresnel - 0.55, 0.0)[:, None]

    return GBuffer(
        width=width,
        height=height,
        color_rgb=color_rgb.reshape(height, width, 3),
        emissive_rgb=emissive_rgb.reshape(height, width, 3),
        depth_z=depth_z.reshape(height, width),
        hit_mask=hit_mask.reshape(height, width),
        focus_depth=focus_depth,
        cam_pos=cam_pos,
        cam_target=cam_target,
        cam_right=cam_right,
        cam_up=cam_up,
        cam_fwd=cam_fwd,
    )
