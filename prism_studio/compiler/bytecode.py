"""
SceneBuffers Flat Float32 Bytecode Packer for PrismStudio.

Packs a CompiledScene into aligned float32 arrays and a single RGBA32F
packed buffer texture so BOTH the ModernGL GLSL shader and the Numba/NumPy
CPU kernel evaluate the exact same mathematical scene definition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..math_ops.materials import hex_to_rgb_linear
from .layout_solver import CompiledScene

SHADER_ID_MAP: Dict[str, int] = {
    "solid": 0,
    "solid_cel": 0,
    "skin_sss": 1,
    "plush_fur": 2,
    "tabby_stripes": 3,
    "anisotropic_hair": 4,
    "brick_masonry": 5,
    "stone_voronoi": 6,
    "foliage_leaf": 7,
    "fabric_twill": 8,
    "fabric_silk": 8,
    "fabric_velvet": 9,
    "plaid_tartan": 10,
    "wood_grain": 11,
    "brushed_metal": 12,
}

EYE_STYLE_MAP: Dict[str, int] = {
    "anime_round": 0,
    "feline_slit": 1,
    "sharp_mystic": 2,
    "visor_glow": 3,
    "closed_happy": 4,
}


@dataclass
class SceneBuffers:
    """Flat, contiguous float32 buffers consumed identically by GPU & CPU backends."""

    # Table dimensions
    n_clusters: int
    n_splines: int
    n_stations: int
    n_sh: int
    n_solids: int
    n_garments: int
    n_decals: int
    n_midground: int
    n_emitters: int

    # Packed arrays (each row is multiple of 4 floats for clean RGBA32F texelFetch!)
    clusters_f32: np.ndarray     # (C, 8)  -> 2 vec4s per cluster
    splines_f32: np.ndarray      # (S, 16) -> 4 vec4s per spline
    stations_f32: np.ndarray     # (K, 16) -> 4 vec4s per station
    sh_f32: np.ndarray           # (H, 32) -> 8 vec4s per cranial SH shell
    solids_f32: np.ndarray       # (P, 20) -> 5 vec4s per solid prop
    garments_f32: np.ndarray     # (G, 12) -> 3 vec4s per garment
    decals_f32: np.ndarray       # (D, 12) -> 3 vec4s per garment decal
    midground_f32: np.ndarray    # (M, 12) -> 3 vec4s per midground item
    emitters_f32: np.ndarray     # (E, 8)  -> 2 vec4s per emitter

    # Unified RGBA32F atlas (rows of width=64 texels, 4 floats per texel -> 256 floats/row)
    packed_rgba32f: np.ndarray   # (TotalRows, 64, 4) float32
    offsets_vec4: Dict[str, int] # start texel index of each table in packed_rgba32f


def pack_scene_buffers(compiled: CompiledScene) -> SceneBuffers:
    """Packs CompiledScene into flat float32 arrays and a unified GPU RGBA32F buffer."""
    # 1. Clusters: 2 vec4 (8 floats)
    # v0: [cx, cy, cz, radius]
    # v1: [spline_start, spline_count, sh_start_or_-1, solid_start_and_counts...]
    # Store exact integers cleanly as floats (exact up to 2^24!)
    c_list = compiled.bvh_clusters
    n_c = len(c_list)
    clusters_f = np.zeros((max(1, n_c), 8), dtype=np.float32)
    c_pad = 0.22 if compiled.garments else 0.10
    for i, c in enumerate(c_list):
        clusters_f[i, 0:3] = c.center
        clusters_f[i, 3] = c.radius + c_pad
        clusters_f[i, 4] = float(c.spline_start)
        clusters_f[i, 5] = float(c.spline_count)
        clusters_f[i, 6] = float(c.sh_start if c.sh_count > 0 else -1)
        clusters_f[i, 7] = float(c.solid_start * 1024 + c.solid_count)

    # 2. Splines & Stations
    n_sp = len(compiled.all_splines)
    stations_chunks = []
    splines_f = np.zeros((max(1, n_sp), 16), dtype=np.float32)
    st_cursor = 0
    for i, sp in enumerate(compiled.all_splines):
        st = sp.stations
        k = st.points.shape[0]
        chunk = np.zeros((k, 16), dtype=np.float32)
        chunk[:, 0:3] = st.points
        chunk[:, 3] = st.radii_a
        chunk[:, 4:7] = st.tangents
        chunk[:, 7] = st.radii_b
        chunk[:, 8:11] = st.normals
        chunk[:, 11] = st.squareness
        chunk[:, 12:15] = st.binormals
        chunk[:, 15] = st.t_vals
        stations_chunks.append(chunk)

        splines_f[i, 0:3] = st.bvh_center
        g_pad = 0.0
        if 0 <= sp.garment_id < len(compiled.garments):
            g_rec = compiled.garments[sp.garment_id]
            g_pad = float(g_rec.fit_offset + g_rec.flare + g_rec.thickness + 0.08)
        splines_f[i, 3] = st.bvh_radius + g_pad
        splines_f[i, 4] = float(st_cursor)
        splines_f[i, 5] = float(k)
        splines_f[i, 6] = float(sp.smooth_k)
        splines_f[i, 7] = float(sp.csg_op)
        splines_f[i, 8:11] = sp.color_rgb
        splines_f[i, 11] = float(sp.garment_id)
        splines_f[i, 12:15] = sp.secondary_rgb
        # Pack role + shader + metallic/roughness/cap into float meta
        shader_id = SHADER_ID_MAP.get(sp.shader, 0)
        splines_f[i, 15] = float(
            sp.bone_role * 100000
            + shader_id * 1000
            + int(round(np.clip(sp.roughness, 0.05, 0.99) * 99)) * 10
            + int(round(np.clip(st.cap_roundness, 0.1, 1.0) * 9))
        )
        st_cursor += k

    stations_f = np.vstack(stations_chunks).astype(np.float32) if stations_chunks else np.zeros((1, 16), dtype=np.float32)

    # 3. Cranial SH Shells: 8 vec4 (32 floats) per shell
    n_sh = len(compiled.actors)
    sh_f = np.zeros((max(1, n_sh), 32), dtype=np.float32)
    for i, ac in enumerate(compiled.actors):
        cr = ac.cranial_shell
        sh_f[i, 0:3] = cr.center
        sh_f[i, 3] = cr.bvh_radius
        sh_f[i, 4:7] = cr.radii
        sh_f[i, 7] = cr.smooth_k
        # rot_inv rows (3 vec3) + proxy_center_local
        sh_f[i, 8:11] = cr.rot_inv[0]
        sh_f[i, 11] = cr.proxy_center_local[0]
        sh_f[i, 12:15] = cr.rot_inv[1]
        sh_f[i, 15] = cr.proxy_center_local[1]
        sh_f[i, 16:19] = cr.rot_inv[2]
        sh_f[i, 19] = cr.proxy_center_local[2]
        # 16 SH coeffs packed as 4 vec4? We need 16 floats + eye params!
        # Let's expand sh_f to 48 floats (12 vec4s) for clean alignment with zero packing compromise!

    sh_f = np.zeros((max(1, n_sh), 48), dtype=np.float32)
    for i, ac in enumerate(compiled.actors):
        cr = ac.cranial_shell
        sh_f[i, 0:3] = cr.center
        sh_f[i, 3] = cr.bvh_radius
        sh_f[i, 4:7] = cr.radii
        sh_f[i, 7] = cr.smooth_k
        sh_f[i, 8:11] = cr.rot_inv[0]
        sh_f[i, 11] = cr.proxy_center_local[0]
        sh_f[i, 12:15] = cr.rot_inv[1]
        sh_f[i, 15] = cr.proxy_center_local[1]
        sh_f[i, 16:19] = cr.rot_inv[2]
        sh_f[i, 19] = cr.proxy_center_local[2]
        sh_f[i, 20:36] = cr.sh_coeffs
        sh_f[i, 36:39] = cr.skin_rgb
        sh_f[i, 39] = float(cr.proxy_weight)
        sh_f[i, 40:43] = cr.iris_rgb
        sh_f[i, 43] = float(cr.eye_scale)
        sh_f[i, 44:47] = cr.lash_rgb
        eye_id = EYE_STYLE_MAP.get(cr.eye_style, 0)
        sh_f[i, 47] = float(
            eye_id * 10000
            + (1 if cr.has_cat_nose else 0) * 1000
            + (1 if cr.fangs else 0) * 100
            + int(round(np.clip(cr.mouth_open, 0.0, 0.99) * 99))
        )

    # 4. Solids: 5 vec4 (20 floats) per solid primitive
    n_so = len(compiled.all_solids)
    solids_f = np.zeros((max(1, n_so), 20), dtype=np.float32)
    for i, so in enumerate(compiled.all_solids):
        solids_f[i, 0:3] = so.center
        solids_f[i, 3] = so.bvh_radius
        solids_f[i, 4:7] = so.half_extents
        solids_f[i, 7] = float(so.kind)
        solids_f[i, 8:11] = so.rot_inv[0]
        solids_f[i, 11] = float(so.param_a)
        solids_f[i, 12:15] = so.rot_inv[1]
        solids_f[i, 15] = float(so.param_b)
        solids_f[i, 16:19] = so.rot_inv[2]
        # Repeat & color packed in second solid table or expand solids_f to 32 floats (8 vec4s)!
    solids_f = np.zeros((max(1, n_so), 32), dtype=np.float32)
    for i, so in enumerate(compiled.all_solids):
        solids_f[i, 0:3] = so.center
        solids_f[i, 3] = so.bvh_radius
        solids_f[i, 4:7] = so.half_extents
        solids_f[i, 7] = float(so.kind)
        solids_f[i, 8:11] = so.rot_inv[0]
        solids_f[i, 11] = float(so.param_a)
        solids_f[i, 12:15] = so.rot_inv[1]
        solids_f[i, 15] = float(so.param_b)
        solids_f[i, 16:19] = so.rot_inv[2]
        solids_f[i, 19] = float(so.smooth_k)
        solids_f[i, 20:23] = np.array(so.repeat_bounds, dtype=np.float32)
        solids_f[i, 23] = float(so.emissive)
        solids_f[i, 24:27] = np.array(so.repeat_pitch, dtype=np.float32)
        solids_f[i, 27] = float(SHADER_ID_MAP.get(so.shader, 0) * 1000 + int(round(np.clip(so.roughness, 0.05, 0.99) * 99)) * 10 + so.csg_op)
        solids_f[i, 28:31] = so.color_rgb
        solids_f[i, 31] = float(so.metallic)
    # Wait: let's also keep secondary color on solids in 36 floats (9 vec4) -> brick mortar needs secondary_rgb!
    solids_f_ext = np.zeros((max(1, n_so), 36), dtype=np.float32)
    solids_f_ext[:, :32] = solids_f
    for i, so in enumerate(compiled.all_solids):
        solids_f_ext[i, 32:35] = so.secondary_rgb
        solids_f_ext[i, 35] = float(so.accent_rgb[0])

    # 5. Garments & Decals
    n_g = len(compiled.garments)
    garments_f = np.zeros((max(1, n_g), 16), dtype=np.float32)
    decals_rows = []
    decal_cursor = 0
    for i, g in enumerate(compiled.garments):
        garments_f[i, 0] = g.fit_offset
        garments_f[i, 1] = g.thickness
        garments_f[i, 2] = g.flare
        garments_f[i, 3] = float(g.pleat_count)
        garments_f[i, 4] = g.pleat_amp
        garments_f[i, 5] = g.ruffle_amp
        garments_f[i, 6] = g.fold_amp
        garments_f[i, 7] = g.open_front_rad
        garments_f[i, 8:11] = g.color_rgb
        garments_f[i, 11] = float(SHADER_ID_MAP.get(g.shader, 8) * 1000 + int(round(np.clip(g.roughness, 0.05, 0.99) * 99)))
        garments_f[i, 12] = g.t_min
        garments_f[i, 13] = g.t_max
        garments_f[i, 14] = float(decal_cursor)
        garments_f[i, 15] = float(len(g.decals))
        for d in g.decals:
            row = np.zeros(12, dtype=np.float32)
            row[0] = float(d.kind)
            row[1] = d.uv_t
            row[2] = d.uv_theta
            row[3] = d.size
            row[4:7] = d.color_rgb
            row[7] = 1.0
            row[8:11] = d.secondary_rgb
            decals_rows.append(row)
            decal_cursor += 1
    decals_f = np.vstack(decals_rows).astype(np.float32) if decals_rows else np.zeros((1, 12), dtype=np.float32)

    # 6. Midground items (Zone 2): 3 vec4 (12 floats)
    n_m = len(compiled.midground)
    mid_f = np.zeros((max(1, n_m), 12), dtype=np.float32)
    for i, m in enumerate(compiled.midground):
        mid_f[i, 0:3] = m.center
        mid_f[i, 3] = m.roof_pitch
        mid_f[i, 4:7] = m.half_extents
        mid_f[i, 7] = 1.0 if m.has_windows else 0.0
        mid_f[i, 8:11] = m.color_rgb
        mid_f[i, 11] = float(
            int(round(np.clip(m.window_glow_rgb[0], 0.0, 1.0) * 255)) * 65536
            + int(round(np.clip(m.window_glow_rgb[1], 0.0, 1.0) * 255)) * 256
            + int(round(np.clip(m.window_glow_rgb[2], 0.0, 1.0) * 255))
        )

    # 7. Emitters (Zone 3): 2 vec4 (8 floats)
    em_list = compiled.dna.zone3_atmosphere.emitters
    n_e = len(em_list)
    em_f = np.zeros((max(1, n_e), 8), dtype=np.float32)
    for i, em in enumerate(em_list):
        d_vec = np.array(em.direction_or_pos, dtype=np.float32)
        d_vec /= max(1e-6, float(np.linalg.norm(d_vec)))
        em_f[i, 0:3] = d_vec
        em_f[i, 3] = float(em.radius)
        em_f[i, 4:7] = hex_to_rgb_linear(em.color) * float(em.intensity)
        em_f[i, 7] = float(em.god_ray_strength)

    # Pack all tables sequentially into flat vec4 stream (width=256 texels per row for 2D RGBA32F sampler!)
    texels = []
    offsets: Dict[str, int] = {}
    cursor = 0
    for name, arr in (
        ("clusters", clusters_f),
        ("splines", splines_f),
        ("stations", stations_f),
        ("sh", sh_f),
        ("solids", solids_f_ext),
        ("garments", garments_f),
        ("decals", decals_f),
        ("midground", mid_f),
        ("emitters", em_f),
    ):
        offsets[name] = cursor
        vec4_arr = arr.reshape(-1, 4)
        texels.append(vec4_arr)
        cursor += vec4_arr.shape[0]

    flat_vec4 = np.vstack(texels).astype(np.float32)
    tex_w = 256
    tex_h = max(1, math.ceil(flat_vec4.shape[0] / tex_w))
    padded = np.zeros((tex_h * tex_w, 4), dtype=np.float32)
    padded[: flat_vec4.shape[0]] = flat_vec4
    packed_tex = padded.reshape(tex_h, tex_w, 4)

    return SceneBuffers(
        n_clusters=n_c,
        n_splines=n_sp,
        n_stations=st_cursor,
        n_sh=n_sh,
        n_solids=n_so,
        n_garments=n_g,
        n_decals=decal_cursor,
        n_midground=n_m,
        n_emitters=n_e,
        clusters_f32=clusters_f,
        splines_f32=splines_f,
        stations_f32=stations_f,
        sh_f32=sh_f,
        solids_f32=solids_f_ext,
        garments_f32=garments_f,
        decals_f32=decals_f,
        midground_f32=mid_f,
        emitters_f32=em_f,
        packed_rgba32f=packed_tex,
        offsets_vec4=offsets,
    )
