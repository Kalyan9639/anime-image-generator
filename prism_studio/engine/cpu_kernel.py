"""
Speed-First Numba / Vectorized NumPy CPU Fallback Raymarcher for PrismStudio.

Executes Pass 1 (G-Buffer raymarcher + 2D UV eyes/decals + triangulated soft
shadows + 5-tap AO + anime_npr/stylized_pbr shading) in cache-friendly 256x256
strips (<20 MB RAM) using Numba @njit(parallel=True) when available, with a
pure-NumPy fallback so the engine runs cleanly on any environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..compiler.bytecode import SceneBuffers
from ..compiler.layout_solver import CompiledScene
from ..math_ops.decals_2d import evaluate_2d_eye_uv
from ..math_ops.materials import hex_to_rgb_linear
from ..math_ops.primitives import (
    sd_cloth_shell,
    sd_rounded_box,
    sd_sh_shell,
    sd_swept_superspline,
    sd_torus,
    smax_cubic,
    smin_cubic_mat,
)
from .gpu_moderngl import compute_camera_basis

try:
    from numba import njit, prange
    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False
    njit = None
    prange = range


@dataclass
class CPUGBuffer:
    width: int
    height: int
    hdr_rim: np.ndarray     # (H, W, 4) float32
    nrm_mat: np.ndarray     # (H, W, 4) float32
    depth_hit: np.ndarray   # (H, W, 2) float32
    focus_depth: float
    sun_uv: Tuple[float, float]
    sun_col: np.ndarray
    godray_strength: float


# ── Numba Accelerated Scalar Kernel (All 4 Logical Cores) ────────────────────
if HAS_NUMBA:

    @njit(fastmath=True, inline="always")
    def _nb_hash2(x: float, y: float) -> float:
        h = math.sin(x * 127.1 + y * 311.7) * 43758.5453123
        return h - math.floor(h)

    @njit(fastmath=True, inline="always")
    def _nb_noise2(u: float, v: float) -> float:
        ix = math.floor(u)
        iy = math.floor(v)
        fx = u - ix
        fy = v - iy
        ux = fx * fx * (3.0 - 2.0 * fx)
        uy = fy * fy * (3.0 - 2.0 * fy)
        a = _nb_hash2(ix, iy)
        b = _nb_hash2(ix + 1.0, iy)
        c = _nb_hash2(ix, iy + 1.0)
        d = _nb_hash2(ix + 1.0, iy + 1.0)
        return (a * (1.0 - ux) + b * ux) * (1.0 - uy) + (c * (1.0 - ux) + d * ux) * uy

    @njit(fastmath=True, inline="always")
    def _nb_smin(a: float, b: float, k: float) -> Tuple[float, float]:
        if k <= 1e-5:
            return (a, 0.0) if a < b else (b, 1.0)
        h = max(k - abs(a - b), 0.0) / k
        d = min(a, b) - (1.0 / 6.0) * k * h * h * h
        m = 0.5 * h * h
        eta = m if a < b else (1.0 - m)
        return d, eta

    @njit(fastmath=True, inline="always")
    def _nb_spline_sdf(
        px: float, py: float, pz: float,
        sp_idx: int,
        splines_f: np.ndarray,
        stations_f: np.ndarray,
    ) -> Tuple[float, float, float, float, float, float]:
        st_start = int(splines_f[sp_idx, 4] + 0.5)
        st_cnt = int(splines_f[sp_idx, 5] + 0.5)
        min_d2 = 1e12
        best_seg = 0
        best_s = 0.0
        bx_c = 0.0
        by_c = 0.0
        bz_c = 0.0

        for i in range(st_cnt - 1):
            a_x = stations_f[st_start + i, 0]
            a_y = stations_f[st_start + i, 1]
            a_z = stations_f[st_start + i, 2]
            sx = stations_f[st_start + i + 1, 0] - a_x
            sy = stations_f[st_start + i + 1, 1] - a_y
            sz = stations_f[st_start + i + 1, 2] - a_z
            l2 = max(sx * sx + sy * sy + sz * sz, 1e-10)
            s = min(1.0, max(0.0, ((px - a_x) * sx + (py - a_y) * sy + (pz - a_z) * sz) / l2))
            cx = a_x + s * sx
            cy = a_y + s * sy
            cz = a_z + s * sz
            dx, dy, dz = px - cx, py - cy, pz - cz
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < min_d2:
                min_d2 = d2
                best_seg = i
                best_s = s
                bx_c, by_c, bz_c = cx, cy, cz

        i0 = st_start + best_seg
        i1 = i0 + 1
        w1 = best_s
        w0 = 1.0 - w1

        tx = stations_f[i0, 4] * w0 + stations_f[i1, 4] * w1
        ty = stations_f[i0, 5] * w0 + stations_f[i1, 5] * w1
        tz = stations_f[i0, 6] * w0 + stations_f[i1, 6] * w1
        tn = math.sqrt(max(tx * tx + ty * ty + tz * tz, 1e-12))
        tx, ty, tz = tx / tn, ty / tn, tz / tn

        nx = stations_f[i0, 8] * w0 + stations_f[i1, 8] * w1
        ny = stations_f[i0, 9] * w0 + stations_f[i1, 9] * w1
        nz = stations_f[i0, 10] * w0 + stations_f[i1, 10] * w1
        ndott = nx * tx + ny * ty + nz * tz
        nx -= ndott * tx
        ny -= ndott * ty
        nz -= ndott * tz
        nn = math.sqrt(max(nx * nx + ny * ny + nz * nz, 1e-12))
        nx, ny, nz = nx / nn, ny / nn, nz / nn

        bx = ty * nz - tz * ny
        by = tz * nx - tx * nz
        bz = tx * ny - ty * nx

        ra = max(0.004, stations_f[i0, 3] * w0 + stations_f[i1, 3] * w1)
        rb = max(0.004, stations_f[i0, 7] * w0 + stations_f[i1, 7] * w1)
        n_exp = min(4.0, max(0.45, stations_f[i0, 11] * w0 + stations_f[i1, 11] * w1))
        t_star = (float(best_seg) + best_s) / max(1.0, float(st_cnt - 1))

        dx, dy, dz = px - bx_c, py - by_c, pz - bz_c
        u = dx * nx + dy * ny + dz * nz
        v = dx * bx + dy * by + dz * bz

        w_ax = 0.0
        if best_seg == 0 and best_s <= 1e-4:
            w_ax = max(-(dx * stations_f[st_start, 4] + dy * stations_f[st_start, 5] + dz * stations_f[st_start, 6]), 0.0)
        elif best_seg == st_cnt - 2 and best_s >= 1.0 - 1e-4:
            e_i = st_start + st_cnt - 1
            w_ax = max(dx * stations_f[e_i, 4] + dy * stations_f[e_i, 5] + dz * stations_f[e_i, 6], 0.0)

        ru = abs(u) / ra
        rv = abs(v) / rb
        min_ab = min(ra, rb)
        max_r = max(ru, rv)
        if max_r < 1e-6:
            d_cross = -min_ab
        else:
            norm_c = max_r * ((ru / max_r) ** n_exp + (rv / max_r) ** n_exp) ** (1.0 / n_exp)
            d_cross = min_ab * (norm_c - 1.0)

        meta = int(splines_f[sp_idx, 15] + 0.5)
        cap_r = max(0.25, float(meta % 10) / 9.0)
        d_raw = d_cross
        if w_ax > 0.0:
            rw = w_ax / (min_ab * cap_r)
            m3 = max(max_r, rw)
            n3 = m3 * ((ru / m3) ** n_exp + (rv / m3) ** n_exp + (rw / m3) ** n_exp) ** (1.0 / n_exp)
            d_raw = min_ab * (n3 - 1.0)

        theta = math.atan2(v, u)
        return d_raw * 0.72, t_star, theta, w_ax, tx, ty, tz

    @njit(fastmath=True, inline="always")
    def _nb_cloth_sdf(d_body: float, t_uv: float, th_uv: float, w_ax: float, g_idx: int, joint_b: float, garments_f: np.ndarray) -> float:
        fit = garments_f[g_idx, 0]
        flare = garments_f[g_idx, 2]
        p_cnt = garments_f[g_idx, 3]
        p_amp = garments_f[g_idx, 4]
        r_amp = garments_f[g_idx, 5]
        f_amp = garments_f[g_idx, 6]
        open_r = garments_f[g_idx, 7]
        t_min = garments_f[g_idx, 12]
        t_max = garments_f[g_idx, 13]

        t_past = max(t_uv - 0.35, 0.0)
        flare_d = flare * (t_past ** 1.35)
        pleat_d = 0.0
        if p_cnt > 0.5:
            pleat_d = min(0.009, p_amp) * math.sin(th_uv * p_cnt) * t_past
        ruff_d = 0.0
        if r_amp > 0.0005:
            ruff_d = min(0.009, r_amp) * math.sin(12.0 * th_uv) * max(1.0 - abs(t_uv - t_max) * 4.5, 0.0)
        fold_d = min(0.005, f_amp) * joint_b * math.exp(-4.0 * abs(t_uv - 0.5)) * math.sin(3.0 * th_uv + 4.0 * t_uv)
        lo_t = min(1.0, max(0.0, (t_uv - (t_min - 0.03)) / 0.15)) if t_min > 0.04 else 1.0
        hi_t = min(1.0, max(0.0, ((t_max + 0.03) - t_uv) / 0.15))
        hem_taper = (lo_t * lo_t * (3.0 - 2.0 * lo_t)) * (hi_t * hi_t * (3.0 - 2.0 * hi_t))
        delta_eff = max(0.004, fit * (0.45 + 0.55 * hem_taper) + flare_d + pleat_d + ruff_d + fold_d)
        d_sh = d_body - delta_eff
        w_mask = w_ax * 0.78 if (t_uv > 0.5 or t_min > 0.02) else 0.0
        d_trim = w_mask + max(t_min - t_uv, t_uv - t_max) * 0.45
        if open_r > 0.02:
            d_trim = max(d_trim, (open_r - abs(th_uv)) * 0.25)
        d_smax, _ = _nb_smin(-d_sh, -d_trim, 0.014)
        return (-d_smax) * 0.68

    @njit(fastmath=True, inline="always")
    def _nb_sh_sdf(px: float, py: float, pz: float, sh_idx: int, sh_f: np.ndarray) -> Tuple[float, float, float, float]:
        dpx = px - sh_f[sh_idx, 0]
        dpy = py - sh_f[sh_idx, 1]
        dpz = pz - sh_f[sh_idx, 2]
        lx = sh_f[sh_idx, 8] * dpx + sh_f[sh_idx, 9] * dpy + sh_f[sh_idx, 10] * dpz
        ly = sh_f[sh_idx, 12] * dpx + sh_f[sh_idx, 13] * dpy + sh_f[sh_idx, 14] * dpz
        lz = sh_f[sh_idx, 16] * dpx + sh_f[sh_idx, 17] * dpy + sh_f[sh_idx, 18] * dpz

        rx = max(sh_f[sh_idx, 4], 1e-4)
        ry = max(sh_f[sh_idx, 5], 1e-4)
        rz = max(sh_f[sh_idx, 6], 1e-4)
        qx, qy, qz = lx / rx, ly / ry, lz / rz
        r_reg = math.sqrt(qx * qx + qy * qy + qz * qz + 1e-8)
        x, y, z = qx / r_reg, qy / r_reg, qz / r_reg
        xx, yy, zz = x * x, y * y, z * z

        s = sh_f[sh_idx, 20] * 0.282095
        s += sh_f[sh_idx, 21] * (0.488603 * y) + sh_f[sh_idx, 22] * (0.488603 * z) + sh_f[sh_idx, 23] * (0.488603 * x)
        s += sh_f[sh_idx, 24] * (1.092548 * x * y) + sh_f[sh_idx, 25] * (1.092548 * y * z)
        s += sh_f[sh_idx, 26] * (0.315392 * (3.0 * zz - 1.0)) + sh_f[sh_idx, 27] * (1.092548 * x * z)
        s += sh_f[sh_idx, 28] * (0.546274 * (xx - yy)) + sh_f[sh_idx, 29] * (0.590044 * y * (3.0 * xx - yy))
        s += sh_f[sh_idx, 30] * (2.890611 * x * y * z) + sh_f[sh_idx, 31] * (0.457046 * y * (5.0 * zz - 1.0))
        s += sh_f[sh_idx, 32] * (0.373176 * z * (5.0 * zz - 3.0)) + sh_f[sh_idx, 33] * (0.457046 * x * (5.0 * zz - 1.0))
        s += sh_f[sh_idx, 34] * (1.445306 * z * (xx - yy)) + sh_f[sh_idx, 35] * (0.590044 * x * (xx - 3.0 * yy))

        r_tgt = min(1.95, max(0.42, 1.0 + s))
        min_s = min(rx, min(ry, rz))
        d_sh = (r_reg - r_tgt) * min_s * 0.55
        return d_sh, lx, ly, lz

    @njit(fastmath=True, inline="always")
    def _nb_solid_sdf(px: float, py: float, pz: float, so_idx: int, solids_f: np.ndarray) -> Tuple[float, float, float]:
        dpx = px - solids_f[so_idx, 0]
        dpy = py - solids_f[so_idx, 1]
        dpz = pz - solids_f[so_idx, 2]
        qx = solids_f[so_idx, 8] * dpx + solids_f[so_idx, 9] * dpy + solids_f[so_idx, 10] * dpz
        qy = solids_f[so_idx, 12] * dpx + solids_f[so_idx, 13] * dpy + solids_f[so_idx, 14] * dpz
        qz = solids_f[so_idx, 16] * dpx + solids_f[so_idx, 17] * dpy + solids_f[so_idx, 18] * dpz

        hx, hy, hz = solids_f[so_idx, 4], solids_f[so_idx, 5], solids_f[so_idx, 6]
        kind = int(solids_f[so_idx, 7] + 0.5)
        pa, pb = solids_f[so_idx, 11], solids_f[so_idx, 15]
        bix, biy = 0.0, 0.0

        if kind == 1:
            spx = max(solids_f[so_idx, 24], 1e-4)
            spy = max(solids_f[so_idx, 25], 1e-4)
            spz = max(solids_f[so_idx, 26], 1e-4)
            bdx, bdy, bdz = solids_f[so_idx, 20], solids_f[so_idx, 21], solids_f[so_idx, 22]
            iy = min(bdy, max(-bdy, round(qy / spy)))
            stag = 0.5 * spx * (abs(iy) % 2.0)
            ix = min(bdx, max(-bdx, round((qx - stag) / spx)))
            iz = min(bdz, max(-bdz, round(qz / spz)))
            bix, biy = ix, iy
            qx -= (ix * spx + stag)
            qy -= iy * spy
            qz -= iz * spz
            qz += (_nb_hash2(ix, iy) - 0.5) * 0.014
        elif kind == 2:
            maj_r = max(0.01, pa)
            min_r = max(0.005, pb)
            r_xz = math.sqrt(qx * qx + qz * qz)
            d_tire = math.sqrt((r_xz - maj_r) ** 2 + qy * qy) - min_r
            d_disc = max(r_xz - maj_r * 0.88, abs(qy) - min_r * 0.35)
            spoke = 0.015 * abs(math.sin(math.atan2(qz, qx) * 5.0))
            return min(d_tire, d_disc + spoke), 0.0, 0.0
        elif kind == 3:
            e1 = min(2.2, max(0.15, pa))
            e2 = min(2.2, max(0.15, pb))
            aqx, aqy, aqz = abs(qx) / max(hx, 1e-4), abs(qy) / max(hy, 1e-4), abs(qz) / max(hz, 1e-4)
            sc = max(aqx, max(aqy, aqz))
            min_a = min(hx, min(hy, hz))
            if sc < 1e-6:
                return -min_a * 0.72, 0.0, 0.0
            nx = max(aqx / sc, 1e-6)
            ny = max(aqy / sc, 1e-6)
            nz = max(aqz / sc, 1e-6)
            xy = nx ** (2.0 / e2) + nz ** (2.0 / e2)
            xyz = (max(xy, 1e-10) ** (e2 / e1)) + ny ** (2.0 / e1)
            f_val = (max(xyz, 1e-10) ** (e1 * 0.5)) * sc
            return (f_val - 1.0) * min_a * 0.72, 0.0, 0.0

        rad = min(pa, min(hx, min(hy, hz)) * 0.48)
        rx = abs(qx) - (hx - rad)
        ry = abs(qy) - (hy - rad)
        rz = abs(qz) - (hz - rad)
        ox = max(rx, 0.0)
        oy = max(ry, 0.0)
        oz = max(rz, 0.0)
        d_box = math.sqrt(ox * ox + oy * oy + oz * oz) + min(max(rx, max(ry, rz)), 0.0) - rad
        return d_box, bix, biy

    @njit(fastmath=True, cache=True)
    def _nb_eval_dist(
        px: float, py: float, pz: float,
        n_c: int, n_sh: int, n_g: int,
        clusters_f: np.ndarray, splines_f: np.ndarray, stations_f: np.ndarray,
        sh_f: np.ndarray, solids_f: np.ndarray, garments_f: np.ndarray,
    ) -> float:
        d_min = 1e4
        for ci in range(n_c):
            dx = px - clusters_f[ci, 0]
            dy = py - clusters_f[ci, 1]
            dz = pz - clusters_f[ci, 2]
            d_sph = math.sqrt(dx * dx + dy * dy + dz * dz) - clusters_f[ci, 3]
            if d_sph > max(0.45, d_min + 0.12):
                d_min = min(d_min, d_sph)
                continue

            sp_start = int(clusters_f[ci, 4] + 0.5)
            sp_cnt = int(clusters_f[ci, 5] + 0.5)
            sh_idx = int(math.floor(clusters_f[ci, 6] + 0.1))
            so_packed = int(clusters_f[ci, 7] + 0.5)
            so_start = so_packed // 1024
            so_cnt = so_packed % 1024

            d_org = 1e4
            if 0 <= sh_idx < n_sh:
                sx = px - sh_f[sh_idx, 0]
                sy = py - sh_f[sh_idx, 1]
                sz = pz - sh_f[sh_idx, 2]
                db = math.sqrt(sx * sx + sy * sy + sz * sz) - sh_f[sh_idx, 3]
                if db < 0.32:
                    d_org, _, _, _ = _nb_sh_sdf(px, py, pz, sh_idx, sh_f)
                else:
                    d_org = db

            d_cl_min = 1e4
            for si in range(sp_cnt):
                idx = sp_start + si
                sx = px - splines_f[idx, 0]
                sy = py - splines_f[idx, 1]
                sz = pz - splines_f[idx, 2]
                db = math.sqrt(sx * sx + sy * sy + sz * sz) - splines_f[idx, 3]
                if db > max(0.32, min(d_org, d_cl_min) + 0.10):
                    d_org = min(d_org, db)
                    continue
                d_sp, t_st, th_st, w_ax, _, _, _ = _nb_spline_sdf(px, py, pz, idx, splines_f, stations_f)
                csg = int(splines_f[idx, 7] + 0.5)
                gid = int(math.floor(splines_f[idx, 11] + 0.1))
                meta = int(splines_f[idx, 15] + 0.5)
                role = meta // 100000

                if role == 4 and 0 <= gid < n_g:
                    d_cl_min = min(d_cl_min, _nb_cloth_sdf(d_sp, t_st, th_st, w_ax, gid, 0.25, garments_f))
                else:
                    if csg == 0:
                        d_org, _ = _nb_smin(d_org, d_sp, splines_f[idx, 6])
                    else:
                        d_org = min(d_org, d_sp)
                    if 0 <= gid < n_g:
                        d_cl_ = _nb_cloth_sdf(d_sp, t_st, th_st, w_ax, gid, 0.4, garments_f)
                        d_cl_min, _ = _nb_smin(d_cl_min, d_cl_, 0.052)

            d_min = min(d_min, min(d_org, d_cl_min))
            for oi in range(so_cnt):
                idx = so_start + oi
                sx = px - solids_f[idx, 0]
                sy = py - solids_f[idx, 1]
                sz = pz - solids_f[idx, 2]
                db = math.sqrt(sx * sx + sy * sy + sz * sz) - solids_f[idx, 3]
                if db > max(0.35, d_min + 0.12):
                    d_min = min(d_min, db)
                    continue
                d_so, _, _ = _nb_solid_sdf(px, py, pz, idx, solids_f)
                csg = int(solids_f[idx, 27] + 0.5) % 10
                if csg == 0:
                    d_min, _ = _nb_smin(d_min, d_so, solids_f[idx, 19])
                else:
                    d_min = min(d_min, d_so)
        return d_min

    @njit(fastmath=True, cache=True)
    def _nb_soft_shadow(
        rox: float, roy: float, roz: float,
        kdx: float, kdy: float, kdz: float, k_soft: float,
        n_c: int, n_sh: int, n_g: int,
        clusters_f: np.ndarray, splines_f: np.ndarray, stations_f: np.ndarray,
        sh_f: np.ndarray, solids_f: np.ndarray, garments_f: np.ndarray,
    ) -> float:
        res = 1.0
        t = 0.045
        for _ in range(18):
            if t > 8.0:
                break
            h = _nb_eval_dist(rox + kdx * t, roy + kdy * t, roz + kdz * t, n_c, n_sh, n_g, clusters_f, splines_f, stations_f, sh_f, solids_f, garments_f)
            if h < 0.0025:
                return 0.20
            res = min(res, k_soft * h / max(t, 1e-3))
            t += min(0.35, max(0.03, h))
        res = min(1.0, max(0.20, res))
        return res * res * (3.0 - 2.0 * res)

    @njit(parallel=True, fastmath=True, cache=True)
    def _nb_render_pass1_rows(
        width: int, height: int,
        cam_pos: np.ndarray, cam_fwd: np.ndarray, cam_right: np.ndarray, cam_up: np.ndarray,
        tan_half: float, aspect: float, style_mode: int, max_steps: int,
        n_c: int, n_sh: int, n_g: int, n_mid: int, n_em: int,
        clusters_f: np.ndarray, splines_f: np.ndarray, stations_f: np.ndarray,
        sh_f: np.ndarray, solids_f: np.ndarray, garments_f: np.ndarray,
        decals_f: np.ndarray, mid_f: np.ndarray, emitters_f: np.ndarray,
        key_dir: np.ndarray, key_col: np.ndarray, shadow_tint: np.ndarray,
        amb_col: np.ndarray, rim_col: np.ndarray, rim_str: float, sss_str: float, sss_col: np.ndarray,
        sky_top: np.ndarray, sky_hor: np.ndarray, sky_bot: np.ndarray,
        g_en: float, g_y: float, g_c1: np.ndarray, g_c2: np.ndarray, haze_col: np.ndarray, haze_str: float,
        out_hdr_rim: np.ndarray, out_nrm_mat: np.ndarray, out_depth_hit: np.ndarray,
    ) -> None:
        rox, roy, roz = cam_pos[0], cam_pos[1], cam_pos[2]
        kdx, kdy, kdz = key_dir[0], key_dir[1], key_dir[2]

        for y in prange(height):
            ny_ndc = 1.0 - 2.0 * ((float(y) + 0.5) / float(height))
            for x in range(width):
                nx_ndc = 2.0 * ((float(x) + 0.5) / float(width)) - 1.0
                rdx = cam_fwd[0] + nx_ndc * aspect * tan_half * cam_right[0] + ny_ndc * tan_half * cam_up[0]
                rdy = cam_fwd[1] + nx_ndc * aspect * tan_half * cam_right[1] + ny_ndc * tan_half * cam_up[1]
                rdz = cam_fwd[2] + nx_ndc * aspect * tan_half * cam_right[2] + ny_ndc * tan_half * cam_up[2]
                rn = math.sqrt(max(rdx * rdx + rdy * rdy + rdz * rdz, 1e-12))
                rdx, rdy, rdz = rdx / rn, rdy / rn, rdz / rn

                # Sky gradient + HDR emitters
                st = min(1.0, max(0.0, rdy * 0.85 + 0.25))
                s1 = min(1.0, max(0.0, st / 0.35))
                s1 = s1 * s1 * (3.0 - 2.0 * s1)
                s2 = min(1.0, max(0.0, (st - 0.32) / 0.63))
                s2 = s2 * s2 * (3.0 - 2.0 * s2)
                bg_r = (sky_bot[0] * (1.0 - s1) + sky_hor[0] * s1) * (1.0 - s2) + sky_top[0] * s2
                bg_g = (sky_bot[1] * (1.0 - s1) + sky_hor[1] * s1) * (1.0 - s2) + sky_top[1] * s2
                bg_b = (sky_bot[2] * (1.0 - s1) + sky_hor[2] * s1) * (1.0 - s2) + sky_top[2] * s2
                for ei in range(n_em):
                    ca = max(0.0, rdx * emitters_f[ei, 0] + rdy * emitters_f[ei, 1] + rdz * emitters_f[ei, 2])
                    ang = math.acos(min(1.0, max(-1.0, ca)))
                    er = emitters_f[ei, 3]
                    dt_d = min(1.0, max(0.0, (er * 1.25 - ang) / max(er * 0.6, 1e-4)))
                    disc = dt_d * dt_d * (3.0 - 2.0 * dt_d)
                    halo = (ca ** 8.0) * 0.28 + (ca ** 48.0) * 0.55
                    w_e = disc * 0.85 + halo * 0.14
                    bg_r += emitters_f[ei, 4] * w_e
                    bg_g += emitters_f[ei, 5] * w_e
                    bg_b += emitters_f[ei, 6] * w_e

                depth_z = 1e4
                hit_flag = 0.0
                col_r, col_g, col_b = bg_r, bg_g, bg_b
                nrm_x, nrm_y, nrm_z = 0.0, 1.0, 0.0
                rim_out = 0.0
                obj_id = 0.0

                # Zone 2 Midground silhouettes
                inv_x = 1.0 / (rdx if abs(rdx) > 1e-5 else 1e-5)
                inv_y = 1.0 / (rdy if abs(rdy) > 1e-5 else 1e-5)
                inv_z = 1.0 / (rdz if abs(rdz) > 1e-5 else 1e-5)
                for mi in range(n_mid):
                    mcx, mcy, mcz = mid_f[mi, 0], mid_f[mi, 1], mid_f[mi, 2]
                    m_rp = mid_f[mi, 3]
                    mhx, mhy, mhz = mid_f[mi, 4], mid_f[mi, 5], mid_f[mi, 6]
                    t0x, t1x = (mcx - mhx - rox) * inv_x, (mcx + mhx - rox) * inv_x
                    t0y, t1y = (mcy - mhy - roy) * inv_y, (mcy + mhy + max(m_rp, 0.0) - roy) * inv_y
                    t0z, t1z = (mcz - mhz - roz) * inv_z, (mcz + mhz - roz) * inv_z
                    te = max(min(t0x, t1x), max(min(t0y, t1y), min(t0z, t1z)))
                    txx = min(max(t0x, t1x), min(max(t0y, t1y), max(t0z, t1z)))
                    if txx >= max(te, 0.2) and te < depth_z:
                        t_hit = max(te, 0.2)
                        hpx = rox + rdx * t_hit
                        hpy = roy + rdy * t_hit
                        ly = hpy - (mcy + mhy)
                        ok = True
                        if m_rp < -1.5:
                            rel_y = (hpy - mcy) / max(0.1, mhy)
                            dx_m = (hpx - mcx) / max(0.1, mhx)
                            if rel_y < -0.15:
                                trunk_w = 0.15 * (1.0 - max(rel_y + 0.65, 0.0) * 0.35)
                                if abs(dx_m - 0.05 * math.sin(rel_y * 3.0)) > trunk_w:
                                    ok = False
                            else:
                                l1 = dx_m * dx_m * 1.15 + ((rel_y - 0.45) / 0.55) ** 2
                                l2 = ((dx_m + 0.44) / 0.58) ** 2 + ((rel_y - 0.16) / 0.46) ** 2
                                l3 = ((dx_m - 0.44) / 0.58) ** 2 + ((rel_y - 0.22) / 0.46) ** 2
                                if min(l1, min(l2, l3)) > 1.0:
                                    ok = False
                        elif m_rp < -0.5:
                            rel_y = (hpy - mcy) / max(0.1, mhy)
                            dx_m = abs(hpx - mcx) / max(0.1, mhx)
                            if rel_y > 0.55:
                                hd = (rel_y - 0.78) / 0.22
                                if dx_m * dx_m * 2.2 + hd * hd > 1.0:
                                    ok = False
                            elif dx_m > 0.85 - max(0.0, rel_y) * 0.2:
                                ok = False
                        elif ly > 0.0 and m_rp > 0.05:
                            rf = 1.0 - min(1.0, max(0.0, ly / m_rp))
                            if abs(hpx - mcx) > mhx * rf + 0.08:
                                ok = False
                        if ok:
                            depth_z = t_hit
                            hit_flag = 0.5
                            obj_id = 0.5
                            mr, mg, mb = mid_f[mi, 8], mid_f[mi, 9], mid_f[mi, 10]
                            hz = min(0.85, max(0.0, (t_hit - 4.0) / 15.0)) * haze_str
                            col_r = mr * (1.0 - hz) + haze_col[0] * hz
                            col_g = mg * (1.0 - hz) + haze_col[1] * hz
                            col_b = mb * (1.0 - hz) + haze_col[2] * hz

                # Ground plane
                if g_en > 0.5 and rdy < -1e-4:
                    tg = (g_y - roy) / rdy
                    if 0.1 < tg < 38.0 and tg < depth_z:
                        pgx, pgz = rox + rdx * tg, roz + rdz * tg
                        gn = _nb_noise2(pgx * 2.2, pgz * 2.2)
                        gar = g_c2[0] * (1.0 - gn) + g_c1[0] * gn
                        gag = g_c2[1] * (1.0 - gn) + g_c1[1] * gn
                        gab = g_c2[2] * (1.0 - gn) + g_c1[2] * gn
                        shg = _nb_soft_shadow(pgx, g_y + 0.02, pgz, kdx, kdy, kdz, 10.0, n_c, n_sh, n_g, clusters_f, splines_f, stations_f, sh_f, solids_f, garments_f)
                        diff_g = max(kdy, 0.15) * shg * 1.2
                        glr = gar * (amb_col[0] * 0.45 + key_col[0] * diff_g)
                        glg = gag * (amb_col[1] * 0.45 + key_col[1] * diff_g)
                        glb = gab * (amb_col[2] * 0.45 + key_col[2] * diff_g)
                        hz = min(0.7, max(0.0, (tg - 6.0) / 24.0)) * haze_str
                        glr = glr * (1.0 - hz) + haze_col[0] * hz
                        glg = glg * (1.0 - hz) + haze_col[1] * hz
                        glb = glb * (1.0 - hz) + haze_col[2] * hz
                        fade = min(1.0, max(0.0, (38.0 - tg) / 28.0))
                        col_r = bg_r * (1.0 - fade) + glr * fade
                        col_g = bg_g * (1.0 - fade) + glg * fade
                        col_b = bg_b * (1.0 - fade) + glb * fade
                        depth_z = tg
                        hit_flag = 0.5
                        obj_id = 0.25

                # Zone 1 Foreground Sphere Trace
                t_cur = 0.05
                t_max = min(depth_z + 0.1, 28.0)
                hit_fg = False
                for _ in range(max_steps):
                    if t_cur > t_max:
                        break
                    px, py, pz = rox + rdx * t_cur, roy + rdy * t_cur, roz + rdz * t_cur
                    d = _nb_eval_dist(px, py, pz, n_c, n_sh, n_g, clusters_f, splines_f, stations_f, sh_f, solids_f, garments_f)
                    if d < 0.0014:
                        hit_fg = True
                        break
                    t_cur += max(d * 0.78, 0.0018)

                if hit_fg and t_cur < depth_z:
                    px, py, pz = rox + rdx * t_cur, roy + rdy * t_cur, roz + rdz * t_cur
                    h = 0.0025
                    d1 = _nb_eval_dist(px + h, py - h, pz - h, n_c, n_sh, n_g, clusters_f, splines_f, stations_f, sh_f, solids_f, garments_f)
                    d2 = _nb_eval_dist(px - h, py - h, pz + h, n_c, n_sh, n_g, clusters_f, splines_f, stations_f, sh_f, solids_f, garments_f)
                    d3 = _nb_eval_dist(px - h, py + h, pz - h, n_c, n_sh, n_g, clusters_f, splines_f, stations_f, sh_f, solids_f, garments_f)
                    d4 = _nb_eval_dist(px + h, py + h, pz + h, n_c, n_sh, n_g, clusters_f, splines_f, stations_f, sh_f, solids_f, garments_f)
                    gx = d1 - d2 - d3 + d4
                    gy = -d1 - d2 + d3 + d4
                    gz = -d1 + d2 - d3 + d4
                    gn = math.sqrt(max(gx * gx + gy * gy + gz * gz, 1e-12))
                    ngx, ngy, ngz = gx / gn, gy / gn, gz / gn

                    # Surface material resolution at hit point
                    alb_r, alb_g, alb_b = 0.8, 0.8, 0.8
                    ns_x, ns_y, ns_z = ngx, ngy, ngz
                    ht_x, ht_y, ht_z = 0.0, 1.0, 0.0
                    rgh, mtl, emi = 0.6, 0.0, 0.0
                    is_hair, is_face = 0.0, 0.0
                    d_best = 1e4

                    for ci in range(n_c):
                        cdx = px - clusters_f[ci, 0]
                        cdy = py - clusters_f[ci, 1]
                        cdz = pz - clusters_f[ci, 2]
                        if math.sqrt(cdx * cdx + cdy * cdy + cdz * cdz) - clusters_f[ci, 3] > 0.18:
                            continue
                        sp_start = int(clusters_f[ci, 4] + 0.5)
                        sp_cnt = int(clusters_f[ci, 5] + 0.5)
                        sh_idx = int(math.floor(clusters_f[ci, 6] + 0.1))
                        so_packed = int(clusters_f[ci, 7] + 0.5)
                        so_start = so_packed // 1024
                        so_cnt = so_packed % 1024

                        d_org = 1e4
                        oa_r, oa_g, oa_b = 0.8, 0.8, 0.8
                        on_x, on_y, on_z = ngx, ngy, ngz
                        ot_x, ot_y, ot_z = 0.0, 1.0, 0.0
                        orgh, ohair, oface = 0.6, 0.0, 0.0

                        if 0 <= sh_idx < n_sh:
                            sx = px - sh_f[sh_idx, 0]
                            sy = py - sh_f[sh_idx, 1]
                            sz = pz - sh_f[sh_idx, 2]
                            if math.sqrt(sx * sx + sy * sy + sz * sz) - sh_f[sh_idx, 3] < 0.15:
                                d_sh, lx, ly, lz = _nb_sh_sdf(px, py, pz, sh_idx, sh_f)
                                d_org = d_sh
                                oa_r, oa_g, oa_b = sh_f[sh_idx, 36], sh_f[sh_idx, 37], sh_f[sh_idx, 38]
                                oface = 1.0
                                # Proxy normal blend
                                plx, ply, plz = sh_f[sh_idx, 11], sh_f[sh_idx, 15], sh_f[sh_idx, 19]
                                pwx = sh_f[sh_idx, 0] + sh_f[sh_idx, 8] * plx + sh_f[sh_idx, 12] * ply + sh_f[sh_idx, 16] * plz
                                pwy = sh_f[sh_idx, 1] + sh_f[sh_idx, 9] * plx + sh_f[sh_idx, 13] * ply + sh_f[sh_idx, 17] * plz
                                pwz = sh_f[sh_idx, 2] + sh_f[sh_idx, 10] * plx + sh_f[sh_idx, 14] * ply + sh_f[sh_idx, 18] * plz
                                npx, npy, npz = px - pwx, py - pwy, pz - pwz
                                npn = math.sqrt(max(npx * npx + npy * npy + npz * npz, 1e-12))
                                npx, npy, npz = npx / npn, npy / npn, npz / npn

                                qx = lx / max(sh_f[sh_idx, 4], 1e-4)
                                qy = ly / max(sh_f[sh_idx, 5], 1e-4)
                                qz = lz / max(sh_f[sh_idx, 6], 1e-4)
                                qn = math.sqrt(max(qx * qx + qy * qy + qz * qz, 1e-12))
                                dx_u, dy_u, dz_u = qx / qn, qy / qn, qz / qn
                                fm = min(1.0, max(0.0, (dz_u - 0.15) / 0.4))
                                pw = sh_f[sh_idx, 39] * fm
                                bx = ngx * (1.0 - pw) + npx * pw
                                by = ngy * (1.0 - pw) + npy * pw
                                bz = ngz * (1.0 - pw) + npz * pw
                                bbn = math.sqrt(max(bx * bx + by * by + bz * bz, 1e-12))
                                on_x, on_y, on_z = bx / bbn, by / bbn, bz / bbn

                                # 2D UV Eyes & Feline Nose directly on front face (+Z)
                                if dz_u > 0.25:
                                    th = math.atan2(dx_u, dz_u)
                                    ph = math.asin(min(1.0, max(-1.0, dy_u)))
                                    meta_sh = int(sh_f[sh_idx, 47] + 0.5)
                                    eye_sty = meta_sh // 10000
                                    cat_n = (meta_sh // 1000) % 10
                                    e_sc = max(0.4, sh_f[sh_idx, 43])
                                    eye_th = 0.44 if cat_n == 1 else 0.37
                                    eye_ph = 0.06 if cat_n == 1 else 0.03
                                    ue = (abs(th) - eye_th) / (0.175 * e_sc)
                                    ve = (ph - eye_ph) / (0.135 * e_sc)
                                    if abs(ue) < 1.25 and abs(ve) < 1.25:
                                        ry_e = 0.74 if eye_sty == 0 else (0.72 if eye_sty == 1 else 0.60)
                                        mscl = min(1.0, max(0.0, 1.0 - ((math.sqrt((ue / 0.78) ** 2 + (ve / ry_e) ** 2) - 1.0) * ry_e) * 28.0))
                                        irx = 0.50 if eye_sty == 1 else 0.44
                                        mir = min(1.0, max(0.0, 1.0 - ((math.sqrt((ue / irx) ** 2 + (ve / 0.54) ** 2) - 1.0) * irx) * 34.0)) * mscl
                                        prx = 0.11 if eye_sty == 1 else (0.16 if eye_sty == 2 else 0.18)
                                        pry = 0.36 if eye_sty == 1 else (0.26 if eye_sty == 2 else 0.23)
                                        mpup = min(1.0, max(0.0, 1.0 - ((math.sqrt((ue / prx) ** 2 + (ve / pry) ** 2) - 1.0) * prx) * 45.0)) * mir
                                        mc1 = min(1.0, max(0.0, 1.0 - ((math.sqrt(((ue + 0.14) / 0.13) ** 2 + ((ve - 0.16) / 0.11) ** 2) - 1.0) * 0.11) * 52.0)) * mscl
                                        gt = min(1.0, max(0.0, ve + 0.5))
                                        ir_r = sh_f[sh_idx, 40] * (0.35 + 0.65 * gt)
                                        ir_g = sh_f[sh_idx, 41] * (0.35 + 0.65 * gt)
                                        ir_b = sh_f[sh_idx, 42] * (0.35 + 0.65 * gt)
                                        er = 0.96 * (1.0 - mir) + ir_r * mir
                                        eg = 0.97 * (1.0 - mir) + ir_g * mir
                                        eb = 0.99 * (1.0 - mir) + ir_b * mir
                                        er = er * (1.0 - mpup) + 0.04 * mpup
                                        eg = eg * (1.0 - mpup) + 0.03 * mpup
                                        eb = eb * (1.0 - mpup) + 0.07 * mpup
                                        er = er * (1.0 - mc1) + 1.85 * mc1
                                        eg = eg * (1.0 - mc1) + 1.90 * mc1
                                        eb = eb * (1.0 - mc1) + 1.95 * mc1
                                        # Upper lash curve approximation
                                        l_y = ry_e * (1.0 - (ue / 0.85) ** 2) * 0.88 + 0.04
                                        mlsh = min(1.0, max(0.0, 1.0 - (abs(ve - l_y) - 0.065) * 32.0)) if abs(ue) < 0.88 else 0.0
                                        er = er * (1.0 - mlsh) + sh_f[sh_idx, 44] * mlsh
                                        eg = eg * (1.0 - mlsh) + sh_f[sh_idx, 45] * mlsh
                                        eb = eb * (1.0 - mlsh) + sh_f[sh_idx, 46] * mlsh
                                        ea = max(mscl, mlsh)
                                        oa_r = oa_r * (1.0 - ea) + er * ea
                                        oa_g = oa_g * (1.0 - ea) + eg * ea
                                        oa_b = oa_b * (1.0 - ea) + eb * ea
                                    if cat_n == 1:
                                        dn = math.sqrt((th * 1.15) ** 2 + ((ph + 0.15) * 1.45) ** 2)
                                        mn = min(1.0, max(0.0, 1.0 - (dn - 0.095) * 32.0))
                                        oa_r = oa_r * (1.0 - mn) + 0.95 * mn
                                        oa_g = oa_g * (1.0 - mn) + 0.45 * mn
                                        oa_b = oa_b * (1.0 - mn) + 0.52 * mn
                                        if abs(th) < 0.22:
                                            dm = abs(ph + 0.26 - 0.03 * math.cos(th * 18.0))
                                            ms = min(1.0, max(0.0, 1.0 - (dm - 0.012) * 55.0)) * 0.85
                                            oa_r = oa_r * (1.0 - ms) + sh_f[sh_idx, 44] * ms
                                            oa_g = oa_g * (1.0 - ms) + sh_f[sh_idx, 45] * ms
                                            oa_b = oa_b * (1.0 - ms) + sh_f[sh_idx, 46] * ms

                        d_cl_min_f = 1e4
                        ca_r, ca_g, ca_b, crgh = 0.8, 0.8, 0.8, 0.78
                        d_sk_min_f = 1e4
                        ska_r, ska_g, ska_b, skrgh = 0.8, 0.8, 0.8, 0.78

                        for si in range(sp_cnt):
                            idx = sp_start + si
                            sx = px - splines_f[idx, 0]
                            sy = py - splines_f[idx, 1]
                            sz = pz - splines_f[idx, 2]
                            if math.sqrt(sx * sx + sy * sy + sz * sz) - splines_f[idx, 3] > 0.24:
                                continue
                            d_sp, t_st, th_st, w_ax, stx, sty, stz = _nb_spline_sdf(px, py, pz, idx, splines_f, stations_f)
                            csg = int(splines_f[idx, 7] + 0.5)
                            gid = int(math.floor(splines_f[idx, 11] + 0.1))
                            meta_sp = int(splines_f[idx, 15] + 0.5)
                            role = meta_sp // 100000
                            rgh_sp = float((meta_sp // 10) % 100) / 99.0
                            if role != 4:
                                sp_r, sp_g, sp_b = splines_f[idx, 8], splines_f[idx, 9], splines_f[idx, 10]
                                hr_flag = 1.0 if role == 1 else 0.0
                                if csg == 0:
                                    d_org, eta = _nb_smin(d_org, d_sp, splines_f[idx, 6])
                                    oa_r = oa_r * (1.0 - eta) + sp_r * eta
                                    oa_g = oa_g * (1.0 - eta) + sp_g * eta
                                    oa_b = oa_b * (1.0 - eta) + sp_b * eta
                                    on_x = on_x * (1.0 - eta) + ngx * eta
                                    on_y = on_y * (1.0 - eta) + ngy * eta
                                    on_z = on_z * (1.0 - eta) + ngz * eta
                                    ot_x = ot_x * (1.0 - eta) + stx * eta
                                    ot_y = ot_y * (1.0 - eta) + sty * eta
                                    ot_z = ot_z * (1.0 - eta) + stz * eta
                                    orgh = orgh * (1.0 - eta) + rgh_sp * eta
                                    ohair = ohair * (1.0 - eta) + hr_flag * eta
                                    oface = oface * (1.0 - eta)
                                elif d_sp < d_org:
                                    d_org = d_sp
                                    oa_r, oa_g, oa_b = sp_r, sp_g, sp_b
                                    on_x, on_y, on_z = ngx, ngy, ngz
                                    ot_x, ot_y, ot_z = stx, sty, stz
                                    orgh, ohair, oface = rgh_sp, hr_flag, 0.0

                            if 0 <= gid < n_g:
                                d_cl = _nb_cloth_sdf(d_sp, t_st, th_st, w_ax, gid, 0.25 if role == 4 else 0.4, garments_f)
                                cl_r, cl_g, cl_b = garments_f[gid, 8], garments_f[gid, 9], garments_f[gid, 10]
                                dec_s = int(garments_f[gid, 14] + 0.5)
                                dec_c = int(garments_f[gid, 15] + 0.5)
                                for di in range(dec_c):
                                    dk = int(decals_f[dec_s + di, 0] + 0.5)
                                    dt = t_st - decals_f[dec_s + di, 1]
                                    dth = ((th_st - decals_f[dec_s + di, 2] + 3.14159265) % 6.2831853) - 3.14159265
                                    dsz = max(0.02, decals_f[dec_s + di, 3])
                                    rd_dec = math.sqrt((dth * 0.45) ** 2 + dt * dt) / dsz
                                    if rd_dec < 1.3:
                                        mdec = 0.0
                                        if dk == 1:
                                            qdx, qdy = abs(dth * 0.45) / dsz, abs(dt) / dsz
                                            dc = min(max(qdx - 0.82, qdy - 0.28), max(qdx - 0.28, qdy - 0.82))
                                            mdec = min(1.0, max(0.0, 1.0 - dc * 22.0))
                                        elif dk == 2:
                                            mdec = min(1.0, max(0.0, 1.0 - (abs(dt) / dsz - 0.25) * 28.0))
                                        else:
                                            mdec = min(1.0, max(0.0, 1.0 - (rd_dec - 0.72) * 20.0))
                                        cl_r = cl_r * (1.0 - mdec) + decals_f[dec_s + di, 4] * mdec
                                        cl_g = cl_g * (1.0 - mdec) + decals_f[dec_s + di, 5] * mdec
                                        cl_b = cl_b * (1.0 - mdec) + decals_f[dec_s + di, 6] * mdec
                                grgh_v = float(int(garments_f[gid, 11] + 0.5) % 1000) / 99.0
                                if role == 4:
                                    if d_cl < d_sk_min_f:
                                        d_sk_min_f = d_cl
                                        ska_r, ska_g, ska_b, skrgh = cl_r, cl_g, cl_b, grgh_v
                                else:
                                    d_cl_min_f, c_eta = _nb_smin(d_cl_min_f, d_cl, 0.052)
                                    ca_r = ca_r * (1.0 - c_eta) + cl_r * c_eta
                                    ca_g = ca_g * (1.0 - c_eta) + cl_g * c_eta
                                    ca_b = ca_b * (1.0 - c_eta) + cl_b * c_eta
                                    crgh = crgh * (1.0 - c_eta) + grgh_v * c_eta

                        if d_sk_min_f < d_best:
                            d_best = d_sk_min_f
                            alb_r, alb_g, alb_b = ska_r, ska_g, ska_b
                            ns_x, ns_y, ns_z = ngx, ngy, ngz
                            rgh, mtl, emi, is_hair, is_face = skrgh, 0.0, 0.0, 0.0, 0.0
                            obj_id = float(ci + 1) * 0.08 + 0.04
                        if d_cl_min_f < d_best:
                            d_best = d_cl_min_f
                            alb_r, alb_g, alb_b = ca_r, ca_g, ca_b
                            ns_x, ns_y, ns_z = ngx, ngy, ngz
                            rgh, mtl, emi, is_hair, is_face = crgh, 0.0, 0.0, 0.0, 0.0
                            obj_id = float(ci + 1) * 0.08 + 0.03

                        if d_org < d_best:
                            d_best = d_org
                            alb_r, alb_g, alb_b = oa_r, oa_g, oa_b
                            ns_x, ns_y, ns_z = on_x, on_y, on_z
                            ht_x, ht_y, ht_z = ot_x, ot_y, ot_z
                            rgh, mtl, emi = orgh, 0.0, 0.0
                            is_hair, is_face = ohair, oface
                            obj_id = float(ci + 1) * 0.08

                        for oi in range(so_cnt):
                            idx = so_start + oi
                            sx = px - solids_f[idx, 0]
                            sy = py - solids_f[idx, 1]
                            sz = pz - solids_f[idx, 2]
                            if math.sqrt(sx * sx + sy * sy + sz * sz) - solids_f[idx, 3] > 0.24:
                                continue
                            d_so, bix, biy = _nb_solid_sdf(px, py, pz, idx, solids_f)
                            if d_so < d_best:
                                d_best = d_so
                                meta_so = int(solids_f[idx, 27] + 0.5)
                                sh_id = meta_so // 1000
                                sr, sg, sb = solids_f[idx, 28], solids_f[idx, 29], solids_f[idx, 30]
                                if sh_id == 5:
                                    bid = _nb_hash2(bix + 3.1, biy + 7.7)
                                    mul = 0.76 + 0.36 * bid
                                    sr = min(1.0, sr * mul)
                                    sg = min(1.0, sg * mul)
                                    sb = min(1.0, sb * mul)
                                alb_r, alb_g, alb_b = sr, sg, sb
                                ns_x, ns_y, ns_z = ngx, ngy, ngz
                                rgh = float((meta_so // 10) % 100) / 99.0
                                mtl = solids_f[idx, 31]
                                emi = solids_f[idx, 23]
                                is_hair, is_face = 0.0, 0.0
                                obj_id = float(ci + 1) * 0.08 + 0.05

                    # Shadows & 4-tap AO
                    sh = _nb_soft_shadow(px + ngx * 0.045, py + ngy * 0.045, pz + ngz * 0.045, kdx, kdy, kdz, 8.5, n_c, n_sh, n_g, clusters_f, splines_f, stations_f, sh_f, solids_f, garments_f)
                    occ = 0.0
                    w_ao = 0.55
                    for ai in range(4):
                        hr = 0.04 + 0.10 * float(ai)
                        da = _nb_eval_dist(px + ngx * hr, py + ngy * hr, pz + ngz * hr, n_c, n_sh, n_g, clusters_f, splines_f, stations_f, sh_f, solids_f, garments_f)
                        occ += max(0.0, hr - da) * w_ao
                        w_ao *= 0.62
                    ao = min(1.0, max(0.25, 1.0 - occ * 1.55))

                    ndotl = ns_x * kdx + ns_y * kdy + ns_z * kdz
                    ndotv = min(1.0, max(0.0, -(ngx * rdx + ngy * rdy + ngz * rdz)))
                    fresnel = ((1.0 - ndotv) ** 3.4) * rim_str * (0.4 + 0.6 * ao) * (1.0 - 0.55 * is_hair)

                    if style_mode == 0:
                        lb = ndotl * sh
                        c_t = min(1.0, max(0.0, (lb - 0.14) / 0.10))
                        cel = c_t * c_t * (3.0 - 2.0 * c_t)
                        sh_r = alb_r * shadow_tint[0] * 0.94 * ao
                        sh_g = alb_g * shadow_tint[1] * 0.94 * ao
                        sh_b = alb_b * shadow_tint[2] * 0.94 * ao
                        br_r = alb_r * key_col[0] * 1.05
                        br_g = alb_g * key_col[1] * 1.05
                        br_b = alb_b * key_col[2] * 1.05
                        lit_r = sh_r * (1.0 - cel) + br_r * cel
                        lit_g = sh_g * (1.0 - cel) + br_g * cel
                        lit_b = sh_b * (1.0 - cel) + br_b * cel
                        sss_b = math.exp(-(((lb - 0.19) / 0.085) ** 2)) * sss_str * (0.42 + 0.58 * is_face)
                        lit_r += alb_r * sss_col[0] * sss_b
                        lit_g += alb_g * sss_col[1] * sss_b
                        lit_b += alb_b * sss_col[2] * sss_b

                        hx, hy, hz = kdx - rdx, kdy - rdy, kdz - rdz
                        hn = math.sqrt(max(hx * hx + hy * hy + hz * hz, 1e-12))
                        hx, hy, hz = hx / hn, hy / hn, hz / hn
                        if is_hair > 0.2:
                            tsx, tsy, tsz = ht_x + ngx * 0.18, ht_y + ngy * 0.18, ht_z + ngz * 0.18
                            tsn = math.sqrt(max(tsx * tsx + tsy * tsy + tsz * tsz, 1e-12))
                            tdoth = (tsx * hx + tsy * hy + tsz * hz) / tsn
                            sin_th = math.sqrt(max(0.0, 1.0 - tdoth * tdoth))
                            l1 = (sin_th ** 26.0) * 0.10
                            l2 = (sin_th ** 96.0) * 0.22
                            lit_r += (alb_r * l1 + key_col[0] * l2) * sh * is_hair
                            lit_g += (alb_g * l1 + key_col[1] * l2) * sh * is_hair
                            lit_b += (alb_b * l1 + key_col[2] * l2) * sh * is_hair
                        else:
                            ndh = max(0.0, ngx * hx + ngy * hy + ngz * hz)
                            sp_t = min(1.0, max(0.0, (ndh - 0.91) / 0.06))
                            sp_mask = min(1.0, max(0.0, (0.52 - rgh) / 0.35))
                            spec = (sp_t * sp_t * (3.0 - 2.0 * sp_t)) * sp_mask * sh * 0.38
                            lit_r += key_col[0] * spec
                            lit_g += key_col[1] * spec
                            lit_b += key_col[2] * spec
                    else:
                        diff = max(ndotl, 0.0) * sh
                        wrap = min(1.0, max(0.0, (ndotl + 0.45) / 1.45)) * sss_str
                        hx, hy, hz = kdx - rdx, kdy - rdy, kdz - rdz
                        hn = math.sqrt(max(hx * hx + hy * hy + hz * hz, 1e-12))
                        ndh = max(0.0, (ngx * hx + ngy * hy + ngz * hz) / hn)
                        ar = max(0.06, rgh * rgh)
                        den = ndh * ndh * (ar * ar - 1.0) + 1.0
                        d_ggx = (ar * ar) / (3.14159 * den * den)
                        spec = d_ggx * sh * (0.08 * (1.0 - mtl) + 0.85 * mtl)
                        sh_f_v = ((1.0 - ndotv) ** 4.0) * 0.45 * (1.0 - mtl)
                        lit_r = alb_r * (amb_col[0] * 0.55 * ao + key_col[0] * diff * 1.35 + sss_col[0] * wrap * 0.35 + sh_f_v) + key_col[0] * spec
                        lit_g = alb_g * (amb_col[1] * 0.55 * ao + key_col[1] * diff * 1.35 + sss_col[1] * wrap * 0.35 + sh_f_v) + key_col[1] * spec
                        lit_b = alb_b * (amb_col[2] * 0.55 * ao + key_col[2] * diff * 1.35 + sss_col[2] * wrap * 0.35 + sh_f_v) + key_col[2] * spec

                    rim_w = fresnel * (0.45 + 0.55 * min(1.0, max(0.0, -(ngx * kdx + ngy * kdy + ngz * kdz) * 0.5 + 0.5)))
                    col_r = lit_r + rim_col[0] * rim_w + alb_r * emi * 1.9
                    col_g = lit_g + rim_col[1] * rim_w + alb_g * emi * 1.9
                    col_b = lit_b + rim_col[2] * rim_w + alb_b * emi * 1.9
                    depth_z = t_cur
                    nrm_x, nrm_y, nrm_z = ngx, ngy, ngz
                    hit_flag = 1.0
                    rim_out = min(1.5, max(0.0, fresnel))

                out_hdr_rim[y, x, 0] = max(0.0, col_r)
                out_hdr_rim[y, x, 1] = max(0.0, col_g)
                out_hdr_rim[y, x, 2] = max(0.0, col_b)
                out_hdr_rim[y, x, 3] = rim_out
                out_nrm_mat[y, x, 0] = nrm_x
                out_nrm_mat[y, x, 1] = nrm_y
                out_nrm_mat[y, x, 2] = nrm_z
                out_nrm_mat[y, x, 3] = obj_id
                out_depth_hit[y, x, 0] = depth_z
                out_depth_hit[y, x, 1] = hit_flag


def _render_cpu_numpy_fallback(
    compiled: CompiledScene,
    width: int,
    height: int,
    cam_pos: np.ndarray,
    cam_fwd: np.ndarray,
    cam_right: np.ndarray,
    cam_up: np.ndarray,
    tan_half: float,
    aspect: float,
    max_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-NumPy 256x256 strip fallback used only when Numba is unavailable."""
    hdr_rim = np.zeros((height, width, 4), dtype=np.float32)
    nrm_mat = np.zeros((height, width, 4), dtype=np.float32)
    depth_hit = np.zeros((height, width, 2), dtype=np.float32)
    depth_hit[:, :, 0] = 1e4

    atmo = compiled.dna.zone3_atmosphere
    lit = compiled.dna.lighting
    sky_top = hex_to_rgb_linear(atmo.top_color)
    sky_hor = hex_to_rgb_linear(atmo.horizon_color)
    sky_bot = hex_to_rgb_linear(atmo.bottom_color)
    key_dir = np.array(lit.key_dir, dtype=np.float32)
    key_dir /= max(1e-6, float(np.linalg.norm(key_dir)))
    key_col = hex_to_rgb_linear(lit.key_color)

    for y0 in range(0, height, 256):
        y1 = min(height, y0 + 256)
        ys = 1.0 - 2.0 * ((np.arange(y0, y1, dtype=np.float32) + 0.5) / height)
        xs = 2.0 * ((np.arange(0, width, dtype=np.float32) + 0.5) / width) - 1.0
        gx, gy = np.meshgrid(xs, ys)
        rd = (
            cam_fwd[None, None, :]
            + (gx * aspect * tan_half)[:, :, None] * cam_right[None, None, :]
            + (gy * tan_half)[:, :, None] * cam_up[None, None, :]
        )
        rd /= np.maximum(np.linalg.norm(rd, axis=2, keepdims=True), 1e-7)
        t_sky = np.clip(rd[:, :, 1] * 0.85 + 0.25, 0.0, 1.0)
        sky = sky_bot[None, None, :] * (1.0 - t_sky[:, :, None]) + sky_hor[None, None, :] * t_sky[:, :, None]
        sky = sky * (1.0 - np.clip((t_sky - 0.35) / 0.65, 0.0, 1.0)[:, :, None]) + sky_top[None, None, :] * np.clip((t_sky - 0.35) / 0.65, 0.0, 1.0)[:, :, None]
        hdr_rim[y0:y1, :, :3] = sky

    # Simple active-ray sphere trace over compiled primitives
    ys = 1.0 - 2.0 * ((np.arange(0, height, dtype=np.float32) + 0.5) / height)
    xs = 2.0 * ((np.arange(0, width, dtype=np.float32) + 0.5) / width) - 1.0
    gx, gy = np.meshgrid(xs, ys)
    rd_flat = (
        cam_fwd[None, :]
        + (gx.reshape(-1) * aspect * tan_half)[:, None] * cam_right[None, :]
        + (gy.reshape(-1) * tan_half)[:, None] * cam_up[None, :]
    )
    rd_flat /= np.maximum(np.linalg.norm(rd_flat, axis=1, keepdims=True), 1e-7)

    def _eval_np(pts: np.ndarray) -> np.ndarray:
        d_m = np.full(pts.shape[0], 1e4, dtype=np.float32)
        for ac in compiled.actors:
            d_m, _ = smin_cubic_mat(d_m, sd_sh_shell((pts - ac.cranial_shell.center[None, :]) @ ac.cranial_shell.rot_inv.T, tuple(ac.cranial_shell.radii.tolist()), ac.cranial_shell.sh_coeffs), ac.cranial_shell.smooth_k)
            for b in ac.bones:
                d_b = sd_swept_superspline(pts, b.stations)
                d_m, _ = smin_cubic_mat(d_m, d_b, b.smooth_k)
        for pr in compiled.props:
            for sp in pr.solids:
                d_s = sd_rounded_box((pts - sp.center[None, :]) @ sp.rot_inv.T, tuple(sp.half_extents.tolist()))
                d_m = np.minimum(d_m, d_s)
        return d_m

    ro_flat = np.broadcast_to(cam_pos[None, :], rd_flat.shape)
    act = np.arange(rd_flat.shape[0])
    t_cur = np.full(act.size, 0.05, dtype=np.float32)
    h_flat = np.zeros(rd_flat.shape[0], dtype=bool)
    z_flat = np.full(rd_flat.shape[0], 1e4, dtype=np.float32)
    for _ in range(min(48, max_steps)):
        if act.size == 0:
            break
        d = _eval_np(ro_flat[act] + rd_flat[act] * t_cur[:, None])
        hit = d < 0.006
        if np.any(hit):
            h_flat[act[hit]] = True
            z_flat[act[hit]] = t_cur[hit]
        t_cur += np.maximum(d * 0.9, 0.006)
        keep = (~hit) & (t_cur < 20.0)
        act = act[keep]
        t_cur = t_cur[keep]

    h_idx = np.nonzero(h_flat)[0]
    if h_idx.size > 0:
        hp = ro_flat[h_idx] + rd_flat[h_idx] * z_flat[h_idx, None]
        eps = 0.004
        nx = _eval_np(hp + np.array([eps, 0, 0], dtype=np.float32)) - _eval_np(hp - np.array([eps, 0, 0], dtype=np.float32))
        ny = _eval_np(hp + np.array([0, eps, 0], dtype=np.float32)) - _eval_np(hp - np.array([0, eps, 0], dtype=np.float32))
        nz = _eval_np(hp + np.array([0, 0, eps], dtype=np.float32)) - _eval_np(hp - np.array([0, 0, eps], dtype=np.float32))
        nrm = np.stack([nx, ny, nz], axis=1)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-7)
        diff = np.clip(nrm @ key_dir, 0.2, 1.0)[:, None] * key_col[None, :]
        hdr_rim.reshape(-1, 4)[h_idx, :3] = diff * 0.85
        nrm_mat.reshape(-1, 4)[h_idx, :3] = nrm
        depth_hit.reshape(-1, 2)[h_idx, 0] = z_flat[h_idx]
        depth_hit.reshape(-1, 2)[h_idx, 1] = 1.0

    return hdr_rim, nrm_mat, depth_hit


def render_cpu_gbuffer(
    compiled: CompiledScene,
    buffers: SceneBuffers,
    width: int,
    height: int,
    max_steps: int = 72,
) -> CPUGBuffer:
    """Renders Pass 1 G-Buffer on CPU using parallel Numba JIT or pure NumPy fallback."""
    cam_pos, cam_fwd, cam_right, cam_up, _, tan_half, aspect, focus_depth = compute_camera_basis(compiled, width, height)
    dna = compiled.dna
    lit = dna.lighting
    atmo = dna.zone3_atmosphere
    mid = dna.zone2_midground
    style_int = 0 if dna.style_mode == "anime_npr" else 1

    hdr_rim = np.zeros((height, width, 4), dtype=np.float32)
    nrm_mat = np.zeros((height, width, 4), dtype=np.float32)
    depth_hit = np.zeros((height, width, 2), dtype=np.float32)

    kd = np.array(lit.key_dir, dtype=np.float32)
    kd /= max(1e-6, float(np.linalg.norm(kd)))

    if HAS_NUMBA:
        _nb_render_pass1_rows(
            width, height,
            cam_pos, cam_fwd, cam_right, cam_up,
            float(tan_half), float(aspect), style_int, int(max_steps),
            buffers.n_clusters, buffers.n_sh, buffers.n_garments, buffers.n_midground, buffers.n_emitters,
            buffers.clusters_f32, buffers.splines_f32, buffers.stations_f32,
            buffers.sh_f32, buffers.solids_f32, buffers.garments_f32,
            buffers.decals_f32, buffers.midground_f32, buffers.emitters_f32,
            kd, hex_to_rgb_linear(lit.key_color), hex_to_rgb_linear(lit.shadow_tint),
            hex_to_rgb_linear(lit.ambient_color), hex_to_rgb_linear(lit.rim_color),
            float(lit.rim_strength), float(lit.sss_strength), hex_to_rgb_linear(lit.sss_color),
            hex_to_rgb_linear(atmo.top_color), hex_to_rgb_linear(atmo.horizon_color), hex_to_rgb_linear(atmo.bottom_color),
            1.0 if atmo.ground_enabled else 0.0, float(atmo.ground_y),
            hex_to_rgb_linear(atmo.ground_color), hex_to_rgb_linear(atmo.ground_secondary_color),
            hex_to_rgb_linear(mid.haze_tint), float(mid.haze_strength),
            hdr_rim, nrm_mat, depth_hit,
        )
    else:
        hdr_rim, nrm_mat, depth_hit = _render_cpu_numpy_fallback(
            compiled, width, height, cam_pos, cam_fwd, cam_right, cam_up, tan_half, aspect, max_steps
        )

    em0 = atmo.emitters[0]
    sun_dir = np.array(em0.direction_or_pos, dtype=np.float32)
    sun_dir /= max(1e-6, float(np.linalg.norm(sun_dir)))
    fwd_z = max(0.15, float(np.dot(sun_dir, cam_fwd)))
    sx = float(np.dot(sun_dir, cam_right)) / (fwd_z * tan_half * aspect)
    sy = float(np.dot(sun_dir, cam_up)) / (fwd_z * tan_half)
    # Top-down image UV for CPU post-FX (note ny=+1 at y=0)
    sun_uv = (0.5 + 0.5 * sx, 0.5 - 0.5 * sy)

    return CPUGBuffer(
        width=width,
        height=height,
        hdr_rim=hdr_rim,
        nrm_mat=nrm_mat,
        depth_hit=depth_hit,
        focus_depth=focus_depth,
        sun_uv=sun_uv,
        sun_col=hex_to_rgb_linear(em0.color),
        godray_strength=float(em0.god_ray_strength),
    )
