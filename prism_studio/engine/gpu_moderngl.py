"""
Headless ModernGL OpenGL 3.3+ GPU Renderer for PrismStudio.

Uses a singleton standalone WGL/EGL OpenGL context (executing natively on
Intel UHD Graphics in ~0.08-0.35s with <30 MB RAM), rendering Pass 1 MRT
G-Buffers and Pass 2 6-stage HDR optical post-processing directly to RGB uint8.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..compiler.bytecode import SceneBuffers
from ..compiler.layout_solver import CompiledScene
from ..math_ops.materials import hex_to_rgb_linear


FULLSCREEN_VERT = """
#version 330 core
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""


def compute_camera_basis(
    compiled: CompiledScene,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Shared deterministic camera framing shared identically by GPU and CPU backends.
    Auto-frames around the hero actor + support prop so subjects fill 55-75% cleanly.
    """
    cam = compiled.dna.camera
    fov_rad = math.radians(cam.fov_deg)
    tan_half = math.tan(fov_rad * 0.5)
    aspect = float(width) / max(1.0, float(height))

    if compiled.actors:
        hero = compiled.actors[0]
        hero_r = max(0.45, float(hero.bvh_radius))
        base_dist = (hero_r / max(0.18, math.sin(fov_rad * 0.5))) * 1.12
    else:
        ext = np.maximum(compiled.scene_bmax - compiled.scene_bmin, 0.8)
        hero_r = float(np.linalg.norm(ext) * 0.45)
        base_dist = (hero_r / max(0.18, math.sin(fov_rad * 0.5))) * 1.15

    shot_mul = {"close_up": 0.64, "full_body": 1.15, "low_angle_hero": 0.98, "high_angle_ledge": 1.08}.get(cam.shot_type, 1.0)
    dist = max(1.35, min(6.5, base_dist * shot_mul * float(cam.distance_scale)))

    el_deg = float(cam.elevation_deg)
    if cam.shot_type == "low_angle_hero":
        el_deg = min(el_deg, 5.0)
    elif cam.shot_type == "high_angle_ledge":
        el_deg = max(el_deg, 18.0)

    az = math.radians(cam.azimuth_deg)
    el = math.radians(el_deg)
    cam_dir = np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)], dtype=np.float32)

    target = compiled.camera_target.copy()
    if cam.shot_type == "close_up" and compiled.actors:
        target = compiled.focus_point.copy()

    cam_pos = (target + cam_dir * dist).astype(np.float32)
    cam_fwd = (-cam_dir / max(1e-6, float(np.linalg.norm(cam_dir)))).astype(np.float32)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    cam_right = np.cross(cam_fwd, world_up)
    rn = float(np.linalg.norm(cam_right))
    cam_right = (cam_right / rn).astype(np.float32) if rn > 1e-5 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cam_up = np.cross(cam_right, cam_fwd).astype(np.float32)

    if abs(cam.roll_deg) > 1e-3:
        rr = math.radians(cam.roll_deg)
        cr, sr = math.cos(rr), math.sin(rr)
        cam_right, cam_up = (cr * cam_right + sr * cam_up).astype(np.float32), (-sr * cam_right + cr * cam_up).astype(np.float32)

    focus_depth = float(np.linalg.norm(compiled.focus_point - cam_pos))
    return cam_pos, cam_fwd, cam_right, cam_up, target, tan_half, aspect, max(0.4, focus_depth)


class ModernGLRenderer:
    """Singleton headless OpenGL 3.3+ renderer compiling shaders once per process."""

    _instance: Optional["ModernGLRenderer"] = None

    def __init__(self) -> None:
        import moderngl  # local import so CPU-only environments can import package cleanly

        self.mgl = moderngl
        self.ctx = moderngl.create_context(standalone=True, require=330)
        shader_dir = Path(__file__).resolve().parent
        p1_src = (shader_dir / "glsl_Pass1_raymarch.glsl").read_text(encoding="utf-8")
        p2_src = (shader_dir / "glsl_Pass2_optics.glsl").read_text(encoding="utf-8")

        self.prog1 = self.ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=p1_src)
        self.prog2 = self.ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=p2_src)

        quad = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao1 = self.ctx.simple_vertex_array(self.prog1, self.vbo, "in_pos")
        self.vao2 = self.ctx.simple_vertex_array(self.prog2, self.vbo, "in_pos")
        self.gl_renderer = str(self.ctx.info.get("GL_RENDERER", "OpenGL 3.3"))

    @classmethod
    def get_instance(cls) -> "ModernGLRenderer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def render(
        self,
        compiled: CompiledScene,
        buffers: SceneBuffers,
        width: int,
        height: int,
        max_steps: int = 84,
    ) -> np.ndarray:
        """Executes Pass 1 (Raymarch G-Buffer) and Pass 2 (6-Stage HDR Optics) on GPU."""
        tex_h, tex_w, _ = buffers.packed_rgba32f.shape
        data_tex = self.ctx.texture((tex_w, tex_h), 4, data=buffers.packed_rgba32f.tobytes(), dtype="f4")
        data_tex.filter = (self.mgl.NEAREST, self.mgl.NEAREST)
        data_tex.use(location=0)

        # Allocate Pass 1 MRT textures
        t_hdr = self.ctx.texture((width, height), 4, dtype="f2")
        t_nrm = self.ctx.texture((width, height), 4, dtype="f2")
        t_dep = self.ctx.texture((width, height), 2, dtype="f4")
        for t in (t_hdr, t_nrm, t_dep):
            t.filter = (self.mgl.LINEAR, self.mgl.LINEAR)
            t.repeat_x = False
            t.repeat_y = False

        fbo1 = self.ctx.framebuffer(color_attachments=[t_hdr, t_nrm, t_dep])
        t_out = self.ctx.texture((width, height), 4, dtype="f1")
        fbo2 = self.ctx.framebuffer(color_attachments=[t_out])

        cam_pos, cam_fwd, cam_right, cam_up, _, tan_half, aspect, focus_depth = compute_camera_basis(compiled, width, height)
        dna = compiled.dna
        lit = dna.lighting
        atmo = dna.zone3_atmosphere
        mid = dna.zone2_midground
        style_int = 0 if dna.style_mode == "anime_npr" else 1

        def _set_u(prog, name, val):
            if name in prog:
                prog[name].value = val

        # Populate Pass 1 uniforms
        _set_u(self.prog1, "u_data", 0)
        _set_u(self.prog1, "u_res", (float(width), float(height)))
        _set_u(self.prog1, "u_cam_pos", tuple(cam_pos.tolist()))
        _set_u(self.prog1, "u_cam_fwd", tuple(cam_fwd.tolist()))
        _set_u(self.prog1, "u_cam_right", tuple(cam_right.tolist()))
        _set_u(self.prog1, "u_cam_up", tuple(cam_up.tolist()))
        _set_u(self.prog1, "u_tan_half_fov", float(tan_half))
        _set_u(self.prog1, "u_aspect", float(aspect))
        _set_u(self.prog1, "u_focus_depth", float(focus_depth))
        _set_u(self.prog1, "u_style_mode", style_int)
        _set_u(self.prog1, "u_max_steps", int(max_steps))

        _set_u(self.prog1, "u_n_clusters", buffers.n_clusters)
        _set_u(self.prog1, "u_n_splines", buffers.n_splines)
        _set_u(self.prog1, "u_n_sh", buffers.n_sh)
        _set_u(self.prog1, "u_n_solids", buffers.n_solids)
        _set_u(self.prog1, "u_n_garments", buffers.n_garments)
        _set_u(self.prog1, "u_n_decals", buffers.n_decals)
        _set_u(self.prog1, "u_n_midground", buffers.n_midground)
        _set_u(self.prog1, "u_n_emitters", buffers.n_emitters)

        for k, off in buffers.offsets_vec4.items():
            _set_u(self.prog1, f"u_off_{k}", int(off))

        kd = np.array(lit.key_dir, dtype=np.float32)
        kd /= max(1e-6, float(np.linalg.norm(kd)))
        _set_u(self.prog1, "u_key_dir", tuple(kd.tolist()))
        _set_u(self.prog1, "u_key_col", tuple(hex_to_rgb_linear(lit.key_color).tolist()))
        _set_u(self.prog1, "u_shadow_tint", tuple(hex_to_rgb_linear(lit.shadow_tint).tolist()))
        _set_u(self.prog1, "u_ambient_col", tuple(hex_to_rgb_linear(lit.ambient_color).tolist()))
        _set_u(self.prog1, "u_rim_col", tuple(hex_to_rgb_linear(lit.rim_color).tolist()))
        _set_u(self.prog1, "u_rim_strength", float(lit.rim_strength))
        _set_u(self.prog1, "u_sss_strength", float(lit.sss_strength))
        _set_u(self.prog1, "u_sss_col", tuple(hex_to_rgb_linear(lit.sss_color).tolist()))

        _set_u(self.prog1, "u_sky_top", tuple(hex_to_rgb_linear(atmo.top_color).tolist()))
        _set_u(self.prog1, "u_sky_horizon", tuple(hex_to_rgb_linear(atmo.horizon_color).tolist()))
        _set_u(self.prog1, "u_sky_bottom", tuple(hex_to_rgb_linear(atmo.bottom_color).tolist()))
        _set_u(self.prog1, "u_sky_glow", tuple(hex_to_rgb_linear(atmo.glow_color).tolist()))
        _set_u(self.prog1, "u_ground_enabled", 1.0 if atmo.ground_enabled else 0.0)
        _set_u(self.prog1, "u_ground_y", float(atmo.ground_y))
        _set_u(self.prog1, "u_ground_col1", tuple(hex_to_rgb_linear(atmo.ground_color).tolist()))
        _set_u(self.prog1, "u_ground_col2", tuple(hex_to_rgb_linear(atmo.ground_secondary_color).tolist()))
        _set_u(self.prog1, "u_haze_col", tuple(hex_to_rgb_linear(mid.haze_tint).tolist()))
        _set_u(self.prog1, "u_haze_strength", float(mid.haze_strength))

        fbo1.use()
        self.vao1.render(self.mgl.TRIANGLE_STRIP)

        # Project primary emitter direction onto 2D screen UV for god-rays
        em0 = atmo.emitters[0]
        sun_dir = np.array(em0.direction_or_pos, dtype=np.float32)
        sun_dir /= max(1e-6, float(np.linalg.norm(sun_dir)))
        fwd_z = max(0.15, float(np.dot(sun_dir, cam_fwd)))
        sx = float(np.dot(sun_dir, cam_right)) / (fwd_z * tan_half * aspect)
        sy = float(np.dot(sun_dir, cam_up)) / (fwd_z * tan_half)
        sun_uv = (0.5 + 0.5 * sx, 0.5 + 0.5 * sy)

        t_hdr.use(location=0)
        t_nrm.use(location=1)
        t_dep.use(location=2)

        _set_u(self.prog2, "u_hdr_rim", 0)
        _set_u(self.prog2, "u_nrm_mat", 1)
        _set_u(self.prog2, "u_depth_hit", 2)
        _set_u(self.prog2, "u_res", (float(width), float(height)))
        _set_u(self.prog2, "u_focus_depth", float(focus_depth))
        _set_u(self.prog2, "u_dof_strength", float(dna.camera.dof_strength))
        _set_u(self.prog2, "u_ink_strength", float(lit.ink_strength))
        _set_u(self.prog2, "u_sun_uv", sun_uv)
        _set_u(self.prog2, "u_sun_col", tuple(hex_to_rgb_linear(em0.color).tolist()))
        _set_u(self.prog2, "u_godray_strength", float(em0.god_ray_strength))
        _set_u(self.prog2, "u_bloom_intensity", float(lit.bloom_intensity))
        _set_u(self.prog2, "u_chrom_ab", float(lit.chromatic_aberration))
        _set_u(self.prog2, "u_film_grain", float(lit.film_grain))
        _set_u(self.prog2, "u_exposure", float(lit.exposure))
        _set_u(self.prog2, "u_style_mode", style_int)
        _set_u(self.prog2, "u_bokeh_count", int(atmo.bokeh_count))
        _set_u(self.prog2, "u_dust_count", int(atmo.dust_motes_count))
        _set_u(self.prog2, "u_seed", int(dna.seed))

        fbo2.use()
        self.vao2.render(self.mgl.TRIANGLE_STRIP)

        raw = np.frombuffer(t_out.read(), dtype=np.uint8).reshape(height, width, 4)
        # Flip OpenGL bottom-up origin to standard top-down image layout
        rgb = np.flipud(raw[:, :, :3]).copy()

        # Release temporary per-render buffers immediately to keep RAM flat
        for obj in (fbo1, fbo2, t_hdr, t_nrm, t_dep, t_out, data_tex):
            obj.release()
        return rgb
