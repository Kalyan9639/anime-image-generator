"""
2D Crisp Vector SVG Renderer for Dynamic Multi-Layer Scenes (Track 1).

Projects resolved 3D scene layers (background gradient, bokeh discs, environment
horizon, and depth-sorted mathematical entity parts) into clean, editable,
infinitely scalable SVG markup using quadratic Bézier / Gielis superformula curves.
"""

from __future__ import annotations

import html
import math
from typing import List, Tuple

import numpy as np

from .math_core import gielis_radius
from .renderer_3d import _build_camera_rays
from .solver import CompiledPart, ResolvedScene


def _project_point(
    pt: np.ndarray,
    cam_pos: np.ndarray,
    cam_right: np.ndarray,
    cam_up: np.ndarray,
    cam_fwd: np.ndarray,
    fov_deg: float,
    width: int,
    height: int,
    is_ortho: bool,
    ortho_scale: float,
) -> Tuple[float, float, float]:
    rel = pt - cam_pos
    z = float(np.dot(rel, cam_fwd))
    x_cam = float(np.dot(rel, cam_right))
    y_cam = float(np.dot(rel, cam_up))
    aspect = width / max(1.0, float(height))

    if is_ortho:
        nx = x_cam / max(1e-4, ortho_scale * aspect)
        ny = y_cam / max(1e-4, ortho_scale)
    else:
        tan_half = math.tan(math.radians(fov_deg) * 0.5)
        denom = max(0.25, z) * tan_half
        nx = x_cam / (denom * aspect)
        ny = y_cam / denom

    sx = (nx * 0.5 + 0.5) * width
    sy = (0.5 - ny * 0.5) * height
    return sx, sy, z


def _gielis_svg_path(cx: float, cy: float, rx: float, ry: float, m: float, n1: float, n2: float, n3: float) -> str:
    angles = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False, dtype=np.float32)
    r = gielis_radius(angles, m=m, n1=n1, n2=n2, n3=n3)
    r /= max(1e-4, float(np.max(r)))
    xs = cx + r * np.cos(angles) * rx
    ys = cy + r * np.sin(angles) * ry
    pts = [f"{xs[i]:.1f},{ys[i]:.1f}" for i in range(len(angles))]
    return "M " + " L ".join(pts) + " Z"


def render_scene_svg(
    resolved: ResolvedScene,
    width: int = 800,
    height: int = 600,
) -> str:
    """Renders a depth-sorted, clean 2D vector SVG illustration of the resolved scene."""
    _, _, cam_pos, cam_target, cam_right, cam_up, cam_fwd, _ = _build_camera_rays(resolved, width, height)
    cam = resolved.scene.camera
    bg = resolved.scene.background
    env = resolved.scene.environment
    is_ortho = cam.projection in ("isometric", "orthographic")
    ext = np.maximum(resolved.scene_bmax - resolved.scene_bmin, 0.6)
    ortho_scale = float(np.linalg.norm(ext) * 0.5 * 1.28 * cam.distance_scale)

    svg_parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "  <defs>",
        '    <linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">',
        f'      <stop offset="0%" stop-color="{html.escape(bg.top_color)}"/>',
        f'      <stop offset="100%" stop-color="{html.escape(bg.bottom_color)}"/>',
        "    </linearGradient>",
        '    <radialGradient id="glowGrad" cx="55%" cy="48%" r="55%">',
        f'      <stop offset="0%" stop-color="{html.escape(bg.glow_color)}" stop-opacity="0.45"/>',
        f'      <stop offset="100%" stop-color="{html.escape(bg.glow_color)}" stop-opacity="0.0"/>',
        "    </radialGradient>",
        '    <filter id="softBlur"><feGaussianBlur stdDeviation="4"/></filter>',
        "  </defs>",
        f'  <rect width="{width}" height="{height}" fill="url(#bgGrad)"/>',
        f'  <rect width="{width}" height="{height}" fill="url(#glowGrad)"/>',
    ]

    # Bokeh backdrop circles in SVG
    bokeh_n = min(45, bg.bokeh_count if bg.bokeh_count > 0 else (30 if bg.style == "bokeh_night" else 0))
    if bokeh_n > 0:
        rng = np.random.default_rng(resolved.scene.seed + 701)
        bcols = bg.bokeh_colors or ["#fde68a", "#fbbf24"]
        svg_parts.append('  <g opacity="0.55" filter="url(#softBlur)">')
        for i in range(bokeh_n):
            bx = float(rng.uniform(0.0, width))
            by = float(rng.uniform(0.0, height * 0.88))
            br = float(rng.uniform(14.0, 44.0) * bg.bokeh_size)
            bc = html.escape(bcols[i % len(bcols)])
            op = float(rng.uniform(0.25, 0.65))
            svg_parts.append(f'    <circle cx="{bx:.1f}" cy="{by:.1f}" r="{br:.1f}" fill="{bc}" opacity="{op:.2f}"/>')
        svg_parts.append("  </g>")

    # Environment horizon band
    if env.enabled and env.kind != "none":
        _, hy, _ = _project_point(
            np.array([cam_target[0], env.y_level, cam_target[2] - 4.0], dtype=np.float32),
            cam_pos, cam_right, cam_up, cam_fwd, cam.fov_deg, width, height, is_ortho, ortho_scale,
        )
        hy = max(height * 0.45, min(height * 0.85, hy))
        svg_parts.append(
            f'  <rect x="0" y="{hy:.1f}" width="{width}" height="{height - hy:.1f}" '
            f'fill="{html.escape(env.primary_color)}" opacity="0.88"/>'
        )

    # Depth-sort all parts from back to front (painter's algorithm)
    all_parts: List[Tuple[float, CompiledPart]] = []
    for ent in resolved.entities:
        for p in ent.parts:
            if p.csg_op == "subtract":
                continue
            _, _, pz = _project_point(
                p.center, cam_pos, cam_right, cam_up, cam_fwd, cam.fov_deg, width, height, is_ortho, ortho_scale
            )
            all_parts.append((pz, p))
    all_parts.sort(key=lambda item: item[0], reverse=True)

    svg_parts.append("  <g>")
    for _, part in all_parts:
        # Project all 8 corners of part's world AABB to get clean 2D screen bounds
        corners = np.array(
            [
                [part.bmin[0], part.bmin[1], part.bmin[2]],
                [part.bmax[0], part.bmin[1], part.bmin[2]],
                [part.bmin[0], part.bmax[1], part.bmin[2]],
                [part.bmax[0], part.bmax[1], part.bmin[2]],
                [part.bmin[0], part.bmin[1], part.bmax[2]],
                [part.bmax[0], part.bmin[1], part.bmax[2]],
                [part.bmin[0], part.bmax[1], part.bmax[2]],
                [part.bmax[0], part.bmax[1], part.bmax[2]],
            ],
            dtype=np.float32,
        )
        proj = [
            _project_point(c, cam_pos, cam_right, cam_up, cam_fwd, cam.fov_deg, width, height, is_ortho, ortho_scale)
            for c in corners
        ]
        xs = [pt[0] for pt in proj]
        ys = [pt[1] for pt in proj]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        pw = max(4.0, x1 - x0)
        ph = max(4.0, y1 - y0)
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        fill = html.escape(part.material.color)
        stroke = html.escape(part.material.secondary_color)

        if part.primitive in ("sphere", "ellipsoid"):
            svg_parts.append(
                f'    <ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{pw * 0.48:.1f}" ry="{ph * 0.48:.1f}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            )
        elif part.primitive == "cone":
            svg_parts.append(
                f'    <polygon points="{cx:.1f},{y0:.1f} {x1:.1f},{y1:.1f} {x0:.1f},{y1:.1f}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            )
        elif part.primitive == "gielis_supershape":
            d_path = _gielis_svg_path(
                cx, cy, pw * 0.48, ph * 0.48,
                m=float(part.params.get("m", 6.0)),
                n1=float(part.params.get("n1", 1.0)),
                n2=float(part.params.get("n2", 1.7)),
                n3=float(part.params.get("n3", 1.7)),
            )
            svg_parts.append(f'    <path d="{d_path}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        elif part.primitive == "torus":
            svg_parts.append(
                f'    <ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{pw * 0.45:.1f}" ry="{ph * 0.35:.1f}" '
                f'fill="none" stroke="{fill}" stroke-width="{max(3.0, min(pw, ph) * 0.16):.1f}"/>'
            )
        else:
            rx = min(pw, ph) * (0.28 if part.primitive in ("superquadric", "capsule", "convolution_limb") else 0.12)
            svg_parts.append(
                f'    <rect x="{x0:.1f}" y="{y0:.1f}" width="{pw:.1f}" height="{ph:.1f}" rx="{rx:.1f}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            )

    svg_parts.append("  </g>")
    svg_parts.append("</svg>")
    return "\n".join(svg_parts)
