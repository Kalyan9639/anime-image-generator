"""
Dynamic Multi-Layer Semantic Scene Engine (Track 1).

Combines:
- Pre-written dynamic mathematical geometry (SDF, Superquadrics, Johan Gielis
  Superformula, Convolution Limbs, Lathe/Bézier Sweeps, L-Systems, CSG, BVH)
- Procedural trig/noise surface shaders (tartan plaid, cable-knit, tiles, wood, metal, PCB)
- Qualitative Spatial Constraint Solver (Holodeck / ShapeAssembly style)
- 2.5D/3D Optical Compositor (soft shadows, AO, Fresnel rim, bokeh, snowflakes, DOF, bloom)
- Ollama Cloud (`gpt-oss:20b-cloud`) / Local Ollama Text-Only Knowledge Planner
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .compositor import composite_scene_to_rgb, rgb_to_png_bytes
from .parser import AsciiBoxNode, AsciiLayout, parse_semantic_ascii
from .planner import (
    DEFAULT_GROQ_HOST,
    DEFAULT_GROQ_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_PROVIDER,
    PlannerAPIError,
    PlanResult,
    build_offline_fallback_scene,
    load_env_config,
    plan_scene,
    plan_scene_with_ollama,
)
from .renderer_2d import render_scene_svg
from .renderer_3d import GBuffer, render_scene_gbuffer
from .schema import (
    BackgroundSpec,
    CameraSpec,
    EntitySpec,
    EnvironmentSpec,
    LightingSpec,
    MaterialSpec,
    ModifierSpec,
    PartNode,
    SceneGraph,
    get_ollama_scene_json_schema,
    scene_from_dict,
)
from .solver import CompiledEntity, CompiledPart, ResolvedScene, solve_scene


@dataclass
class RenderOutput:
    png_bytes: bytes
    rgb_array: np.ndarray
    svg_content: str
    resolved: ResolvedScene
    render_seconds: float


def render_scene_graph(
    scene: SceneGraph,
    *,
    width: int = 640,
    height: int = 480,
    max_steps: int = 72,
    include_svg: bool = True,
) -> RenderOutput:
    """
    Deterministically solves constraints, raymarches 3D layers, composites optical
    background/bokeh/particles/DOF/bloom, and optionally exports 2D vector SVG.
    """
    t0 = time.perf_counter()
    resolved = solve_scene(scene)
    gbuf = render_scene_gbuffer(resolved, width=width, height=height, max_steps=max_steps)
    rgb = composite_scene_to_rgb(
        gbuf,
        bg=scene.background,
        lit=scene.lighting,
        dof_strength=scene.camera.dof_strength,
        seed=scene.seed,
    )
    png_bytes = rgb_to_png_bytes(rgb)
    svg = render_scene_svg(resolved, width=width, height=height) if include_svg else ""
    elapsed = time.perf_counter() - t0
    return RenderOutput(
        png_bytes=png_bytes,
        rgb_array=rgb,
        svg_content=svg,
        resolved=resolved,
        render_seconds=elapsed,
    )


__all__ = [
    "AsciiBoxNode",
    "AsciiLayout",
    "BackgroundSpec",
    "CameraSpec",
    "CompiledEntity",
    "CompiledPart",
    "DEFAULT_GROQ_HOST",
    "DEFAULT_GROQ_MODEL",
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_PROVIDER",
    "EntitySpec",
    "EnvironmentSpec",
    "GBuffer",
    "LightingSpec",
    "MaterialSpec",
    "ModifierSpec",
    "PartNode",
    "PlanResult",
    "PlannerAPIError",
    "RenderOutput",
    "ResolvedScene",
    "SceneGraph",
    "build_offline_fallback_scene",
    "composite_scene_to_rgb",
    "get_ollama_scene_json_schema",
    "load_env_config",
    "parse_semantic_ascii",
    "plan_scene",
    "plan_scene_with_ollama",
    "render_scene_gbuffer",
    "render_scene_graph",
    "render_scene_svg",
    "rgb_to_png_bytes",
    "scene_from_dict",
    "solve_scene",
]
