"""
PrismStudio: Standalone Compositional Pure-Math Natural-Language-to-Image Engine.

Provides a clean end-to-end API from natural language prompt -> SceneDNA ->
Deterministic Bone-Graph / Body-UV / 3-Zone Compiler -> ModernGL GPU or fast
Numba/NumPy CPU renderer -> optional VLM draft critique -> final PNG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .compiler import CompiledScene, pack_scene_buffers, solve_and_compile_scene, validate_and_repair_dna
from .director import (
    CritiqueResult,
    DirectorAPIError,
    DirectorPlanResult,
    PRESET_SCENES,
    build_offline_director_scene,
    critique_and_patch_scene,
    load_director_env,
    plan_scene_dna,
)
from .engine import RenderOutput, render_scene_dna, rgb_to_png_bytes
from .schema import SceneDNA, ScenePatch, apply_scene_patch, get_director_json_schema, scene_dna_from_dict


@dataclass
class GenerationResult:
    png_bytes: bytes
    rgb_array: np.ndarray
    dna: SceneDNA
    plan_result: DirectorPlanResult
    render_output: RenderOutput
    critique: Optional[CritiqueResult] = None
    notices: List[str] = field(default_factory=list)


def generate_image(
    prompt: str,
    *,
    style_mode: str = "anime_npr",
    width: int = 960,
    height: int = 720,
    use_vlm_critique: bool = False,
    prefer_backend: str = "auto",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    vision_model: Optional[str] = None,
    allow_fallback: bool = True,
) -> GenerationResult:
    """One-call natural-language-to-image synthesis via compositional pure math."""
    plan = plan_scene_dna(
        prompt,
        style_mode=style_mode,
        provider=provider,
        model=model,
        allow_fallback=allow_fallback,
    )
    notices = list(plan.warnings)
    dna_final = plan.dna
    critique_res: Optional[CritiqueResult] = None

    if use_vlm_critique:
        critique_res = critique_and_patch_scene(
            dna_final,
            prompt,
            provider=provider,
            vision_model=vision_model,
            prefer_backend=prefer_backend,
        )
        dna_final = critique_res.patched_dna
        if critique_res.notice:
            notices.append(critique_res.notice)

    render_out = render_scene_dna(dna_final, width=width, height=height, prefer_backend=prefer_backend)
    if render_out.backend_notice:
        notices.append(render_out.backend_notice)

    return GenerationResult(
        png_bytes=render_out.png_bytes,
        rgb_array=render_out.rgb_array,
        dna=dna_final,
        plan_result=plan,
        render_output=render_out,
        critique=critique_res,
        notices=notices,
    )


__all__ = [
    "CompiledScene",
    "CritiqueResult",
    "DirectorAPIError",
    "DirectorPlanResult",
    "GenerationResult",
    "PRESET_SCENES",
    "RenderOutput",
    "SceneDNA",
    "ScenePatch",
    "apply_scene_patch",
    "build_offline_director_scene",
    "critique_and_patch_scene",
    "generate_image",
    "get_director_json_schema",
    "load_director_env",
    "pack_scene_buffers",
    "plan_scene_dna",
    "render_scene_dna",
    "rgb_to_png_bytes",
    "scene_dna_from_dict",
    "solve_and_compile_scene",
    "validate_and_repair_dna",
]
