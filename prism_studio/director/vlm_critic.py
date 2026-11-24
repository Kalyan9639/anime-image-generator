"""
Closed-Loop Vision-Language Model (VLM) Draft Self-Critique Pass for PrismStudio.

Renders a fast 320x240 draft image, sends it + the user prompt to a Groq or
Ollama vision model to diagnose concrete framing, contact, or lighting issues,
and applies bounded `ScenePatch` refinements before the final HD render.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..engine.backend_manager import render_scene_dna
from ..schema.patch_schema import ScenePatch, apply_scene_patch, get_patch_json_schema, patch_from_dict
from ..schema.scene_dna import SceneDNA
from .llm_client import call_vision_llm, extract_json_object, load_director_env


VLM_CRITIC_SYSTEM = """You are a senior Anime Cinematography & 3D Technical Supervisor.
Inspect the 320x240 draft render against the user prompt. Identify ONLY concrete visual
flaws (e.g., subject too small/far, camera too high/low, rim light weak, prop wall too short,
tail/exposure off) and output a bounded ScenePatch JSON delta. Keep deltas conservative."""


@dataclass
class CritiqueResult:
    patched_dna: SceneDNA
    patch: ScenePatch
    draft_rgb: np.ndarray
    draft_png: bytes
    applied: bool
    notice: str = ""
    warnings: List[str] = field(default_factory=list)


def critique_and_patch_scene(
    dna: SceneDNA,
    prompt: str,
    *,
    provider: Optional[str] = None,
    host: Optional[str] = None,
    vision_model: Optional[str] = None,
    api_key: Optional[str] = None,
    prefer_backend: str = "auto",
    timeout: int = 60,
) -> CritiqueResult:
    """Renders a 320x240 draft and invokes a VLM critic to apply a bounded ScenePatch."""
    draft = render_scene_dna(dna, width=320, height=240, max_steps=64, prefer_backend=prefer_backend)
    cfg = load_director_env()
    eff_prov = (provider or cfg["provider"]).strip().lower()
    if eff_prov not in ("groq", "ollama"):
        eff_prov = "groq"

    if eff_prov == "groq":
        eff_host = (host or cfg["groq_host"]).strip()
        eff_model = (vision_model or cfg["groq_vision_model"]).strip()
        eff_key = api_key if api_key is not None else cfg["groq_api_key"]
    else:
        eff_host = (host or cfg["ollama_host"]).strip()
        eff_model = (vision_model or cfg["ollama_vision_model"]).strip()
        eff_key = api_key if api_key is not None else cfg["ollama_api_key"]

    user_msg = (
        f"Original Prompt: {prompt or dna.prompt}\n"
        f"Style Mode: {dna.style_mode}\n"
        f"Current Camera: azimuth={dna.camera.azimuth_deg:.1f}, elevation={dna.camera.elevation_deg:.1f}, "
        f"distance_scale={dna.camera.distance_scale:.2f}\n"
        "Inspect the draft image carefully and emit a ScenePatch JSON improving composition, lighting, or proportions."
    )

    try:
        raw = call_vision_llm(
            VLM_CRITIC_SYSTEM,
            user_msg,
            draft.png_bytes,
            provider=eff_prov,
            host=eff_host,
            model=eff_model,
            api_key=eff_key,
            schema=get_patch_json_schema(),
            timeout=timeout,
        )
        patch = patch_from_dict(extract_json_object(raw))
        patched = apply_scene_patch(dna, patch)
        return CritiqueResult(
            patched_dna=patched,
            patch=patch,
            draft_rgb=draft.rgb_array,
            draft_png=draft.png_bytes,
            applied=True,
            notice=f"VLM Critique ({eff_prov}/{eff_model}): {patch.summary or 'Applied bounded refinement.'}",
            warnings=draft.warnings,
        )
    except Exception as exc:
        notice = f"VLM Critique skipped ({exc}); proceeding with deterministic validated SceneDNA."
        return CritiqueResult(
            patched_dna=dna,
            patch=ScenePatch(summary="Skipped"),
            draft_rgb=draft.rgb_array,
            draft_png=draft.png_bytes,
            applied=False,
            notice=notice,
            warnings=draft.warnings + [notice],
        )
