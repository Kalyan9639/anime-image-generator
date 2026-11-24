"""
Dual-Backend Execution Manager (ModernGL GPU + Explicit-Notify Fast CPU Fallback).

Routes rendering to the headless ModernGL OpenGL 3.3+ GPU kernel first (Intel UHD
Graphics, ~0.08-0.35s). Whenever GPU context initialization fails or CPU fallback
is forced, it ALWAYS notifies the user explicitly via RenderOutput.backend_notice
and Python warnings while executing on the fast Numba/NumPy CPU pipeline.
"""

from __future__ import annotations

import io
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PIL import Image

from ..compiler.bytecode import pack_scene_buffers
from ..compiler.layout_solver import CompiledScene, solve_and_compile_scene
from ..schema.scene_dna import SceneDNA
from .cpu_kernel import HAS_NUMBA, render_cpu_gbuffer
from .gpu_moderngl import ModernGLRenderer
from .optics_cpu import composite_cpu_optics


@dataclass
class RenderOutput:
    png_bytes: bytes
    rgb_array: np.ndarray
    compiled: CompiledScene
    render_seconds: float
    backend_used: str                  # "moderngl_gpu" | "cpu_numba" | "cpu_numpy"
    backend_notice: str = ""           # Non-empty explicit notification whenever falling back to CPU
    warnings: List[str] = field(default_factory=list)


def rgb_to_png_bytes(rgb_u8: np.ndarray) -> bytes:
    img = Image.fromarray(rgb_u8, mode="RGB")
    bio = io.BytesIO()
    img.save(bio, format="PNG", optimize=True)
    return bio.getvalue()


def render_scene_dna(
    dna: SceneDNA,
    *,
    width: int = 960,
    height: int = 720,
    max_steps: int = 84,
    prefer_backend: str = "auto",
) -> RenderOutput:
    """
    Compiles SceneDNA deterministically and renders via ModernGL GPU or fast CPU fallback.
    """
    t0 = time.perf_counter()
    compiled = solve_and_compile_scene(dna)
    buffers = pack_scene_buffers(compiled)

    backend_used = ""
    backend_notice = ""
    all_warnings = list(compiled.diagnostics)

    rgb: Optional[np.ndarray] = None
    if prefer_backend in ("auto", "gpu", "moderngl"):
        try:
            gpu = ModernGLRenderer.get_instance()
            rgb = gpu.render(compiled, buffers, width=width, height=height, max_steps=max_steps)
            backend_used = f"moderngl_gpu ({gpu.gl_renderer})"
        except Exception as exc:
            backend_notice = (
                f"[Notice] ModernGL GPU backend unavailable ({exc}); "
                f"falling back to fast {'Numba multi-core' if HAS_NUMBA else 'vectorized NumPy'} CPU renderer."
            )
            warnings.warn(backend_notice, RuntimeWarning, stacklevel=2)
            all_warnings.append(backend_notice)
    else:
        backend_notice = (
            f"[Notice] CPU fallback explicitly selected; "
            f"running on {'Numba multi-core' if HAS_NUMBA else 'vectorized NumPy'} CPU renderer."
        )
        all_warnings.append(backend_notice)

    if rgb is None:
        gbuf = render_cpu_gbuffer(compiled, buffers, width=width, height=height, max_steps=min(max_steps, 68))
        rgb = composite_cpu_optics(gbuf, compiled.dna)
        backend_used = "cpu_numba" if HAS_NUMBA else "cpu_numpy"

    png_bytes = rgb_to_png_bytes(rgb)
    elapsed = time.perf_counter() - t0
    return RenderOutput(
        png_bytes=png_bytes,
        rgb_array=rgb,
        compiled=compiled,
        render_seconds=elapsed,
        backend_used=backend_used,
        backend_notice=backend_notice,
        warnings=all_warnings,
    )
