"""Dual-backend GPU (ModernGL) & CPU (Numba/NumPy) rendering engine exports."""

from .backend_manager import RenderOutput, render_scene_dna, rgb_to_png_bytes
from .cpu_kernel import CPUGBuffer, HAS_NUMBA, render_cpu_gbuffer
from .gpu_moderngl import ModernGLRenderer, compute_camera_basis
from .optics_cpu import composite_cpu_optics

__all__ = [
    "CPUGBuffer",
    "HAS_NUMBA",
    "ModernGLRenderer",
    "RenderOutput",
    "composite_cpu_optics",
    "compute_camera_basis",
    "render_cpu_gbuffer",
    "render_scene_dna",
    "rgb_to_png_bytes",
]
