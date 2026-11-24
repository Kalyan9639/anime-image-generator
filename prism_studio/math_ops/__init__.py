"""Universal mathematical operators, 2D UV decals, and procedural materials."""

from .decals_2d import evaluate_2d_eye_uv, evaluate_2d_garment_decal
from .materials import evaluate_procedural_material, fbm_2d, hex_to_rgb_linear, value_noise_2d
from .primitives import (
    BishopSplineStations,
    compute_bishop_frames,
    domain_repeat_bounded,
    domain_repeat_masonry_staggered,
    evaluate_sh_basis_l3,
    rotation_matrix_xyz,
    sd_cloth_shell,
    sd_rounded_box,
    sd_sh_shell,
    sd_swept_superspline,
    sd_torus,
    smax_cubic,
    smin_cubic_mat,
)

__all__ = [
    "BishopSplineStations",
    "compute_bishop_frames",
    "domain_repeat_bounded",
    "domain_repeat_masonry_staggered",
    "evaluate_2d_eye_uv",
    "evaluate_2d_garment_decal",
    "evaluate_procedural_material",
    "evaluate_sh_basis_l3",
    "fbm_2d",
    "hex_to_rgb_linear",
    "rotation_matrix_xyz",
    "sd_cloth_shell",
    "sd_rounded_box",
    "sd_sh_shell",
    "sd_swept_superspline",
    "sd_torus",
    "smax_cubic",
    "smin_cubic_mat",
    "value_noise_2d",
]
