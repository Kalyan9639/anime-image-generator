"""Deterministic Math Compiler exports for PrismStudio."""

from .bytecode import EYE_STYLE_MAP, SHADER_ID_MAP, SceneBuffers, pack_scene_buffers
from .cranial_sh import CompiledCranialShell, build_cranial_sh_coeffs, compile_cranial_shell
from .garment_uv import CompiledDecalRecord, CompiledGarmentRecord, compile_actor_garments
from .hair_fur import compile_hair_clumps
from .kinematics import (
    CompiledActorRig,
    CompiledSolidPrimitive,
    CompiledSplineBone,
    compile_actor_rig,
    solve_two_bone_ik,
)
from .layout_solver import BVHCluster, CompiledScene, solve_and_compile_scene
from .validator import validate_and_repair_dna
from .world_zones import CompiledMidgroundItem, CompiledProp, compile_zone1_prop, compile_zone2_midground

__all__ = [
    "BVHCluster",
    "CompiledActorRig",
    "CompiledCranialShell",
    "CompiledDecalRecord",
    "CompiledGarmentRecord",
    "CompiledMidgroundItem",
    "CompiledProp",
    "CompiledScene",
    "CompiledSolidPrimitive",
    "CompiledSplineBone",
    "EYE_STYLE_MAP",
    "SHADER_ID_MAP",
    "SceneBuffers",
    "build_cranial_sh_coeffs",
    "compile_actor_garments",
    "compile_actor_rig",
    "compile_cranial_shell",
    "compile_hair_clumps",
    "compile_zone1_prop",
    "compile_zone2_midground",
    "pack_scene_buffers",
    "solve_and_compile_scene",
    "solve_two_bone_ik",
    "validate_and_repair_dna",
]
