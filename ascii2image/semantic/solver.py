"""
Qualitative Spatial Constraint Solver & Geometry Sanity Validator (Track 1).

Inspired by Holodeck (CVPR 2024) and ShapeAssembly (SIGGRAPH Asia 2020):
- Converts qualitative attachment/placement words (`on_top_of`, `underneath`,
  `sitting_on`, `next_to`, `behind`, `around_perimeter`, `draped_over`) into
  exact world-space transforms and tight 3D Axis-Aligned Bounding Boxes (AABBs).
- Prevents floating parts, unintended overlap, and off-camera framing.
- Produces human-readable geometry validation diagnostics for the LLM retry loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .math_core import (
    apply_domain_modifiers,
    evaluate_surface_displacement,
    generate_lsystem_segments,
    op_intersect,
    op_smooth_union,
    op_subtract,
    op_union,
    rotation_matrix_xyz,
    sd_bezier_tube,
    sd_cone_frustum,
    sd_convolution_limb,
    sd_cylinder,
    sd_ellipsoid,
    sd_gielis_supershape,
    sd_lathe_profile,
    sd_lsystem_segments,
    sd_rounded_box,
    sd_sphere,
    sd_superquadric,
    sd_torus,
)
from .schema import EntitySpec, MaterialSpec, PartNode, SceneGraph


@dataclass
class CompiledPart:
    name: str
    primitive: str
    center: np.ndarray              # (3,) float32 world-space center
    half_extents: np.ndarray        # (3,) float32 local primitive half extents
    rot_inv: np.ndarray             # (3, 3) float32 world->local rotation (R^T)
    rot_fwd: np.ndarray             # (3, 3) float32 local->world rotation (R)
    bmin: np.ndarray                # (3,) float32 world AABB min
    bmax: np.ndarray                # (3,) float32 world AABB max
    csg_op: str
    smooth_k: float
    params: Dict[str, Any]
    modifiers: List[Dict[str, Any]]
    material: MaterialSpec
    material_id: int
    lsystem_segments: Optional[List[Tuple[np.ndarray, np.ndarray, float, float]]] = None

    def eval_sdf(self, p_world: np.ndarray) -> np.ndarray:
        """Evaluates this compiled part's exact SDF at N world-space points."""
        p_local = (p_world - self.center[None, :]) @ self.rot_inv.T
        q, lip = apply_domain_modifiers(p_local, self.modifiers)

        hx, hy, hz = float(self.half_extents[0]), float(self.half_extents[1]), float(self.half_extents[2])
        prim = self.primitive
        pr = self.params

        if prim == "sphere":
            d = sd_sphere(q, radius=min(hx, hy, hz))
        elif prim == "ellipsoid":
            d = sd_ellipsoid(q, (hx, hy, hz))
        elif prim == "superquadric":
            d = sd_superquadric(
                q,
                (hx, hy, hz),
                e1=float(pr.get("e1", 0.4)),
                e2=float(pr.get("e2", 0.4)),
            )
        elif prim == "cylinder":
            d = sd_cylinder(q, radius=max(hx, hz), half_height=hy, bevel=float(pr.get("bevel", 0.025)))
        elif prim == "cone":
            r_bot = max(hx, hz) * float(pr.get("bottom_ratio", 1.0))
            r_top = max(hx, hz) * float(pr.get("top_ratio", 0.12))
            d = sd_cone_frustum(q, r_bottom=r_bot, r_top=r_top, half_height=hy)
        elif prim == "capsule":
            r = min(hx, hz)
            a = np.array([0.0, -max(0.01, hy - r), 0.0], dtype=np.float32)
            b = np.array([0.0, max(0.01, hy - r), 0.0], dtype=np.float32)
            d = sd_convolution_limb(q, [tuple(a.tolist()), tuple(b.tolist())], [r, r], smooth_k=0.02)
        elif prim == "torus":
            major_r = max(0.04, max(hx, hz) - min(hy, max(hx, hz) * 0.35))
            minor_r = max(0.015, min(hy, major_r * 0.45))
            d = sd_torus(q, major_r=major_r, minor_r=minor_r)
        elif prim == "gielis_supershape":
            d = sd_gielis_supershape(
                q,
                (hx, hy, hz),
                m=float(pr.get("m", 6.0)),
                n1=float(pr.get("n1", 1.0)),
                n2=float(pr.get("n2", 1.7)),
                n3=float(pr.get("n3", 1.7)),
                m2=float(pr["m2"]) if "m2" in pr else None,
            )
        elif prim == "lathe_profile":
            raw_prof = pr.get("profile_radii") or [0.75, 1.0, 0.95, 0.8]
            d = sd_lathe_profile(
                q,
                half_height=hy,
                max_radius=max(hx, hz),
                profile_radii=[float(v) for v in raw_prof],
                hollow=bool(pr.get("hollow", False)),
                wall_thickness=float(pr.get("wall_thickness", 0.08)),
            )
        elif prim == "bezier_tube":
            raw_pts = pr.get("control_points")
            if isinstance(raw_pts, list) and len(raw_pts) >= 3:
                pts = [(float(pt[0]) * hx, float(pt[1]) * hy, float(pt[2]) * hz) for pt in raw_pts[:4]]
            else:
                pts = [(-hx, -hy * 0.5, 0.0), (0.0, hy, 0.0), (hx, -hy * 0.5, 0.0)]
            r_s = max(0.02, float(pr.get("radius_start", min(hx, hy, hz) * 0.22)))
            r_e = max(0.02, float(pr.get("radius_end", r_s)))
            d = sd_bezier_tube(q, pts, r_start=r_s, r_end=r_e, segments=8)
        elif prim == "convolution_limb":
            raw_pts = pr.get("control_points")
            if isinstance(raw_pts, list) and len(raw_pts) >= 2:
                pts = [(float(pt[0]) * hx, float(pt[1]) * hy, float(pt[2]) * hz) for pt in raw_pts]
            else:
                pts = [(0.0, -hy, 0.0), (0.0, 0.0, hz * 0.2), (0.0, hy, 0.0)]
            raw_rads = pr.get("radii")
            if isinstance(raw_rads, list) and raw_rads:
                rads = [max(0.015, float(rv) * min(hx, hz)) for rv in raw_rads]
            else:
                rads = [max(0.02, min(hx, hz) * 0.65)] * len(pts)
            d = sd_convolution_limb(q, pts, rads, smooth_k=max(0.02, self.smooth_k))
        elif prim == "lsystem_branch":
            d = sd_lsystem_segments(q, self.lsystem_segments or [], smooth_k=0.025)
        else:  # rounded_box
            bevel = float(pr.get("bevel", min(hx, hy, hz) * 0.14))
            d = sd_rounded_box(q, (hx, hy, hz), radius=bevel)

        if self.modifiers:
            disp = evaluate_surface_displacement(p_local, self.modifiers)
            d = d + disp

        return (d * lip).astype(np.float32)


@dataclass
class CompiledEntity:
    id: str
    label: str
    role: str
    placement: str
    parts: List[CompiledPart]
    bmin: np.ndarray                # (3,) tight world AABB min
    bmax: np.ndarray                # (3,) tight world AABB max

    def eval_sdf_and_mat(self, p_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates combined CSG SDF and winning material_id for all parts in this entity."""
        n = p_world.shape[0]
        if not self.parts:
            return np.full(n, 1e4, dtype=np.float32), np.zeros(n, dtype=np.int32)

        d_acc = self.parts[0].eval_sdf(p_world)
        mat_acc = np.full(n, self.parts[0].material_id, dtype=np.int32)

        for part in self.parts[1:]:
            d_p = part.eval_sdf(p_world)
            op = part.csg_op
            if op == "subtract":
                d_new = op_subtract(d_acc, d_p)
                # Cut interior can adopt cutter material if desired
                mat_acc = np.where(d_new > d_acc, part.material_id, mat_acc)
                d_acc = d_new
            elif op == "intersect":
                mat_acc = np.where(d_p > d_acc, part.material_id, mat_acc)
                d_acc = op_intersect(d_acc, d_p)
            elif op == "smooth_union":
                mat_acc = np.where(d_p < d_acc, part.material_id, mat_acc)
                d_acc = op_smooth_union(d_acc, d_p, part.smooth_k)
            else:  # union
                mat_acc = np.where(d_p < d_acc, part.material_id, mat_acc)
                d_acc = op_union(d_acc, d_p)

        return d_acc, mat_acc


@dataclass
class ResolvedScene:
    scene: SceneGraph
    entities: List[CompiledEntity]
    materials: List[MaterialSpec]
    scene_bmin: np.ndarray
    scene_bmax: np.ndarray
    diagnostics: List[str] = field(default_factory=list)


def _compute_modified_half_extents(
    base_half: np.ndarray,
    modifiers: List[Dict[str, Any]],
) -> np.ndarray:
    """Expands local AABB half-extents to account for mirror/array/radial/warp modifiers."""
    hx, hy, hz = float(base_half[0]), float(base_half[1]), float(base_half[2])
    for mod in modifiers:
        kind = mod.get("kind", "")
        pr = mod.get("params", {}) or {}
        if kind == "mirror_x":
            hx = hx + abs(float(pr.get("offset", 0.0)))
        elif kind == "mirror_z":
            hz = hz + abs(float(pr.get("offset", 0.0)))
        elif kind == "array_linear":
            ax = str(pr.get("axis", "x")).lower()
            cnt = max(1, min(32, int(pr.get("count", 4))))
            sp = max(0.02, float(pr.get("spacing", 0.3)))
            span = (cnt - 1) * 0.5 * sp
            if ax == "x":
                hx += span
            elif ax == "y":
                hy += span
            else:
                hz += span
        elif kind == "array_grid":
            nx = max(1, min(24, int(pr.get("nx", 4))))
            nz = max(1, min(24, int(pr.get("nz", 4))))
            sx = max(0.02, float(pr.get("spacing_x", 0.2)))
            sz = max(0.02, float(pr.get("spacing_z", 0.2)))
            hx += (nx - 1) * 0.5 * sx
            hz += (nz - 1) * 0.5 * sz
        elif kind == "array_radial":
            ring_r = abs(float(pr.get("radius", 0.0)))
            r_max = max(hx, hz) + ring_r
            hx = r_max
            hz = r_max
        elif kind in ("bend", "twist", "taper"):
            hx *= 1.35
            hy *= 1.25
            hz *= 1.35
    return np.array([hx, hy, hz], dtype=np.float32)


def _rotated_aabb(center: np.ndarray, half_ext: np.ndarray, rot_fwd: np.ndarray, pad: float = 0.06) -> Tuple[np.ndarray, np.ndarray]:
    """Computes tight world AABB of an oriented box with a small safety margin."""
    abs_r = np.abs(rot_fwd)
    world_half = abs_r @ half_ext + pad
    return (center - world_half).astype(np.float32), (center + world_half).astype(np.float32)


def _flatten_parts(parts: List[PartNode], parent_name: str = "") -> List[PartNode]:
    out: List[PartNode] = []
    for p in parts:
        p_copy = PartNode(
            name=p.name,
            primitive=p.primitive,
            size=p.size,
            attachment=p.attachment,
            target=p.target or parent_name,
            offset=p.offset,
            rotation_deg=p.rotation_deg,
            csg_op=p.csg_op,
            smooth_k=p.smooth_k,
            params=dict(p.params),
            modifiers=list(p.modifiers),
            material=p.material,
            children=[],
        )
        out.append(p_copy)
        if p.children:
            out.extend(_flatten_parts(p.children, parent_name=p.name))
    return out


def _resolve_entity_parts_local(
    entity: EntitySpec,
    materials: List[MaterialSpec],
    seed: int,
) -> Tuple[List[CompiledPart], np.ndarray, np.ndarray]:
    """Resolves all parts inside one entity using qualitative attachment constraints."""
    flat_parts = _flatten_parts(entity.parts)
    resolved: List[CompiledPart] = []
    by_name: Dict[str, CompiledPart] = {}

    for idx, node in enumerate(flat_parts):
        mod_dicts = [{"kind": m.kind, "params": dict(m.params)} for m in node.modifiers]
        base_half = np.array([node.size[0] * 0.5, node.size[1] * 0.5, node.size[2] * 0.5], dtype=np.float32)
        eff_half = _compute_modified_half_extents(base_half, mod_dicts)
        if node.primitive == "lathe_profile" and isinstance(node.params.get("profile_radii"), list) and node.params["profile_radii"]:
            try:
                max_p = max(0.25, min(1.0, max(float(v) for v in node.params["profile_radii"])))
                eff_half[0] *= max_p
                eff_half[2] *= max_p
            except (TypeError, ValueError):
                pass

        lsys_segs = None
        if node.primitive == "lsystem_branch":
            lsys_segs = generate_lsystem_segments(
                height=float(node.size[1]),
                base_radius=max(0.03, min(float(node.size[0]), float(node.size[2])) * 0.14),
                depth=int(node.params.get("depth", 3)),
                angle_deg=float(node.params.get("angle_deg", 32.0)),
                seed=seed + idx * 17,
            )
            if lsys_segs:
                all_pts = np.vstack([s[0] for s in lsys_segs] + [s[1] for s in lsys_segs])
                max_r = max(max(s[2], s[3]) for s in lsys_segs)
                eff_half = np.maximum(eff_half, np.max(np.abs(all_pts), axis=0) + max_r)

        rot_fwd = rotation_matrix_xyz(*node.rotation_deg)
        rot_inv = rot_fwd.T
        rot_half = np.abs(rot_fwd) @ eff_half

        # Resolve target part for qualitative attachment
        ref: Optional[CompiledPart] = None
        if node.target and node.target in by_name:
            ref = by_name[node.target]
        elif resolved:
            ref = resolved[0] if node.attachment in ("on_top_of", "underneath", "around_perimeter") and not node.target else resolved[-1]

        att = node.attachment
        off = np.array(node.offset, dtype=np.float32)

        if ref is None or att == "base_center":
            center = np.array([0.0, float(rot_half[1]), 0.0], dtype=np.float32) + off
        else:
            ref_center = 0.5 * (ref.bmin + ref.bmax)
            # Strip AABB safety pad (~0.06) when snapping faces flush
            r_min = ref.bmin + 0.06
            r_max = ref.bmax - 0.06
            nest_y = 0.06 if (ref.primitive == "cone" and att == "on_top_of") else 0.025
            if att == "on_top_of":
                center = np.array([ref_center[0], r_max[1] + rot_half[1] - nest_y, ref_center[2]], dtype=np.float32) + off
            elif att == "underneath":
                center = np.array([ref_center[0], r_min[1] - rot_half[1] + 0.025, ref_center[2]], dtype=np.float32) + off
            elif att == "left_of":
                center = np.array([r_min[0] - rot_half[0] + 0.03, ref_center[1], ref_center[2]], dtype=np.float32) + off
            elif att == "right_of":
                center = np.array([r_max[0] + rot_half[0] - 0.03, ref_center[1], ref_center[2]], dtype=np.float32) + off
            elif att == "in_front_of":
                center = np.array([ref_center[0], ref_center[1], r_max[2] + rot_half[2] - 0.025], dtype=np.float32) + off
            elif att == "behind":
                center = np.array([ref_center[0], ref_center[1], r_min[2] - rot_half[2] + 0.025], dtype=np.float32) + off
            elif att in ("inside_center", "around_perimeter"):
                center = ref_center + off
            elif att == "draped_over":
                center = np.array([ref_center[0], r_max[1] + rot_half[1] * 0.25, ref_center[2]], dtype=np.float32) + off
            elif att == "hanging_from":
                center = np.array([ref_center[0], r_min[1] - rot_half[1] * 0.85, ref_center[2]], dtype=np.float32) + off
            else:  # attached_to
                center = ref_center + off

        bmin, bmax = _rotated_aabb(center, eff_half, rot_fwd, pad=0.06)
        mat_id = len(materials)
        materials.append(node.material)

        cp = CompiledPart(
            name=node.name,
            primitive=node.primitive,
            center=center,
            half_extents=base_half,
            rot_inv=rot_inv,
            rot_fwd=rot_fwd,
            bmin=bmin,
            bmax=bmax,
            csg_op=node.csg_op,
            smooth_k=node.smooth_k,
            params=dict(node.params),
            modifiers=mod_dicts,
            material=node.material,
            material_id=mat_id,
            lsystem_segments=lsys_segs,
        )
        resolved.append(cp)
        by_name[node.name] = cp

    if not resolved:
        z = np.zeros(3, dtype=np.float32)
        return [], z, z

    # Normalize local entity so its bottom rests at y=0 and XZ is centered around 0
    ent_min = np.min(np.vstack([p.bmin for p in resolved]), axis=0)
    ent_max = np.max(np.vstack([p.bmax for p in resolved]), axis=0)
    shift = np.array([-(ent_min[0] + ent_max[0]) * 0.5, -ent_min[1], -(ent_min[2] + ent_max[2]) * 0.5], dtype=np.float32)

    s = float(entity.scale)
    e_rot = rotation_matrix_xyz(*entity.rotation_deg)

    scaled_parts: List[CompiledPart] = []
    for p in resolved:
        c_norm = (p.center + shift) * s
        c_rot = e_rot @ c_norm
        new_rot_fwd = (e_rot @ p.rot_fwd).astype(np.float32)
        new_half = (p.half_extents * s).astype(np.float32)

        # Scale modifier spacings/offsets proportionally to entity scale
        scaled_mods: List[Dict[str, Any]] = []
        for m in p.modifiers:
            m_params = dict(m["params"])
            for k in ("offset", "spacing", "spacing_x", "spacing_z", "radius", "amplitude"):
                if k in m_params:
                    m_params[k] = float(m_params[k]) * s
            scaled_mods.append({"kind": m["kind"], "params": m_params})

        scaled_lsys = None
        if p.lsystem_segments is not None:
            scaled_lsys = [(a * s, b * s, r0 * s, r1 * s) for a, b, r0, r1 in p.lsystem_segments]

        eff_half = _compute_modified_half_extents(new_half, scaled_mods)
        if scaled_lsys:
            all_pts = np.vstack([seg[0] for seg in scaled_lsys] + [seg[1] for seg in scaled_lsys])
            max_r = max(max(seg[2], seg[3]) for seg in scaled_lsys)
            eff_half = np.maximum(eff_half, np.max(np.abs(all_pts), axis=0) + max_r + 0.04)

        bmin, bmax = _rotated_aabb(c_rot, eff_half, new_rot_fwd, pad=0.05)
        scaled_parts.append(
            CompiledPart(
                name=p.name,
                primitive=p.primitive,
                center=c_rot.astype(np.float32),
                half_extents=new_half,
                rot_inv=new_rot_fwd.T.astype(np.float32),
                rot_fwd=new_rot_fwd,
                bmin=bmin,
                bmax=bmax,
                csg_op=p.csg_op,
                smooth_k=p.smooth_k * s,
                params=p.params,
                modifiers=scaled_mods,
                material=p.material,
                material_id=p.material_id,
                lsystem_segments=scaled_lsys,
            )
        )

    final_min = np.min(np.vstack([p.bmin for p in scaled_parts]), axis=0)
    final_max = np.max(np.vstack([p.bmax for p in scaled_parts]), axis=0)
    # Re-zero y min after rotation so entity sits flat at y=0
    y_fix = -float(final_min[1])
    for p in scaled_parts:
        p.center[1] += y_fix
        p.bmin[1] += y_fix
        p.bmax[1] += y_fix
    final_min[1] += y_fix
    final_max[1] += y_fix
    return scaled_parts, final_min, final_max


def _translate_entity(ent: CompiledEntity, delta: np.ndarray) -> None:
    for p in ent.parts:
        p.center = (p.center + delta).astype(np.float32)
        p.bmin = (p.bmin + delta).astype(np.float32)
        p.bmax = (p.bmax + delta).astype(np.float32)
    ent.bmin = (ent.bmin + delta).astype(np.float32)
    ent.bmax = (ent.bmax + delta).astype(np.float32)


def solve_scene(scene: SceneGraph) -> ResolvedScene:
    """
    Compiles every entity in the SceneGraph, resolves multi-object qualitative
    placements, applies non-overlap separation, and produces validation diagnostics.
    """
    materials: List[MaterialSpec] = [MaterialSpec()]  # id 0 = default fallback
    compiled_entities: List[CompiledEntity] = []
    by_id: Dict[str, CompiledEntity] = {}
    env_y = float(scene.environment.y_level) if scene.environment.enabled else 0.0

    # Preset stage anchors for qualitative scene placements
    stage_anchors = {
        "center": np.array([0.0, env_y, 0.0], dtype=np.float32),
        "grounded_on": np.array([0.0, env_y, 0.0], dtype=np.float32),
        "sitting_on": np.array([0.0, env_y, 0.0], dtype=np.float32),
        "left": np.array([-1.8, env_y, 0.1], dtype=np.float32),
        "right": np.array([1.8, env_y, 0.1], dtype=np.float32),
        "foreground_left": np.array([-1.5, env_y, 1.3], dtype=np.float32),
        "foreground_right": np.array([1.5, env_y, 1.3], dtype=np.float32),
        "background_left": np.array([-2.1, env_y, -1.6], dtype=np.float32),
        "background_right": np.array([2.1, env_y, -1.6], dtype=np.float32),
        "floating_above": np.array([0.0, env_y + 1.8, 0.0], dtype=np.float32),
    }

    for idx, e_spec in enumerate(scene.entities):
        parts, bmin, bmax = _resolve_entity_parts_local(e_spec, materials, seed=scene.seed + idx * 97)
        ent = CompiledEntity(
            id=e_spec.id,
            label=e_spec.label,
            role=e_spec.role,
            placement=e_spec.placement,
            parts=parts,
            bmin=bmin.copy(),
            bmax=bmax.copy(),
        )
        size = ent.bmax - ent.bmin
        off = np.array(e_spec.offset, dtype=np.float32)

        # Priority 1: Explicit Semantic ASCII layout bbox overrides XZ position
        if e_spec.ascii_bbox is not None:
            x0, y0, x1, y1 = e_spec.ascii_bbox
            cx = ((x0 + x1) * 0.5 - 0.5) * 6.0
            cz = ((y0 + y1) * 0.5 - 0.5) * 4.5
            delta = np.array([cx, env_y, cz], dtype=np.float32) + off
            _translate_entity(ent, delta)
        else:
            ref = by_id.get(e_spec.target) if e_spec.target else (compiled_entities[0] if compiled_entities else None)
            pl = e_spec.placement
            if pl == "on_top_of" and ref is not None:
                rc = 0.5 * (ref.bmin + ref.bmax)
                delta = np.array([rc[0], ref.bmax[1] - 0.03, rc[2]], dtype=np.float32) + off
            elif pl == "next_to" and ref is not None:
                rc = 0.5 * (ref.bmin + ref.bmax)
                side = 1.0 if idx % 2 == 1 else -1.0
                dx = side * ((ref.bmax[0] - ref.bmin[0]) * 0.5 + size[0] * 0.5 + 0.25)
                delta = np.array([rc[0] + dx, env_y, rc[2]], dtype=np.float32) + off
            elif pl == "behind" and ref is not None:
                rc = 0.5 * (ref.bmin + ref.bmax)
                dz = -((ref.bmax[2] - ref.bmin[2]) * 0.5 + size[2] * 0.5 + 0.45)
                delta = np.array([rc[0], env_y, rc[2] + dz], dtype=np.float32) + off
            elif pl == "in_front_of" and ref is not None:
                rc = 0.5 * (ref.bmin + ref.bmax)
                dz = (ref.bmax[2] - ref.bmin[2]) * 0.5 + size[2] * 0.5 + 0.35
                delta = np.array([rc[0], env_y, rc[2] + dz], dtype=np.float32) + off
            elif pl == "leaning_against" and ref is not None:
                rc = 0.5 * (ref.bmin + ref.bmax)
                dx = (ref.bmax[0] - ref.bmin[0]) * 0.5 + size[0] * 0.42
                delta = np.array([rc[0] + dx, env_y, rc[2]], dtype=np.float32) + off
            elif pl == "floating_above" and ref is not None:
                rc = 0.5 * (ref.bmin + ref.bmax)
                delta = np.array([rc[0], ref.bmax[1] + 0.65, rc[2]], dtype=np.float32) + off
            else:
                anchor = stage_anchors.get(pl, np.array([0.0, env_y, 0.0], dtype=np.float32))
                delta = anchor + off
            _translate_entity(ent, delta)

        compiled_entities.append(ent)
        by_id[ent.id] = ent

    # Non-overlap relaxation for unrelated sibling entities sharing the same ground plane
    for _ in range(3):
        for i in range(len(compiled_entities)):
            for j in range(i + 1, len(compiled_entities)):
                ei, ej = compiled_entities[i], compiled_entities[j]
                if ei.placement in ("on_top_of", "floating_above") or ej.placement in ("on_top_of", "floating_above"):
                    continue
                # Check 3D AABB intersection (with tighter interior margin)
                overlap_x = min(ei.bmax[0], ej.bmax[0]) - max(ei.bmin[0], ej.bmin[0]) - 0.10
                overlap_y = min(ei.bmax[1], ej.bmax[1]) - max(ei.bmin[1], ej.bmin[1]) - 0.10
                overlap_z = min(ei.bmax[2], ej.bmax[2]) - max(ei.bmin[2], ej.bmin[2]) - 0.10
                if overlap_x > 0.05 and overlap_y > 0.05 and overlap_z > 0.05:
                    ci = 0.5 * (ei.bmin + ei.bmax)
                    cj = 0.5 * (ej.bmin + ej.bmax)
                    if overlap_x <= overlap_z:
                        sign = 1.0 if cj[0] >= ci[0] else -1.0
                        push = np.array([sign * (overlap_x * 0.55 + 0.08), 0.0, 0.0], dtype=np.float32)
                    else:
                        sign = 1.0 if cj[2] >= ci[2] else -1.0
                        push = np.array([0.0, 0.0, sign * (overlap_z * 0.55 + 0.08)], dtype=np.float32)
                    _translate_entity(ej, push)

    if compiled_entities:
        scene_bmin = np.min(np.vstack([e.bmin for e in compiled_entities]), axis=0)
        scene_bmax = np.max(np.vstack([e.bmax for e in compiled_entities]), axis=0)
    else:
        scene_bmin = np.array([-1.0, 0.0, -1.0], dtype=np.float32)
        scene_bmax = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    diagnostics = validate_resolved_scene(scene, compiled_entities, env_y)
    return ResolvedScene(
        scene=scene,
        entities=compiled_entities,
        materials=materials,
        scene_bmin=scene_bmin.astype(np.float32),
        scene_bmax=scene_bmax.astype(np.float32),
        diagnostics=diagnostics,
    )


def validate_resolved_scene(
    scene: SceneGraph,
    entities: List[CompiledEntity],
    env_y: float,
) -> List[str]:
    """
    Runs deterministic geometric sanity checks and returns plain-English error
    messages suitable for feeding back into the LLM retry loop.
    """
    issues: List[str] = []
    if not entities:
        issues.append("Scene has zero entities; include at least one subject or object.")
        return issues

    for ent in entities:
        ext = ent.bmax - ent.bmin
        if float(np.min(ext)) < 0.02:
            issues.append(f"Entity '{ent.id}' has collapsed near-zero thickness; increase part sizes.")
        if scene.environment.enabled and ent.placement not in ("on_top_of", "floating_above"):
            gap = float(ent.bmin[1]) - env_y
            if gap > 0.35:
                issues.append(f"Entity '{ent.id}' is floating {gap:.2f} units above the ground; set placement='grounded_on' or 'sitting_on'.")

    return issues
