"""
Topological Layout Solver & Surface Contact Snapper for PrismStudio.

Resolves Zone 1 architectural props and actors in strict topological dependency
order (support targets ALWAYS compile before supported subjects, regardless of
JSON array order), snaps paws/feet/haunches flush onto support surface tops,
and auto-frames the camera so hero + context occupy 60-80% of the frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..schema.scene_dna import SceneDNA
from .garment_uv import CompiledGarmentRecord, compile_actor_garments
from .hair_fur import compile_hair_clumps
from .kinematics import CompiledActorRig, CompiledSolidPrimitive, CompiledSplineBone, compile_actor_rig
from .validator import validate_and_repair_dna
from .world_zones import CompiledMidgroundItem, CompiledProp, compile_zone1_prop, compile_zone2_midground


@dataclass
class BVHCluster:
    center: np.ndarray
    radius: float
    spline_start: int
    spline_count: int
    sh_start: int
    sh_count: int
    solid_start: int
    solid_count: int


@dataclass
class CompiledScene:
    dna: SceneDNA
    actors: List[CompiledActorRig]
    props: List[CompiledProp]
    midground: List[CompiledMidgroundItem]
    garments: List[CompiledGarmentRecord]
    bvh_clusters: List[BVHCluster]
    all_splines: List[CompiledSplineBone]
    all_solids: List[CompiledSolidPrimitive]
    focus_point: np.ndarray
    camera_target: np.ndarray
    scene_bmin: np.ndarray
    scene_bmax: np.ndarray
    diagnostics: List[str] = field(default_factory=list)


def _topological_sort_ids(nodes: List[Tuple[str, str]]) -> List[str]:
    """Kahn topological sort of (id, support_target_id) pairs."""
    ids = [nid for nid, _ in nodes]
    id_set = set(ids)
    deps: Dict[str, Optional[str]] = {nid: (tgt if tgt in id_set and tgt != nid else None) for nid, tgt in nodes}
    children: Dict[str, List[str]] = {nid: [] for nid in ids}
    indeg: Dict[str, int] = {nid: 0 for nid in ids}

    for nid, tgt in deps.items():
        if tgt is not None:
            children[tgt].append(nid)
            indeg[nid] += 1

    queue = [nid for nid in ids if indeg[nid] == 0]
    order: List[str] = []
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for ch in children[cur]:
            indeg[ch] -= 1
            if indeg[ch] == 0:
                queue.append(ch)

    # Append any cyclic leftovers safely in original order
    for nid in ids:
        if nid not in order:
            order.append(nid)
    return order


def solve_and_compile_scene(raw_dna: SceneDNA) -> CompiledScene:
    """
    Validates SceneDNA, resolves topological support dependencies, compiles all
    actors, garments, hair clumps, 3-zone props, and packs tight BVH clusters.
    """
    dna, diag = validate_and_repair_dna(raw_dna)
    ground_y = float(dna.zone3_atmosphere.ground_y) if dna.zone3_atmosphere.ground_enabled else 0.0

    # Stage placement XZ anchors
    stage_anchors = {
        "center": np.array([0.0, ground_y, 0.0], dtype=np.float32),
        "left": np.array([-1.15, ground_y, 0.1], dtype=np.float32),
        "right": np.array([1.15, ground_y, 0.1], dtype=np.float32),
        "foreground_left": np.array([-0.95, ground_y, 0.75], dtype=np.float32),
        "foreground_right": np.array([0.95, ground_y, 0.75], dtype=np.float32),
        "background_left": np.array([-1.35, ground_y, -1.1], dtype=np.float32),
        "background_right": np.array([1.35, ground_y, -1.1], dtype=np.float32),
    }

    # Build unified node map with prefixed IDs so prop/actor names never collide
    prop_map = {p.id: p for p in dna.zone1_props}
    actor_map = {a.id: a for a in dna.actors}

    dep_pairs: List[Tuple[str, str]] = []
    for p in dna.zone1_props:
        tgt = f"prop:{p.support_target}" if p.support_target in prop_map else ""
        dep_pairs.append((f"prop:{p.id}", tgt))
    for a in dna.actors:
        if a.support_target in prop_map:
            tgt = f"prop:{a.support_target}"
        elif a.support_target in actor_map:
            tgt = f"actor:{a.support_target}"
        else:
            tgt = ""
        dep_pairs.append((f"actor:{a.id}", tgt))

    order = _topological_sort_ids(dep_pairs)

    compiled_props: Dict[str, CompiledProp] = {}
    compiled_actors: Dict[str, CompiledActorRig] = {}
    garments_table: List[CompiledGarmentRecord] = []

    def _resolve_origin(placement: str, support_id: str, offset: Tuple[float, float, float]) -> np.ndarray:
        off = np.array(offset, dtype=np.float32)
        if support_id in compiled_props:
            sup = compiled_props[support_id]
            return np.array([sup.origin_world[0], sup.top_surface_y - 0.008, sup.origin_world[2]], dtype=np.float32) + off
        if support_id in compiled_actors:
            sup_a = compiled_actors[support_id]
            return np.array([sup_a.origin_world[0], sup_a.bvh_center[1] + sup_a.bvh_radius * 0.5, sup_a.origin_world[2]], dtype=np.float32) + off
        base = stage_anchors.get(placement, np.array([0.0, ground_y, 0.0], dtype=np.float32))
        return (base + off).astype(np.float32)

    for key in order:
        kind, nid = key.split(":", 1)
        if kind == "prop":
            p_spec = prop_map[nid]
            org = _resolve_origin(p_spec.placement, p_spec.support_target, p_spec.offset)
            # If sitting next_to / behind another prop
            if not p_spec.support_target and p_spec.placement in ("next_to", "behind", "in_front_of") and compiled_props:
                ref_p = next(iter(compiled_props.values()))
                if p_spec.placement == "next_to":
                    org = np.array([ref_p.bmax[0] + p_spec.size[0] * 0.5 + 0.35, ground_y, ref_p.origin_world[2]], dtype=np.float32)
                elif p_spec.placement == "in_front_of":
                    org = np.array([ref_p.origin_world[0], ground_y, ref_p.bmax[2] + p_spec.size[2] * 0.5 + 0.35], dtype=np.float32)
                elif p_spec.placement == "behind":
                    org = np.array([ref_p.origin_world[0], ground_y, ref_p.bmin[2] - p_spec.size[2] * 0.5 - 0.35], dtype=np.float32)
            compiled_props[nid] = compile_zone1_prop(p_spec, org, seed=dna.seed)
        else:
            a_spec = actor_map[nid]
            org = _resolve_origin(a_spec.placement, a_spec.support_target, a_spec.offset)
            # Compile at local origin=0 first to inspect lowest ground/paw contact y
            prelim = compile_actor_rig(a_spec, origin_world=np.zeros(3, dtype=np.float32), style_mode=dna.style_mode)
            lowest_y = 0.0
            if prelim.bones:
                # Exclude deliberately hanging tails on narrow edge perches when snapping paws/haunches!
                contact_ys = []
                for b in prelim.bones:
                    if b.name == "tail" and a_spec.posture == "sitting_edge":
                        continue
                    min_b_y = float(np.min(b.stations.points[:, 1] - np.maximum(b.stations.radii_a, b.stations.radii_b)))
                    contact_ys.append(min_b_y)
                for sp in prelim.solid_parts:
                    contact_ys.append(float(sp.center[1] - sp.half_extents[1]))
                if contact_ys:
                    lowest_y = min(contact_ys)
            org_snapped = np.array([org[0], org[1] - lowest_y, org[2]], dtype=np.float32)
            rig = compile_actor_rig(a_spec, origin_world=org_snapped, style_mode=dna.style_mode)
            rig.bones.extend(compile_hair_clumps(a_spec, rig.cranial_shell))
            compile_actor_garments(a_spec, rig, garments_table)

            # Recompute tight rig BVH including hair & garment shells
            pts_all = [rig.cranial_shell.center[None, :]] + [b.stations.points for b in rig.bones] + [sp.center[None, :] for sp in rig.solid_parts]
            merged = np.vstack(pts_all)
            bmin = np.min(merged, axis=0)
            bmax = np.max(merged, axis=0)
            rig.bvh_center = (0.5 * (bmin + bmax)).astype(np.float32)
            rig.bvh_radius = float(np.max(np.linalg.norm(merged - rig.bvh_center[None, :], axis=1))) + 0.30 * float(a_spec.scale)
            compiled_actors[nid] = rig

    props_list = [compiled_props[p.id] for p in dna.zone1_props if p.id in compiled_props]
    actors_list = [compiled_actors[a.id] for a in dna.actors if a.id in compiled_actors]

    # Compile Zone 2 midground silhouettes positioned below top Zone 1 support height
    zone1_top = max((p.top_surface_y for p in props_list), default=ground_y + 1.0)
    midground = compile_zone2_midground(dna.zone2_midground, zone1_top, seed=dna.seed)

    # Flatten primitives into contiguous BVH clusters
    all_splines: List[CompiledSplineBone] = []
    all_solids: List[CompiledSolidPrimitive] = []
    clusters: List[BVHCluster] = []

    for ac in actors_list:
        sp_s = len(all_splines)
        all_splines.extend(ac.bones)
        sp_c = len(ac.bones)
        so_s = len(all_solids)
        all_solids.extend(ac.solid_parts)
        so_c = len(ac.solid_parts)
        sh_idx = actors_list.index(ac)
        clusters.append(
            BVHCluster(
                center=ac.bvh_center,
                radius=ac.bvh_radius,
                spline_start=sp_s,
                spline_count=sp_c,
                sh_start=sh_idx,
                sh_count=1,
                solid_start=so_s,
                solid_count=so_c,
            )
        )

    for pr in props_list:
        sp_s = len(all_splines)
        all_splines.extend(pr.splines)
        sp_c = len(pr.splines)
        so_s = len(all_solids)
        all_solids.extend(pr.solids)
        so_c = len(pr.solids)
        clusters.append(
            BVHCluster(
                center=pr.bvh_center,
                radius=pr.bvh_radius,
                spline_start=sp_s,
                spline_count=sp_c,
                sh_start=0,
                sh_count=0,
                solid_start=so_s,
                solid_count=so_c,
            )
        )

    # Compute scene AABB & cinematic focus/target
    if clusters:
        c_pts = np.vstack([c.center - c.radius for c in clusters] + [c.center + c.radius for c in clusters])
        bmin = np.min(c_pts, axis=0)
        bmax = np.max(c_pts, axis=0)
    else:
        bmin = np.array([-1.0, 0.0, -1.0], dtype=np.float32)
        bmax = np.array([1.0, 1.5, 1.0], dtype=np.float32)

    if actors_list:
        hero = actors_list[0]
        focus_pt = hero.cranial_shell.center.copy()
        # Frame camera around hero torso/head + upper portion of support wall
        cam_tgt = 0.55 * hero.bvh_center + 0.45 * hero.cranial_shell.center
        if props_list and dna.camera.shot_type != "close_up":
            cam_tgt[1] -= 0.14
    else:
        focus_pt = 0.5 * (bmin + bmax)
        cam_tgt = focus_pt.copy()

    return CompiledScene(
        dna=dna,
        actors=actors_list,
        props=props_list,
        midground=midground,
        garments=garments_table,
        bvh_clusters=clusters,
        all_splines=all_splines,
        all_solids=all_solids,
        focus_point=focus_pt.astype(np.float32),
        camera_target=cam_tgt.astype(np.float32),
        scene_bmin=bmin.astype(np.float32),
        scene_bmax=bmax.astype(np.float32),
        diagnostics=diag,
    )
