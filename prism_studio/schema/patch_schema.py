"""
VLM Critique ScenePatch Schema & Bounded Delta Applicator.

Allows a Vision-Language Model inspecting a fast 320x240 draft render to emit
compact, bounded corrections (camera angle, framing, support elevation, rim/key
lighting, or actor posture/tail tweaks) without rewriting the entire SceneDNA.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from .scene_dna import SceneDNA, _clamp, _clamp_hex, scene_dna_from_dict


@dataclass
class ScenePatch:
    summary: str = ""
    azimuth_delta_deg: float = 0.0
    elevation_delta_deg: float = 0.0
    distance_scale_mul: float = 1.0
    rim_strength_delta: float = 0.0
    exposure_delta: float = 0.0
    bloom_delta: float = 0.0
    actor_y_offset_delta: float = 0.0
    actor_facing_delta_deg: float = 0.0
    tail_curl_delta: float = 0.0
    prop_height_mul: float = 1.0
    rim_color_override: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def patch_from_dict(raw: Dict[str, Any]) -> ScenePatch:
    if not isinstance(raw, dict):
        return ScenePatch()
    rim_ov = str(raw.get("rim_color_override") or "").strip()
    return ScenePatch(
        summary=str(raw.get("summary") or "").strip(),
        azimuth_delta_deg=_clamp(raw.get("azimuth_delta_deg"), -45.0, 45.0, 0.0),
        elevation_delta_deg=_clamp(raw.get("elevation_delta_deg"), -25.0, 25.0, 0.0),
        distance_scale_mul=_clamp(raw.get("distance_scale_mul"), 0.72, 1.45, 1.0),
        rim_strength_delta=_clamp(raw.get("rim_strength_delta"), -0.5, 0.7, 0.0),
        exposure_delta=_clamp(raw.get("exposure_delta"), -0.35, 0.35, 0.0),
        bloom_delta=_clamp(raw.get("bloom_delta"), -0.4, 0.4, 0.0),
        actor_y_offset_delta=_clamp(raw.get("actor_y_offset_delta"), -0.35, 0.35, 0.0),
        actor_facing_delta_deg=_clamp(raw.get("actor_facing_delta_deg"), -45.0, 45.0, 0.0),
        tail_curl_delta=_clamp(raw.get("tail_curl_delta"), -0.5, 0.5, 0.0),
        prop_height_mul=_clamp(raw.get("prop_height_mul"), 0.7, 1.6, 1.0),
        rim_color_override=_clamp_hex(rim_ov, "") if rim_ov.startswith("#") else "",
        warnings=[str(w) for w in (raw.get("warnings") or [])],
    )


def apply_scene_patch(dna: SceneDNA, patch: ScenePatch) -> SceneDNA:
    """Applies bounded VLM delta corrections and re-validates SceneDNA invariants."""
    data = copy.deepcopy(dna.to_dict())
    cam = data.setdefault("camera", {})
    cam["azimuth_deg"] = float(cam.get("azimuth_deg", 22.0)) + patch.azimuth_delta_deg
    cam["elevation_deg"] = float(cam.get("elevation_deg", 10.0)) + patch.elevation_delta_deg
    cam["distance_scale"] = float(cam.get("distance_scale", 1.0)) * patch.distance_scale_mul

    lit = data.setdefault("lighting", {})
    lit["rim_strength"] = float(lit.get("rim_strength", 0.85)) + patch.rim_strength_delta
    lit["exposure"] = float(lit.get("exposure", 1.08)) + patch.exposure_delta
    lit["bloom_intensity"] = float(lit.get("bloom_intensity", 0.55)) + patch.bloom_delta
    if patch.rim_color_override:
        lit["rim_color"] = patch.rim_color_override

    for actor in data.get("actors", []):
        off = list(actor.get("offset", [0.0, 0.0, 0.0]))
        off[1] = float(off[1]) + patch.actor_y_offset_delta
        actor["offset"] = off
        actor["facing_deg"] = float(actor.get("facing_deg", 12.0)) + patch.actor_facing_delta_deg
        props = actor.setdefault("proportions", {})
        props["tail_curl"] = float(props.get("tail_curl", 0.45)) + patch.tail_curl_delta

    if data.get("zone1_props"):
        sz = list(data["zone1_props"][0].get("size", [3.4, 1.45, 0.52]))
        sz[1] = float(sz[1]) * patch.prop_height_mul
        data["zone1_props"][0]["size"] = sz

    return scene_dna_from_dict(data)


def get_patch_json_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "azimuth_delta_deg": {"type": "number"},
            "elevation_delta_deg": {"type": "number"},
            "distance_scale_mul": {"type": "number"},
            "rim_strength_delta": {"type": "number"},
            "exposure_delta": {"type": "number"},
            "bloom_delta": {"type": "number"},
            "actor_y_offset_delta": {"type": "number"},
            "actor_facing_delta_deg": {"type": "number"},
            "tail_curl_delta": {"type": "number"},
            "prop_height_mul": {"type": "number"},
            "rim_color_override": {"type": "string"},
        },
        "required": ["summary"],
    }
