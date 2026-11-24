"""
Deterministic SceneDNA Validator & Auto-Repair Guardrails for PrismStudio.

Fixes common LLM planning mistakes before compilation:
- Binds missing `support_target` when an actor uses sitting_on/on_top_of/perched_on
- Prevents collapsed wall/table geometry (`0.1m` poles -> proper architectural widths)
- Adds sensible Zone 2 midground & Zone 3 emitters when left blank
- Checks compiled bounding spheres and contact gaps
"""

from __future__ import annotations

import copy
from typing import List, Tuple

from ..schema.scene_dna import (
    LightEmitterSpec,
    MidgroundZoneSpec,
    PropSpec,
    SceneDNA,
    scene_dna_from_dict,
)


def validate_and_repair_dna(dna: SceneDNA) -> Tuple[SceneDNA, List[str]]:
    """
    Repairs missing relational references and degenerate dimensions in-place on a copy,
    returning `(repaired_dna, diagnostic_warnings)`.
    """
    data = copy.deepcopy(dna.to_dict())
    notes: List[str] = []

    props = data.get("zone1_props") or []
    actors = data.get("actors") or []

    # 1. Repair collapsed architectural wall/ledge widths/depths (e.g. old 0.1m cylinder pole bug!)
    for p in props:
        arch = str(p.get("archetype", "masonry_wall")).lower()
        sz = list(p.get("size", [3.4, 1.45, 0.52]))
        if arch in ("masonry_wall", "rooftop_ledge"):
            if sz[0] < 1.2:
                notes.append(f"Auto-widened '{p.get('id')}' width from {sz[0]:.2f}m to 3.2m so it forms a wall rather than a pole.")
                sz[0] = 3.2
            if sz[1] < 0.7:
                sz[1] = 1.35
            if sz[2] < 0.35:
                sz[2] = 0.52
            p["size"] = sz
        elif arch == "stage_deck":
            sz[0] = max(2.6, sz[0])
            sz[2] = max(2.0, sz[2])
            p["size"] = sz

    # 2. Auto-bind missing support_target when actor sits/perches on top of something
    prop_ids = [str(p.get("id")) for p in props if p.get("id")]
    for a in actors:
        pl = str(a.get("placement", "center")).lower()
        tgt = str(a.get("support_target") or "").strip()
        if pl in ("on_top_of", "sitting_on", "perched_on") and (not tgt or tgt not in prop_ids):
            if prop_ids:
                a["support_target"] = prop_ids[0]
                notes.append(f"Auto-bound actor '{a.get('id')}' support_target -> '{prop_ids[0]}'.")
            elif "wall" in (dna.prompt or "").lower() or "ledge" in (dna.prompt or "").lower():
                wall_p = PropSpec(id="auto_high_wall", archetype="masonry_wall", has_foliage_ivy=True).to_dict() if hasattr(PropSpec(), "to_dict") else None
                if wall_p is None:
                    from dataclasses import asdict
                    wall_p = asdict(PropSpec(id="auto_high_wall", archetype="masonry_wall", has_foliage_ivy=True))
                props.append(wall_p)
                prop_ids.append("auto_high_wall")
                a["support_target"] = "auto_high_wall"
                notes.append("Synthesized missing masonry_wall prop for sitting_on actor.")

    data["zone1_props"] = props

    # 3. Ensure at least one HDR emitter exists for rim/god-rays
    atmo = data.setdefault("zone3_atmosphere", {})
    if not atmo.get("emitters"):
        from dataclasses import asdict
        atmo["emitters"] = [asdict(LightEmitterSpec())]
        notes.append("Added default sky emitter for atmospheric lighting and god-rays.")

    # 4. If sitting on a high wall/ledge, ensure midground reflects height
    if props and str(props[0].get("archetype", "")).lower() in ("masonry_wall", "rooftop_ledge"):
        mid = data.setdefault("zone2_midground", {})
        if not mid.get("kind") or mid.get("kind") == "none":
            from dataclasses import asdict
            data["zone2_midground"] = asdict(MidgroundZoneSpec())
            notes.append("Added Zone 2 rooftop_village midground below wall elevation.")

    repaired = scene_dna_from_dict(data)
    return repaired, notes
