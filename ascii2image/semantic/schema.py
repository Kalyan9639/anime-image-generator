"""
Dynamic Multi-Layer Scene Graph Schema (Track 1).

Defines the pure-data specification shared between:
1. The LLM Knowledge/Scene Planner (gpt-oss:20b-cloud / local Ollama via JSON Schema),
2. The Semantic ASCII Layout Parser,
3. The Pre-Written Qualitative Constraint Solver, and
4. The Pre-Written Dynamic Math & Optical Rendering Engine.

Zero executable code or hardcoded object names exist anywhere in this schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


VALID_PRIMITIVES = (
    "rounded_box",
    "superquadric",
    "sphere",
    "ellipsoid",
    "cylinder",
    "cone",
    "capsule",
    "torus",
    "gielis_supershape",
    "lathe_profile",
    "bezier_tube",
    "convolution_limb",
    "lsystem_branch",
)

VALID_CSG_OPS = ("union", "smooth_union", "subtract", "intersect")

VALID_PATTERNS = (
    "solid",
    "grid_tiles",
    "plaid_tartan",
    "cable_knit",
    "brushed_metal",
    "wood_grain",
    "stone_voronoi",
    "circuit_traces",
    "stripes",
    "dots",
    "fbm_noise",
)

VALID_ATTACHMENTS = (
    "base_center",
    "on_top_of",
    "underneath",
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "inside_center",
    "around_perimeter",
    "attached_to",
    "draped_over",
    "hanging_from",
)

VALID_PLACEMENTS = (
    "center",
    "left",
    "right",
    "foreground_left",
    "foreground_right",
    "background_left",
    "background_right",
    "on_top_of",
    "sitting_on",
    "grounded_on",
    "next_to",
    "behind",
    "in_front_of",
    "leaning_against",
    "floating_above",
)

VALID_MODIFIER_KINDS = (
    "bend",
    "twist",
    "taper",
    "mirror_x",
    "mirror_z",
    "array_linear",
    "array_grid",
    "array_radial",
    "curve_chain",
    "surface_displace",
)

VALID_BG_STYLES = (
    "gradient",
    "bokeh_night",
    "twilight_glow",
    "studio",
    "starfield",
    "mountain_horizon",
)

VALID_ENV_KINDS = (
    "none",
    "ground_plane",
    "tiled_ledge",
    "tabletop",
    "terrain_fbm",
)


def _clamp_vec3(val: Any, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if isinstance(val, (list, tuple)) and len(val) >= 3:
        try:
            return (float(val[0]), float(val[1]), float(val[2]))
        except (TypeError, ValueError):
            pass
    return default


def _clamp_hex(val: Any, default: str = "#cbd5e1") -> str:
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("#") and len(s) in (4, 7):
            hex_part = s[1:]
            if all(c in "0123456789abcdefABCDEF" for c in hex_part):
                if len(hex_part) == 3:
                    return "#" + "".join(c * 2 for c in hex_part).lower()
                return "#" + hex_part.lower()
    return default


@dataclass
class CameraSpec:
    projection: str = "perspective"   # perspective | isometric | orthographic
    azimuth_deg: float = 28.0
    elevation_deg: float = 22.0
    roll_deg: float = 0.0
    fov_deg: float = 42.0
    distance_scale: float = 1.0
    dof_strength: float = 0.25        # [0.0, 1.0]
    focus_offset: float = 0.0         # shift relative to scene center


@dataclass
class LightingSpec:
    key_dir: Tuple[float, float, float] = (0.65, 0.85, 0.55)
    key_color: str = "#fff7ed"
    ambient_color: str = "#334155"
    rim_color: str = "#fde68a"
    rim_strength: float = 0.45
    bloom_intensity: float = 0.35
    cel_shading: bool = False
    exposure: float = 1.05


@dataclass
class BackgroundSpec:
    style: str = "gradient"           # gradient | bokeh_night | twilight_glow | studio | starfield | mountain_horizon
    top_color: str = "#0f172a"
    bottom_color: str = "#1e293b"
    glow_color: str = "#fbbf24"
    bokeh_count: int = 0
    bokeh_colors: List[str] = field(default_factory=lambda: ["#fde68a", "#fbbf24", "#fff7ed"])
    bokeh_size: float = 1.0
    particle_type: str = "none"       # none | snowflakes | sparks | stars
    particle_count: int = 0
    particle_color: str = "#ffffff"


@dataclass
class EnvironmentSpec:
    enabled: bool = True
    kind: str = "ground_plane"        # none | ground_plane | tiled_ledge | tabletop | terrain_fbm
    y_level: float = 0.0
    tilt_deg: float = 0.0
    pattern: str = "grid_tiles"
    primary_color: str = "#475569"
    secondary_color: str = "#334155"
    pattern_scale: float = 2.0
    roughness: float = 0.75
    metallic: float = 0.05


@dataclass
class MaterialSpec:
    color: str = "#cbd5e1"
    secondary_color: str = "#334155"
    accent_color: str = "#f59e0b"
    pattern: str = "solid"
    pattern_scale: float = 3.0
    metallic: float = 0.1
    roughness: float = 0.55
    emissive: float = 0.0             # [0.0, 3.0]


@dataclass
class ModifierSpec:
    kind: str                         # bend | twist | taper | mirror_x | mirror_z | array_linear | array_grid | array_radial | curve_chain | surface_displace
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PartNode:
    name: str
    primitive: str = "rounded_box"
    size: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    attachment: str = "base_center"
    target: str = ""                  # optional sibling/parent part name
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    csg_op: str = "union"
    smooth_k: float = 0.08
    params: Dict[str, Any] = field(default_factory=dict)
    modifiers: List[ModifierSpec] = field(default_factory=list)
    material: MaterialSpec = field(default_factory=MaterialSpec)
    children: List["PartNode"] = field(default_factory=list)


@dataclass
class EntitySpec:
    id: str
    label: str = ""
    role: str = "object"              # subject | object | prop | accent
    placement: str = "center"
    target: str = ""                  # optional entity id or "environment"
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0
    rotation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ascii_bbox: Optional[Tuple[float, float, float, float]] = None  # optional normalized (x0, y0, x1, y1)
    parts: List[PartNode] = field(default_factory=list)


@dataclass
class SceneGraph:
    title: str = "Procedural Math Scene"
    description: str = ""
    seed: int = 42
    camera: CameraSpec = field(default_factory=CameraSpec)
    lighting: LightingSpec = field(default_factory=LightingSpec)
    background: BackgroundSpec = field(default_factory=BackgroundSpec)
    environment: EnvironmentSpec = field(default_factory=EnvironmentSpec)
    entities: List[EntitySpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_material(raw: Any) -> MaterialSpec:
    if not isinstance(raw, dict):
        return MaterialSpec()
    pat = str(raw.get("pattern", "solid")).strip().lower()
    if pat not in VALID_PATTERNS:
        pat = "solid"
    return MaterialSpec(
        color=_clamp_hex(raw.get("color"), "#cbd5e1"),
        secondary_color=_clamp_hex(raw.get("secondary_color"), "#334155"),
        accent_color=_clamp_hex(raw.get("accent_color"), "#f59e0b"),
        pattern=pat,
        pattern_scale=max(0.1, min(40.0, float(raw.get("pattern_scale", 3.0) or 3.0))),
        metallic=max(0.0, min(1.0, float(raw.get("metallic", 0.1) or 0.0))),
        roughness=max(0.05, min(1.0, float(raw.get("roughness", 0.55) or 0.55))),
        emissive=max(0.0, min(4.0, float(raw.get("emissive", 0.0) or 0.0))),
    )


def _parse_modifier(raw: Any) -> Optional[ModifierSpec]:
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("kind", "")).strip().lower()
    if kind not in VALID_MODIFIER_KINDS:
        return None
    params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
    return ModifierSpec(kind=kind, params=dict(params))


def _parse_part(raw: Any, idx: int = 0) -> PartNode:
    if not isinstance(raw, dict):
        return PartNode(name=f"part_{idx}")
    prim = str(raw.get("primitive", "rounded_box")).strip().lower()
    if prim not in VALID_PRIMITIVES:
        prim = "rounded_box"
    attach = str(raw.get("attachment", "base_center")).strip().lower()
    if attach not in VALID_ATTACHMENTS:
        attach = "base_center"
    csg = str(raw.get("csg_op", "union")).strip().lower()
    if csg not in VALID_CSG_OPS:
        csg = "union"
    sx, sy, sz = _clamp_vec3(raw.get("size"), (1.0, 1.0, 1.0))
    size = (max(0.02, min(20.0, abs(sx))), max(0.02, min(20.0, abs(sy))), max(0.02, min(20.0, abs(sz))))
    mods = []
    for m in raw.get("modifiers", []) or []:
        parsed_m = _parse_modifier(m)
        if parsed_m:
            mods.append(parsed_m)
    children = [
        _parse_part(c, i) for i, c in enumerate(raw.get("children", []) or []) if isinstance(c, dict)
    ]
    return PartNode(
        name=str(raw.get("name") or f"part_{idx}").strip(),
        primitive=prim,
        size=size,
        attachment=attach,
        target=str(raw.get("target") or "").strip(),
        offset=_clamp_vec3(raw.get("offset"), (0.0, 0.0, 0.0)),
        rotation_deg=_clamp_vec3(raw.get("rotation_deg"), (0.0, 0.0, 0.0)),
        csg_op=csg,
        smooth_k=max(0.0, min(1.0, float(raw.get("smooth_k", 0.08) or 0.08))),
        params=dict(raw.get("params")) if isinstance(raw.get("params"), dict) else {},
        modifiers=mods,
        material=_parse_material(raw.get("material")),
        children=children,
    )


def scene_from_dict(data: Dict[str, Any]) -> SceneGraph:
    """Safely deserializes and clamps any raw JSON dict into a validated SceneGraph."""
    if not isinstance(data, dict):
        return SceneGraph()

    cam_raw = data.get("camera") if isinstance(data.get("camera"), dict) else {}
    proj = str(cam_raw.get("projection", "perspective")).strip().lower()
    if proj not in ("perspective", "isometric", "orthographic"):
        proj = "perspective"
    camera = CameraSpec(
        projection=proj,
        azimuth_deg=float(cam_raw.get("azimuth_deg", 28.0) or 28.0),
        elevation_deg=max(-15.0, min(85.0, float(cam_raw.get("elevation_deg", 22.0) or 22.0))),
        roll_deg=max(-25.0, min(25.0, float(cam_raw.get("roll_deg", 0.0) or 0.0))),
        fov_deg=max(18.0, min(75.0, float(cam_raw.get("fov_deg", 42.0) or 42.0))),
        distance_scale=max(0.5, min(2.5, float(cam_raw.get("distance_scale", 1.0) or 1.0))),
        dof_strength=max(0.0, min(1.0, float(cam_raw.get("dof_strength", 0.25) or 0.0))),
        focus_offset=float(cam_raw.get("focus_offset", 0.0) or 0.0),
    )

    lit_raw = data.get("lighting") if isinstance(data.get("lighting"), dict) else {}
    lighting = LightingSpec(
        key_dir=_clamp_vec3(lit_raw.get("key_dir"), (0.65, 0.85, 0.55)),
        key_color=_clamp_hex(lit_raw.get("key_color"), "#fff7ed"),
        ambient_color=_clamp_hex(lit_raw.get("ambient_color"), "#334155"),
        rim_color=_clamp_hex(lit_raw.get("rim_color"), "#fde68a"),
        rim_strength=max(0.0, min(1.5, float(lit_raw.get("rim_strength", 0.45) or 0.0))),
        bloom_intensity=max(0.0, min(1.5, float(lit_raw.get("bloom_intensity", 0.35) or 0.0))),
        cel_shading=bool(lit_raw.get("cel_shading", False)),
        exposure=max(0.5, min(2.0, float(lit_raw.get("exposure", 1.05) or 1.05))),
    )

    bg_raw = data.get("background") if isinstance(data.get("background"), dict) else {}
    bg_style = str(bg_raw.get("style", "gradient")).strip().lower()
    if bg_style not in VALID_BG_STYLES:
        bg_style = "gradient"
    p_type = str(bg_raw.get("particle_type", "none")).strip().lower()
    if p_type not in ("none", "snowflakes", "sparks", "stars"):
        p_type = "none"
    raw_bcols = bg_raw.get("bokeh_colors")
    if isinstance(raw_bcols, list) and raw_bcols:
        bcols = [_clamp_hex(c, "#fde68a") for c in raw_bcols[:6]]
    else:
        bcols = ["#fde68a", "#fbbf24", "#fff7ed"]
    background = BackgroundSpec(
        style=bg_style,
        top_color=_clamp_hex(bg_raw.get("top_color"), "#0f172a"),
        bottom_color=_clamp_hex(bg_raw.get("bottom_color"), "#1e293b"),
        glow_color=_clamp_hex(bg_raw.get("glow_color"), "#fbbf24"),
        bokeh_count=max(0, min(220, int(bg_raw.get("bokeh_count", 0) or 0))),
        bokeh_colors=bcols,
        bokeh_size=max(0.3, min(3.0, float(bg_raw.get("bokeh_size", 1.0) or 1.0))),
        particle_type=p_type,
        particle_count=max(0, min(120, int(bg_raw.get("particle_count", 0) or 0))),
        particle_color=_clamp_hex(bg_raw.get("particle_color"), "#ffffff"),
    )

    env_raw = data.get("environment") if isinstance(data.get("environment"), dict) else {}
    env_kind = str(env_raw.get("kind", "ground_plane")).strip().lower()
    if env_kind not in VALID_ENV_KINDS:
        env_kind = "ground_plane"
    env_pat = str(env_raw.get("pattern", "grid_tiles")).strip().lower()
    if env_pat not in VALID_PATTERNS:
        env_pat = "grid_tiles"
    environment = EnvironmentSpec(
        enabled=bool(env_raw.get("enabled", env_kind != "none")),
        kind=env_kind,
        y_level=float(env_raw.get("y_level", 0.0) or 0.0),
        tilt_deg=max(-25.0, min(25.0, float(env_raw.get("tilt_deg", 0.0) or 0.0))),
        pattern=env_pat,
        primary_color=_clamp_hex(env_raw.get("primary_color"), "#475569"),
        secondary_color=_clamp_hex(env_raw.get("secondary_color"), "#334155"),
        pattern_scale=max(0.2, min(20.0, float(env_raw.get("pattern_scale", 2.0) or 2.0))),
        roughness=max(0.05, min(1.0, float(env_raw.get("roughness", 0.75) or 0.75))),
        metallic=max(0.0, min(1.0, float(env_raw.get("metallic", 0.05) or 0.05))),
    )

    entities: List[EntitySpec] = []
    for e_idx, e_raw in enumerate(data.get("entities", []) or []):
        if not isinstance(e_raw, dict):
            continue
        place = str(e_raw.get("placement", "center")).strip().lower()
        if place not in VALID_PLACEMENTS:
            place = "center"
        role = str(e_raw.get("role", "object")).strip().lower()
        if role not in ("subject", "object", "prop", "accent"):
            role = "object"
        raw_bbox = e_raw.get("ascii_bbox")
        ascii_bbox = None
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
            try:
                ascii_bbox = (float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3]))
            except (TypeError, ValueError):
                ascii_bbox = None
        parts = [
            _parse_part(p, p_idx)
            for p_idx, p in enumerate(e_raw.get("parts", []) or [])
            if isinstance(p, dict)
        ]
        if not parts:
            parts = [PartNode(name="body", primitive="rounded_box")]
        entities.append(
            EntitySpec(
                id=str(e_raw.get("id") or f"entity_{e_idx}").strip(),
                label=str(e_raw.get("label") or e_raw.get("id") or f"entity_{e_idx}").strip(),
                role=role,
                placement=place,
                target=str(e_raw.get("target") or "").strip(),
                offset=_clamp_vec3(e_raw.get("offset"), (0.0, 0.0, 0.0)),
                scale=max(0.1, min(10.0, float(e_raw.get("scale", 1.0) or 1.0))),
                rotation_deg=_clamp_vec3(e_raw.get("rotation_deg"), (0.0, 0.0, 0.0)),
                ascii_bbox=ascii_bbox,
                parts=parts,
            )
        )

    return SceneGraph(
        title=str(data.get("title") or "Procedural Math Scene").strip(),
        description=str(data.get("description") or "").strip(),
        seed=int(data.get("seed", 42) or 42),
        camera=camera,
        lighting=lighting,
        background=background,
        environment=environment,
        entities=entities,
    )


def get_ollama_scene_json_schema() -> Dict[str, Any]:
    """
    Returns a clean, flat-friendly JSON Schema accepted by Ollama's `format`
    parameter (GBNF grammar constrained decoding) so gpt-oss:20b-cloud outputs
    strictly valid, multi-layer SceneGraph specifications without writing code.
    """
    vec3_schema = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
    }
    material_schema = {
        "type": "object",
        "properties": {
            "color": {"type": "string"},
            "secondary_color": {"type": "string"},
            "accent_color": {"type": "string"},
            "pattern": {"type": "string", "enum": list(VALID_PATTERNS)},
            "pattern_scale": {"type": "number"},
            "metallic": {"type": "number"},
            "roughness": {"type": "number"},
            "emissive": {"type": "number"},
        },
        "required": ["color", "pattern"],
    }
    modifier_schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(VALID_MODIFIER_KINDS)},
            "params": {"type": "object"},
        },
        "required": ["kind"],
    }
    part_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "primitive": {"type": "string", "enum": list(VALID_PRIMITIVES)},
            "size": vec3_schema,
            "attachment": {"type": "string", "enum": list(VALID_ATTACHMENTS)},
            "target": {"type": "string"},
            "offset": vec3_schema,
            "rotation_deg": vec3_schema,
            "csg_op": {"type": "string", "enum": list(VALID_CSG_OPS)},
            "smooth_k": {"type": "number"},
            "params": {"type": "object"},
            "modifiers": {"type": "array", "items": modifier_schema},
            "material": material_schema,
        },
        "required": ["name", "primitive", "size", "attachment", "material"],
    }
    entity_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "label": {"type": "string"},
            "role": {"type": "string", "enum": ["subject", "object", "prop", "accent"]},
            "placement": {"type": "string", "enum": list(VALID_PLACEMENTS)},
            "target": {"type": "string"},
            "offset": vec3_schema,
            "scale": {"type": "number"},
            "rotation_deg": vec3_schema,
            "parts": {"type": "array", "items": part_schema, "minItems": 1},
        },
        "required": ["id", "label", "role", "placement", "scale", "parts"],
    }
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "seed": {"type": "integer"},
            "camera": {
                "type": "object",
                "properties": {
                    "projection": {"type": "string", "enum": ["perspective", "isometric", "orthographic"]},
                    "azimuth_deg": {"type": "number"},
                    "elevation_deg": {"type": "number"},
                    "roll_deg": {"type": "number"},
                    "fov_deg": {"type": "number"},
                    "distance_scale": {"type": "number"},
                    "dof_strength": {"type": "number"},
                },
                "required": ["projection", "azimuth_deg", "elevation_deg"],
            },
            "lighting": {
                "type": "object",
                "properties": {
                    "key_dir": vec3_schema,
                    "key_color": {"type": "string"},
                    "ambient_color": {"type": "string"},
                    "rim_color": {"type": "string"},
                    "rim_strength": {"type": "number"},
                    "bloom_intensity": {"type": "number"},
                    "cel_shading": {"type": "boolean"},
                },
                "required": ["key_color", "ambient_color", "rim_color"],
            },
            "background": {
                "type": "object",
                "properties": {
                    "style": {"type": "string", "enum": list(VALID_BG_STYLES)},
                    "top_color": {"type": "string"},
                    "bottom_color": {"type": "string"},
                    "glow_color": {"type": "string"},
                    "bokeh_count": {"type": "integer"},
                    "bokeh_colors": {"type": "array", "items": {"type": "string"}},
                    "bokeh_size": {"type": "number"},
                    "particle_type": {"type": "string", "enum": ["none", "snowflakes", "sparks", "stars"]},
                    "particle_count": {"type": "integer"},
                    "particle_color": {"type": "string"},
                },
                "required": ["style", "top_color", "bottom_color"],
            },
            "environment": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "kind": {"type": "string", "enum": list(VALID_ENV_KINDS)},
                    "y_level": {"type": "number"},
                    "tilt_deg": {"type": "number"},
                    "pattern": {"type": "string", "enum": list(VALID_PATTERNS)},
                    "primary_color": {"type": "string"},
                    "secondary_color": {"type": "string"},
                    "pattern_scale": {"type": "number"},
                },
                "required": ["enabled", "kind", "pattern", "primary_color"],
            },
            "entities": {
                "type": "array",
                "items": entity_schema,
                "minItems": 1,
            },
        },
        "required": ["title", "camera", "lighting", "background", "environment", "entities"],
    }
