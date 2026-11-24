"""
Semantic SceneDNA Specification for PrismStudio.

Pure-data compositional grammar shared between the LLM Scene Director,
Deterministic Math Compiler, Topological Layout Solver, and Dual-Backend
Render Engine. Zero legacy dependencies or hardcoded asset templates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


VALID_STYLE_MODES = ("anime_npr", "stylized_pbr")
VALID_ARCHETYPES = ("humanoid", "quadruped", "avian", "serpentine", "custom_dag")
VALID_POSTURES = (
    "standing",
    "sitting_haunches",
    "sitting_edge",
    "walking",
    "crouching",
    "singing_pose",
    "kneeling",
    "skating_lean",
    "hovering",
)
VALID_EYE_STYLES = ("anime_round", "feline_slit", "sharp_mystic", "visor_glow", "closed_happy")
VALID_HAIR_STYLES = (
    "none",
    "high_ponytail",
    "twin_tails",
    "long_flowing",
    "bob_cut",
    "short_spiky",
    "braided_beard",
    "lion_mane",
    "feline_cheek_tufts",
)
VALID_GARMENT_REGIONS = (
    "upper_torso",
    "full_dress",
    "lower_skirt",
    "trousers",
    "outer_coat",
    "headwear",
    "footwear",
    "cloak",
)
VALID_NECKLINES = ("crew", "v_neck", "scoop", "off_shoulder", "mandarin", "hoodie_cowl", "open_lapel")
VALID_FABRIC_SHADERS = (
    "fabric_twill",
    "fabric_silk",
    "fabric_velvet",
    "cable_knit",
    "plaid_tartan",
    "leather_matte",
    "solid_cel",
    "metallic_trim",
)
VALID_DECAL_KINDS = ("star_badge", "medical_cross", "safety_stripes", "button_seam", "lace_border", "emblem_circle")
VALID_CHAIN_KINDS = ("arm_ik", "tentacle", "wing", "tail", "extra_head", "horn_pair")
VALID_MECH_KINDS = (
    "wheel_assembly",
    "visor_helmet",
    "blade_arm",
    "wing_membrane",
    "horn_pair",
    "handheld_mic",
    "magic_staff",
    "crown_tiara",
    "jetpack",
)
VALID_PROP_ARCHETYPES = (
    "masonry_wall",
    "rooftop_ledge",
    "stage_deck",
    "microphone_stand",
    "piano_keyboard",
    "tree_canopy",
    "moss_boulder",
    "wooden_table",
    "street_lamp",
    "custom_parts",
)
VALID_MIDGROUND_KINDS = (
    "none",
    "rooftop_village",
    "city_skyline",
    "forest_canopy",
    "concert_stage_crowd",
    "mountain_valley",
    "interior_room_walls",
)
VALID_SKY_STYLES = (
    "sunset_horizon",
    "twilight_glow",
    "concert_stage",
    "clear_day",
    "moonlit_night",
    "studio",
    "forest_mist",
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
    "perched_on",
    "next_to",
    "in_front_of",
    "behind",
)


def _clamp(val: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(val)
        if v != v:  # NaN guard
            return default
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return default


def _clamp_int(val: Any, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(val)))
    except (TypeError, ValueError):
        return default


def _clamp_vec3(val: Any, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if isinstance(val, (list, tuple)) and len(val) >= 3:
        try:
            x, y, z = float(val[0]), float(val[1]), float(val[2])
            if x == x and y == y and z == z:
                return (x, y, z)
        except (TypeError, ValueError):
            pass
    return default


def _clamp_hex(val: Any, default: str = "#cbd5e1") -> str:
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("#") and len(s) in (4, 7):
            h = s[1:]
            if all(c in "0123456789abcdefABCDEF" for c in h):
                if len(h) == 3:
                    return "#" + "".join(c * 2 for c in h).lower()
                return "#" + h.lower()
    return default


@dataclass
class MorphProportions:
    """Bounded semantic morphology weights translated to SH bases and bone lengths."""

    stature: float = 1.0          # [0.45, 1.8]
    limb_length: float = 1.0      # [0.5, 1.8]
    torso_width: float = 1.0      # [0.5, 1.8]
    head_scale: float = 1.0       # [0.6, 1.8]
    jaw_taper: float = 0.25       # [-1.0, 1.0]
    chin_projection: float = 0.2  # [-1.0, 1.0]
    cheek_fullness: float = 0.35  # [-1.0, 1.0]
    brow_ridge: float = 0.0       # [-1.0, 1.0]
    snout_length: float = 0.0     # [0.0, 1.0]
    muzzle_width: float = 0.0     # [-1.0, 1.0]
    cranium_dome: float = 0.15    # [-1.0, 1.0]
    asymmetry_smirk: float = 0.0  # [-1.0, 1.0]
    ear_length: float = 0.18      # [0.05, 1.2]
    ear_tilt_deg: float = 18.0    # [-45.0, 65.0]
    ear_taper: float = 0.35       # [0.02, 0.9]
    tail_length: float = 0.0      # [0.0, 2.2]
    tail_curl: float = 0.45       # [-1.0, 1.5]
    fur_plushness: float = 0.0    # [0.0, 1.0]


@dataclass
class ExpressionSpec:
    emotion: str = "calm"            # calm | happy | singing | fierce | curious | serene
    mouth_open: float = 0.0          # [0.0, 1.0]
    fangs: bool = False
    gaze_target: str = "camera"      # camera | left | right | up | down | support
    eye_style: str = "anime_round"
    iris_color: str = "#38bdf8"
    iris_secondary: str = "#1e3a8a"
    lash_color: str = "#1e1b4b"
    eye_scale: float = 1.0           # [0.5, 1.6]


@dataclass
class HairManeSpec:
    style: str = "none"
    length: float = 0.6              # [0.1, 1.8]
    clump_count: int = 16            # [4, 36]
    color: str = "#1e293b"
    secondary_color: str = "#475569"
    wind: Tuple[float, float, float] = (0.12, 0.0, -0.05)


@dataclass
class UVDecalSpec:
    kind: str = "star_badge"
    uv_t: float = 0.65               # [0.0, 1.0] along garment bone
    uv_theta_deg: float = 0.0        # [-180, 180] around torso/limb
    size: float = 0.14               # [0.03, 0.45]
    color: str = "#facc15"
    secondary_color: str = "#ffffff"


@dataclass
class GarmentLayerSpec:
    name: str = "tunic"
    region: str = "upper_torso"
    fit_offset: float = 0.018        # [0.004, 0.18]
    thickness: float = 0.012         # [0.004, 0.05]
    flare: float = 0.0               # [0.0, 1.8]
    pleat_count: int = 0             # [0, 32]
    pleat_amp: float = 0.0           # [0.0, 0.12]
    ruffle_amp: float = 0.0          # [0.0, 0.1]
    fold_amp: float = 0.018          # [0.0, 0.08]
    neckline: str = "crew"
    open_front_deg: float = 0.0      # [0.0, 90.0]
    sleeve_length: float = 0.85      # [0.0, 1.1]
    hem_length: float = 0.9          # [0.1, 1.4]
    collar: bool = False
    fabric_shader: str = "fabric_twill"
    primary_color: str = "#3b82f6"
    secondary_color: str = "#1e3a8a"
    accent_color: str = "#facc15"
    roughness: float = 0.65
    metallic: float = 0.0
    uv_decals: List[UVDecalSpec] = field(default_factory=list)


@dataclass
class LimbChainSpec:
    """Arbitrary extra articulated bone chain (extra arms, tentacles, wings, extra heads)."""

    id: str = "extra_chain_1"
    kind: str = "arm_ik"
    parent_socket: str = "mid_torso"  # upper_torso | mid_torso | pelvis | skull_crown | skull_back
    side: str = "pair"                # left | right | center | pair
    offset_local: Tuple[float, float, float] = (0.18, -0.05, 0.05)
    l1: float = 0.28
    l2: float = 0.26
    radius_base: float = 0.055
    radius_tip: float = 0.032
    ik_target: Tuple[float, float, float] = (0.35, -0.1, 0.28)
    pole_dir: Tuple[float, float, float] = (-0.2, -0.3, -1.0)
    color: str = ""                   # inherits actor skin/coat color if empty
    has_hand: bool = True


@dataclass
class MechanicalSocketSpec:
    """Attaches procedural wheels, visors, blades, horns, staffs, or props onto any bone socket."""

    name: str = "socket_gear"
    kind: str = "wheel_assembly"
    socket_bone: str = "left_ankle"   # any bone name or "both_ankles" / "both_wrists" / "head" / "right_hand"
    scale: float = 1.0
    offset_local: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    primary_color: str = "#0f172a"
    secondary_color: str = "#38bdf8"
    accent_color: str = "#facc15"
    metallic: float = 0.65
    roughness: float = 0.25
    emissive: float = 0.0


@dataclass
class ActorSpec:
    id: str = "hero_actor"
    label: str = "Character"
    archetype: str = "humanoid"
    placement: str = "center"
    support_target: str = ""
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0
    facing_deg: float = 12.0          # yaw relative to +Z (camera faces -Z)
    posture: str = "standing"
    proportions: MorphProportions = field(default_factory=MorphProportions)
    skin_color: str = "#fde2cf"
    skin_shadow_color: str = "#c98b7c"
    secondary_fur_color: str = "#fff7ed"
    coat_pattern: str = "solid"       # solid | tabby_stripes | spots | tuxedo
    left_hand_target: str = "auto"    # auto | hip_side | near_mouth | forward_reach | piano_keys | held_staff
    right_hand_target: str = "auto"
    expression: ExpressionSpec = field(default_factory=ExpressionSpec)
    hair_or_mane: HairManeSpec = field(default_factory=HairManeSpec)
    garments: List[GarmentLayerSpec] = field(default_factory=list)
    omit_bones: List[str] = field(default_factory=list)
    extra_limb_chains: List[LimbChainSpec] = field(default_factory=list)
    mechanical_attachments: List[MechanicalSocketSpec] = field(default_factory=list)


@dataclass
class CustomPartSpec:
    name: str = "part"
    primitive: str = "rounded_box"    # rounded_box | superquadric | swept_spline | torus | lathe
    size: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    color: str = "#94a3b8"
    secondary_color: str = "#475569"
    shader: str = "solid_cel"
    metallic: float = 0.1
    roughness: float = 0.55
    emissive: float = 0.0
    repeat_grid: Tuple[int, int, int] = (1, 1, 1)
    repeat_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4)
    staggered_masonry: bool = False


@dataclass
class PropSpec:
    id: str = "prop_1"
    label: str = "Architecture Prop"
    archetype: str = "masonry_wall"
    placement: str = "center"
    support_target: str = ""
    size: Tuple[float, float, float] = (3.4, 1.45, 0.52)
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    primary_color: str = "#b4533c"
    secondary_color: str = "#d6cfc7"
    accent_color: str = "#3f7d46"
    shader: str = "brick_masonry"
    has_foliage_ivy: bool = False
    has_top_coping: bool = True
    metallic: float = 0.05
    roughness: float = 0.72
    emissive: float = 0.0
    custom_parts: List[CustomPartSpec] = field(default_factory=list)


@dataclass
class MidgroundZoneSpec:
    kind: str = "rooftop_village"
    below_ledge_m: float = 0.75
    depth_start_m: float = 5.5
    depth_end_m: float = 15.0
    density: int = 12                 # [0, 32]
    silhouette_color: str = "#47354d"
    window_glow_color: str = "#fbbf24"
    haze_tint: str = "#f59e0b"
    haze_strength: float = 0.45


@dataclass
class LightEmitterSpec:
    kind: str = "sun_disc"            # sun_disc | moon_disc | stage_spotlight | lantern_orb
    direction_or_pos: Tuple[float, float, float] = (-0.45, 0.35, -0.82)
    color: str = "#ffd180"
    intensity: float = 10.0           # [1.0, 25.0] HDR
    radius: float = 0.14              # [0.02, 0.6]
    god_ray_strength: float = 0.55    # [0.0, 1.5]


@dataclass
class AtmosphereSpec:
    sky_style: str = "sunset_horizon"
    top_color: str = "#1e1b4b"
    horizon_color: str = "#fb7185"
    bottom_color: str = "#fbbf24"
    glow_color: str = "#fde68a"
    ground_enabled: bool = True
    ground_y: float = 0.0
    ground_pattern: str = "stone_voronoi"
    ground_color: str = "#475569"
    ground_secondary_color: str = "#334155"
    emitters: List[LightEmitterSpec] = field(default_factory=lambda: [LightEmitterSpec()])
    bokeh_count: int = 36
    bokeh_colors: List[str] = field(default_factory=lambda: ["#fde68a", "#fbbf24", "#fff7ed"])
    bokeh_size: float = 1.0
    dust_motes_count: int = 28


@dataclass
class CameraSpec:
    shot_type: str = "three_quarter"  # auto_medium | close_up | full_body | low_angle_hero | high_angle_ledge | three_quarter
    azimuth_deg: float = 22.0
    elevation_deg: float = 10.0
    roll_deg: float = 0.0
    fov_deg: float = 38.0
    distance_scale: float = 1.0
    focus_target: str = "subject_head"
    dof_strength: float = 0.35        # [0.0, 1.0]
    f_stop: float = 2.0               # [1.2, 8.0]


@dataclass
class LightingSpec:
    key_dir: Tuple[float, float, float] = (-0.55, 0.65, 0.52)
    key_color: str = "#fff1e0"
    shadow_tint: str = "#4a386b"
    ambient_color: str = "#3b3355"
    rim_color: str = "#ffb74d"
    rim_strength: float = 0.85
    sss_strength: float = 0.45
    sss_color: str = "#ff7847"
    ink_strength: float = 0.65
    bloom_intensity: float = 0.55
    chromatic_aberration: float = 0.0012
    film_grain: float = 0.012
    exposure: float = 1.08


@dataclass
class SceneDNA:
    title: str = "PrismStudio Scene"
    prompt: str = ""
    style_mode: str = "anime_npr"
    seed: int = 42
    camera: CameraSpec = field(default_factory=CameraSpec)
    lighting: LightingSpec = field(default_factory=LightingSpec)
    actors: List[ActorSpec] = field(default_factory=list)
    zone1_props: List[PropSpec] = field(default_factory=list)
    zone2_midground: MidgroundZoneSpec = field(default_factory=MidgroundZoneSpec)
    zone3_atmosphere: AtmosphereSpec = field(default_factory=AtmosphereSpec)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_morph(raw: Any) -> MorphProportions:
    if not isinstance(raw, dict):
        return MorphProportions()
    return MorphProportions(
        stature=_clamp(raw.get("stature"), 0.45, 1.8, 1.0),
        limb_length=_clamp(raw.get("limb_length"), 0.5, 1.8, 1.0),
        torso_width=_clamp(raw.get("torso_width"), 0.5, 1.8, 1.0),
        head_scale=_clamp(raw.get("head_scale"), 0.6, 1.8, 1.0),
        jaw_taper=_clamp(raw.get("jaw_taper"), -1.0, 1.0, 0.25),
        chin_projection=_clamp(raw.get("chin_projection"), -1.0, 1.0, 0.2),
        cheek_fullness=_clamp(raw.get("cheek_fullness"), -1.0, 1.0, 0.35),
        brow_ridge=_clamp(raw.get("brow_ridge"), -1.0, 1.0, 0.0),
        snout_length=_clamp(raw.get("snout_length"), 0.0, 1.0, 0.0),
        muzzle_width=_clamp(raw.get("muzzle_width"), -1.0, 1.0, 0.0),
        cranium_dome=_clamp(raw.get("cranium_dome"), -1.0, 1.0, 0.15),
        asymmetry_smirk=_clamp(raw.get("asymmetry_smirk"), -1.0, 1.0, 0.0),
        ear_length=_clamp(raw.get("ear_length"), 0.05, 1.2, 0.18),
        ear_tilt_deg=_clamp(raw.get("ear_tilt_deg"), -45.0, 65.0, 18.0),
        ear_taper=_clamp(raw.get("ear_taper"), 0.02, 0.9, 0.35),
        tail_length=_clamp(raw.get("tail_length"), 0.0, 2.2, 0.0),
        tail_curl=_clamp(raw.get("tail_curl"), -1.0, 1.5, 0.45),
        fur_plushness=_clamp(raw.get("fur_plushness"), 0.0, 1.0, 0.0),
    )


def _parse_expression(raw: Any) -> ExpressionSpec:
    if not isinstance(raw, dict):
        return ExpressionSpec()
    eye_style = str(raw.get("eye_style", "anime_round")).strip().lower()
    if eye_style not in VALID_EYE_STYLES:
        eye_style = "anime_round"
    return ExpressionSpec(
        emotion=str(raw.get("emotion", "calm")).strip().lower(),
        mouth_open=_clamp(raw.get("mouth_open"), 0.0, 1.0, 0.0),
        fangs=bool(raw.get("fangs", False)),
        gaze_target=str(raw.get("gaze_target", "camera")).strip().lower(),
        eye_style=eye_style,
        iris_color=_clamp_hex(raw.get("iris_color"), "#38bdf8"),
        iris_secondary=_clamp_hex(raw.get("iris_secondary"), "#1e3a8a"),
        lash_color=_clamp_hex(raw.get("lash_color"), "#1e1b4b"),
        eye_scale=_clamp(raw.get("eye_scale"), 0.5, 1.6, 1.0),
    )


def _parse_hair(raw: Any) -> HairManeSpec:
    if not isinstance(raw, dict):
        return HairManeSpec()
    style = str(raw.get("style", "none")).strip().lower()
    if style not in VALID_HAIR_STYLES:
        style = "long_flowing"
    return HairManeSpec(
        style=style,
        length=_clamp(raw.get("length"), 0.1, 1.8, 0.6),
        clump_count=_clamp_int(raw.get("clump_count"), 4, 36, 16),
        color=_clamp_hex(raw.get("color"), "#1e293b"),
        secondary_color=_clamp_hex(raw.get("secondary_color"), "#475569"),
        wind=_clamp_vec3(raw.get("wind"), (0.12, 0.0, -0.05)),
    )


def _parse_garment(raw: Any, idx: int) -> GarmentLayerSpec:
    if not isinstance(raw, dict):
        return GarmentLayerSpec(name=f"garment_{idx}")
    region = str(raw.get("region", "upper_torso")).strip().lower()
    if region not in VALID_GARMENT_REGIONS:
        region = "upper_torso"
    neck = str(raw.get("neckline", "crew")).strip().lower()
    if neck not in VALID_NECKLINES:
        neck = "crew"
    shader = str(raw.get("fabric_shader", "fabric_twill")).strip().lower()
    if shader not in VALID_FABRIC_SHADERS:
        shader = "fabric_twill"
    decals: List[UVDecalSpec] = []
    for d in raw.get("uv_decals", []) or []:
        if not isinstance(d, dict):
            continue
        dk = str(d.get("kind", "star_badge")).strip().lower()
        if dk not in VALID_DECAL_KINDS:
            dk = "star_badge"
        decals.append(
            UVDecalSpec(
                kind=dk,
                uv_t=_clamp(d.get("uv_t"), 0.0, 1.0, 0.65),
                uv_theta_deg=_clamp(d.get("uv_theta_deg"), -180.0, 180.0, 0.0),
                size=_clamp(d.get("size"), 0.03, 0.45, 0.14),
                color=_clamp_hex(d.get("color"), "#facc15"),
                secondary_color=_clamp_hex(d.get("secondary_color"), "#ffffff"),
            )
        )
    return GarmentLayerSpec(
        name=str(raw.get("name") or f"garment_{idx}").strip(),
        region=region,
        fit_offset=_clamp(raw.get("fit_offset"), 0.004, 0.18, 0.018),
        thickness=_clamp(raw.get("thickness"), 0.004, 0.05, 0.012),
        flare=_clamp(raw.get("flare"), 0.0, 1.8, 0.0),
        pleat_count=_clamp_int(raw.get("pleat_count"), 0, 32, 0),
        pleat_amp=_clamp(raw.get("pleat_amp"), 0.0, 0.12, 0.0),
        ruffle_amp=_clamp(raw.get("ruffle_amp"), 0.0, 0.1, 0.0),
        fold_amp=_clamp(raw.get("fold_amp"), 0.0, 0.08, 0.018),
        neckline=neck,
        open_front_deg=_clamp(raw.get("open_front_deg"), 0.0, 90.0, 0.0),
        sleeve_length=_clamp(raw.get("sleeve_length"), 0.0, 1.1, 0.85),
        hem_length=_clamp(raw.get("hem_length"), 0.1, 1.4, 0.9),
        collar=bool(raw.get("collar", False)),
        fabric_shader=shader,
        primary_color=_clamp_hex(raw.get("primary_color"), "#3b82f6"),
        secondary_color=_clamp_hex(raw.get("secondary_color"), "#1e3a8a"),
        accent_color=_clamp_hex(raw.get("accent_color"), "#facc15"),
        roughness=_clamp(raw.get("roughness"), 0.05, 1.0, 0.65),
        metallic=_clamp(raw.get("metallic"), 0.0, 1.0, 0.0),
        uv_decals=decals,
    )


def _parse_chain(raw: Any, idx: int) -> LimbChainSpec:
    if not isinstance(raw, dict):
        return LimbChainSpec(id=f"chain_{idx}")
    kind = str(raw.get("kind", "arm_ik")).strip().lower()
    if kind not in VALID_CHAIN_KINDS:
        kind = "arm_ik"
    side = str(raw.get("side", "pair")).strip().lower()
    if side not in ("left", "right", "center", "pair"):
        side = "pair"
    return LimbChainSpec(
        id=str(raw.get("id") or f"chain_{idx}").strip(),
        kind=kind,
        parent_socket=str(raw.get("parent_socket") or "mid_torso").strip().lower(),
        side=side,
        offset_local=_clamp_vec3(raw.get("offset_local"), (0.18, -0.05, 0.05)),
        l1=_clamp(raw.get("l1"), 0.08, 1.2, 0.28),
        l2=_clamp(raw.get("l2"), 0.08, 1.2, 0.26),
        radius_base=_clamp(raw.get("radius_base"), 0.015, 0.25, 0.055),
        radius_tip=_clamp(raw.get("radius_tip"), 0.01, 0.2, 0.032),
        ik_target=_clamp_vec3(raw.get("ik_target"), (0.35, -0.1, 0.28)),
        pole_dir=_clamp_vec3(raw.get("pole_dir"), (-0.2, -0.3, -1.0)),
        color=_clamp_hex(raw.get("color"), "") if raw.get("color") else "",
        has_hand=bool(raw.get("has_hand", True)),
    )


def _parse_mech(raw: Any, idx: int) -> MechanicalSocketSpec:
    if not isinstance(raw, dict):
        return MechanicalSocketSpec(name=f"mech_{idx}")
    kind = str(raw.get("kind", "wheel_assembly")).strip().lower()
    if kind not in VALID_MECH_KINDS:
        kind = "wheel_assembly"
    return MechanicalSocketSpec(
        name=str(raw.get("name") or f"mech_{idx}").strip(),
        kind=kind,
        socket_bone=str(raw.get("socket_bone") or "both_ankles").strip().lower(),
        scale=_clamp(raw.get("scale"), 0.2, 3.0, 1.0),
        offset_local=_clamp_vec3(raw.get("offset_local"), (0.0, 0.0, 0.0)),
        rotation_deg=_clamp_vec3(raw.get("rotation_deg"), (0.0, 0.0, 0.0)),
        primary_color=_clamp_hex(raw.get("primary_color"), "#0f172a"),
        secondary_color=_clamp_hex(raw.get("secondary_color"), "#38bdf8"),
        accent_color=_clamp_hex(raw.get("accent_color"), "#facc15"),
        metallic=_clamp(raw.get("metallic"), 0.0, 1.0, 0.65),
        roughness=_clamp(raw.get("roughness"), 0.05, 1.0, 0.25),
        emissive=_clamp(raw.get("emissive"), 0.0, 6.0, 0.0),
    )


def scene_dna_from_dict(data: Dict[str, Any]) -> SceneDNA:
    """Safely deserializes and clamps any raw JSON dict into a validated SceneDNA."""
    if not isinstance(data, dict):
        return SceneDNA()

    style_mode = str(data.get("style_mode", "anime_npr")).strip().lower()
    if style_mode not in VALID_STYLE_MODES:
        style_mode = "anime_npr"

    cam_raw = data.get("camera") if isinstance(data.get("camera"), dict) else {}
    camera = CameraSpec(
        shot_type=str(cam_raw.get("shot_type", "three_quarter")).strip().lower(),
        azimuth_deg=_clamp(cam_raw.get("azimuth_deg"), -180.0, 180.0, 22.0),
        elevation_deg=_clamp(cam_raw.get("elevation_deg"), -25.0, 75.0, 10.0),
        roll_deg=_clamp(cam_raw.get("roll_deg"), -25.0, 25.0, 0.0),
        fov_deg=_clamp(cam_raw.get("fov_deg"), 18.0, 72.0, 38.0),
        distance_scale=_clamp(cam_raw.get("distance_scale"), 0.55, 2.4, 1.0),
        focus_target=str(cam_raw.get("focus_target", "subject_head")).strip().lower(),
        dof_strength=_clamp(cam_raw.get("dof_strength"), 0.0, 1.0, 0.35),
        f_stop=_clamp(cam_raw.get("f_stop"), 1.2, 8.0, 2.0),
    )

    lit_raw = data.get("lighting") if isinstance(data.get("lighting"), dict) else {}
    lighting = LightingSpec(
        key_dir=_clamp_vec3(lit_raw.get("key_dir"), (-0.55, 0.65, 0.52)),
        key_color=_clamp_hex(lit_raw.get("key_color"), "#fff1e0"),
        shadow_tint=_clamp_hex(lit_raw.get("shadow_tint"), "#4a386b"),
        ambient_color=_clamp_hex(lit_raw.get("ambient_color"), "#3b3355"),
        rim_color=_clamp_hex(lit_raw.get("rim_color"), "#ffb74d"),
        rim_strength=_clamp(lit_raw.get("rim_strength"), 0.0, 2.0, 0.85),
        sss_strength=_clamp(lit_raw.get("sss_strength"), 0.0, 1.5, 0.45),
        sss_color=_clamp_hex(lit_raw.get("sss_color"), "#ff7847"),
        ink_strength=_clamp(lit_raw.get("ink_strength"), 0.0, 1.5, 0.65),
        bloom_intensity=_clamp(lit_raw.get("bloom_intensity"), 0.0, 1.5, 0.55),
        chromatic_aberration=_clamp(lit_raw.get("chromatic_aberration"), 0.0, 0.005, 0.0012),
        film_grain=_clamp(lit_raw.get("film_grain"), 0.0, 0.05, 0.012),
        exposure=_clamp(lit_raw.get("exposure"), 0.5, 2.0, 1.08),
    )

    actors: List[ActorSpec] = []
    for a_idx, a_raw in enumerate(data.get("actors", []) or []):
        if not isinstance(a_raw, dict):
            continue
        arch = str(a_raw.get("archetype", "humanoid")).strip().lower()
        if arch not in VALID_ARCHETYPES:
            arch = "humanoid"
        post = str(a_raw.get("posture", "standing")).strip().lower()
        if post not in VALID_POSTURES:
            post = "standing"
        place = str(a_raw.get("placement", "center")).strip().lower()
        if place not in VALID_PLACEMENTS:
            place = "center"
        actors.append(
            ActorSpec(
                id=str(a_raw.get("id") or f"actor_{a_idx}").strip(),
                label=str(a_raw.get("label") or f"Character {a_idx}").strip(),
                archetype=arch,
                placement=place,
                support_target=str(a_raw.get("support_target") or "").strip(),
                offset=_clamp_vec3(a_raw.get("offset"), (0.0, 0.0, 0.0)),
                scale=_clamp(a_raw.get("scale"), 0.2, 4.0, 1.0),
                facing_deg=_clamp(a_raw.get("facing_deg"), -180.0, 180.0, 12.0),
                posture=post,
                proportions=_parse_morph(a_raw.get("proportions")),
                skin_color=_clamp_hex(a_raw.get("skin_color"), "#fde2cf"),
                skin_shadow_color=_clamp_hex(a_raw.get("skin_shadow_color"), "#c98b7c"),
                secondary_fur_color=_clamp_hex(a_raw.get("secondary_fur_color"), "#fff7ed"),
                coat_pattern=str(a_raw.get("coat_pattern", "solid")).strip().lower(),
                left_hand_target=str(a_raw.get("left_hand_target", "auto")).strip().lower(),
                right_hand_target=str(a_raw.get("right_hand_target", "auto")).strip().lower(),
                expression=_parse_expression(a_raw.get("expression")),
                hair_or_mane=_parse_hair(a_raw.get("hair_or_mane")),
                garments=[_parse_garment(g, gi) for gi, g in enumerate(a_raw.get("garments", []) or [])],
                omit_bones=[str(b).strip().lower() for b in (a_raw.get("omit_bones") or []) if str(b).strip()],
                extra_limb_chains=[_parse_chain(c, ci) for ci, c in enumerate(a_raw.get("extra_limb_chains", []) or [])],
                mechanical_attachments=[_parse_mech(m, mi) for mi, m in enumerate(a_raw.get("mechanical_attachments", []) or [])],
            )
        )

    props: List[PropSpec] = []
    for p_idx, p_raw in enumerate(data.get("zone1_props", []) or []):
        if not isinstance(p_raw, dict):
            continue
        arch = str(p_raw.get("archetype", "masonry_wall")).strip().lower()
        if arch not in VALID_PROP_ARCHETYPES:
            arch = "masonry_wall"
        place = str(p_raw.get("placement", "center")).strip().lower()
        if place not in VALID_PLACEMENTS:
            place = "center"
        c_parts: List[CustomPartSpec] = []
        for c_i, cp in enumerate(p_raw.get("custom_parts", []) or []):
            if not isinstance(cp, dict):
                continue
            rg = cp.get("repeat_grid") or (1, 1, 1)
            c_parts.append(
                CustomPartSpec(
                    name=str(cp.get("name") or f"part_{c_i}").strip(),
                    primitive=str(cp.get("primitive", "rounded_box")).strip().lower(),
                    size=_clamp_vec3(cp.get("size"), (0.5, 0.5, 0.5)),
                    offset=_clamp_vec3(cp.get("offset"), (0.0, 0.0, 0.0)),
                    rotation_deg=_clamp_vec3(cp.get("rotation_deg"), (0.0, 0.0, 0.0)),
                    color=_clamp_hex(cp.get("color"), "#94a3b8"),
                    secondary_color=_clamp_hex(cp.get("secondary_color"), "#475569"),
                    shader=str(cp.get("shader", "solid_cel")).strip().lower(),
                    metallic=_clamp(cp.get("metallic"), 0.0, 1.0, 0.1),
                    roughness=_clamp(cp.get("roughness"), 0.05, 1.0, 0.55),
                    emissive=_clamp(cp.get("emissive"), 0.0, 6.0, 0.0),
                    repeat_grid=(
                        _clamp_int(rg[0] if len(rg) > 0 else 1, 1, 16, 1),
                        _clamp_int(rg[1] if len(rg) > 1 else 1, 1, 16, 1),
                        _clamp_int(rg[2] if len(rg) > 2 else 1, 1, 16, 1),
                    ),
                    repeat_spacing=_clamp_vec3(cp.get("repeat_spacing"), (0.4, 0.4, 0.4)),
                    staggered_masonry=bool(cp.get("staggered_masonry", False)),
                )
            )
        props.append(
            PropSpec(
                id=str(p_raw.get("id") or f"prop_{p_idx}").strip(),
                label=str(p_raw.get("label") or f"Prop {p_idx}").strip(),
                archetype=arch,
                placement=place,
                support_target=str(p_raw.get("support_target") or "").strip(),
                size=_clamp_vec3(p_raw.get("size"), (3.4, 1.45, 0.52)),
                offset=_clamp_vec3(p_raw.get("offset"), (0.0, 0.0, 0.0)),
                rotation_deg=_clamp_vec3(p_raw.get("rotation_deg"), (0.0, 0.0, 0.0)),
                primary_color=_clamp_hex(p_raw.get("primary_color"), "#b4533c"),
                secondary_color=_clamp_hex(p_raw.get("secondary_color"), "#d6cfc7"),
                accent_color=_clamp_hex(p_raw.get("accent_color"), "#3f7d46"),
                shader=str(p_raw.get("shader", "brick_masonry")).strip().lower(),
                has_foliage_ivy=bool(p_raw.get("has_foliage_ivy", False)),
                has_top_coping=bool(p_raw.get("has_top_coping", True)),
                metallic=_clamp(p_raw.get("metallic"), 0.0, 1.0, 0.05),
                roughness=_clamp(p_raw.get("roughness"), 0.05, 1.0, 0.72),
                emissive=_clamp(p_raw.get("emissive"), 0.0, 6.0, 0.0),
                custom_parts=c_parts,
            )
        )

    mid_raw = data.get("zone2_midground") if isinstance(data.get("zone2_midground"), dict) else {}
    mid_kind = str(mid_raw.get("kind", "rooftop_village")).strip().lower()
    if mid_kind not in VALID_MIDGROUND_KINDS:
        mid_kind = "rooftop_village"
    zone2 = MidgroundZoneSpec(
        kind=mid_kind,
        below_ledge_m=_clamp(mid_raw.get("below_ledge_m"), -2.0, 5.0, 0.75),
        depth_start_m=_clamp(mid_raw.get("depth_start_m"), 2.5, 18.0, 5.5),
        depth_end_m=_clamp(mid_raw.get("depth_end_m"), 4.0, 28.0, 15.0),
        density=_clamp_int(mid_raw.get("density"), 0, 32, 12),
        silhouette_color=_clamp_hex(mid_raw.get("silhouette_color"), "#47354d"),
        window_glow_color=_clamp_hex(mid_raw.get("window_glow_color"), "#fbbf24"),
        haze_tint=_clamp_hex(mid_raw.get("haze_tint"), "#f59e0b"),
        haze_strength=_clamp(mid_raw.get("haze_strength"), 0.0, 1.0, 0.45),
    )

    atmo_raw = data.get("zone3_atmosphere") if isinstance(data.get("zone3_atmosphere"), dict) else {}
    sky = str(atmo_raw.get("sky_style", "sunset_horizon")).strip().lower()
    if sky not in VALID_SKY_STYLES:
        sky = "sunset_horizon"
    emitters: List[LightEmitterSpec] = []
    for em in atmo_raw.get("emitters", []) or []:
        if not isinstance(em, dict):
            continue
        emitters.append(
            LightEmitterSpec(
                kind=str(em.get("kind", "sun_disc")).strip().lower(),
                direction_or_pos=_clamp_vec3(em.get("direction_or_pos"), (-0.45, 0.35, -0.82)),
                color=_clamp_hex(em.get("color"), "#ffd180"),
                intensity=_clamp(em.get("intensity"), 1.0, 25.0, 10.0),
                radius=_clamp(em.get("radius"), 0.02, 0.6, 0.14),
                god_ray_strength=_clamp(em.get("god_ray_strength"), 0.0, 1.5, 0.55),
            )
        )
    if not emitters:
        emitters = [LightEmitterSpec()]
    bcols_raw = atmo_raw.get("bokeh_colors")
    bcols = [_clamp_hex(c, "#fde68a") for c in bcols_raw[:6]] if isinstance(bcols_raw, list) and bcols_raw else ["#fde68a", "#fbbf24", "#fff7ed"]
    zone3 = AtmosphereSpec(
        sky_style=sky,
        top_color=_clamp_hex(atmo_raw.get("top_color"), "#1e1b4b"),
        horizon_color=_clamp_hex(atmo_raw.get("horizon_color"), "#fb7185"),
        bottom_color=_clamp_hex(atmo_raw.get("bottom_color"), "#fbbf24"),
        glow_color=_clamp_hex(atmo_raw.get("glow_color"), "#fde68a"),
        ground_enabled=bool(atmo_raw.get("ground_enabled", True)),
        ground_y=_clamp(atmo_raw.get("ground_y"), -4.0, 2.0, 0.0),
        ground_pattern=str(atmo_raw.get("ground_pattern", "stone_voronoi")).strip().lower(),
        ground_color=_clamp_hex(atmo_raw.get("ground_color"), "#475569"),
        ground_secondary_color=_clamp_hex(atmo_raw.get("ground_secondary_color"), "#334155"),
        emitters=emitters,
        bokeh_count=_clamp_int(atmo_raw.get("bokeh_count"), 0, 120, 36),
        bokeh_colors=bcols,
        bokeh_size=_clamp(atmo_raw.get("bokeh_size"), 0.3, 3.0, 1.0),
        dust_motes_count=_clamp_int(atmo_raw.get("dust_motes_count"), 0, 120, 28),
    )

    return SceneDNA(
        title=str(data.get("title") or "PrismStudio Scene").strip(),
        prompt=str(data.get("prompt") or "").strip(),
        style_mode=style_mode,
        seed=_clamp_int(data.get("seed"), 0, 999999, 42),
        camera=camera,
        lighting=lighting,
        actors=actors,
        zone1_props=props,
        zone2_midground=zone2,
        zone3_atmosphere=zone3,
    )


def get_director_json_schema() -> Dict[str, Any]:
    """Returns a flat-friendly JSON Schema for Groq/Ollama structured SceneDNA generation."""
    vec3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "style_mode": {"type": "string", "enum": list(VALID_STYLE_MODES)},
            "seed": {"type": "integer"},
            "camera": {
                "type": "object",
                "properties": {
                    "shot_type": {"type": "string"},
                    "azimuth_deg": {"type": "number"},
                    "elevation_deg": {"type": "number"},
                    "fov_deg": {"type": "number"},
                    "distance_scale": {"type": "number"},
                    "dof_strength": {"type": "number"},
                    "f_stop": {"type": "number"},
                },
                "required": ["azimuth_deg", "elevation_deg"],
            },
            "lighting": {
                "type": "object",
                "properties": {
                    "key_dir": vec3,
                    "key_color": {"type": "string"},
                    "shadow_tint": {"type": "string"},
                    "ambient_color": {"type": "string"},
                    "rim_color": {"type": "string"},
                    "rim_strength": {"type": "number"},
                    "sss_strength": {"type": "number"},
                    "ink_strength": {"type": "number"},
                    "bloom_intensity": {"type": "number"},
                },
                "required": ["key_color", "rim_color"],
            },
            "actors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "archetype": {"type": "string", "enum": list(VALID_ARCHETYPES)},
                        "placement": {"type": "string", "enum": list(VALID_PLACEMENTS)},
                        "support_target": {"type": "string"},
                        "scale": {"type": "number"},
                        "facing_deg": {"type": "number"},
                        "posture": {"type": "string", "enum": list(VALID_POSTURES)},
                        "skin_color": {"type": "string"},
                        "skin_shadow_color": {"type": "string"},
                        "secondary_fur_color": {"type": "string"},
                        "coat_pattern": {"type": "string"},
                        "left_hand_target": {"type": "string"},
                        "right_hand_target": {"type": "string"},
                        "proportions": {"type": "object"},
                        "expression": {"type": "object"},
                        "hair_or_mane": {"type": "object"},
                        "garments": {"type": "array", "items": {"type": "object"}},
                        "omit_bones": {"type": "array", "items": {"type": "string"}},
                        "extra_limb_chains": {"type": "array", "items": {"type": "object"}},
                        "mechanical_attachments": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": ["id", "archetype", "posture"],
                },
            },
            "zone1_props": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "archetype": {"type": "string", "enum": list(VALID_PROP_ARCHETYPES)},
                        "placement": {"type": "string", "enum": list(VALID_PLACEMENTS)},
                        "support_target": {"type": "string"},
                        "size": vec3,
                        "primary_color": {"type": "string"},
                        "secondary_color": {"type": "string"},
                        "accent_color": {"type": "string"},
                        "shader": {"type": "string"},
                        "has_foliage_ivy": {"type": "boolean"},
                        "has_top_coping": {"type": "boolean"},
                    },
                    "required": ["id", "archetype", "size"],
                },
            },
            "zone2_midground": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(VALID_MIDGROUND_KINDS)},
                    "below_ledge_m": {"type": "number"},
                    "density": {"type": "integer"},
                    "silhouette_color": {"type": "string"},
                    "window_glow_color": {"type": "string"},
                    "haze_tint": {"type": "string"},
                    "haze_strength": {"type": "number"},
                },
                "required": ["kind"],
            },
            "zone3_atmosphere": {
                "type": "object",
                "properties": {
                    "sky_style": {"type": "string", "enum": list(VALID_SKY_STYLES)},
                    "top_color": {"type": "string"},
                    "horizon_color": {"type": "string"},
                    "bottom_color": {"type": "string"},
                    "glow_color": {"type": "string"},
                    "ground_enabled": {"type": "boolean"},
                    "ground_pattern": {"type": "string"},
                    "ground_color": {"type": "string"},
                    "bokeh_count": {"type": "integer"},
                    "dust_motes_count": {"type": "integer"},
                    "emitters": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["sky_style", "top_color", "horizon_color"],
            },
        },
        "required": ["title", "style_mode", "actors", "zone1_props", "zone2_midground", "zone3_atmosphere"],
    }
