"""
LLM Scene Director & Compositional Offline Presets for PrismStudio.

Translates arbitrary natural-language prompts into structured SceneDNA using
Groq / Ollama purely as a semantic director (never guessing raw 3D vertex math),
paired with built-in showcase presets and smart compositional offline synthesis.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..compiler.validator import validate_and_repair_dna
from ..schema.scene_dna import (
    SceneDNA,
    VALID_ARCHETYPES,
    VALID_CHAIN_KINDS,
    VALID_DECAL_KINDS,
    VALID_EYE_STYLES,
    VALID_FABRIC_SHADERS,
    VALID_GARMENT_REGIONS,
    VALID_HAIR_STYLES,
    VALID_MECH_KINDS,
    VALID_MIDGROUND_KINDS,
    VALID_NECKLINES,
    VALID_POSTURES,
    VALID_PROP_ARCHETYPES,
    VALID_SKY_STYLES,
    VALID_STYLE_MODES,
    get_director_json_schema,
    scene_dna_from_dict,
)
from .llm_client import DirectorAPIError, call_text_llm, extract_json_object, load_director_env


@dataclass
class DirectorPlanResult:
    dna: SceneDNA
    source: str                      # "groq" | "ollama" | "offline" | "preset"
    model_used: str = ""
    attempts: int = 1
    warnings: List[str] = field(default_factory=list)


DIRECTOR_SYSTEM_PROMPT = f"""You are a visionary Anime & Stylized 3D Art Director.
Convert any natural-language image prompt into a strict `SceneDNA` JSON specification
for PrismStudio's compositional pure-math compiler. NEVER output code or raw coordinates
for body limbs; compose living beings, outfits, cybernetics, and 3-zone worlds semantically.

COMPOSITIONAL GRAMMAR CAPABILITIES:
1. Style Modes: {list(VALID_STYLE_MODES)} (`anime_npr` for expressive cel shading + ink outlines + Kajiya-Kay hair sheen; `stylized_pbr` for soft plush realistic shading).
2. Living Beings (`actors`):
   - `archetype`: {list(VALID_ARCHETYPES)} (`quadruped` for cats/foxes/dogs; `humanoid` for humans/elves/dwarfs/vampires/cyborgs/aliens).
   - `posture`: {list(VALID_POSTURES)} (`sitting_edge` drapes tails cleanly over wall/ledge edges!).
   - `proportions`: morph weights (`stature`, `limb_length`, `torso_width`, `head_scale`, `jaw_taper`, `chin_projection`, `cheek_fullness`, `snout_length`, `ear_length`, `ear_tilt_deg`, `ear_taper`, `tail_length`, `tail_curl`, `fur_plushness`).
     * Elf: humanoid with `ear_length: 0.42`, `ear_tilt_deg: 28`, `ear_taper: 0.15`.
     * Dwarf: `stature: 0.72`, `torso_width: 1.38`, `hair_or_mane: {{"style": "braided_beard"}}`.
     * Cat: `archetype: "quadruped"`, `posture: "sitting_edge"` (on walls!), `snout_length: 0.28`, `cheek_fullness: 0.65`, `ear_length: 0.16`, `tail_length: 0.78`, `fur_plushness: 0.75`, `expression: {{"eye_style": "feline_slit", "eye_scale": 1.15}}`.
   - Generative Multi-Limb & Cybernetic Composability:
     * `omit_bones`: e.g. `["left_foot", "right_foot"]` when replacing feet with wheels or wings.
     * `extra_limb_chains`: attach extra 2-bone IK arms/wings/tentacles (`kind`: {list(VALID_CHAIN_KINDS)}, `parent_socket`: `"mid_torso"` | `"upper_torso"` | `"pelvis"`, `side`: `"pair"`, `ik_target`: `[x, y, z]`). Use this for 4-armed characters!
     * `mechanical_attachments`: `kind`: {list(VALID_MECH_KINDS)}, `socket_bone`: `"both_ankles"` (e.g. `wheel_assembly` for XLR8-style wheel-legs!), `"head"` (`visor_helmet` / `horn_pair`), `"right_hand"` (`magic_staff` / `handheld_mic`).
3. Body-UV Outfits (`garments`):
   - `region`: {list(VALID_GARMENT_REGIONS)}, `neckline`: {list(VALID_NECKLINES)}, `fabric_shader`: {list(VALID_FABRIC_SHADERS)}
   - Use `flare`, `pleat_count`, `pleat_amp`, `ruffle_amp`, `open_front_deg`, `fit_offset`, and `uv_decals` (`kind`: {list(VALID_DECAL_KINDS)}) for nurse dresses (`medical_cross`), police uniforms (`star_badge`), firefighter coats (`safety_stripes`), hoodies, or fantasy cloaks.
4. Hair & Expression:
   - `hair_or_mane.style`: {list(VALID_HAIR_STYLES)}
   - `expression.eye_style`: {list(VALID_EYE_STYLES)}
5. MANDATORY 3-ZONE WORLD COMPOSITION:
   - `zone1_props`: Hero support & foreground architecture (`archetype`: {list(VALID_PROP_ARCHETYPES)}). ALWAYS build walls as `masonry_wall` (`size: [3.6, 1.5, 0.54]`, `has_top_coping: true`, `has_foliage_ivy: true`) and bind sitting actors' `support_target` to the wall's `id`!
   - `zone2_midground`: Distant silhouettes (`kind`: {list(VALID_MIDGROUND_KINDS)}) with `below_ledge_m: 0.75` for elevated wall/rooftop scenes so the viewer feels height!
   - `zone3_atmosphere`: Sky gradient (`sky_style`: {list(VALID_SKY_STYLES)}) + HDR `emitters` (`intensity: 8.0-18.0`, `god_ray_strength: 0.5-0.9`) + `bokeh_count` + `dust_motes_count`.
"""


PRESET_SCENES: Dict[str, Dict[str, Any]] = {
    "Cute Cat on High Sunset Wall (Anime)": {
        "title": "Cute Cat on High Sunset Wall",
        "prompt": "cute cat sitting on a high wall at sunset",
        "style_mode": "anime_npr",
        "seed": 17,
        "camera": {
            "shot_type": "three_quarter",
            "azimuth_deg": 24.0,
            "elevation_deg": 8.0,
            "fov_deg": 38.0,
            "distance_scale": 0.96,
            "dof_strength": 0.42,
            "f_stop": 1.8,
        },
        "lighting": {
            "key_dir": [-0.55, 0.48, -0.68],
            "key_color": "#ffd8b0",
            "shadow_tint": "#5b4475",
            "ambient_color": "#45375c",
            "rim_color": "#ffb347",
            "rim_strength": 1.15,
            "sss_strength": 0.55,
            "sss_color": "#ff7a45",
            "ink_strength": 0.48,
            "bloom_intensity": 0.42,
            "exposure": 1.06,
        },
        "actors": [
            {
                "id": "sunset_cat",
                "label": "Cute Orange Tabby Cat",
                "archetype": "quadruped",
                "placement": "sitting_on",
                "support_target": "high_brick_wall",
                "offset": [0.0, 0.0, 0.04],
                "scale": 1.0,
                "facing_deg": 18.0,
                "posture": "sitting_edge",
                "skin_color": "#f4a261",
                "skin_shadow_color": "#a45a49",
                "secondary_fur_color": "#fff3e2",
                "coat_pattern": "tabby_stripes",
                "proportions": {
                    "stature": 1.0,
                    "limb_length": 1.0,
                    "torso_width": 1.0,
                    "head_scale": 1.08,
                    "cheek_fullness": 0.68,
                    "snout_length": 0.28,
                    "muzzle_width": 0.2,
                    "ear_length": 0.17,
                    "ear_tilt_deg": 16.0,
                    "tail_length": 0.85,
                    "tail_curl": 0.65,
                    "fur_plushness": 0.78,
                },
                "expression": {
                    "emotion": "curious",
                    "mouth_open": 0.0,
                    "gaze_target": "camera",
                    "eye_style": "feline_slit",
                    "iris_color": "#34d399",
                    "iris_secondary": "#065f46",
                    "lash_color": "#2b1d2e",
                    "eye_scale": 1.18,
                },
            }
        ],
        "zone1_props": [
            {
                "id": "high_brick_wall",
                "label": "High Sunset Masonry Wall",
                "archetype": "masonry_wall",
                "placement": "center",
                "size": [3.8, 1.55, 0.54],
                "primary_color": "#a84c38",
                "secondary_color": "#c9bfb5",
                "accent_color": "#3d7a47",
                "shader": "brick_masonry",
                "has_foliage_ivy": True,
                "has_top_coping": True,
            }
        ],
        "zone2_midground": {
            "kind": "rooftop_village",
            "below_ledge_m": 0.85,
            "depth_start_m": 5.5,
            "depth_end_m": 15.0,
            "density": 14,
            "silhouette_color": "#43314c",
            "window_glow_color": "#fbbf24",
            "haze_tint": "#f59e0b",
            "haze_strength": 0.55,
        },
        "zone3_atmosphere": {
            "sky_style": "sunset_horizon",
            "top_color": "#1e1b4b",
            "horizon_color": "#f4726a",
            "bottom_color": "#fbbf24",
            "glow_color": "#fde68a",
            "ground_enabled": True,
            "ground_y": 0.0,
            "ground_color": "#474056",
            "ground_secondary_color": "#2d2838",
            "emitters": [
                {
                    "kind": "sun_disc",
                    "direction_or_pos": [-0.38, 0.22, -0.90],
                    "color": "#ffd27d",
                    "intensity": 9.5,
                    "radius": 0.085,
                    "god_ray_strength": 0.65,
                }
            ],
            "bokeh_count": 28,
            "dust_motes_count": 24,
        },
    },
    "Cute Cat on High Sunset Wall (Stylized PBR)": {
        "title": "Cute Cat on High Sunset Wall (PBR)",
        "prompt": "cute cat sitting on a high wall at sunset in soft stylized realistic 3D",
        "style_mode": "stylized_pbr",
        "seed": 17,
        "camera": {
            "shot_type": "three_quarter",
            "azimuth_deg": 24.0,
            "elevation_deg": 8.0,
            "fov_deg": 38.0,
            "distance_scale": 0.96,
            "dof_strength": 0.45,
        },
        "lighting": {
            "key_dir": [-0.55, 0.48, -0.68],
            "key_color": "#ffe2c2",
            "shadow_tint": "#4c3b61",
            "ambient_color": "#4a3b5e",
            "rim_color": "#ffb347",
            "rim_strength": 0.95,
            "sss_strength": 0.48,
            "sss_color": "#ff7a45",
            "ink_strength": 0.0,
            "bloom_intensity": 0.55,
            "exposure": 1.12,
        },
        "actors": [
            {
                "id": "sunset_cat_pbr",
                "label": "Plush Tabby Cat",
                "archetype": "quadruped",
                "placement": "sitting_on",
                "support_target": "high_brick_wall",
                "offset": [0.0, 0.0, 0.04],
                "scale": 1.0,
                "facing_deg": 18.0,
                "posture": "sitting_edge",
                "skin_color": "#f4a261",
                "skin_shadow_color": "#9a5040",
                "secondary_fur_color": "#fff3e2",
                "coat_pattern": "tabby_stripes",
                "proportions": {
                    "head_scale": 1.06,
                    "cheek_fullness": 0.65,
                    "snout_length": 0.30,
                    "ear_length": 0.17,
                    "ear_tilt_deg": 15.0,
                    "tail_length": 0.85,
                    "tail_curl": 0.65,
                    "fur_plushness": 0.85,
                },
                "expression": {
                    "eye_style": "feline_slit",
                    "iris_color": "#34d399",
                    "lash_color": "#2b1d2e",
                    "eye_scale": 1.14,
                },
            }
        ],
        "zone1_props": [
            {
                "id": "high_brick_wall",
                "label": "High Sunset Masonry Wall",
                "archetype": "masonry_wall",
                "placement": "center",
                "size": [3.8, 1.55, 0.54],
                "primary_color": "#a84c38",
                "secondary_color": "#c9bfb5",
                "accent_color": "#3d7a47",
                "shader": "brick_masonry",
                "has_foliage_ivy": True,
                "has_top_coping": True,
            }
        ],
        "zone2_midground": {
            "kind": "rooftop_village",
            "below_ledge_m": 0.85,
            "density": 14,
            "silhouette_color": "#43314c",
            "window_glow_color": "#fbbf24",
            "haze_tint": "#f59e0b",
            "haze_strength": 0.55,
        },
        "zone3_atmosphere": {
            "sky_style": "sunset_horizon",
            "top_color": "#1e1b4b",
            "horizon_color": "#f4726a",
            "bottom_color": "#fbbf24",
            "emitters": [{"kind": "sun_disc", "direction_or_pos": [-0.38, 0.24, -0.89], "color": "#ffd27d", "intensity": 14.0, "radius": 0.13, "god_ray_strength": 0.78}],
            "bokeh_count": 28,
            "dust_motes_count": 24,
        },
    },
    "Anime Stage Singer (Ponytail & Hoodie)": {
        "title": "Anime Stage Vocalist",
        "prompt": "anime girl with high ponytail and oversized hoodie singing into a microphone on a lit stage",
        "style_mode": "anime_npr",
        "seed": 23,
        "camera": {
            "shot_type": "three_quarter",
            "azimuth_deg": 28.0,
            "elevation_deg": 9.0,
            "fov_deg": 36.0,
            "distance_scale": 0.92,
            "dof_strength": 0.45,
        },
        "lighting": {
            "key_dir": [0.28, 0.68, 0.68],
            "key_color": "#ffe8cc",
            "shadow_tint": "#453a6b",
            "ambient_color": "#2f294d",
            "rim_color": "#ff9e44",
            "rim_strength": 1.2,
            "sss_strength": 0.52,
            "sss_color": "#ff7355",
            "ink_strength": 0.62,
            "bloom_intensity": 0.72,
            "exposure": 1.12,
        },
        "actors": [
            {
                "id": "singer_hero",
                "label": "Anime Vocalist",
                "archetype": "humanoid",
                "placement": "on_top_of",
                "support_target": "concert_stage",
                "scale": 1.0,
                "facing_deg": 24.0,
                "posture": "singing_pose",
                "skin_color": "#f4c7ad",
                "skin_shadow_color": "#c98b7c",
                "left_hand_target": "near_mouth",
                "right_hand_target": "hip_side",
                "proportions": {
                    "jaw_taper": 0.42,
                    "chin_projection": 0.28,
                    "cheek_fullness": 0.32,
                    "head_scale": 1.04,
                },
                "expression": {
                    "emotion": "singing",
                    "mouth_open": 0.65,
                    "eye_style": "anime_round",
                    "iris_color": "#38bdf8",
                    "lash_color": "#1e1b4b",
                    "eye_scale": 1.12,
                },
                "hair_or_mane": {
                    "style": "high_ponytail",
                    "length": 0.65,
                    "clump_count": 18,
                    "color": "#6b3725",
                    "secondary_color": "#a85c3a",
                    "wind": [0.24, 0.06, -0.16],
                },
                "garments": [
                    {
                        "name": "oversized_hoodie",
                        "region": "upper_torso",
                        "fit_offset": 0.020,
                        "thickness": 0.012,
                        "fold_amp": 0.012,
                        "neckline": "hoodie_cowl",
                        "sleeve_length": 0.88,
                        "fabric_shader": "fabric_twill",
                        "roughness": 0.82,
                        "primary_color": "#3b4f82",
                        "secondary_color": "#1e293b",
                    },
                    {
                        "name": "pleated_skirt",
                        "region": "lower_skirt",
                        "fit_offset": 0.016,
                        "flare": 0.44,
                        "pleat_count": 16,
                        "pleat_amp": 0.008,
                        "hem_length": 0.82,
                        "fabric_shader": "plaid_tartan",
                        "roughness": 0.78,
                        "primary_color": "#b91c1c",
                        "secondary_color": "#1e293b",
                    },
                    {
                        "name": "stage_leggings",
                        "region": "trousers",
                        "fit_offset": 0.009,
                        "fabric_shader": "fabric_twill",
                        "roughness": 0.76,
                        "primary_color": "#262b38",
                        "secondary_color": "#0f172a",
                    },
                ],
                "mechanical_attachments": [
                    {
                        "name": "stage_mic",
                        "kind": "handheld_mic",
                        "socket_bone": "left_hand",
                        "primary_color": "#1e293b",
                        "secondary_color": "#cbd5e1",
                    }
                ],
            }
        ],
        "zone1_props": [
            {
                "id": "concert_stage",
                "label": "Stage Floor",
                "archetype": "stage_deck",
                "placement": "center",
                "size": [6.5, 0.25, 4.8],
                "primary_color": "#292524",
                "secondary_color": "#1c1917",
                "shader": "wood_grain",
            },
            {
                "id": "stage_piano",
                "label": "Grand Keyboard",
                "archetype": "piano_keyboard",
                "placement": "behind",
                "size": [1.5, 0.82, 0.58],
                "offset": [0.75, 0.25, -0.65],
                "rotation_deg": [0.0, -28.0, 0.0],
                "primary_color": "#0f172a",
            },
        ],
        "zone2_midground": {
            "kind": "concert_stage_crowd",
            "below_ledge_m": 0.15,
            "density": 14,
            "silhouette_color": "#1e1b2e",
            "haze_tint": "#f59e0b",
            "haze_strength": 0.42,
        },
        "zone3_atmosphere": {
            "sky_style": "concert_stage",
            "top_color": "#090814",
            "horizon_color": "#3b2140",
            "bottom_color": "#581c3a",
            "emitters": [{"kind": "stage_spotlight", "direction_or_pos": [-0.32, 0.42, -0.84], "color": "#ffcc70", "intensity": 16.0, "radius": 0.11, "god_ray_strength": 0.85}],
            "bokeh_count": 38,
            "dust_motes_count": 32,
        },
    },
    "Four-Armed Elf Mystic (Multi-Limb + Cloak)": {
        "title": "Four-Armed Elf Mystic",
        "prompt": "creative four-armed elf mystic in a flared emerald cloak with glowing magic staff in a misty forest",
        "style_mode": "anime_npr",
        "seed": 31,
        "camera": {
            "shot_type": "three_quarter",
            "azimuth_deg": 20.0,
            "elevation_deg": 9.0,
            "fov_deg": 38.0,
            "distance_scale": 1.0,
            "dof_strength": 0.35,
        },
        "lighting": {
            "key_dir": [-0.5, 0.72, 0.48],
            "key_color": "#e8ffe8",
            "shadow_tint": "#273b4f",
            "ambient_color": "#1e3a3a",
            "rim_color": "#6ee7b7",
            "rim_strength": 1.05,
            "sss_strength": 0.45,
            "ink_strength": 0.58,
            "bloom_intensity": 0.65,
        },
        "actors": [
            {
                "id": "elf_mystic",
                "label": "Four-Armed Elf Mystic",
                "archetype": "humanoid",
                "placement": "center",
                "scale": 1.0,
                "facing_deg": 10.0,
                "posture": "standing",
                "skin_color": "#f2c5ab",
                "secondary_fur_color": "#ffffff",
                "left_hand_target": "raised_wave",
                "right_hand_target": "held_staff",
                "proportions": {
                    "jaw_taper": 0.45,
                    "chin_projection": 0.32,
                    "ear_length": 0.44,
                    "ear_tilt_deg": 26.0,
                    "ear_taper": 0.14,
                },
                "expression": {
                    "emotion": "serene",
                    "eye_style": "sharp_mystic",
                    "iris_color": "#d97706",
                    "lash_color": "#0f172a",
                    "eye_scale": 1.15,
                },
                "hair_or_mane": {
                    "style": "long_flowing",
                    "length": 0.75,
                    "clump_count": 16,
                    "color": "#94a3b8",
                    "secondary_color": "#64748b",
                },
                "garments": [
                    {
                        "name": "mystic_tunic",
                        "region": "upper_torso",
                        "fit_offset": 0.014,
                        "neckline": "v_neck",
                        "sleeve_length": 0.68,
                        "fabric_shader": "fabric_silk",
                        "roughness": 0.72,
                        "primary_color": "#15803d",
                        "secondary_color": "#0d9488",
                    },
                    {
                        "name": "flared_cloak",
                        "region": "cloak",
                        "fit_offset": 0.012,
                        "flare": 0.42,
                        "open_front_deg": 0.0,
                        "hem_length": 0.96,
                        "fabric_shader": "fabric_velvet",
                        "roughness": 0.80,
                        "primary_color": "#0f766e",
                        "secondary_color": "#fbbf24",
                    },
                    {
                        "name": "ranger_leggings",
                        "region": "trousers",
                        "fit_offset": 0.009,
                        "fabric_shader": "fabric_twill",
                        "roughness": 0.75,
                        "primary_color": "#1f2937",
                        "secondary_color": "#0f172a",
                    },
                ],
                "extra_limb_chains": [
                    {
                        "id": "lower_arms",
                        "kind": "arm_ik",
                        "parent_socket": "mid_torso",
                        "side": "pair",
                        "offset_local": [0.14, -0.05, 0.02],
                        "l1": 0.27,
                        "l2": 0.25,
                        "radius_base": 0.044,
                        "radius_tip": 0.026,
                        "ik_target": [0.46, -0.18, 0.24],
                    }
                ],
                "mechanical_attachments": [
                    {
                        "name": "mystic_staff",
                        "kind": "magic_staff",
                        "socket_bone": "right_hand",
                        "primary_color": "#78350f",
                        "secondary_color": "#34d399",
                        "emissive": 2.2,
                    }
                ],
            }
        ],
        "zone1_props": [
            {
                "id": "moss_stone",
                "label": "Ancient Stone Pedestal",
                "archetype": "moss_boulder",
                "placement": "behind",
                "size": [1.8, 0.75, 1.2],
                "offset": [-1.2, 0.0, -0.8],
                "primary_color": "#475569",
                "secondary_color": "#3f7d46",
                "shader": "stone_voronoi",
            }
        ],
        "zone2_midground": {
            "kind": "forest_canopy",
            "below_ledge_m": 0.0,
            "density": 14,
            "silhouette_color": "#132a28",
            "haze_tint": "#34d399",
            "haze_strength": 0.45,
        },
        "zone3_atmosphere": {
            "sky_style": "forest_mist",
            "top_color": "#062c2b",
            "horizon_color": "#2dd4bf",
            "bottom_color": "#a7f3d0",
            "ground_enabled": True,
            "ground_color": "#2d4a35",
            "ground_secondary_color": "#1d3023",
            "emitters": [{"kind": "sun_disc", "direction_or_pos": [-0.35, 0.38, -0.85], "color": "#bbf7d0", "intensity": 11.0, "radius": 0.14, "god_ray_strength": 0.75}],
            "bokeh_count": 24,
            "dust_motes_count": 30,
        },
    },
    "XLR8 Wheel-Leg Speedster (Alien Hybrid)": {
        "title": "Wheel-Leg Alien Speedster",
        "prompt": "futuristic alien speedster hero with wheel-legs, aerodynamic tail, and glowing visor helmet",
        "style_mode": "anime_npr",
        "seed": 77,
        "camera": {
            "shot_type": "three_quarter",
            "azimuth_deg": 32.0,
            "elevation_deg": 8.0,
            "fov_deg": 38.0,
            "distance_scale": 1.0,
            "dof_strength": 0.35,
        },
        "lighting": {
            "key_dir": [-0.48, 0.62, 0.62],
            "key_color": "#e0f2fe",
            "shadow_tint": "#1e293b",
            "ambient_color": "#0f172a",
            "rim_color": "#38bdf8",
            "rim_strength": 1.25,
            "sss_strength": 0.25,
            "ink_strength": 0.68,
            "bloom_intensity": 0.7,
        },
        "actors": [
            {
                "id": "speedster_hero",
                "label": "Wheel-Leg Speedster",
                "archetype": "humanoid",
                "placement": "center",
                "scale": 1.0,
                "facing_deg": 24.0,
                "posture": "skating_lean",
                "skin_color": "#0284c7",
                "skin_shadow_color": "#0c4a6e",
                "secondary_fur_color": "#0f172a",
                "left_hand_target": "skating_back",
                "right_hand_target": "forward_reach",
                "omit_bones": ["left_foot", "right_foot"],
                "proportions": {
                    "limb_length": 1.08,
                    "torso_width": 0.88,
                    "snout_length": 0.25,
                    "tail_length": 1.35,
                    "tail_curl": 0.35,
                },
                "expression": {
                    "eye_style": "sharp_mystic",
                    "iris_color": "#4ade80",
                    "lash_color": "#020617",
                },
                "garments": [
                    {
                        "name": "speed_suit",
                        "region": "upper_torso",
                        "fit_offset": 0.010,
                        "sleeve_length": 0.85,
                        "fabric_shader": "fabric_silk",
                        "roughness": 0.65,
                        "primary_color": "#111827",
                        "secondary_color": "#0284c7",
                    },
                    {
                        "name": "speed_leggings",
                        "region": "trousers",
                        "fit_offset": 0.010,
                        "fabric_shader": "fabric_silk",
                        "roughness": 0.65,
                        "primary_color": "#111827",
                        "secondary_color": "#0284c7",
                    }
                ],
                "mechanical_attachments": [
                    {
                        "name": "ankle_wheels",
                        "kind": "wheel_assembly",
                        "socket_bone": "both_ankles",
                        "scale": 1.15,
                        "primary_color": "#0f172a",
                        "secondary_color": "#38bdf8",
                        "emissive": 1.4,
                    },
                    {
                        "name": "aero_helmet",
                        "kind": "visor_helmet",
                        "socket_bone": "head",
                        "scale": 1.06,
                        "primary_color": "#020617",
                        "secondary_color": "#22d3ee",
                        "emissive": 1.25,
                    },
                ],
            }
        ],
        "zone1_props": [],
        "zone2_midground": {
            "kind": "city_skyline",
            "below_ledge_m": 0.0,
            "density": 16,
            "silhouette_color": "#0f172a",
            "window_glow_color": "#38bdf8",
            "haze_tint": "#0284c7",
            "haze_strength": 0.45,
        },
        "zone3_atmosphere": {
            "sky_style": "twilight_glow",
            "top_color": "#020617",
            "horizon_color": "#1e3a8a",
            "bottom_color": "#0ea5e9",
            "ground_enabled": True,
            "ground_color": "#1e293b",
            "ground_secondary_color": "#0f172a",
            "emitters": [{"kind": "moon_disc", "direction_or_pos": [-0.42, 0.34, -0.84], "color": "#7dd3fc", "intensity": 12.0, "radius": 0.11, "god_ray_strength": 0.65}],
            "bokeh_count": 26,
            "dust_motes_count": 28,
        },
    },
}


def build_offline_director_scene(prompt: str, *, style_mode: str = "anime_npr") -> SceneDNA:
    """
    Deterministic compositional offline fallback (zero API call) when LLM keys
    are absent or offline. Assembles characters, outfits, multi-limbs, wheel-legs,
    and 3-zone worlds from prompt semantics cleanly.
    """
    p = prompt.lower().strip()
    if any(k in p for k in ("xlr8", "wheel", "speedster", "roller")):
        base = copy.deepcopy(PRESET_SCENES["XLR8 Wheel-Leg Speedster (Alien Hybrid)"])
    elif any(k in p for k in ("singer", "singing", "microphone", "concert", "stage", "ponytail")):
        base = copy.deepcopy(PRESET_SCENES["Anime Stage Singer (Ponytail & Hoodie)"])
    elif any(k in p for k in ("four-arm", "four arm", "4 arm", "4-arm", "elf", "mystic")):
        base = copy.deepcopy(PRESET_SCENES["Four-Armed Elf Mystic (Multi-Limb + Cloak)"])
    elif any(k in p for k in ("cat", "kitten", "feline", "kitty")):
        key = "Cute Cat on High Sunset Wall (Stylized PBR)" if style_mode == "stylized_pbr" else "Cute Cat on High Sunset Wall (Anime)"
        base = copy.deepcopy(PRESET_SCENES[key])
    else:
        base = copy.deepcopy(PRESET_SCENES["Four-Armed Elf Mystic (Multi-Limb + Cloak)"])
        actor = base["actors"][0]
        actor["extra_limb_chains"] = []
        actor["mechanical_attachments"] = []
        actor["proportions"]["ear_length"] = 0.12
        actor["proportions"]["ear_tilt_deg"] = 10.0

        if "nurse" in p:
            actor["garments"] = [{
                "name": "nurse_dress", "region": "full_dress", "fit_offset": 0.018, "flare": 0.45,
                "neckline": "v_neck", "sleeve_length": 0.32, "hem_length": 0.95,
                "primary_color": "#f8fafc", "secondary_color": "#cbd5e1",
                "uv_decals": [{"kind": "medical_cross", "uv_t": 0.62, "uv_theta_deg": 18.0, "size": 0.12, "color": "#dc2626"}],
            }]
        elif "police" in p:
            actor["garments"] = [{
                "name": "police_uniform", "region": "upper_torso", "fit_offset": 0.016,
                "neckline": "mandarin", "sleeve_length": 0.85, "primary_color": "#1e3a8a", "secondary_color": "#0f172a",
                "uv_decals": [{"kind": "star_badge", "uv_t": 0.66, "uv_theta_deg": 24.0, "size": 0.10, "color": "#facc15"}],
            }, {"name": "police_trousers", "region": "trousers", "fit_offset": 0.014, "primary_color": "#0f172a"}]
        elif "firefighter" in p:
            actor["garments"] = [{
                "name": "fire_coat", "region": "outer_coat", "fit_offset": 0.036, "sleeve_length": 0.95,
                "primary_color": "#b45309", "secondary_color": "#1e293b",
                "uv_decals": [{"kind": "safety_stripes", "uv_t": 0.35, "uv_theta_deg": 0.0, "size": 0.18, "color": "#bef264"}],
            }]

        if "4 arm" in p or "four arm" in p:
            actor["extra_limb_chains"] = [{"id": "extra_arms", "kind": "arm_ik", "parent_socket": "mid_torso", "side": "pair"}]
        if "elf" in p:
            actor["proportions"]["ear_length"] = 0.44
            actor["proportions"]["ear_tilt_deg"] = 26.0
        if "dwarf" in p:
            actor["proportions"]["stature"] = 0.72
            actor["proportions"]["torso_width"] = 1.38
            actor["hair_or_mane"] = {"style": "braided_beard", "color": "#9a3412", "length": 0.55, "clump_count": 10}
        if "horn" in p or "succubus" in p or "demon" in p:
            actor["mechanical_attachments"].append({"name": "horns", "kind": "horn_pair", "socket_bone": "head", "primary_color": "#7f1d1d"})

    base["prompt"] = prompt
    base["style_mode"] = style_mode if style_mode in VALID_STYLE_MODES else "anime_npr"
    dna, _ = validate_and_repair_dna(scene_dna_from_dict(base))
    return dna


def plan_scene_dna(
    prompt: str,
    *,
    style_mode: str = "anime_npr",
    provider: Optional[str] = None,
    host: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_retries: int = 2,
    allow_fallback: bool = True,
    timeout: int = 90,
) -> DirectorPlanResult:
    """
    Plans a structured SceneDNA JSON via Groq/Ollama Scene Director, validates and
    repairs geometry constraints, and optionally falls back offline on API errors.
    """
    if not prompt.strip():
        dna = build_offline_director_scene("cute cat sitting on a high wall at sunset", style_mode=style_mode)
        return DirectorPlanResult(dna=dna, source="offline", warnings=["Empty prompt; loaded default showcase."])

    cfg = load_director_env()
    eff_prov = (provider or cfg["provider"]).strip().lower()
    if eff_prov not in ("groq", "ollama"):
        eff_prov = "groq"

    if eff_prov == "groq":
        eff_host = (host or cfg["groq_host"]).strip()
        eff_model = (model or cfg["groq_text_model"]).strip()
        eff_key = api_key if api_key is not None else cfg["groq_api_key"]
    else:
        eff_host = (host or cfg["ollama_host"]).strip()
        eff_model = (model or cfg["ollama_text_model"]).strip()
        eff_key = api_key if api_key is not None else cfg["ollama_api_key"]

    schema = get_director_json_schema()
    messages = [
        {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"STYLE MODE: {style_mode}\n"
                f"USER IMAGE PROMPT:\n{prompt.strip()}\n\n"
                "Return a complete SceneDNA JSON covering actors, creative anatomy/garments, "
                "Zone 1 hero props/walls, Zone 2 midground depth silhouettes, and Zone 3 HDR atmosphere."
            ),
        },
    ]

    warnings: List[str] = []
    last_dna: Optional[SceneDNA] = None

    for attempt in range(1, max_retries + 2):
        try:
            raw_text = call_text_llm(
                messages,
                provider=eff_prov,
                host=eff_host,
                model=eff_model,
                api_key=eff_key,
                schema=schema,
                schema_name="scene_dna",
                timeout=timeout,
            )
            parsed = extract_json_object(raw_text)
            parsed["prompt"] = prompt.strip()
            parsed["style_mode"] = style_mode if style_mode in VALID_STYLE_MODES else parsed.get("style_mode", "anime_npr")
            dna_raw = scene_dna_from_dict(parsed)
            dna_fixed, notes = validate_and_repair_dna(dna_raw)
            last_dna = dna_fixed
            warnings.extend(notes)

            if not dna_fixed.actors and not dna_fixed.zone1_props and attempt <= max_retries:
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({"role": "user", "content": "Output had empty actors and zone1_props. Populate both and return JSON."})
                continue

            return DirectorPlanResult(dna=dna_fixed, source=eff_prov, model_used=eff_model, attempts=attempt, warnings=warnings)
        except DirectorAPIError as exc:
            if allow_fallback:
                warnings.append(f"LLM API unavailable ({exc}); synthesized via compositional offline director.")
                return DirectorPlanResult(
                    dna=build_offline_director_scene(prompt, style_mode=style_mode),
                    source="offline",
                    model_used=eff_model,
                    attempts=attempt,
                    warnings=warnings,
                )
            raise
        except Exception as exc:
            warnings.append(f"Director parse attempt {attempt} failed: {exc}")
            if attempt > max_retries:
                if last_dna is not None:
                    return DirectorPlanResult(dna=last_dna, source=eff_prov, model_used=eff_model, attempts=attempt, warnings=warnings)
                if allow_fallback:
                    return DirectorPlanResult(
                        dna=build_offline_director_scene(prompt, style_mode=style_mode),
                        source="offline",
                        model_used=eff_model,
                        attempts=attempt,
                        warnings=warnings,
                    )
                raise
            messages.append({"role": "user", "content": f"JSON parse error ({exc}). Return ONLY valid SceneDNA JSON."})

    return DirectorPlanResult(dna=build_offline_director_scene(prompt, style_mode=style_mode), source="offline", warnings=warnings)
