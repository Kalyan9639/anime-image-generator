"""
Streamlit Web Application for ASCII to Image Generator (ascii2image).

Provides two workspaces:
1. Classic ASCII → Diagram & Art (pure math/vector pipeline, zero AI).
2. Scene Studio (Track 1: Groq `openai/gpt-oss-20b` or Ollama as a text-only
   knowledge/scene planner + pre-written dynamic multi-layer 3D/2D math engine
   & optical bokeh/DOF compositor).
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
from pathlib import Path
from typing import Any, Callable, Dict

import cv2
from PIL import Image
import streamlit as st

from ascii2image import (
    classify_ascii,
    diagram_to_svg,
    get_available_engines,
    reconstruct_art_photo,
    render_art_crisp_png,
    render_art_crisp_svg,
    sanitize_ascii_text,
    svg_to_png,
    vectorize_line_art_svg,
)
from ascii2image.semantic import (
    PlannerAPIError,
    build_offline_fallback_scene,
    load_env_config,
    plan_scene,
    render_scene_graph,
    scene_from_dict,
)

ROOT_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT_DIR / "samples"
FALLBACK_SAMPLE = "+-------+     +-------+\n| Input | --> | Image |\n+-------+     +-------+"

st.set_page_config(
    page_title="ASCII → Image Generator & Scene Studio",
    page_icon="▦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Monospace styling with white-space: pre and horizontal scrolling so wide ASCII
# lines NEVER soft-wrap or become visually disoriented in the editor.
st.markdown(
    """
    <style>
    textarea,
    [data-testid="stTextArea"] textarea {
        font-family: "Consolas", "DejaVu Sans Mono", "Courier New", monospace !important;
        font-size: 13px !important;
        line-height: 1.35 !important;
        white-space: pre !important;
        overflow-wrap: normal !important;
        word-wrap: normal !important;
        overflow-x: auto !important;
        tab-size: 4 !important;
    }
    .det-card {
        padding: 0.85rem 1rem;
        border-radius: 0.6rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        background-color: rgba(56, 189, 248, 0.06);
        margin-bottom: 0.75rem;
    }
    .det-badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: #ffffff;
        margin-right: 0.5rem;
    }
    .badge-diagram { background-color: #0284c7; }
    .badge-art { background-color: #7c3aed; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _stretch_kwargs(widget_fn: Callable[..., Any]) -> Dict[str, Any]:
    """
    Uses `width="stretch"` on Streamlit >= 1.49 (avoiding deprecation warnings)
    and falls back to `use_container_width=True` on older Streamlit 1.32–1.48.
    """
    try:
        params = inspect.signature(widget_fn).parameters
        if "width" in params:
            ver_parts = tuple(int(p) for p in st.__version__.split(".")[:2] if p.isdigit())
            if ver_parts >= (1, 49):
                return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


@st.cache_data
def load_samples() -> Dict[str, str]:
    """Load built-in .txt sample files from the samples/ directory."""
    samples: Dict[str, str] = {}
    if SAMPLES_DIR.is_dir():
        for file in sorted(SAMPLES_DIR.glob("*.txt")):
            try:
                label = file.stem.replace("_", " ").title()
                samples[label] = file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
    return samples


SCENE_PRESETS: Dict[str, Dict[str, Any]] = {
    "Winter Bokeh Ledge (Multi-Object)": {
        "title": "Cozy Winter Night Ledge",
        "seed": 17,
        "camera": {
            "projection": "perspective",
            "azimuth_deg": 24,
            "elevation_deg": 18,
            "roll_deg": -3,
            "fov_deg": 40,
            "distance_scale": 0.98,
            "dof_strength": 0.30,
        },
        "lighting": {
            "key_dir": [0.7, 0.85, 0.55],
            "key_color": "#ffedd5",
            "ambient_color": "#312e47",
            "rim_color": "#fde68a",
            "rim_strength": 0.70,
            "bloom_intensity": 0.55,
            "cel_shading": False,
            "exposure": 1.08,
        },
        "background": {
            "style": "bokeh_night",
            "top_color": "#1a1426",
            "bottom_color": "#2b2030",
            "glow_color": "#f59e0b",
            "bokeh_count": 110,
            "bokeh_colors": ["#fef3c7", "#fde68a", "#fbbf24", "#fff7ed", "#f59e0b"],
            "bokeh_size": 1.15,
            "particle_type": "snowflakes",
            "particle_count": 24,
            "particle_color": "#fffbeb",
        },
        "environment": {
            "enabled": True,
            "kind": "tiled_ledge",
            "y_level": 0.0,
            "tilt_deg": 0.0,
            "pattern": "grid_tiles",
            "primary_color": "#645a68",
            "secondary_color": "#2e2833",
            "pattern_scale": 1.6,
            "roughness": 0.72,
        },
        "entities": [
            {
                "id": "winter_lantern",
                "label": "Glowing Winter Lantern",
                "role": "subject",
                "placement": "center",
                "scale": 1.15,
                "parts": [
                    {"name": "base", "primitive": "cylinder", "size": [0.95, 0.14, 0.95], "attachment": "base_center", "material": {"color": "#3f2e21", "pattern": "brushed_metal", "metallic": 0.7, "roughness": 0.35}},
                    {"name": "glow_core", "primitive": "lathe_profile", "size": [0.64, 0.95, 0.64], "attachment": "on_top_of", "target": "base", "params": {"profile_radii": [0.75, 1.0, 0.96, 0.72]}, "material": {"color": "#fef08a", "pattern": "solid", "emissive": 1.7, "roughness": 0.2}},
                    {"name": "pillars", "primitive": "cylinder", "size": [0.09, 0.98, 0.09], "attachment": "around_perimeter", "target": "glow_core", "modifiers": [{"kind": "array_radial", "params": {"count": 4, "radius": 0.37}}], "material": {"color": "#292018", "pattern": "brushed_metal", "metallic": 0.8, "roughness": 0.3}},
                    {"name": "roof_dome", "primitive": "cone", "size": [1.05, 0.38, 1.05], "attachment": "on_top_of", "target": "glow_core", "params": {"bottom_ratio": 1.0, "top_ratio": 0.18}, "material": {"color": "#3f2e21", "pattern": "brushed_metal", "metallic": 0.75, "roughness": 0.3}},
                    {"name": "top_ring", "primitive": "torus", "size": [0.30, 0.09, 0.30], "attachment": "on_top_of", "target": "roof_dome", "rotation_deg": [90, 0, 0], "material": {"color": "#d97706", "pattern": "solid", "metallic": 0.9, "roughness": 0.2}},
                ],
            },
            {
                "id": "knit_mug",
                "label": "Cable-Knit Ceramic Mug",
                "role": "object",
                "placement": "foreground_left",
                "offset": [0.3, 0.0, -0.4],
                "scale": 0.85,
                "parts": [
                    {"name": "cup_body", "primitive": "lathe_profile", "size": [0.85, 0.92, 0.85], "attachment": "base_center", "params": {"profile_radii": [0.78, 0.96, 1.0, 0.95], "hollow": True, "wall_thickness": 0.12}, "material": {"color": "#f8fafc", "secondary_color": "#cbd5e1", "pattern": "cable_knit", "pattern_scale": 3.5, "roughness": 0.82}},
                    {"name": "handle", "primitive": "bezier_tube", "size": [0.45, 0.52, 0.25], "attachment": "right_of", "target": "cup_body", "offset": [-0.12, 0.0, 0.0], "params": {"control_points": [[-0.6, 0.7, 0.0], [1.0, 0.35, 0.0], [1.0, -0.35, 0.0], [-0.6, -0.7, 0.0]], "radius_start": 0.075, "radius_end": 0.075}, "material": {"color": "#f1f5f9", "pattern": "solid", "roughness": 0.35}},
                ],
            },
            {
                "id": "tartan_gift_bundle",
                "label": "Tartan Plaid Gift Box",
                "role": "object",
                "placement": "right",
                "offset": [-0.35, 0.0, 0.15],
                "scale": 0.95,
                "rotation_deg": [0, -22, 0],
                "parts": [
                    {"name": "plaid_wrap", "primitive": "superquadric", "size": [1.15, 0.72, 0.95], "attachment": "base_center", "params": {"e1": 0.32, "e2": 0.32}, "material": {"color": "#b91c1c", "secondary_color": "#172554", "accent_color": "#facc15", "pattern": "plaid_tartan", "pattern_scale": 2.4, "roughness": 0.85}},
                    {"name": "star_bow", "primitive": "gielis_supershape", "size": [0.55, 0.24, 0.55], "attachment": "on_top_of", "target": "plaid_wrap", "params": {"m": 6.0, "n1": 0.55, "n2": 1.6, "n3": 1.6}, "material": {"color": "#f59e0b", "pattern": "solid", "metallic": 0.65, "roughness": 0.25}},
                ],
            },
            {
                "id": "mini_pine",
                "label": "Festive Pine Tree",
                "role": "prop",
                "placement": "background_left",
                "offset": [0.55, 0.0, 0.75],
                "scale": 1.1,
                "parts": [
                    {"name": "trunk", "primitive": "cylinder", "size": [0.22, 0.42, 0.22], "attachment": "base_center", "material": {"color": "#5c3a21", "secondary_color": "#3b2313", "pattern": "wood_grain", "roughness": 0.85}},
                    {"name": "tier1", "primitive": "cone", "size": [1.15, 0.65, 1.15], "attachment": "on_top_of", "target": "trunk", "offset": [0.0, -0.1, 0.0], "material": {"color": "#166534", "secondary_color": "#14532d", "pattern": "fbm_noise", "roughness": 0.8}},
                    {"name": "tier2", "primitive": "cone", "size": [0.85, 0.58, 0.85], "attachment": "on_top_of", "target": "tier1", "offset": [0.0, -0.22, 0.0], "material": {"color": "#15803d", "secondary_color": "#14532d", "pattern": "fbm_noise", "roughness": 0.8}},
                    {"name": "top_star", "primitive": "gielis_supershape", "size": [0.30, 0.30, 0.16], "attachment": "on_top_of", "target": "tier2", "params": {"m": 5.0, "n1": 0.38, "n2": 1.2, "n3": 1.2}, "material": {"color": "#fde047", "pattern": "solid", "emissive": 1.6, "metallic": 0.5}},
                ],
            },
        ],
    },
    "Microprocessor & Copper Heatsink": {
        "title": "Microprocessor Die & BGA Package",
        "seed": 7,
        "camera": {"projection": "isometric", "azimuth_deg": 38, "elevation_deg": 30, "fov_deg": 38, "distance_scale": 1.0, "dof_strength": 0.10},
        "lighting": {"key_dir": [0.6, 0.9, 0.5], "key_color": "#ffffff", "ambient_color": "#334155", "rim_color": "#38bdf8", "rim_strength": 0.55, "bloom_intensity": 0.35},
        "background": {"style": "studio", "top_color": "#0b1120", "bottom_color": "#1e293b", "glow_color": "#0284c7", "particle_type": "none"},
        "environment": {"enabled": True, "kind": "ground_plane", "y_level": 0.0, "pattern": "circuit_traces", "primary_color": "#0f2e24", "secondary_color": "#081c16", "pattern_scale": 1.8, "roughness": 0.5, "metallic": 0.25},
        "entities": [
            {
                "id": "cpu_package",
                "label": "Microprocessor Package",
                "role": "subject",
                "placement": "center",
                "scale": 1.25,
                "parts": [
                    {"name": "bga_balls", "primitive": "sphere", "size": [0.11, 0.11, 0.11], "attachment": "base_center", "modifiers": [{"kind": "array_grid", "params": {"nx": 9, "nz": 9, "spacing_x": 0.21, "spacing_z": 0.21}}], "material": {"color": "#e2e8f0", "pattern": "solid", "metallic": 0.92, "roughness": 0.18}},
                    {"name": "substrate", "primitive": "rounded_box", "size": [2.25, 0.18, 2.25], "attachment": "on_top_of", "target": "bga_balls", "params": {"bevel": 0.03}, "material": {"color": "#065f46", "secondary_color": "#064e3b", "accent_color": "#f59e0b", "pattern": "circuit_traces", "pattern_scale": 3.5, "metallic": 0.25, "roughness": 0.45}},
                    {"name": "silicon_die", "primitive": "superquadric", "size": [1.35, 0.14, 1.35], "attachment": "on_top_of", "target": "substrate", "params": {"e1": 0.18, "e2": 0.18}, "material": {"color": "#334155", "secondary_color": "#0f172a", "accent_color": "#38bdf8", "pattern": "brushed_metal", "metallic": 0.85, "roughness": 0.2}},
                    {"name": "cpu_cores", "primitive": "rounded_box", "size": [0.36, 0.12, 0.36], "attachment": "on_top_of", "target": "silicon_die", "modifiers": [{"kind": "array_grid", "params": {"nx": 2, "nz": 2, "spacing_x": 0.56, "spacing_z": 0.56}}], "material": {"color": "#38bdf8", "pattern": "solid", "emissive": 0.75, "metallic": 0.6, "roughness": 0.2}},
                    {"name": "smd_caps", "primitive": "rounded_box", "size": [0.10, 0.09, 0.18], "attachment": "on_top_of", "target": "substrate", "offset": [0.0, 0.0, 0.88], "modifiers": [{"kind": "array_linear", "params": {"axis": "x", "count": 6, "spacing": 0.28}}], "material": {"color": "#d97706", "pattern": "solid", "metallic": 0.7, "roughness": 0.3}},
                ],
            },
            {
                "id": "heatsink_fin_block",
                "label": "Copper Heatsink",
                "role": "object",
                "placement": "right",
                "offset": [0.45, 0.0, -0.15],
                "scale": 0.9,
                "parts": [
                    {"name": "hs_base", "primitive": "rounded_box", "size": [1.3, 0.18, 1.3], "attachment": "base_center", "material": {"color": "#b45309", "pattern": "brushed_metal", "metallic": 0.9, "roughness": 0.25}},
                    {"name": "fins", "primitive": "rounded_box", "size": [0.07, 0.75, 1.22], "attachment": "on_top_of", "target": "hs_base", "modifiers": [{"kind": "array_linear", "params": {"axis": "x", "count": 7, "spacing": 0.18}}], "material": {"color": "#d97706", "pattern": "brushed_metal", "metallic": 0.9, "roughness": 0.22}},
                ],
            },
        ],
    },
}


def resolve_effective_mode(mode: str, category: str, subtype: str) -> str:
    if mode != "auto":
        return mode
    if category == "diagram":
        return "diagram"
    if subtype == "art:tone":
        return "art:reconstruct"
    return "art:crisp"


def resolve_effective_theme(theme: str, effective_mode: str) -> str:
    if theme != "auto":
        return theme
    return "light" if effective_mode == "diagram" else "dark"


def _on_sample_change(samples: Dict[str, str]) -> None:
    chosen = st.session_state.get("sample_selector", "Custom Input")
    if chosen == "Custom Input":
        st.session_state["ascii_text"] = ""
    elif chosen in samples:
        st.session_state["ascii_text"] = samples[chosen]


def _on_text_change(samples: Dict[str, str]) -> None:
    current = st.session_state.get("ascii_text", "")
    chosen = st.session_state.get("sample_selector", "Custom Input")
    if chosen != "Custom Input" and samples.get(chosen) != current:
        st.session_state["sample_selector"] = "Custom Input"


def _on_clear() -> None:
    st.session_state["ascii_text"] = ""
    st.session_state["sample_selector"] = "Custom Input"
    st.session_state["last_upload_hash"] = None


def _on_fix_alignment() -> None:
    current = st.session_state.get("ascii_text", "")
    cleaned, _ = sanitize_ascii_text(current, repair_alignment=True, strip_blank_edges=True)
    st.session_state["ascii_text"] = cleaned


def _on_file_upload() -> None:
    uploaded = st.session_state.get("ascii_uploader")
    if uploaded is None:
        st.session_state["last_upload_hash"] = None
        return
    raw_bytes = uploaded.getvalue()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    if st.session_state.get("last_upload_hash") != content_hash:
        st.session_state["ascii_text"] = raw_bytes.decode("utf-8", errors="replace")
        st.session_state["sample_selector"] = "Custom Input"
        st.session_state["last_upload_hash"] = content_hash


def _render_scene_studio_tab() -> None:
    """Renders Track 1 Scene Studio: Prompt / Semantic ASCII -> LLM Knowledge (Groq/Ollama) -> Dynamic Math."""
    env_cfg = load_env_config()
    if "studio_scene_json" not in st.session_state:
        st.session_state["studio_scene_json"] = json.dumps(
            SCENE_PRESETS["Winter Bokeh Ledge (Multi-Object)"], indent=2
        )
    if "studio_source_label" not in st.session_state:
        st.session_state["studio_source_label"] = "preset"
    if "studio_warnings" not in st.session_state:
        st.session_state["studio_warnings"] = []

    st.caption(
        "Track 1: **Groq (`openai/gpt-oss-20b`) / Ollama** acts strictly as a text knowledge & scene planner "
        "(never writing Python/SVG code), while our **pre-written dynamic math engine** (SDF, Superquadrics, "
        "Gielis Superformula, Convolution Limbs, Lathe/Bézier, CSG, Bokeh/DOF Compositor) synthesizes the image."
    )

    s_left, s_right = st.columns([1, 1], gap="large")

    with s_left:
        st.subheader("1. Describe Scene or Load Preset")
        preset_names = ["— Select Math Demo Preset —"] + list(SCENE_PRESETS.keys())
        chosen_preset = st.selectbox("Instant Math Demo Presets (Zero API Cost)", preset_names, key="studio_preset_sel")
        if chosen_preset != preset_names[0] and st.session_state.get("studio_last_preset") != chosen_preset:
            st.session_state["studio_scene_json"] = json.dumps(SCENE_PRESETS[chosen_preset], indent=2)
            st.session_state["studio_source_label"] = f"preset ({chosen_preset})"
            st.session_state["studio_warnings"] = []
            st.session_state["studio_last_preset"] = chosen_preset

        prompt_text = st.text_area(
            "Image Prompt (Background + Subject + Multiple Objects)",
            value="A cozy winter tiled stone ledge at night with warm golden bokeh lights, snowflakes, a glowing brass lantern, a cable-knit ceramic mug, a tartan plaid gift box, and a small pine tree.",
            height=110,
            key="studio_prompt",
        )

        with st.expander("Optional: Semantic ASCII Spatial Layout Lock", expanded=False):
            studio_ascii = st.text_area(
                "Semantic ASCII boxes (optional)",
                value="",
                height=140,
                key="studio_ascii",
                placeholder="+------------------------------------+\n| microprocessor   view=iso          |\n|  +---------+  +-------+  +------+  |\n|  | core x4 |  | cache |  | gpu  |  |\n|  +---------+  +-------+  +------+  |\n|  pins=bga 10x10                    |\n+------------------------------------+",
            )

        with st.expander("LLM Provider & Model Settings (Groq / Ollama)", expanded=False):
            provider_options = ["groq", "ollama"]
            def_idx = 0 if env_cfg.get("provider", "groq") == "groq" else 1
            llm_provider = st.radio(
                "LLM Provider",
                options=provider_options,
                index=def_idx,
                format_func=lambda p: "Groq Cloud (Fast / Free Tier)" if p == "groq" else "Ollama (Cloud / Local)",
                horizontal=True,
                key="studio_llm_provider",
            )
            if llm_provider == "groq":
                llm_host = st.text_input("Groq API Base URL", value=env_cfg["groq_host"], key="studio_groq_host")
                llm_model = st.text_input("Groq Model", value=env_cfg["groq_model"], key="studio_groq_model")
                llm_key = st.text_input("GROQ_API_KEY", value=env_cfg["groq_api_key"], type="password", key="studio_groq_key")
            else:
                llm_host = st.text_input("Ollama Host", value=env_cfg["ollama_host"], key="studio_ollama_host")
                llm_model = st.text_input("Ollama Model", value=env_cfg["ollama_model"], key="studio_ollama_model")
                llm_key = st.text_input("OLLAMA_API_KEY", value=env_cfg["ollama_api_key"], type="password", key="studio_ollama_key")

            allow_fb = st.checkbox(
                "Fallback to offline math if LLM returns network/API error",
                value=True,
                key="studio_allow_fallback",
            )

        res_map = {"480×360 (Fast)": (480, 360), "640×480 (Balanced)": (640, 480), "800×600 (HD)": (800, 600)}
        res_choice = st.select_slider("Render Resolution", options=list(res_map.keys()), value="640×480 (Balanced)", key="studio_res")
        rw, rh = res_map[res_choice]

        b1, b2 = st.columns(2)
        with b1:
            plan_clicked = st.button(
                f"✨ Plan & Render via {llm_provider.title()}",
                type="primary",
                key="btn_studio_plan",
                **_stretch_kwargs(st.button),
            )
        with b2:
            offline_clicked = st.button(
                "🧮 Render Scene JSON (No LLM)",
                key="btn_studio_render",
                **_stretch_kwargs(st.button),
            )

        if plan_clicked:
            try:
                with st.spinner(f"Querying {llm_provider.title()} ({llm_model}) for multi-layer Scene Graph..."):
                    plan_res = plan_scene(
                        prompt_text,
                        ascii_text=studio_ascii,
                        provider=llm_provider,
                        host=llm_host,
                        model=llm_model,
                        api_key=llm_key,
                        allow_fallback=allow_fb,
                    )
                st.session_state["studio_scene_json"] = json.dumps(plan_res.scene.to_dict(), indent=2)
                st.session_state["studio_source_label"] = f"{plan_res.source} ({plan_res.model_used}, attempts={plan_res.attempts})"
                st.session_state["studio_warnings"] = plan_res.warnings
            except PlannerAPIError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Scene planning failed: {exc}")
        elif offline_clicked and studio_ascii.strip() and not st.session_state.get("studio_scene_json"):
            fb = build_offline_fallback_scene(prompt_text, studio_ascii)
            st.session_state["studio_scene_json"] = json.dumps(fb.to_dict(), indent=2)
            st.session_state["studio_source_label"] = "offline-ascii"

        with st.expander("Camera, Lighting & Optical Post-FX Overrides (Instant)", expanded=False):
            c_az = st.slider("Camera Azimuth (°)", -180, 180, 24, key="studio_az")
            c_el = st.slider("Camera Elevation (°)", -5, 75, 18, key="studio_el")
            c_dof = st.slider("Depth-of-Field Blur", 0.0, 1.0, 0.25, 0.05, key="studio_dof")
            c_bloom = st.slider("Optical Bloom", 0.0, 1.2, 0.45, 0.05, key="studio_bloom")
            c_cel = st.checkbox("Cel / Anime Shading Quantization", value=False, key="studio_cel")
            override_cam = st.checkbox("Apply above camera/FX overrides", value=False, key="studio_override_cam")

        with st.expander("Editable Scene Graph JSON (Live Math Input)", expanded=False):
            st.text_area("Scene Graph JSON", key="studio_scene_json", height=240, label_visibility="collapsed")

    with s_right:
        st.subheader("2. Synthesized Multi-Layer Output")
        for warn in st.session_state.get("studio_warnings", []):
            st.warning(warn)

        try:
            raw_dict = json.loads(st.session_state["studio_scene_json"])
            scene_obj = scene_from_dict(raw_dict)
            if override_cam:
                scene_obj.camera.azimuth_deg = float(c_az)
                scene_obj.camera.elevation_deg = float(c_el)
                scene_obj.camera.dof_strength = float(c_dof)
                scene_obj.lighting.bloom_intensity = float(c_bloom)
                scene_obj.lighting.cel_shading = bool(c_cel)

            with st.spinner(f"Evaluating dynamic math & optics ({rw}×{rh})..."):
                out = render_scene_graph(scene_obj, width=rw, height=rh, include_svg=True)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇ Download 3D PNG",
                    data=out.png_bytes,
                    file_name="scene_studio.png",
                    mime="image/png",
                    type="primary",
                    key="dl_studio_png",
                    **_stretch_kwargs(st.download_button),
                )
            with dl2:
                st.download_button(
                    "⬇ Download 2D Vector SVG",
                    data=out.svg_content,
                    file_name="scene_studio.svg",
                    mime="image/svg+xml",
                    key="dl_studio_svg",
                    **_stretch_kwargs(st.download_button),
                )

            t3d, t2d = st.tabs(["🖼️ 3D Optical Render", "📐 2D Vector SVG"])
            with t3d:
                st.image(out.png_bytes, **_stretch_kwargs(st.image))
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Dimensions", f"{rw}×{rh}")
                m2.metric("Render Time", f"{out.render_seconds:.2f}s")
                m3.metric("Entities", str(len(out.resolved.entities)))
                m4.metric("Source", st.session_state.get("studio_source_label", "preset").split("(")[0].strip())
            with t2d:
                st.code(out.svg_content, language="xml")
        except Exception as exc:
            st.error(f"Scene Graph render error: {exc}")


def main() -> None:
    samples = load_samples()
    default_sample_name = "Flowchart" if "Flowchart" in samples else (next(iter(samples), None))
    default_text = samples.get(default_sample_name, FALLBACK_SAMPLE)
    sample_options = ["Custom Input"] + list(samples.keys())

    if "sample_selector" not in st.session_state:
        st.session_state["sample_selector"] = (
            default_sample_name if default_sample_name in sample_options else "Custom Input"
        )
    if "ascii_text" not in st.session_state:
        st.session_state["ascii_text"] = default_text
    if "last_upload_hash" not in st.session_state:
        st.session_state["last_upload_hash"] = None

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("▦ ASCII → Image Generator & Scene Studio")
    st.caption(
        "Pure mathematical signal-processing & vector graphics pipeline • "
        "Plus Track 1 Dynamic Multi-Layer Scene Studio (Groq / Ollama Knowledge + Pre-Written Math)"
    )

    # ── Sidebar Controls (Classic ASCII Converter) ───────────────────────────
    with st.sidebar:
        st.header("⚙️ ASCII Converter Settings")

        st.selectbox(
            "Load Built-in Sample",
            options=sample_options,
            key="sample_selector",
            on_change=_on_sample_change,
            args=(samples,),
            help="Select a built-in example, or choose 'Custom Input' for a blank canvas.",
        )

        st.file_uploader(
            "Or upload a .txt file",
            type=["txt", "bob", "asc"],
            key="ascii_uploader",
            on_change=_on_file_upload,
        )

        st.divider()

        mode_labels = {
            "auto": "Auto-Detect (Recommended)",
            "diagram": "Diagram (Vector Paths)",
            "art:reconstruct": "Art: Photo Reconstruction (Bilateral + CLAHE)",
            "art:crisp": "Art: Crisp Monospace Typography",
            "art:vector": "Art: Line Vector Contours (Bézier)",
        }
        mode = st.selectbox(
            "Conversion Mode",
            options=list(mode_labels.keys()),
            format_func=lambda k: mode_labels[k],
            index=0,
        )

        theme_labels = {
            "auto": "Auto (Light for Diagram, Dark for Art)",
            "light": "Light",
            "dark": "Dark Slate",
            "blueprint": "Blueprint CAD",
            "tinted": "Tinted Pastel",
            "terminal_green": "Retro CRT Green",
            "amber": "Retro Amber",
        }
        theme = st.selectbox(
            "Color Theme",
            options=list(theme_labels.keys()),
            format_func=lambda k: theme_labels[k],
            index=0,
        )

        scale = st.select_slider(
            "PNG Resolution Scale (DPI)",
            options=[1.0, 1.5, 2.0, 3.0],
            value=2.0,
            format_func=lambda v: f"{v:.1f}x",
        )

        available_engines = get_available_engines()
        engine = st.selectbox("Diagram Engine", options=["auto"] + available_engines, index=0)

        auto_repair = st.checkbox(
            "Auto-repair pasted alignment",
            value=True,
            help=(
                "Automatically cleans Unicode spaces/tabs, unwraps Markdown ``` code fences, "
                "and repairs missing Line-1 indentation from drag-select copy-paste."
            ),
        )

        with st.expander("Photo Reconstruction Tuning", expanded=False):
            target_width = st.slider("Target Width (px)", min_value=400, max_value=2000, value=1200, step=100)
            denoise = st.slider("Denoise Smoothing", min_value=0.2, max_value=2.5, value=1.0, step=0.1)
            clahe_clip = st.slider("CLAHE Contrast Limit", min_value=1.0, max_value=5.0, value=2.0, step=0.2)
            invert = st.selectbox("Invert Luminance", options=["auto", "yes", "no"], index=0)

    # ── Top Workspace Tabs ───────────────────────────────────────────────────
    tab_classic, tab_studio = st.tabs(["▦ ASCII → Diagram & Art", "🎨 Scene Studio (LLM + Dynamic Math)"])

    with tab_classic:
        col_in, col_out = st.columns([1, 1], gap="large")

        with col_in:
            st.subheader("1. Input ASCII Text")
            ascii_text = st.text_area(
                "Paste your ASCII diagram or ASCII art below:",
                key="ascii_text",
                height=360,
                label_visibility="collapsed",
                placeholder="Paste your ASCII diagram or ASCII art here, then click 'Generate Image' (or press Ctrl+Enter)...",
                on_change=_on_text_change,
                args=(samples,),
            )

            act_col1, act_col2, act_col3 = st.columns([2, 1, 1])
            with act_col1:
                st.button(
                    "⚡ Generate Image",
                    type="primary",
                    **_stretch_kwargs(st.button),
                    help="Apply any newly pasted ASCII text and render the image.",
                )
            with act_col2:
                st.button(
                    "🔧 Fix Align",
                    on_click=_on_fix_alignment,
                    **_stretch_kwargs(st.button),
                    help="Normalize tabs/spaces, unwrap code fences, and fix misaligned Line-1 indent in the editor.",
                )
            with act_col3:
                st.button(
                    "🧹 Clear",
                    on_click=_on_clear,
                    **_stretch_kwargs(st.button),
                    help="Clear the input box to paste fresh ASCII.",
                )

            if not ascii_text.strip():
                st.info("Paste ASCII text above (and click **Generate Image** or press **Ctrl+Enter**), or load a sample from the sidebar.")
                with col_out:
                    st.subheader("2. Converted Output")
                    st.info("Your generated SVG & PNG preview will appear here once you paste ASCII text or select a sample.")
                effective_text = ""
            else:
                effective_text, repair_notes = sanitize_ascii_text(
                    ascii_text,
                    repair_alignment=auto_repair,
                    strip_blank_edges=True,
                )
                if not effective_text.strip():
                    st.info("Paste ASCII text above to generate an image.")
                    with col_out:
                        st.subheader("2. Converted Output")
                        st.info("Your generated SVG & PNG preview will appear here.")
                else:
                    if repair_notes:
                        st.info("🔧 " + " ".join(repair_notes))

                    detection = classify_ascii(effective_text)
                    badge_cls = "badge-diagram" if detection.category == "diagram" else "badge-art"
                    st.markdown(
                        f"""
                        <div class="det-card">
                            <div>
                                <span class="det-badge {badge_cls}">{detection.category.upper()} · {detection.subtype}</span>
                                <strong>{detection.confidence * 100:.1f}% Confidence</strong>
                            </div>
                            <div style="margin-top: 0.4rem; font-size: 0.88rem; opacity: 0.85;">
                                {detection.explanation}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.expander("6-Feature Classifier Diagnostics"):
                        if detection.scores:
                            score_cols = st.columns(3)
                            items = list(detection.scores.items())
                            for idx, (name, val) in enumerate(items):
                                score_cols[idx % 3].metric(name.replace("_", " ").title(), f"{val:.3f}")

        if effective_text.strip():
            effective_mode = resolve_effective_mode(mode, detection.category, detection.subtype)
            effective_theme = resolve_effective_theme(theme, effective_mode)
            diagram_theme = effective_theme if effective_theme in ("light", "dark", "blueprint", "tinted") else "light"
            art_theme = "light" if effective_theme == "tinted" else effective_theme

            svg_content: str | None = None
            png_bytes: bytes | None = None

            with col_out:
                st.subheader("2. Converted Output")
                try:
                    with st.spinner("Rendering..."):
                        if effective_mode == "diagram":
                            svg_content = diagram_to_svg(
                                effective_text,
                                engine=engine,
                                theme=diagram_theme,
                                repair_alignment=auto_repair,
                            )
                            png_bytes = svg_to_png(svg_content, scale=scale)
                        elif effective_mode == "art:reconstruct":
                            mat = reconstruct_art_photo(
                                effective_text,
                                target_width=target_width,
                                invert=invert,
                                denoise_strength=denoise,
                                clahe_clip=clahe_clip,
                            )
                            ok, buf = cv2.imencode(".png", mat)
                            if ok:
                                png_bytes = buf.tobytes()
                            svg_content = render_art_crisp_svg(
                                effective_text,
                                theme=art_theme,
                                repair_alignment=auto_repair,
                            )
                        elif effective_mode == "art:vector":
                            svg_content = vectorize_line_art_svg(effective_text, theme=art_theme)
                            png_bytes = svg_to_png(svg_content, scale=scale)
                        else:  # art:crisp
                            svg_content = render_art_crisp_svg(
                                effective_text,
                                theme=art_theme,
                                repair_alignment=auto_repair,
                            )
                            png_bytes = render_art_crisp_png(
                                effective_text,
                                theme=art_theme,
                                scale=scale,
                                repair_alignment=auto_repair,
                            )

                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        if png_bytes:
                            st.download_button(
                                label="⬇ Download PNG",
                                data=png_bytes,
                                file_name="ascii_output.png",
                                mime="image/png",
                                type="primary",
                                **_stretch_kwargs(st.download_button),
                            )
                    with dl_col2:
                        if svg_content:
                            st.download_button(
                                label="⬇ Download SVG",
                                data=svg_content,
                                file_name="ascii_output.svg",
                                mime="image/svg+xml",
                                **_stretch_kwargs(st.download_button),
                            )

                    tab_img, tab_svg = st.tabs(["🖼️ Image Preview", "📐 SVG Source"])

                    with tab_img:
                        if png_bytes:
                            st.image(png_bytes, **_stretch_kwargs(st.image))
                            pil_img = Image.open(io.BytesIO(png_bytes))
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Dimensions", f"{pil_img.width} × {pil_img.height} px")
                            m2.metric("PNG Size", f"{len(png_bytes) / 1024:.1f} KB")
                            m3.metric("Render Mode", f"{effective_mode} ({effective_theme})")
                        else:
                            st.error("Image rendering returned empty output. Try switching Conversion Mode or Diagram Engine.")

                    with tab_svg:
                        if svg_content:
                            st.code(svg_content, language="xml")
                        else:
                            st.info("No SVG output for this mode.")

                except Exception as exc:
                    st.error(f"Conversion failed: {exc}")

    with tab_studio:
        _render_scene_studio_tab()


if __name__ == "__main__":
    main()
