"""
PrismStudio Streamlit Web Application.

Standalone natural-language-to-image pure-math studio featuring:
- Open-ended Compositional Scene Director (Groq / Ollama + Offline Director)
- Dual Aesthetic Modes (`anime_npr` vs `stylized_pbr`)
- Optional 320x240 VLM Draft Self-Critique loop with Draft-vs-Final preview
- Live Zero-LLM Camera / Aperture / Lighting / Cel / Exposure controls
- ModernGL Intel UHD GPU renderer + Explicit-Notify Fast CPU Fallback
"""

from __future__ import annotations

import json
import streamlit as st

from prism_studio import (
    PRESET_SCENES,
    build_offline_director_scene,
    critique_and_patch_scene,
    load_director_env,
    plan_scene_dna,
    render_scene_dna,
    scene_dna_from_dict,
)


st.set_page_config(
    page_title="PrismStudio — Compositional Math Image Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    env_cfg = load_director_env()

    if "prism_dna_json" not in st.session_state:
        st.session_state["prism_dna_json"] = json.dumps(
            PRESET_SCENES["Cute Cat on High Sunset Wall (Anime)"], indent=2
        )
    if "prism_notices" not in st.session_state:
        st.session_state["prism_notices"] = []
    if "prism_draft_png" not in st.session_state:
        st.session_state["prism_draft_png"] = None

    st.title("✦ PrismStudio — Pure-Math Generative Image Studio")
    st.caption(
        "Compositional Bone-Graph Anatomy (4 arms, wheel-legs, elf/cat rigs) • Body-UV Outfits • "
        "3-Zone World Architecture • ModernGL Intel UHD GPU + Explicit-Notify Fast CPU Fallback"
    )

    with st.sidebar:
        st.header("🎨 Studio Controls")
        preset_list = ["— Custom Prompt —"] + list(PRESET_SCENES.keys())
        sel_preset = st.selectbox("Instant Showcase Presets", preset_list, index=1, key="prism_preset_sel")
        if sel_preset != preset_list[0] and st.session_state.get("prism_last_preset") != sel_preset:
            st.session_state["prism_dna_json"] = json.dumps(PRESET_SCENES[sel_preset], indent=2)
            st.session_state["prism_last_preset"] = sel_preset
            st.session_state["prism_draft_png"] = None
            st.session_state["prism_notices"] = []

        style_mode = st.radio(
            "Visual Aesthetic Mode",
            options=["anime_npr", "stylized_pbr"],
            format_func=lambda s: "Anime / NPR Cel + Ink + Kajiya-Kay" if s == "anime_npr" else "Stylized PBR (Soft Plush + GGX)",
            key="prism_style",
        )

        res_map = {"640×480 (Fast)": (640, 480), "960×720 (HD)": (960, 720), "1280×720 (Widescreen)": (1280, 720)}
        res_key = st.select_slider("Render Resolution", options=list(res_map.keys()), value="960×720 (HD)")
        rw, rh = res_map[res_key]

        backend_choice = st.selectbox(
            "Execution Backend",
            options=["auto", "gpu", "cpu"],
            format_func=lambda b: {"auto": "Auto (ModernGL GPU -> Fast CPU Fallback)", "gpu": "Force ModernGL GPU", "cpu": "Force Fast CPU Fallback"}[b],
        )

        use_vlm = st.checkbox("Enable VLM 320×240 Draft Self-Critique", value=False)
        offline_only = st.checkbox("Offline Compositional Director (Zero API)", value=False)

        with st.expander("LLM / VLM Provider Settings", expanded=False):
            prov = st.radio("Provider", ["groq", "ollama"], index=0 if env_cfg["provider"] == "groq" else 1, horizontal=True)
            if prov == "groq":
                api_host = st.text_input("Groq Host", value=env_cfg["groq_host"])
                text_model = st.text_input("Text Model", value=env_cfg["groq_text_model"])
                vis_model = st.text_input("Vision Critic Model", value=env_cfg["groq_vision_model"])
                api_key = st.text_input("GROQ_API_KEY", value=env_cfg["groq_api_key"], type="password")
            else:
                api_host = st.text_input("Ollama Host", value=env_cfg["ollama_host"])
                text_model = st.text_input("Text Model", value=env_cfg["ollama_text_model"])
                vis_model = st.text_input("Vision Critic Model", value=env_cfg["ollama_vision_model"])
                api_key = st.text_input("OLLAMA_API_KEY", value=env_cfg["ollama_api_key"], type="password")

        with st.expander("Instant Camera & Optical FX Overrides", expanded=False):
            ov_enable = st.checkbox("Override SceneDNA Camera/Optics", value=False)
            ov_az = st.slider("Camera Azimuth (°)", -180, 180, 24)
            ov_el = st.slider("Camera Elevation (°)", -15, 65, 8)
            ov_dof = st.slider("Vogel Bokeh DOF", 0.0, 1.0, 0.42, 0.05)
            ov_ink = st.slider("Ink Outline Strength", 0.0, 1.2, 0.62, 0.05)
            ov_bloom = st.slider("HDR Bloom", 0.0, 1.5, 0.65, 0.05)
            ov_exp = st.slider("Exposure", 0.6, 1.8, 1.12, 0.04)

    col_left, col_right = st.columns([1, 1.15], gap="large")

    with col_left:
        st.subheader("1. Natural-Language Prompt")
        prompt = st.text_area(
            "Describe any scene, creature, outfit, or world:",
            value="cute cat sitting on a high wall at sunset",
            height=115,
        )
        c1, c2 = st.columns(2)
        with c1:
            gen_btn = st.button("✦ Generate Scene", type="primary", use_container_width=True)
        with c2:
            st.button("🔄 Re-Render SceneDNA JSON", use_container_width=True)

        if gen_btn:
            st.session_state["prism_draft_png"] = None
            st.session_state["prism_notices"] = []
            if offline_only:
                dna = build_offline_director_scene(prompt, style_mode=style_mode)
                notices = ["Synthesized via offline compositional director."]
            else:
                with st.spinner("Planning SceneDNA via LLM Director..."):
                    plan = plan_scene_dna(
                        prompt,
                        style_mode=style_mode,
                        provider=prov,
                        host=api_host,
                        model=text_model,
                        api_key=api_key,
                        allow_fallback=True,
                    )
                dna = plan.dna
                notices = list(plan.warnings)

            if use_vlm:
                with st.spinner("Rendering 320×240 draft & running VLM self-critique..."):
                    cr = critique_and_patch_scene(
                        dna,
                        prompt,
                        provider=prov,
                        host=api_host,
                        vision_model=vis_model,
                        api_key=api_key,
                        prefer_backend=backend_choice,
                    )
                dna = cr.patched_dna
                st.session_state["prism_draft_png"] = cr.draft_png
                if cr.notice:
                    notices.append(cr.notice)

            st.session_state["prism_dna_json"] = json.dumps(dna.to_dict(), indent=2)
            st.session_state["prism_notices"] = notices

        with st.expander("Editable SceneDNA JSON (Deterministic Compiler Input)", expanded=True):
            st.text_area("SceneDNA JSON", key="prism_dna_json", height=380, label_visibility="collapsed")

    with col_right:
        st.subheader("2. Synthesized Pure-Math Render")
        try:
            dna_obj = scene_dna_from_dict(json.loads(st.session_state["prism_dna_json"]))
            dna_obj.style_mode = style_mode
            if ov_enable:
                dna_obj.camera.azimuth_deg = float(ov_az)
                dna_obj.camera.elevation_deg = float(ov_el)
                dna_obj.camera.dof_strength = float(ov_dof)
                dna_obj.lighting.ink_strength = float(ov_ink)
                dna_obj.lighting.bloom_intensity = float(ov_bloom)
                dna_obj.lighting.exposure = float(ov_exp)

            out = render_scene_dna(dna_obj, width=rw, height=rh, prefer_backend=backend_choice)

            if out.backend_notice:
                st.warning(out.backend_notice)
            for msg in st.session_state.get("prism_notices", []):
                st.info(msg)

            st.image(out.png_bytes, use_container_width=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Backend", out.backend_used.split("(")[0].strip())
            m2.metric("Render Time", f"{out.render_seconds:.3f}s")
            m3.metric("Splines / Solids", f"{len(out.compiled.all_splines)} / {len(out.compiled.all_solids)}")
            m4.metric("Midground Items", str(len(out.compiled.midground)))

            st.download_button(
                "⬇ Download Final PNG",
                data=out.png_bytes,
                file_name="prism_studio_render.png",
                mime="image/png",
                type="primary",
                use_container_width=True,
            )

            if st.session_state.get("prism_draft_png") is not None:
                with st.expander("VLM Draft (320×240) Before Critique Patch"):
                    st.image(st.session_state["prism_draft_png"], width=360)
        except Exception as exc:
            st.error(f"Render error: {exc}")


if __name__ == "__main__":
    main()
