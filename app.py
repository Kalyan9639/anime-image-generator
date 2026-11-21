"""
Streamlit Web Application for ASCII to Image Generator (ascii2image).

Deployment-ready interactive UI for converting ASCII diagrams and ASCII art
into high-quality SVG and PNG images using pure mathematical signal processing
and vector graphics (zero AI).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict

import cv2
from PIL import Image
import streamlit as st

from ascii2image import (
    classify_ascii,
    diagram_to_svg,
    get_available_engines,
    render_art_crisp_png,
    render_art_crisp_svg,
    reconstruct_art_photo,
    svg_to_png,
    vectorize_line_art_svg,
)

ROOT_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT_DIR / "samples"

st.set_page_config(
    page_title="ASCII → Image Generator",
    page_icon="▦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle styling for monospace input and clean metric cards
st.markdown(
    """
    <style>
    textarea {
        font-family: "Consolas", "Courier New", monospace !important;
        font-size: 13px !important;
        line-height: 1.35 !important;
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


def main() -> None:
    samples = load_samples()
    default_sample_name = "Flowchart" if "Flowchart" in samples else (next(iter(samples), None))
    default_text = samples.get(default_sample_name, "+-------+     +-------+\n| Input | --> | Image |\n+-------+     +-------+")

    if "ascii_text" not in st.session_state:
        st.session_state["ascii_text"] = default_text

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("▦ ASCII → Image Generator")
    st.caption(
        "Pure mathematical signal-processing & vector graphics pipeline • "
        "Zero AI • Exports scalable SVG & high-DPI PNG"
    )

    # ── Sidebar Controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        # Sample selector
        sample_options = ["Custom Input"] + list(samples.keys())
        default_idx = sample_options.index(default_sample_name) if default_sample_name in sample_options else 0
        selected_sample = st.selectbox("Load Built-in Sample", sample_options, index=default_idx)

        if selected_sample != "Custom Input" and st.session_state.get("last_sample") != selected_sample:
            st.session_state["ascii_text"] = samples[selected_sample]
            st.session_state["last_sample"] = selected_sample

        uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt", "bob", "asc"])
        if uploaded_file is not None:
            file_id = f"{uploaded_file.name}:{uploaded_file.size}"
            if st.session_state.get("last_upload") != file_id:
                st.session_state["ascii_text"] = uploaded_file.read().decode("utf-8", errors="replace")
                st.session_state["last_upload"] = file_id

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

        with st.expander("Photo Reconstruction Tuning", expanded=False):
            target_width = st.slider("Target Width (px)", min_value=400, max_value=2000, value=1200, step=100)
            denoise = st.slider("Denoise Smoothing", min_value=0.2, max_value=2.5, value=1.0, step=0.1)
            clahe_clip = st.slider("CLAHE Contrast Limit", min_value=1.0, max_value=5.0, value=2.0, step=0.2)
            invert = st.selectbox("Invert Luminance", options=["auto", "yes", "no"], index=0)

    # ── Main Two-Column Layout ────────────────────────────────────────────────
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("1. Input ASCII Text")
        ascii_text = st.text_area(
            "Paste your ASCII diagram or ASCII art below:",
            key="ascii_text",
            height=360,
            label_visibility="collapsed",
        )

        if not ascii_text.strip():
            st.info("Paste ASCII text above or choose a sample from the sidebar to begin.")
            return

        detection = classify_ascii(ascii_text)
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

    # ── Conversion Execution ──────────────────────────────────────────────────
    effective_mode = resolve_effective_mode(mode, detection.category, detection.subtype)
    effective_theme = resolve_effective_theme(theme, effective_mode)

    # diagram_to_svg accepts light/dark/blueprint/tinted
    diagram_theme = effective_theme if effective_theme in ("light", "dark", "blueprint", "tinted") else "light"

    svg_content: str | None = None
    png_bytes: bytes | None = None

    with col_out:
        st.subheader("2. Converted Output")
        try:
            with st.spinner("Rendering..."):
                if effective_mode == "diagram":
                    svg_content = diagram_to_svg(ascii_text, engine=engine, theme=diagram_theme)
                    png_bytes = svg_to_png(svg_content, scale=scale)
                elif effective_mode == "art:reconstruct":
                    mat = reconstruct_art_photo(
                        ascii_text,
                        target_width=target_width,
                        invert=invert,
                        denoise_strength=denoise,
                        clahe_clip=clahe_clip,
                    )
                    ok, buf = cv2.imencode(".png", mat)
                    if ok:
                        png_bytes = buf.tobytes()
                    svg_content = render_art_crisp_svg(ascii_text, theme=effective_theme)
                elif effective_mode == "art:vector":
                    svg_content = vectorize_line_art_svg(ascii_text, theme=effective_theme)
                    png_bytes = svg_to_png(svg_content, scale=scale)
                else:  # art:crisp
                    svg_content = render_art_crisp_svg(ascii_text, theme=effective_theme)
                    png_bytes = render_art_crisp_png(ascii_text, theme=effective_theme)

            # Action bar: Download buttons
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                if png_bytes:
                    st.download_button(
                        label="⬇ Download PNG",
                        data=png_bytes,
                        file_name="ascii_output.png",
                        mime="image/png",
                        use_container_width=True,
                        type="primary",
                    )
            with dl_col2:
                if svg_content:
                    st.download_button(
                        label="⬇ Download SVG",
                        data=svg_content,
                        file_name="ascii_output.svg",
                        mime="image/svg+xml",
                        use_container_width=True,
                    )

            tab_img, tab_svg = st.tabs(["🖼️ Image Preview", "📐 SVG Source"])

            with tab_img:
                if png_bytes:
                    st.image(png_bytes, use_container_width=True)
                    pil_img = Image.open(io.BytesIO(png_bytes))
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Dimensions", f"{pil_img.width} × {pil_img.height} px")
                    m2.metric("PNG Size", f"{len(png_bytes) / 1024:.1f} KB")
                    m3.metric("Render Mode", f"{effective_mode} ({effective_theme})")

            with tab_svg:
                if svg_content:
                    st.code(svg_content, language="xml")
                else:
                    st.info("No SVG output for this mode.")

        except Exception as exc:
            st.error(f"Conversion failed: {exc}")


if __name__ == "__main__":
    main()
