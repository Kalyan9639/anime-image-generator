"""
Streamlit Web Application for ASCII to Image Generator (ascii2image).

Deployment-ready interactive UI for converting ASCII diagrams and ASCII art
into high-quality SVG and PNG images using pure mathematical signal processing
and vector graphics (zero AI).
"""

from __future__ import annotations

import hashlib
import inspect
import io
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

ROOT_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT_DIR / "samples"
FALLBACK_SAMPLE = "+-------+     +-------+\n| Input | --> | Image |\n+-------+     +-------+"

st.set_page_config(
    page_title="ASCII → Image Generator",
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
    st.title("▦ ASCII → Image Generator")
    st.caption(
        "Pure mathematical signal-processing & vector graphics pipeline • "
        "Zero AI • Exports scalable SVG & high-DPI PNG"
    )

    # ── Sidebar Controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

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

    # ── Main Two-Column Layout ────────────────────────────────────────────────
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

        # Explicit action buttons so pasting new ASCII ALWAYS has an immediate trigger
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
            return

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
            return

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

    # ── Conversion Execution ──────────────────────────────────────────────────
    effective_mode = resolve_effective_mode(mode, detection.category, detection.subtype)
    effective_theme = resolve_effective_theme(theme, effective_mode)

    # diagram_to_svg accepts light/dark/blueprint/tinted; art modes accept light/dark/blueprint/terminal_green/amber
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

            # Action bar: Download buttons
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


if __name__ == "__main__":
    main()
