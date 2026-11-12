"""
Flask Web Application for ASCII to Image Generator (ascii2image).

Local web server providing an interactive browser interface to paste ASCII text,
inspect real-time topological classification, preview SVG/PNG outputs side-by-side,
and download high-resolution images.
"""

import base64
import io
import os
import sys
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request, send_file

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ascii2image import (
    classify_ascii,
    diagram_to_svg,
    get_available_engines,
    render_art_crisp_svg,
    render_art_crisp_png,
    reconstruct_art_photo,
    vectorize_line_art_svg,
    svg_to_png,
)

app = Flask(__name__)
SAMPLES_DIR = PROJECT_ROOT / "samples"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns available sample ASCII text snippets."""
    samples = []
    if SAMPLES_DIR.is_dir():
        for file in sorted(SAMPLES_DIR.glob("*.txt")):
            try:
                content = file.read_text(encoding="utf-8", errors="replace")
                name = file.stem.replace("_", " ").title()
                samples.append({"id": file.name, "name": name, "text": content})
            except Exception:
                pass
    return jsonify({"samples": samples})


@app.route("/api/detect", methods=["POST"])
def detect_ascii():
    """Analyzes text and returns the 6-feature classification results."""
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    result = classify_ascii(text)
    return jsonify({
        "category": result.category,
        "subtype": result.subtype,
        "confidence": result.confidence,
        "explanation": result.explanation,
        "scores": result.scores,
    })


@app.route("/api/convert", methods=["POST"])
def convert_ascii():
    """Converts ASCII text to SVG and PNG preview according to parameters."""
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    mode = data.get("mode", "auto")
    theme = data.get("theme", "auto")
    scale = float(data.get("scale", 2.0))
    engine = data.get("engine", "auto")
    denoise = float(data.get("denoise", 1.0))
    clahe_clip = float(data.get("clahe", 2.0))
    width = int(data.get("width", 1000))
    invert = data.get("invert", "auto")

    detection = classify_ascii(text)

    # Resolve automatic mode
    effective_mode = mode
    if effective_mode == "auto":
        if detection.category == "diagram":
            effective_mode = "diagram"
        elif detection.subtype == "art:tone":
            effective_mode = "art:reconstruct"
        elif detection.subtype == "art:line":
            effective_mode = "art:crisp"
        else:
            effective_mode = "art:crisp"

    # Resolve automatic theme
    effective_theme = theme
    if effective_theme == "auto":
        effective_theme = "light" if effective_mode == "diagram" else "dark"

    svg_content = None
    png_bytes = None

    try:
        if effective_mode == "diagram":
            svg_content = diagram_to_svg(text, engine=engine, theme=effective_theme)
            png_bytes = svg_to_png(svg_content, scale=scale)

        elif effective_mode == "art:reconstruct":
            mat = reconstruct_art_photo(
                text,
                target_width=width,
                invert=invert,
                denoise_strength=denoise,
                clahe_clip=clahe_clip
            )
            is_success, buffer = cv2.imencode(".png", mat)
            if is_success:
                png_bytes = buffer.tobytes()
            # Also generate vector crisp SVG
            svg_content = render_art_crisp_svg(text, theme=effective_theme)

        elif effective_mode == "art:vector":
            svg_content = vectorize_line_art_svg(text, theme=effective_theme)
            png_bytes = svg_to_png(svg_content, scale=scale)

        else:  # art:crisp
            svg_content = render_art_crisp_svg(text, theme=effective_theme)
            png_bytes = render_art_crisp_png(text, theme=effective_theme)

        png_b64 = base64.b64encode(png_bytes).decode("ascii") if png_bytes else ""

        return jsonify({
            "status": "success",
            "effective_mode": effective_mode,
            "effective_theme": effective_theme,
            "detection": {
                "category": detection.category,
                "subtype": detection.subtype,
                "confidence": detection.confidence,
                "explanation": detection.explanation,
                "scores": detection.scores,
            },
            "svg": svg_content,
            "png_b64": png_b64,
            "png_size_bytes": len(png_bytes) if png_bytes else 0,
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


@app.route("/api/download/<file_type>", methods=["POST"])
def download_image(file_type):
    """Direct binary download endpoint for generated SVG or PNG."""
    data = request.form
    text = data.get("text", "")
    mode = data.get("mode", "auto")
    theme = data.get("theme", "auto")
    scale = float(data.get("scale", 2.0))
    engine = data.get("engine", "auto")
    denoise = float(data.get("denoise", 1.0))
    clahe_clip = float(data.get("clahe", 2.0))
    width = int(data.get("width", 1200))
    invert = data.get("invert", "auto")

    detection = classify_ascii(text)
    effective_mode = mode
    if effective_mode == "auto":
        effective_mode = "diagram" if detection.category == "diagram" else "art:crisp"

    effective_theme = theme
    if effective_theme == "auto":
        effective_theme = "light" if effective_mode == "diagram" else "dark"

    if file_type == "svg":
        if effective_mode == "diagram":
            svg = diagram_to_svg(text, engine=engine, theme=effective_theme)
        elif effective_mode == "art:vector":
            svg = vectorize_line_art_svg(text, theme=effective_theme)
        else:
            svg = render_art_crisp_svg(text, theme=effective_theme)

        return send_file(
            io.BytesIO(svg.encode("utf-8")),
            mimetype="image/svg+xml",
            as_attachment=True,
            download_name="ascii_output.svg"
        )

    elif file_type == "png":
        if effective_mode == "diagram":
            svg = diagram_to_svg(text, engine=engine, theme=effective_theme)
            png = svg_to_png(svg, scale=scale)
        elif effective_mode == "art:reconstruct":
            mat = reconstruct_art_photo(
                text,
                target_width=width,
                invert=invert,
                denoise_strength=denoise,
                clahe_clip=clahe_clip
            )
            _, buf = cv2.imencode(".png", mat)
            png = buf.tobytes()
        elif effective_mode == "art:vector":
            svg = vectorize_line_art_svg(text, theme=effective_theme)
            png = svg_to_png(svg, scale=scale)
        else:
            png = render_art_crisp_png(text, theme=effective_theme)

        return send_file(
            io.BytesIO(png),
            mimetype="image/png",
            as_attachment=True,
            download_name="ascii_output.png"
        )

    return ("Invalid format", 400)


def start_server(host="127.0.0.1", port=5000, debug=False):
    print(f"\n========================================================")
    print(f"  ASCII to Image Generator Web Server")
    print(f"  Open in your browser: http://{host}:{port}")
    print(f"========================================================\n")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    start_server()
