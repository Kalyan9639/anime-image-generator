"""
Command-Line Interface for ASCII to Image Generator (ascii2image).

Converts ASCII diagrams and ASCII art pictures into high-quality SVG and PNG
images using pure mathematical signal processing and graphics frameworks.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
from PIL import Image

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


HELP_BANNER = """
================================================================================
          ASCII to Image Generator (ascii2image) - CLI Manual
================================================================================

USAGE:
  python cli.py <input.txt> [OPTIONS]
  python cli.py help                    (Show this help guide)
  python cli.py --help                  (Standard argument help)

DESCRIPTION:
  Converts text-based ASCII diagrams (boxes, arrows, flowcharts) and ASCII art
  (portraits, cartoons, block elements) into clean, modern SVG vector graphics
  and high-resolution PNG images without using any AI image generators.

POSITIONAL ARGUMENTS:
  input                 Path to the ASCII text file to convert (e.g., diagram.txt)

OUTPUT OPTIONS:
  -o, --outdir DIR      Output folder where files will be saved (default: ./out)
  --format {both,svg,png}
                        Output format to export:
                          • both (default): Saves both .svg and .png
                          • svg:            Saves only scalable vector .svg
                          • png:            Saves only raster .png
  -s, --scale FLOAT     PNG resolution multiplier (default: 2.0 for 2x Retina).
                        Use 1.0 for standard, 3.0 for ultra-high-definition.

CONVERSION MODES (-m, --mode):
  auto                  (Default) Uses 6-feature classifier to auto-detect
                        between diagrams and art with high confidence.
  diagram               Renders structured diagrams (boxes, rounded corners,
                        clean lines, arrows, and styled labels) into SVG paths.
  art:crisp             Renders ASCII art with exact monospace grid alignment
                        and high-DPI anti-aliased typography.
  art:reconstruct       5-stage mathematical reconstruction: 8x4 sub-cell spatial
                        integration, bilateral edge-preserving filter & CLAHE
                        producing smooth continuous-tone grayscale photos.
  art:vector            Traces sparse line-art sketches into smooth SVG Bézier paths.

THEMES (-t, --theme):
  light                 Crisp slate-gray lines on clean white (default for diagrams)
  dark                  Vibrant cyan/white strokes on slate-900 (default for art)
  blueprint             Deep CAD blue background with soft cyan lines and text
  tinted                Warm background with soft pastel fills inside boxes
  terminal_green        Retro CRT monochrome phosphor green on black
  amber                 Retro monochrome amber on dark chocolate

ADVANCED RECONSTRUCTION OPTIONS:
  --reconstruct         Shortcut flag to force 5-stage photo reconstruction.
  --width INT           Target width in pixels for reconstruction (default: 1200).
  --denoise FLOAT       Bilateral filter denoise factor (default: 1.0).
  --clahe FLOAT         CLAHE contrast enhancement clip limit (default: 2.0).
  --invert {auto,yes,no}
                        Inverts luminance polarity (default: auto).
  --engine {auto,svgbob,aafigure}
                        Diagram parsing engine (default: auto -> svgbob).

EXAMPLES:
  # 1. Quick automatic conversion (auto-detects and exports both .svg and .png)
  python cli.py samples/flowchart.txt

  # 2. Convert an architecture diagram with blueprint CAD styling
  python cli.py cc_diag.txt --theme blueprint -o out/

  # 3. High-resolution (3x Print DPI) diagram export
  python cli.py samples/flowchart.txt --scale 3.0 -o out/

  # 4. Photographic reconstruction of shaded ASCII art
  python cli.py samples/shading_bourke.txt --reconstruct -o out/

  # 5. Convert line-art sketch (cat) into smooth vector Bézier curves
  python cli.py samples/cat.txt -m art:vector -o out/

  # 6. Detailed diagnostic score breakdown
  python cli.py cc_diag.txt -v
================================================================================
"""


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert ASCII diagrams and ASCII art pictures into high-quality SVG and PNG images.",
        add_help=True
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        type=str,
        help="Path to the ASCII text file to convert (or 'help' to display manual)."
    )
    parser.add_argument(
        "-o", "--outdir",
        type=str,
        default="out",
        help="Directory to save output image files (default: ./out)."
    )
    parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=["auto", "diagram", "art:crisp", "art:reconstruct", "art:vector"],
        default="auto",
        help="Conversion mode. 'auto' uses the 6-feature classifier (default: auto)."
    )
    parser.add_argument(
        "-t", "--theme",
        type=str,
        choices=["light", "dark", "blueprint", "tinted", "terminal_green", "amber"],
        default=None,
        help="Visual theme (defaults to 'light' for diagrams, 'dark' for art)."
    )
    parser.add_argument(
        "-s", "--scale",
        type=float,
        default=2.0,
        help="High-DPI supersampling factor for PNG rasterization (default: 2.0)."
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["auto", "svgbob", "aafigure"],
        default="auto",
        help="Diagram rendering engine (default: auto -> svgbob)."
    )
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="Force 5-stage mathematical reconstruction for ASCII art (outputs photo-like grayscale)."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1200,
        help="Target pixel width for photographic reconstruction (default: 1200)."
    )
    parser.add_argument(
        "--denoise",
        type=float,
        default=1.0,
        help="Bilateral filter denoise strength factor (default: 1.0)."
    )
    parser.add_argument(
        "--clahe",
        type=float,
        default=2.0,
        help="CLAHE contrast enhancement clip limit (default: 2.0)."
    )
    parser.add_argument(
        "--invert",
        type=str,
        choices=["auto", "yes", "no"],
        default="auto",
        help="Inversion mode for reconstruction (default: auto)."
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["both", "svg", "png"],
        default="both",
        help="Output format: 'both' (default), 'svg', or 'png'."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print diagnostic detection scores and font metrics."
    )
    return parser


def main():
    parser = build_parser()

    # If called with no arguments or explicitly with "help"
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1].lower() in ("help", "-help", "--help-full")):
        print(HELP_BANNER)
        sys.exit(0)

    args = parser.parse_args()

    if not args.input or args.input.lower() == "help":
        print(HELP_BANNER)
        sys.exit(0)

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"\n[Error] Input file not found: {input_path}", file=sys.stderr)
        print("Tip: Run 'python cli.py help' to see usage instructions and examples.\n", file=sys.stderr)
        sys.exit(1)

    text = input_path.read_text(encoding="utf-8", errors="replace")
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    # 1. Detection Phase
    detection = classify_ascii(text)
    print(f"\n========================================================")
    print(f"  ASCII to Image Generator (ascii2image v1.0.0)")
    print(f"========================================================")
    print(f"Input:       {input_path.name} ({len(text.splitlines())} lines)")
    print(f"Classified:  {detection.category.upper()} [{detection.subtype}] ({detection.confidence * 100:.1f}% confidence)")
    print(f"Explanation: {detection.explanation}")

    if args.verbose:
        print("\nDiagnostic Scores:")
        for k, v in detection.scores.items():
            print(f"  - {k:<20}: {v:.4f}")
    print("--------------------------------------------------------")

    # 2. Determine Mode & Theme
    chosen_mode = args.mode
    if chosen_mode == "auto":
        if args.reconstruct or detection.subtype == "art:tone":
            chosen_mode = "art:reconstruct" if args.reconstruct else ("art:tone" if detection.category == "art" else "diagram")
        elif detection.category == "diagram":
            chosen_mode = "diagram"
        elif detection.subtype == "art:line":
            chosen_mode = "art:crisp"
        else:
            chosen_mode = "art:crisp"

    # Default themes based on category
    if args.theme is not None:
        theme = args.theme
    else:
        theme = "light" if (chosen_mode == "diagram" or detection.category == "diagram") else "dark"

    print(f"Mode:        {chosen_mode}")
    print(f"Theme:       {theme}")
    print(f"Scale:       {args.scale}x")

    created_files = []

    # 3. Execution Phase
    if chosen_mode == "diagram":
        svg_content = diagram_to_svg(text, engine=args.engine, theme=theme)
        if args.format in ("both", "svg"):
            svg_file = out_dir / f"{stem}_diagram.svg"
            svg_file.write_text(svg_content, encoding="utf-8")
            created_files.append(svg_file)

        if args.format in ("both", "png"):
            png_bytes = svg_to_png(svg_content, scale=args.scale)
            png_file = out_dir / f"{stem}_diagram.png"
            png_file.write_bytes(png_bytes)
            created_files.append(png_file)

    elif chosen_mode in ("art:reconstruct", "art:tone"):
        # Photographic grayscale reconstruction
        reconstructed_mat = reconstruct_art_photo(
            text,
            target_width=args.width,
            invert=args.invert,
            denoise_strength=args.denoise,
            clahe_clip=args.clahe
        )
        png_file = out_dir / f"{stem}_reconstructed.png"
        cv2.imwrite(str(png_file), reconstructed_mat)
        created_files.append(png_file)

        # Also provide crisp vector SVG alongside
        if args.format in ("both", "svg"):
            crisp_svg = render_art_crisp_svg(text, theme=theme)
            svg_file = out_dir / f"{stem}_crisp.svg"
            svg_file.write_text(crisp_svg, encoding="utf-8")
            created_files.append(svg_file)

    elif chosen_mode == "art:vector":
        vector_svg = vectorize_line_art_svg(text, theme=theme)
        if args.format in ("both", "svg"):
            svg_file = out_dir / f"{stem}_vector.svg"
            svg_file.write_text(vector_svg, encoding="utf-8")
            created_files.append(svg_file)

        if args.format in ("both", "png"):
            png_bytes = svg_to_png(vector_svg, scale=args.scale)
            png_file = out_dir / f"{stem}_vector.png"
            png_file.write_bytes(png_bytes)
            created_files.append(png_file)

    else:  # art:crisp
        crisp_svg = render_art_crisp_svg(text, theme=theme)
        if args.format in ("both", "svg"):
            svg_file = out_dir / f"{stem}_crisp.svg"
            svg_file.write_text(crisp_svg, encoding="utf-8")
            created_files.append(svg_file)

        if args.format in ("both", "png"):
            png_bytes = render_art_crisp_png(text, theme=theme, scale=args.scale)
            png_file = out_dir / f"{stem}_crisp.png"
            png_file.write_bytes(png_bytes)
            created_files.append(png_file)

    print("--------------------------------------------------------")
    print("Files successfully generated:")
    for f in created_files:
        size_kb = f.stat().st_size / 1024.0
        print(f"  -> {f} ({size_kb:.1f} KB)")
    print("========================================================\n")


if __name__ == "__main__":
    main()
