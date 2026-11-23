# SceneStudio: ASCII Vectorizer & Procedural 3D Scene Synthesizer

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-49%20Passed-brightgreen.svg)]()
[![Engine](https://img.shields.io/badge/Engine-Dual%20(Vector%20%2B%20Procedural%203D)-indigo.svg)]()
[![LLM Planner](https://img.shields.io/badge/LLM%20Knowledge-Groq%20%7C%20Ollama-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

**SceneStudio** (formerly `ascii2image`) is a dual-engine graphics synthesis framework combining:
1. **Deterministic ASCII Vectorizer & Reconstructor:** Converts plain-text ASCII diagrams and ASCII art pictures into infinitely scalable SVG vectors and high-resolution PNGs without any neural network hallucinations.
2. **Procedural 3D Scene Studio (Track 1):** Transforms natural-language prompts and semantic ASCII layouts into rich, multi-layered 3D/2.5D optical renders using **Groq** (`openai/gpt-oss-20b`) or **Ollama** strictly as a text knowledge architect, coupled with a pre-written mathematical rendering engine (SDFs, Barr Superquadrics, Johan Gielis Superformula, procedural shaders, and optical bokeh/particle compositing).

---

## Table of Contents

- [Dual-Engine Overview](#dual-engine-overview)
- [Engine 1: Deterministic ASCII Vectorizer & Art Reconstructor](#engine-1-deterministic-ascii-vectorizer--art-reconstructor)
  - [Mathematical Loss-Minimization Pipeline](#mathematical-loss-minimization-pipeline)
  - [Visual Modes & Examples](#visual-modes--examples)
- [Engine 2: Procedural 3D Scene Studio (Track 1)](#engine-2-procedural-3d-scene-studio-track-1)
  - [Dynamic Math Primitives & Optical Shaders](#dynamic-math-primitives--optical-shaders)
  - [Knowledge & Spatial Constraint Solver](#knowledge--spatial-constraint-solver)
- [Quick Start](#quick-start)
  - [1. Streamlit Web Dashboard (Both Workspaces)](#1-streamlit-web-dashboard-both-workspaces)
  - [2. CLI Command-Line Interface](#2-cli-command-line-interface)
  - [3. Standalone Flask Web App](#3-standalone-flask-web-app)
- [Configuration (.env)](#configuration-env)
- [Python API Reference](#python-api-reference)
- [Project Architecture](#project-architecture)
- [Automated Test Suite](#automated-test-suite)
- [GitHub Release & Versioning Guide](#github-release--versioning-guide)

---

## Dual-Engine Overview

```
                                      ┌─────────────────────────────────────────────────────────┐
                                      │                    SceneStudio Input                    │
                                      └────────────────────────────┬────────────────────────────┘
                                                                   │
                                     ┌─────────────────────────────┴─────────────────────────────┐
                                     ▼                                                           ▼
                      [Plain ASCII Diagram or Art]                             [Natural Language Prompt & Semantic ASCII]
                                     │                                                           │
                                     ▼                                                           ▼
                        ┌────────────────────────┐                                  ┌────────────────────────┐
                        │   Engine 1: Classic    │                                  │ Engine 2: Scene Studio │
                        │  ASCII Vectorizer &    │                                  │  (Groq/Ollama Planner  │
                        │   Art Reconstructor    │                                  │   + 3D Math Pipeline)  │
                        └────────────┬───────────┘                                  └────────────┬───────────┘
                                     │                                                           │
          ┌──────────────────────────┼──────────────────────────┐                                │
          ▼                          ▼                          ▼                                ▼
   [Vector SVG Paths]       [5-Stage Tone Photo]       [Monospace CRT]              [3D Raymarched G-Buffer]
   (svgbob / aafigure)      (Bilateral + CLAHE)         (Vector Text)                            │
          │                          │                          │                                ▼
          └──────────────────────────┼──────────────────────────┘                   [2.5D Optical Bokeh & DOF]
                                     ▼                                                           │
                        ┌────────────────────────┐                                               ▼
                        │ High-Res SVG / Retina  │                                  ┌────────────────────────┐
                        │       PNG Output       │                                  │ Multi-Layer 3D PNG/SVG │
                        └────────────────────────┘                                  └────────────────────────┘
```

---

## Engine 1: Deterministic ASCII Vectorizer & Art Reconstructor

Operates 100% locally with **zero AI image generation**. Built on topological grammar parsing, sub-cell spatial calibration, structure-tensor anisotropy, and edge-preserving non-linear bilateral filtering.

### Mathematical Loss-Minimization Pipeline

1. **$8 \times 4$ Sub-Cell Spatial Discretization (32 Samples/Character):**
   Renders each character onto a $4\times$ supersampled fixed-cell canvas baseline-locked with `anchor="ls"`, downsampling via exact area integration (`cv2.INTER_AREA`) into an $8\text{-row} \times 4\text{-column}$ sub-cell spatial matrix to preserve sub-character stroke locations (`^`, `_`, `/`, `\`).
2. **Structure Tensor & Directional Coherence:**
   Calculates the symmetric structure tensor from Sobel gradients $J = \begin{bmatrix} \langle I_x^2 \rangle & \langle I_x I_y \rangle \\ \langle I_x I_y \rangle & \langle I_y^2 \rangle \end{bmatrix}$ to compute directional coherence score $\alpha = \left(\frac{\lambda_1 - \lambda_2}{\lambda_1 + \lambda_2 + \epsilon}\right)^2$, preserving directional strokes while smoothing isotropic shading (`@`, `#`, `%`).
3. **Edge-Preserving Bilateral Filtering:**
   Weights neighboring samples simultaneously by spatial Euclidean distance and photometric luminance difference to dissolve character-grid seams without blurring object contours.
4. **CLAHE (Contrast Limited Adaptive Histogram Equalization):**
   Restores subtle local illumination gradients without over-amplifying background noise.

### Visual Modes & Examples

- **Diagram Mode:** Converts flowcharts (`+---+`, `| |`, `-->`) into true SVG rectangles, rounded corners, arrows, and styled typography.
- **Tonal Art Mode:** Reconstructs shaded ASCII art (portraits, landscapes, continuous tones) into smooth photographic grayscales.
- **Line Art Mode:** Traces cartoons and sketch contours into smooth cubic Bézier SVG curves.
- **Crisp Monospace Mode:** Vector typography with customizable CRT themes (`light`, `dark`, `blueprint`, `tinted`, `terminal_green`, `amber`).

---

## Engine 2: Procedural 3D Scene Studio (Track 1)

Generates multi-object, atmospheric 3D images from natural language or boxy semantic ASCII layouts using **Groq** (`openai/gpt-oss-20b`) or **Ollama** strictly as a scene architect—**never writing Python, GLSL, or Blender code**.

### Dynamic Math Primitives & Optical Shaders

- **Signed Distance Fields (SDFs):** Rounded boxes, cylinders, cones, capsules, tori, and CSG booleans (`union`, `smooth_union`, `subtract`, `intersect`).
- **Barr Superquadrics:** Smooth parametric forms governed by exponents $e_1, e_2$ (pill capsules, beveled cushions, rounded slabs).
- **Johan Gielis Superformula:** Intricate multi-symmetry procedural geometry (6-arm snowflakes, crystalline forms, flower petals).
- **Lathe & Spline Sweeps:** Lathe revolution profiles (cups, mugs, lanterns, vases) and cubic Bézier sweep tubes (handles, cables, arches).
- **Branching L-Systems:** Centered parametric branching structures for trees and botanical elements.
- **Procedural Shaders:** Vectorized NumPy textures including tartan plaids, cable-knit corduroy, Voronoi stone tiles, wood grain, brushed metal, and etched PCB circuit traces.
- **2.5D Optical Compositor:** Depth-sorted bokeh discs with optical iris rings, falling snowflakes/sparks, ACES filmic tone mapping, bloom highlights, and depth-of-field blurring.

---

## Quick Start

### 1. Streamlit Web Dashboard (Both Workspaces)

The recommended visual interface containing both the **Classic ASCII Converter** and the **Scene Studio**:

```powershell
# Launch the Streamlit dashboard
streamlit run app.py
```

- **Tab 1: ▦ ASCII → Diagram & Art:** Live ASCII editor with automatic diagram/art classification, auto-indent repair, multiple styling themes, and SVG/PNG downloads.
- **Tab 2: 🎨 Scene Studio (LLM + Dynamic Math):** Prompt-based scene generation using Groq or Ollama, zero-cost offline math presets, semantic ASCII box spatial locking, camera/optics overrides, and live 3D PNG / 2D SVG viewers.

---

### 2. CLI Command-Line Interface

#### A. Classic ASCII Diagram & Art Conversion
```powershell
# Auto-detect and convert an ASCII diagram to SVG and PNG
python cli.py samples/flowchart.txt -o out/

# Convert architecture diagram with Blueprint theme
python cli.py samples/network.txt --theme blueprint -o out/

# Reconstruct continuous-tone ASCII portrait/landscape art
python cli.py samples/house.txt --reconstruct -o out/

# Vectorize cartoon line sketch into smooth Bézier SVG paths
python cli.py samples/cat.txt -m art:vector -o out/
```

#### B. Scene Studio Procedural 3D Generation
```powershell
# Plan and render a 3D scene via Groq (openai/gpt-oss-20b) + dynamic math
python cli.py --scene-prompt "A glowing brass lantern on a tiled stone ledge next to a ceramic mug and a pine tree, with warm golden bokeh lights" --outdir out/

# Render with custom resolution (default: 640x480)
python cli.py --scene-prompt "Microprocessor chip floating above circuit board with copper traces" --scene-size 800x600 --outdir out/

# Combine natural language prompt with an ASCII spatial layout file
python cli.py dcn.txt --scene-prompt "Cloud network nodes with holographic links" --outdir out/
```

---

### 3. Standalone Flask Web App

For lightweight, dedicated ASCII-to-image conversion:
```powershell
python web/app.py
# Open http://127.0.0.1:5000 in your browser
```

---

## Configuration (`.env`)

Configure your preferred LLM provider for Scene Studio:

```ini
# LLM Provider: "groq" (recommended, fast LPU inference) or "ollama"
LLM_PROVIDER=groq

# Groq Cloud Configuration
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_HOST=https://api.groq.com/openai/v1
GROQ_MODEL=openai/gpt-oss-20b

# Ollama Configuration (Local or Cloud)
OLLAMA_API_KEY=your_ollama_cloud_key_if_applicable
OLLAMA_HOST=https://ollama.com
OLLAMA_MODEL=gpt-oss:20b-cloud
```

*Note: If no API key or network connection is available, Scene Studio includes built-in offline math presets and deterministic ASCII-box fallback generation.*

---

## Python API Reference

```python
# 1. Classic ASCII Conversion
from ascii2image import classify_ascii, diagram_to_svg, svg_to_png, reconstruct_art_photo

text = """
+---------+       +---------+
| Client  | ----> | Server  |
+---------+       +---------+
"""

diag_type = classify_ascii(text)
svg_str = diagram_to_svg(text, theme="blueprint")
png_bytes = svg_to_png(svg_str, scale=2.0)

# 2. Scene Studio Procedural Generation
from ascii2image.semantic import plan_scene, render_scene_graph

# Plan scene graph using Groq / Ollama text knowledge
plan_res = plan_scene("A glowing brass lantern on a stone ledge with bokeh lights")

# Deterministically evaluate 3D geometry & composite optical effects
out = render_scene_graph(plan_res.scene, width=640, height=480, include_svg=True)

# Access outputs
with open("scene.png", "wb") as f:
    f.write(out.png_bytes)
```

---

## Project Architecture

```
SceneStudio/
├── app.py                      # Streamlit dual-workspace application
├── cli.py                      # CLI supporting both classic and scene-studio modes
├── requirements.txt            # Python dependencies
├── .env                        # API keys and provider settings
├── ascii2image/                # Core Python package
│   ├── detect.py               # 6-feature ASCII diagram/art classifier
│   ├── normalize.py            # Indent repair, fence removal, sanitation
│   ├── ink.py                  # Photometric sub-cell font atlas calibration
│   ├── diagram.py              # svgbob & aafigure vector diagram parser
│   ├── art.py                  # Tone reconstruction, crisp text, Bézier tracing
│   ├── export.py               # resvg_py high-DPI rasterization
│   └── semantic/               # Track 1 Procedural 3D Scene Studio Engine
│       ├── schema.py           # SceneGraph specifications & JSON Schemas
│       ├── parser.py           # Semantic ASCII layout parser
│       ├── math_core.py        # Vectorized SDFs, Superquadrics, Gielis, L-Systems
│       ├── patterns.py         # Procedural shaders (tartan, knit, wood, tiles, PCB)
│       ├── solver.py           # Constraint solver & geometric collision relaxer
│       ├── renderer_3d.py      # BVH ray-AABB sphere-tracer & G-buffer
│       ├── compositor.py       # 2.5D optical compositor (bokeh, DOF, bloom)
│       ├── renderer_2d.py      # Vector SVG isometric/perspective projector
│       └── planner.py          # Groq / Ollama LLM knowledge & planning bridge
├── samples/                    # Verified ASCII sample files
├── tests/                      # Automated test suite (49 unit tests)
└── web/                        # Lightweight Flask web app
```

---

## Automated Test Suite

All 49 unit and integration tests run offline in ~2–4 seconds:

```powershell
.\a2i\Scripts\python.exe -m pytest tests -v
```

Tests cover:
- Photometric font monotonicity and sub-cell distribution
- 6-feature topological classification
- Box diagram parsing, rounded corner preservation, and auto-indent repair
- Analytical SDF zeroes, surface gradients, and BVH ray-AABB culling
- Procedural shaders and texture generators
- Multi-object spatial constraint solver and flush attachment
- Mocked Groq / Ollama planning and automatic retry logic
- Web endpoints and Streamlit AppTest UI validation

---

## GitHub Release & Versioning Guide

When releasing this project on GitHub:

### Release Tagging Strategy
- **Release `v1.0.0` (Tag: `v1.0.0`):** *Classic ASCII to Diagram & Art Vectorizer*  
  Captures the standalone mathematical ASCII diagram/art conversion tool.
- **Release `v2.0.0` (Tag: `v2.0.0`):** *SceneStudio — Dual Engine ASCII Vectorizer & Procedural 3D Studio*  
  Adds the Track 1 dynamic 3D math engine, Groq/Ollama scene planner, and Streamlit Scene Studio.

### How GitHub Release Snapshots Work
- When you create a release on GitHub, **GitHub freezes an immutable ZIP/TAR snapshot of your entire codebase at that exact commit**.
- Your previous release code is **never lost or overwritten**. You and other users can download older releases anytime under GitHub's **Releases** tab or check them out with `git checkout v1.0.0`.
