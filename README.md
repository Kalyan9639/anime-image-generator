# ASCII to Image Generator (`ascii2image`)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-26%20Passed-brightgreen.svg)]()
[![Zero AI](https://img.shields.io/badge/Engine-Zero%20AI%20%2F%20Pure%20Math-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

A pure-code, mathematical signal-processing and computer-graphics tool to convert **ASCII diagrams** and **ASCII art pictures** into high-quality **SVG** (infinitely scalable vector graphics) and **PNG** (high-resolution raster images).

> **Zero AI Image Generators:** Operates 100% locally on your computer with zero neural network hallucination. Built purely on topological grammar parsing, fixed-cell photometric font calibration, structure-tensor eigenvalue analysis, non-linear edge-preserving bilateral filtering, and high-DPI vector rasterization.

---

## Table of Contents

- [Capabilities Overview](#capabilities-overview)
- [How It Minimizes Loss (Mathematical Concepts)](#how-it-minimizes-loss-mathematical-concepts)
- [Visual Examples](#visual-examples)
  - [Example 1: Software Architecture Diagram](#example-1-software-architecture-diagram)
  - [Example 2: Landscape & Scene Art](#example-2-landscape--scene-art)
  - [Example 3: Continuous-Tone Photographic Shading](#example-3-continuous-tone-photographic-shading)
  - [Example 4: Line-Art Cartoon Sketch](#example-4-line-art-cartoon-sketch)
- [Quick Start](#quick-start)
  - [CLI Command-Line Interface](#1-using-the-command-line-interface-clipy)
  - [Interactive Web Dashboard](#2-using-the-interactive-local-web-app)
- [Python Library API](#python-library-api)
- [Theme Palette Reference](#theme-palette-reference)
- [Project Architecture](#project-architecture)
- [Running Automated Tests](#running-automated-tests)

---

## Capabilities Overview

| ASCII Input Type | Sub-Type | Output Modes Provided |
|---|---|---|
| **ASCII Diagram** | Flowcharts (`+---+`), architecture blocks, network topologies (`.---.`), sequence flows, tables | **Vector Diagram Mode:** True geometric SVG paths (boxes, smooth rounded corners, arrows, modern typography) + 2×/3× Retina PNG |
| **ASCII Art (Tonal)** | Portraits, photographic halftones (`@#%*+=-:. `) | **5-Stage Mathematical Reconstruction:** Continuous-tone grayscale image via $8 \times 4$ sub-cell density integration, bilateral filtering, and CLAHE |
| **ASCII Art (Line/Sketch)** | Cartoons, animal sketches (`/\_/\`), typography | **Crisp Monospace Mode** (exact vector text) or **Vector Contour Mode** (Bézier curves) |
| **ASCII Art (Block)** | Unicode blocks (`█▀▄▌▐░▒▓`), Braille (`⠀-⣿`) | High-contrast geometric reconstruction and crisp typography |

---

## How It Minimizes Loss (Mathematical Concepts)

Turning discrete text characters back into an image is inherently ill-posed because ASCII throws away almost all microscopic pixel information. Naive scripts average each character into a single $1 \times 1$ pixel and apply a simple Gaussian blur, producing muddy, unrecognizable results.

`ascii2image` minimizes information loss using **four mathematical frameworks**:

### 1. $8 \times 4$ Sub-Cell Spatial Discretization (32 Samples/Character)
Instead of collapsing each character into a single flat $1 \times 1$ pixel, each character is rendered onto a $4\times$ supersampled fixed-cell canvas (`cell_width` $\times$ `cell_height`, baseline-locked with `anchor="ls"`) and downsampled via exact area integration (`cv2.INTER_AREA`) into an **$8\text{-row} \times 4\text{-column}$ sub-cell spatial matrix**.
- Preserves sub-character stroke locations:
  - `^` ink is concentrated in the top $25\%$
  - `.` and `_` ink sits strictly at the bottom
  - `/` and `\` preserve diagonal slopes
  - `▀` and `▄` occupy top and bottom halves

### 2. Structure Tensor & Eigenvalue Anisotropy Analysis
To prevent dense shading characters (`@`, `#`, `%`) from creating "screen-door" noise while keeping line strokes crisp, we calculate the $2 \times 2$ symmetric structure tensor using Sobel image gradients $\nabla I = (I_x, I_y)$:

$$J = \begin{bmatrix} \langle I_x^2 \rangle & \langle I_x I_y \rangle \\ \langle I_x I_y \rangle & \langle I_y^2 \rangle \end{bmatrix}$$

We solve for its eigenvalues $\lambda_1 \ge \lambda_2 \ge 0$ to calculate a directional coherence score $\alpha \in [0, 1]$:

$$\alpha = \left(\frac{\lambda_1 - \lambda_2}{\lambda_1 + \lambda_2 + \epsilon}\right)^2$$

- Directional characters (`/`, `\`, `|`, `-`) maintain their exact sub-cell stroke geometry.
- Isotropic shading characters (`@`, `#`, `%`, `&`) are smoothly blended to prevent interior grid holes.

### 3. Non-Linear Edge-Preserving Bilateral Filtering
Standard linear filters blur across object boundaries, destroying silhouettes. The bilateral filter weights neighboring pixels simultaneously by **spatial Euclidean distance** $g_{\sigma_s}$ and **photometric luminance difference** $g_{\sigma_r}$:

$$I_{\text{filtered}}(p) = \frac{1}{W_p} \sum_{q \in S} I(q) \, g_{\sigma_s}(\|p - q\|) \, g_{\sigma_r}(|I(q) - I(p)|)$$

- Inside uniform character regions, $g_{\sigma_r} \approx 1$, mathematically dissolving the blocky character-grid seams.
- Across sharp boundaries where character density jumps abruptly (e.g. an outline meeting the background), $g_{\sigma_r} \to 0$, keeping silhouettes and contours crisp.

### 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
Restores subtle local shading gradients without blowing out highlights or over-amplifying noise.

---

## Visual Examples

### Example 1: Software Architecture Diagram
**Input file:** `samples/cc_diag.txt`
```text
Cloud Node 1 (VM / Server)                   Cloud Node 2 (VM / Server)
+------------------------------------------+   +------------------------------------------+
|  [Process 1]              [Process 2]    |   |  [Process 3]              [Process 4]    |
|       |                        |         |   |       ^                        |         |
|       +-----------+------------+         |   |       |                        |         |
|                   v                      |   |       |                        v         |
|        Cloud Haskell Runtime &           |   |        Cloud Haskell Runtime &           |
|        Network Transport Layer           |==>|        Network Transport Layer           |
+------------------------------------------+   +------------------------------------------+
                   Async Message Passing (No Shared Memory)
```

**Command:**
```powershell
python cli.py samples/cc_diag.txt -o out/
```

**Console Output:**
```
========================================================
  ASCII to Image Generator (ascii2image v1.0.0)
========================================================
Input:       cc_diag.txt (10 lines)
Classified:  DIAGRAM [diagram:box] (95.6% confidence)
Explanation: Detected diagram structure: 10 verified 90° box corner(s), 4 relational arrow vector(s), 50.1% orthogonal grid lines, 7 enclosed label(s).
--------------------------------------------------------
Mode:        diagram
Theme:       light
Scale:       2.0x
--------------------------------------------------------
Files successfully generated:
  -> out/cc_diag_diagram.svg (5.3 KB)
  -> out/cc_diag_diagram.png (26.7 KB)
========================================================
```
* **Result:** Generates `cc_diag_diagram.svg` and `cc_diag_diagram.png` ($1472 \times 352\text{ px}$) with true closed rectangular boxes, routed arrow lines, and clean typography.

---

### Example 2: Landscape & Scene Art
**Input file:** `samples/house.txt`
```text
            /\                                   \ | /
           /  \                                 -- O --
          /    \                                 / | \
         /      \
        /        \
       /          \
      /            \
     /______________\
      | .--.  .--. |                  .@@@@.
      | |  |  |  | |                .@@@@@@@@.
      | '--'  '--' |               @@@@@@@@@@@@
      |    ____    |               @@@@@@@@@@@@
      |   |    |   |                '@@@@@@@@'
      |   |  o |   |                   ||||
      |___|____|___|                   ||||
==========================================================
```

**Command:**
```powershell
python cli.py samples/house.txt -o out/
```

**Console Output:**
```
========================================================
  ASCII to Image Generator (ascii2image v1.0.0)
========================================================
Input:       house.txt (16 lines)
Classified:  ART [art:tone] (90.4% confidence)
Explanation: ASCII art illustration/scene: contains drawn geometry (roof/windows) with 20.8% shading (e.g. foliage) and no relational arrows.
--------------------------------------------------------
Mode:        art:tone
Theme:       dark
Scale:       2.0x
--------------------------------------------------------
Files successfully generated:
  -> out/house_reconstructed.png (120.5 KB)
  -> out/house_crisp.svg (1.9 KB)
========================================================
```
* **Result:** Automatically identifies the drawing as an art scene (due to high shading and zero relational arrows). Generates `house_reconstructed.png`, where the tree canopy is mathematically smoothed into a continuous foliage crown, the trunk is solid, the roof lines are crisp, and the sun radiates cleanly.

---

### Example 3: Continuous-Tone Photographic Shading
**Input file:** `samples/shading_bourke.txt` (Paul Bourke 70-character grayscale ramp)
```text
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$    
  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  
 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 
$$$$$$$$$$$$$$$$$$@@@@@@$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$@@@@%%%%@@@@$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$@@%%##****##%%@@$$$$$$$$$$$$$$$$
$$$$$$$$$$@@%#**++====++*#%@@$$$$$$$$$$$$$$$
$$$$$$$$$@%#*+=--::::--=+*#%@$$$$$$$$$$$$$$$
$$$$$$$$$@%#*+=--:....:--=+*#%@$$$$$$$$$$$$$
$$$$$$$$$@%#*+=--:....:--=+*#%@$$$$$$$$$$$$$
$$$$$$$$$@%#*+=--::::--=+*#%@$$$$$$$$$$$$$$$
$$$$$$$$$$@@%#**++====++*#%@@$$$$$$$$$$$$$$$
$$$$$$$$$$$$@@%%##****##%%@@$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$@@@@%%%%@@@@$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$@@@@@@$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 
  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$    
```

**Command:**
```powershell
python cli.py samples/shading_bourke.txt --reconstruct -o out/
```
* **Result:** Produces `shading_bourke_reconstructed.png`, resolving the concentric density levels into a smooth, spherical shaded 3D gradient with zero character-grid seams.

---

### Example 4: Line-Art Cartoon Sketch
**Input file:** `samples/cat.txt`
```text
       /\_____/\
      /  o   o  \
     ( ==  ^  == )
      )         (
     (           )
    ( (  )   (  ) )
   (__(__)___(__)__)
```

**Command:**
```powershell
# Vector Bézier contour mode
python cli.py samples/cat.txt -m art:vector -o out/

# Or Crisp Monospace Typography mode with retro green CRT theme
python cli.py samples/cat.txt -m art:crisp -t terminal_green -o out/
```
* **Result:** `cat_vector.svg` traces the cat's ears, whiskers, and paws into smooth cubic Bézier SVG paths. `cat_crisp.png` renders it with anti-aliased monospace typography in phosphor green.

---

## Quick Start

### 1. Using the Command-Line Interface (`cli.py`)

Run the CLI using your Python environment:

```powershell
# 1. View the full interactive manual and help
python cli.py help

# 2. Convert any ASCII diagram (auto-detects and exports SVG + PNG)
python cli.py samples/flowchart.txt

# 3. Choose a styling theme (light, dark, blueprint, tinted, terminal_green, amber)
python cli.py samples/network.txt --theme blueprint -o out/

# 4. Ultra-High-Definition 3.0x print export
python cli.py samples/flowchart.txt --scale 3.0 -o out/

# 5. Photographic reconstruction of shaded ASCII art (Bilateral + CLAHE)
python cli.py samples/house.txt --reconstruct -o out/

# 6. Convert line art into smooth vector Bézier curves
python cli.py samples/cat.txt -m art:vector -o out/

# 7. Print detailed 6-feature classification scores
python cli.py samples/cc_diag.txt -v
```

#### CLI Options Reference:
| Option | Argument | Description | Default |
|---|---|---|---|
| `input` | `path` | Path to ASCII text file (or `help` to show manual) | Required |
| `-o, --outdir` | `path` | Output directory to save generated files | `./out` |
| `-m, --mode` | `choice` | `auto`, `diagram`, `art:crisp`, `art:reconstruct`, `art:vector` | `auto` |
| `-t, --theme` | `choice` | `light`, `dark`, `blueprint`, `tinted`, `terminal_green`, `amber` | Contextual |
| `-s, --scale` | `float` | PNG supersampling zoom factor ($1.0\times$, $2.0\times$, $3.0\times$) | `2.0` |
| `--engine` | `choice` | Diagram engine: `auto`, `svgbob`, `aafigure` | `auto` |
| `--reconstruct` | flag | Force 5-stage photographic reconstruction | Off |
| `--width` | `int` | Target pixel width for photographic reconstruction | `1200` |
| `--denoise` | `float` | Bilateral filter smoothing strength factor | `1.0` |
| `--clahe` | `float` | CLAHE local contrast clip limit | `2.0` |
| `--format` | `choice` | Output format: `both` (SVG + PNG), `svg`, or `png` | `both` |
| `-v, --verbose` | flag | Print diagnostic detection scores and font metrics | Off |

---

### 2. Using the Interactive Local Web App

Start the built-in Flask web server:

```powershell
python web/app.py
```

Then open your browser at **`http://127.0.0.1:5000`**.

Features:
* **One-Click Sample Loader:** Select from pre-loaded architecture flowcharts, network topologies, cat sketches, or house scenes.
* **Real-Time Classification Badge:** Shows category (`DIAGRAM` vs `ART`), subtype, confidence percentage, and structural explanation.
* **Interactive Controls:** Toggle modes, visual themes, DPI scale, and denoise sliders with live preview.
* **Side-by-Side Inspector:** View the rendered image preview alongside the raw SVG vector code.
* **One-Click Downloads:** Download the final SVG and PNG directly to your computer.

---

## Python Library API

You can also import and use `ascii2image` as a Python module in your own scripts:

```python
from ascii2image import (
    classify_ascii,
    diagram_to_svg,
    render_art_crisp_svg,
    render_art_crisp_png,
    reconstruct_art_photo,
    vectorize_line_art_svg,
    svg_to_png
)

text = """
+---------+       +---------+
| Client  | ----> | Server  |
+---------+       +---------+
"""

# 1. Classify ASCII input
result = classify_ascii(text)
print(f"Type: {result.category} [{result.subtype}], Confidence: {result.confidence:.1%}")

# 2. Render diagram to SVG
svg_data = diagram_to_svg(text, theme="blueprint")

# 3. Export SVG to high-DPI PNG bytes
png_bytes = svg_to_png(svg_data, scale=2.0)
with open("diagram.png", "wb") as f:
    f.write(png_bytes)

# 4. Photographic reconstruction of shaded art
art_text = open("samples/house.txt").read()
gray_image = reconstruct_art_photo(art_text, target_width=1200)
# Returns a standard OpenCV uint8 numpy array (H x W)
```

---

## Theme Palette Reference

| Theme Name | Background | Stroke / Foreground | Fill | Best For |
|---|---|---|---|---|
| `light` | `#ffffff` | `#1e293b` (Slate 800) | `#f8fafc` | Technical documentation, white papers |
| `dark` | `#0f172a` (Slate 900) | `#38bdf8` (Sky 400) | `#1e293b` | Modern dark-mode IDEs, presentations |
| `blueprint` | `#1e3a8a` (CAD Blue) | `#93c5fd` (Soft Blue) | `#172554` | Engineering & network architecture |
| `tinted` | `#fafafa` (Warm Gray) | `#334155` (Slate 700) | `#e0f2fe` | Clean product blogs & infographics |
| `terminal_green` | `#051005` (Deep Black) | `#22c55e` (Phosphor) | `#051005` | Retro CRT aesthetic, hacker line-art |
| `amber` | `#120a00` (Dark Choc) | `#f59e0b` (Amber CRT) | `#120a00` | Warm retro terminal aesthetic |

---

## Project Architecture

```
ASCII to Image Generator/
├── .gitignore          # Ignores venvs, cache, and output folders
├── cli.py              # User-facing CLI with built-in manual
├── README.md           # Documentation & guides
├── requirements.txt    # Project dependencies
├── ascii2image/        # Core universal package
│   ├── __init__.py     # Package exports
│   ├── detect.py       # 6-feature topological & statistical classifier
│   ├── ink.py          # Exact photometric fixed-cell calibration & sub-cell atlas
│   ├── diagram.py      # Diagram parsing (svgbob / aafigure) + themes
│   ├── art.py          # Crisp typography, 5-stage reconstruction & Bézier contours
│   └── export.py       # resvg_py high-DPI SVG -> PNG rasterizer
├── samples/            # Verified sample ASCII inputs (.txt)
│   ├── bright.txt
│   ├── cat.txt
│   ├── cc_diag.txt
│   ├── flowchart.txt
│   ├── house.txt
│   ├── network.txt
│   ├── portrait.txt
│   └── shading_bourke.txt
├── tests/              # Pytest test suite (26 unit tests)
└── web/                # Local Flask application & interactive UI
    ├── app.py
    ├── templates/index.html
    └── static/ (style.css, app.js)
```

---

## Running Automated Tests

The test suite covers photometric monotonicity, tight-bbox prevention, 6-feature classification, diagram rendering, reconstruction, and web API endpoints:

```powershell
python -m pytest tests -v
```

All 26 unit tests run in ~1.1 seconds completely offline.
