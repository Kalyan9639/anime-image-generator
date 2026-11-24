"""Console entry point for `prism-studio`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import PRESET_SCENES, build_offline_director_scene, generate_image, render_scene_dna, scene_dna_from_dict


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prism-studio",
        description="PrismStudio: Pure-math compositional natural-language-to-image generator.",
    )
    p.add_argument("prompt", nargs="?", default="", help="Natural language prompt describing the scene.")
    p.add_argument("--preset", type=str, default=None, help="Load a built-in SceneDNA preset directly (zero LLM call).")
    p.add_argument("--list-presets", action="store_true", help="Print available preset names and exit.")
    p.add_argument("--style", choices=["anime_npr", "stylized_pbr"], default="anime_npr", help="Visual shader mode.")
    p.add_argument("--size", type=str, default="960x720", help="Output WxH resolution (default: 960x720).")
    p.add_argument("--backend", choices=["auto", "gpu", "cpu"], default="auto", help="Rendering backend.")
    p.add_argument("--provider", choices=["groq", "ollama"], default=None, help="LLM provider override.")
    p.add_argument("--model", type=str, default=None, help="Text LLM model override.")
    p.add_argument("--vlm-critique", action="store_true", help="Enable 320x240 VLM draft self-critique pass.")
    p.add_argument("--offline", action="store_true", help="Force compositional offline director (no API call).")
    p.add_argument("-o", "--out", type=str, default="out/prism_render.png", help="Output PNG path.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_presets:
        for name in PRESET_SCENES:
            print(f"- {name}")
        return 0

    try:
        w_s, h_s = args.size.lower().split("x", 1)
        w, h = max(160, int(w_s)), max(120, int(h_s))
    except Exception:
        w, h = 960, 720

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")

    if args.preset:
        if args.preset not in PRESET_SCENES:
            print(f"[Error] Unknown preset '{args.preset}'. Run --list-presets.", file=sys.stderr)
            return 1
        dna = scene_dna_from_dict(PRESET_SCENES[args.preset])
        dna.style_mode = args.style
        res_out = render_scene_dna(dna, width=w, height=h, prefer_backend=args.backend)
        if res_out.backend_notice:
            print(res_out.backend_notice, file=sys.stderr)
        out_path.write_bytes(res_out.png_bytes)
        json_path.write_text(json.dumps(dna.to_dict(), indent=2), encoding="utf-8")
        print(f"Rendered preset '{args.preset}' ({res_out.backend_used}) in {res_out.render_seconds:.3f}s -> {out_path}")
        return 0

    if not args.prompt.strip():
        parser.print_help()
        return 0

    if args.offline:
        dna = build_offline_director_scene(args.prompt, style_mode=args.style)
        res_out = render_scene_dna(dna, width=w, height=h, prefer_backend=args.backend)
        if res_out.backend_notice:
            print(res_out.backend_notice, file=sys.stderr)
        out_path.write_bytes(res_out.png_bytes)
        json_path.write_text(json.dumps(dna.to_dict(), indent=2), encoding="utf-8")
        print(f"Rendered offline ({res_out.backend_used}) in {res_out.render_seconds:.3f}s -> {out_path}")
        return 0

    gen = generate_image(
        args.prompt,
        style_mode=args.style,
        width=w,
        height=h,
        use_vlm_critique=args.vlm_critique,
        prefer_backend=args.backend,
        provider=args.provider,
        model=args.model,
        allow_fallback=True,
    )
    for note in gen.notices:
        print(f"[Notice] {note}", file=sys.stderr)
    out_path.write_bytes(gen.png_bytes)
    json_path.write_text(json.dumps(gen.dna.to_dict(), indent=2), encoding="utf-8")
    print(
        f"PrismStudio ({gen.plan_result.source} | {gen.render_output.backend_used}) "
        f"completed in {gen.render_output.render_seconds:.3f}s -> {out_path}"
    )
    return 0
