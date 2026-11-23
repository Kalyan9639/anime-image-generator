"""
LLM Scene Planner & Knowledge Bridge (Track 1 — Groq, Ollama Cloud, & Local Ollama).

Uses `openai/gpt-oss-20b` on Groq (or Ollama `gpt-oss:20b-cloud` / local) strictly
as a TEXT KNOWLEDGE & SCENE DECOMPOSITION MODEL—never generating Python, Blender,
OpenSCAD, or SVG code. All geometry and optics are evaluated by our pre-written math engine.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .parser import AsciiLayout, parse_semantic_ascii
from .schema import (
    BackgroundSpec,
    CameraSpec,
    EntitySpec,
    EnvironmentSpec,
    LightingSpec,
    MaterialSpec,
    ModifierSpec,
    PartNode,
    SceneGraph,
    VALID_ATTACHMENTS,
    VALID_BG_STYLES,
    VALID_CSG_OPS,
    VALID_ENV_KINDS,
    VALID_MODIFIER_KINDS,
    VALID_PATTERNS,
    VALID_PLACEMENTS,
    VALID_PRIMITIVES,
    get_ollama_scene_json_schema,
    scene_from_dict,
)
from .solver import ResolvedScene, solve_scene


DEFAULT_PROVIDER = "groq"
DEFAULT_GROQ_HOST = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_OLLAMA_HOST = "https://ollama.com"
DEFAULT_OLLAMA_MODEL = "gpt-oss:20b-cloud"


class PlannerAPIError(RuntimeError):
    """Raised when the LLM endpoint (Groq / Ollama) returns an HTTP or network error."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class PlanResult:
    scene: SceneGraph
    resolved: ResolvedScene
    source: str                      # "groq" | "ollama" | "fallback"
    model_used: str = ""
    attempts: int = 1
    warnings: List[str] = field(default_factory=list)


def load_env_config(project_root: Optional[Path] = None) -> Dict[str, str]:
    """Reads Groq & Ollama configuration from environment variables or `.env`."""
    root = project_root or Path(__file__).resolve().parent.parent.parent
    env_file = root / ".env"
    file_vals: Dict[str, str] = {}
    if env_file.is_file():
        try:
            for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                k, v = stripped.split("=", 1)
                file_vals[k.strip()] = v.strip().strip('"').strip("'")
        except Exception:
            pass

    def _get(key: str, default: str = "") -> str:
        return os.environ.get(key) or file_vals.get(key, default)

    provider = _get("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if provider not in ("groq", "ollama"):
        provider = DEFAULT_PROVIDER

    groq_api_key = _get("GROQ_API_KEY", "")
    groq_host = _get("GROQ_HOST", DEFAULT_GROQ_HOST)
    groq_model = _get("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    ollama_api_key = _get("OLLAMA_API_KEY", "")
    ollama_host = _get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    ollama_model = _get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    if provider == "groq":
        active_key, active_host, active_model = groq_api_key, groq_host, groq_model
    else:
        active_key, active_host, active_model = ollama_api_key, ollama_host, ollama_model

    return {
        "provider": provider,
        "api_key": active_key,
        "host": active_host,
        "model": active_model,
        "groq_api_key": groq_api_key,
        "groq_host": groq_host,
        "groq_model": groq_model,
        "ollama_api_key": ollama_api_key,
        "ollama_host": ollama_host,
        "ollama_model": ollama_model,
    }


SYSTEM_PROMPT = f"""You are a 3D visual knowledge & scene decomposition architect.
Your ONLY job is to convert the user's description (and optional ASCII spatial layout) into a structured JSON SceneGraph that our pre-written mathematical rendering engine will synthesize into an image.

STRICT RULES:
1. NEVER write Python, SVG, GLSL, or code. Output ONLY valid JSON matching the schema.
2. COMPLETE OBJECT COVERAGE (CRITICAL):
   - Every single subject, prop, and environment item mentioned in the prompt MUST appear as its own entry in `entities` with distinct `placement` (`left`, `center`, `right`, `foreground_left`, `foreground_right`, `background_left`, `background_right`).
   - NEVER omit requested objects! If the prompt mentions 3 objects (e.g., lantern + mug + pine tree), output 3 distinct entities.
3. MATCH BACKGROUND & ENVIRONMENT EXACTLY TO PROMPT:
   - Choose `background.style` from {list(VALID_BG_STYLES)}:
     * Twilight / sunset / gradient sky (WITHOUT bokeh requested): use `twilight_glow` (or `gradient`) with `top_color` like `#1e1b4b` (deep twilight indigo), `bottom_color` `#fb7185` or `#f97316` (sunset glow), `glow_color: "#fde047"`, `bokeh_count: 0`, `particle_type: "stars"`, `particle_count: 24`.
     * Night / festive / bokeh lights requested: use `bokeh_night` with dark sky (`#0b1120` to `#1e293b`), `bokeh_count: 65`, warm `bokeh_colors: ["#fde68a", "#fbbf24", "#f59e0b"]`, `bokeh_size: 1.0`.
   - Choose `environment.kind` from {list(VALID_ENV_KINDS)} with matching surface pattern (`grid_tiles` or `stone_voronoi` for stone/patio/ledge, `wood_grain` for wooden shelf/table, `circuit_traces` for PCB).
4. PRE-WRITTEN PRIMITIVES, ATTACHMENTS, MODIFIERS & PATTERNS:
   - Primitives: {list(VALID_PRIMITIVES)}
   - Attachments: {list(VALID_ATTACHMENTS)}
   - Entity placements: {list(VALID_PLACEMENTS)}
   - Modifiers: {list(VALID_MODIFIER_KINDS)} (`array_grid`, `array_linear`, `array_radial`, `mirror_x`, `surface_displace`)
   - Patterns: {list(VALID_PATTERNS)}
   - CSG ops: {list(VALID_CSG_OPS)}
5. RECOGNIZABLE 3D PART RECIPES (DO NOT use random radial white cones for trees or spheres for crystals):
   - **Pine Tree**: dark-brown `cylinder` trunk (`#5c3a21`, `size: [0.24, 0.55, 0.24]`) + 3 stacked evergreen `cone` tiers (`#15803d`, `#166534`, `params: {{"bottom_ratio": 1.0, "top_ratio": 0.03}}`, sizes `[1.1, 0.65, 1.1]`, `[0.85, 0.55, 0.85]`, `[0.55, 0.45, 0.55]`, each `on_top_of` targeting the previous part).
   - **Crystal Cluster**: 3–5 angled `cone` or faceted `superquadric` shards (`e1: 0.2, e2: 0.2`, sizes around `[0.25, 0.9, 0.25]` with tilted `rotation_deg` like `[14, 0, -18]`, `[-15, 25, 20]`, `[5, -15, 12]`), icy/amethyst colors (`#7dd3fc`, `#c084fc`, `#a855f7`), `roughness: 0.12`, `emissive: 0.35`.
   - **Brass Lantern**: dark-bronze `rounded_box` base (`#78350f`, `metallic: 0.85`) + glowing amber `sphere` core (`#fde047`, `emissive: 2.2`, `on_top_of` base) + wide-brim brass `cone` cap (`#d97706`, `on_top_of` core).
   - **Ceramic Mug**: `lathe_profile` cup (`params: {{"profile_radii": [0.68, 0.72, 0.76, 0.80], "hollow": true}}`, `size: [0.55, 0.55, 0.55]`, `#e2e8f0` or patterned `cable_knit`) + `torus` handle (`right_of` cup, `rotation_deg: [0, 0, 90]`, `size: [0.22, 0.22, 0.06]`).
   - Other primitives: `gielis_supershape` (stars/flowers), `bezier_tube` (arches/handles/cables), `convolution_limb` (curved organic forms), `lsystem_branch` (bare tree branches/coral).
6. FRAMING, PROPORTIONS & TARGETING:
   - Keep `scale` between `0.75` and `1.3` and main part sizes around `0.45` to `1.3` world units so objects are clearly legible.
   - ALWAYS set `"target"` to the exact `"name"` of the earlier part when using `on_top_of`, `inside_center`, `left_of`, `right_of`, etc.
   - Keep `camera.elevation_deg` between `12` and `22` (unless isometric), `particle_count` between `15` and `40`.
"""


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Robustly extracts a JSON object from model output (handles optional markdown fences)."""
    s = raw_text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        parsed = json.loads(s[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Model response did not contain a valid JSON object.")


def _call_ollama_chat(
    messages: List[Dict[str, str]],
    *,
    host: str,
    model: str,
    api_key: str,
    schema: Dict[str, Any],
    temperature: float = 0.15,
    timeout: int = 90,
) -> str:
    base = host.rstrip("/")
    url = f"{base}/api/chat" if not base.endswith("/api") else f"{base}/chat"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": schema,
        "options": {"temperature": float(temperature)},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace").strip()
        if exc.code == 402:
            raise PlannerAPIError(
                f"Ollama Cloud returned HTTP 402 (Payment/Credits Required): {err_body}. "
                "Add credits at https://ollama.com/plans or switch Ollama Host to local http://localhost:11434.",
                status_code=402,
            ) from exc
        if exc.code in (401, 403):
            raise PlannerAPIError(
                f"Ollama authentication failed (HTTP {exc.code}): {err_body}. Verify OLLAMA_API_KEY.",
                status_code=exc.code,
            ) from exc
        raise PlannerAPIError(f"Ollama HTTP {exc.code}: {err_body}", status_code=exc.code) from exc
    except Exception as exc:
        raise PlannerAPIError(f"Could not reach Ollama endpoint {url}: {exc}") from exc

    msg = body.get("message") or {}
    content = msg.get("content", "")
    if not content:
        raise PlannerAPIError(f"Empty response from Ollama model '{model}'.")
    return str(content)


def _call_groq_chat(
    messages: List[Dict[str, str]],
    *,
    host: str,
    model: str,
    api_key: str,
    schema: Dict[str, Any],
    temperature: float = 0.15,
    timeout: int = 90,
) -> str:
    """
    Calls Groq's OpenAI-compatible Chat Completions endpoint using structured
    `json_schema` output (with automatic fallback to `json_object` if a model
    does not support `json_schema`).
    """
    if not api_key or not api_key.strip():
        raise PlannerAPIError(
            "Missing GROQ_API_KEY. Provide your Groq API key in Settings or `.env`.",
            status_code=401,
        )

    base = host.rstrip("/")
    if base.endswith("/chat/completions"):
        url = base
    elif base.endswith("/v1"):
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/openai/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
        # Cloudflare WAF blocks the default Python-urllib User-Agent with HTTP 403/1010
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    def _post(resp_format: Dict[str, Any], msgs: List[Dict[str, str]]) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "temperature": float(temperature),
            "max_completion_tokens": 8192,
            "response_format": resp_format,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
        choices = body.get("choices") or []
        if not choices:
            raise PlannerAPIError(f"Empty choices in Groq response for model '{model}'.")
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
        if not content:
            raise PlannerAPIError(f"Empty response content from Groq model '{model}'.")
        return str(content)

    schema_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "scene_graph",
            "strict": False,
            "schema": schema,
        },
    }

    try:
        return _post(schema_format, messages)
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace").strip()
        # Some Groq models reject json_schema; transparently retry once with json_object
        if exc.code == 400 and ("json_schema" in err_body.lower() or "response_format" in err_body.lower()):
            fallback_msgs = list(messages)
            fallback_msgs[0] = {
                "role": "system",
                "content": (
                    fallback_msgs[0]["content"]
                    + "\n\nOUTPUT JSON SCHEMA (ALL top-level keys including environment and entities are REQUIRED):\n"
                    + json.dumps(schema)
                ),
            }
            try:
                return _post({"type": "json_object"}, fallback_msgs)
            except urllib.error.HTTPError as exc2:
                err_body2 = exc2.read().decode("utf-8", errors="replace").strip()
                raise PlannerAPIError(f"Groq HTTP {exc2.code}: {err_body2}", status_code=exc2.code) from exc2
            except Exception as exc2:
                raise PlannerAPIError(f"Could not reach Groq endpoint {url}: {exc2}") from exc2

        if exc.code in (401, 403):
            raise PlannerAPIError(
                f"Groq authentication/access failed (HTTP {exc.code}): {err_body}. Verify GROQ_API_KEY.",
                status_code=exc.code,
            ) from exc
        if exc.code == 429:
            raise PlannerAPIError(
                f"Groq rate limit reached (HTTP 429): {err_body}. Wait a moment and retry.",
                status_code=429,
            ) from exc
        raise PlannerAPIError(f"Groq HTTP {exc.code}: {err_body}", status_code=exc.code) from exc
    except PlannerAPIError:
        raise
    except Exception as exc:
        raise PlannerAPIError(f"Could not reach Groq endpoint {url}: {exc}") from exc


def build_offline_fallback_scene(
    prompt: str = "",
    ascii_text: str = "",
) -> SceneGraph:
    """
    Pure deterministic fallback (zero AI, zero object-name hardcoding) when Ollama
    is offline or out of credits: turns any Semantic ASCII boxes directly into
    beveled 3D math geometry with explicit grid counts (`x4`, `16x16`) and layout!
    """
    layout: Optional[AsciiLayout] = parse_semantic_ascii(ascii_text) if ascii_text.strip() else None
    palette = ["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f472b6", "#a78bfa"]

    entities: List[EntitySpec] = []
    if layout and layout.boxes:
        top_boxes = [b for b in layout.boxes if b.parent_id is None]
        by_id = {b.id: b for b in layout.boxes}
        for idx, tb in enumerate(top_boxes):
            w_norm = max(0.15, tb.norm_bbox[2] - tb.norm_bbox[0])
            h_norm = max(0.15, tb.norm_bbox[3] - tb.norm_bbox[1])
            base_w = max(0.7, w_norm * 4.0)
            base_d = max(0.7, h_norm * 3.0)
            parts: List[PartNode] = [
                PartNode(
                    name="base_slab",
                    primitive="rounded_box",
                    size=(base_w, 0.28, base_d),
                    attachment="base_center",
                    material=MaterialSpec(
                        color=palette[idx % len(palette)],
                        secondary_color="#1e293b",
                        accent_color="#fbbf24",
                        pattern="circuit_traces" if tb.grid_repeat or tb.children_ids else "brushed_metal",
                        pattern_scale=3.0,
                        metallic=0.35,
                        roughness=0.4,
                    ),
                )
            ]
            if tb.grid_repeat:
                nx, nz = min(14, tb.grid_repeat[0]), min(14, tb.grid_repeat[1])
                sp_x = (base_w * 0.78) / max(1, nx)
                sp_z = (base_d * 0.78) / max(1, nz)
                parts.append(
                    PartNode(
                        name="grid_array",
                        primitive="sphere",
                        size=(min(sp_x, sp_z) * 0.65,) * 3,
                        attachment="underneath",
                        target="base_slab",
                        modifiers=[
                            ModifierSpec(
                                kind="array_grid",
                                params={"nx": nx, "nz": nz, "spacing_x": sp_x, "spacing_z": sp_z},
                            )
                        ],
                        material=MaterialSpec(color="#e2e8f0", metallic=0.85, roughness=0.2),
                    )
                )
            for c_i, cid in enumerate(tb.children_ids):
                cb = by_id[cid]
                cw = max(0.20, (cb.norm_bbox[2] - cb.norm_bbox[0]) * 3.2 * 0.86)
                cd = max(0.20, (cb.norm_bbox[3] - cb.norm_bbox[1]) * 2.4 * 0.86)
                rel_cx = ((cb.norm_bbox[0] + cb.norm_bbox[2]) * 0.5 - (tb.norm_bbox[0] + tb.norm_bbox[2]) * 0.5) * 3.8
                rel_cz = ((cb.norm_bbox[1] + cb.norm_bbox[3]) * 0.5 - (tb.norm_bbox[1] + tb.norm_bbox[3]) * 0.5) * 2.8
                mods: List[ModifierSpec] = []
                part_w = cw
                if cb.count > 1:
                    cnt = min(8, cb.count)
                    sp = cw / max(1, cnt)
                    part_w = sp * 0.68
                    mods.append(ModifierSpec(kind="array_linear", params={"axis": "x", "count": cnt, "spacing": sp}))
                parts.append(
                    PartNode(
                        name=f"child_{c_i}",
                        primitive="superquadric",
                        size=(part_w, 0.24, cd),
                        attachment="on_top_of",
                        target="base_slab",
                        offset=(rel_cx, 0.0, rel_cz),
                        params={"e1": 0.25, "e2": 0.25},
                        modifiers=mods,
                        material=MaterialSpec(
                            color=palette[(idx + c_i + 1) % len(palette)],
                            secondary_color="#0f172a",
                            pattern="brushed_metal",
                            metallic=0.55,
                            roughness=0.3,
                        ),
                    )
                )
            entities.append(
                EntitySpec(
                    id=tb.id,
                    label=tb.label,
                    role="subject" if idx == 0 else "object",
                    placement="center",
                    ascii_bbox=tb.norm_bbox,
                    parts=parts,
                )
            )
    else:
        # Generic geometric sculpture placeholder when neither LLM nor ASCII boxes exist
        entities.append(
            EntitySpec(
                id="fallback_sculpture",
                label=prompt[:40] or "Math Sculpture",
                role="subject",
                placement="center",
                parts=[
                    PartNode(
                        name="pedestal",
                        primitive="cylinder",
                        size=(1.4, 0.25, 1.4),
                        attachment="base_center",
                        material=MaterialSpec(color="#475569", pattern="stone_voronoi", roughness=0.7),
                    ),
                    PartNode(
                        name="supershape_form",
                        primitive="gielis_supershape",
                        size=(1.1, 1.1, 1.1),
                        attachment="on_top_of",
                        target="pedestal",
                        params={"m": 6.0, "n1": 1.0, "n2": 1.7, "n3": 1.7},
                        material=MaterialSpec(color="#38bdf8", metallic=0.45, roughness=0.25),
                    ),
                ],
            )
        )

    view_attr = layout.global_attributes.get("view", "") if layout else ""
    proj = "isometric" if view_attr.lower() in ("iso", "isometric") else "perspective"
    return SceneGraph(
        title=(prompt[:48] or (layout.boxes[0].label if layout and layout.boxes else "ASCII 3D Scene")),
        description="Offline deterministic mathematical synthesis (no LLM call).",
        camera=CameraSpec(projection=proj, azimuth_deg=32.0, elevation_deg=26.0, dof_strength=0.15),
        lighting=LightingSpec(rim_strength=0.55, bloom_intensity=0.25),
        background=BackgroundSpec(style="studio", top_color="#0f172a", bottom_color="#1e293b", glow_color="#38bdf8"),
        environment=EnvironmentSpec(enabled=True, kind="ground_plane", pattern="grid_tiles", primary_color="#334155", secondary_color="#1e293b"),
        entities=entities,
    )


def plan_scene(
    prompt: str,
    *,
    ascii_text: str = "",
    provider: Optional[str] = None,
    host: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_retries: int = 2,
    allow_fallback: bool = False,
    timeout: int = 90,
) -> PlanResult:
    """
    Calls the configured LLM provider (`groq` with `openai/gpt-oss-20b`, or `ollama`)
    to synthesize a dynamic multi-layer SceneGraph JSON, runs our pre-written
    Constraint Solver + Geometry Validator, and retries up to `max_retries` times
    if geometric issues or empty entity lists occur.
    """
    cfg = load_env_config()
    eff_provider = (provider or cfg["provider"] or DEFAULT_PROVIDER).strip().lower()
    if eff_provider not in ("groq", "ollama"):
        eff_provider = DEFAULT_PROVIDER

    if eff_provider == "groq":
        eff_host = (host or cfg["groq_host"] or DEFAULT_GROQ_HOST).strip()
        eff_model = (model or cfg["groq_model"] or DEFAULT_GROQ_MODEL).strip()
        eff_key = api_key if api_key is not None else cfg["groq_api_key"]
    else:
        eff_host = (host or cfg["ollama_host"] or DEFAULT_OLLAMA_HOST).strip()
        eff_model = (model or cfg["ollama_model"] or DEFAULT_OLLAMA_MODEL).strip()
        eff_key = api_key if api_key is not None else cfg["ollama_api_key"]

    ascii_layout: Optional[AsciiLayout] = parse_semantic_ascii(ascii_text) if ascii_text.strip() else None

    user_sections = []
    if prompt.strip():
        user_sections.append(f"USER SCENE PROMPT:\n{prompt.strip()}")
    if ascii_layout and ascii_layout.boxes:
        user_sections.append(
            "SEMANTIC ASCII LAYOUT (respect these boxes, nesting, counts, and attributes):\n"
            + json.dumps(ascii_layout.to_summary_dict(), indent=2)
        )
    elif ascii_text.strip():
        user_sections.append(f"ASCII REFERENCE:\n{ascii_text.strip()}")

    if not user_sections:
        fallback = build_offline_fallback_scene("", "")
        return PlanResult(scene=fallback, resolved=solve_scene(fallback), source="fallback", warnings=["Empty prompt."])

    user_sections.append(
        "Return a rich, complete multi-layer SceneGraph JSON (background + environment + all entities & detailed parts)."
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_sections)},
    ]
    schema = get_ollama_scene_json_schema()
    warnings: List[str] = []

    last_scene: Optional[SceneGraph] = None
    last_resolved: Optional[ResolvedScene] = None

    for attempt in range(1, max_retries + 2):
        try:
            if eff_provider == "groq":
                raw_content = _call_groq_chat(
                    messages,
                    host=eff_host,
                    model=eff_model,
                    api_key=eff_key,
                    schema=schema,
                    temperature=0.15,
                    timeout=timeout,
                )
            else:
                raw_content = _call_ollama_chat(
                    messages,
                    host=eff_host,
                    model=eff_model,
                    api_key=eff_key,
                    schema=schema,
                    temperature=0.15,
                    timeout=timeout,
                )

            parsed = _extract_json_object(raw_content)
            scene = scene_from_dict(parsed)

            # Lock top-level ASCII box positions if an ASCII layout was provided
            if ascii_layout and ascii_layout.boxes:
                top_boxes = [b for b in ascii_layout.boxes if b.parent_id is None]
                for i, ent in enumerate(scene.entities):
                    if ent.ascii_bbox is None and i < len(top_boxes):
                        ent.ascii_bbox = top_boxes[i].norm_bbox

            resolved = solve_scene(scene)
            diagnostics = list(resolved.diagnostics)
            if not scene.entities:
                diagnostics.append("The `entities` array was empty; you MUST populate 1 or more detailed entities with parts.")

            last_scene = scene
            last_resolved = resolved

            if not diagnostics or attempt > max_retries:
                if diagnostics:
                    warnings.extend(diagnostics)
                return PlanResult(
                    scene=scene,
                    resolved=resolved,
                    source=eff_provider,
                    model_used=eff_model,
                    attempts=attempt,
                    warnings=warnings,
                )

            # Feed plain-English geometry errors back to the model for refinement
            messages.append({"role": "assistant", "content": raw_content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Our geometric solver detected these layout issues:\n- "
                        + "\n- ".join(diagnostics)
                        + "\nFix ONLY those placements/attachments/sizes and return the corrected complete JSON."
                    ),
                }
            )
        except PlannerAPIError as exc:
            if allow_fallback:
                warnings.append(str(exc))
                fallback = build_offline_fallback_scene(prompt, ascii_text)
                return PlanResult(
                    scene=fallback,
                    resolved=solve_scene(fallback),
                    source="fallback",
                    model_used=eff_model,
                    attempts=attempt,
                    warnings=warnings,
                )
            raise
        except Exception as exc:
            warnings.append(f"Attempt {attempt} parse error: {exc}")
            if attempt > max_retries:
                if last_scene is not None and last_resolved is not None:
                    return PlanResult(
                        scene=last_scene,
                        resolved=last_resolved,
                        source=eff_provider,
                        model_used=eff_model,
                        attempts=attempt,
                        warnings=warnings,
                    )
                if allow_fallback:
                    fallback = build_offline_fallback_scene(prompt, ascii_text)
                    return PlanResult(
                        scene=fallback,
                        resolved=solve_scene(fallback),
                        source="fallback",
                        model_used=eff_model,
                        attempts=attempt,
                        warnings=warnings,
                    )
                raise PlannerAPIError(f"Failed to parse valid SceneGraph after {attempt} attempts: {exc}") from exc
            messages.append(
                {
                    "role": "user",
                    "content": f"Your previous output could not be parsed ({exc}). Return ONLY valid JSON matching the schema.",
                }
            )

    fallback = build_offline_fallback_scene(prompt, ascii_text)
    return PlanResult(scene=fallback, resolved=solve_scene(fallback), source="fallback", warnings=warnings)


def plan_scene_with_ollama(
    prompt: str,
    *,
    ascii_text: str = "",
    host: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_retries: int = 2,
    allow_fallback: bool = False,
    timeout: int = 90,
) -> PlanResult:
    """Backward-compatible wrapper targeting the Ollama provider."""
    return plan_scene(
        prompt,
        ascii_text=ascii_text,
        provider="ollama",
        host=host,
        model=model,
        api_key=api_key,
        max_retries=max_retries,
        allow_fallback=allow_fallback,
        timeout=timeout,
    )
