"""
Groq & Ollama Text/Vision LLM Client for PrismStudio.

Reads `.env` configuration from `prism_studio/.env` (falling back to parent
project `.env` so existing credentials work automatically) and calls Groq
OpenAI-compatible or Ollama endpoints with structured JSON schema decoding.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_PROVIDER = "groq"
DEFAULT_GROQ_HOST = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_TEXT_MODEL = "openai/gpt-oss-20b"
DEFAULT_GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_TEXT_MODEL = "gpt-oss:20b-cloud"
DEFAULT_OLLAMA_VISION_MODEL = "qwen2.5vl:7b"


class DirectorAPIError(RuntimeError):
    """Raised when the Groq or Ollama endpoint returns an API/network failure."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def load_director_env(project_root: Optional[Path] = None) -> Dict[str, str]:
    """Loads Groq/Ollama keys from env vars, `prism_studio/.env`, or parent `.env`."""
    pkg_root = project_root or Path(__file__).resolve().parent.parent.parent
    candidates = [pkg_root / ".env", pkg_root.parent / ".env"]
    file_vals: Dict[str, str] = {}
    for env_file in candidates:
        if env_file.is_file():
            try:
                for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        continue
                    k, v = stripped.split("=", 1)
                    k = k.strip()
                    if k not in file_vals:
                        file_vals[k] = v.strip().strip('"').strip("'")
            except Exception:
                pass

    def _get(key: str, default: str = "") -> str:
        return os.environ.get(key) or file_vals.get(key, default)

    provider = _get("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if provider not in ("groq", "ollama"):
        provider = DEFAULT_PROVIDER

    return {
        "provider": provider,
        "groq_api_key": _get("GROQ_API_KEY", ""),
        "groq_host": _get("GROQ_HOST", DEFAULT_GROQ_HOST),
        "groq_text_model": _get("GROQ_TEXT_MODEL", _get("GROQ_MODEL", DEFAULT_GROQ_TEXT_MODEL)),
        "groq_vision_model": _get("GROQ_VISION_MODEL", DEFAULT_GROQ_VISION_MODEL),
        "ollama_api_key": _get("OLLAMA_API_KEY", ""),
        "ollama_host": _get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
        "ollama_text_model": _get("OLLAMA_TEXT_MODEL", _get("OLLAMA_MODEL", DEFAULT_OLLAMA_TEXT_MODEL)),
        "ollama_vision_model": _get("OLLAMA_VISION_MODEL", DEFAULT_OLLAMA_VISION_MODEL),
    }


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Robustly extracts a JSON object from LLM response text."""
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
    raise ValueError("LLM response did not contain a valid JSON object.")


def _call_groq_raw(
    messages: List[Dict[str, Any]],
    *,
    host: str,
    model: str,
    api_key: str,
    schema: Dict[str, Any],
    schema_name: str,
    temperature: float = 0.15,
    timeout: int = 90,
) -> str:
    if not api_key or not api_key.strip():
        raise DirectorAPIError("Missing GROQ_API_KEY. Set it in `.env` or UI Settings.", status_code=401)

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
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    def _post(resp_fmt: Dict[str, Any], msgs: List[Dict[str, Any]]) -> str:
        payload = {
            "model": model,
            "messages": msgs,
            "temperature": float(temperature),
            "max_completion_tokens": 6144,
            "response_format": resp_fmt,
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
        choices = body.get("choices") or []
        if not choices:
            raise DirectorAPIError(f"Empty choices returned by Groq '{model}'.")
        content = (choices[0].get("message") or {}).get("content") or ""
        if not content:
            raise DirectorAPIError(f"Empty message content returned by Groq '{model}'.")
        return str(content)

    try:
        return _post({"type": "json_schema", "json_schema": {"name": schema_name, "strict": False, "schema": schema}}, messages)
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", errors="replace").strip()
        if exc.code == 400 and ("json_schema" in err.lower() or "response_format" in err.lower()):
            fb_msgs = list(messages)
            sys_txt = fb_msgs[0]["content"] if isinstance(fb_msgs[0]["content"], str) else ""
            fb_msgs[0] = {"role": "system", "content": sys_txt + "\n\nREQUIRED JSON SCHEMA:\n" + json.dumps(schema)}
            try:
                return _post({"type": "json_object"}, fb_msgs)
            except urllib.error.HTTPError as exc2:
                raise DirectorAPIError(f"Groq HTTP {exc2.code}: {exc2.read().decode('utf-8', errors='replace')}", status_code=exc2.code) from exc2
            except Exception as exc2:
                raise DirectorAPIError(f"Groq request error: {exc2}") from exc2
        raise DirectorAPIError(f"Groq HTTP {exc.code}: {err}", status_code=exc.code) from exc
    except DirectorAPIError:
        raise
    except Exception as exc:
        raise DirectorAPIError(f"Could not reach Groq endpoint {url}: {exc}") from exc


def _call_ollama_raw(
    messages: List[Dict[str, Any]],
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
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": schema,
        "options": {"temperature": float(temperature)},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", errors="replace").strip()
        raise DirectorAPIError(f"Ollama HTTP {exc.code}: {err}", status_code=exc.code) from exc
    except Exception as exc:
        raise DirectorAPIError(f"Could not reach Ollama endpoint {url}: {exc}") from exc
    content = (body.get("message") or {}).get("content") or ""
    if not content:
        raise DirectorAPIError(f"Empty response from Ollama '{model}'.")
    return str(content)


def call_text_llm(
    messages: List[Dict[str, Any]],
    *,
    provider: str,
    host: str,
    model: str,
    api_key: str,
    schema: Dict[str, Any],
    schema_name: str = "scene_dna",
    temperature: float = 0.15,
    timeout: int = 90,
) -> str:
    if provider == "ollama":
        return _call_ollama_raw(messages, host=host, model=model, api_key=api_key, schema=schema, temperature=temperature, timeout=timeout)
    return _call_groq_raw(messages, host=host, model=model, api_key=api_key, schema=schema, schema_name=schema_name, temperature=temperature, timeout=timeout)


def call_vision_llm(
    system_prompt: str,
    user_prompt: str,
    png_bytes: bytes,
    *,
    provider: str,
    host: str,
    model: str,
    api_key: str,
    schema: Dict[str, Any],
    timeout: int = 60,
) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    if provider == "ollama":
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt, "images": [b64]},
        ]
        return _call_ollama_raw(msgs, host=host, model=model, api_key=api_key, schema=schema, temperature=0.05, timeout=timeout)

    msgs = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        },
    ]
    return _call_groq_raw(msgs, host=host, model=model, api_key=api_key, schema=schema, schema_name="scene_patch", temperature=0.05, timeout=timeout)
