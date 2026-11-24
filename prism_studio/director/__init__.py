"""LLM Scene Director & VLM Self-Critique exports for PrismStudio."""

from .llm_client import (
    DEFAULT_GROQ_HOST,
    DEFAULT_GROQ_TEXT_MODEL,
    DEFAULT_GROQ_VISION_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_TEXT_MODEL,
    DEFAULT_OLLAMA_VISION_MODEL,
    DEFAULT_PROVIDER,
    DirectorAPIError,
    call_text_llm,
    call_vision_llm,
    extract_json_object,
    load_director_env,
)
from .scene_director import (
    DIRECTOR_SYSTEM_PROMPT,
    DirectorPlanResult,
    PRESET_SCENES,
    build_offline_director_scene,
    plan_scene_dna,
)
from .vlm_critic import CritiqueResult, critique_and_patch_scene

__all__ = [
    "CritiqueResult",
    "DEFAULT_GROQ_HOST",
    "DEFAULT_GROQ_TEXT_MODEL",
    "DEFAULT_GROQ_VISION_MODEL",
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_OLLAMA_TEXT_MODEL",
    "DEFAULT_OLLAMA_VISION_MODEL",
    "DEFAULT_PROVIDER",
    "DIRECTOR_SYSTEM_PROMPT",
    "DirectorAPIError",
    "DirectorPlanResult",
    "PRESET_SCENES",
    "build_offline_director_scene",
    "call_text_llm",
    "call_vision_llm",
    "critique_and_patch_scene",
    "extract_json_object",
    "load_director_env",
    "plan_scene_dna",
]
