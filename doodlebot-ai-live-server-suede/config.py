from pathlib import Path
import os

from typing import TypedDict, Literal, TypeAlias

PENDING_DIR = Path("pending")
PENDING_DIR.mkdir(exist_ok=True)
SKETCHES_DIR = Path("sketches")
SKETCHES_DIR.mkdir(exist_ok=True)
COMBINED_DIR = Path("combined")
COMBINED_DIR.mkdir(exist_ok=True)
PRESETS_FILE = Path("presets.json")

STATIC_DIR = Path("static")
"""
Front-end HTML cache. Pages are fetched from `STATIC_HOST` on first request and cached here. 
POST `/static/bust` (or navigating to `/static/bust?token=...` in browser using `ADMIN_TOKEN`) 
clears it to pick up a redeployed front-end.
"""

STATIC_DIR.mkdir(exist_ok=True)
STATIC_HOST = os.environ.get("STATIC_HOST", "https://mitmedialab.github.io")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "test")

Provider: TypeAlias = Literal["openai", "gemini"]


class ModelInfo(TypedDict):
    label: str
    provider: Provider


MODELS: dict[str, ModelInfo] = {
    "gpt-image-1": {"label": "GPT Image 1", "provider": "openai"},
    "gpt-image-2": {"label": "GPT Image 2", "provider": "openai"},
    "gemini-2.5-flash-image": {"label": "Nano Banana Flash", "provider": "gemini"},
    "gemini-3.1-flash-image-preview": {"label": "Nano Banana 2", "provider": "gemini"},
    "gemini-3-pro-image-preview": {"label": "Nano Banana Pro", "provider": "gemini"},
}

PROMPT_PRESETS = {
    "Default": (
        "doodle creatively combining all sketches in one adding elements where the flow needs it, use words as inspiration, no words or letters in the drawing. Use simple arcs and straight lines to make the doodle."
        "Pure white background, thin clean black lines only, no fill, no shading, no color, "
        "no hatching. Style: sparse contour drawing, like a zen brushstroke illustration."
        "Keeo the drawing as minimal as possible. Minimal and less clear is better than complicated."
    ),
    "Longer Default": (
        "doodle creatively combining all sketches in one adding elements where the flow needs it, use words as inspiration, no words or letters in the drawing. Use simple arcs and straight lines to make the doodle."
        "Combine all the provided sketches into a single unified minimal line drawing. "
        "Use as few continuous strokes as possible — every line must be essential. "
        "Pure white background, thin clean black lines only, no fill, no shading, no color, "
        "no hatching. Style: sparse contour drawing, like a zen brushstroke illustration."
        "Keep the drawing as minimal as possible. Minimal and less clear is better than complicated."
    ),
    "Simple Merge": (
        "Combine all the provided sketches into one unique unified drawing. "
        "Blend the elements together creatively so they feel like a single cohesive piece, not separate drawings placed side by side. "
        "Keep it simple — use clean minimal lines, remove unnecessary detail and fine-grained detail. Less is more. "
        "Pure white background, thin clean black lines only, no fill, no shading, no color."
    ),
}
