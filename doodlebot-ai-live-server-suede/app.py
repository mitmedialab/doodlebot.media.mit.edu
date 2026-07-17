#!/usr/bin/env python3
"""
Sketch Server
  /          -> phone drawing/text interface
  /gallery   -> curator: moderation queue + combine
  /display   -> big screen: approved sketches only
  /stream    -> SSE event stream
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .config import ADMIN_TOKEN, OPENAI_API_KEY, GEMINI_API_KEY
from . import (
    pages,
    stream,
    submit,
    moderation,
    sketches,
    models,
    presets,
    combine,
    vectorize,
    robots,
    v2,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mitmedialab.github.io", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for module in (
    pages,
    stream,
    submit,
    moderation,
    sketches,
    models,
    presets,
    combine,
    vectorize,
    robots,
    v2,
):
    app.include_router(module.router)


if __name__ == "__main__":
    print("=" * 60)
    print("  Sketch Server")
    print("=" * 60)
    print(f"  Draw page    : http://localhost:5000/")
    print(f"  Gallery page : http://localhost:5000/gallery")
    print(f"  Display page : http://localhost:5000/display")
    if ADMIN_TOKEN == "test":
        print("  ⚠  WARNING: Using default ADMIN_TOKEN!")
    else:
        print(f"  Admin token  : {ADMIN_TOKEN}")
    print(f"  OpenAI key   : {'set ✓' if OPENAI_API_KEY else 'NOT SET ✗'}")
    print(f"  Gemini key   : {'set ✓' if GEMINI_API_KEY else 'NOT SET ✗'}")
    print("=" * 60)

    print()
    uvicorn.run(app, host="0.0.0.0", port=5000)
