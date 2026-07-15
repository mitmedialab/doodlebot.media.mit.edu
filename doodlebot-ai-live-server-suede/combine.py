"""Combine endpoint — merge 2-4 approved sketches with one or more image models."""

import json, base64, asyncio
from pathlib import Path
from datetime import datetime
from typing import Any, List, Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from google.genai import types as genai_types
from PIL import Image

from .common import (
    Phone,
    Broadcast,
    broadcast,
    load_presets,
    require_admin,
)
from .config import (
    MODELS,
    SKETCHES_DIR,
    COMBINED_DIR,
    OPENAI_API_KEY,
    GEMINI_API_KEY,
)
from .llms import client

router = APIRouter()


class Combine:
    @classmethod
    def openai(cls, model_id: str, image_paths: list[Path], prompt: str) -> str:
        openai = client("openai")
        open_files = [open(p, "rb") for p in image_paths]
        try:
            resp = openai.images.edit(
                model=model_id,
                image=open_files if len(open_files) > 1 else open_files[0],
                prompt=prompt,
                size="1024x1024",
            )
        finally:
            for fh in open_files:
                fh.close()
        if not resp.data or resp.data[0].b64_json is None:
            raise RuntimeError("OpenAI returned no image")
        return resp.data[0].b64_json

    @classmethod
    def openai_s3(cls, model_id: str, images: list[bytes], prompt: str) -> str:
        """Same as ``openai`` but takes raw PNG bytes (e.g. pulled from S3)
        instead of disk paths, so the v2 pipeline never needs the trio on local
        disk. The OpenAI SDK accepts ``(filename, bytes, content_type)`` tuples
        anywhere it accepts a file handle."""
        openai = client("openai")
        files: list[Any] = [
            (f"image_{i}.png", body, "image/png") for i, body in enumerate(images)
        ]
        resp = openai.images.edit(
            model=model_id,
            image=files if len(files) > 1 else files[0],
            prompt=prompt,
            size="1024x1024",
            quality="medium",
        )
        if not resp.data or resp.data[0].b64_json is None:
            raise RuntimeError("OpenAI returned no image")
        return resp.data[0].b64_json

    @classmethod
    def gemini(cls, model_id: str, image_paths: list[Path], prompt: str) -> str:
        gemini = client("gemini")
        contents: list[Any] = [prompt]
        for p in image_paths:
            img = Image.open(p)
            contents.append(img)
        response = gemini.models.generate_content(
            model=model_id,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )
        # extract image bytes from response
        candidates = response.candidates
        if not candidates:
            raise RuntimeError("Gemini returned no candidates")
        content = candidates[0].content
        if content is None or content.parts is None:
            raise RuntimeError("Gemini returned no content")
        for part in content.parts:
            inline = part.inline_data
            if (
                inline
                and inline.mime_type
                and inline.mime_type.startswith("image/")
                and inline.data
            ):
                return base64.b64encode(inline.data).decode()
        raise RuntimeError("Gemini returned no image")

    class Request(BaseModel):
        filenames: List[str] = []
        models: Optional[List[str]] = None
        model: str = "gpt-image-1"
        preset: str = "Minimal Zen"
        prompt: str = ""

    class Response(BaseModel):
        class Result(BaseModel):
            model: str
            model_label: str
            image: Optional[str] = None
            saved_as: Optional[str] = None
            error: Optional[str] = None

        success: bool
        results: list["Combine.Response.Result"]


Combine.Response.model_rebuild()


@router.post("/api/combine")
async def combine_sketches(
    payload: Combine.Request, request: Request
) -> Combine.Response:
    require_admin(request)

    filenames = payload.filenames
    model_ids = payload.models if payload.models is not None else [payload.model]
    preset = payload.preset
    extra = payload.prompt.strip()

    if not (2 <= len(filenames) <= 4):
        raise HTTPException(status_code=400, detail="Select between 2 and 4 sketches")
    if not model_ids:
        raise HTTPException(status_code=400, detail="Select at least one model")

    for model_id in model_ids:
        if model_id not in MODELS:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")

    prompt = (
        extra
        or load_presets().get(preset, "")
        or "Combine all sketches into a single creative drawing."
    )

    # gather files
    involved_phones: list[Phone] = []
    image_files: list[Path] = []
    for name in filenames:
        file = SKETCHES_DIR / name
        if not file.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {name}")
        image_files.append(file)
        meta_path = file.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            involved_phones.append(
                Phone(phoneId=meta.get("phoneId"), phoneColor=meta.get("phoneColor"))
            )

    broadcast(
        "combining", Broadcast.Combining(filenames=filenames, phones=involved_phones)
    )

    # run each selected model
    results: list[Combine.Response.Result] = []
    for model_id in model_ids:
        provider = MODELS[model_id]["provider"]
        label = MODELS[model_id]["label"]
        if provider == "openai" and not OPENAI_API_KEY:
            results.append(
                Combine.Response.Result(
                    model=model_id, model_label=label, error="OPENAI_API_KEY not set"
                )
            )
            continue
        if provider == "gemini" and not GEMINI_API_KEY:
            results.append(
                Combine.Response.Result(
                    model=model_id, model_label=label, error="GEMINI_API_KEY not set"
                )
            )
            continue
        try:
            if provider == "openai":
                image_b64 = await asyncio.to_thread(
                    Combine.openai, model_id, image_files, prompt
                )
            else:
                image_b64 = await asyncio.to_thread(
                    Combine.gemini, model_id, image_files, prompt
                )

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_file = COMBINED_DIR / f"combined_{model_id}_{ts}.png"
            with open(image_file, "wb") as f:
                f.write(base64.b64decode(image_b64))

            results.append(
                Combine.Response.Result(
                    model=model_id,
                    model_label=label,
                    image=f"data:image/png;base64,{image_b64}",
                    saved_as=image_file.name,
                )
            )
            print(f"[+] Combined with {label} -> {image_file.name}")
        except Exception as e:
            print(f"[!] Combine error ({label}): {e}")
            results.append(
                Combine.Response.Result(model=model_id, model_label=label, error=str(e))
            )

    broadcast(
        "combined", Broadcast.Combined(filenames=filenames, phones=involved_phones)
    )
    return Combine.Response(success=True, results=results)
