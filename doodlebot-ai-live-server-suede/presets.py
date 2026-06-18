"""Prompt presets — list the available presets and save new custom ones."""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from .common import load_presets, save_preset, require_admin, SuccessResponse

router = APIRouter()


class Preset:
    class Request(BaseModel):
        name: str
        prompt: str

    class Item(BaseModel):
        name: str
        prompt: str


@router.get("/api/presets")
async def list_presets() -> list[Preset.Item]:
    presets = load_presets()
    return [Preset.Item(name=k, prompt=v) for k, v in presets.items()]


@router.post("/api/presets")
async def add_preset(payload: Preset.Request, request: Request) -> SuccessResponse:
    require_admin(request)
    name = payload.name.strip()
    prompt = payload.prompt.strip()
    if not name or not prompt:
        raise HTTPException(status_code=400, detail="Name and prompt required")
    save_preset(name, prompt)
    print(f"[+] Saved preset: {name}")
    return SuccessResponse()
