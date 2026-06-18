"""Approved sketches — list, delete, and the curator's selection notification."""

import json, base64
from typing import List

from fastapi import APIRouter, Request, Body
from pydantic import BaseModel

from .common import (
    Kind,
    Broadcast,
    broadcast,
    require_admin,
    SuccessResponse,
)
from .config import SKETCHES_DIR

router = APIRouter()


class Sketches:
    class Item(BaseModel):
        filename: str
        dataUrl: str
        phoneId: str = "unknown"
        phoneColor: str = "#888888"
        kind: Kind = "drawing"
        created: str = ""


@router.get("/api/sketches")
async def list_sketches() -> list[Sketches.Item]:
    sketches: list[Sketches.Item] = []
    for path in sorted(SKETCHES_DIR.glob("*.png"), reverse=True):
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        with open(path, "rb") as file:
            b64 = base64.b64encode(file.read()).decode()
        sketches.append(
            Sketches.Item(
                filename=path.name,
                dataUrl=f"data:image/png;base64,{b64}",
                phoneId=meta.get("phoneId", "unknown"),
                phoneColor=meta.get("phoneColor", "#888888"),
                kind=meta.get("kind", "drawing"),
                created=meta.get("created", ""),
            )
        )
    return sketches


@router.delete("/api/sketches/{filename}")
async def delete_sketch(filename: str, request: Request) -> SuccessResponse:
    require_admin(request)
    stem = filename.replace(".png", "")
    for ext in (".png", ".json"):
        p = SKETCHES_DIR / (stem + ext)
        if p.exists():
            p.unlink()
    broadcast("deleted_sketch", Broadcast.DeletedSketch(filename=filename))
    return SuccessResponse()


@router.post("/api/select")
async def notify_selected(
    request: Request, selected: List[str] = Body(default=[], embed=True)
) -> SuccessResponse:
    require_admin(request)
    broadcast("selection_changed", Broadcast.SelectionChanged(selected=selected))
    return SuccessResponse()
