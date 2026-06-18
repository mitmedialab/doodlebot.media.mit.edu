"""Moderation queue — list pending sketches, approve them, or reject them."""

import json, base64
from typing import Any

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from .common import (
    Kind,
    Broadcast,
    broadcast,
    require_admin,
    SuccessResponse,
)
from .config import PENDING_DIR, SKETCHES_DIR

router = APIRouter()


class Pending:
    class Item(BaseModel):
        filename: str
        dataUrl: str
        phoneId: str = "unknown"
        phoneColor: str = "#888888"
        kind: Kind = "drawing"
        created: str = ""
        status: str = "pending"


@router.get("/api/pending")
async def list_pending(request: Request) -> list[Pending.Item]:
    require_admin(request)
    items: list[Pending.Item] = []
    for path in sorted(PENDING_DIR.glob("*.png"), reverse=True):
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if meta.get("status") != "pending":
            continue
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        items.append(Pending.Item(**meta, dataUrl=f"data:image/png;base64,{b64}"))
    return items


@router.post("/api/pending/{filename}/approve")
async def approve(filename: str, request: Request) -> SuccessResponse:
    require_admin(request)

    src_png = PENDING_DIR / filename
    src_json = src_png.with_suffix(".json")
    if not src_png.exists():
        raise HTTPException(status_code=404, detail="Not found")

    dst_png = SKETCHES_DIR / filename
    dst_json = SKETCHES_DIR / src_json.name

    src_png.rename(dst_png)

    meta = json.loads(src_json.read_text()) if src_json.exists() else {}
    meta["status"] = "approved"
    with open(dst_json, "w") as f:
        json.dump(meta, f)
    if src_json.exists():
        src_json.unlink()

    with open(dst_png, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    print(f"[+] Approved {filename}")
    broadcast(
        "approved_sketch",
        Broadcast.Sketch(
            filename=meta.get("filename", filename),
            dataUrl=f"data:image/png;base64,{b64}",
            phoneId=meta.get("phoneId", "unknown"),
            phoneColor=meta.get("phoneColor", "#888888"),
            kind=meta.get("kind", "drawing"),
            created=meta.get("created", ""),
            status=meta.get("status", "approved"),
        ),
    )
    return SuccessResponse()


@router.post("/api/pending/{filename}/reject")
async def reject(filename: str, request: Request) -> SuccessResponse:
    require_admin(request)

    src_png = PENDING_DIR / filename
    src_json = src_png.with_suffix(".json")

    meta: dict[str, Any] = {}
    if src_json.exists():
        meta = json.loads(src_json.read_text())
        meta["status"] = "rejected"
        src_json.unlink()
    if src_png.exists():
        src_png.unlink()

    print(f"[-] Rejected {filename}")
    return SuccessResponse()
