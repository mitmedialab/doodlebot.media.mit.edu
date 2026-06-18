"""Submit endpoint — a phone posts a drawing/text sketch into the pending queue."""

import json, base64
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from .common import Kind, Broadcast, broadcast
from .config import PENDING_DIR

router = APIRouter()


class Submit:
    class Request(BaseModel):
        image: str
        phoneId: str = "unknown"
        phoneColor: str = "#888888"
        kind: Kind = "drawing"

    class Response(BaseModel):
        success: bool
        filename: str
        phoneId: str
        phoneColor: str
        kind: Kind
        created: str
        status: str


@router.post("/api/submit")
async def submit(payload: Submit.Request) -> Submit.Response:
    phone_id = payload.phoneId
    phone_color = payload.phoneColor
    kind = payload.kind

    image_data = payload.image
    if image_data.startswith("data:image/png;base64,"):
        image_data = image_data[len("data:image/png;base64,") :]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"sketch_{timestamp}.png"

    with open(PENDING_DIR / filename, "wb") as file:
        file.write(base64.b64decode(image_data))

    created = datetime.now().isoformat()
    meta = {
        "filename": filename,
        "phoneId": phone_id,
        "phoneColor": phone_color,
        "kind": kind,
        "created": created,
        "status": "pending",
    }
    with open(PENDING_DIR / f"sketch_{timestamp}.json", "w") as file:
        json.dump(meta, file)

    print(f"[+] Pending {filename} from {phone_id} ({kind})")
    broadcast(
        "new_pending",
        Broadcast.Sketch(
            filename=filename,
            dataUrl="data:image/png;base64," + image_data,
            phoneId=phone_id,
            phoneColor=phone_color,
            kind=kind,
            created=created,
            status="pending",
        ),
    )

    return Submit.Response(
        success=True,
        filename=filename,
        phoneId=phone_id,
        phoneColor=phone_color,
        kind=kind,
        created=created,
        status="pending",
    )
