"""SSE event stream — clients subscribe here to receive broadcast events."""

import asyncio, queue
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .common import add_listener, remove_listener

router = APIRouter()

HEARTBEAT_TIMEOUT_SECONDS = 25
"""
Seconds to wait for an event before sending a keep-alive ping,
so idle connections aren't dropped by proxies or the browser.
"""


@router.get("/stream")
async def stream() -> StreamingResponse:
    message_queue = add_listener()

    async def generate() -> AsyncGenerator[str, None]:
        yield "event: connected\ndata: {}\n\n"
        try:
            while True:
                try:
                    msg = await asyncio.to_thread(
                        message_queue.get, True, HEARTBEAT_TIMEOUT_SECONDS
                    )
                    yield msg
                except queue.Empty:
                    yield ": ping\n\n"
        finally:
            remove_listener(message_queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
