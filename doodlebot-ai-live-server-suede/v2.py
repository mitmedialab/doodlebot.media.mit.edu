"""DoodleBot AI-Live server — v2 routes.

A production implementation of the backend modelled on the hand-written test
server (the one that exercised the re-worked SvelteKit client). It contributes
the same four routes and speaks the same wire protocol, but replaces the test
server's fakery with the real pipeline:

  * sketches are persisted to disk (not held in memory) and served from there;
  * approved sketches are grouped into trios *incrementally* — a submitter is
    told about each companion the moment it appears, rather than waiting for the
    whole trio to complete;
  * a completed trio is combined into a single drawing by GPT Image 1 and that
    result is vectorized into robot-drawable strokes; the rendered vectorization
    is served back to every member of the trio.

Routes (identical to the test server), exported on ``router`` and mounted by
release/app.py alongside the existing v1 routes:

  GET  /client                 -> { "client": "<uuid>" }
  POST /sketch                 -> { "sketch": "<sha256>" }   (body: client + data URL)
  GET  /resource/{resource_id} -> the sketch PNG or the vectorization SVG
  GET  /events?client=<id>     -> text/event-stream of SSEPayload objects

Design decisions worth revisiting are collected in DESIGN NOTE comments
throughout.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel

from .arc_line_vectorization_suede.visualize import commands_to_svg
from .combine import Combine
from .config import COMBINED_DIR
from .robots import DrawingCommand, coordinator, enqueue_drawing, parse_commands
from .vectorize import run_vectorization

# --- Tunables --------------------------------------------------------------

# How many approved sketches make a trio that gets combined + vectorized.
TRIO_SIZE = 3

# The image model + prompt used to combine a trio into one drawing.
COMBINE_MODEL = "gpt-image-1"
COMBINE_PROMPT = """\
doodle creatively combining all sketches into one, adding elements where the \
flow needs it, use words as inspiration, no words or letters in the drawing. \
Use simple arcs and straight lines to make the doodle.
Pure white background, thin clean black lines only, no fill, no shading, no \
color, no hatching. Style: sparse contour drawing, like a zen brushstroke \
illustration.
Keep the drawing as minimal as possible. Minimal and less clear is better \
than complicated.
Your doodle will be drawn by a wheeled drawing robot that is only able to \
draw circular arcs and lines. 
Circular arcs are preferred, since they only require a single command, \
while a line effectively requires two commands (one command to spin in place to orient \
and then the actual line to draw).
Limit your final drawing to 30 or less shapes (lines + arcs) to allow for fast drawing.
"""

# Robot dispatch: after vectorization, the drawing is enqueued for a real
# Doodlebot (see release/robots.py). The client sits in "robot-selection" until
# a bot actually picks the job up; we poll the coordinator for that.
ROBOT_POLL_INTERVAL = float(os.environ.get("V2_ROBOT_POLL_INTERVAL", "1.0"))
# How long to wait for a bot to claim the drawing before completing anyway. The
# job stays queued in the coordinator regardless, so a bot coming online later
# still draws it — this only bounds how long the *client* waits before its
# screen advances. Set to 0 to wait indefinitely for a real assignment.
ROBOT_ASSIGN_TIMEOUT = float(os.environ.get("V2_ROBOT_ASSIGN_TIMEOUT", "120.0"))

HEARTBEAT = 15.0  # seconds between SSE keep-alive comments

# Retries around the (network/CPU-bound) combine + vectorize work.
PIPELINE_ATTEMPTS = 2

# DESIGN NOTE (data dir): v2 keeps its state in its own directory so it never
# collides with the v1 admin flow's pending/ + sketches/ trees. Everything here
# is content-addressed and safe to back with S3/NFS later.
V2_DATA_DIR = Path(os.environ.get("V2_DATA_DIR", "v2_data"))
RESOURCES_DIR = V2_DATA_DIR / "resources"  # served PNG/SVG blobs, named {sha}.{ext}
SKETCH_META_DIR = V2_DATA_DIR / "sketches"  # per-sketch JSON: state + event log
for _d in (V2_DATA_DIR, RESOURCES_DIR, SKETCH_META_DIR, COMBINED_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --- Wire types (must mirror the client's types) ---------------------------

SketchStatus = Literal["approved", "innapropriate", "complex"]
RobotKind = Literal["doughnut"]


class SSEPayload(BaseModel):
    """One update pushed to a client over the SSE feed. Matches the client's
    ``SSEPayload`` type exactly, including the (intentional) ``innapropriate``
    spelling. Optional fields are omitted on the wire via ``exclude_none``."""

    sketch: str
    status: SketchStatus | None = None
    companions: list[str] | None = None
    vectorization: str | None = None
    robot: str | None = None

    def encode(self) -> str:
        return self.model_dump_json(exclude_none=True)


class ClientResponse(BaseModel):
    client: str


class SketchRequest(BaseModel):
    client: str
    """The image content, as a ``data:image/png;base64,...`` URL exported by the
    client's SketchPad."""
    sketch: str


class SketchResponse(BaseModel):
    sketch: str


# --- Server-side domain model ---------------------------------------------

# The pipeline states a sketch moves through. Mirrors the client's ScreenId.
#
# DESIGN NOTE (states vs companions): a sketch stays "approved" while it waits in
# the grouping pool AND while it accumulates companions one at a time. It only
# flips to "combining" once its trio is full (it has TRIO_SIZE-1 companions).
# The client therefore should treat "combining" as "len(companions) == 2", not
# "companions field present". See _Group.admit().
ServerState = Literal[
    "approval-pending",
    "approved",
    "combining",
    "robot-selection",
    "complete",
]

# States we re-load on startup (all of them). In-flight pipeline states are
# reloaded for replay only — they are NOT resumed. See the DURABILITY note.
_LOADABLE_STATES = {
    "approval-pending",
    "approved",
    "combining",
    "robot-selection",
    "complete",
}


@dataclass
class Resource:
    """Something the ``/resource`` route can serve directly as an <img> src.

    Backed by a file on disk; the bytes are read on demand so we don't keep
    every blob resident in memory."""

    content_type: str
    path: Path

    def read(self) -> bytes:
        return self.path.read_bytes()


@dataclass
class Sketch:
    id: str
    client_id: str
    created: str
    state: ServerState = "approval-pending"
    # The full ordered log of every payload emitted for this sketch, so a
    # reconnecting client can be replayed back to the sketch's current state.
    events: list[SSEPayload] = field(default_factory=list)

    def to_meta(self) -> dict:
        return {
            "id": self.id,
            "client_id": self.client_id,
            "created": self.created,
            "state": self.state,
            "events": [e.model_dump(exclude_none=True) for e in self.events],
        }

    @classmethod
    def from_meta(cls, meta: dict) -> "Sketch":
        return cls(
            id=meta["id"],
            client_id=meta["client_id"],
            created=meta.get("created", ""),
            state=meta.get("state", "approval-pending"),
            events=[SSEPayload(**e) for e in meta.get("events", [])],
        )


@dataclass
class Client:
    id: str
    # Sketch ids in submission order (oldest first) — the replay order.
    sketch_ids: list[str] = field(default_factory=list)
    # Live SSE subscribers. A client may briefly have more than one (e.g. a
    # reconnect racing an old connection's teardown).
    subscribers: set["asyncio.Queue[str]"] = field(default_factory=set)


@dataclass
class _Group:
    """A trio being assembled incrementally out of approved sketches.

    Unlike the test server (which batched three at once), members join one at a
    time. Each join tells the newcomer about everyone already inside, and tells
    everyone already inside about the newcomer — so every member's companion
    list grows to its final size as the trio fills."""

    manager: "Manager"
    members: list[str] = field(default_factory=list)

    def admit(self, sketch: Sketch) -> bool:
        """Add ``sketch`` and re-broadcast companion lists. Returns True when the
        trio is now full and should be handed to the combine pipeline."""
        self.members.append(sketch.id)
        full = len(self.members) >= TRIO_SIZE
        for member_id in self.members:
            member = self.manager.sketches[member_id]
            companions = [other for other in self.members if other != member_id]
            # A full companion list means the trio is complete → start combining.
            member.state = "combining" if full else "approved"
            self.manager._emit(
                member, SSEPayload(sketch=member_id, companions=companions)
            )
        return full


class Manager:
    """Single source of truth for clients, their sketches, servable resources,
    and the cross-client grouping of approved sketches into trios.

    DESIGN NOTE (durability): sketch content, metadata, event logs and served
    resources are all persisted to disk, so ``/resource`` and the reconnect
    replay survive a restart. What does NOT survive a restart is *in-flight
    pipeline execution* — a sketch caught mid-combine won't resume on its own.
    Restarts should be rare; a durable job queue (or moving the pipeline onto a
    worker + DB) is the recommended next step.
    """

    def __init__(self) -> None:
        self.clients: dict[str, Client] = {}
        self.sketches: dict[str, Sketch] = {}
        self.resources: dict[str, Resource] = {}
        # The single trio currently being assembled. A new one is started as
        # soon as the previous one fills.
        self._forming = _Group(self)
        # Keep strong references to background tasks so they aren't GC'd.
        self._tasks: set[asyncio.Task[None]] = set()
        self._load_from_disk()

    # -- startup recovery ---------------------------------------------------

    def _load_from_disk(self) -> None:
        """Rebuild the served resources and each client's sketch history so
        reconnecting clients can be replayed after a restart."""
        for path in RESOURCES_DIR.iterdir():
            if path.is_file():
                ct = _content_type_for(path)
                if ct:
                    self.resources[path.stem] = Resource(content_type=ct, path=path)

        metas: list[Sketch] = []
        for path in sorted(SKETCH_META_DIR.glob("*.json")):
            try:
                metas.append(Sketch.from_meta(json.loads(path.read_text())))
            except Exception as exc:  # noqa: BLE001
                print(f"[v2] skipping unreadable sketch meta {path.name}: {exc}")
        # Oldest first so each client's sketch_ids stay in submission order.
        for sketch in sorted(metas, key=lambda s: s.created):
            if sketch.state not in _LOADABLE_STATES:
                continue
            self.sketches[sketch.id] = sketch
            self.ensure_client(sketch.client_id).sketch_ids.append(sketch.id)

    # -- clients ------------------------------------------------------------

    def ensure_client(self, client_id: str) -> Client:
        client = self.clients.get(client_id)
        if client is None:
            client = Client(id=client_id)
            self.clients[client_id] = client
        return client

    def subscribe(self, client_id: str) -> "asyncio.Queue[str]":
        """Register a live SSE subscriber and pre-load it with the client's
        replayed history. Synchronous (no awaits) so the snapshot + registration
        is atomic against concurrently-emitted live events."""
        client = self.ensure_client(client_id)
        queue: "asyncio.Queue[str]" = asyncio.Queue()
        client.subscribers.add(queue)
        for sketch_id in client.sketch_ids:
            for payload in self.sketches[sketch_id].events:
                queue.put_nowait(payload.encode())
        return queue

    def unsubscribe(self, client_id: str, queue: "asyncio.Queue[str]") -> None:
        client = self.clients.get(client_id)
        if client is not None:
            client.subscribers.discard(queue)

    # -- resources ----------------------------------------------------------

    def get_resource(self, resource_id: str) -> Resource | None:
        return self.resources.get(resource_id)

    def _store_resource(self, body: bytes, content_type: str) -> str:
        """Content-address ``body``, write it to the resources dir, and register
        it as servable. Returns the resource id (sha256 hex). Idempotent."""
        resource_id = hashlib.sha256(body).hexdigest()
        if resource_id not in self.resources:
            ext = _ext_for(content_type)
            path = RESOURCES_DIR / f"{resource_id}{ext}"
            path.write_bytes(body)
            self.resources[resource_id] = Resource(content_type=content_type, path=path)
        return resource_id

    # -- sketches -----------------------------------------------------------

    def store_sketch(self, client_id: str, data_url: str) -> str:
        """Persist a submitted sketch and kick off its pipeline. Returns the
        sha256 content id. Idempotent for a re-submitted identical sketch."""
        content_type, body = _decode_data_url(data_url)
        sketch_id = hashlib.sha256(body).hexdigest()

        client = self.ensure_client(client_id)

        if sketch_id in self.sketches:
            # Same content submitted again — hand back the existing id untouched.
            return sketch_id

        # The sketch PNG is itself a servable resource, addressed by its hash.
        self._store_resource(body, content_type)

        sketch = Sketch(id=sketch_id, client_id=client_id, created=_now())
        self.sketches[sketch_id] = sketch
        client.sketch_ids.append(sketch_id)

        # The creation event materialises the pipeline on the client. (A live
        # submission already created it eagerly and treats this as a no-op; a
        # reconnecting client needs it to rebuild the model.)
        self._emit(sketch, SSEPayload(sketch=sketch_id))
        self._spawn(self._review(sketch))
        return sketch_id

    def _persist(self, sketch: Sketch) -> None:
        """Write the sketch's current state + event log to disk. Small JSON, so
        a synchronous write on the event loop is fine."""
        (SKETCH_META_DIR / f"{sketch.id}.json").write_text(json.dumps(sketch.to_meta()))

    # -- event emission -----------------------------------------------------

    def _emit(self, sketch: Sketch, payload: SSEPayload) -> None:
        """Append to the sketch's log, persist, and fan out to live feeds."""
        sketch.events.append(payload)
        self._persist(sketch)
        data = payload.encode()
        owner = self.clients.get(sketch.client_id)
        if owner is not None:
            for queue in owner.subscribers:
                queue.put_nowait(data)

    # -- pipeline orchestration --------------------------------------------

    async def _review(self, sketch: Sketch) -> None:
        """Moderate the sketch, then either reject it or admit it to a trio."""
        status = await _classify(self.get_resource(sketch.id))

        if status != "approved":
            # "innapropriate" / "complex" are terminal — the sketch never groups.
            sketch.state = "approval-pending"  # stays out of the pool
            self._emit(sketch, SSEPayload(sketch=sketch.id, status=status))
            return

        sketch.state = "approved"
        self._emit(sketch, SSEPayload(sketch=sketch.id, status="approved"))

        # DESIGN NOTE (incremental grouping): admit the newly-approved sketch to
        # the forming trio right now. Everyone already waiting learns about it
        # immediately, and it learns about them — this is the "pair the moment
        # there's more than one" behaviour the test server flagged as a TODO.
        # The whole read-modify-write below is synchronous, so it is atomic
        # against other approvals landing on the event loop.
        if self._forming.admit(sketch):
            trio = self._forming.members
            self._forming = _Group(self)
            self._spawn(self._combine(trio))

    async def _combine(self, trio_ids: list[str]) -> None:
        """Combine a full trio into one drawing, vectorize it, hand every member
        the same vectorization resource, dispatch it to a real robot, then
        complete the pipeline once a bot has claimed the drawing."""
        trio = [self.sketches[sid] for sid in trio_ids]
        image_paths = [self.resources[sid].path for sid in trio_ids]

        try:
            vectorization_id, commands = await self._combine_and_vectorize(image_paths)
        except Exception as exc:  # noqa: BLE001
            # DESIGN NOTE (failure surface): the wire protocol has no "failed"
            # status, so a hard failure here currently leaves the trio stuck in
            # "combining" on the client. We log loudly; adding an error status
            # (a client-side change) is the recommended fix.
            print(f"[v2] combine pipeline failed for {trio_ids}: {exc}")
            return

        for sketch in trio:
            sketch.state = "robot-selection"
            self._emit(
                sketch, SSEPayload(sketch=sketch.id, vectorization=vectorization_id)
            )

        # Dispatch the vectorized drawing to the robot pool and wait for a real
        # Doodlebot to claim it before advancing every member to "complete".
        await self._dispatch_to_robot(trio, commands, source=vectorization_id)

    async def _dispatch_to_robot(
        self, trio: list[Sketch], commands: list[DrawingCommand], source: str
    ) -> None:
        """Enqueue the drawing for placement on a ready bot, then poll the
        coordinator until a bot claims it (bounded by ROBOT_ASSIGN_TIMEOUT).

        DESIGN NOTE (robots): this replaces the test server's hard-coded
        "doughnut" with real dispatch through release/robots.py. The low-geometry
        commands are the same ones the bot's renderer consumes; the coordinator
        owns placement (region, rotation, scale) and hands the job to the
        next-best ready bot on its ~1s check-in. RobotKind is currently a single
        "doughnut", so that's what we report; when the pool grows, map the
        assigned bot to its kind here."""
        job = await asyncio.to_thread(enqueue_drawing, commands, None, source)

        # Poll for a real assignment. The job remains queued in the coordinator
        # even past the timeout, so a bot that comes online later still draws it;
        # the timeout only bounds how long the client's screen waits.
        waited = 0.0
        assigned: str | None = None
        while True:
            assigned = coordinator.assigned_robot(job.jobId)
            if assigned is not None:
                break
            if ROBOT_ASSIGN_TIMEOUT and waited >= ROBOT_ASSIGN_TIMEOUT:
                print(
                    f"[v2] {job.jobId} still unassigned after {waited:.0f}s; "
                    "completing client while it stays queued for the next bot"
                )
                break
            await asyncio.sleep(ROBOT_POLL_INTERVAL)
            waited += ROBOT_POLL_INTERVAL

        if assigned is not None:
            print(f"[v2] {job.jobId} claimed by robot '{assigned}'")

        for sketch in trio:
            sketch.state = "complete"
            self._emit(sketch, SSEPayload(sketch=sketch.id, robot=assigned))

    async def _combine_and_vectorize(
        self, image_paths: list[Path]
    ) -> tuple[str, list[DrawingCommand]]:
        """Run GPT Image 1 over the trio and vectorize the result. Returns the
        served SVG's resource id plus the low-geometry drawing commands (for
        robot dispatch). Retries the whole (network + CPU) chain a couple of
        times before giving up."""
        last_exc: Exception | None = None
        for attempt in range(1, PIPELINE_ATTEMPTS + 1):
            try:
                # 1) Combine the trio into a single PNG (reuses combine.py).
                image_b64 = await asyncio.to_thread(
                    Combine.openai, COMBINE_MODEL, image_paths, COMBINE_PROMPT
                )
                combined_png = base64.b64decode(image_b64)

                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                (COMBINED_DIR / f"combined_{ts}.png").write_bytes(combined_png)

                # 2) Vectorize the combined drawing into robot strokes: the
                #    low-geometry commands drive the robot; we also render them to
                #    a clean black-line SVG (no pen-up travel moves) to serve.
                commands, svg = await asyncio.to_thread(
                    _vectorize_for_robot_and_svg, combined_png
                )
                vectorization_id = self._store_resource(
                    svg.encode("utf-8"), "image/svg+xml"
                )
                return vectorization_id, commands
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                print(f"[v2] combine/vectorize attempt {attempt} failed: {exc}")
        assert last_exc is not None
        raise last_exc

    # -- task bookkeeping ---------------------------------------------------

    def _spawn(self, coro) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


# --- moderation ------------------------------------------------------------


async def _classify(resource: Resource | None) -> SketchStatus:
    """Classify a submitted sketch as approved / innapropriate / complex.

    DESIGN NOTE (moderation): this is the seam where real moderation goes. The
    test server approved everything; production needs both a safety check
    (inappropriate content) and a tractability check (too complex to combine or
    vectorize well). Both want a vision model call — e.g. OpenAI's moderation /
    a vision classifier — and are intentionally left as a clearly-marked hook
    rather than guessed at, since no moderation model was specified.

    Defaults to "approved" so the combine + vectorize path is exercised end to
    end. Wire a classifier in here; it may run via asyncio.to_thread.
    """
    return "approved"


# --- vectorization helper --------------------------------------------------


def _vectorize_for_robot_and_svg(png_bytes: bytes) -> tuple[list[DrawingCommand], str]:
    """Vectorize a combined PNG into the low-geometry (robot-drawn) commands and
    a clean standalone SVG rendered from those same commands.

    Runs the existing pipeline (release/vectorize.py). The low-geometry commands
    are what the robot actually draws (and what we hand to robot dispatch); we
    re-render them with pen-up travel moves hidden so the served image is just
    the black contour, matching the zen aesthetic rather than the debug SVG."""
    pil = Image.open(io.BytesIO(png_bytes))
    pil.load()
    result = run_vectorization(np.asarray(pil))
    # run_vectorization returns commands as plain JSON-able dicts. commands_to_svg
    # consumes that dict form directly, but robot dispatch needs the typed
    # DrawingCommand models (attribute access), so parse them for the return.
    raw_commands = result["low_geometry"]
    svg = commands_to_svg(
        raw_commands,
        show_pen_up=False,
        stroke_width=4.0,
        stroke="black",
        show_endpoints=False,
    )
    commands = parse_commands(raw_commands)
    return commands, svg


# --- misc helpers ----------------------------------------------------------

_EXT_BY_CT = {"image/png": ".png", "image/svg+xml": ".svg", "image/jpeg": ".jpg"}
_CT_BY_EXT = {ext: ct for ct, ext in _EXT_BY_CT.items()}


def _ext_for(content_type: str) -> str:
    return _EXT_BY_CT.get(content_type, ".bin")


def _content_type_for(path: Path) -> str | None:
    return _CT_BY_EXT.get(path.suffix)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decode_data_url(data_url: str) -> tuple[str, bytes]:
    """Split a ``data:<mime>;base64,<payload>`` URL into (mime, bytes)."""
    if not data_url.startswith("data:"):
        raise HTTPException(status_code=400, detail="sketch must be a data URL")
    header, _, encoded = data_url.partition(",")
    if not encoded:
        raise HTTPException(status_code=400, detail="malformed data URL")
    mime = header[len("data:") :].split(";")[0] or "application/octet-stream"
    try:
        body = base64.b64decode(encoded)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="invalid base64 payload") from exc
    return mime, body


# --- routes ----------------------------------------------------------------
#
# DESIGN NOTE (mounting): these are contributed as a router, mounted by
# release/app.py next to the v1 routes. CORS is applied app-wide in app.py
# (restricted to the deployed front-end origin), so nothing CORS-related lives
# here. The four paths below don't collide with any existing v1 path.

router = APIRouter()

manager = Manager()


@router.get("/client", response_model=ClientResponse)
async def request_client() -> ClientResponse:
    """Mint a fresh client id. uuid4 is 122 bits of randomness, so a collision
    is astronomically unlikely and no manager-side uniqueness check is needed."""
    return ClientResponse(client=uuid.uuid4().hex)


@router.post("/sketch", response_model=SketchResponse)
async def store_sketch(request: SketchRequest) -> SketchResponse:
    sketch_id = manager.store_sketch(request.client, request.sketch)
    return SketchResponse(sketch=sketch_id)


@router.get("/resource/{resource_id}")
async def get_resource(resource_id: str) -> Response:
    resource = manager.get_resource(resource_id)
    if resource is None:
        raise HTTPException(status_code=404, detail="unknown resource")
    return Response(
        content=resource.read(),
        media_type=resource.content_type,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/events")
async def events(
    request: Request, client: str = Query(..., description="client id")
) -> StreamingResponse:
    """SSE feed for one client: replayed history first, then live updates."""
    queue = manager.subscribe(client)

    async def stream() -> AsyncIterator[str]:
        try:
            yield ": connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"  # comment frame; ignored by EventSource
                    continue
                yield f"data: {data}\n\n"
        finally:
            manager.unsubscribe(client, queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
