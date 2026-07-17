"""DoodleBot AI-Live server — v2 routes.

A production implementation of the backend modelled on the hand-written test
server (the one that exercised the re-worked SvelteKit client). It contributes
the same four routes and speaks the same wire protocol, but replaces the test
server's fakery with the real pipeline:

  * sketch/vectorization image blobs are persisted to S3 (storage.py) and served
    via presigned-URL redirects; only per-sketch meta JSON stays on local disk;
  * every submitted sketch is held for a **human admin** to approve before it can
    group — this is the moderation gate that keeps inappropriate drawings out;
  * approved sketches are grouped into trios *incrementally* — a submitter is
    told about each companion the moment it appears, rather than waiting for the
    whole trio to complete;
  * a completed trio is combined into a single drawing by GPT Image 1 and that
    result is vectorized into robot-drawable strokes; that vectorization is then
    held for the **admin** to approve (and optionally simplify) before it is
    served back to every member of the trio and dispatched to a robot.

Routes (the client-facing four match the test server), exported on ``router`` and
mounted by release/app.py alongside the existing v1 routes:

  GET  /client                 -> { "client": "<uuid>" }
  POST /sketch                 -> { "sketch": "<sha256>" }   (body: client + data URL)
  GET  /resource/{resource_id} -> the sketch PNG or the vectorization SVG
  GET  /events?client=<id>     -> text/event-stream of SSEPayload objects

Admin routes (gated by ``require_admin``; a separate admin front-end drives them):

  GET  /admin/events           -> text/event-stream of items awaiting approval
  POST /admin/sketch           -> record an admin's verdict on a sketch
  POST /admin/vectorization    -> choose a vectorization's final commands
  GET  /admin/sessions         -> the active session tokens
  POST /admin/sessions         -> replace the active session tokens

DESIGN NOTE (admin as resource locators): like ``ClientSSEPayload``'s ids, the
``sketch_id`` and ``vectorization_id`` the admin stream hands out are opaque
locators. The admin echoes one back on its response endpoint and the server
resolves it to the exact pending entity — no session state on the admin side.

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
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Literal, Coroutine, Any, Sequence, cast

import numpy as np
import openai
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel

from .arc_line_vectorization_suede.visualize import (
    commands_to_svg,
    DrawingCommand as VectorDrawingCommand,
)
from .combine import Combine
from .common import SuccessResponse, require_admin
from .robots import DrawingCommand, coordinator, enqueue_drawing, parse_commands
from .storage import StoredResource, storage
from .vectorize import run_vectorization

# --- Tunables --------------------------------------------------------------

# How many approved sketches make a trio that gets combined + vectorized.
TRIO_SIZE = 3

# How many independent combine+vectorize runs are offered to the admin per trio.
# The admin picks the best of these. Fixed at 2 to match the two-option shape of
# Admin.Vectorization.SSEPayload.command_options — bump both together if changed.
VECTORIZATION_OPTIONS = 2

# Cap on how many combine (GPT Image 1) calls may be in flight at once, across
# every trio. OpenAI enforces per-minute rate limits (RPM / images-per-minute,
# tiered by usage) rather than a hard concurrency cap; a global gate keeps bursts
# under those limits, and the retry-with-backoff below absorbs the occasional 429
# that still slips through. Each trio issues VECTORIZATION_OPTIONS calls, so
# without this the in-flight count is 2 x (concurrent trios), unbounded. Default
# 50 suits a Tier 5 org (whose image limits are in the thousands/min); lower it
# on smaller tiers via the env var.
MAX_CONCURRENT_COMBINES = int(os.environ.get("V2_MAX_CONCURRENT_COMBINES", "50"))

# Retry-with-backoff around the combine call, for rate-limit (429) and transient
# server/connection errors. On each retry we wait an exponentially-growing delay
# with full jitter (or the server's Retry-After, if given), capped at
# COMBINE_BACKOFF_MAX. The gate slot is released while backing off so a rate-
# limited call doesn't hold up others. Set retries to 0 to disable.
COMBINE_MAX_RETRIES = int(os.environ.get("V2_COMBINE_MAX_RETRIES", "5"))
COMBINE_BACKOFF_BASE = float(os.environ.get("V2_COMBINE_BACKOFF_BASE", "1.0"))
COMBINE_BACKOFF_MAX = float(os.environ.get("V2_COMBINE_BACKOFF_MAX", "30.0"))

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
Limit your final drawing to 20 or less shapes (individual lines + arcs) to allow for extra fast drawing.
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

# Presigned-URL redirects for /resource (see get_resource). We mint a URL valid
# for PRESIGN_TTL and cache it per resource, so repeated requests for the same
# blob redirect to the *same* S3 URL — letting a browser reuse its cached bytes.
# We stop handing out a cached URL (and cap the redirect's own max-age) once it's
# within PRESIGN_REFRESH_MARGIN of expiry, so a client never follows a dead URL.
PRESIGN_TTL = int(os.environ.get("V2_PRESIGN_TTL", "7200"))  # 2h
PRESIGN_REFRESH_MARGIN = int(os.environ.get("V2_PRESIGN_REFRESH_MARGIN", "600"))

# DESIGN NOTE (data dir): v2 keeps its metadata in its own directory so it never
# collides with the v1 admin flow's pending/ + sketches/ trees. Image blobs
# (served resources + combined debug snapshots) now live in S3 via storage.py;
# only the per-sketch meta JSON stays on local disk.
V2_DATA_DIR = Path(os.environ.get("V2_DATA_DIR", "v2_data"))
SKETCH_META_DIR = V2_DATA_DIR / "sketches"  # per-sketch JSON: state + event log
for _d in (V2_DATA_DIR, SKETCH_META_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# The admin-configured active session tokens live here (JSON: {"sessions": [...]}).
# Unlike in-flight pipeline state, this is durable *config*, so it is loaded on
# startup and rewritten on every change — it survives restarts.
SESSIONS_FILE = V2_DATA_DIR / "sessions.json"


# --- Wire types (must mirror the client's types) ---------------------------

# The admin's moderation verdicts, carried on the client SSE feed's ``status``.
SketchStatus = Literal["approved", "innapropriate", "complex"]
# The synchronous rejection returned from POST /sketch when the submission's
# session token isn't active. Distinct from SketchStatus: it's decided at
# submission time (not by an admin) and never enters the moderation pipeline.
SubmissionRejection = Literal["inactive session"]
RobotKind = Literal["doughnut"]


class ClientSSEPayload(BaseModel):
    """One update pushed to a client over the SSE feed. Matches the client's
    ``SSEPayload`` type exactly, including the (intentional) ``innapropriate``
    spelling. Optional fields are omitted on the wire via ``exclude_none``."""

    sketch: str
    # The session the sketch was submitted under. Stamped onto every payload for a
    # sketch (see Manager._emit) so it rides along on the client feed — the default
    # is only a placeholder for construction and for old persisted events. Always a
    # real token on the wire for a live sketch.
    session: str = ""
    status: SketchStatus | None = None
    companions: list[str] | None = None
    vectorization: str | None = None
    robot: str | None = None
    color: str | None = None

    def encode(self) -> str:
        return self.model_dump_json(exclude_none=True)


class ClientResponse(BaseModel):
    client: str


class SketchRequest(BaseModel):
    client: str
    """The session token the submission was made under (collected by the client
    from its URL). Empty or unknown ⇒ the sketch is rejected as "inactive
    session"; it must match one of the admin's active sessions to be accepted."""
    session: str = ""
    """The image content, as a ``data:image/png;base64,...`` URL exported by the
    client's SketchPad."""
    sketch: str


class SketchResponse(BaseModel):
    sketch: str
    # Set only when the submission was rejected up-front for an inactive session;
    # omitted (None) when the sketch was accepted into the moderation pipeline.
    status: SubmissionRejection | None = None


class Admin:
    class Sketch:
        class SubmitterStats(BaseModel):
            """A snapshot — baked in at the moment the payload is sent, not live —
            of the submitting client's sketch history across the whole system.

            Lets the admin prioritize the queue. The intent is to surface users who
            haven't yet had an honest attempt approved: prefer low ``approved`` and
            low ``pending`` (likely a first-timer waiting on their debut), and treat
            a lone ``rejected_complex`` with little else as worth a quick re-look (a
            user stuck on a too-detailed drawing), while a pile of
            ``rejected_innapropriate`` flags someone likely pushing content through.

            The four buckets partition the client's sketches, so
            ``submitted == pending + approved + rejected_complex +
            rejected_innapropriate``."""

            submitted: int
            """Total sketches this client has ever submitted."""
            pending: int
            """Awaiting an admin verdict (includes the sketch this payload is about)."""
            approved: int
            """Given the "approved" verdict (whether or not they've drawn yet)."""
            rejected_complex: int
            """Rejected with the "complex" verdict (too detailed to combine/vectorize)."""
            rejected_innapropriate: int
            """Rejected with the "innapropriate" verdict (unsafe content)."""

        class SSEPayload(BaseModel):
            type: Literal["sketch"]
            sketch_id: str
            submitter: "Admin.Sketch.SubmitterStats"
            """The submitting client's history, for queue prioritization."""

        class Request(BaseModel):
            """How the admin client responds"""

            sketch_id: str
            status: SketchStatus

    class Vectorization:
        class SSEPayload(BaseModel):
            type: Literal["vectorization"]
            vectorization_id: str

            source_trio: tuple[str, str, str]
            """
            The three source sketch ids the trio was combined from. Each is also a
            /resource/{id} locator for that sketch's PNG, so the admin can judge a
            vectorization against the drawings that produced it.
            """

            command_options: tuple[list[DrawingCommand], list[DrawingCommand]]
            """
            Two independently-combined vectorizations of the same trio. The admin
            picks whichever reads best and trims any unnecessary parts; the chosen
            (and possibly edited) list is returned in the Request.
            """

        class Request(BaseModel):
            """How the admin client responds"""

            vectorization_id: str

            commands: list[DrawingCommand]
            """
            The command list the admin selected (from ``command_options``) and
            optionally trimmed. This is drawable as-is: it is re-vectorized to an
            SVG, stored at ``vectorization_id``, and dispatched to a robot.
            """

    class Sessions:
        class Config(BaseModel):
            """The complete set of active session tokens. A sketch is accepted only
            if its ``session`` matches one of these. Used by both admin session
            endpoints: GET returns the current set, POST replaces it wholesale."""

            sessions: list[str]


# Resolve the forward reference to the (nested) SubmitterStats now that Admin
# exists — the annotation couldn't be a bare name across the nested class scope.
Admin.Sketch.SSEPayload.model_rebuild()


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
class Sketch:
    id: str
    client_id: str
    created: str
    # The active session this sketch was submitted under. Stamped onto every
    # payload emitted for the sketch (see Manager._emit) so the client feed always
    # carries it. Persisted so it survives a restart.
    session: str = ""
    state: ServerState = "approval-pending"
    # The admin's moderation verdict, or None while still awaiting one. Distinct
    # from `state` (which a rejected sketch leaves at "approval-pending"): this is
    # what tells pending/approved/complex/innapropriate apart, so it drives the
    # submitter stats surfaced to the admin. Persisted so counts survive a restart.
    verdict: "SketchStatus | None" = None
    # The full ordered log of every payload emitted for this sketch, so a
    # reconnecting client can be replayed back to the sketch's current state.
    events: list[ClientSSEPayload] = field(default_factory=list)

    def to_meta(self) -> dict:
        return {
            "id": self.id,
            "client_id": self.client_id,
            "created": self.created,
            "session": self.session,
            "state": self.state,
            "verdict": self.verdict,
            "events": [e.model_dump(exclude_none=True) for e in self.events],
        }

    @classmethod
    def from_meta(cls, meta: dict) -> "Sketch":
        return cls(
            id=meta["id"],
            client_id=meta["client_id"],
            created=meta.get("created", ""),
            session=meta.get("session", ""),
            state=meta.get("state", "approval-pending"),
            verdict=meta.get("verdict"),
            events=[ClientSSEPayload(**e) for e in meta.get("events", [])],
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
                member, ClientSSEPayload(sketch=member_id, companions=companions)
            )
        return full


@dataclass
class _PendingVectorization:
    """A combined trio's vectorization options waiting on the admin's choice.

    Held out of every client's feed until the admin picks a drawing — the admin
    reviews two independently-combined ``command_options`` (against the source
    trio) and returns the chosen, possibly-trimmed list. ``id`` is the stable
    resource locator the served SVG is stored under once chosen."""

    id: str
    trio_ids: list[str]
    command_options: list[list[DrawingCommand]]


class Manager:
    """Single source of truth for clients, their sketches, servable resources,
    and the cross-client grouping of approved sketches into trios.

    DESIGN NOTE (durability): sketch metadata and event logs persist to local
    disk; served image blobs persist to S3 (storage.py). So ``/resource`` and the
    reconnect replay survive a restart. What does NOT survive a restart is
    *in-flight pipeline execution* — a sketch caught mid-combine won't resume on
    its own. Restarts should be rare; a durable job queue (or moving the pipeline
    onto a worker + DB) is the recommended next step.
    """

    def __init__(self) -> None:
        self.clients: dict[str, Client] = {}
        self.sketches: dict[str, Sketch] = {}
        self.resources: dict[str, StoredResource] = {}
        # Cache of minted presigned URLs: resource_id -> (url, created_monotonic).
        # See resource_redirect() for the refresh/expiry policy.
        self._presigned: dict[str, tuple[str, float]] = {}
        # The single trio currently being assembled. A new one is started as
        # soon as the previous one fills.
        self._forming = _Group(self)
        # Keep strong references to background tasks so they aren't GC'd.
        self._tasks: set[asyncio.Task[None]] = set()
        # Process-wide gate bounding concurrent OpenAI combine calls. Built lazily
        # in the running loop (see _combine_gate).
        self._combine_gate_sem: "asyncio.Semaphore | None" = None
        self._combine_gate_loop: "asyncio.AbstractEventLoop | None" = None
        # -- admin approval state --------------------------------------------
        # Live admin SSE subscribers (see subscribe_admin). A fresh connection is
        # replayed the current backlog below, then fed live notifications.
        self._admin_subscribers: set["asyncio.Queue[str]"] = set()
        # Sketches parked in "approval-pending", keyed by id, awaiting a verdict.
        self.pending_sketches: dict[str, Sketch] = {}
        # Combined vectorizations awaiting a verdict, keyed by their locator id.
        self.pending_vectorizations: dict[str, _PendingVectorization] = {}
        # Locator ids of vectorizations already resolved, so a duplicate/second-
        # admin resolve is an idempotent no-op rather than a 404. Ids are short
        # hex strings and trios are infrequent, so this set stays tiny.
        self.resolved_vectorizations: set[str] = set()
        # Admin-configured active session tokens. A sketch is accepted only if its
        # submission's session is in here. Loaded from disk below.
        self.active_sessions: set[str] = set()
        self._load_from_disk()

    # -- startup recovery ---------------------------------------------------

    def _load_from_disk(self) -> None:
        """Rebuild the served resources and each client's sketch history so
        reconnecting clients can be replayed after a restart.

        The resource catalogue is enumerated from S3 (blocking, but this runs at
        construction time before the event loop is serving)."""
        self.resources = storage.list_resources()
        self._load_sessions()

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

    # -- sessions -----------------------------------------------------------

    def _load_sessions(self) -> None:
        """Restore the admin's active session tokens from disk (empty if none)."""
        if not SESSIONS_FILE.exists():
            return
        try:
            data = json.loads(SESSIONS_FILE.read_text())
            self.active_sessions = set(data.get("sessions", []))
        except Exception as exc:  # noqa: BLE001
            print(f"[v2] could not load sessions from {SESSIONS_FILE.name}: {exc}")

    def _persist_sessions(self) -> None:
        SESSIONS_FILE.write_text(
            json.dumps({"sessions": sorted(self.active_sessions)})
        )

    def is_active_session(self, session: str) -> bool:
        """Whether a submission's session token is currently active. An empty or
        unknown token is never active (so a sketch without a valid session link is
        rejected rather than erroring)."""
        return bool(session) and session in self.active_sessions

    def get_active_sessions(self) -> list[str]:
        return sorted(self.active_sessions)

    def set_active_sessions(self, sessions: list[str]) -> None:
        """Replace the active session set wholesale (the admin sends the complete
        desired list) and persist it. Blank tokens are dropped. Only gates *new*
        submissions — sketches already in the pipeline are unaffected."""
        self.active_sessions = {s for s in sessions if s}
        self._persist_sessions()

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

    # -- admin feed ---------------------------------------------------------

    def subscribe_admin(self) -> "asyncio.Queue[str]":
        """Register a live admin SSE subscriber, pre-loaded with everything that
        currently needs a verdict. Synchronous (no awaits) so the backlog snapshot
        + registration is atomic against concurrently-emitted pending items.

        Replay is a snapshot of the *outstanding* work — pending sketches then
        pending vectorizations — not a full history: once an item is resolved it
        no longer concerns the admin, so a reconnecting admin only sees what's
        still open."""
        queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._admin_subscribers.add(queue)
        for sketch in self.pending_sketches.values():
            queue.put_nowait(_encode_admin(self._sketch_payload(sketch)))
        for pending in self.pending_vectorizations.values():
            queue.put_nowait(_encode_admin(_vectorization_payload(pending)))
        return queue

    def unsubscribe_admin(self, queue: "asyncio.Queue[str]") -> None:
        self._admin_subscribers.discard(queue)

    def _notify_admin(self, payload: BaseModel) -> None:
        """Fan an admin payload out to every live admin feed."""
        data = _encode_admin(payload)
        for queue in self._admin_subscribers:
            queue.put_nowait(data)

    # -- resources ----------------------------------------------------------

    def get_resource(self, resource_id: str) -> StoredResource | None:
        return self.resources.get(resource_id)

    def resource_redirect(self, resource_id: str) -> tuple[str, int] | None:
        """Resolve a resource to a presigned S3 URL to redirect to, plus the
        max-age (seconds) the redirect itself may be cached for.

        Returns ``None`` for an unknown resource. A URL is reused from the cache
        until it's within PRESIGN_REFRESH_MARGIN of expiry, so repeated requests
        for the same blob point a browser at the same S3 URL (cache-friendly);
        the returned max-age always leaves at least PRESIGN_REFRESH_MARGIN of the
        URL's life, so a cached redirect never resolves to a dead URL.

        generate_presigned_url is a local computation (no network), so this is
        safe to call straight from the event loop."""
        stored = self.resources.get(resource_id)
        if stored is None:
            return None
        now = time.monotonic()
        cached = self._presigned.get(resource_id)
        if cached is None or now - cached[1] >= PRESIGN_TTL - PRESIGN_REFRESH_MARGIN:
            url = storage.presigned_url(stored.key, expires=PRESIGN_TTL)
            self._presigned[resource_id] = (url, now)
            age = 0.0
        else:
            url, created = cached
            age = now - created
        max_age = int(PRESIGN_TTL - PRESIGN_REFRESH_MARGIN - age)
        return url, max(max_age, 0)

    def _store_resource(self, body: bytes, content_type: str) -> str:
        """Content-address ``body``, upload it to S3, and register it as
        servable. Returns the resource id (sha256 hex). Idempotent.

        Blocking (S3 PUT) — call it via ``asyncio.to_thread`` from async code."""
        resource_id = storage.put_resource(body, content_type)
        if resource_id not in self.resources:
            self.resources[resource_id] = StoredResource(
                key=storage.resource_key(resource_id, content_type),
                content_type=content_type,
            )
        return resource_id

    def _register_resource_at(
        self, resource_id: str, body: bytes, content_type: str
    ) -> None:
        """Upload ``body`` under a caller-chosen ``resource_id`` and register it as
        servable. Unlike ``_store_resource`` the id is *not* the content hash — it's
        a stable locator handed out ahead of the bytes (an admin-approved
        vectorization; see _request_vectorization_approval).

        Blocking (S3 PUT) — call it via ``asyncio.to_thread`` from async code."""
        storage.put_resource_at(resource_id, body, content_type)
        self.resources[resource_id] = StoredResource(
            key=storage.resource_key(resource_id, content_type),
            content_type=content_type,
        )

    # -- sketches -----------------------------------------------------------

    async def store_sketch(
        self, client_id: str, session: str, data_url: str
    ) -> SketchResponse:
        """Validate the submission's session, then (if active) persist the sketch
        and kick off its pipeline. Returns the sha256 content id, plus a
        ``status`` of "inactive session" when the session gate rejected it.

        The session check comes first and, on failure, short-circuits before any
        state is created or any blob is uploaded — an inactive session never
        touches storage, the admin queue, or the submitter stats. Idempotent for a
        re-submitted identical sketch."""
        content_type, body = _decode_data_url(data_url)
        sketch_id = hashlib.sha256(body).hexdigest()

        if not self.is_active_session(session):
            # Rejected up front — no client/sketch/resource state is created, so a
            # bad-session flood can't fill storage or the admin queue.
            return SketchResponse(sketch=sketch_id, status="inactive session")

        client = self.ensure_client(client_id)

        if sketch_id in self.sketches:
            # Same content submitted again — hand back the existing id untouched.
            return SketchResponse(sketch=sketch_id)

        # Reserve the sketch synchronously (before any await) so two concurrent
        # identical submissions dedup to one pipeline rather than racing.
        sketch = Sketch(
            id=sketch_id, client_id=client_id, created=_now(), session=session
        )
        self.sketches[sketch_id] = sketch
        client.sketch_ids.append(sketch_id)

        # The sketch PNG is itself a servable resource, addressed by its hash.
        # Upload off the event loop (S3 PUT is blocking). The POST only returns
        # after this completes, so by the time the client has the id and requests
        # /resource, the blob is registered and live.
        await asyncio.to_thread(self._store_resource, body, content_type)

        # The creation event materialises the pipeline on the client. (A live
        # submission already created it eagerly and treats this as a no-op; a
        # reconnecting client needs it to rebuild the model.)
        self._emit(sketch, ClientSSEPayload(sketch=sketch_id))
        self._request_sketch_approval(sketch)
        return SketchResponse(sketch=sketch_id)

    def _persist(self, sketch: Sketch) -> None:
        """Write the sketch's current state + event log to disk. Small JSON, so
        a synchronous write on the event loop is fine."""
        (SKETCH_META_DIR / f"{sketch.id}.json").write_text(json.dumps(sketch.to_meta()))

    # -- event emission -----------------------------------------------------

    def _emit(self, sketch: Sketch, payload: ClientSSEPayload) -> None:
        """Append to the sketch's log, persist, and fan out to live feeds."""
        # Stamp the sketch's session onto every payload here — the single emit
        # choke point — so the session rides along on every client event without
        # each call site having to thread it through.
        payload.session = sketch.session
        sketch.events.append(payload)
        self._persist(sketch)
        data = payload.encode()
        owner = self.clients.get(sketch.client_id)
        if owner is not None:
            for queue in owner.subscribers:
                queue.put_nowait(data)

    # -- sketch approval (admin) -------------------------------------------

    def _request_sketch_approval(self, sketch: Sketch) -> None:
        """Park a freshly-submitted sketch for an admin verdict and notify the
        admin feed. The sketch sits in "approval-pending" (its default state, out
        of the grouping pool) and its owner sees only the creation event until the
        admin decides — so nothing about it reaches the client, not even a status,
        before a human has looked at it.

        DESIGN NOTE (durability): the parked sketch is *not* re-queued to the admin
        after a server restart. The blob and event log persist, but resuming the
        approval hand-off is in-flight pipeline state, which — consistent with the
        combine/robot stages — is not resumed. See the Manager durability note."""
        self.pending_sketches[sketch.id] = sketch
        self._notify_admin(self._sketch_payload(sketch))

    def _sketch_payload(self, sketch: Sketch) -> "Admin.Sketch.SSEPayload":
        """Build the admin SSE payload for a pending sketch, with a fresh snapshot
        of its submitter's stats. Shared by the live notify and the on-connect
        backlog replay so both carry stats and never drift."""
        return Admin.Sketch.SSEPayload(
            type="sketch",
            sketch_id=sketch.id,
            submitter=self._submitter_stats(sketch.client_id),
        )

    def _submitter_stats(self, client_id: str) -> "Admin.Sketch.SubmitterStats":
        """Count a client's sketches by verdict, right now. Cheap (a client has
        few sketches) and computed at send time, so the numbers are a point-in-time
        snapshot rather than a live-updating figure."""
        client = self.clients.get(client_id)
        sketch_ids = client.sketch_ids if client is not None else []
        pending = approved = complex_ = innapropriate = 0
        for sketch_id in sketch_ids:
            sketch = self.sketches.get(sketch_id)
            verdict = sketch.verdict if sketch is not None else None
            if verdict is None:
                pending += 1
            elif verdict == "approved":
                approved += 1
            elif verdict == "complex":
                complex_ += 1
            elif verdict == "innapropriate":
                innapropriate += 1
        return Admin.Sketch.SubmitterStats(
            submitted=len(sketch_ids),
            pending=pending,
            approved=approved,
            rejected_complex=complex_,
            rejected_innapropriate=innapropriate,
        )

    def resolve_sketch(self, sketch_id: str, status: SketchStatus) -> None:
        """Apply an admin's verdict on a pending sketch. Synchronous (no awaits)
        so admit-to-trio is atomic against other approvals on the event loop.

        The verdict becomes the client's ``status``: "approved" moves the sketch
        into the grouping pool exactly as the old auto-classifier did; the
        "innapropriate"/"complex" verdicts are terminal — the client learns why
        and the sketch never groups."""
        sketch = self.sketches.get(sketch_id)
        if sketch is None:
            raise HTTPException(status_code=404, detail="unknown sketch")
        if self.pending_sketches.pop(sketch_id, None) is None:
            # Already resolved — by another admin, or a duplicate of this request.
            # First write wins: this is an idempotent no-op, not an error, and it
            # leaves the existing verdict/state untouched. (No await runs before
            # the pop above, so concurrent resolves can't both pass it.)
            return

        # Record the verdict (drives the submitter stats) before anything else.
        sketch.verdict = status

        if status != "approved":
            # Terminal. Keep it in "approval-pending" (out of the pool); the client
            # learns the verdict via the status field.
            self._emit(sketch, ClientSSEPayload(sketch=sketch_id, status=status))
            return

        sketch.state = "approved"
        self._emit(sketch, ClientSSEPayload(sketch=sketch_id, status="approved"))

        # DESIGN NOTE (incremental grouping): admit the newly-approved sketch to
        # the forming trio right now. Everyone already waiting learns about it
        # immediately, and it learns about them — this is the "pair the moment
        # there's more than one" behaviour the test server flagged as a TODO.
        if self._forming.admit(sketch):
            trio = self._forming.members
            self._forming = _Group(self)
            self._spawn(self._combine(trio))

    # -- pipeline orchestration --------------------------------------------

    async def _combine(self, trio_ids: list[str]) -> None:
        """Combine a full trio into VECTORIZATION_OPTIONS independent drawings,
        vectorize each, then hand both options to the admin to choose from.
        Nothing reaches the trio's clients here — they stay in "combining" until
        the admin picks a vectorization."""
        trio_resources = [self.resources[sid] for sid in trio_ids]

        try:
            # Two independent combine+vectorize runs (GPT Image 1 is
            # non-deterministic, so the same trio yields two distinct options).
            # Run them concurrently; both must succeed to offer a real choice.
            options = await asyncio.gather(
                *(
                    self._combine_and_vectorize(trio_resources)
                    for _ in range(VECTORIZATION_OPTIONS)
                )
            )
        except Exception as exc:  # noqa: BLE001
            # DESIGN NOTE (failure surface): the wire protocol has no "failed"
            # status, so a hard failure here currently leaves the trio stuck in
            # "combining" on the client. We log loudly; adding an error status
            # (a client-side change) is the recommended fix.
            print(f"[v2] combine pipeline failed for {trio_ids}: {exc}")
            return

        self._request_vectorization_approval(trio_ids, list(options))

    # -- vectorization choice (admin) --------------------------------------

    def _request_vectorization_approval(
        self, trio_ids: list[str], command_options: list[list[DrawingCommand]]
    ) -> None:
        """Park a combined trio's vectorization options for the admin's choice and
        notify the admin feed with both option sets plus the source trio.

        The locator is a fresh, non-content-addressed id: the served SVG doesn't
        exist yet (the admin picks + trims the commands), so the id is minted up
        front and the blob is filled in under it once chosen. The trio stays in
        "combining" — no ``vectorization`` reaches any client until then."""
        vectorization_id = uuid.uuid4().hex
        pending = _PendingVectorization(
            id=vectorization_id,
            trio_ids=list(trio_ids),
            command_options=command_options,
        )
        self.pending_vectorizations[vectorization_id] = pending
        self._notify_admin(_vectorization_payload(pending))

    def resolve_vectorization(self, req: "Admin.Vectorization.Request") -> None:
        """Record the admin's chosen (and possibly trimmed) command list for a
        pending vectorization. Synchronous and fast: the actual finalize (render +
        store + robot dispatch, which can block for a robot assignment) runs as a
        background task so the admin's HTTP call returns immediately.

        Every response is a drawable vectorization — there is no reject path.

        Idempotent under concurrent admins: the first resolve wins and the rest are
        no-ops. No await runs before the pop, so two concurrent resolves can't both
        take the pending entry; a later one finds it already resolved and returns
        without re-finalizing (so the drawing isn't dispatched twice)."""
        pending = self.pending_vectorizations.pop(req.vectorization_id, None)
        if pending is None:
            if req.vectorization_id in self.resolved_vectorizations:
                # Already resolved — first write wins, this is a no-op.
                return
            raise HTTPException(
                status_code=404, detail="unknown vectorization"
            )
        self.resolved_vectorizations.add(req.vectorization_id)
        self._spawn(self._finalize_vectorization(pending, req.commands))

    async def _finalize_vectorization(
        self, pending: _PendingVectorization, commands: list[DrawingCommand]
    ) -> None:
        """Render the chosen commands to the served SVG, store it under the
        vectorization's locator id, reveal it to every trio member, and dispatch
        the drawing to a robot."""
        trio = [self.sketches[sid] for sid in pending.trio_ids]

        svg = _render_commands_svg(commands)
        await asyncio.to_thread(
            self._register_resource_at, pending.id, svg.encode("utf-8"), "image/svg+xml"
        )

        for sketch in trio:
            sketch.state = "robot-selection"
            self._emit(
                sketch,
                ClientSSEPayload(sketch=sketch.id, vectorization=pending.id),
            )

        # Dispatch the vectorized drawing to the robot pool and wait for a real
        # Doodlebot to claim it before advancing every member to "complete".
        await self._dispatch_to_robot(trio, commands, source=pending.id)

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
                color = coordinator.color_for_robot(assigned)
                self._emit(
                    sketch,
                    ClientSSEPayload(sketch=sketch.id, robot=assigned, color=color),
                )
        else:
            for sketch in trio:
                sketch.state = "complete"
                self._emit(
                    sketch,
                    ClientSSEPayload(sketch=sketch.id, robot="<invalid>"),
                )

    async def _combine_and_vectorize(
        self, resources: list[StoredResource]
    ) -> list[DrawingCommand]:
        """Run GPT Image 1 over the trio and vectorize the result into the
        low-geometry drawing commands (for admin review + robot dispatch). Retries
        the whole (network + CPU) chain a couple of times before giving up.

        No SVG is stored here: the servable image is only rendered once the admin
        has approved (and possibly simplified) these commands — see
        _finalize_vectorization."""
        last_exc: Exception | None = None
        for attempt in range(1, PIPELINE_ATTEMPTS + 1):
            try:
                # 1) Pull the trio's PNG bytes from S3, then combine them into a
                #    single PNG (reuses combine.py's bytes-based entrypoint).
                images = await asyncio.to_thread(
                    lambda: [storage.read(r.key) for r in resources]
                )
                image_b64 = await self._combine_call(images)
                combined_png = base64.b64decode(image_b64)

                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                await asyncio.to_thread(
                    storage.put_combined_debug, combined_png, f"combined_{ts}.png"
                )

                # 2) Vectorize the combined drawing into the low-geometry robot
                #    strokes. These are what the admin reviews and what ultimately
                #    drives the robot.
                commands = await asyncio.to_thread(_vectorize_for_robot, combined_png)
                return commands
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                print(f"[v2] combine/vectorize attempt {attempt} failed: {exc}")
        assert last_exc is not None
        raise last_exc

    def _combine_gate(self) -> "asyncio.Semaphore":
        """The process-wide semaphore bounding concurrent OpenAI combine calls.

        Created lazily inside the running loop (and rebuilt if the loop changes,
        e.g. between tests) so the semaphore is never awaited from a loop other
        than the one it was created on."""
        loop = asyncio.get_running_loop()
        if self._combine_gate_sem is None or self._combine_gate_loop is not loop:
            self._combine_gate_sem = asyncio.Semaphore(MAX_CONCURRENT_COMBINES)
            self._combine_gate_loop = loop
        return self._combine_gate_sem

    async def _combine_call(self, images: list[bytes]) -> str:
        """Run one GPT Image 1 combine, gated for concurrency and retried with
        backoff on rate-limit / transient errors. Returns the combined PNG as
        base64.

        The gate is acquired *per attempt* and released while backing off, so a
        rate-limited call frees its slot for others instead of holding it idle
        through the wait."""
        attempt = 0
        while True:
            attempt += 1
            try:
                # Gate only the OpenAI call — the rate-limited resource — so no
                # more than MAX_CONCURRENT_COMBINES are ever in flight at once,
                # however many trios are combining. S3 reads + vectorization
                # aren't gated.
                async with self._combine_gate():
                    return await asyncio.to_thread(
                        Combine.openai_s3, COMBINE_MODEL, images, COMBINE_PROMPT
                    )
            except Exception as exc:  # noqa: BLE001
                if attempt > COMBINE_MAX_RETRIES or not _is_retryable(exc):
                    raise
                delay = _combine_backoff(attempt, exc)
                print(
                    f"[v2] combine call hit {type(exc).__name__} "
                    f"(attempt {attempt}/{COMBINE_MAX_RETRIES}); "
                    f"retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)

    # -- task bookkeeping ---------------------------------------------------

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


# --- combine retry/backoff -------------------------------------------------


def _is_retryable(exc: Exception) -> bool:
    """Whether a failed combine call is worth retrying: OpenAI rate limits (429)
    and transient server/connection errors, not client errors (bad request, auth)
    which would just fail again."""
    if isinstance(
        exc,
        (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ),
    ):
        return True
    # Fall back to the status code for anything that isn't one of those exact
    # types (e.g. a wrapped/re-raised error, or a test double).
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    return status == 429 or (isinstance(status, int) and 500 <= status < 600)


def _retry_after_seconds(exc: Exception) -> float | None:
    """The server-requested wait from a Retry-After header, if the error carries
    one — respected over our own backoff so we don't hammer ahead of the reset."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _combine_backoff(attempt: int, exc: Exception) -> float:
    """Seconds to wait before combine retry ``attempt`` (1-based). Prefers the
    server's Retry-After; otherwise exponential backoff with full jitter, capped
    at COMBINE_BACKOFF_MAX."""
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return retry_after
    ceiling = min(COMBINE_BACKOFF_MAX, COMBINE_BACKOFF_BASE * (2 ** (attempt - 1)))
    return random.uniform(0.0, ceiling)


# --- admin feed encoding ---------------------------------------------------


def _encode_admin(payload: BaseModel) -> str:
    """Serialize an admin SSE payload. Both payload types carry a ``type``
    discriminator ("sketch"/"vectorization") so the single admin feed can
    interleave them and the front-end can tell them apart."""
    return payload.model_dump_json()


def _vectorization_payload(
    pending: "_PendingVectorization",
) -> "Admin.Vectorization.SSEPayload":
    """Build the admin SSE payload for a pending vectorization: the source trio's
    sketch ids plus both command options. Shared by the live notify and the
    on-connect backlog replay so the two never drift."""
    trio = pending.trio_ids
    options = pending.command_options
    return Admin.Vectorization.SSEPayload(
        type="vectorization",
        vectorization_id=pending.id,
        source_trio=(trio[0], trio[1], trio[2]),
        command_options=(options[0], options[1]),
    )


# --- vectorization helper --------------------------------------------------


def _commands_to_svg(commands: Sequence[VectorDrawingCommand]):
    return commands_to_svg(
        commands,
        show_pen_up=False,
        stroke_width=4.0,
        stroke="black",
        show_endpoints=False,
    )


def _vectorize_for_robot(png_bytes: bytes) -> list[DrawingCommand]:
    """Vectorize a combined PNG into the low-geometry (robot-drawn) commands.

    Runs the existing pipeline (release/vectorize.py). run_vectorization returns
    commands as plain JSON-able dicts; robot dispatch and the admin payload need
    the typed DrawingCommand models (attribute access), so parse them here."""
    pil = Image.open(io.BytesIO(png_bytes))
    pil.load()
    result = run_vectorization(np.asarray(pil))
    return parse_commands(result["low_geometry"])


def _render_commands_svg(commands: list[DrawingCommand]) -> str:
    """Render approved (possibly admin-simplified) drawing commands to a clean
    standalone SVG — pen-up travel moves hidden so the served image is just the
    black contour. The typed models are dumped back to the dict form
    commands_to_svg consumes."""
    return _commands_to_svg(
        [cast(VectorDrawingCommand, command.model_dump()) for command in commands]
    )


# --- misc helpers ----------------------------------------------------------


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
    """Accept a sketch submission. A ``status`` of "inactive session" in the
    response means the submission's ``session`` wasn't active and nothing was
    stored; otherwise the sketch entered the moderation pipeline."""
    return await manager.store_sketch(request.client, request.session, request.sketch)


@router.get("/resource/{resource_id}")
async def get_resource(resource_id: str) -> Response:
    """Redirect to a presigned S3 URL rather than proxying the bytes, so the
    server never streams blob traffic. The redirect is cacheable only for as long
    as its target URL stays valid (see Manager.resource_redirect); the S3 object
    itself carries an immutable, year-long Cache-Control so the browser caches the
    bytes under that URL."""
    redirect = manager.resource_redirect(resource_id)
    if redirect is None:
        raise HTTPException(status_code=404, detail="unknown resource")
    url, max_age = redirect
    return RedirectResponse(
        url,
        status_code=307,
        headers={"Cache-Control": f"public, max-age={max_age}"},
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


# --- admin routes ----------------------------------------------------------
#
# DESIGN NOTE (admin auth): all three routes are gated by require_admin (from
# .common), so they share the X-Admin-Token header / ?token= query auth the v1
# admin flow uses. The admin front-end is built separately; these endpoints are
# its whole contract.


@router.get("/admin/events")
async def admin_events(request: Request) -> StreamingResponse:
    """SSE feed of everything awaiting an admin verdict: on connect, a snapshot of
    the current backlog (pending sketches, then pending vectorizations), then live
    items as they appear. Each frame is a JSON payload discriminated by ``type``
    ("sketch" | "vectorization")."""
    require_admin(request)
    queue = manager.subscribe_admin()

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
            manager.unsubscribe_admin(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/admin/sketch", response_model=SuccessResponse)
async def admin_resolve_sketch(
    request: Request, payload: Admin.Sketch.Request
) -> SuccessResponse:
    """Record an admin's verdict on a pending sketch. The ``sketch_id`` locates
    the sketch; the ``status`` becomes the client's status (approved → grouping;
    innapropriate/complex → terminal)."""
    require_admin(request)
    manager.resolve_sketch(payload.sketch_id, payload.status)
    return SuccessResponse()


@router.post("/admin/vectorization", response_model=SuccessResponse)
async def admin_resolve_vectorization(
    request: Request, payload: Admin.Vectorization.Request
) -> SuccessResponse:
    """Record the admin's chosen (and optionally trimmed) command list for a
    pending vectorization. Idempotent: a duplicate/second-admin resolve is a no-op.
    Returns immediately; the render + robot dispatch runs in the background."""
    require_admin(request)
    manager.resolve_vectorization(payload)
    return SuccessResponse()


@router.get("/admin/sessions", response_model=Admin.Sessions.Config)
async def admin_get_sessions(request: Request) -> Admin.Sessions.Config:
    """Return the current set of active session tokens (sorted)."""
    require_admin(request)
    return Admin.Sessions.Config(sessions=manager.get_active_sessions())


@router.post("/admin/sessions", response_model=Admin.Sessions.Config)
async def admin_set_sessions(
    request: Request, payload: Admin.Sessions.Config
) -> Admin.Sessions.Config:
    """Replace the active session set with ``payload.sessions`` (send the complete
    desired list — this is a wholesale replace, not a merge). Persisted to disk and
    effective immediately for new submissions. Returns the stored set (sorted)."""
    require_admin(request)
    manager.set_active_sessions(payload.sessions)
    return Admin.Sessions.Config(sessions=manager.get_active_sessions())
