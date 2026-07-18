"""Robot coordination — the polling protocol between Doodlebots and the server.

See the architecture and state-machine diagrams in the repo README. Each bot runs
a **Locate → Poll → Draw** loop and the server is the matchmaker:

1. **Locate** — the bot fetches the known global positions of the aruco markers
   (`GET /api/robots/markers`), detects which ones its camera can see, and solves
   for its own pose in the shared global frame.
2. **Poll** — roughly once a second the bot checks in (`POST /api/robots/checkin`)
   reporting its `name`, `pose` and `status`. The server answers either ``wait``
   (nothing to draw) or ``draw`` (where to *navigate*, what to *draw*, how to
   *exit*).
3. **Draw** — the bot drives to the job's start pose, runs the drawing commands,
   follows the exit path off the canvas, then loops back to Locate.

Unlike a naive design, the **server owns the canvas model**: every canvas, its
markers, its regions (one region per robot), and — crucially — a per-region
*occupancy grid* so a new drawing never overlaps an existing one. When a bot is
handed a job the coordinator runs a placement search (rotation + offset) in that
bot's region, reserves the footprint, and returns the resulting start pose. The
heavy geometry (footprint rasterization, occupancy, placement) lives in
``canvas.py``; this module is the wire protocol, config, and matchmaking.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
import math
from typing import (
    Annotated,
    Any,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypeAlias,
    Union,
)

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, TypeAdapter

from dataclasses import replace

from . import canvas as canvas_engine
from .canvas import (
    Canvas,
    CanvasStore,
    Marker,
    PlacementConfig,
    Region,
    Stroke,
    Placement,
    PlacedDrawing,
)
from .common import require_admin

import numpy as np
from shapely.geometry import LineString, Point as ShapelyPoint
from shapely.ops import unary_union

router = APIRouter()


# --------------------------------------------------------------------------- #
# Shared geometry + drawing-command wire types
#
# The drawing commands mirror the discriminated union produced by the
# vectorizer (`arc_line_vectorization_suede.commands.DrawingCommand`) and the
# bot's TypeScript renderer.
# --------------------------------------------------------------------------- #


class Point(BaseModel):
    """A position in the shared global frame (millimetres, x-right, y-down)."""

    x: float
    y: float


class Pose(BaseModel):
    """A position plus a heading (degrees, CCW positive, matching the vectorizer)."""

    x: float
    y: float
    headingDegrees: float = 0.0


class LineCommand(BaseModel):
    kind: Literal["line"] = "line"
    distance: float
    penDown: bool


class SpinCommand(BaseModel):
    kind: Literal["spin"] = "spin"
    degrees: float


class ArcCommand(BaseModel):
    kind: Literal["arc"] = "arc"
    radius: float
    degrees: float


DrawingCommand: TypeAlias = Annotated[
    Union[LineCommand, SpinCommand, ArcCommand], Field(discriminator="kind")
]


_DRAWING_COMMANDS_ADAPTER = TypeAdapter(list[DrawingCommand])


def parse_commands(raw: Iterable[Mapping[str, Any]]) -> list[DrawingCommand]:
    """Validate raw vectorizer command dicts into the typed discriminated-union
    models the coordinator consumes.

    The vectorizer (``release/vectorize.py``) emits commands as plain JSON-able
    dicts (its ``DrawingCommand`` *TypedDict*). The coordinator reaches into
    commands by attribute (``cmd.kind``, ``cmd.distance``, ...), so those dicts
    must be parsed into the Pydantic ``DrawingCommand`` models here before
    dispatch — don't rely on ``DrawingJob`` coercing them implicitly."""
    return _DRAWING_COMMANDS_ADAPTER.validate_python(list(raw))


# --------------------------------------------------------------------------- #
# Test shapes — canned drawings for exercising a robot end-to-end
#
# These let an admin confirm a newly-online bot can render a vectorization
# without pushing a real sketch through the combine/vectorize pipeline. They are
# ordinary DrawingCommands, so they flow through the exact same placement +
# dispatch path as a real drawing (see build_test_shape / enqueue_drawing).
# --------------------------------------------------------------------------- #

TestShape: TypeAlias = Literal["line", "square", "triangle", "sine_wave"]


def _polyline_to_commands(points: Sequence[tuple[float, float]]) -> list[DrawingCommand]:
    """Turn a pen-down polyline (local mm, starting at the pen origin facing +x)
    into spin+line drawing commands.

    Each segment becomes a Spin to face it, then a pen-down Line of its length —
    the exact inverse of ``canvas.commands_to_strokes``'s turtle integration, so
    the bot redraws the polyline. Absolute size is irrelevant: the coordinator
    rescales the drawing to the target region's footprint at placement time."""
    commands: list[DrawingCommand] = []
    heading = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue  # skip zero-length segments (e.g. a repeated point)
        target = math.degrees(math.atan2(dy, dx))
        turn = (target - heading + 180.0) % 360.0 - 180.0  # shortest turn to face it
        if abs(turn) > 1e-9:
            commands.append(SpinCommand(degrees=turn))
        commands.append(LineCommand(distance=length, penDown=True))
        heading = target
    return commands


def build_test_shape(shape: TestShape) -> list[DrawingCommand]:
    """Drawing commands for a simple calibration shape. Built at an arbitrary base
    size (~100mm); placement scales it to the target region's ``target_footprint``,
    so only the proportions here matter."""
    if shape == "line":
        return _polyline_to_commands([(0.0, 0.0), (100.0, 0.0)])
    if shape == "square":
        return _polyline_to_commands(
            [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (0.0, 0.0)]
        )
    if shape == "triangle":
        height = 100.0 * math.sqrt(3.0) / 2.0  # equilateral, base 100 along +x
        return _polyline_to_commands(
            [(0.0, 0.0), (100.0, 0.0), (50.0, height), (0.0, 0.0)]
        )
    if shape == "sine_wave":
        amplitude, width, cycles, samples = 25.0, 200.0, 2, 48
        points = [
            (
                width * i / samples,
                amplitude * math.sin(2.0 * math.pi * cycles * i / samples),
            )
            for i in range(samples + 1)
        ]
        return _polyline_to_commands(points)
    raise ValueError(f"unknown test shape: {shape!r}")


class ArucoMarker(BaseModel):
    """A fiducial marker at a known global position the bot localizes against."""

    id: int
    position: Point
    sizeMm: Optional[float] = None
    yawRadians: Optional[float] = (
        None  # server-derived from the canvas edge; ignored on input
    )


# --------------------------------------------------------------------------- #
# Canvas configuration (static config + admin endpoint)
# --------------------------------------------------------------------------- #


class RegionConfig(BaseModel):
    id: str
    x: float
    y: float
    width: float
    height: float
    robot: Optional[str] = None  # the robot name assigned to draw this region
    color: Optional[str] = "#000"


class PlacementSettings(BaseModel):
    cellMm: float = 5.0
    penMm: float = 5.0
    clearanceMm: float = 8.0
    searchStepCells: int = 2
    angleStepDeg: float = 15.0  # rotations tried = 0, step, 2·step, … < 360
    strategy: Literal["origin", "scatter"] = "scatter"
    targetFootprintMm: float = 200.0  # scale each drawing so its longest side is ~this
    minFootprintScale: float = 0.4  # shrink floor, as a fraction of targetFootprintMm


class CanvasConfig(BaseModel):
    id: str
    width: float
    height: float
    general_buffer: float = 30.0  # mm of clearance between separate drawings
    markers: list[ArucoMarker] = []
    regions: list[RegionConfig] = []
    placement: PlacementSettings = PlacementSettings()
    drawings: list[PlacedDrawing] = []


# Default canvas layout. In a real deployment this is measured per-venue; defined
# here (and overridable via `POST /api/robots/canvases`) so the system boots with
# something. One 1m × 1m canvas, four corner markers, split into two regions.
DEFAULT_CANVASES: list[CanvasConfig] = [
    CanvasConfig(
        id="main",
        width=1000.0,
        height=1000.0,
        markers=[
            ArucoMarker(id=0, position=Point(x=0.0, y=0.0), sizeMm=100.0),
            ArucoMarker(id=1, position=Point(x=1000.0, y=0.0), sizeMm=100.0),
            ArucoMarker(id=2, position=Point(x=1000.0, y=1000.0), sizeMm=100.0),
            ArucoMarker(id=3, position=Point(x=0.0, y=1000.0), sizeMm=100.0),
        ],
        regions=[
            RegionConfig(id="left", x=0.0, y=0.0, width=500.0, height=1000.0),
            RegionConfig(id="right", x=500.0, y=0.0, width=500.0, height=1000.0),
        ],
        drawings=[],
        general_buffer=30,
    )
]


def _build_canvas(cfg: CanvasConfig) -> Canvas:
    step = max(1.0, cfg.placement.angleStepDeg)
    angles = tuple(i * step for i in range(int(360.0 / step)))
    placement = PlacementConfig(
        cell_mm=cfg.placement.cellMm,
        pen_mm=cfg.placement.penMm,
        clearance_mm=cfg.placement.clearanceMm,
        search_step_cells=cfg.placement.searchStepCells,
        angles_deg=angles,
        strategy=cfg.placement.strategy,
        target_footprint_mm=cfg.placement.targetFootprintMm,
        min_footprint_scale=cfg.placement.minFootprintScale,
    )
    return Canvas(
        id=cfg.id,
        width=cfg.width,
        height=cfg.height,
        markers=[
            Marker(
                id=m.id,
                x=m.position.x,
                y=m.position.y,
                size_mm=m.sizeMm,
                yaw=canvas_engine.edge_yaw(
                    m.position.x, m.position.y, cfg.width, cfg.height
                ),
            )
            for m in cfg.markers
        ],
        regions=[
            Region(
                id=r.id,
                x=r.x,
                y=r.y,
                width=r.width,
                height=r.height,
                robot=r.robot,
                color=r.color,
                config=placement,
            )
            for r in cfg.regions
        ],
        general_buffer=cfg.general_buffer,
    )


# --------------------------------------------------------------------------- #
# Poll (check-in) request / response
# --------------------------------------------------------------------------- #


RobotStatus: TypeAlias = Literal["locating", "ready", "drawing"]


class CheckIn:
    """The ~1s poll: bot → server report, and the server → bot reply."""

    class Request(BaseModel):
        name: str
        status: RobotStatus
        pose: Pose

    class Wait(BaseModel):
        """Nothing to draw yet — keep polling."""

        action: Literal["wait"] = "wait"
        pollAfterSeconds: float = 1.0

    class Draw(BaseModel):
        """A drawing assignment: navigate to ``navigateTo`` then run ``commands``.

        ``navigateTo`` is the drawing's first ink point in the global frame, and
        ``navigateTo.headingDegrees`` is the approach heading — the drawing's own
        start heading plus the rotation the placement search chose for a tight
        fit. ``commands`` is the drawing with its (pen-up) lead-in stripped off,
        since the bot drives straight to the first ink point instead.
        """

        action: Literal["draw"] = "draw"
        jobId: str
        navigateTo: Pose
        navigateFrom: Pose
        commands: list[DrawingCommand]
        exitPose: Optional[Pose] = None


CheckInResponse: TypeAlias = Annotated[
    Union[CheckIn.Draw, CheckIn.Wait], Field(discriminator="action")
]


class DrawingJob(BaseModel):
    """A unit of work: a vectorized drawing awaiting a bot + a place to put it."""

    jobId: str
    commands: list[DrawingCommand]
    exitPose: Optional[Pose] = None
    sourceFilename: Optional[str] = None
    robotName: Optional[str] = None
    """If set, only this bot may draw the job — the placement search considers it
    alone and the job never reroutes to a different bot (used by the admin test-draw
    flow to target one robot). None ⇒ the normal next-best-ready-bot selection."""


# --------------------------------------------------------------------------- #
# Coordinator — ready pool, job queue, and placement against canvas occupancy
# --------------------------------------------------------------------------- #


def _span(strokes: list) -> float:
    """Longest side of the strokes' bounding box (mm), or 0 for no ink."""
    pts = [p for stroke in strokes for p in stroke]
    if not pts:
        return 0.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(max(xs) - min(xs), max(ys) - min(ys))


@dataclass
class _StagedJob:
    job: DrawingJob
    navigate_to: Pose  # resolved start pose (first ink point + approach heading)
    navigate_from: Pose
    commands: list[DrawingCommand]  # the drawing with its lead-in stripped off


@dataclass
class _QueuedJob:
    """A queued drawing plus the inputs reused across every placement attempt.

    ``drawing`` is the native (lead-in-stripped) commands and ``heading0`` its
    start heading, both derived once. ``native_span`` is the longest side of the
    native ink's bounding box — the coordinator divides each canvas's
    ``target_footprint_mm`` by it to get the uniform scale that sizes the drawing
    to that canvas (aspect ratio preserved), before the placement search runs.
    """

    job: DrawingJob
    drawing: list[DrawingCommand]
    heading0: float
    native_span: float

    _footprints: dict[float, canvas_engine.FootprintCache] = field(default_factory=dict)
    """Rotated, rasterized footprints, keyed by the absolute scale they were built at.

    A queued drawing is re-tested against every ready bot on every poll, and the
    scales it gets tried at are deterministic (the target, then a fixed bisection),
    so the same rotations get rasterized over and over. Keyed by absolute scale
    rather than by region, so two regions that size a drawing the same way share the
    work. Lives and dies with the job.
    """

    def footprints_at(
        self, scale: float, strokes: Sequence[canvas_engine.Stroke]
    ) -> canvas_engine.FootprintCache:
        key = round(scale, 6)
        cache = self._footprints.get(key)
        if cache is None:
            cache = canvas_engine.FootprintCache(strokes)
            self._footprints[key] = cache
        return cache


@dataclass
class _RobotRecord:
    name: str
    pose: Pose
    status: RobotStatus
    last_seen: float
    ready_since: Optional[float]  # monotonic time the bot entered "ready", else None
    staged: Optional[_StagedJob] = None


def compute_exit_pose(
    strokes,
    markers: list[Marker],
    allowed_region: Region,
    robot_radius: float = 80,
    min_marker_distance: float = 300,
    max_marker_distance: float = 700,
    distance_step: float = 5,
    boundary: float = 80,
    marker_weight: float = 1.0,
    center_weight: float = 1.0,
    drawing_weight: float = 2.0,
    debug_plot: bool = False,
) -> Pose:

    # Original drawing (used for distance scoring)
    drawing_lines = unary_union(
        [LineString(stroke) for stroke in strokes if len(stroke) >= 2]
    )

    # Buffered drawing (used for collision checking)
    drawing = drawing_lines.buffer(robot_radius)

    best_pose = None
    best_score = float("inf")

    robot = np.array(strokes[-1][-1])

    if allowed_region is not None:
        region_center = np.array(
            [
                allowed_region.x + allowed_region.width / 2,
                allowed_region.y + allowed_region.height / 2,
            ]
        )
    else:
        region_center = None

    for marker in markers:

        if marker.yaw is None:
            continue

        marker_pos = np.array([marker.x, marker.y])

        normal = np.array(
            [
                math.cos(marker.yaw),
                math.sin(marker.yaw),
            ]
        )

        for d in np.arange(
            min_marker_distance,
            max_marker_distance + distance_step,
            distance_step,
        ):

            candidate = marker_pos + d * normal
            point = ShapelyPoint(*candidate)

            # ----------------------------------------------------------
            # Must be inside region
            # ----------------------------------------------------------

            if allowed_region is not None:

                if not (
                    allowed_region.x
                    <= candidate[0]
                    <= allowed_region.x + allowed_region.width
                    and allowed_region.y
                    <= candidate[1]
                    <= allowed_region.y + allowed_region.height
                ):
                    continue

                distance_to_boundary = min(
                    candidate[0] - allowed_region.x,
                    allowed_region.x + allowed_region.width - candidate[0],
                    candidate[1] - allowed_region.y,
                    allowed_region.y + allowed_region.height - candidate[1],
                )

                if distance_to_boundary < boundary:
                    continue

            # ----------------------------------------------------------
            # Collision with buffered drawing
            # ----------------------------------------------------------

            if drawing.contains(point):
                continue

            # ----------------------------------------------------------
            # Distances for scoring
            # ----------------------------------------------------------

            marker_distance = np.linalg.norm(candidate - marker_pos)

            if region_center is not None:
                center_distance = np.linalg.norm(candidate - region_center)
            else:
                center_distance = 0

            # Distance to the actual drawing (not the buffered version)
            drawing_distance = drawing_lines.distance(point)

            score = (
                marker_weight * marker_distance
                + center_weight * center_distance
                + drawing_weight * drawing_distance
            )

            if score < best_score:

                heading = math.atan2(
                    marker.y - candidate[1],
                    marker.x - candidate[0],
                )

                best_score = score

                best_pose = Pose(
                    x=float(candidate[0]),
                    y=float(candidate[1]),
                    headingDegrees=math.degrees(heading),
                )
    if best_pose is not None:
        return best_pose

    print(
        f"Unable to compute best pose for region: {allowed_region.id}, robot: {allowed_region.robot}. Computing default."
    )

    # ------------------------------------------------------------------
    # Fallback: no candidate survived the region / collision / scoring
    # filters (e.g. the drawing fills the region, or no marker had a
    # usable yaw). Default to the center of the allowed region, facing
    # the nearest marker — mirroring the primary loop's heading, which
    # always points from the pose back toward a marker.
    # ------------------------------------------------------------------
    fallback_center = region_center if region_center is not None else robot

    heading_degrees = 0.0
    if markers:
        nearest = min(
            markers,
            key=lambda m: (m.x - fallback_center[0]) ** 2
            + (m.y - fallback_center[1]) ** 2,
        )
        heading_degrees = math.degrees(
            math.atan2(
                nearest.y - fallback_center[1],
                nearest.x - fallback_center[0],
            )
        )

    return Pose(
        x=float(fallback_center[0]),
        y=float(fallback_center[1]),
        headingDegrees=heading_degrees,
    )


class _Coordinator:
    """Thread-safe matchmaker between approved drawings and the ready bot pool."""

    #: Wall-clock ceiling for one ``_assign_locked`` pass, in seconds.
    #:
    #: That pass runs inside the check-in lock, so it delays *every* robot's poll.
    #: Placement is unbounded by nature — a tight canvas means more rotations
    #: searched, more shrink-to-fit steps, more candidate bots — so it is capped
    #: rather than trusted. Whatever it doesn't get to stays queued for the next
    #: check-in, roughly a second later; nothing is dropped.
    assign_budget_s: float = 5.0

    #: Wall-clock ceiling for one *job* within a pass, in seconds.
    #:
    #: Bounds the damage a single awkward drawing can do. Without it, a drawing
    #: that is expensive to fail at (fits nowhere, so every bot runs a full
    #: bisection) would eat the whole pass, every pass, and the jobs behind it
    #: would never be looked at. Exceed this without placing and the job goes to
    #: the back of the queue.
    job_budget_s: float = 1.0

    def __init__(
        self, canvases: list[CanvasConfig], seed: Optional[int] = None
    ) -> None:
        self._lock = threading.Lock()
        self._robots: dict[str, _RobotRecord] = {}
        self._queue: deque[_QueuedJob] = deque()
        self._store = CanvasStore(_build_canvas(c) for c in canvases)
        self._job_counter = 0
        self._drawings: dict[str, list[PlacedDrawing]] = {}
        self._rng = random.Random(
            seed
        )  # drives scatter placement (deterministic if seeded)
        self.drawingDictionary: dict[str, list[Stroke]] = {}

    # -- canvas config ------------------------------------------------------ #

    def markers(self) -> list[Marker]:
        with self._lock:
            return self._store.all_markers()

    def markers_for_robot(self, robot_name: str) -> list[Marker]:
        """Markers of the canvas the robot is placed on (empty if unassigned)."""
        with self._lock:
            canvas = self._store.canvas_for_robot(robot_name)
            return list(canvas.markers) if canvas is not None else []

    def canvases(self) -> list[Canvas]:
        with self._lock:
            return self._store.all()

    def set_canvas(self, cfg: CanvasConfig) -> None:
        with self._lock:
            self._store.upsert(_build_canvas(cfg))
            self.insert_drawings(cfg.id, cfg.drawings)

    def insert_drawings(self, canvas_id: str, drawings: list[PlacedDrawing]) -> None:
        if canvas_id not in self._drawings:
            self._drawings[canvas_id] = []
        for drawing in drawings:
            self._drawings[canvas_id].append(drawing)
            canvas = self._store.get(canvas_id)
            if canvas is None:
                return
            region = canvas.region_for_robot(drawing.robot_name)
            if region is None:
                return
            region.add_drawings([drawing])

    def remove_canvas(self, canvas_id: str) -> None:
        with self._lock:
            self._store.remove(canvas_id)
            self._drawings.pop(canvas_id, None)

    def clear_drawings(self, canvas_id: str) -> int:
        with self._lock:
            if canvas_id not in self._drawings:
                raise KeyError(canvas_id)
            count = len(self._drawings[canvas_id])
            self._drawings[canvas_id] = []
            # Also free the reserved space: without this the record goes empty but
            # every region grid stays full, so nothing new can be placed.
            canvas = self._store.get(canvas_id)
            if canvas is not None:
                for region in canvas.regions:
                    region.clear()
            return count

    # -- enqueue (on approval) ---------------------------------------------- #

    def next_job_id(self) -> str:
        with self._lock:
            self._job_counter += 1
            return f"job_{int(time.time())}_{self._job_counter}"

    def enqueue(self, job: DrawingJob) -> None:
        # Derive the placement inputs once; they're reused across every attempt
        # this job makes while queued. Strip the lead-in so we place only the ink
        # and have the bot navigate straight to the first ink point. ``native_span``
        # lets each canvas scale the drawing to its own target footprint size.
        _lead_in, drawing, _p0, heading0 = canvas_engine.split_lead_in(job.commands)
        native_span = _span(canvas_engine.commands_to_strokes(drawing))
        queued = _QueuedJob(
            job=job,
            drawing=drawing,
            heading0=heading0,
            native_span=native_span,
        )
        with self._lock:
            self._queue.append(queued)
            self._assign_locked()

    def queued(self) -> list[DrawingJob]:
        """The jobs still waiting for a fitting bot (excludes already-staged ones)."""
        with self._lock:
            return [qj.job for qj in self._queue]

    def clear_queue(self) -> int:
        """Drop all waiting jobs. Does not touch jobs already staged on a bot
        (those have reserved space on a canvas and are mid-delivery)."""
        with self._lock:
            n = len(self._queue)
            self._queue.clear()
            return n

    # -- poll (check-in) ---------------------------------------------------- #

    def check_in(self, req: "CheckIn.Request") -> CheckInResponse:
        now = time.monotonic()
        with self._lock:
            record = self._robots.get(req.name)
            if record is None:
                record = _RobotRecord(
                    name=req.name,
                    pose=req.pose,
                    status=req.status,
                    last_seen=now,
                    ready_since=now if req.status == "ready" else None,
                )
                self._robots[req.name] = record
            else:
                if req.status == "ready" and record.status != "ready":
                    record.ready_since = now
                elif req.status != "ready":
                    record.ready_since = None
                record.pose = req.pose
                record.status = req.status
                record.last_seen = now
            if req.status == "ready" or req.status == "locating":
                # Not drawing any more, so stop treating it as a live obstacle.
                # ``pop`` with a default rather than indexing: a bot that has never
                # drawn has no entry at all, and checking in is the first thing it
                # ever does.
                self.drawingDictionary.pop(req.name, None)

            self._assign_locked()

            region = self._store.region_for_robot(req.name)
            canvas = self._store.canvas_for_robot(req.name)

            if canvas is None:
                print(f"ERROR: No canvas for robot {req.name}")
                return CheckIn.Wait()

            if region is None:
                print(f"ERROR: No region for robot {req.name}")
                return CheckIn.Wait()

            if req.status == "ready" and record.staged is not None:
                staged = record.staged
                record.staged = None
                record.status = "drawing"
                record.ready_since = None
                strokes = self.replay_to_world(
                    staged.commands,
                    staged.navigate_to.x,
                    staged.navigate_to.y,
                    staged.navigate_to.headingDegrees,
                )
                exit_pose = compute_exit_pose(strokes, canvas.markers, region)
                self.drawingDictionary[req.name] = strokes
                return CheckIn.Draw(
                    jobId=staged.job.jobId,
                    navigateTo=staged.navigate_to,
                    navigateFrom=staged.navigate_from,
                    commands=staged.commands,
                    exitPose=exit_pose,
                )

            return CheckIn.Wait()

    # -- introspection (admin) ---------------------------------------------- #

    def snapshot(self) -> "RobotPool":
        now = time.monotonic()
        with self._lock:
            bots = []
            for r in self._robots.values():
                region = self._store.region_for_robot(r.name)
                bots.append(
                    RobotPool.Bot(
                        name=r.name,
                        status=r.status,
                        pose=r.pose,
                        region=region.id if region else None,
                        regionFreeFraction=region.free_fraction if region else None,
                        idleSeconds=(now - r.ready_since) if r.ready_since else 0.0,
                        secondsSinceSeen=now - r.last_seen,
                        stagedJobId=r.staged.job.jobId if r.staged else None,
                    )
                )
            return RobotPool(bots=bots, queuedJobs=len(self._queue))

    def assigned_robot(self, job_id: str) -> Optional[str]:
        """The name of the robot a queued job has landed on, or None if it's
        still waiting for a bot (or unknown).

        A job counts as assigned once it's either staged on a bot (a placement
        was found and the bot will collect it on its next check-in) or already
        recorded as a placed drawing. Callers (e.g. the v2 pipeline) poll this to
        learn when a real Doodlebot has taken the drawing."""
        with self._lock:
            for record in self._robots.values():
                if record.staged is not None and record.staged.job.jobId == job_id:
                    return record.name
            for placed_list in self._drawings.values():
                for placed in placed_list:
                    if placed.job_id == job_id:
                        return placed.robot_name
            return None

    def robot_ready(self, robot_name: str) -> bool:
        """Whether ``robot_name`` is checked in, ready, idle, and has a region to
        draw in — i.e. a drawing pinned to it could be staged right now. Used by the
        admin test-draw flow to fail fast rather than queue a job for a bot that
        can't take it."""
        with self._lock:
            record = self._robots.get(robot_name)
            return (
                record is not None
                and record.status == "ready"
                and record.staged is None
                and self._store.region_for_robot(robot_name) is not None
            )

    def color_for_robot(self, robot_name: str) -> str:
        region = self._store.region_for_robot(robot_name)
        color = region.color if region is not None else "#FF0000"
        return color if color is not None else "#FF0000"

    # -- internals ---------------------------------------------------------- #

    def scale_commands(self, commands, scale):
        scaled = []

        for cmd in commands:
            if cmd.kind == "line":
                scaled.append(
                    LineCommand(
                        kind="line",
                        distance=cmd.distance * scale,
                        penDown=cmd.penDown,
                    )
                )

            elif cmd.kind == "arc":
                scaled.append(
                    ArcCommand(
                        kind="arc",
                        radius=cmd.radius * scale,
                        degrees=cmd.degrees,
                    )
                )

            else:  # spin
                scaled.append(cmd)

        return scaled

    def _assign_locked(self) -> None:
        """Place as many queued jobs as possible onto ready bots. Caller holds lock.

        For each queued job we consider every ready bot that has a region and no
        job already staged, ranked best-first (most idle, then most canvas free),
        and run a placement search in that bot's region. The first bot whose
        region can fit the drawing gets it: we reserve the footprint and stage the
        resolved start pose for delivery on that bot's next check-in. Jobs that
        currently fit nowhere stay queued (a region only fills up, so they wait
        for a different ready bot — re-tried on every check-in).

        This runs inside the check-in lock, so it is on the critical path of every
        robot's poll, and a placement search is not cheap — hundreds of ms on a
        tight canvas, times the shrink-to-fit bisection, times each candidate bot.
        Two budgets keep that bounded (see ``assign_budget_s``/``job_budget_s``):
        the pass stops starting new work once it's out of time, and a job that eats
        its own budget without placing goes to the *back* of the queue so it can't
        starve everything behind it. Both are checked between units of work, so the
        real ceiling is the budget plus whatever search was already running.
        """

        if not self._queue:
            return

        started = time.monotonic()
        deadline = started + self.assign_budget_s

        # One prepared snapshot per region, reused for every job and every candidate
        # bot in this pass. Building it costs an occupancy dilation, an FFT and the
        # neighbours' body sweep, and none of that depends on the drawing being
        # placed. Keyed by canvas as well as region: region ids are only unique
        # within their own canvas.
        contexts: dict[tuple[str, str], canvas_engine.PlacementContext] = {}

        def context_for(
            region: Region, canvas: Canvas
        ) -> canvas_engine.PlacementContext:
            key = (canvas.id, region.id)
            context = contexts.get(key)
            if context is None:
                context = region.prepare(
                    general_buffer=canvas.general_buffer,
                    canvas=canvas,
                    active_drawings=self.drawingDictionary,
                )
                contexts[key] = context
            return context

        def invalidate(region: Region, canvas: Canvas) -> None:
            """Drop the snapshots that committing into ``region`` just invalidated.

            Exactly two kinds go stale, and nothing else:

            1. ``region`` itself — it has new ink, so its occupancy moved.
            2. Regions on this canvas that ``adjoins`` it — their occupancy folds in
               the live bodies of robots next door, and the bot we just staged has
               become one.

            Everything else is untouched: a region that doesn't adjoin this one
            can't see the new robot (``_active_robot_keepout`` gates on exactly that
            predicate), and other canvases can't see it at all.
            """
            contexts.pop((canvas.id, region.id), None)
            for other in canvas.regions:
                if other.id != region.id and other.adjoins(region):
                    contexts.pop((canvas.id, other.id), None)

        still_queued: deque[_QueuedJob] = deque()
        deferred: deque[_QueuedJob] = deque()  # blew their budget — go to the back
        while self._queue:
            if time.monotonic() >= deadline:
                break  # out of time; whatever's left in _queue is simply untried

            qj = self._queue.popleft()
            placed = False
            job_deadline = min(deadline, time.monotonic() + self.job_budget_s)

            candidates = [
                r
                for r in self._robots.values()
                if r.status == "ready"
                and r.staged is None
                and self._store.region_for_robot(r.name) is not None
            ]
            # A pinned job (admin test-draw) may only land on its target bot — never
            # rerouted to another ready bot. If that bot isn't among the candidates
            # (not ready, mid-draw, or regionless) the job simply stays queued.
            if qj.job.robotName is not None:
                candidates = [r for r in candidates if r.name == qj.job.robotName]
            candidates.sort(key=self._score, reverse=True)

            for bot in candidates:
                if time.monotonic() >= job_deadline:
                    break  # this job has had its turn; the rest of the queue waits
                region = self._store.region_for_robot(bot.name)
                canvas = self._store.canvas_for_robot(bot.name)
                assert region is not None
                assert canvas is not None
                # Scale the drawing to this canvas's target footprint size (up or
                # down, aspect ratio preserved), then place it — shrinking below
                # the target only as far as needed if it won't fit here. The search
                # rotates the ink for a tighter fit; that rotation rides on the
                # approach heading, so the drawing commands are sent unchanged.
                placement, scaled_commands = self._place_scaled(
                    region, qj, context_for(region, canvas), deadline=job_deadline
                )
                if placement is None:
                    continue  # won't fit even at min scale — try another bot

                region.commit(placement)
                staged_angle = qj.heading0 + placement.angle_deg

                _, navigateFrom = canvas_engine.commands_to_strokes_with_pose(
                    scaled_commands,
                    Pose(
                        x=placement.anchor_x,
                        y=placement.anchor_y,
                        headingDegrees=staged_angle,
                    ),
                )

                bot.staged = _StagedJob(
                    job=qj.job,
                    navigate_to=Pose(
                        x=placement.anchor_x,
                        y=placement.anchor_y,
                        headingDegrees=staged_angle,
                    ),
                    navigate_from=Pose(
                        x=navigateFrom.x,
                        y=navigateFrom.y,
                        headingDegrees=navigateFrom.headingDegrees,
                    ),
                    commands=scaled_commands,
                )
                drawing_strokes = self.replay_to_world(
                    commands=scaled_commands,
                    start_x=placement.anchor_x,
                    start_y=placement.anchor_y,
                    heading_deg=staged_angle,
                )
                exit_pose = compute_exit_pose(drawing_strokes, canvas.markers, region)

                self.add_drawing(
                    canvas.id,
                    qj.job.jobId,
                    bot.name,
                    scaled_commands,
                    placement,
                    qj.heading0,
                    exit_pose,
                )
                self.drawingDictionary[bot.name] = drawing_strokes

                invalidate(region, canvas)
                placed = True
                break

            if placed:
                continue
            if candidates and time.monotonic() >= job_deadline:
                # Had somewhere to try, spent its whole budget, still homeless. Send
                # it to the back so the jobs behind it get a turn — otherwise a
                # drawing that is merely expensive to *fail* at would monopolise
                # every pass forever. The ``candidates`` test matters: a job with no
                # ready bot didn't burn anything, it just had nowhere to go, and
                # demoting it for that would shuffle the queue on every idle poll.
                deferred.append(qj)
            else:
                still_queued.append(qj)

        # Order out: tried-and-didn't-fit first (unchanged relative order), then
        # anything we never got to, then the jobs that timed out.
        still_queued.extend(self._queue)
        still_queued.extend(deferred)
        self._queue = still_queued

    def _score(self, record: _RobotRecord) -> tuple[float, float]:
        region = self._store.region_for_robot(record.name)
        free = region.free_fraction if region else 0.0
        idle = time.monotonic() - record.ready_since if record.ready_since else 0.0
        return (idle, free)

    def replay_to_world(
        self, commands, start_x: float, start_y: float, heading_deg: float
    ):
        """Mirror the robot: run the (lead-in-stripped) commands from the given start
        pose and return pen-down polylines in global mm. This deliberately re-derives
        the ink from the wire message rather than peeking at the occupancy grid."""
        local = canvas_engine.commands_to_strokes(
            commands
        )  # local frame, first ink at (0,0)
        rad = math.radians(heading_deg)
        cos_t, sin_t = math.cos(rad), math.sin(rad)
        world = []
        for stroke in local:
            world.append(
                [
                    (
                        px * cos_t - py * sin_t + start_x,
                        px * sin_t + py * cos_t + start_y,
                    )
                    for px, py in stroke
                ]
            )
        return world

    def _place_scaled(
        self,
        region: Region,
        qj: "_QueuedJob",
        context: canvas_engine.PlacementContext,
        scale_tol: float = 0.02,
        max_iters: int = 12,
        deadline: Optional[float] = None,
    ) -> tuple[Optional[Placement], list]:
        """Place the drawing at this region's target footprint size, shrinking only
        if it won't fit. Returns ``(placement, scaled_commands)``.

        ``context`` is the region's prepared snapshot (occupancy, its FFT, and the
        live neighbours' bodies). It's passed in rather than built here because this
        method runs up to ``max_iters + 1`` searches and every one of them would
        otherwise rebuild it — none of it depends on the drawing or its scale. The
        caller owns it, and must rebuild it whenever the region's ink or the live
        drawings change.

        The base scale ``target = target_footprint_mm / native_span`` sizes the
        drawing (uniformly, so aspect ratio is kept and the longest side hits the
        target) — it may scale up or down. We try that first; if it fits we're
        done. Otherwise footprint-fits-region is monotonic in size, so we
        binary-search a fraction ``s`` of the target in ``[min_footprint_scale, 1)``
        for the largest that still places, and return ``(None, drawing)`` if even
        ``min_footprint_scale * target`` won't fit (caller leaves the job queued
        rather than drawing an illegible speck).

        The loop stops once the scale interval is narrower than ``scale_tol``:
        each step costs a full placement search, and refining below a couple of
        percent moves the footprint less than an occupancy cell. ``max_iters`` is
        a hard backstop.

        ``deadline`` (a ``time.monotonic`` stamp) stops the bisection early. The
        full-size attempt always runs — it's the common case and the reason we're
        here — but the refinement is the expensive part and the most expendable:
        giving up returns the best fit found so far, which is a real placement, just
        possibly smaller than one more step would have found.
        """

        if qj.native_span <= 0:
            return None, qj.drawing
        min_scale = region.config.min_footprint_scale
        target = region.config.target_footprint_mm / qj.native_span

        def attempt(s: float) -> tuple[Optional[Placement], list]:
            scale = target * s
            commands = self.scale_commands(qj.drawing, scale)
            strokes = self.replay_to_world(commands, 0, 0, qj.heading0)
            return (
                region.try_place(
                    strokes,
                    context,
                    rng=self._rng,
                    footprints=qj.footprints_at(scale, strokes),
                ),
                commands,
            )

        # Target size first (s = 1.0): the common case on a canvas with free space.
        placement, commands = attempt(1.0)
        if placement is not None:
            return placement, commands

        # Doesn't fit at target — shrink below it as little as possible.
        lo, hi = min_scale, 1.0
        best: Optional[Placement] = None
        best_commands: list = qj.drawing
        iters = 0
        while hi - lo > scale_tol and iters < max_iters:
            if deadline is not None and time.monotonic() >= deadline:
                break  # out of time — keep the best fit we've already proved
            iters += 1
            mid = (lo + hi) / 2.0
            placement, commands = attempt(mid)
            if placement is not None:
                best, best_commands, lo = placement, commands, mid  # fits → go bigger
            else:
                hi = mid  # too big → shrink the upper bound
        return best, best_commands

    def add_drawing(
        self,
        canvas_id: str,
        job_id: str,
        robot_name: str,
        commands: list,
        placement: Placement,
        heading0: float,
        exit_pose: Pose | None,
    ) -> None:
        # The reserved footprint is the ink at orientation ``heading0 + angle``
        # (the lead-in heading baked into the strokes, plus the placement search's
        # rotation) — the same heading the robot is told to approach with. Replay
        # at that heading so the recorded strokes match the committed occupancy and
        # what the bot actually draws; using ``angle`` alone drops ``heading0`` and
        # swings the drawing off its real pose.
        world_heading = heading0 + placement.angle_deg
        world_strokes = self.replay_to_world(
            commands,
            placement.anchor_x,
            placement.anchor_y,
            world_heading,
        )
        if canvas_id not in self._drawings:
            self._drawings[canvas_id] = []
        if exit_pose:
            self._drawings[canvas_id].append(
                PlacedDrawing(
                    job_id=job_id,
                    robot_name=robot_name,
                    anchor_x=placement.anchor_x,
                    anchor_y=placement.anchor_y,
                    angle_deg=world_heading,
                    commands=commands,
                    strokes=world_strokes,
                    exit_pose_x=exit_pose.x,
                    exit_pose_y=exit_pose.y,
                    exit_pose_deg=exit_pose.headingDegrees,
                )
            )
        else:
            self._drawings[canvas_id].append(
                PlacedDrawing(
                    job_id=job_id,
                    robot_name=robot_name,
                    anchor_x=placement.anchor_x,
                    anchor_y=placement.anchor_y,
                    angle_deg=world_heading,
                    commands=commands,
                    strokes=world_strokes,
                    exit_pose_x=0,
                    exit_pose_y=0,
                    exit_pose_deg=0,
                )
            )


class RobotPool(BaseModel):
    """Admin/debug view of the coordinator state."""

    class Bot(BaseModel):
        name: str
        status: RobotStatus
        pose: Pose
        region: Optional[str] = None
        regionFreeFraction: Optional[float] = None
        idleSeconds: float
        secondsSinceSeen: float
        stagedJobId: Optional[str] = None

    bots: list["RobotPool.Bot"]
    queuedJobs: int


RobotPool.model_rebuild()


# Single process-wide coordinator. Other modules (e.g. moderation on approval)
# enqueue work through `enqueue_drawing`.
coordinator = _Coordinator(DEFAULT_CANVASES)


def enqueue_drawing(
    commands: list[DrawingCommand],
    exit_pose: Optional[Pose] = None,
    source_filename: Optional[str] = None,
    robot_name: Optional[str] = None,
) -> DrawingJob:
    """Queue a vectorized drawing for placement on the next-best ready bot.

    Intended to be called from the approval flow once a combined sketch has been
    vectorized into drawing commands. Where the drawing physically lands (region,
    rotation, offset) is decided by the placement search at assignment time.

    ``robot_name`` pins the job to one specific bot (admin test-draw); the drawing
    is then only ever placed on that bot, never rerouted. Omit it for the normal
    next-best-ready-bot selection.
    """

    job = DrawingJob(
        jobId=coordinator.next_job_id(),
        commands=commands,
        exitPose=exit_pose or None,
        sourceFilename=source_filename,
        robotName=robot_name,
    )
    coordinator.enqueue(job)
    print(f"[robots] queued {job.jobId} ({len(commands)} commands)")
    return job


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


class Markers(BaseModel):
    markers: list[ArucoMarker]


@router.get("/api/robots/markers")
async def get_markers(robot: Optional[str] = None) -> Markers:
    """Locate step: the known global positions of aruco markers.

    Without ``robot`` returns every canvas's markers. With ``?robot=<name>`` it
    returns only the markers of the canvas that robot is placed on (the canvas
    owning the region assigned to it), so a bot localizes against its own canvas
    alone. An unknown or unassigned robot yields an empty list.
    """
    markers = (
        coordinator.markers_for_robot(robot)
        if robot is not None
        else coordinator.markers()
    )
    return Markers(
        markers=[
            ArucoMarker(
                id=m.id,
                position=Point(x=m.x, y=m.y),
                sizeMm=m.size_mm,
                yawRadians=m.yaw,
            )
            for m in markers
        ]
    )


@router.post("/api/robots/checkin")
async def checkin(payload: CheckIn.Request) -> CheckInResponse:
    """Poll step: report pose/status, receive ``wait`` or a ``draw`` job."""
    return coordinator.check_in(payload)


def _placement_settings(pc: PlacementConfig) -> PlacementSettings:
    """Recover the wire ``PlacementSettings`` from a region's engine ``PlacementConfig``.

    The inverse of the transform in ``_build_canvas``: that expands ``angleStepDeg``
    into the explicit ``angles_deg`` tuple (0, step, 2·step, …), so the step is just
    the spacing between its first two entries; every other field maps one-to-one.
    """
    angles = pc.angles_deg
    if len(angles) >= 2:
        angle_step = angles[1] - angles[0]
    elif angles:
        angle_step = 360.0
    else:
        angle_step = PlacementSettings().angleStepDeg
    return PlacementSettings(
        cellMm=pc.cell_mm,
        penMm=pc.pen_mm,
        clearanceMm=pc.clearance_mm,
        searchStepCells=pc.search_step_cells,
        angleStepDeg=angle_step,
        strategy=pc.strategy,
        targetFootprintMm=pc.target_footprint_mm,
        minFootprintScale=pc.min_footprint_scale,
    )


class Canvases(BaseModel):
    class Item(CanvasConfig):
        """A canvas's full config (as POSTed) plus live per-region occupancy.

        Subclasses ``CanvasConfig`` so the GET response round-trips everything a
        POST accepts — notably ``placement`` and ``general_buffer``, which the old
        hand-rolled shape dropped — and adds the one field that only exists at read
        time: how full each region currently is.
        """

        freeFractionByRegion: dict[str, float]

    canvases: list["Canvases.Item"]


Canvases.model_rebuild()


@router.get("/api/robots/canvases")
async def get_canvases(request: Request) -> Canvases:
    """Admin: the configured canvases (full config) with live per-region occupancy."""
    require_admin(request)
    items: list[Canvases.Item] = []
    for c in coordinator.canvases():
        drawings = coordinator._drawings.get(c.id, [])
        # Placement is shared across a canvas's regions; recover it from the first
        # (or fall back to defaults for a region-less canvas).
        placement = (
            _placement_settings(c.regions[0].config)
            if c.regions
            else PlacementSettings()
        )
        items.append(
            Canvases.Item(
                id=c.id,
                width=c.width,
                height=c.height,
                general_buffer=c.general_buffer,
                markers=[
                    ArucoMarker(
                        id=m.id,
                        position=Point(x=m.x, y=m.y),
                        sizeMm=m.size_mm,
                        yawRadians=m.yaw,
                    )
                    for m in c.markers
                ],
                regions=[
                    RegionConfig(
                        id=r.id,
                        x=r.x,
                        y=r.y,
                        width=r.width,
                        height=r.height,
                        robot=r.robot,
                        color=r.color,
                    )
                    for r in c.regions
                ],
                placement=placement,
                drawings=list(drawings),
                freeFractionByRegion={r.id: r.free_fraction for r in c.regions},
            )
        )
    return Canvases(canvases=items)


@router.delete("/api/robots/canvases/{canvas_id}")
async def delete_canvas(canvas_id: str, request: Request) -> None:
    """Admin: delete a canvas by id."""
    require_admin(request)
    try:
        coordinator.remove_canvas(canvas_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Canvas '{canvas_id}' not found")


class ClearedDrawings(BaseModel):
    cleared: int


@router.delete("/api/robots/canvases/{canvas_id}/drawings")
async def clear_drawings(canvas_id: str, request: Request) -> ClearedDrawings:
    """Admin: clear all placed drawings from a canvas (also frees the occupancy)."""
    require_admin(request)
    try:
        cleared = coordinator.clear_drawings(canvas_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Canvas '{canvas_id}' not found")
    print(f"[robots] cleared {cleared} drawing(s) from canvas '{canvas_id}'")
    return ClearedDrawings(cleared=cleared)


@router.post("/api/robots/canvases")
async def post_canvas(payload: CanvasConfig, request: Request) -> CanvasConfig:
    """Admin: register or replace a canvas definition (resets its occupancy)."""
    require_admin(request)
    coordinator.set_canvas(payload)
    print(f"[robots] canvas '{payload.id}' configured ({len(payload.regions)} regions)")
    return payload


@router.get("/api/robots")
async def list_robots(request: Request) -> RobotPool:
    """Admin: inspect the ready pool, region occupancy, and queued-job depth."""
    require_admin(request)
    return coordinator.snapshot()


class EnqueueDrawing(BaseModel):
    commands: list[DrawingCommand]
    exitPose: Optional[Pose] = None
    sourceFilename: Optional[str] = None


@router.post("/api/robots/jobs")
async def post_job(payload: EnqueueDrawing, request: Request) -> DrawingJob:
    """Admin: manually enqueue a drawing job (also the internal approval hook)."""
    require_admin(request)
    if not payload.commands:
        raise HTTPException(status_code=400, detail="No drawing commands")
    return enqueue_drawing(
        commands=payload.commands,
        exit_pose=payload.exitPose,
        source_filename=payload.sourceFilename,
    )


class QueuedJobs(BaseModel):
    class Item(BaseModel):
        jobId: str
        commandCount: int
        sourceFilename: Optional[str] = None

    jobs: list["QueuedJobs.Item"]
    count: int


QueuedJobs.model_rebuild()


@router.get("/api/robots/jobs")
async def list_jobs(request: Request) -> QueuedJobs:
    """Admin: inspect the jobs still waiting to be placed on a bot."""
    require_admin(request)
    jobs = coordinator.queued()
    return QueuedJobs(
        jobs=[
            QueuedJobs.Item(
                jobId=j.jobId,
                commandCount=len(j.commands),
                sourceFilename=j.sourceFilename,
            )
            for j in jobs
        ],
        count=len(jobs),
    )


class ClearedJobs(BaseModel):
    cleared: int


@router.delete("/api/robots/jobs")
async def clear_jobs(request: Request) -> ClearedJobs:
    """Admin: drop all waiting jobs (does not cancel jobs already staged on a bot)."""
    require_admin(request)
    cleared = coordinator.clear_queue()
    print(f"[robots] cleared {cleared} queued job(s)")
    return ClearedJobs(cleared=cleared)
