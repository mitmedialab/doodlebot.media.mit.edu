"""Canvas occupancy + drawing placement engine.

The server owns a model of every physical canvas: its size, its aruco markers,
and the regions it is carved into (one region drawn by one robot). As drawings
are committed to a region we must guarantee a new drawing never overlaps an
existing one. This module is the algorithm-heavy half of that:

* **Occupancy** is a raster *occupancy grid* per region — the robotics-standard
  representation (cf. ROS costmaps). One cell ≈ a couple of millimetres; a 1 m²
  region is only ~250 KB as ``uint8``, so resolution/memory is a non-issue at
  the scale a pen plotter cares about.

* **Footprint** of a drawing is computed by turtle-integrating its line/spin/arc
  commands into pen-down polylines, then rasterizing those strokes (dilated by
  pen width + a clearance margin) into a small boolean mask.

* **Placement** is the irregular-nesting problem. We do a raster placement
  search: for each candidate rotation we slide the footprint mask over the grid
  and accept the first collision-free pose. A *summed-area table* (integral
  image) makes the common case — the footprint's bounding box landing on blank
  canvas — an O(1) test, so we only pay for an exact mask test near existing ink.

Rotation is essentially free for a turtle path: rotating a drawing by θ is
identical to starting the pen at heading θ. So placement first strips the lead-in
(the pen-up drive to the first ink) via ``split_lead_in``, packs and rotates only
the ink, and reports the first ink point as the start pose with an approach
heading of ``θ0 + angle`` — the drawing commands themselves are sent unchanged.

This module is pure geometry + numpy/PIL and has no FastAPI/pydantic deps; the
wire models and routing live in ``robots.py``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import (
    Iterable,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
    Tuple,
    cast,
)

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw
from pydantic import BaseModel
from scipy.fft import next_fast_len
import cv2


@dataclass
class PlacedDrawing:
    job_id: str
    robot_name: str
    anchor_x: float
    anchor_y: float
    angle_deg: float
    commands: list
    strokes: list
    exit_pose_x: float
    exit_pose_y: float
    exit_pose_deg: float


# --------------------------------------------------------------------------- #
# Command protocol (duck-typed against robots.py's pydantic models)
# --------------------------------------------------------------------------- #


class LineCommand(Protocol):
    kind: Literal["line"]
    distance: float
    penDown: bool


class SpinCommand(Protocol):
    kind: Literal["spin"]
    degrees: float


class ArcCommand(Protocol):
    kind: Literal["arc"]
    radius: float
    degrees: float


Command: TypeAlias = LineCommand | SpinCommand | ArcCommand

# Bound to the command protocol so functions that slice/return commands (e.g.
# ``split_lead_in``) preserve the caller's *concrete* type — robots.py passes its
# pydantic models in and gets pydantic models back, not bare protocol objects.
CommandT = TypeVar("CommandT", bound=Command)


Point = tuple[float, float]
Stroke = list[Point]  # a contiguous pen-down polyline, in drawing-local mm

BoolMask: TypeAlias = npt.NDArray[np.bool_]
"""A boolean raster over grid cells — ``True`` = covered/blocked.

Footprint masks, keep-out maps and free-offset maps are all this shape of thing;
they differ in what they're indexed by, which the parameter names carry.
"""

Grid: TypeAlias = npt.NDArray[np.uint8]
"""A region's committed occupancy grid: 0/1 per cell.

``uint8`` rather than ``bool`` because ``commit`` stamps footprints in with
``|=`` and callers treat it as counts-adjacent (``grid.sum()``).
"""

Spectrum: TypeAlias = npt.NDArray[np.complex128]
"""Half-spectrum of a real 2-D grid, as returned by ``np.fft.rfft2``.

``complex128`` (not ``complex64``) is load-bearing — see ``_free_offsets``.
"""

ROBOT_LENGTH_MM: float = 226.0
"""Nose to tail, along the direction of travel."""

ROBOT_WIDTH_MM: float = 149.0
"""Across the direction of travel."""

ROBOT_PEN_FROM_NOSE_MM: float = 76.0
"""Nose to pen centre.

Well forward of centre, which is the whole reason the body model is worth having:
only 76mm of robot leads the ink, while 150mm of it trails behind.
"""


@dataclass(frozen=True)
class RobotBody:
    """A robot's physical footprint, relative to its pen.

    A robot mid-drawing is a body, not a point — and its ink tells us both where
    its pen is and, from the stroke's direction, which way it is pointing. That's
    enough to say what space it actually occupies, which is far tighter than
    "somewhere within body-length of the pen": see ``Region._active_robot_keepout``.
    """

    length_mm: float = ROBOT_LENGTH_MM
    width_mm: float = ROBOT_WIDTH_MM
    pen_from_nose_mm: float = ROBOT_PEN_FROM_NOSE_MM

    @property
    def pen_from_tail_mm(self) -> float:
        """How much body trails behind the pen. The dominant direction, here."""
        return self.length_mm - self.pen_from_nose_mm

    @property
    def reach_mm(self) -> float:
        """Farthest any part of the body can sit from the pen, over all headings.

        The far tail corner, so ``hypot(pen_from_tail, width/2)``. The body is
        always *inside* ``disc(pen, reach_mm)``, so a pen further than this from an
        obstacle is safe no matter which way the robot points.
        """
        along = max(self.pen_from_nose_mm, self.pen_from_tail_mm)
        return math.hypot(along, self.width_mm / 2.0)

    @property
    def min_overhang_mm(self) -> float:
        """Nearest any part of the body's *edge* can sit to the pen, over all headings.

        The pen sits inside the chassis rectangle, so the disc of this radius is
        inscribed in the body — and the body therefore *contains* ``disc(pen,
        min_overhang_mm)`` at every heading. The mirror of ``reach_mm``: where that
        one says "beyond this, certainly safe", this one says "within this,
        certainly touching", whichever way the robot happens to be pointing.

        That makes it the exact broad-phase bound. Blocking a live robot's body
        grown by this radius rejects every placement that is guaranteed to collide
        and no placement that isn't — so it prunes without ever costing a solution.
        """
        return min(self.pen_from_nose_mm, self.pen_from_tail_mm, self.width_mm / 2.0)


ROBOT_BODY = RobotBody()
"""The fleet's body geometry."""


def _body_sweep(
    a: Point, b: Point, body: RobotBody, margin: float = 0.0
) -> Optional[list[Point]]:
    """The region a robot's body covers while its pen travels from ``a`` to ``b``.

    The robot drives along the segment, so its heading is the segment's direction
    and the swept region is the Minkowski sum of the segment with the body
    rectangle — which, because the body is aligned *to that same heading*, is just
    one bigger rectangle: the body extended by the segment's length. Returned as 4
    corners in the strokes' own frame.

    ``margin`` inflates the rectangle on every side. Callers pass a grid cell's
    worth: rasterizing samples cell *centres*, so a cell straddling the true edge
    can otherwise read as free (measured: up to 0.62 cells of the body left
    unblocked). A keep-out may over-cover safely but must never under-cover, and
    one cell is a rounding error against a 149mm chassis.

    ``None`` for a zero-length segment, where the heading is undefined.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    dist = math.hypot(dx, dy)
    if dist == 0.0:
        return None

    ux, uy = dx / dist, dy / dist  # along travel
    nx, ny = -uy, ux  # across it

    # The nose leads the pen by ``pen_from_nose`` and the tail trails it by the
    # rest, so the swept rectangle starts behind `a` and ends ahead of `b`.
    back = body.pen_from_tail_mm + margin
    front = body.pen_from_nose_mm + margin
    rx, ry = a[0] - ux * back, a[1] - uy * back
    fx, fy = b[0] + ux * front, b[1] + uy * front

    hw = body.width_mm / 2.0 + margin
    return [
        (rx + nx * hw, ry + ny * hw),
        (fx + nx * hw, fy + ny * hw),
        (fx - nx * hw, fy - ny * hw),
        (rx - nx * hw, ry - ny * hw),
    ]


class Pose(BaseModel):
    """A position plus a heading (degrees, CCW positive, matching the vectorizer)."""

    x: float
    y: float
    headingDegrees: float = 0.0


# --------------------------------------------------------------------------- #
# Geometry: drawing commands -> pen-down strokes
# --------------------------------------------------------------------------- #


def commands_to_strokes(
    commands: Sequence[Command], arc_step_deg: float = 6.0
) -> list[Stroke]:
    strokes, _ = commands_to_strokes_with_pose(
        commands, Pose(x=0, y=0, headingDegrees=0), arc_step_deg
    )
    return strokes


def commands_to_strokes_with_pose(
    commands: Sequence[Command], start: Pose, arc_step_deg: float = 6.0
) -> tuple[list[Stroke], Pose]:
    """Same turtle-integration as commands_to_strokes, but also returns the
    final local pose (x, y, headingDegrees). Uses the exact same math as the
    stroke generation (including chord-flattened arcs), so the returned pose
    is guaranteed consistent with the last point of the last stroke."""

    x, y, heading = start.x, start.y, start.headingDegrees
    strokes: list[Stroke] = []
    current: Stroke = []

    def extend(nx: float, ny: float) -> None:
        nonlocal current
        if not current:
            current = [(x, y)]
        current.append((nx, ny))

    def flush() -> None:
        nonlocal current
        if len(current) >= 2:
            strokes.append(current)
        current = []

    for cmd in commands:
        if cmd.kind == "line":
            rad = math.radians(heading)
            nx = x + cmd.distance * math.cos(rad)
            ny = y + cmd.distance * math.sin(rad)
            if cmd.penDown:
                extend(nx, ny)
            else:
                flush()
            x, y = nx, ny
        elif cmd.kind == "spin":
            heading += cmd.degrees
        elif cmd.kind == "arc":
            steps = max(1, int(math.ceil(abs(cmd.degrees) / arc_step_deg)))
            dtheta = cmd.degrees / steps
            seg_len = cmd.radius * math.radians(abs(dtheta))
            for _ in range(steps):
                rad = math.radians(heading)
                nx = x + seg_len * math.cos(rad)
                ny = y + seg_len * math.sin(rad)
                extend(nx, ny)
                x, y = nx, ny
                heading += dtheta
        else:
            raise ValueError(f"Unknown drawing command kind: {cmd.kind!r}")

    flush()
    return strokes, Pose(x=x, y=y, headingDegrees=heading)


def rotate_strokes(strokes: Sequence[Stroke], degrees: float) -> list[Stroke]:
    """Rotate strokes about (0, 0). Rigid, so it only changes the footprint's
    orientation; the placement search re-rasterizes per angle so the pivot choice
    doesn't matter to the mask — only to how we map the anchor back to mm."""
    if degrees == 0.0:
        return [list(s) for s in strokes]
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    return [
        [(px * c - py * s, px * s + py * c) for px, py in stroke] for stroke in strokes
    ]


def split_lead_in(
    commands: Sequence[CommandT],
) -> tuple[list[CommandT], list[CommandT], Point, float]:
    """Separate a drawing's lead-in from its actual ink.

    A vectorization starts at the pen origin and typically *drives pen-up* (and
    maybe spins) to reach the first place it draws. That lead-in is an artifact of
    the vectorizer's origin and shouldn't constrain placement: if we rotated it
    along with the ink, the robot's start pose could end up off-canvas.

    Returns ``(lead_in, drawing, start_point, start_heading)`` where ``drawing``
    begins at the first pen-down command, ``start_point`` is the local-mm point
    where the first ink begins, and ``start_heading`` is the heading (degrees)
    there. The robot is told to navigate straight to ``start_point`` (placed +
    rotated) facing ``start_heading + angle``, then run ``drawing`` — so the
    lead-in is recomputed as a plain "drive to the start" rather than rotated.
    """

    x, y, heading = 0.0, 0.0, 0.0
    for i, cmd in enumerate(commands):
        # Read attributes through the union view: the checkers narrow ``Command``
        # on ``.kind`` but not the ``CommandT`` type var, so cast to the union for
        # the geometry; slice ``commands`` itself so the returned lists keep the
        # caller's concrete command type.
        c = cast(Command, cmd)
        if c.kind == "line":
            if c.penDown:
                return list(commands[:i]), list(commands[i:]), (x, y), heading
            rad = math.radians(heading)
            x += c.distance * math.cos(rad)
            y += c.distance * math.sin(rad)
        elif c.kind == "spin":
            heading += c.degrees
        elif c.kind == "arc":  # arcs are pen-down ink — drawing starts here
            return list(commands[:i]), list(commands[i:]), (x, y), heading
        else:
            raise ValueError(f"Unknown drawing command kind: {c.kind!r}")

    return list(commands), [], (x, y), heading  # no ink at all


# --------------------------------------------------------------------------- #
# Rasterization: strokes -> boolean footprint mask
# --------------------------------------------------------------------------- #


@dataclass
class Footprint:
    mask: BoolMask  # shape (rows, cols) — True = covered
    min_x: float  # local-mm coords of the mask's (0,0) cell corner
    min_y: float
    pad: int
    cell_mm: float

    def to_px(self, point: Point) -> tuple[float, float]:
        """Map a local-mm point to (col, row) within the mask."""
        return (
            (point[0] - self.min_x) / self.cell_mm + self.pad,
            (point[1] - self.min_y) / self.cell_mm + self.pad,
        )


def rasterize(
    strokes: Sequence[Stroke], cell_mm: float, half_width_cells: int
) -> Footprint:
    """Rasterize pen-down strokes into a dilated boolean mask.

    ``half_width_cells`` is half the drawn line thickness in cells — i.e. the
    pen radius plus the clearance margin — so the mask already encodes the
    keep-out buffer around the ink.
    """

    pts = [p for stroke in strokes for p in stroke]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, min_y = min(xs), min(ys)
    max_x, max_y = max(xs), max(ys)

    pad = half_width_cells + 1
    width_cells = int(math.ceil((max_x - min_x) / cell_mm)) + 2 * pad + 1
    height_cells = int(math.ceil((max_y - min_y) / cell_mm)) + 2 * pad + 1

    img = Image.new("1", (width_cells, height_cells), 0)
    draw = ImageDraw.Draw(img)

    def to_px(p: Point) -> tuple[float, float]:
        return ((p[0] - min_x) / cell_mm + pad, (p[1] - min_y) / cell_mm + pad)

    line_width = 2 * half_width_cells + 1
    for stroke in strokes:
        if len(stroke) >= 2:
            draw.line(
                [to_px(p) for p in stroke], fill=1, width=line_width, joint="curve"
            )
        elif len(stroke) == 1:
            px, py = to_px(stroke[0])
            r = half_width_cells
            draw.ellipse([px - r, py - r, px + r, py + r], fill=1)

    mask: BoolMask = np.array(img, dtype=np.bool_)
    return Footprint(mask=mask, min_x=min_x, min_y=min_y, pad=pad, cell_mm=cell_mm)


# --------------------------------------------------------------------------- #
# Placement search over an occupancy grid
# --------------------------------------------------------------------------- #


@dataclass
class PlacementConfig:
    cell_mm: float = 5.0
    """Occupancy-grid resolution. ~2mm is sub-pen-width; finer just costs memory."""

    pen_mm: float = 5.0
    """Drawn line width."""

    clearance_mm: float = 8.0
    """Keep-out margin enforced between separate drawings."""

    search_step_cells: int = 2
    """Stride of the sliding-window search (in cells). Larger = faster, coarser."""

    angles_deg: tuple[float, ...] = tuple(float(a) for a in range(0, 360, 15))
    """Candidate rotations, tried in order (0° first = prefer upright)."""

    strategy: Literal["origin", "scatter"] = "scatter"
    """How to choose among valid poses.

    ``origin`` packs every drawing toward the region's (0,0) corner — the
    classic Bottom-Left-Fill heuristic, though in this screen-like y-down frame
    (0,0) is the top-left. Dense, but it looks like a print head filling a page.
    ``scatter`` picks a random valid pose (position + rotation), so drawings
    appear spread across the region like several artists working it at once; it
    jams at lower coverage (random sequential packing always does), trading
    density for that organic look.
    """

    target_footprint_mm: float = 200.0
    """Desired size of a placed drawing given ample free space: the coordinator
    uniformly scales each drawing (preserving aspect ratio) so its *longest*
    footprint dimension is this many mm, then shrinks below it only if it won't
    fit. Carried here for convenience; the scaling itself lives in ``robots.py``.
    """

    min_footprint_scale: float = 0.4
    """Floor for the shrink-to-fit fallback, as a fraction of ``target_footprint_mm``.
    A drawing that won't fit even at this size is left queued rather than drawn
    illegibly small. Also carried for convenience; applied in ``robots.py``.
    """


@dataclass
class Placement:
    angle_deg: float
    anchor_x: float  # global mm — where the first ink point (P0) lands
    anchor_y: float
    _top: int  # grid row of the mask's top-left (internal, for stamping)
    _left: int
    _footprint: Footprint


@dataclass(frozen=True)
class PlacementContext:
    """Everything a placement search needs about a region that won't change during it.

    ``try_place`` is called many times for one placement — once per candidate
    rotation internally, and the whole search again on every step of the
    coordinator's shrink-to-fit binary search, and again on the next poll if the
    drawing stayed queued. None of *this* varies across those calls: it depends on
    the region's committed ink and on who is drawing next door, not on the drawing
    being placed. So it's built once by ``Region.prepare`` and handed in.

    Fields, in the order the search uses them:

    ``occupancy``
        The broad-phase keep-out — committed ink grown by ``general_buffer``, plus
        live neighbours' bodies grown by the query robot's overhang. What the
        candidate footprint is correlated against.
    ``grid_fft`` / ``fft_shape``
        ``occupancy`` transformed once, reused for every rotation (and every call).
        ``None`` when the region is empty, where every offset is trivially free.
    ``active_bodies``
        Live neighbours' swept bodies, *ungrown* — the narrow phase's ground truth.
        Empty when nobody adjacent is drawing, which is the common case and lets
        ``try_place`` skip the narrow phase entirely.
    ``active_reach_bbox``
        ``active_bodies``' bounding box grown by ``robot_body.reach_mm``, in cells,
        as ``(top, left, bottom, right)``. A candidate whose footprint misses this
        cannot possibly collide, so it skips the exact check on an O(1) test.
        ``None`` when no neighbour is live.
    """

    occupancy: BoolMask
    grid_fft: Optional[Spectrum]
    fft_shape: tuple[int, int]
    active_bodies: BoolMask
    active_reach_bbox: Optional[tuple[int, int, int, int]]
    robot_body: RobotBody

    @property
    def has_live_neighbour(self) -> bool:
        """Is any robot drawing close enough to this region to need the narrow phase?"""
        return self.active_reach_bbox is not None


def _fft_shape(grid_shape: tuple[int, int]) -> tuple[int, int]:
    """Transform size to use for a grid of ``grid_shape`` — the next FFT-friendly size.

    An FFT is only fast when its length factors into small primes; a prime length
    degrades toward the O(n²) DFT. Grid sizes here are ``ceil(width_mm / cell_mm)``
    — an arbitrary integer chosen by whoever posted the canvas — so they land on
    hostile lengths regularly (a 1000mm canvas at 1.5mm cells gives 667 = 23·29,
    which transforms ~2x slower than the 675 next to it). Rounding *up* to a
    5-smooth length costs a little zero-padding and buys that back.

    Padding is safe for the wraparound argument in ``_free_offsets``: enlarging the
    period only adds zeros past the grid, and the valid offsets we read back
    (``t <= rows - mh``) stay well inside the un-aliased range.
    """

    return (
        next_fast_len(grid_shape[0], real=True),  # type: ignore
        next_fast_len(grid_shape[1], real=True),
    )


def _free_offsets(
    grid_fft: Spectrum,
    mask: BoolMask,
    grid_shape: tuple[int, int],
    fft_shape: tuple[int, int],
) -> BoolMask:
    """Boolean map of collision-free top-left offsets, via FFT cross-correlation.

    ``corr[t, l]`` is the number of occupied cells the footprint would overlap if
    its mask's top-left sat at grid cell ``(t, l)``. The cross-correlation theorem
    gives every offset at once from one inverse FFT —

        corr = irfft2( FFT(grid) · conj(FFT(mask)) )

    — replacing the per-position sweep that used to dominate the runtime. Overlap
    counts are integers, so a free cell is exactly where ``corr`` rounds to zero
    (``< 0.5`` absorbs FFT floating-point error — which is also why this stays in
    float64: the transform's DC term runs to ~1e8 here, and float32's epsilon
    against that is far wider than the 0.5 the threshold allows).

    ``grid_fft`` is precomputed once per placement and reused across rotations
    (only the mask changes per angle); it must have been transformed at
    ``fft_shape``, which ``_fft_shape`` derives from ``grid_shape``.

    Returns an array over valid top-left offsets, shape
    ``(rows - mh + 1, cols - mw + 1)``; ``True`` means that offset is collision-free.
    """
    rows, cols = grid_shape
    mh, mw = mask.shape
    mask_fft = np.fft.rfft2(mask.astype(np.float64), fft_shape)
    corr = np.fft.irfft2(grid_fft * np.conj(mask_fft), fft_shape)
    valid = corr[: rows - mh + 1, : cols - mw + 1]
    return valid < 0.5


FootprintCacheKey: TypeAlias = tuple[
    float, float, int
]  # (angle, cell_mm, half_width_cells)

FootprintCacheValue: TypeAlias = tuple[
    Optional[Footprint], Optional[Point]
]  # (footprint, first-ink point)

BodySweepKey: TypeAlias = tuple[
    float, float, int, RobotBody, float
]  # (angle, cell_mm, half_width_cells, body, margin)


@dataclass(frozen=True)
class BodySweep:
    """The space the query robot's own chassis covers while drawing, at one rotation.

    The counterpart of ``Footprint``: that one is the ink, this is the machine that
    lays it. Like the footprint it depends only on the *rotation*, never on where
    the drawing ends up — so it is built once per angle and then simply shifted to
    each candidate offset, which is what makes the narrow phase affordable.

    ``d_row``/``d_col`` place it relative to the ink mask: the sweep is bigger (the
    chassis overhangs the ink), so its origin sits this many cells above and left of
    the ink mask's own. Given a placement's ``_top``/``_left``, the sweep's top-left
    grid cell is ``(_top - d_row, _left - d_col)``.
    """

    mask: BoolMask
    d_row: int
    d_col: int


class FootprintCache:
    """Memoizes one drawing's rotated, rasterized footprints across placements.

    A queued drawing is re-tested for placement on every poll and against every
    candidate bot, but its footprint at a given rotation and grid resolution never
    changes — only the occupancy it's tested against does. Caching the rotate +
    rasterize step (the expensive part) turns that repeated work into a one-time
    cost per ``(angle, cell, half)``. The cache is keyed so distinct region
    resolutions coexist correctly, and it's meant to live with the queued job, so
    it's discarded once the job is placed.
    """

    def __init__(self, strokes: Sequence[Stroke]) -> None:
        self._strokes = strokes
        self._cache: dict[FootprintCacheKey, FootprintCacheValue] = {}
        self._rotated: dict[float, list[Stroke]] = {}
        self._bodies: dict[BodySweepKey, BodySweep] = {}

    def _rotated_at(self, angle: float) -> list[Stroke]:
        hit = self._rotated.get(angle)
        if hit is None:
            hit = rotate_strokes(self._strokes, angle)
            self._rotated[angle] = hit
        return hit

    def body_at(
        self,
        angle: float,
        cell_mm: float,
        half_width_cells: int,
        body: RobotBody,
        margin: float,
    ) -> BodySweep:
        """The chassis swept along this rotation's ink — see ``BodySweep``.

        Rasterized in the same frame as ``at``'s footprint (same ``min_x``/``min_y``,
        just a wider pad), so the two masks differ only by a fixed cell offset. That
        keeps the narrow phase to a shift and an ``&`` per candidate instead of
        redrawing a few hundred polygons.
        """
        key = (angle, cell_mm, half_width_cells, body, margin)
        hit = self._bodies.get(key)
        if hit is not None:
            return hit

        rotated = self._rotated_at(angle)
        footprint, _ = self.at(angle, cell_mm, half_width_cells)
        assert footprint is not None

        pts = [p for stroke in rotated for p in stroke]
        min_x = min(p[0] for p in pts)
        min_y = min(p[1] for p in pts)
        max_x = max(p[0] for p in pts)
        max_y = max(p[1] for p in pts)

        # Wide enough for the chassis at any heading, plus the sub-cell margin.
        pad = int(math.ceil((body.reach_mm + margin) / cell_mm)) + 1
        width = int(math.ceil((max_x - min_x) / cell_mm)) + 2 * pad + 1
        height = int(math.ceil((max_y - min_y) / cell_mm)) + 2 * pad + 1

        img = Image.new("1", (width, height), 0)
        draw = ImageDraw.Draw(img)

        def to_px(p: Point) -> tuple[float, float]:
            return ((p[0] - min_x) / cell_mm + pad, (p[1] - min_y) / cell_mm + pad)

        for stroke in rotated:
            for a, b in zip(stroke[:-1], stroke[1:]):
                sweep = _body_sweep(a, b, body, margin)
                if sweep is not None:
                    draw.polygon([to_px(p) for p in sweep], fill=1)
            if len(stroke) == 1:
                cx, cy = to_px(stroke[0])
                r = (body.reach_mm + margin) / cell_mm
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=1)

        # ``rasterize`` uses the same min_x/min_y with its own (smaller) pad, so the
        # two origins differ by exactly the pad difference.
        delta = pad - footprint.pad
        hit = BodySweep(mask=np.array(img, dtype=np.bool_), d_row=delta, d_col=delta)
        self._bodies[key] = hit
        return hit

    def at(
        self, angle: float, cell_mm: float, half_width_cells: int
    ) -> tuple[Optional[Footprint], Optional[Point]]:
        """Rotated footprint at ``angle``, covering the ink only.

        The general keep-out buffer is *not* applied here — it's dilated onto the
        occupancy map instead (see ``Region.compute_occupancy``), which is the same
        clearance for one dilation per placement rather than one per rotation. It
        also keeps ``Region.commit`` stamping bare ink into the grid, so the buffer
        stays a property of the test rather than something baked into the region.
        """
        key = self._key(angle, cell_mm, half_width_cells)
        hit = self._cache.get(key)
        if hit is None:
            rotated = self._rotated_at(angle)
            footprint = rasterize(rotated, cell_mm, half_width_cells)
            anchor = rotated[0][0]
            hit = (footprint, anchor)
            self._cache[key] = hit
        return hit

    def _key(
        self, angle: float, cell_mm: float, half_width_cells: int
    ) -> FootprintCacheKey:
        return (angle, cell_mm, half_width_cells)


# --------------------------------------------------------------------------- #
# Region + Canvas
# --------------------------------------------------------------------------- #


@dataclass
class Region:
    """A rectangular area of a canvas drawn by one robot, with its occupancy grid."""

    id: str
    x: float  # min corner in global/canvas mm
    y: float
    width: float
    height: float
    robot: Optional[str] = None
    color: Optional[str] = "#000"
    config: PlacementConfig = field(default_factory=PlacementConfig)

    grid: Grid = field(init=False)

    def __post_init__(self) -> None:
        cols = max(1, int(math.ceil(self.width / self.config.cell_mm)))
        rows = max(1, int(math.ceil(self.height / self.config.cell_mm)))
        self.grid = np.zeros((rows, cols), dtype=np.uint8)

    @property
    def free_fraction(self) -> float:
        return 1.0 - float(self.grid.sum()) / float(self.grid.size)

    def add_drawings(self, drawings: list[PlacedDrawing]) -> None:
        """Rebuild the occupancy grid from already-placed drawings."""

        cell = self.config.cell_mm
        half = int(
            math.ceil((self.config.pen_mm / 2.0 + self.config.clearance_mm) / cell)
        )

        for drawing in drawings:
            footprint = rasterize(drawing.strokes, cell, half)
            if footprint is None:
                continue

            top = int(round((footprint.min_y - self.y) / cell))
            left = int(round((footprint.min_x - self.x) / cell))

            mask = footprint.mask
            mh, mw = mask.shape

            # Clip in case the saved drawing lies partially outside the region.
            grid_top = max(0, top)
            grid_left = max(0, left)
            grid_bottom = min(self.grid.shape[0], top + mh)
            grid_right = min(self.grid.shape[1], left + mw)

            if grid_bottom <= grid_top or grid_right <= grid_left:
                continue

            mask_top = grid_top - top
            mask_left = grid_left - left
            mask_bottom = mask_top + (grid_bottom - grid_top)
            mask_right = mask_left + (grid_right - grid_left)

            self.grid[grid_top:grid_bottom, grid_left:grid_right] |= mask[
                mask_top:mask_bottom, mask_left:mask_right
            ].astype(np.uint8)

    @property
    def _rect(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y) in global/canvas mm."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def adjoins(self, other: "Region", tol: float = 1e-6) -> bool:
        """True if a robot working ``other`` could reach into this region.

        Both are axis-aligned rects, so this is: they are not separated by a gap,
        and they meet in more than a point. That covers the two layouts in use —

        * **Tiled** regions that share an edge: flush on one axis, genuinely
          overlapping on the other.
        * **Stacked** regions that cover the same ground, which is how two robots
          are put on one shared canvas (see ``tests/test_active_drawings.py``). They
          overlap on both axes, and a robot in one is quite literally in the other.

        — while excluding regions that only touch at a corner (zero overlap on both
        axes: nothing reaches through a point) and any region separated by a gap.

        Note the corner exclusion is a modelling choice, not a physical fact: a bot
        drawing just past a shared corner is within reach, we just don't count it.
        """
        ax0, ay0, ax1, ay1 = self._rect
        bx0, by0, bx1, by1 = other._rect

        # Positive = overlap, zero = flush, negative = gap.
        x_overlap = min(ax1, bx1) - max(ax0, bx0)
        y_overlap = min(ay1, by1) - max(ay0, by0)

        if x_overlap < -tol or y_overlap < -tol:
            return False  # a gap on either axis separates them entirely
        if abs(x_overlap) <= tol and abs(y_overlap) <= tol:
            return False  # they meet at a corner and nowhere else
        return True

    def _active_robot_keepout(
        self,
        canvas: "Canvas",
        active_drawings: Mapping[str, Optional[Sequence[Stroke]]],
        body: RobotBody,
        clearance: float = 0.0,
    ) -> BoolMask:
        """Cells this region must not draw in because a *neighbouring* robot is there.

        A robot mid-drawing is a physical body, not just its ink. But we know more
        than "it's near its pen": the stroke it is laying down gives its heading as
        well as its position, so we block the space the body actually *sweeps* —
        one oriented rectangle per segment (see ``_body_sweep``) — rather than a
        disc of body-length around every ink point. The disc is the worst case over
        every orientation at once, and it over-blocks badly: at 226mm it claimed
        ~37% of a 500mm region from a single stroke.

        Only robots that are on this canvas and drawing in a region that shares an
        edge with ours can reach us; everything else is skipped.

        The strokes arrive in **global canvas mm** (``replay_to_world`` bakes in the
        placement anchor), while our grid is region-local — hence the ``ox``/``oy``
        shift below rather than a bare ``p / cell``.

        ``clearance`` grows the result by that many mm. Callers pass
        ``max(general_buffer, body.min_overhang_mm)`` — see ``compute_occupancy``.

        Not modelled: a robot pivoting in place at a stroke's corner sweeps a small
        wedge outside both adjoining rectangles. Arcs are flattened into ~6° steps
        so that wedge is slivers; a real spin between strokes is the case to watch.
        """

        blocked = np.zeros(self.grid.shape, dtype=np.bool_)
        cell = self.config.cell_mm

        # One cell of slop so the raster can't leave a sliver of body unblocked;
        # see ``_body_sweep``. Folded into the reach too, so the bbox reject below
        # stays a strict over-approximation of what we actually draw.
        margin = cell
        clearance = max(0.0, clearance)
        reach = body.reach_mm + margin + clearance
        if not active_drawings or reach <= 0:
            return blocked

        rows, cols = self.grid.shape

        # The zone a robot has to be inside to reach us at all: our rect grown by the
        # body's worst-case reach. Testing each neighbour's stroke bbox against it
        # first is the cheap half of "don't process what can't affect us" — a bot
        # drawing at the far end of an adjacent region costs only a bbox compare.
        zx0, zy0 = self.x - reach, self.y - reach
        zx1 = self.x + self.width + reach
        zy1 = self.y + self.height + reach

        relevant: list[Stroke] = []
        for name, strokes in active_drawings.items():
            if not strokes:
                continue  # idle, or between drawings (robots.py parks a None here)
            other = canvas.region_for_robot(name)
            if other is None or other.id == self.id:
                continue  # not on this canvas, or it's us — our own ink is committed
            if not self.adjoins(other):
                continue
            for stroke in strokes:
                if not stroke:
                    continue
                sxs = [p[0] for p in stroke]
                sys_ = [p[1] for p in stroke]
                if (
                    min(sxs) > zx1
                    or max(sxs) < zx0
                    or min(sys_) > zy1
                    or max(sys_) < zy0
                ):
                    continue  # bbox out of reach; a stroke never leaves its own bbox
                relevant.append(stroke)

        if not relevant:
            return blocked

        # With no clearance the swept rectangles go straight into our own grid: PIL
        # clips whatever hangs outside and a cell is either covered or it isn't.
        # With clearance we need ``pad`` cells of margin around us, because a body
        # sitting outside the region can still grow *into* it — and the distance
        # transform can only measure away from sources it can see.
        pad = int(math.ceil(clearance / cell)) + (1 if clearance > 0 else 0)
        box_rows, box_cols = rows + 2 * pad, cols + 2 * pad
        ox, oy = self.x - pad * cell, self.y - pad * cell

        img = Image.new("1", (box_cols, box_rows), 0)
        draw = ImageDraw.Draw(img)

        def to_cell(p: Point) -> tuple[float, float]:
            return ((p[0] - ox) / cell, (p[1] - oy) / cell)

        for stroke in relevant:
            for a, b in zip(stroke[:-1], stroke[1:]):
                sweep = _body_sweep(a, b, body, margin)
                if sweep is None:
                    continue  # zero-length segment: no heading, and no motion either
                draw.polygon([to_cell(p) for p in sweep], fill=1)
            if len(stroke) == 1:
                # A lone pen-down point says where the pen is but not which way the
                # robot faces, so fall back to the worst case over all headings.
                cx, cy = to_cell(stroke[0])
                r = (body.reach_mm + margin) / cell
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=1)

        swept: BoolMask = np.array(img, dtype=np.bool_)
        if clearance > 0 and swept.any():
            # Grow by ``clearance``. A dilation by a disc that big would need a
            # ~75-cell kernel; "within R of the body" is exactly
            # ``distanceTransform(...) <= R``, at two linear passes regardless of R.
            src: Grid = np.where(swept, 0, 255).astype(np.uint8)
            dist = cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            swept = dist <= clearance / cell

        return cast(BoolMask, swept[pad : pad + rows, pad : pad + cols])

    def compute_occupancy(
        self,
        *,
        general_buffer: float,
        canvas: "Canvas",
        active_drawings: Mapping[str, Optional[Sequence[Stroke]]],
        robot_body: RobotBody = ROBOT_BODY,
    ) -> BoolMask:
        """Snapshot the keep-out map ``try_place`` tests against. Two sources:

        * **Committed ink** in this region's grid, grown by ``general_buffer`` (mm) —
          the margin every new drawing leaves around an existing one. Growing the
          *occupied* side is equivalent to growing each candidate footprint
          (Minkowski sum) but costs one dilation per placement instead of one per
          candidate rotation — hence ``try_place`` takes the result rather than
          deriving it.

        * **Neighbouring robots mid-drawing**, from ``active_drawings`` (robot name ->
          strokes in global canvas mm, as ``robots.py`` records them). Those aren't
          ink to steer around, they're a moving obstacle: see
          ``_active_robot_keepout``. ``canvas`` is the canvas containing this region,
          needed to resolve which region each active robot is drawing in and whether
          it's close enough to matter.

        **This map alone does not prove two robots won't touch** — it is the *broad
        phase*. It tests our ink, but what collides is our chassis, which reaches
        ``robot_body.reach_mm`` from our pen. So the live-robot layer is grown by
        ``max(general_buffer, robot_body.min_overhang_mm)``: within the overhang a
        collision is certain at *any* heading, which prunes every hopeless placement
        while never rejecting a workable one. Survivors still need
        ``body_collides`` — the narrow phase — to rule out the ambiguous band
        between ``min_overhang_mm`` and ``reach_mm``.

        Why ``max`` and not a sum: those are two constraints over the same set, not
        two stacked margins. ``general_buffer`` keeps our *ink* off their *ink*; the
        overhang keeps our *pen* off their *body*. Their ink lies inside their swept
        body, so the wider of the two already implies the other — with the stock
        149mm chassis the 74.5mm overhang swallows the buffer whole.

        Returns a boolean array the same shape as ``self.grid`` (region-grid cells,
        ``True`` = keep out), so it drops straight into the FFT correlation.
        """

        occupied: BoolMask = self.grid.astype(np.bool_)

        buffer_cells = math.ceil(general_buffer / self.config.cell_mm)
        if buffer_cells > 0 and occupied.any():
            h, w = occupied.shape

            padded = np.pad(
                occupied.astype(np.uint8),
                buffer_cells,
                mode="constant",
                constant_values=0,
            )

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * buffer_cells + 1, 2 * buffer_cells + 1),
            )

            # cv2 is untyped here, so pin the result back to a known dtype rather
            # than letting Any leak into the return.
            dilated: BoolMask = cv2.dilate(padded, kernel).astype(np.bool_)

            # Crop back to the region grid: the buffer only pushes drawings apart
            # from each other, it doesn't extend the region's own bounds.
            occupied = dilated[
                buffer_cells : buffer_cells + h, buffer_cells : buffer_cells + w
            ]

        return occupied | self._active_robot_keepout(
            canvas,
            active_drawings,
            robot_body,
            clearance=max(general_buffer, robot_body.min_overhang_mm),
        )

    def prepare(
        self,
        *,
        general_buffer: float,
        canvas: "Canvas",
        active_drawings: Mapping[str, Optional[Sequence[Stroke]]],
        robot_body: RobotBody = ROBOT_BODY,
    ) -> PlacementContext:
        """Snapshot this region for a placement search. Call once, pass to ``try_place``.

        Everything here is a function of the region and its neighbours, not of the
        drawing being placed — so it survives every rotation ``try_place`` tries,
        every step of the coordinator's shrink-to-fit search, and every re-test of a
        still-queued drawing. Building it once is the difference between paying for
        the FFT and the neighbours' body sweep once or a dozen times per placement.

        Rebuild it whenever the region's ink or the live drawings change — i.e. after
        ``commit``, and on each poll.
        """
        occupancy = self.compute_occupancy(
            general_buffer=general_buffer,
            canvas=canvas,
            active_drawings=active_drawings,
            robot_body=robot_body,
        )
        # The same sweep again, ungrown: the broad phase needed it inflated by the
        # query robot's overhang, the narrow phase needs the bodies themselves. The
        # rasterization is ~0.15ms; only the inflation's distance transform is
        # expensive, and that isn't repeated here.
        active_bodies = self._active_robot_keepout(canvas, active_drawings, robot_body)

        rows, cols = self.grid.shape
        fft_shape = _fft_shape((rows, cols))
        grid_fft: Optional[Spectrum] = (
            np.fft.rfft2(occupancy.astype(np.float64), fft_shape)
            if occupancy.any()
            else None
        )

        bbox: Optional[tuple[int, int, int, int]] = None
        if active_bodies.any():
            live_rows = np.where(active_bodies.any(axis=1))[0]
            live_cols = np.where(active_bodies.any(axis=0))[0]
            pad = int(math.ceil(robot_body.reach_mm / self.config.cell_mm)) + 1
            bbox = (
                max(0, int(live_rows[0]) - pad),
                max(0, int(live_cols[0]) - pad),
                min(rows - 1, int(live_rows[-1]) + pad),
                min(cols - 1, int(live_cols[-1]) + pad),
            )

        return PlacementContext(
            occupancy=occupancy,
            grid_fft=grid_fft,
            fft_shape=fft_shape,
            active_bodies=active_bodies,
            active_reach_bbox=bbox,
            robot_body=robot_body,
        )

    def try_place(
        self,
        strokes: Sequence[Stroke],
        context: PlacementContext,
        rng: Optional[random.Random] = None,
        footprints: Optional["FootprintCache"] = None,
    ) -> Optional[Placement]:
        """Find a collision-free pose (rotation + offset) for ``strokes``, or None.

        For each candidate rotation we get the full map of collision-free offsets
        in one shot via FFT cross-correlation (see ``_free_offsets``), then select
        per ``config.strategy``: ``origin`` keeps the offset nearest the
        ``(0,0)`` corner (ties prefer the smaller angle, so drawings stay upright
        when rotating buys nothing); ``scatter`` picks a uniformly random free
        offset for an organic spread. ``search_step_cells`` subsamples the offset
        grid (coarser = faster, fewer candidate positions).

        This is the definitive answer to "can this drawing go here" — a pose it
        returns is one the robot can actually draw. That takes two tests, because
        two different things must not overlap: our *ink* must clear committed ink
        and the general buffer, and our *chassis* must clear any neighbouring robot
        mid-drawing. The first is the correlation below, against
        ``context.occupancy``. The second is ``body_collides``, run on candidates
        the first accepts — a grid cell holds one value, so it cannot also encode
        which way we'd be pointing when we reach it, and heading is exactly what
        decides whether 150mm of tail swings into the robot next door.

        The narrow phase only runs when a neighbour is actually live, and only for
        candidates whose footprint lands near one (``context.active_reach_bbox``).
        Nobody drawing next door — the usual case — costs nothing at all.

        ``context`` comes from ``Region.prepare``: it carries the occupancy, its
        FFT, and the neighbours' swept bodies. It's a required argument rather than
        something derived here because a single placement runs this search many
        times (once per rotation, and repeatedly while binary-searching the
        drawing's scale), and none of it changes across those.

        ``footprints`` is an optional ``FootprintCache`` for the drawing; the same
        queued drawing is re-tested across polls and candidate bots, and its
        rotated/rasterized footprints never change, so a shared cache avoids
        recomputing them. When omitted a throwaway cache is used (single call).

        DENSITY EXTENSION POINT (origin): ranking by the *mask's corner*
        position is a standard bottom-left-fill heuristic — good enough, not optimal,
        and it leaves the canvas fragmented once it fills. If packing density
        becomes a problem, upgrade the ranking to a true minimal-waste score: a
        contact-point metric (favour poses whose perimeter touches existing ink /
        walls) or full No-Fit-Polygon nesting. Either way the rest of the pipeline
        (footprint raster, occupancy grid, lead-in stripping) is unchanged — only
        the pose selection below changes.
        """

        if not strokes:
            return None

        cell = self.config.cell_mm
        half = int(
            math.ceil((self.config.pen_mm / 2.0 + self.config.clearance_mm) / cell)
        )
        step = max(1, self.config.search_step_cells)
        rows, cols = self.grid.shape

        if context.occupancy.shape != self.grid.shape:
            raise ValueError(
                f"occupancy shape {context.occupancy.shape} does not match region "
                f"{self.id!r} grid shape {self.grid.shape}"
            )

        if footprints is None:
            footprints = FootprintCache(strokes)

        scatter = self.config.strategy == "scatter"
        if scatter and rng is None:
            rng = random.Random()
        angles = list(self.config.angles_deg)
        if scatter:
            rng.shuffle(angles)  # type: ignore[union-attr]

        grid_fft = context.grid_fft
        narrow = context.has_live_neighbour  # nobody drawing next door -> no 2nd test

        best: Optional[Placement] = None
        best_key: Optional[tuple[int, int]] = None

        for angle in angles:
            footprint, anchor_local = footprints.at(angle, cell, half)
            if footprint is None:
                return None  # empty drawing — nothing to place at any angle
            assert anchor_local is not None  # set together with footprint
            mask = footprint.mask

            mh, mw = mask.shape
            if mh > rows or mw > cols:
                continue  # this rotation can't fit in the region at all

            if grid_fft is None:
                free = np.ones((rows - mh + 1, cols - mw + 1), dtype=np.bool_)
            else:
                free = _free_offsets(grid_fft, mask, (rows, cols), context.fft_shape)

            coords = np.argwhere(free[::step, ::step])  # row-major: corner-first
            if coords.size == 0:
                continue

            def at(index: int) -> tuple[tuple[int, int], Placement]:
                row, col = coords[index]
                pos = (int(row) * step, int(col) * step)
                return pos, self._placement(
                    angle,
                    cast(Point, anchor_local),
                    cast(Footprint, footprint),
                    pos,
                    cell,
                )

            if scatter:
                if not narrow:
                    # Nothing to veto, so take one uniform pick — same draw from
                    # ``rng`` as before the narrow phase existed, which keeps
                    # seeded scatter runs reproducible.
                    pos, candidate = at(rng.randrange(len(coords)))  # type: ignore[union-attr]
                    return candidate
                order = list(range(len(coords)))
                rng.shuffle(order)  # type: ignore[union-attr]
                for index in order:
                    pos, candidate = at(index)
                    if not self.body_collides(candidate, strokes, context, footprints):
                        return candidate
                continue  # every free offset at this angle would clip a live robot

            # ``origin``: coords are row-major, so the first that survives both
            # tests is this angle's best. Stop as soon as we can't beat the
            # incumbent — everything later in ``coords`` is further from the corner.
            for index in range(len(coords)):
                pos, candidate = at(index)
                if best_key is not None and pos >= best_key:
                    break
                if narrow and self.body_collides(
                    candidate, strokes, context, footprints
                ):
                    continue  # ink fits, chassis wouldn't
                best_key, best = pos, candidate
                break

            if best_key == (0, 0):
                # Nothing beats the corner itself, and a later angle that merely
                # ties loses to the `pos >= best_key` test above anyway — so the
                # remaining rotations cannot change the answer. Mostly this is the
                # empty-region case, where every angle is free at (0,0) and we'd
                # otherwise rasterize all of them to learn nothing.
                break

        return best

    def placed_strokes(
        self, placement: Placement, strokes: Sequence[Stroke]
    ) -> list[Stroke]:
        """``strokes`` as they will actually be drawn: rotated, in global canvas mm.

        Mirrors what ``robots.py`` reconstructs via ``replay_to_world`` — rotate by
        the placement's angle, then slide so the first ink point lands on the
        anchor — so the geometry here matches the ink the robot really lays down.
        """
        rotated = rotate_strokes(strokes, placement.angle_deg)
        ax, ay = rotated[0][0]
        dx, dy = placement.anchor_x - ax, placement.anchor_y - ay
        return [[(px + dx, py + dy) for px, py in stroke] for stroke in rotated]

    def _body_hits(
        self, sweep: BodySweep, top: int, left: int, context: PlacementContext
    ) -> bool:
        """Does ``sweep``, placed for an ink mask at ``(top, left)``, touch a live body?

        The whole narrow phase, once the sweep is built: shift, clip to the region,
        ``&``. Everything expensive already happened in ``FootprintCache.body_at``.
        """
        rows, cols = self.grid.shape
        bh, bw = sweep.mask.shape
        r0, c0 = top - sweep.d_row, left - sweep.d_col

        # Clip the sweep to the grid — it overhangs the ink, so it routinely pokes
        # outside the region, and a chassis outside our region is not our problem.
        lo_r, lo_c = max(0, r0), max(0, c0)
        hi_r, hi_c = min(rows, r0 + bh), min(cols, c0 + bw)
        if lo_r >= hi_r or lo_c >= hi_c:
            return False

        ours = sweep.mask[lo_r - r0 : hi_r - r0, lo_c - c0 : hi_c - c0]
        theirs = context.active_bodies[lo_r:hi_r, lo_c:hi_c]
        return bool((ours & theirs).any())

    def body_collides(
        self,
        placement: Placement,
        strokes: Sequence[Stroke],
        context: PlacementContext,
        footprints: Optional["FootprintCache"] = None,
    ) -> bool:
        """Narrow phase: would our chassis actually touch a live neighbour's?

        ``try_place`` runs this for you — it's public because it's the answer to a
        real question ("would this pose collide?"), not because you must remember
        to call it.

        ``context.occupancy`` is the broad phase, and it is conservative in exactly
        one direction: it rejects poses that are *certain* to collide (pen within
        ``min_overhang_mm`` of a live body, which the chassis contains at any
        heading). Between ``min_overhang_mm`` and ``reach_mm`` the answer depends on
        which way we end up pointing — and an occupancy grid cannot say that, since
        a cell holds one value and heading isn't a property of a cell.

        So: take the pose the search actually chose, sweep *our* body along *our*
        ink the same way ``_active_robot_keepout`` swept theirs, and intersect.
        Exact, and affordable because it runs once per surviving candidate rather
        than once per rotation — which is the whole reason the phases are split.

        ``footprints`` is the drawing's cache; pass the same one you gave
        ``try_place``. The swept chassis depends only on the rotation, so caching it
        turns each candidate into a shift and an ``&``. Omitted, a throwaway cache
        rebuilds it — fine for a one-off check, ruinous inside the search loop.

        Returns ``True`` if the placement must be rejected.
        """
        if not strokes or not context.has_live_neighbour:
            return False  # nobody live within reach: nothing to hit

        top, left = placement._top, placement._left
        mh, mw = placement._footprint.mask.shape

        # O(1) reject: our ink lands nowhere near a live robot, so no heading of
        # ours can reach one. Most candidates in a big region exit here, before we
        # even look at the sweep.
        btop, bleft, bbottom, bright = context.active_reach_bbox  # type: ignore[misc]
        if (
            top > bbottom
            or top + mh - 1 < btop
            or left > bright
            or left + mw - 1 < bleft
        ):
            return False

        if footprints is None:
            footprints = FootprintCache(strokes)
        cell = self.config.cell_mm
        half = int(
            math.ceil((self.config.pen_mm / 2.0 + self.config.clearance_mm) / cell)
        )
        sweep = footprints.body_at(
            placement.angle_deg, cell, half, context.robot_body, cell
        )
        return self._body_hits(sweep, top, left, context)

    def _placement(
        self,
        angle: float,
        anchor_local: Point,
        footprint: Footprint,
        pos: tuple[int, int],
        cell: float,
    ) -> Placement:
        top, left = pos
        anchor_col, anchor_row = footprint.to_px(anchor_local)  # first ink point
        return Placement(
            angle_deg=angle,
            anchor_x=self.x + (left + anchor_col) * cell,
            anchor_y=self.y + (top + anchor_row) * cell,
            _top=top,
            _left=left,
            _footprint=footprint,
        )

    def commit(self, placement: Placement) -> None:
        """Stamp a placed footprint into the occupancy grid (reserves the space)."""

        mask = placement._footprint.mask
        mh, mw = mask.shape
        top, left = placement._top, placement._left
        self.grid[top : top + mh, left : left + mw] |= mask.astype(np.uint8)

    def clear(self) -> None:
        self.grid[:] = 0


def edge_yaw(x: float, y: float, width: float, height: float) -> float:
    """Yaw (radians) of a marker sitting on the canvas boundary.

    Frame is screen-like (origin top-left, +x right, +y down). Yaw is the standard
    heading of the marker's inward-facing normal — ``atan2(ny, nx)`` measured from
    +x, increasing clockwise — i.e. the direction the marker faces into the canvas:
    left (x=0) faces right -> 0, top (y=0) faces down -> +pi/2,
    right (x=width) faces left -> pi, bottom (y=height) faces up -> -pi/2. The
    nearest edge wins; a corner ties two edges, and we break ties toward the
    horizontal edge (top/bottom) — consistent with the default corner markers.
    """
    d_top, d_bottom, d_left, d_right = abs(y), abs(height - y), abs(x), abs(width - x)
    nearest = min(d_top, d_bottom, d_left, d_right)
    if nearest == d_top:
        return math.pi / 2
    if nearest == d_bottom:
        return -math.pi / 2
    if nearest == d_left:
        return 0.0
    return math.pi


@dataclass
class Marker:
    id: int
    x: float
    y: float
    size_mm: Optional[float] = None
    yaw: Optional[float] = (
        None  # radians; derived from the canvas edge, not stored config
    )


@dataclass
class Canvas:
    id: str
    width: float
    height: float

    general_buffer: float
    """mm of clearance to leave between separate drawings."""

    markers: list[Marker] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)

    def region_for_robot(self, robot_name: str) -> Optional[Region]:
        for region in self.regions:
            if region.robot == robot_name:
                return region
        return None


class CanvasStore:
    """Holds the live canvases. Occupancy lives in the Region grids in memory."""

    def __init__(self, canvases: Optional[Iterable[Canvas]] = None) -> None:
        self._canvases: dict[str, Canvas] = {}
        for canvas in canvases or []:
            self._canvases[canvas.id] = canvas

    def all(self) -> list[Canvas]:
        return list(self._canvases.values())

    def get(self, canvas_id: str) -> Optional[Canvas]:
        return self._canvases.get(canvas_id)

    def upsert(self, canvas: Canvas) -> None:
        self._canvases[canvas.id] = canvas

    def remove(self, canvas_id: str) -> None:
        # Remove the canvas; raises KeyError if not found
        if canvas_id not in self._canvases:
            raise KeyError(canvas_id)
        del self._canvases[canvas_id]

    def all_markers(self) -> list[Marker]:
        return [m for canvas in self._canvases.values() for m in canvas.markers]

    def region_for_robot(self, robot_name: str) -> Optional[Region]:
        for canvas in self._canvases.values():
            region = canvas.region_for_robot(robot_name)
            if region is not None:
                return region
        return None

    def canvas_for_robot(self, robot_name: str) -> Optional[Canvas]:
        for canvas in self._canvases.values():
            region = canvas.region_for_robot(robot_name)
            if region is not None:
                return canvas
        return None
