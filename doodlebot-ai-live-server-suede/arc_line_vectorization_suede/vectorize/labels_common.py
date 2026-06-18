"""Shared command-labeling types and a geometry-only labeler.

Both vectorizers attach raw-segment provenance to their drawing
commands, but they get there by different routes:

* **low geometry** tracks provenance structurally — each primitive
  came from a known ``ChainPiece`` of a known polyline, so it maps
  command → primitive → final-polyline range → raw-segment span
  exactly (see ``low_geometry/labels.py``).

* **high geometry** is a frozen baseline whose primitives carry only
  geometry (endpoints, center/radius), with no back-pointer to the
  polyline or point range they came from. Threading index tracking
  through that pipeline would mean editing the frozen logic, so we
  recover its labels *geometrically* instead: sample each drawing
  command's path, find the nearest raw-segment pixel per sample, and
  compress the result into spans.

The geometric labeler here is deliberately general — it consumes only
the emitted command stream plus the raw segments, so it also works on
the route-optimized command stream (which neither structural labeler
covers, since ``OptimizeRoute`` reorders and re-emits transit moves).

Both routes produce the same ``CommandSpan`` / ``LabeledCommand``
types, defined here so consumers can treat low- and high-geometry
labels uniformly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from ..commands import DrawingCommand


@dataclass(frozen=True)
class CommandSpan:
    """One raw-segment contribution to a single drawing command.

    * ``raw_segment_id`` — pointer back to the raw segment list
      (``Segment.segments``).
    * ``raw_start`` / ``raw_end`` — INCLUSIVE indices in the raw
      segment; ``raw_end < raw_start`` means the command walks the raw
      segment in reverse. To slice regardless of direction, use
      ``raw[min(raw_start, raw_end) : max(raw_start, raw_end) + 1]``.
    * ``command_start_ratio`` / ``command_end_ratio`` — fraction along
      the command's drawn geometry (0 at its start, 1 at its end).
    """

    raw_segment_id: int
    raw_start: int
    raw_end: int
    command_start_ratio: float
    command_end_ratio: float


@dataclass
class LabeledCommand:
    """A command paired with what (if anything) it draws.

    ``primitive_id`` is ``None`` for transit / spin commands. For the
    structural (low-geometry) labeler it indexes the primitives list;
    for the geometric labeler it's the drawing command's ordinal in
    draw order (a stable handle, not an index into any primitive
    list). ``final_segment_index`` is the source polyline index when
    known (structural labeler) and ``None`` otherwise. ``spans`` tiles
    the drawing command's geometry — empty for non-drawing commands.
    """

    command: DrawingCommand
    primitive_id: Optional[int]
    final_segment_index: Optional[int]
    spans: List[CommandSpan] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Geometry-only labeler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DrawnGeometry:
    """The drawn path of one pen-down command, recovered by replaying
    the command stream. ``command_index`` ties it back to its slot in
    the command list."""

    command_index: int
    kind: str  # "line" | "arc"
    # Line:
    p0: NDArray[np.float64]
    p1: NDArray[np.float64]
    # Arc (unused for lines):
    center: Optional[NDArray[np.float64]]
    radius: float
    start_angle: float
    sweep: float
    length: float


def _simulate_drawn(
    commands: Sequence[DrawingCommand],
    start_pos: Tuple[float, float],
    start_heading: float,
) -> List[_DrawnGeometry]:
    """Replay ``commands`` and recover the spatial path of every
    pen-down ``line`` / ``arc`` command. Mirrors the simulator in
    ``release.visualize`` but keeps only what the labeler needs.
    """
    pos = np.asarray(start_pos, dtype=float)
    heading = float(start_heading)
    out: List[_DrawnGeometry] = []

    for idx, cmd in enumerate(commands):
        kind = cmd["kind"]
        if kind == "spin":
            heading += math.radians(cmd["degrees"])
        elif kind == "line":
            direction = np.array([math.cos(heading), math.sin(heading)])
            new_pos = pos + cmd["distance"] * direction
            if cmd["penDown"]:
                out.append(
                    _DrawnGeometry(
                        command_index=idx,
                        kind="line",
                        p0=pos.copy(),
                        p1=new_pos.copy(),
                        center=None,
                        radius=0.0,
                        start_angle=0.0,
                        sweep=0.0,
                        length=float(cmd["distance"]),
                    )
                )
            pos = new_pos
        elif kind == "arc":
            r = float(cmd["radius"])
            sweep = math.radians(cmd["degrees"])
            ccw = sweep > 0
            normal_angle = heading + (math.pi / 2 if ccw else -math.pi / 2)
            center = pos + r * np.array(
                [math.cos(normal_angle), math.sin(normal_angle)]
            )
            start_a = math.atan2(pos[1] - center[1], pos[0] - center[0])
            end_a = start_a + sweep
            new_pos = center + r * np.array([math.cos(end_a), math.sin(end_a)])
            out.append(
                _DrawnGeometry(
                    command_index=idx,
                    kind="arc",
                    p0=pos.copy(),
                    p1=new_pos.copy(),
                    center=center.copy(),
                    radius=r,
                    start_angle=start_a,
                    sweep=sweep,
                    length=abs(sweep) * r,
                )
            )
            pos = new_pos
            heading += sweep
        else:
            raise ValueError(f"Unknown command kind: {cmd!r}")

    return out


def _sample_geometry(
    geom: _DrawnGeometry, n_samples: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(ts, xys)`` of ``n_samples`` points along the drawn
    geometry: ``ts`` are parameters in [0, 1], ``xys`` their positions.
    """
    ts = np.linspace(0.0, 1.0, n_samples)
    if geom.kind == "line":
        xys = geom.p0[None, :] + ts[:, None] * (geom.p1 - geom.p0)[None, :]
        return ts, xys
    angles = geom.start_angle + ts * geom.sweep
    xys = geom.center[None, :] + geom.radius * np.stack(
        [np.cos(angles), np.sin(angles)], axis=1
    )
    return ts, xys


def label_commands_geometric(
    commands: Sequence[DrawingCommand],
    raw_segments: Sequence[NDArray[np.float64]],
    start_pos: Tuple[float, float] = (0.0, 0.0),
    start_heading: float = 0.0,
    samples_per_pixel: float = 1.0,
    min_samples: int = 8,
    max_samples: int = 4000,
) -> List[LabeledCommand]:
    """Label each drawing command by sampling its geometry and matching
    each sample to the nearest raw-segment pixel.

    Works on any command stream — high-geometry, low-geometry, or the
    route-optimized output — because it only reads the emitted
    geometry, not internal provenance.

    Arguments:
        commands: the command stream to label.
        raw_segments: ``Segment.segments`` (the trace output). Each
            entry's index is its raw-segment id.
        start_pos / start_heading: the robot pose ``commands`` was
            emitted from — MUST match, or the replayed geometry won't
            line up with the source.
        samples_per_pixel: sampling density along each command's path.
            One sample per pixel resolves raw-segment transitions to
            ~1px; raise for finer span boundaries.
        min_samples / max_samples: clamp the per-command sample count.

    Returns one ``LabeledCommand`` per command, in the same order.
    Non-drawing commands (spins, pen-ups) get an empty span list.
    """
    labeled: List[LabeledCommand] = [
        LabeledCommand(command=c, primitive_id=None, final_segment_index=None)
        for c in commands
    ]
    if not commands:
        return labeled

    # KDTree over every raw-segment pixel, remembering its raw id and
    # local index — the same construction the segment-stage labeler uses.
    all_points: List[NDArray[np.float64]] = []
    all_ids: List[NDArray[np.int32]] = []
    all_local: List[NDArray[np.int32]] = []
    for raw_id, poly in enumerate(raw_segments):
        arr = np.asarray(poly, dtype=float)
        if len(arr) == 0:
            continue
        all_points.append(arr)
        all_ids.append(np.full(len(arr), raw_id, dtype=np.int32))
        all_local.append(np.arange(len(arr), dtype=np.int32))
    if not all_points:
        return labeled

    raw_pixels = np.concatenate(all_points, axis=0)
    raw_ids = np.concatenate(all_ids, axis=0)
    raw_local = np.concatenate(all_local, axis=0)
    tree = cKDTree(raw_pixels)

    drawn = _simulate_drawn(commands, start_pos, start_heading)

    for draw_order, geom in enumerate(drawn):
        n = int(round(geom.length * samples_per_pixel))
        n = max(min_samples, min(max_samples, n))
        ts, xys = _sample_geometry(geom, n)

        _, idx = tree.query(xys, k=1)
        per_sample_id = raw_ids[idx]
        per_sample_local = raw_local[idx]

        spans: List[CommandSpan] = []
        i = 0
        while i < n:
            j = i + 1
            while j < n and per_sample_id[j] == per_sample_id[i]:
                j += 1
            spans.append(
                CommandSpan(
                    raw_segment_id=int(per_sample_id[i]),
                    raw_start=int(per_sample_local[i]),
                    raw_end=int(per_sample_local[j - 1]),
                    command_start_ratio=float(ts[i]),
                    command_end_ratio=float(ts[j - 1]),
                )
            )
            i = j

        labeled[geom.command_index] = LabeledCommand(
            command=commands[geom.command_index],
            primitive_id=draw_order,
            final_segment_index=None,
            spans=spans,
        )

    return labeled
