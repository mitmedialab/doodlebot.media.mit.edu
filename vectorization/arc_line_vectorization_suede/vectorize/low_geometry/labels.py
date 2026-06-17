"""Per-command labels: trace each drawing command back to raw segments.

After the vectorizer emits a flat command stream (``commands_consolidated``)
this module walks that stream and pairs each *drawing* command — a
pen-down ``line`` or any ``arc`` — with the raw-segment pieces that
fed the primitive it draws. Spins and pen-up transit commands carry
no label, since they don't correspond to any raw geometry.

The trip from a command to a raw-segment span is three hops:

* command → primitive: the tour's k-th entry produces 1-4 commands;
  the last one is always the drawing command (spins and pen-up
  movements are emitted first, by construction of ``to_commands``).

* primitive → final-polyline range: each primitive came from one
  ``ChainPiece``, which carries ``start_idx``/``end_idx`` into the
  *subsampled* polyline that ``fit_polyline`` saw. ``build_chains``
  applies a deterministic ``np.linspace`` subsample at
  ``polyline_subsample_cap``; we recompute it here to map subsampled
  indices back to raw (= final-segment) indices.

* final-polyline range → raw-segment spans: the segment stage's
  ``LabeledSegment`` already carries per-pixel ``raw_ids`` and
  ``raw_indices``, so we just slice through the relevant pixel range
  and compress consecutive same-id pixels into ``CommandSpan`` runs.

Each ``CommandSpan`` carries:

* ``raw_segment_id`` — pointer back to ``Segment.segments``.
* ``raw_start``/``raw_end`` — inclusive range within the raw segment;
  ``raw_end < raw_start`` means the command walks the raw segment in
  reverse (which happens whenever the tour traverses a primitive
  end → start). To slice the raw segment, use
  ``raw_segments[id][min(raw_start, raw_end) : max(raw_start, raw_end) + 1]``.
* ``command_start_ratio``/``command_end_ratio`` — fraction along the
  command's geometry (0 at the command's start, 1 at its end), so
  callers can map ratio → spatial position via the command's primitive
  type.

Most raw segments fit into one primitive entirely and therefore yield
exactly one ``CommandSpan`` covering ``[0, raw_len)`` and
``[0.0, 1.0]``; spans that don't cover the full raw segment fall out
of the same code path when a primitive consumed only part of a raw
segment (e.g. a corner split inside one raw stroke).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ...commands import DrawingCommand
from ...segment.labels import LabeledSegment
from ..labels_common import CommandSpan, LabeledCommand
from .primitives import Arc, Circle, Line, Primitive, endpoint
from .routing import _reverse_primitive  # internal helper, OK to reuse
from .solve import FittedSegment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_to_primitive(p: NDArray[np.float64], prim: Primitive) -> float:
    """Return the parameter ``t`` ∈ [0, 1] of the closest point on
    ``prim`` to ``p``.

    For a Line this is the unsigned projection onto the chord
    clamped to [0, 1]. For an Arc/Circle it's the angular distance
    from the start to the radial projection of ``p`` onto the
    primitive's circle, divided by the sweep (so 0 is at the
    primitive's start, 1 at its end). Out-of-range projections are
    clamped to the endpoints; on a degenerate primitive (zero
    length / radius) the function returns 0.0.
    """
    if isinstance(prim, Line):
        chord = prim.p1 - prim.p0
        L2 = float(np.dot(chord, chord))
        if L2 < 1e-12:
            return 0.0
        t = float(np.dot(p - prim.p0, chord) / L2)
        return max(0.0, min(1.0, t))
    if isinstance(prim, Arc):
        sweep = prim.sweep()
        if abs(sweep) < 1e-9:
            # Degraded to a line; project onto chord.
            return _project_to_primitive(p, Line(prim.p0, prim.p1))
        c = prim.center()
        theta_p = math.atan2(float(p[1] - c[1]), float(p[0] - c[0]))
        theta0 = prim.theta0()
        # Bring theta_p into the same revolution as theta0+sweep along
        # the sweep direction. The arc covers theta in
        # [theta0, theta0 + sweep] (sweep signed); we want the
        # smallest t with t * sweep + theta0 ≈ theta_p (modulo 2π).
        delta = (theta_p - theta0) if sweep > 0 else (theta0 - theta_p)
        delta %= 2.0 * math.pi
        # Normalize to [0, 2π); divide by |sweep|. For the arc's actual
        # angular extent (|sweep| in [0, 2π]) this lands in [0, 2π/|sweep|).
        t = delta / abs(sweep)
        return max(0.0, min(1.0, t))
    if isinstance(prim, Circle):
        theta_p = math.atan2(float(p[1] - prim.center[1]), float(p[0] - prim.center[0]))
        # Circle traversal goes CCW from theta = 0.
        if theta_p < 0:
            theta_p += 2.0 * math.pi
        t = theta_p / (2.0 * math.pi)
        return max(0.0, min(1.0, t))
    raise TypeError(f"unknown primitive type {type(prim)!r}")


def _subsample_index_map(
    raw_len: int,
    subsample_cap: int,
) -> NDArray[np.int32]:
    """Return the subsample-index array ``build_chains`` produced for a
    polyline of length ``raw_len``. Must match exactly so we can map
    chain-piece indices back to raw-polyline indices.
    """
    cap = max(2, int(subsample_cap))
    if raw_len > cap:
        return np.linspace(0, raw_len - 1, cap, dtype=np.int32)
    return np.arange(raw_len, dtype=np.int32)


def _replay_drawing_indices(
    prims: Sequence[Primitive],
    tour: List[Tuple[int, bool]],
    start_pos: NDArray[np.float64],
    start_heading: float,
    pen_up_join_tol: float,
) -> List[Optional[int]]:
    """Replay ``to_commands`` to find, for each tour entry, the index
    in the emitted command list that corresponds to the primitive's
    drawing command (the last command emitted for that iteration).
    Returns ``None`` for tour entries that emitted no drawing command
    at all (a zero-length degenerate primitive).

    This duplicates the *control flow* of ``to_commands`` but not its
    actual emission — we only care about whether each branch would
    append a command. Keeping this here rather than threading an
    output channel into ``to_commands`` avoids changing the public
    signature ``OptimizeRoute`` consumes.
    """
    cur_pos = np.asarray(start_pos, dtype=float)
    cur_heading = float(start_heading)
    drawing_idx_per_tour: List[Optional[int]] = []
    cmd_count = 0

    def wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def heading_change_deg(fh: float, th: float) -> float:
        return float(math.degrees(wrap_to_pi(th - fh)))

    def heading_of(vec: NDArray[np.float64]) -> float:
        return float(math.atan2(vec[1], vec[0]))

    for (pid, reverse) in tour:
        prim = prims[pid]
        if reverse:
            prim = _reverse_primitive(prim)

        # Start position of primitive.
        if isinstance(prim, Circle):
            start = prim.point_at(0.0)
        else:
            start = endpoint(prim, "start")

        # Pen-up transit (mirrors to_commands).
        gap = float(np.linalg.norm(start - cur_pos))
        if gap > pen_up_join_tol:
            target_heading = heading_of(start - cur_pos)
            spin_deg = heading_change_deg(cur_heading, target_heading)
            if abs(spin_deg) > 1e-3:
                cmd_count += 1
                cur_heading = target_heading
            cmd_count += 1  # pen-up line
            cur_pos = start.copy()

        # Decide primitive emission target.
        emit_as_line: Optional[float] = None
        if isinstance(prim, Arc):
            sweep_deg = float(math.degrees(prim.sweep()))
            r = prim.radius()
            sweep_rad = prim.sweep()
            if math.isfinite(r) and r > 0:
                sagitta = abs(r * (1.0 - math.cos(sweep_rad / 2.0)))
                chord_over_r = prim.chord() / r
            else:
                sagitta = 0.0
                chord_over_r = 0.0
            degenerate = (
                not math.isfinite(r)
                or r > 1e6
                or sagitta < 0.5
                or (abs(sweep_deg) > 270.0 and chord_over_r < 0.15)
            )
            if degenerate:
                emit_as_line = float(prim.chord())

        if isinstance(prim, Line) or emit_as_line is not None:
            chord_vec = prim.p1 - prim.p0
            primitive_start_heading = heading_of(chord_vec)
        elif isinstance(prim, Arc):
            from .primitives import tangent_at_end
            start_tan = tangent_at_end(prim, "start")
            primitive_start_heading = heading_of(start_tan)
        else:  # Circle
            start_tan = prim.tangent_at(0.0)
            primitive_start_heading = heading_of(start_tan)

        spin_deg = heading_change_deg(cur_heading, primitive_start_heading)
        if abs(spin_deg) > 1e-3:
            cmd_count += 1
            cur_heading = primitive_start_heading

        # Emit primitive.
        emitted_drawing_idx: Optional[int] = None
        if isinstance(prim, Line):
            d = prim.length()
            if d > 1e-9:
                emitted_drawing_idx = cmd_count
                cmd_count += 1
            cur_pos = prim.p1.copy()
            cur_heading = primitive_start_heading
        elif isinstance(prim, Arc):
            if emit_as_line is not None:
                if emit_as_line > 1e-9:
                    emitted_drawing_idx = cmd_count
                    cmd_count += 1
                cur_pos = prim.p1.copy()
                cur_heading = primitive_start_heading
            else:
                emitted_drawing_idx = cmd_count
                cmd_count += 1
                cur_pos = prim.p1.copy()
                # Update heading to end tangent.
                from .primitives import tangent_at_end
                end_tan = tangent_at_end(prim, "end")
                cur_heading = heading_of(end_tan)
        elif isinstance(prim, Circle):
            emitted_drawing_idx = cmd_count
            cmd_count += 1
        else:
            raise TypeError(type(prim))

        drawing_idx_per_tour.append(emitted_drawing_idx)

    return drawing_idx_per_tour


def _spans_for_drawing_command(
    prim: Primitive,
    reverse: bool,
    final_lo: int,
    final_hi: int,
    labeled: LabeledSegment,
) -> List[CommandSpan]:
    """Compress consecutive same-raw-id pixels across the command's
    portion of the labeled final polyline into ``CommandSpan`` runs.

    The pixels in ``labeled.points[final_lo:final_hi]`` correspond to
    the primitive's geometry; if the tour traverses the primitive in
    reverse, the pixels are walked in reverse order so the resulting
    spans are listed in command-traversal order.
    """
    if final_hi <= final_lo:
        return []

    sl = slice(final_lo, final_hi)
    points = labeled.points[sl]
    raw_ids = labeled.raw_ids[sl]
    raw_indices = labeled.raw_indices[sl]
    if reverse:
        # Walk from the primitive's start (which is the polyline's end
        # of the piece range, by construction of routing/_reverse) toward
        # its end.
        points = points[::-1]
        raw_ids = raw_ids[::-1]
        raw_indices = raw_indices[::-1]

    n = len(points)
    spans: List[CommandSpan] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and raw_ids[j] == raw_ids[i]:
            j += 1
        t_start = _project_to_primitive(points[i], prim)
        t_end = _project_to_primitive(points[j - 1], prim)
        spans.append(
            CommandSpan(
                raw_segment_id=int(raw_ids[i]),
                raw_start=int(raw_indices[i]),
                raw_end=int(raw_indices[j - 1]),
                command_start_ratio=t_start,
                command_end_ratio=t_end,
            )
        )
        i = j
    return spans


def label_commands(
    commands: Sequence[DrawingCommand],
    tour: List[Tuple[int, bool]],
    primitives: List[Primitive],
    fitted_segments: List[FittedSegment],
    polylines: Sequence[NDArray[np.float64]],
    labeled_segments: Sequence[LabeledSegment],
    polyline_subsample_cap: int,
    start_pos: NDArray[np.float64],
    start_heading: float,
    pen_up_join_tol: float,
) -> List[LabeledCommand]:
    """Build a per-command label list parallel to ``commands``.

    ``commands``, ``tour``, ``primitives``, ``fitted_segments``,
    ``start_pos``, ``start_heading``, ``pen_up_join_tol`` are the same
    objects ``to_commands`` consumed when producing ``commands``.
    ``polylines`` and ``labeled_segments`` come from the segment stage
    (``StrokeGraph.polylines`` and ``Segment.labeled_segments``).
    ``polyline_subsample_cap`` mirrors ``FitConfig.polyline_subsample_cap``.

    Returns one ``LabeledCommand`` per command, in the same order.
    """
    labeled: List[LabeledCommand] = [
        LabeledCommand(command=c, primitive_id=None, final_segment_index=None)
        for c in commands
    ]

    if not commands or not tour:
        return labeled

    # primitive_id -> (fitted_segment_index, piece_index)
    pid_to_seg_piece: Dict[int, Tuple[int, int]] = {}
    for seg_idx, seg in enumerate(fitted_segments):
        for piece_idx, pid in enumerate(seg.primitive_ids):
            pid_to_seg_piece[pid] = (seg_idx, piece_idx)

    drawing_indices = _replay_drawing_indices(
        primitives, tour, start_pos, start_heading, pen_up_join_tol
    )

    for tour_idx, (pid, reverse) in enumerate(tour):
        draw_cmd_idx = drawing_indices[tour_idx]
        if draw_cmd_idx is None or draw_cmd_idx >= len(commands):
            continue
        if pid not in pid_to_seg_piece:
            continue
        seg_idx, piece_idx = pid_to_seg_piece[pid]
        seg = fitted_segments[seg_idx]
        if seg.polyline_index >= len(polylines) or seg.polyline_index >= len(
            labeled_segments
        ):
            continue
        raw_poly = polylines[seg.polyline_index]
        ls = labeled_segments[seg.polyline_index]
        piece = seg.pieces[piece_idx]

        sub_idx = _subsample_index_map(len(raw_poly), polyline_subsample_cap)
        if piece.start_idx >= len(sub_idx) or piece.end_idx > len(sub_idx):
            continue
        final_lo = int(sub_idx[piece.start_idx])
        # piece.end_idx is exclusive in subsampled space; convert
        # ``sub_idx[end_idx-1]`` to raw-space then bump by one for a
        # half-open raw range.
        final_hi = int(sub_idx[piece.end_idx - 1]) + 1

        # Walking the primitive in command order may correspond to
        # walking the polyline forward or backward depending on which
        # endpoint the chain put first; the chain was built from the
        # polyline in forward order, so a reverse tour entry walks the
        # polyline backward through the piece's index range.
        prim = primitives[pid]
        emitted_prim = _reverse_primitive(prim) if reverse else prim
        spans = _spans_for_drawing_command(
            emitted_prim, reverse, final_lo, final_hi, ls
        )

        labeled[draw_cmd_idx] = LabeledCommand(
            command=commands[draw_cmd_idx],
            primitive_id=int(pid),
            final_segment_index=int(seg.polyline_index),
            spans=spans,
        )

    return labeled
