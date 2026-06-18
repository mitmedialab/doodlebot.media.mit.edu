"""Time-optimal re-ordering of an existing drawing-command sequence.

This is the REQUIRED final stage of the pipeline. The vectorizers and
``routing.py`` produce a command stream that is pen-up-minimal but not
time-minimal (the Eulerian router does not optimize in-place turning,
which is a large share of firmware draw time). This stage closes that
gap, and is what makes the low-geometry output reliably beat the naive
high-geometry baseline. Its contract is "never worse than the input":
it seeds local search from both a nearest-neighbour tour and the input
ordering, and falls back to the untouched input if neither wins.

Takes any ``List[DrawingCommand]`` (low or high geometry output), pulls
out each pen-down primitive (a ``line`` with ``penDown=True`` or an
``arc``), and re-sequences them to minimize total wall-clock drawing
time on the firmware's motion model.

What the optimizer touches:

* The order of pen-down primitives.
* Each primitive's traversal direction (forward / reverse).
* The pen-up jumps and alignment spins between them — these are
  regenerated from scratch to match the new order.

What it does NOT touch:

* The geometry of each individual primitive (a line of length L stays
  a line of length L; an arc of radius r / sweep θ keeps both).
* Anything inside a single line/arc command — only the inter-primitive
  transitions are re-planned.

The cost model is a port of the firmware's TMC429 motion estimator (see
``estimate_line_time`` / ``estimate_arc_time``). Distances in the input
commands are in *image pixels*; ``pixels_per_inch`` converts them to
physical inches before they're handed to the step-rate estimator.

A spin-in-place is handled as the radius=0 case of the arc estimator
(both wheels travel ±wheelbase × θ).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, TypedDict

import numpy as np
from numpy.typing import NDArray

from .commands import DrawingCommand

# ---------------------------------------------------------------------------
# Motion model (mirror of the firmware estimator in the task description)
# ---------------------------------------------------------------------------

MAX_TURN_SPEED = 3200  # steps/sec
ACCELERATION_MAX_DRAWING = 1000  # steps/sec^2

WHEELBASE_RADIUS = 2.93  # inches
STEP_LENGTH = 0.055  # inches per full step
MICROSTEPS_PER_STEP = 16

# inches per microstep
_INCHES_PER_MICROSTEP = STEP_LENGTH / MICROSTEPS_PER_STEP


def _estimate_motion_time(
    distance_steps: float, max_speed: float, acceleration: float
) -> float:
    distance_steps = abs(distance_steps)
    if distance_steps <= 0.0 or max_speed <= 0.0 or acceleration <= 0.0:
        return 0.0
    ramp_distance = (max_speed * max_speed) / acceleration
    if distance_steps <= ramp_distance:
        return 2.0 * math.sqrt(distance_steps / acceleration)
    accel_time = max_speed / acceleration
    cruise_distance = distance_steps - ramp_distance
    cruise_time = cruise_distance / max_speed
    return 2.0 * accel_time + cruise_time


def estimate_line_time(distance_steps: float, speed: float = 2000.0) -> float:
    """Time for a straight-line motion of ``distance_steps`` microsteps."""
    return _estimate_motion_time(abs(distance_steps), speed, ACCELERATION_MAX_DRAWING)


def estimate_arc_time(radius_inches: float, angle_deg: float) -> float:
    """Time for an arc of ``radius_inches`` (turn radius) and signed
    ``angle_deg``. Radius 0 reduces to a spin-in-place (both wheels at
    ±wheelbase × θ).
    """
    angle_deg = abs(angle_deg)

    outer_radius = WHEELBASE_RADIUS + radius_inches
    outer_circumference = 2.0 * math.pi * outer_radius
    outer_distance = (outer_circumference * (angle_deg / 360.0)) / _INCHES_PER_MICROSTEP

    inner_radius = abs(radius_inches - WHEELBASE_RADIUS)
    inner_circumference = 2.0 * math.pi * inner_radius
    inner_distance = (inner_circumference * (angle_deg / 360.0)) / _INCHES_PER_MICROSTEP

    if outer_distance <= 0.0:
        return 0.0

    outer_travel_time_no_accel = outer_distance / MAX_TURN_SPEED
    inner_speed = (
        abs(inner_distance / outer_travel_time_no_accel)
        if outer_travel_time_no_accel > 0.0
        else 0.0
    )

    if inner_speed <= MAX_TURN_SPEED:
        outer_accel = ACCELERATION_MAX_DRAWING
        inner_accel = (
            (outer_accel * inner_speed) / MAX_TURN_SPEED if MAX_TURN_SPEED > 0 else 0.0
        )
    else:
        inner_accel = ACCELERATION_MAX_DRAWING
        outer_accel = (
            (inner_accel * MAX_TURN_SPEED) / inner_speed if inner_speed > 0 else 0.0
        )

    outer_time = _estimate_motion_time(outer_distance, MAX_TURN_SPEED, outer_accel)
    inner_time = _estimate_motion_time(inner_distance, inner_speed, inner_accel)
    return max(outer_time, inner_time)


def estimate_spin_time(angle_deg: float) -> float:
    """Time to rotate in place by ``angle_deg`` (sign ignored)."""
    return estimate_arc_time(0.0, angle_deg)


# ---------------------------------------------------------------------------
# Command-sequence cost estimator (used for before/after reporting)
# ---------------------------------------------------------------------------


def estimate_total_time(
    commands: Sequence[DrawingCommand], pixels_per_inch: float = 1.0
) -> float:
    """Sum the firmware-model time for an entire command sequence."""
    if pixels_per_inch <= 0.0:
        raise ValueError("pixels_per_inch must be > 0")
    total = 0.0
    for c in commands:
        if c["kind"] == "spin":
            total += estimate_spin_time(c["degrees"])
        elif c["kind"] == "line":
            distance_inches = c["distance"] / pixels_per_inch
            distance_microsteps = distance_inches / _INCHES_PER_MICROSTEP
            total += estimate_line_time(distance_microsteps)
        elif c["kind"] == "arc":
            radius_inches = c["radius"] / pixels_per_inch
            total += estimate_arc_time(radius_inches, c["degrees"])
        else:
            raise ValueError(f"unknown command kind {c['kind']!r}")
    return total


# ---------------------------------------------------------------------------
# Command parsing: extract pen-down primitives from a command stream
# ---------------------------------------------------------------------------


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _heading_change_deg(from_heading: float, to_heading: float) -> float:
    """Signed degrees for the shortest rotation between two headings."""
    return math.degrees(_wrap_to_pi(to_heading - from_heading))


def _heading_of(vec: NDArray[np.float64]) -> float:
    return float(math.atan2(vec[1], vec[0]))


def _step_command(
    pos: NDArray[np.float64], heading: float, cmd: DrawingCommand
) -> Tuple[NDArray[np.float64], float]:
    """Return ``(pos, heading)`` after applying ``cmd``. Matches the
    semantics used in ``release.visualize._simulate`` so reconstructed
    commands draw the same picture.
    """
    if cmd["kind"] == "spin":
        return pos, heading + math.radians(cmd["degrees"])
    if cmd["kind"] == "line":
        d = float(cmd["distance"])
        new_pos = pos + d * np.array([math.cos(heading), math.sin(heading)])
        return new_pos, heading
    if cmd["kind"] == "arc":
        r = float(cmd["radius"])
        sweep = math.radians(cmd["degrees"])
        ccw = sweep > 0.0
        normal_angle = heading + (math.pi / 2.0 if ccw else -math.pi / 2.0)
        center = pos + r * np.array([math.cos(normal_angle), math.sin(normal_angle)])
        start_a = math.atan2(pos[1] - center[1], pos[0] - center[0])
        end_a = start_a + sweep
        new_pos = center + r * np.array([math.cos(end_a), math.sin(end_a)])
        return new_pos, heading + sweep
    raise ValueError(f"unknown command kind {cmd['kind']!r}")


@dataclass
class _Primitive:
    """One pen-down drawing command, expanded with the world-frame
    endpoints / headings the optimizer needs.
    """

    cmd: DrawingCommand  # the original line/arc command (always forward)
    entry_pos: NDArray[np.float64]
    entry_heading: float
    exit_pos: NDArray[np.float64]
    exit_heading: float
    draw_time: float  # firmware-model time for this primitive only


def _reverse_primitive(p: _Primitive) -> Tuple[DrawingCommand, float, float]:
    """Build the command + entry/exit heading for the reverse traversal.

    The reverse traversal starts at the original exit and ends at the
    original entry, with both headings flipped by π.

    * Line reversal: same distance, traveled in the opposite direction.
    * Arc reversal: same radius, negated sweep — keeps the same circle
      but flips CCW ↔ CW so the geometry is identical when entered with
      the flipped heading.
    """
    cmd = p.cmd
    if cmd["kind"] == "line":
        rev: DrawingCommand = {
            "kind": "line",
            "distance": cmd["distance"],
            "penDown": cmd["penDown"],
        }
    elif cmd["kind"] == "arc":
        rev = {
            "kind": "arc",
            "radius": cmd["radius"],
            "degrees": -cmd["degrees"],
        }
    else:
        raise ValueError(f"unreversable command kind {cmd['kind']!r}")
    return rev, p.exit_heading + math.pi, p.entry_heading + math.pi


def _extract_primitives(
    commands: Sequence[DrawingCommand],
    start_pos: NDArray[np.float64],
    start_heading: float,
    pixels_per_inch: float,
) -> List[_Primitive]:
    """Walk the command sequence, simulate motion, and pull out every
    pen-down primitive with its world-frame entry/exit state.

    Spins and pen-up lines are dropped — those are transitions the
    optimizer regenerates from scratch.
    """
    pos = np.asarray(start_pos, dtype=float).copy()
    heading = float(start_heading)
    out: List[_Primitive] = []

    for cmd in commands:
        if cmd["kind"] == "spin":
            heading = heading + math.radians(cmd["degrees"])
            continue
        if cmd["kind"] == "line" and not cmd["penDown"]:
            pos, heading = _step_command(pos, heading, cmd)
            continue

        entry_pos = pos.copy()
        entry_heading = heading
        new_pos, new_heading = _step_command(pos, heading, cmd)

        if cmd["kind"] == "line":
            distance_inches = cmd["distance"] / pixels_per_inch
            distance_microsteps = distance_inches / _INCHES_PER_MICROSTEP
            draw_time = estimate_line_time(distance_microsteps)
        elif cmd["kind"] == "arc":
            radius_inches = cmd["radius"] / pixels_per_inch
            draw_time = estimate_arc_time(radius_inches, cmd["degrees"])
        else:
            raise ValueError(f"unknown command kind {cmd['kind']!r}")

        out.append(
            _Primitive(
                cmd=cmd,
                entry_pos=entry_pos,
                entry_heading=entry_heading,
                exit_pos=new_pos.copy(),
                exit_heading=new_heading,
                draw_time=draw_time,
            )
        )

        pos = new_pos
        heading = new_heading

    return out


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------


# Entry pose for a (primitive_index, direction) choice — used by the
# command re-emitter to know where each primitive's pen-down starts.
def _entry(p: _Primitive, reverse: bool) -> Tuple[NDArray[np.float64], float]:
    if not reverse:
        return p.entry_pos, p.entry_heading
    return p.exit_pos, p.exit_heading + math.pi


# ---------------------------------------------------------------------------
# Precomputed cost cache
# ---------------------------------------------------------------------------


@dataclass
class _CostCache:
    """Vectorized lookups for the hot path.

    Layout: for each primitive ``i`` and direction ``r ∈ {0, 1}`` (0 =
    forward, 1 = reverse), store entry/exit pose. Then build:

    * ``trans[i, ri, j, rj]`` — transition from exit of (i, ri) to entry
      of (j, rj).
    * ``trans_start[j, rj]`` — transition from the robot's start state
      to entry of (j, rj).
    * ``draw[i]`` — drawing time of primitive ``i`` (direction-invariant).
    """

    trans: NDArray[np.float64]  # shape (N, 2, N, 2)
    trans_start: NDArray[np.float64]  # shape (N, 2)
    draw_sum: float


def _build_cost_cache(
    prims: List[_Primitive],
    start_pos: NDArray[np.float64],
    start_heading: float,
    pixels_per_inch: float,
    pen_up_join_tol: float,
) -> _CostCache:
    n = len(prims)
    # Stack entry / exit pose into (N, 2, ...) arrays.
    entry_pos = np.zeros((n, 2, 2), dtype=np.float64)  # (i, dir, xy)
    entry_heading = np.zeros((n, 2), dtype=np.float64)  # (i, dir)
    exit_pos = np.zeros((n, 2, 2), dtype=np.float64)
    exit_heading = np.zeros((n, 2), dtype=np.float64)
    draw_sum = 0.0
    for i, p in enumerate(prims):
        entry_pos[i, 0] = p.entry_pos
        entry_pos[i, 1] = p.exit_pos
        entry_heading[i, 0] = p.entry_heading
        entry_heading[i, 1] = p.exit_heading + math.pi
        exit_pos[i, 0] = p.exit_pos
        exit_pos[i, 1] = p.entry_pos
        exit_heading[i, 0] = p.exit_heading
        exit_heading[i, 1] = p.entry_heading + math.pi
        draw_sum += p.draw_time

    # Reshape to flat 2N axes for the pairwise broadcast, then unflatten.
    flat_entry_pos = entry_pos.reshape(2 * n, 2)
    flat_entry_h = entry_heading.reshape(2 * n)
    flat_exit_pos = exit_pos.reshape(2 * n, 2)
    flat_exit_h = exit_heading.reshape(2 * n)

    # gap_vec[a, b] = flat_entry_pos[b] - flat_exit_pos[a]
    gap_vec = flat_entry_pos[None, :, :] - flat_exit_pos[:, None, :]
    gap_dist = np.linalg.norm(gap_vec, axis=-1)  # (2N, 2N)
    has_pen_up = gap_dist > pen_up_join_tol

    # Spin times for the "no pen-up" branch: just the heading change
    # between exit_h[a] and entry_h[b].
    delta = flat_entry_h[None, :] - flat_exit_h[:, None]
    short_delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
    short_spin_deg = np.abs(np.degrees(short_delta))

    spin_only_time = _vec_spin_time(short_spin_deg)

    # Spin1: from exit_h to gap_heading; Spin2: from gap_heading to entry_h.
    gap_heading = np.arctan2(gap_vec[..., 1], gap_vec[..., 0])
    spin1_delta = gap_heading - flat_exit_h[:, None]
    spin2_delta = flat_entry_h[None, :] - gap_heading
    spin1_deg = np.abs(np.degrees((spin1_delta + math.pi) % (2.0 * math.pi) - math.pi))
    spin2_deg = np.abs(np.degrees((spin2_delta + math.pi) % (2.0 * math.pi) - math.pi))
    spin1_time = _vec_spin_time(spin1_deg)
    spin2_time = _vec_spin_time(spin2_deg)
    gap_microsteps = (gap_dist / pixels_per_inch) / _INCHES_PER_MICROSTEP
    line_time = _vec_line_time(gap_microsteps)
    pen_up_time = spin1_time + line_time + spin2_time

    full = np.where(has_pen_up, pen_up_time, spin_only_time)
    trans = full.reshape(n, 2, n, 2)

    # Start row.
    start_exit_pos = np.asarray(start_pos, dtype=float).reshape(1, 2)
    start_exit_h = np.array([float(start_heading)])
    gap_vec_s = flat_entry_pos - start_exit_pos  # (2N, 2)
    gap_dist_s = np.linalg.norm(gap_vec_s, axis=-1)
    has_pen_up_s = gap_dist_s > pen_up_join_tol
    delta_s = flat_entry_h - start_exit_h
    short_delta_s = (delta_s + math.pi) % (2.0 * math.pi) - math.pi
    spin_only_s = _vec_spin_time(np.abs(np.degrees(short_delta_s)))
    gap_heading_s = np.arctan2(gap_vec_s[..., 1], gap_vec_s[..., 0])
    spin1_d = gap_heading_s - start_exit_h
    spin2_d = flat_entry_h - gap_heading_s
    spin1_deg_s = np.abs(np.degrees((spin1_d + math.pi) % (2.0 * math.pi) - math.pi))
    spin2_deg_s = np.abs(np.degrees((spin2_d + math.pi) % (2.0 * math.pi) - math.pi))
    gap_microsteps_s = (gap_dist_s / pixels_per_inch) / _INCHES_PER_MICROSTEP
    pen_up_s = (
        _vec_spin_time(spin1_deg_s)
        + _vec_line_time(gap_microsteps_s)
        + _vec_spin_time(spin2_deg_s)
    )
    start_full = np.where(has_pen_up_s, pen_up_s, spin_only_s)
    trans_start = start_full.reshape(n, 2)

    return _CostCache(trans=trans, trans_start=trans_start, draw_sum=draw_sum)


def _vec_line_time(distance_steps: NDArray[np.float64]) -> NDArray[np.float64]:
    """Vectorized ``estimate_line_time``."""
    d = np.abs(distance_steps)
    max_speed = 2000.0
    accel = ACCELERATION_MAX_DRAWING
    ramp = (max_speed * max_speed) / accel
    triangle = 2.0 * np.sqrt(np.maximum(d, 0.0) / accel)
    accel_time = max_speed / accel
    cruise = np.maximum(d - ramp, 0.0) / max_speed
    trapezoid = 2.0 * accel_time + cruise
    out = np.where(d <= ramp, triangle, trapezoid)
    return np.where(d > 0.0, out, 0.0)


def _vec_spin_time(angle_deg: NDArray[np.float64]) -> NDArray[np.float64]:
    """Vectorized ``estimate_spin_time`` (= ``estimate_arc_time`` at r=0).
    Both wheels travel ``wheelbase * θ`` so the slower wheel is just one
    of the two, both with full drawing acceleration and full speed.
    """
    angle = np.abs(angle_deg)
    circ = 2.0 * math.pi * WHEELBASE_RADIUS
    distance = (circ * (angle / 360.0)) / _INCHES_PER_MICROSTEP
    return _vec_line_time_with(distance, MAX_TURN_SPEED, ACCELERATION_MAX_DRAWING)


def _vec_line_time_with(
    distance_steps: NDArray[np.float64], max_speed: float, accel: float
) -> NDArray[np.float64]:
    d = np.abs(distance_steps)
    ramp = (max_speed * max_speed) / accel
    triangle = 2.0 * np.sqrt(np.maximum(d, 0.0) / accel)
    accel_time = max_speed / accel
    cruise = np.maximum(d - ramp, 0.0) / max_speed
    trapezoid = 2.0 * accel_time + cruise
    out = np.where(d <= ramp, triangle, trapezoid)
    return np.where(d > 0.0, out, 0.0)


# ---------------------------------------------------------------------------
# Solver: nearest-neighbor + 2-opt with direction flips
# ---------------------------------------------------------------------------


# When True, every incremental delta computed by ``_two_opt`` / ``_or_opt``
# is cross-checked against a full ``_tour_cost_cached`` recompute. Kept off
# in normal operation (it defeats the point of the incremental delta); flip
# to True to validate the delta math after touching the cost model.
_VERIFY_DELTAS = False


@dataclass
class _Tour:
    order: List[int]  # primitive indices in visit order
    reverse: List[bool]  # whether each is traversed reversed

    def copy(self) -> "_Tour":
        return _Tour(list(self.order), list(self.reverse))


def _tour_cost_cached(tour: _Tour, cache: _CostCache) -> float:
    """Total firmware-model time using a precomputed transition matrix.

    Drawing time is direction-invariant and stored once as a scalar; we
    just sum the N transition lookups plus the per-tour ``draw_sum``.
    """
    order = tour.order
    reverse = tour.reverse
    if not order:
        return 0.0
    trans = cache.trans
    trans_start = cache.trans_start
    total = float(trans_start[order[0], 1 if reverse[0] else 0])
    for k in range(len(order) - 1):
        a = order[k]
        ra = 1 if reverse[k] else 0
        b = order[k + 1]
        rb = 1 if reverse[k + 1] else 0
        total += float(trans[a, ra, b, rb])
    return total + cache.draw_sum


def _edge_cost(
    cache: _CostCache,
    prev: Optional[Tuple[int, int]],
    node: Tuple[int, int],
    is_first: bool,
) -> float:
    """Cost of the directed edge entering ``node`` (a ``(prim, dir)``
    pair). If ``is_first`` the edge comes from the robot start state
    (``trans_start``); otherwise from ``prev``.

    The cost-cache transition matrix has a reverse-traversal symmetry
    that the incremental 2-opt / or-opt deltas below rely on:

        trans[a, ra, b, rb] == trans[b, 1-rb, a, 1-ra]

    i.e. reversing a directed edge and flipping both endpoints' traversal
    directions leaves the transition cost unchanged. (Provable from
    ``_build_cost_cache``: entry/exit pose of the flipped traversal is
    the other endpoint's pose with heading rotated by pi; the gap vector
    negates, gap distance is invariant, and every spin magnitude is
    invariant under the simultaneous pi-rotation of both headings.)

    Consequence: when a 2-opt move reverses a tour slice (and flips the
    direction of every primitive in it), the *internal* transitions of
    that slice keep their cost — only the two boundary transitions
    change. Same for an or-opt segment relocation. That is what makes
    the delta O(1) instead of O(n).
    """
    if is_first:
        return float(cache.trans_start[node[0], node[1]])
    assert prev is not None
    return float(cache.trans[prev[0], prev[1], node[0], node[1]])


def _nearest_neighbor(cache: _CostCache, n: int) -> _Tour:
    """Greedy seed: at each step pick the (primitive, direction) with
    the lowest cached transition cost from the current state.
    """
    used = np.zeros(n, dtype=bool)
    order: List[int] = []
    reverse: List[bool] = []
    # First step uses trans_start.
    flat_start = cache.trans_start.reshape(2 * n)
    # mask out used = none yet, so just argmin over all
    best = int(np.argmin(flat_start))
    j0, r0 = divmod(best, 2)
    order.append(j0)
    reverse.append(bool(r0))
    used[j0] = True
    cur = j0
    cur_r = r0
    for _ in range(n - 1):
        # Row of transitions from (cur, cur_r), mask out used.
        row = cache.trans[cur, cur_r]  # shape (n, 2)
        masked = np.where(used[:, None], np.inf, row)
        idx_flat = int(np.argmin(masked))
        j, r = divmod(idx_flat, 2)
        order.append(j)
        reverse.append(bool(r))
        used[j] = True
        cur = j
        cur_r = r
    return _Tour(order, reverse)


def _two_opt(
    tour: _Tour,
    cache: _CostCache,
    max_passes: int = 8,
) -> _Tour:
    """2-opt with segment reversal. Reversing the slice ``tour[i..j]``
    also flips the direction of every primitive in the slice — that's
    how direction is searched here.

    Each candidate move is evaluated by an O(1) incremental delta: the
    reversal preserves the cost of every transition *inside* the slice
    (see ``_edge_cost`` for why), so only the two boundary transitions
    — the edge entering position ``i`` and the edge leaving position
    ``j`` — change. Total work per pass is O(n^2) move evaluations,
    each O(1), instead of O(n^3).
    """
    n = len(tour.order)
    if n < 3:
        return tour
    order = list(tour.order)
    rev = [1 if x else 0 for x in tour.reverse]

    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                oi, ri = order[i], rev[i]
                oj, rj = order[j], rev[j]
                # Boundary edge entering position i.
                if i == 0:
                    old_enter = _edge_cost(cache, None, (oi, ri), True)
                    new_enter = _edge_cost(cache, None, (oj, 1 - rj), True)
                else:
                    prev = (order[i - 1], rev[i - 1])
                    old_enter = _edge_cost(cache, prev, (oi, ri), False)
                    new_enter = _edge_cost(cache, prev, (oj, 1 - rj), False)
                # Boundary edge leaving position j.
                if j == n - 1:
                    old_leave = 0.0
                    new_leave = 0.0
                else:
                    succ = (order[j + 1], rev[j + 1])
                    old_leave = _edge_cost(cache, (oj, rj), succ, False)
                    new_leave = _edge_cost(cache, (oi, 1 - ri), succ, False)
                delta = (new_enter + new_leave) - (old_enter + old_leave)
                if delta < -1e-9:
                    if _VERIFY_DELTAS:
                        before = _tour_cost_cached(
                            _Tour(order, [bool(x) for x in rev]), cache
                        )
                    order[i : j + 1] = order[i : j + 1][::-1]
                    seg = rev[i : j + 1]
                    rev[i : j + 1] = [1 - x for x in reversed(seg)]
                    if _VERIFY_DELTAS:
                        after = _tour_cost_cached(
                            _Tour(order, [bool(x) for x in rev]), cache
                        )
                        assert abs((after - before) - delta) < 1e-6, (
                            f"2-opt delta mismatch: predicted {delta}, "
                            f"actual {after - before}"
                        )
                    improved = True
        if not improved:
            break
    return _Tour(order, [bool(x) for x in rev])


def _or_opt(
    tour: _Tour,
    cache: _CostCache,
    max_segment_len: int = 3,
    max_passes: int = 4,
) -> _Tour:
    """Or-opt: relocate a short consecutive run, optionally flipped.
    Catches moves 2-opt can't represent.

    Like ``_two_opt`` this uses an O(1) incremental delta per candidate.
    Removing a segment destroys its two incident transitions and adds a
    closure transition; reinserting it splits one transition and adds
    two. The segment's internal transitions are cost-invariant under the
    optional flip (``_edge_cost``), so they cancel and never enter the
    delta. Per pass: O(seg_len * n^2) instead of O(seg_len * n^3).
    """
    n = len(tour.order)
    if n < 4:
        return tour
    order = list(tour.order)
    rev = [1 if x else 0 for x in tour.reverse]

    for _ in range(max_passes):
        improved = False
        for seg_len in range(1, max_segment_len + 1):
            i = 0
            while i + seg_len <= n:
                # Segment B[i .. i+seg_len-1]; pre/post are its
                # neighbours in the current tour.
                pre = (order[i - 1], rev[i - 1]) if i > 0 else None
                has_post = i + seg_len < n
                post = (
                    (order[i + seg_len], rev[i + seg_len])
                    if has_post
                    else None
                )
                seg_nodes = [
                    (order[i + k], rev[i + k]) for k in range(seg_len)
                ]
                # Remaining tour (segment cut out).
                rem = (
                    [(order[k], rev[k]) for k in range(i)]
                    + [(order[k], rev[k]) for k in range(i + seg_len, n)]
                )
                m = len(rem)

                # Edges destroyed by the cut, and the closure edge that
                # heals the gap in the remaining tour.
                d1 = _edge_cost(cache, pre, seg_nodes[0], i == 0)
                d2 = (
                    _edge_cost(cache, seg_nodes[-1], post, False)
                    if post is not None
                    else 0.0
                )
                g_close = (
                    _edge_cost(cache, pre, post, pre is None)
                    if post is not None
                    else 0.0
                )

                best_delta = -1e-9
                best_move: Optional[Tuple[int, bool]] = None
                for g in range(m + 1):
                    rg_prev = rem[g - 1] if g > 0 else None
                    rg = rem[g] if g < m else None
                    r_split = (
                        _edge_cost(cache, rg_prev, rg, g == 0)
                        if rg is not None
                        else 0.0
                    )
                    for flip in (False, True):
                        if flip:
                            seg_first = (
                                seg_nodes[-1][0],
                                1 - seg_nodes[-1][1],
                            )
                            seg_last = (
                                seg_nodes[0][0],
                                1 - seg_nodes[0][1],
                            )
                        else:
                            seg_first = seg_nodes[0]
                            seg_last = seg_nodes[-1]
                        a1 = _edge_cost(cache, rg_prev, seg_first, g == 0)
                        a2 = (
                            _edge_cost(cache, seg_last, rg, False)
                            if rg is not None
                            else 0.0
                        )
                        delta = (g_close + a1 + a2) - (d1 + d2 + r_split)
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (g, flip)

                if best_move is not None:
                    g, flip = best_move
                    if flip:
                        seg_out = [
                            (p, 1 - r) for (p, r) in reversed(seg_nodes)
                        ]
                    else:
                        seg_out = list(seg_nodes)
                    new_nodes = rem[:g] + seg_out + rem[g:]
                    if _VERIFY_DELTAS:
                        before = _tour_cost_cached(
                            _Tour(order, [bool(x) for x in rev]), cache
                        )
                    order = [p for (p, _) in new_nodes]
                    rev = [r for (_, r) in new_nodes]
                    if _VERIFY_DELTAS:
                        after = _tour_cost_cached(
                            _Tour(order, [bool(x) for x in rev]), cache
                        )
                        assert abs((after - before) - best_delta) < 1e-6, (
                            f"or-opt delta mismatch: predicted {best_delta}, "
                            f"actual {after - before}"
                        )
                    improved = True
                    # Tour mutated; rescan this seg_len from the start.
                    i = 0
                    continue
                i += 1
        if not improved:
            break
    return _Tour(order, [bool(x) for x in rev])


# ---------------------------------------------------------------------------
# Command re-emission
# ---------------------------------------------------------------------------


def _emit_transition(
    cur_pos: NDArray[np.float64],
    cur_heading: float,
    entry_pos: NDArray[np.float64],
    entry_heading: float,
    pen_up_join_tol: float,
    cmds: List[DrawingCommand],
) -> Tuple[NDArray[np.float64], float]:
    """Append the spin/pen-up/spin sequence needed to move from
    ``(cur_pos, cur_heading)`` to ``(entry_pos, entry_heading)``. Returns
    the updated pose.
    """
    gap_vec = entry_pos - cur_pos
    gap = float(np.linalg.norm(gap_vec))
    if gap > pen_up_join_tol:
        gap_heading = _heading_of(gap_vec)
        spin1_deg = _heading_change_deg(cur_heading, gap_heading)
        if abs(spin1_deg) > 1e-3:
            cmds.append({"kind": "spin", "degrees": float(spin1_deg)})
            cur_heading = gap_heading
        cmds.append({"kind": "line", "distance": float(gap), "penDown": False})
        cur_pos = entry_pos.copy()
    spin2_deg = _heading_change_deg(cur_heading, entry_heading)
    if abs(spin2_deg) > 1e-3:
        cmds.append({"kind": "spin", "degrees": float(spin2_deg)})
        cur_heading = entry_heading
    return cur_pos, cur_heading


def _emit_tour(
    tour: _Tour,
    prims: List[_Primitive],
    start_pos: NDArray[np.float64],
    start_heading: float,
    pen_up_join_tol: float,
) -> List[DrawingCommand]:
    out: List[DrawingCommand] = []
    cur_pos = np.asarray(start_pos, dtype=float).copy()
    cur_heading = float(start_heading)
    for idx, rev in zip(tour.order, tour.reverse):
        p = prims[idx]
        ep, eh = _entry(p, rev)
        cur_pos, cur_heading = _emit_transition(
            cur_pos, cur_heading, ep, eh, pen_up_join_tol, out
        )
        if rev:
            rev_cmd, _entry_h, exit_h = _reverse_primitive(p)
            out.append(rev_cmd)
            # Simulate to keep cur_pos / cur_heading consistent with the
            # actual emitted command (rather than trusting analytical
            # exit data, which could drift on degenerate arcs).
            cur_pos, cur_heading = _step_command(cur_pos, cur_heading, rev_cmd)
        else:
            out.append(p.cmd)
            cur_pos, cur_heading = _step_command(cur_pos, cur_heading, p.cmd)
    return out


# ---------------------------------------------------------------------------
# Public Config + class
# ---------------------------------------------------------------------------


class OptimizeDict(TypedDict, total=False):
    pixels_per_inch: float
    pen_up_join_tol: float
    two_opt_passes: int
    or_opt_passes: int
    or_opt_max_segment_len: int


class OptimizeRoute:
    """Re-order a drawing-command sequence to minimize firmware-model
    drawing time.

    Args:
        commands: the command sequence to optimize (e.g.
            ``LowGeometryVectorize(...).commands_consolidated`` or
            ``HighGeometryVectorize(...).commands``).
        start_pos: robot's starting position in image coordinates. MUST
            match the start used when generating the input commands.
        start_heading: robot's starting heading in radians. MUST match
            the start used when generating the input commands.
        optimize: solver tuning (see ``OptimizeDict``).

    Result fields:
        commands: the re-ordered command sequence.
        estimated_time_before / estimated_time_after: firmware-model
            total time for input / output, in seconds.
    """

    class Config:
        Optimize = OptimizeDict

    def __init__(
        self,
        commands: Sequence[DrawingCommand],
        start_pos: NDArray[np.float64],
        start_heading: float,
        cfg: OptimizeDict,
    ):
        self.pixels_per_inch = float(cfg.get("pixels_per_inch", 1.0))
        if self.pixels_per_inch <= 0.0:
            raise ValueError("pixels_per_inch must be > 0")
        self.pen_up_join_tol = float(cfg.get("pen_up_join_tol", 0.5))
        self.two_opt_passes = int(cfg.get("two_opt_passes", 8))
        self.or_opt_passes = int(cfg.get("or_opt_passes", 4))
        self.or_opt_max_segment_len = int(cfg.get("or_opt_max_segment_len", 3))

        self.input_commands: List[DrawingCommand] = list(commands)
        self.start_pos = np.asarray(start_pos, dtype=float).copy()
        self.start_heading = float(start_heading)

        self._run()

    def _run(self) -> None:
        self.estimated_time_before = estimate_total_time(
            self.input_commands, self.pixels_per_inch
        )

        primitives = _extract_primitives(
            self.input_commands,
            self.start_pos,
            self.start_heading,
            self.pixels_per_inch,
        )
        self._primitives = primitives

        if not primitives:
            self.commands: List[DrawingCommand] = list(self.input_commands)
            self.estimated_time_after = self.estimated_time_before
            self.tour: List[Tuple[int, bool]] = []
            self.improved = False
            return

        n = len(primitives)
        cache = _build_cost_cache(
            primitives,
            self.start_pos,
            self.start_heading,
            self.pixels_per_inch,
            self.pen_up_join_tol,
        )

        # Local search is seeded from TWO starting tours and the best
        # result is kept:
        #   1. a nearest-neighbour greedy tour, and
        #   2. the input ordering itself.
        # ``_extract_primitives`` returns primitives in input order, all
        # traversed forward, so ``_Tour(range(n), [False]*n)`` is exactly
        # the order this optimizer was handed. Seeding from it guarantees
        # 2-opt/or-opt can never converge to something worse than the
        # input — without this seed, a nearest-neighbour start can land
        # in a basin whose local optimum is slower than the (often
        # already decent) Eulerian routing the input came from. This is
        # the bug that made ``beachnugget`` regress by ~2%.
        seeds = [
            _nearest_neighbor(cache, n),
            _Tour(list(range(n)), [False] * n),
        ]
        best_tour: Optional[_Tour] = None
        best_cost = float("inf")
        for seed in seeds:
            t = _two_opt(seed, cache, max_passes=self.two_opt_passes)
            t = _or_opt(
                t,
                cache,
                max_segment_len=self.or_opt_max_segment_len,
                max_passes=self.or_opt_passes,
            )
            c = _tour_cost_cached(t, cache)
            if c < best_cost:
                best_cost = c
                best_tour = t
        assert best_tour is not None

        self.commands = _emit_tour(
            best_tour,
            primitives,
            self.start_pos,
            self.start_heading,
            self.pen_up_join_tol,
        )
        self.tour = list(zip(best_tour.order, best_tour.reverse))
        self.estimated_time_after = estimate_total_time(
            self.commands, self.pixels_per_inch
        )

        # Final safety net: if the re-emitted tour still came out slower
        # than the literal input command stream (possible when the input
        # was already near-optimal and command re-emission introduced a
        # rounding-scale difference), fall back to the input untouched.
        # The optimizer's contract is "never worse than the input".
        if self.estimated_time_after > self.estimated_time_before + 1e-9:
            self.commands = list(self.input_commands)
            self.estimated_time_after = self.estimated_time_before
            self.tour = []
            self.improved = False
        else:
            self.improved = self.estimated_time_after < self.estimated_time_before

    def stats(self) -> str:
        n_prim = len(self._primitives)
        before = self.estimated_time_before
        after = self.estimated_time_after
        if before > 0:
            pct = 100.0 * (before - after) / before
        else:
            pct = 0.0
        return (
            f"{n_prim} primitives, "
            f"{before:.2f}s -> {after:.2f}s "
            f"({pct:+.1f}% change)"
        )
