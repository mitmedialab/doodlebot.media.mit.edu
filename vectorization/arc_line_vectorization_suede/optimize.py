"""Time-optimal re-ordering of an existing drawing-command sequence.

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
    """
    n = len(tour.order)
    if n < 3:
        return tour
    best = tour.copy()
    best_cost = _tour_cost_cached(best, cache)
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                candidate_order = best.order.copy()
                candidate_reverse = best.reverse.copy()
                candidate_order[i : j + 1] = list(reversed(candidate_order[i : j + 1]))
                candidate_reverse[i : j + 1] = [
                    not r for r in reversed(candidate_reverse[i : j + 1])
                ]
                candidate = _Tour(candidate_order, candidate_reverse)
                c = _tour_cost_cached(candidate, cache)
                if c + 1e-9 < best_cost:
                    best = candidate
                    best_cost = c
                    improved = True
        if not improved:
            break
    return best


def _or_opt(
    tour: _Tour,
    cache: _CostCache,
    max_segment_len: int = 3,
    max_passes: int = 4,
) -> _Tour:
    """Or-opt: relocate a short consecutive run, optionally flipped.
    Catches moves 2-opt can't represent.
    """
    n = len(tour.order)
    if n < 4:
        return tour
    best = tour.copy()
    best_cost = _tour_cost_cached(best, cache)
    for _ in range(max_passes):
        improved = False
        for seg_len in range(1, max_segment_len + 1):
            for i in range(n - seg_len + 1):
                segment_order = best.order[i : i + seg_len]
                segment_reverse = best.reverse[i : i + seg_len]
                remaining_order = best.order[:i] + best.order[i + seg_len :]
                remaining_reverse = best.reverse[:i] + best.reverse[i + seg_len :]
                for j in range(len(remaining_order) + 1):
                    for flip in (False, True):
                        if flip:
                            new_seg_order = list(reversed(segment_order))
                            new_seg_reverse = [not r for r in reversed(segment_reverse)]
                        else:
                            new_seg_order = segment_order
                            new_seg_reverse = segment_reverse
                        candidate = _Tour(
                            remaining_order[:j] + new_seg_order + remaining_order[j:],
                            remaining_reverse[:j]
                            + new_seg_reverse
                            + remaining_reverse[j:],
                        )
                        if (
                            candidate.order == best.order
                            and candidate.reverse == best.reverse
                        ):
                            continue
                        c = _tour_cost_cached(candidate, cache)
                        if c + 1e-9 < best_cost:
                            best = candidate
                            best_cost = c
                            improved = True
        if not improved:
            break
    return best


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
            ``LowGeometryVectorize(...).consolidated`` or
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
            return

        cache = _build_cost_cache(
            primitives,
            self.start_pos,
            self.start_heading,
            self.pixels_per_inch,
            self.pen_up_join_tol,
        )
        initial = _nearest_neighbor(cache, len(primitives))
        after_2opt = _two_opt(initial, cache, max_passes=self.two_opt_passes)
        after_or_opt = _or_opt(
            after_2opt,
            cache,
            max_segment_len=self.or_opt_max_segment_len,
            max_passes=self.or_opt_passes,
        )

        self.commands = _emit_tour(
            after_or_opt,
            primitives,
            self.start_pos,
            self.start_heading,
            self.pen_up_join_tol,
        )
        self.tour = list(zip(after_or_opt.order, after_or_opt.reverse))
        self.estimated_time_after = estimate_total_time(
            self.commands, self.pixels_per_inch
        )

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
