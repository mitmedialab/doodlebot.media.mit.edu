"""Post-pass over a finished DrawingCommand list.

The vectorizer fits each segment as an independent line or arc, so a
single underlying circle in the input (a wheel, an eye, a head outline)
that gets fragmented at junctions ends up as several arcs with slightly
different (center, radius) parameters. Visually the result looks broken
even though each piece is a good fit on its own.

This module does a pure post-pass on the emitted commands:

  1. Reverse-simulate the command sequence to recover geometric
     primitives (each with start, end, and arc params if applicable).
  2. Cluster pen-down arcs whose (center, radius) are similar.
  3. For every cluster of size >= 2, compute a consensus circle
     (sweep-weighted average) and snap every arc in the cluster onto
     that circle: its center and radius are replaced, and its endpoints
     are projected radially onto the new circle.
  4. Snapping moves endpoints, which creates disjoints with neighbouring
     primitives. Walk the primitive list and stitch every gap so that
     prim[i+1].start == prim[i].end exactly. For neighbouring arcs that
     are not in any cluster, refit center and radius so the moved
     endpoint stays on a circle (preserving the original bend
     direction).
  5. Re-emit a fresh command sequence from the modified primitives.

The output is self-consistent: every arc primitive satisfies
|start - center| = |end - center| = radius, and adjacent primitives
share their connecting endpoint exactly. That is what guarantees the
robot doesn't drift -- the emitter computes spin/distance/sweep from
the primitive's geometry, and any inconsistency would manifest as
accumulating offset along the chain.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from commands import DrawingCommand, SpinCommand, LineCommand, ArcCommand

# =============================================================================
# Internal geometric representation
# =============================================================================


@dataclass
class _Primitive:
    """One geometric primitive between two robot states.

    `pen_down` is False only for inter-stroke pen-up traversals (which
    have free endpoints and therefore aren't constrained by the stitcher
    on their END side -- they just go wherever the next primitive starts).
    Arc fields are populated only when kind == "arc".
    """

    kind: str  # "line" | "arc"
    start: NDArray[np.float64]
    end: NDArray[np.float64]
    pen_down: bool = True
    # arc-only:
    center: Optional[NDArray[np.float64]] = None
    radius: float = 0.0
    ccw: bool = True


def _wrap_pi(a: float) -> float:
    return ((a + math.pi) % (2 * math.pi)) - math.pi


# =============================================================================
# Stage 1: simulate commands -> primitives
# =============================================================================


def _commands_to_primitives(
    commands: Sequence[DrawingCommand],
    start_pos: NDArray[np.float64],
    start_heading: float,
) -> List[_Primitive]:
    """Walk the command list while tracking (pos, heading); emit one
    primitive per line/arc command (spins are absorbed into the heading).
    """
    pos = np.asarray(start_pos, dtype=float).copy()
    heading = float(start_heading)
    prims: List[_Primitive] = []
    for c in commands:
        kind = c["kind"]
        if kind == "spin":
            spin = cast(SpinCommand, c)
            heading = _wrap_pi(heading + math.radians(spin["degrees"]))
        elif kind == "line":
            line = cast(LineCommand, c)
            dist = float(line["distance"])
            new_pos = pos + dist * np.array([math.cos(heading), math.sin(heading)])
            prims.append(
                _Primitive(
                    kind="line",
                    start=pos.copy(),
                    end=new_pos.copy(),
                    pen_down=bool(line.get("penDown", True)),
                )
            )
            pos = new_pos
        elif kind == "arc":
            arc = cast(ArcCommand, c)
            r = float(arc["radius"])
            sweep = math.radians(float(arc["degrees"]))
            ccw = sweep > 0
            # Center is perpendicular to heading at distance r; left for
            # CCW, right for CW.
            if ccw:
                perp = np.array([-math.sin(heading), math.cos(heading)])
            else:
                perp = np.array([math.sin(heading), -math.cos(heading)])
            center = pos + r * perp
            # Rotate the radial vector around the center by `sweep`.
            radial = pos - center
            cs, sn = math.cos(sweep), math.sin(sweep)
            new_radial = np.array(
                [
                    radial[0] * cs - radial[1] * sn,
                    radial[0] * sn + radial[1] * cs,
                ]
            )
            new_pos = center + new_radial
            prims.append(
                _Primitive(
                    kind="arc",
                    start=pos.copy(),
                    end=new_pos.copy(),
                    pen_down=True,
                    center=center,
                    radius=r,
                    ccw=ccw,
                )
            )
            pos = new_pos
            heading = _wrap_pi(heading + sweep)
        else:
            raise ValueError(f"unknown command kind: {kind!r}")
    return prims


# =============================================================================
# Stage 2: cluster arcs by shared underlying circle
# =============================================================================


def _cluster_arcs(
    prims: List[_Primitive],
    center_tol_rel: float,
    radius_tol_rel: float,
    center_tol_abs: float,
    radius_tol_abs: float,
) -> List[List[int]]:
    """Union-find arcs with similar (center, radius). Tolerances are
    'either / or': two arcs are linked if their centers AND radii are
    close, where close = max(rel * R, abs).
    Returns clusters as lists of primitive indices (only clusters of
    size >= 2 are returned)."""
    arc_idxs = [i for i, p in enumerate(prims) if p.kind == "arc" and p.pen_down]
    n = len(arc_idxs)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for ii in range(n):
        pi = prims[arc_idxs[ii]]
        if pi.center is None:
            continue
        for jj in range(ii + 1, n):
            pj = prims[arc_idxs[jj]]
            if pj.center is None:
                continue
            r_max = max(pi.radius, pj.radius)
            ctol = max(center_tol_rel * r_max, center_tol_abs)
            rtol = max(radius_tol_rel * r_max, radius_tol_abs)
            cdist = float(np.linalg.norm(pi.center - pj.center))
            rdiff = abs(pi.radius - pj.radius)
            if cdist < ctol and rdiff < rtol:
                union(ii, jj)

    bucket: Dict[int, List[int]] = {}
    for ii in range(n):
        bucket.setdefault(find(ii), []).append(arc_idxs[ii])
    return [g for g in bucket.values() if len(g) >= 2]


def _arc_signed_sweep(p: _Primitive) -> float:
    """Sweep angle in radians (positive for CCW, negative for CW), as
    you would compute from the primitive's start, end and center."""
    if p.center is None:
        return 0.0
    a_s = math.atan2(p.start[1] - p.center[1], p.start[0] - p.center[0])
    a_e = math.atan2(p.end[1] - p.center[1], p.end[0] - p.center[0])
    sweep = a_e - a_s
    if p.ccw and sweep <= 0:
        sweep += 2 * math.pi
    elif not p.ccw and sweep >= 0:
        sweep -= 2 * math.pi
    return sweep


def _consensus_circle(
    prims: List[_Primitive], cluster: List[int]
) -> Tuple[NDArray, float]:
    """Sweep-weighted average of (center, radius). Longer arcs vote more
    because they carry more information about the underlying circle."""
    weights, centers, radii = [], [], []
    for idx in cluster:
        p = prims[idx]
        # Minimum weight floors the contribution of vanishingly short arcs
        # without dropping them entirely.
        w = max(abs(_arc_signed_sweep(p)), 0.05)
        weights.append(w)
        centers.append(p.center)
        radii.append(p.radius)
    weights = np.asarray(weights)
    weights = weights / weights.sum()
    cc = np.zeros(2)
    for w, c in zip(weights, centers):
        cc = cc + w * c
    cr = float(sum(w * r for w, r in zip(weights, radii)))
    return cc, cr


# =============================================================================
# Stage 3: snap each clustered arc to its consensus circle
# =============================================================================


def _project_onto_circle(p: NDArray, center: NDArray, radius: float) -> NDArray:
    v = p - center
    d = float(np.linalg.norm(v))
    if d < 1e-9:
        return center + np.array([radius, 0.0])
    return center + (radius / d) * v


def _snap_arc_to_consensus(p: _Primitive, cc: NDArray, cr: float) -> None:
    """Replace the arc's center and radius with the cluster consensus,
    and project both endpoints radially onto the new circle. The arc's
    ccw direction is preserved."""
    p.center = cc.copy()
    p.radius = cr
    p.start = _project_onto_circle(p.start, cc, cr)
    p.end = _project_onto_circle(p.end, cc, cr)


# =============================================================================
# Stage 2b/3b: cluster collinear lines and snap them to a shared consensus
# =============================================================================


def _line_geometry(p: _Primitive) -> Tuple[NDArray, float]:
    """Return the line's unit direction and length. Caller is expected to
    have filtered out zero-length lines already."""
    v = p.end - p.start
    L = float(np.linalg.norm(v))
    return v / L, L


def _cluster_lines(
    prims: List[_Primitive],
    angle_tol_rad: float,
    offset_tol_abs: float,
    min_length: float,
) -> List[List[int]]:
    """Group pen-down lines that are collinear: same direction within
    `angle_tol_rad` (sign-ambiguous, since a line has no preferred
    orientation), and same perpendicular offset within `offset_tol_abs`
    pixels. Lines shorter than `min_length` are excluded because their
    direction estimate is too noisy."""
    line_idxs: List[int] = []
    dirs: List[NDArray] = []
    for i, p in enumerate(prims):
        if p.kind != "line" or not p.pen_down:
            continue
        L = float(np.linalg.norm(p.end - p.start))
        if L < min_length:
            continue
        line_idxs.append(i)
        dirs.append((p.end - p.start) / L)

    n = len(line_idxs)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    cos_tol = math.cos(angle_tol_rad)
    for ii in range(n):
        pi = prims[line_idxs[ii]]
        di = dirs[ii]
        for jj in range(ii + 1, n):
            pj = prims[line_idxs[jj]]
            dj = dirs[jj]
            # Direction parallel? Use |dot| because a line has no orientation.
            if abs(float(np.dot(di, dj))) < cos_tol:
                continue
            # Perpendicular offset: distance from one line's start to the
            # other's infinite line.
            perp_j = np.array([-dj[1], dj[0]])
            offset = abs(float(np.dot(perp_j, pi.start - pj.start)))
            if offset < offset_tol_abs:
                union(ii, jj)

    bucket: Dict[int, List[int]] = {}
    for ii in range(n):
        bucket.setdefault(find(ii), []).append(line_idxs[ii])
    return [g for g in bucket.values() if len(g) >= 2]


def _consensus_line(
    prims: List[_Primitive],
    cluster: List[int],
) -> Tuple[NDArray, NDArray]:
    """Total-least-squares line through all endpoints of the clustered
    lines. Returns (centroid, unit_direction); the infinite consensus
    line is the set of points centroid + t * direction for any t in R."""
    pts = []
    for idx in cluster:
        p = prims[idx]
        pts.append(p.start)
        pts.append(p.end)
    arr = np.asarray(pts)
    centroid = arr.mean(axis=0)
    centered = arr - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]
    return centroid, direction


def _project_onto_line(p: NDArray, centroid: NDArray, direction: NDArray) -> NDArray:
    return centroid + float(np.dot(p - centroid, direction)) * direction


def _snap_line_to_consensus(
    p: _Primitive,
    centroid: NDArray,
    direction: NDArray,
) -> None:
    """Project both endpoints of a line primitive onto the consensus
    infinite line (perpendicular projection)."""
    p.start = _project_onto_line(p.start, centroid, direction)
    p.end = _project_onto_line(p.end, centroid, direction)


def _max_line_endpoint_snap_distance(
    prims: List[_Primitive],
    cluster: List[int],
    centroid: NDArray,
    direction: NDArray,
) -> float:
    """Max perpendicular distance any endpoint must travel to reach the
    consensus line. Large values indicate the cluster is bogus."""
    perp = np.array([-direction[1], direction[0]])
    max_d = 0.0
    for idx in cluster:
        p = prims[idx]
        for endpoint in (p.start, p.end):
            d = abs(float(np.dot(perp, endpoint - centroid)))
            if d > max_d:
                max_d = d
    return max_d


# =============================================================================
# Stage 4: stitching (refit non-clustered arcs whose endpoints moved)
# =============================================================================


def _refit_arc_through_endpoints(p: _Primitive) -> None:
    """The arc's start and end no longer satisfy
    |x - center| = radius. Find the unique arc that PASSES THROUGH the
    current start and end with center on the perpendicular bisector,
    closest to the original center. Preserves the original bending side
    (and therefore the ccw flag)."""
    if p.center is None:
        return
    chord = p.end - p.start
    chord_len = float(np.linalg.norm(chord))
    if chord_len < 1e-9:
        return  # degenerate
    midpoint = 0.5 * (p.start + p.end)
    # perpendicular to chord (unit)
    perp = np.array([-chord[1], chord[0]]) / chord_len
    # Match the original bending side
    if float(np.dot(p.center - midpoint, perp)) < 0:
        perp = -perp
    # Project the original center onto the perpendicular line
    t = float(np.dot(p.center - midpoint, perp))
    new_center = midpoint + t * perp
    new_radius = math.sqrt(t * t + (chord_len / 2.0) ** 2)
    p.center = new_center
    p.radius = new_radius
    # ccw is preserved


def _stitch(
    prims: List[_Primitive],
    consensus_arcs: Dict[int, Tuple[NDArray, float]],
    consensus_lines: Dict[int, Tuple[NDArray, NDArray]],
) -> None:
    """Walk the primitive list in order. For each adjacent pen-down pair
    whose connecting endpoints disagree, reconcile while preserving each
    side's consensus geometry where possible.

    Resolution rules:
      - Pen-up "line" primitives are unconstrained on both ends; they
        just snap to whatever the neighbours dictate.
      - If exactly one of (prev, cur) is on a consensus geometry (arc or
        line), the other adapts: the moving endpoint is set to match the
        clustered side; non-clustered arc neighbours get refit through
        their new endpoints, non-clustered line neighbours just store
        the new endpoint.
      - If neither side is clustered (only happens if a non-clustered arc
        was already disturbed indirectly), take the midpoint and refit.
      - If both sides are clustered (e.g. a circle meeting a line at
        a junction), take the midpoint, project each side onto its own
        consensus, average once more. Sub-pixel residual is accepted.
    """
    EPS = 1e-3

    def project_to_own(idx: int, attr: str) -> None:
        """Project the (start|end) of prims[idx] onto its consensus
        geometry, if any. Otherwise no-op."""
        p = prims[idx]
        pt = getattr(p, attr)
        if idx in consensus_arcs:
            cc, cr = consensus_arcs[idx]
            v = pt - cc
            d = float(np.linalg.norm(v))
            if d > 1e-9:
                setattr(p, attr, cc + (cr / d) * v)
        elif idx in consensus_lines:
            centroid, direction = consensus_lines[idx]
            setattr(p, attr, _project_onto_line(pt, centroid, direction))

    def is_clustered(idx: int) -> bool:
        return idx in consensus_arcs or idx in consensus_lines

    for i in range(1, len(prims)):
        prev = prims[i - 1]
        cur = prims[i]

        if cur.kind == "line" and not cur.pen_down:
            cur.start = prev.end.copy()
            continue

        gap = float(np.linalg.norm(cur.start - prev.end))
        if gap < EPS:
            cur.start = prev.end.copy()
            continue

        prev_clustered = is_clustered(i - 1)
        cur_clustered = is_clustered(i)

        if prev_clustered and not cur_clustered:
            cur.start = prev.end.copy()
            if cur.kind == "arc":
                _refit_arc_through_endpoints(cur)
        elif cur_clustered and not prev_clustered:
            prev.end = cur.start.copy()
            if prev.kind == "arc":
                _refit_arc_through_endpoints(prev)
        elif prev_clustered and cur_clustered:
            mid = 0.5 * (prev.end + cur.start)
            prev.end = mid.copy()
            cur.start = mid.copy()
            project_to_own(i - 1, "end")
            project_to_own(i, "start")
            mid = 0.5 * (prev.end + cur.start)
            prev.end = mid.copy()
            cur.start = mid.copy()
        else:
            mid = 0.5 * (prev.end + cur.start)
            prev.end = mid.copy()
            cur.start = mid.copy()
            if prev.kind == "arc":
                _refit_arc_through_endpoints(prev)
            if cur.kind == "arc":
                _refit_arc_through_endpoints(cur)


# =============================================================================
# Stage 5: primitives -> commands (re-emit)
# =============================================================================


def _primitives_to_commands(
    prims: List[_Primitive],
    start_pos: NDArray,
    start_heading: float,
) -> List[DrawingCommand]:
    """Walk the (now self-consistent) primitive list and emit
    spin / line / arc commands. Mirrors the existing `strokes_to_commands`
    behaviour but drives off the primitive list rather than strokes."""
    pos = np.asarray(start_pos, dtype=float).copy()
    heading = float(start_heading)
    out: List[DrawingCommand] = []
    EPS_HEADING = 1e-4
    EPS_DIST = 1e-6

    for p in prims:
        if p.kind == "line":
            delta = p.end - pos
            dist = float(np.linalg.norm(delta))
            if dist < EPS_DIST:
                continue
            target_h = math.atan2(delta[1], delta[0])
            spin = _wrap_pi(target_h - heading)
            if abs(spin) > EPS_HEADING:
                out.append({"kind": "spin", "degrees": math.degrees(spin)})
                heading = target_h
            out.append({"kind": "line", "distance": dist, "penDown": p.pen_down})
            pos = p.end.copy()
            continue

        if p.center is None:
            continue
        # arc
        radial = p.start - p.center
        if p.ccw:
            tangent = np.array([-radial[1], radial[0]])
        else:
            tangent = np.array([radial[1], -radial[0]])
        target_h = math.atan2(tangent[1], tangent[0])
        spin = _wrap_pi(target_h - heading)
        if abs(spin) > EPS_HEADING:
            out.append({"kind": "spin", "degrees": math.degrees(spin)})
            heading = target_h
        sweep = _arc_signed_sweep(p)
        out.append(
            {"kind": "arc", "radius": float(p.radius), "degrees": math.degrees(sweep)}
        )
        pos = p.end.copy()
        heading = _wrap_pi(heading + sweep)

    return out


# =============================================================================
# Public API
# =============================================================================


def max_drift(
    commands: Sequence[DrawingCommand],
    start_pos: Sequence[float] = (0.0, 0.0),
    start_heading: float = 0.0,
) -> float:
    """Maximum within-stroke disjoint, in pixels, in a command list.

    A correctly-emitted DrawingCommand list should have zero drift: every
    pen-down primitive's start should equal the previous primitive's end.
    Useful as an assertion after any post-processing.
    """
    sp = np.asarray(start_pos, dtype=float)
    prims = _commands_to_primitives(commands, sp, start_heading)
    max_d = 0.0
    for i in range(1, len(prims)):
        prev, cur = prims[i - 1], prims[i]
        if cur.kind == "line" and not cur.pen_down:
            continue
        max_d = max(max_d, float(np.linalg.norm(cur.start - prev.end)))
    return max_d


@dataclass
class ConsolidationReport:
    """Diagnostics returned alongside the consolidated commands."""

    n_arc_clusters: int
    arc_cluster_sizes: List[int]
    n_arc_clusters_rejected: int = 0
    n_line_clusters: int = 0
    line_cluster_sizes: List[int] = field(default_factory=list)
    n_line_clusters_rejected: int = 0
    n_disjoints_fixed: int = 0
    max_disjoint_gap: float = 0.0


def _max_endpoint_snap_distance(
    prims: List[_Primitive],
    cluster: List[int],
    cc: NDArray,
    cr: float,
) -> float:
    """Maximum radial distance any endpoint would have to travel to reach
    the consensus circle. Used as a safety check: if even the best
    consensus circle requires moving an endpoint by a lot, the arcs in
    the cluster aren't really part of one circle."""
    max_d = 0.0
    for idx in cluster:
        p = prims[idx]
        for endpoint in (p.start, p.end):
            d = float(np.linalg.norm(endpoint - cc))
            max_d = max(max_d, abs(d - cr))
    return max_d

    # start_pos: Sequence[float] = (0.0, 0.0),
    # start_heading: float = 0.0,
    # # arc-cluster knobs
    # center_tol_rel: float = 0.25,
    # radius_tol_rel: float = 0.25,
    # center_tol_abs: float = 3.0,
    # radius_tol_abs: float = 3.0,
    # max_endpoint_snap_rel: float = 0.15,
    # max_endpoint_snap_abs: float = 6.0,
    # # line-cluster knobs
    # line_angle_tol_deg: float = 6.0,
    # line_offset_tol_abs: float = 5.0,
    # min_line_length: float = 5.0,
    # max_line_endpoint_snap_abs: float = 5.0,
    # # behaviour switches
    # merge_arcs: bool = True,
    # merge_lines: bool = True,
    # return_report: bool = False,


class ConsolidateConfig(TypedDict):
    center_tol_rel: float
    radius_tol_rel: float
    center_tol_abs: float
    radius_tol_abs: float
    max_endpoint_snap_rel: float
    max_endpoint_snap_abs: float
    line_angle_tol_deg: float
    line_offset_tol_abs: float
    min_line_length: float
    max_line_endpoint_snap_abs: float
    merge_arcs: bool
    merge_lines: bool
    return_report: bool


def consolidate_commands(
    commands: Sequence[DrawingCommand],
    start_pos: NDArray[np.float64],
    start_heading: float,
    config: ConsolidateConfig,
):
    """Merge fragmented arcs that probably belong to one circle, merge
    fragmented lines that probably belong to one infinite line, and
    stitch all disjoints introduced by either operation. The output is
    self-consistent (zero drift on the robot).

    Args:
        commands:                  the DrawingCommand list to post-process.
        start_pos, start_heading:  the robot's initial state, used to
            simulate the input commands back into geometric primitives.

      Arc-cluster tolerances (used when merge_arcs=True):
        center_tol_rel/_abs:       cluster two arcs if their centers are
            within max(rel * R_max, abs) pixels.
        radius_tol_rel/_abs:       same for radius difference.
        max_endpoint_snap_rel/_abs: a candidate cluster is REJECTED if
            its consensus circle would require any endpoint to move
            more than max(rel * R, abs) pixels. Catches the
            "two unrelated arcs that just happen to look similar" case.

      Line-cluster tolerances (used when merge_lines=True):
        line_angle_tol_deg:    two lines cluster only if their direction
            unit vectors agree (up to sign) within this many degrees.
        line_offset_tol_abs:   ...AND their perpendicular offset agrees
            within this many pixels.
        min_line_length:       lines shorter than this are excluded from
            clustering (their direction estimate is too noisy).
        max_line_endpoint_snap_abs: a candidate line cluster is REJECTED
            if any endpoint would have to move more than this many
            pixels to reach the consensus line.

      Switches:
        merge_arcs:    set False to disable the arc-cluster pass.
        merge_lines:   set False to disable the line-cluster pass.
        return_report: also return a ConsolidationReport.

    Returns:
        consolidated_commands  -- a fresh DrawingCommand list, or
        (consolidated_commands, ConsolidationReport) if return_report.
    """
    prims = _commands_to_primitives(commands, start_pos, start_heading)

    # ---------------- arc clustering ----------------
    accepted_arc_clusters: List[List[int]] = []
    rejected_arc_clusters: List[List[int]] = []
    consensus_arcs: Dict[int, Tuple[NDArray, float]] = {}
    if config["merge_arcs"]:
        candidate_arc_clusters = _cluster_arcs(
            prims,
            config["center_tol_rel"],
            config["radius_tol_rel"],
            config["center_tol_abs"],
            config["radius_tol_abs"],
        )
        for cluster in candidate_arc_clusters:
            cc, cr = _consensus_circle(prims, cluster)
            snap_d = _max_endpoint_snap_distance(prims, cluster, cc, cr)
            snap_allowed = max(
                config["max_endpoint_snap_abs"], config["max_endpoint_snap_rel"] * cr
            )
            if snap_d <= snap_allowed:
                accepted_arc_clusters.append(cluster)
                for idx in cluster:
                    _snap_arc_to_consensus(prims[idx], cc, cr)
                    consensus_arcs[idx] = (cc, cr)
            else:
                rejected_arc_clusters.append(cluster)

    # ---------------- line clustering ----------------
    accepted_line_clusters: List[List[int]] = []
    rejected_line_clusters: List[List[int]] = []
    consensus_lines: Dict[int, Tuple[NDArray, NDArray]] = {}
    if config["merge_lines"]:
        candidate_line_clusters = _cluster_lines(
            prims,
            angle_tol_rad=math.radians(config["line_angle_tol_deg"]),
            offset_tol_abs=config["line_offset_tol_abs"],
            min_length=config["min_line_length"],
        )
        for cluster in candidate_line_clusters:
            centroid, direction = _consensus_line(prims, cluster)
            snap_d = _max_line_endpoint_snap_distance(
                prims, cluster, centroid, direction
            )
            if snap_d <= config["max_line_endpoint_snap_abs"]:
                accepted_line_clusters.append(cluster)
                for idx in cluster:
                    _snap_line_to_consensus(prims[idx], centroid, direction)
                    consensus_lines[idx] = (centroid, direction)
            else:
                rejected_line_clusters.append(cluster)

    # ---------------- pre-stitch diagnostics ----------------
    max_gap = 0.0
    n_disjoints = 0
    for i in range(1, len(prims)):
        prev, cur = prims[i - 1], prims[i]
        if cur.kind == "line" and not cur.pen_down:
            continue
        g = float(np.linalg.norm(cur.start - prev.end))
        if g > 1e-3:
            n_disjoints += 1
            if g > max_gap:
                max_gap = g

    _stitch(prims, consensus_arcs, consensus_lines)
    out = _primitives_to_commands(prims, start_pos, start_heading)

    if config["return_report"]:
        report = ConsolidationReport(
            n_arc_clusters=len(accepted_arc_clusters),
            arc_cluster_sizes=[len(c) for c in accepted_arc_clusters],
            n_arc_clusters_rejected=len(rejected_arc_clusters),
            n_line_clusters=len(accepted_line_clusters),
            line_cluster_sizes=[len(c) for c in accepted_line_clusters],
            n_line_clusters_rejected=len(rejected_line_clusters),
            n_disjoints_fixed=n_disjoints,
            max_disjoint_gap=max_gap,
        )
        return out, report
    return out, None
