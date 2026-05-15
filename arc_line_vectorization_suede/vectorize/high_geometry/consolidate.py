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
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
from numpy.typing import NDArray

from .junction_graph import JunctionGraph
from ...commands import DrawingCommand

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
            heading = _wrap_pi(heading + math.radians(c["degrees"]))
        elif kind == "line":
            dist = float(c["distance"])
            new_pos = pos + dist * np.array([math.cos(heading), math.sin(heading)])
            prims.append(
                _Primitive(
                    kind="line",
                    start=pos.copy(),
                    end=new_pos.copy(),
                    pen_down=bool(c.get("penDown", True)),
                )
            )
            pos = new_pos
        elif kind == "arc":
            r = float(c["radius"])
            sweep = math.radians(float(c["degrees"]))
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
        for jj in range(ii + 1, n):
            pj = prims[arc_idxs[jj]]
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


def _move_endpoint(
    prims: List[_Primitive],
    idx: int,
    attr: str,
    new_pos: NDArray,
    jg: Optional[JunctionGraph],
    skip_refit: Optional[set] = None,
) -> List[Tuple[int, str]]:
    """Move a single endpoint, optionally propagating to its junction-mates.

    If ``jg`` is None, only ``prims[idx].<attr>`` is updated. If ``jg`` is
    provided, every endpoint sharing the same junction is moved to
    ``new_pos``, and each propagated-to arc whose index is NOT in
    ``skip_refit`` is refit through its (now-changed) endpoints. The
    ``skip_refit`` set typically contains all arc cluster members --
    those will have their (center, radius) controlled by their own
    cluster's consensus and shouldn't be refit by accident.

    Returns the propagation list (excluding the requested endpoint).
    """
    if jg is None:
        p = prims[idx]
        np_pos = np.asarray(new_pos, dtype=float).copy()
        if attr == "start":
            p.start = np_pos
        else:
            p.end = np_pos
        return []
    moved_others = jg.move(prims, idx, attr, new_pos)
    if skip_refit is None:
        skip_refit = set()
    for oi, _ in moved_others:
        if prims[oi].kind == "arc" and oi not in skip_refit:
            _refit_arc_through_endpoints(prims[oi])
    return moved_others


def _snap_arc_to_consensus(
    p: _Primitive,
    idx: int,
    cc: NDArray,
    cr: float,
    prims: List[_Primitive],
    jg: Optional[JunctionGraph] = None,
    skip_refit: Optional[set] = None,
) -> None:
    """Replace the arc's center and radius with the cluster consensus,
    and project both endpoints radially onto the new circle. The arc's
    ccw direction is preserved. If ``jg`` is provided, the endpoint
    moves are propagated to all junction-mates (so that other strokes
    that touched this arc at its endpoints follow along)."""
    p.center = cc.copy()
    p.radius = cr
    new_start = _project_onto_circle(p.start, cc, cr)
    new_end = _project_onto_circle(p.end, cc, cr)
    _move_endpoint(prims, idx, "start", new_start, jg, skip_refit)
    _move_endpoint(prims, idx, "end", new_end, jg, skip_refit)


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
    idx: int,
    centroid: NDArray,
    direction: NDArray,
    prims: List[_Primitive],
    jg: Optional[JunctionGraph] = None,
    skip_refit: Optional[set] = None,
) -> None:
    """Project both endpoints of a line primitive onto the consensus
    infinite line (perpendicular projection). If ``jg`` is provided,
    the moves propagate to all junction-mates."""
    new_start = _project_onto_line(p.start, centroid, direction)
    new_end = _project_onto_line(p.end, centroid, direction)
    _move_endpoint(prims, idx, "start", new_start, jg, skip_refit)
    _move_endpoint(prims, idx, "end", new_end, jg, skip_refit)


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
    jg: Optional[JunctionGraph] = None,
) -> None:
    """Walk the primitive list in order. For each adjacent pen-down pair
    whose connecting endpoints disagree, reconcile while preserving each
    side's consensus geometry where possible. Endpoint mutations go
    through the junction graph if provided, so any fix here also
    propagates to other strokes that touched the same junction.

    Resolution rules:
      - Pen-up "line" primitives are unconstrained on both ends; they
        just snap to whatever the neighbours dictate. The symmetric
        case (prev is pen-up) is also handled, so a stroke whose
        terminal endpoint moved earlier (e.g. by a proximity snap)
        doesn't get pulled back by an unconstrained pen-up.
      - If exactly one of (prev, cur) is on a consensus geometry, the
        other adapts: the moving endpoint matches the clustered side;
        non-clustered arc neighbours get refit through new endpoints.
      - If neither side is clustered, take the midpoint and refit.
      - If both sides are clustered (rare, e.g. a circle meeting a
        line at a junction), take the midpoint, project each onto its
        own consensus, average once more. Sub-pixel residual accepted.
    """
    EPS = 1e-3
    cluster_member_set = set(consensus_arcs) | set(consensus_lines)

    def set_endpoint(idx: int, attr: str, new_pos: NDArray) -> None:
        _move_endpoint(prims, idx, attr, new_pos, jg, cluster_member_set)

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
                set_endpoint(idx, attr, cc + (cr / d) * v)
        elif idx in consensus_lines:
            centroid, direction = consensus_lines[idx]
            set_endpoint(idx, attr, _project_onto_line(pt, centroid, direction))

    def is_clustered(idx: int) -> bool:
        return idx in consensus_arcs or idx in consensus_lines

    def is_pen_up(p: _Primitive) -> bool:
        return p.kind == "line" and not p.pen_down

    for i in range(1, len(prims)):
        prev = prims[i - 1]
        cur = prims[i]

        # Pen-up traversals adapt to whichever neighbour is constrained.
        if is_pen_up(cur):
            set_endpoint(i, "start", prev.end)
            continue
        if is_pen_up(prev):
            set_endpoint(i - 1, "end", cur.start)
            continue

        gap = float(np.linalg.norm(cur.start - prev.end))
        if gap < EPS:
            set_endpoint(i, "start", prev.end)
            continue

        prev_clustered = is_clustered(i - 1)
        cur_clustered = is_clustered(i)

        if prev_clustered and not cur_clustered:
            set_endpoint(i, "start", prev.end)
            if cur.kind == "arc":
                _refit_arc_through_endpoints(cur)
        elif cur_clustered and not prev_clustered:
            set_endpoint(i - 1, "end", cur.start)
            if prev.kind == "arc":
                _refit_arc_through_endpoints(prev)
        elif prev_clustered and cur_clustered:
            mid = 0.5 * (prev.end + cur.start)
            set_endpoint(i - 1, "end", mid)
            set_endpoint(i, "start", mid)
            project_to_own(i - 1, "end")
            project_to_own(i, "start")
            mid = 0.5 * (prev.end + cur.start)
            set_endpoint(i - 1, "end", mid)
            set_endpoint(i, "start", mid)
        else:
            mid = 0.5 * (prev.end + cur.start)
            set_endpoint(i - 1, "end", mid)
            set_endpoint(i, "start", mid)
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
    commands: Sequence[Dict[str, Any]],
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
    n_arcs_added_by_proximity: int = 0
    n_line_clusters: int = 0
    line_cluster_sizes: List[int] = field(default_factory=list)
    n_line_clusters_rejected: int = 0
    n_lines_added_by_proximity: int = 0
    n_disjoints_fixed: int = 0
    max_disjoint_gap: float = 0.0
    n_proximate_endpoints_snapped: int = 0
    # Junction graph statistics
    n_junctions: int = 0
    n_shared_junctions: int = 0
    shared_junction_sizes: List[int] = field(default_factory=list)


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


def _extend_arc_clusters_with_proximate(
    prims: List[_Primitive],
    accepted_clusters: List[List[int]],
    consensus_circles: List[Tuple[NDArray, float]],
    max_endpoint_snap_rel: float,
    max_endpoint_snap_abs: float,
    min_radius_ratio: float = 0.4,
) -> List[List[int]]:
    """Add arcs to clusters by ENDPOINT proximity, regardless of how
    well their own fitted (center, radius) match.

    Rationale: an arc that is geometrically a piece of a wheel can have
    a noisy individual fit (badly off radius/center) yet still have both
    of its endpoints sitting close to the underlying circle. The primary
    union-find pass clusters by similarity of fitted parameters and
    misses these. This pass looks at every pen-down arc not yet in any
    cluster and, if both its endpoints are within the snap-safety
    threshold of some cluster's consensus circle, adds it to that
    cluster. The "best" cluster (closest endpoints) wins ties.

    Critically, the consensus circle is NOT recomputed after extension:
    the original cluster members determined a good circle, and the
    newcomer's noisy fit wouldn't improve it. The newcomer just snaps
    to the existing consensus.

    Guard against tiny detail arcs (small eyes, nose tips, whisker
    crossings) whose endpoints happen to sit near a big circle by
    requiring the candidate's own radius to be within a factor of
    ``min_radius_ratio`` of the consensus radius. Without this guard,
    a tiny arc forced onto a big circle ends up with two snapped
    endpoints far apart on the big circle, and its preserved ``ccw``
    direction can sweep the long way around -- producing a 350°
    phantom arc on the consensus circle.
    """
    cluster_membership: Dict[int, int] = {}
    for ci, cluster in enumerate(accepted_clusters):
        for idx in cluster:
            cluster_membership[idx] = ci

    for i, p in enumerate(prims):
        if p.kind != "arc" or not p.pen_down or i in cluster_membership:
            continue
        best_ci = None
        best_max_d = float("inf")
        for ci, (cc, cr) in enumerate(consensus_circles):
            r_ratio = min(p.radius, cr) / max(p.radius, cr)
            if r_ratio < min_radius_ratio:
                continue
            d_start = abs(float(np.linalg.norm(p.start - cc)) - cr)
            d_end = abs(float(np.linalg.norm(p.end - cc)) - cr)
            max_d = max(d_start, d_end)
            snap_allowed = max(max_endpoint_snap_abs, max_endpoint_snap_rel * cr)
            if max_d > snap_allowed:
                continue
            if max_d < best_max_d:
                best_max_d = max_d
                best_ci = ci
        if best_ci is not None:
            accepted_clusters[best_ci].append(i)
            cluster_membership[i] = best_ci
    return accepted_clusters


def _snap_proximate_endpoints_to_consensus_circles(
    prims: List[_Primitive],
    consensus_circles: List[Tuple[NDArray, float]],
    cluster_members: set,
    max_snap_rel: float,
    max_snap_abs: float,
    jg: JunctionGraph,
    min_radius_ratio: float = 0.4,
) -> int:
    """For every pen-down primitive that is NOT a cluster member, look
    at each endpoint and check whether it lies within snap threshold of
    any consensus circle. If so, project it onto the circle and
    propagate through the junction graph so any junction-mates follow.

    The junction graph already handles cases where a non-cluster
    endpoint exactly coincided with a cluster-member endpoint -- those
    moved during cluster snapping. This pass picks up the OTHER cases:
    an endpoint that's geometrically near the consensus circle but
    didn't share a junction with any cluster arc. The bike's spokes
    are the canonical example: each spoke ends at the rim, but the
    rim's arc-arc junctions and the rim's spoke-touching points are at
    different angular positions, so they don't share junctions in the
    graph.

    Same tiny-arc guard as the extension pass: for arc primitives, the
    candidate's own radius must be within ``min_radius_ratio`` of the
    candidate consensus radius. Lines have no inherent curvature, so
    the gate is skipped for them.

    Only TERMINAL endpoints are eligible -- an endpoint is terminal
    when it sits at the start or end of the primitive list, or when
    the adjacent primitive is a pen-up traversal. Interior endpoints
    (mid-stroke) are skipped. Without this restriction, a short
    interior line segment with BOTH endpoints near a consensus circle
    gets snapped on both ends, turning it into a chord of the circle
    and producing a visible bump along the consolidated outline. The
    bike's spokes don't have this problem because their non-rim
    endpoint is at the hub, far from the consensus circle.

    Returns the count of endpoints that were snapped.
    """
    n = len(prims)
    n_snapped = 0

    def is_pen_up_line(p: _Primitive) -> bool:
        return p.kind == "line" and not p.pen_down

    def is_terminal(i: int, attr: str) -> bool:
        if attr == "start":
            return i == 0 or is_pen_up_line(prims[i - 1])
        return i == n - 1 or is_pen_up_line(prims[i + 1])

    def maybe_snap_target(pt: NDArray, p: _Primitive) -> Optional[NDArray]:
        best_ci = None
        best_d = float("inf")
        for ci, (cc, cr) in enumerate(consensus_circles):
            # Arc-only: skip consensus circles whose curvature is
            # incompatible with this primitive's own radius.
            if p.kind == "arc":
                r_ratio = min(p.radius, cr) / max(p.radius, cr)
                if r_ratio < min_radius_ratio:
                    continue
            d = abs(float(np.linalg.norm(pt - cc)) - cr)
            snap_allowed = max(max_snap_abs, max_snap_rel * cr)
            if d < snap_allowed and d < best_d:
                best_d = d
                best_ci = ci
        if best_ci is None:
            return None
        cc, cr = consensus_circles[best_ci]
        v = pt - cc
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            return None
        return cc + (cr / norm) * v

    for i, p in enumerate(prims):
        if not p.pen_down or i in cluster_members:
            continue
        endpoint_changed = False
        for attr in ("start", "end"):
            if not is_terminal(i, attr):
                continue
            new = maybe_snap_target(getattr(p, attr), p)
            if new is not None:
                _move_endpoint(prims, i, attr, new, jg, cluster_members)
                endpoint_changed = True
                n_snapped += 1
        # Refit our own arc if its endpoints moved off-circle
        if endpoint_changed and p.kind == "arc":
            d_start = float(np.linalg.norm(p.start - p.center))
            d_end = float(np.linalg.norm(p.end - p.center))
            if abs(d_start - p.radius) > 0.5 or abs(d_end - p.radius) > 0.5:
                _refit_arc_through_endpoints(p)
    return n_snapped


def _extend_line_clusters_with_proximate(
    prims: List[_Primitive],
    accepted_clusters: List[List[int]],
    consensus_lines: List[Tuple[NDArray, NDArray]],
    angle_tol_rad: float,
    max_line_endpoint_snap_abs: float,
    min_length: float,
) -> List[List[int]]:
    """Same idea as `_extend_arc_clusters_with_proximate` but for lines:
    add a line to a cluster if its direction is parallel to the
    cluster's consensus line (within `angle_tol_rad`) AND both endpoints
    sit within `max_line_endpoint_snap_abs` perpendicular pixels of that
    consensus line.
    """
    cluster_membership: Dict[int, int] = {}
    for ci, cluster in enumerate(accepted_clusters):
        for idx in cluster:
            cluster_membership[idx] = ci

    cos_tol = math.cos(angle_tol_rad)
    for i, p in enumerate(prims):
        if p.kind != "line" or not p.pen_down or i in cluster_membership:
            continue
        L = float(np.linalg.norm(p.end - p.start))
        if L < min_length:
            continue
        u = (p.end - p.start) / L
        best_ci = None
        best_max_d = float("inf")
        for ci, (centroid, direction) in enumerate(consensus_lines):
            if abs(float(np.dot(u, direction))) < cos_tol:
                continue
            perp = np.array([-direction[1], direction[0]])
            d_start = abs(float(np.dot(perp, p.start - centroid)))
            d_end = abs(float(np.dot(perp, p.end - centroid)))
            max_d = max(d_start, d_end)
            if max_d > max_line_endpoint_snap_abs:
                continue
            if max_d < best_max_d:
                best_max_d = max_d
                best_ci = ci
        if best_ci is not None:
            accepted_clusters[best_ci].append(i)
            cluster_membership[i] = best_ci
    return accepted_clusters


class ConsolidateConfig(TypedDict):
    center_tol_rel: float
    """
    cluster two arcs if their centers are within max(rel * R_max, abs) pixels.
    """

    radius_tol_rel: float
    """
    cluster two arcs if their radii differ by less than max(rel * R_max, abs).
    """

    center_tol_abs: float
    """
    absolute fallback for center clustering tolerance in pixels.
    """

    radius_tol_abs: float
    """
    absolute fallback for radius clustering tolerance in pixels.
    """

    max_endpoint_snap_rel: float
    """
    reject an arc cluster if any endpoint would need to move more than
    max(rel * R, abs) pixels to match the consensus circle.
    """

    max_endpoint_snap_abs: float
    """
    absolute fallback for endpoint snap threshold in pixels when validating
    arc clusters.
    """

    proximity_min_radius_ratio: float

    line_angle_tol_deg: float
    """
    cluster two lines only if their direction vectors (up to sign) agree
    within this many degrees.
    """

    line_offset_tol_abs: float
    """
    cluster two lines only if their perpendicular offsets differ by no more
    than this many pixels.
    """

    min_line_length: float
    """
    lines shorter than this are excluded from line clustering because their
    direction estimate is too noisy.
    """

    max_line_endpoint_snap_abs: float
    """
    reject a line cluster if any endpoint would need to move more than this
    many pixels to reach the consensus line.
    """

    junction_epsilon: float
    """
    when building the junction graph, consider two endpoints to be linked if
    they are within this many pixels of each other in the initial primitive set.
    """

    merge_arcs: bool
    """
    set False to disable the arc-cluster pass.
    """

    merge_lines: bool
    """
    set False to disable the line-cluster pass.
    """

    return_report: bool
    """
    if True, return (commands, ConsolidationReport) instead of only commands.
    """


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
    sp = np.asarray(start_pos, dtype=float)
    prims = _commands_to_primitives(commands, sp, start_heading)

    # Build the junction graph from the *initial* primitive positions.
    # All subsequent endpoint moves go through this graph so endpoints
    # that originally coincided in 2D travel together. This is what
    # keeps a spoke aligned with the wheel rim when the rim snaps onto
    # a consensus circle. Membership is fixed at construction; later
    # mutations don't change which endpoints are considered linked.
    jg = JunctionGraph(prims, epsilon=config["junction_epsilon"])

    # ---------------- arc clustering ----------------
    accepted_arc_clusters: List[List[int]] = []
    rejected_arc_clusters: List[List[int]] = []
    arc_consensus_circles: List[Tuple[NDArray, float]] = []
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
                arc_consensus_circles.append((cc, cr))
            else:
                rejected_arc_clusters.append(cluster)

        # Extension pass: include any arc whose endpoints are already
        # close to one of the accepted consensus circles, regardless of
        # how noisy its own fit was.
        n_arc_added_by_proximity = sum(len(c) for c in accepted_arc_clusters)
        _extend_arc_clusters_with_proximate(
            prims,
            accepted_arc_clusters,
            arc_consensus_circles,
            config["max_endpoint_snap_rel"],
            config["max_endpoint_snap_abs"],
            min_radius_ratio=config["proximity_min_radius_ratio"],
        )
        n_arc_added_by_proximity = (
            sum(len(c) for c in accepted_arc_clusters) - n_arc_added_by_proximity
        )
    else:
        n_arc_added_by_proximity = 0

    # ---------------- line clustering ----------------
    accepted_line_clusters: List[List[int]] = []
    rejected_line_clusters: List[List[int]] = []
    line_consensus_geoms: List[Tuple[NDArray, NDArray]] = []
    consensus_lines: Dict[int, Tuple[NDArray, NDArray]] = {}
    if config["merge_lines"]:
        line_angle_tol_rad = math.radians(config["line_angle_tol_deg"])
        candidate_line_clusters = _cluster_lines(
            prims,
            angle_tol_rad=line_angle_tol_rad,
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
                line_consensus_geoms.append((centroid, direction))
            else:
                rejected_line_clusters.append(cluster)

        # Extension pass for lines.
        n_line_added_by_proximity = sum(len(c) for c in accepted_line_clusters)
        _extend_line_clusters_with_proximate(
            prims,
            accepted_line_clusters,
            line_consensus_geoms,
            angle_tol_rad=line_angle_tol_rad,
            max_line_endpoint_snap_abs=config["max_line_endpoint_snap_abs"],
            min_length=config["min_line_length"],
        )
        n_line_added_by_proximity = (
            sum(len(c) for c in accepted_line_clusters) - n_line_added_by_proximity
        )
    else:
        n_line_added_by_proximity = 0

    # Pre-compute the set of all cluster members across both kinds.
    # Propagation must not refit cluster member arcs (that would
    # overwrite their freshly-set consensus center/radius).
    all_cluster_members: set = set()
    for c in accepted_arc_clusters:
        all_cluster_members.update(c)
    for c in accepted_line_clusters:
        all_cluster_members.update(c)

    # Now do the actual snapping, with junction-graph propagation.
    if config["merge_arcs"]:
        for cluster, (cc, cr) in zip(accepted_arc_clusters, arc_consensus_circles):
            for idx in cluster:
                _snap_arc_to_consensus(
                    prims[idx],
                    idx,
                    cc,
                    cr,
                    prims,
                    jg,
                    all_cluster_members,
                )
                consensus_arcs[idx] = (cc, cr)
    if config["merge_lines"]:
        for cluster, (centroid, direction) in zip(
            accepted_line_clusters,
            line_consensus_geoms,
        ):
            for idx in cluster:
                _snap_line_to_consensus(
                    prims[idx],
                    idx,
                    centroid,
                    direction,
                    prims,
                    jg,
                    all_cluster_members,
                )
                consensus_lines[idx] = (centroid, direction)

    # Proximate-endpoint pass: catches non-cluster endpoints (typically
    # belonging to spokes or other strokes meeting a consolidated curve)
    # that are near a consensus circle but didn't share a junction with
    # any cluster member. Without this, those endpoints stay at their
    # original positions while the rim moves, producing visible
    # spoke-overshoots-rim artifacts.
    n_endpoints_snapped = 0
    if config["merge_arcs"] and arc_consensus_circles:
        n_endpoints_snapped = _snap_proximate_endpoints_to_consensus_circles(
            prims,
            arc_consensus_circles,
            all_cluster_members,
            config["max_endpoint_snap_rel"],
            config["max_endpoint_snap_abs"],
            jg,
            min_radius_ratio=config["proximity_min_radius_ratio"],
        )

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

    _stitch(prims, consensus_arcs, consensus_lines, jg)

    # Final reprojection pass: cross-cluster propagation through the
    # junction graph could have pulled a clustered arc's endpoint
    # slightly off its consensus circle. Re-snap clustered arc/line
    # endpoints back onto their own consensus geometry. Don't propagate
    # this final fix (we don't want to set off another wave of moves);
    # just clamp each cluster member onto its consensus.
    for idx, (cc, cr) in consensus_arcs.items():
        p = prims[idx]
        p.start = _project_onto_circle(p.start, cc, cr)
        p.end = _project_onto_circle(p.end, cc, cr)
    for idx, (centroid, direction) in consensus_lines.items():
        p = prims[idx]
        p.start = _project_onto_line(p.start, centroid, direction)
        p.end = _project_onto_line(p.end, centroid, direction)

    out = _primitives_to_commands(prims, sp, start_heading)

    if config["return_report"]:
        report = ConsolidationReport(
            n_arc_clusters=len(accepted_arc_clusters),
            arc_cluster_sizes=[len(c) for c in accepted_arc_clusters],
            n_arc_clusters_rejected=len(rejected_arc_clusters),
            n_arcs_added_by_proximity=n_arc_added_by_proximity,
            n_line_clusters=len(accepted_line_clusters),
            line_cluster_sizes=[len(c) for c in accepted_line_clusters],
            n_line_clusters_rejected=len(rejected_line_clusters),
            n_lines_added_by_proximity=n_line_added_by_proximity,
            n_disjoints_fixed=n_disjoints,
            max_disjoint_gap=max_gap,
            n_proximate_endpoints_snapped=n_endpoints_snapped,
            n_junctions=jg.n_junctions(),
            n_shared_junctions=jg.n_shared_junctions(),
            shared_junction_sizes=jg.shared_junction_sizes(),
        )
        return out, report
    return out, None
