"""Build a stroke graph from repaired polylines."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np
from numpy.typing import NDArray


class Role(Enum):
    TERMINAL = "terminal"
    CROSSING = "crossing"
    CUSP = "cusp"


@dataclass
class Participation:
    polyline_index: int
    point_index: int
    role: Role
    terminal_end: Optional[int]
    arc_length_t: float
    tangent_in: NDArray[np.float64]
    tangent_out: Optional[NDArray[np.float64]]


@dataclass
class Junction:
    location: NDArray[np.float64]
    participants: List[Participation]

    @property
    def is_terminal(self):
        return any(p.role == Role.TERMINAL for p in self.participants)

    @property
    def is_anchored(self):
        return any(p.role in (Role.TERMINAL, Role.CUSP) for p in self.participants)

    @property
    def has_crossing(self):
        return any(p.role == Role.CROSSING for p in self.participants)

    def pair_angles_deg(self):
        def directional(p):
            if p.role == Role.CROSSING and p.tangent_out is not None:
                v = p.tangent_out - p.tangent_in
                n = float(np.linalg.norm(v))
                if n > 1e-9:
                    return v / n
            return p.tangent_in

        out = []
        n = len(self.participants)
        for i in range(n):
            ti = directional(self.participants[i])
            for j in range(i + 1, n):
                tj = directional(self.participants[j])
                cos = float(np.clip(abs(np.dot(ti, tj)), 0.0, 1.0))
                deg = float(np.degrees(np.arccos(cos)))
                out.append((i, j, deg))
        return out


class BuildConfig(TypedDict):
    junction_tol: float
    """Max cross-polyline distance for clustering into one junction."""

    terminal_tangent_window: int
    """Number of points to PCA-fit for the tangent at a TERMINAL endpoint."""

    crossing_tangent_skip: int
    """Baseline number of polyline points to skip on each side of an
    interior rep before sampling for the tangent PCA. The sampling
    point is then advanced FURTHER past any bridge points
    (fractional-coord interpolation that ``Segment.Config.Repair``
    adds around an LS-solved jp) until the next real skeleton point
    is reached — see ``_crossing_tangents``. The dynamic-bridge walk
    means this baseline can stay small (e.g. 2-3) and the function
    still handles arbitrarily long cascaded-repair bridges; the
    baseline is for skipping past non-bridge local noise (a small
    skeletonization jog right next to a real X crossing, etc).
    Set to 0 to disable the baseline skip entirely (dynamic bridge
    walk still applies)."""

    crossing_tangent_half_window: int
    """Half-width of the tangent window on each side of a crossing
    rep, applied AFTER ``crossing_tangent_skip`` is applied. The two
    windows that PCA-fit ``tangent_in`` and ``tangent_out`` each
    contain up to ``half_window + 1`` polyline points; larger = more
    robust to noise but less local. ~6 is reasonable."""

    cusp_angle_threshold_deg: float
    """Deflection (degrees) above which an interior junction point
    flips from CROSSING to CUSP. Default ~30°: catches real artistic
    corners (typically 60-120°) while accepting smooth crossings (0°)
    and modest noise."""

    cluster_merge_centroid_distance: float
    """Max centroid-to-centroid distance for two spatial clusters to
    be considered as merge candidates by the post-clustering merge
    pass. A single logical X-crossing whose polylines pass within
    ``junction_tol`` at the entrance, briefly diverge to more than
    ``junction_tol`` in the middle, then re-converge within
    ``junction_tol`` at the exit ends up split into TWO clusters by
    strict spatial clustering — this happens often after
    ``resolve_crossings`` draws two close-running Bresenham lines
    through a shallow-angle ribbon collapse. The merge pass identifies
    such pairs and unions them so the graph reports one junction.
    Set to 0 to disable. Default 10.0 (~4x junction_tol)."""

    cluster_merge_index_gap: int
    """Max polyline-index gap between two clusters' index ranges for
    them to be merged, applied PER shared polyline. Two clusters are
    only unioned if every shared polyline has a small gap, which is
    the test that distinguishes "one logical junction whose cluster
    fragmented" (small gaps; the polylines hardly traverse any real
    geometry between the fragments) from "two genuinely distinct
    junctions involving the same polyline pair" (large gaps; the
    polylines traverse real geometry between, e.g. a whisker that
    crosses the head outline at two separate points). Default 10."""


class StrokeGraph:
    class Config:
        class Build(BuildConfig):
            pass

    polylines: List[NDArray[np.float64]]
    junctions: List[Junction]
    polyline_to_junctions: Dict[int, List[Tuple[int, int]]]

    def __init__(self, polylines, build):
        self.polylines = [np.asarray(p, dtype=float) for p in polylines]
        self.junctions = _build_junctions(self.polylines, build)
        self.polyline_to_junctions = _index_polyline_to_junctions(self.junctions)

    def junctions_for(self, polyline_index):
        out = []
        for ji, pi in self.polyline_to_junctions.get(polyline_index, []):
            j = self.junctions[ji]
            out.append((j, j.participants[pi]))
        return out

    def is_closed(self, polyline_index):
        return _is_closed_polyline(self.polylines[polyline_index])

    def closed_polyline_indices(self):
        return [i for i in range(len(self.polylines)) if self.is_closed(i)]

    def stats(self):
        n_term_only = sum(
            1 for j in self.junctions if j.is_terminal and not j.has_crossing
        )
        n_cross_only = sum(
            1 for j in self.junctions if j.has_crossing and not j.is_terminal
        )
        n_mixed = sum(1 for j in self.junctions if j.is_terminal and j.has_crossing)
        return (
            f"{len(self.polylines)} polylines, "
            f"{len(self.junctions)} junctions "
            f"({n_term_only} pure-terminal, "
            f"{n_cross_only} pure-crossing, "
            f"{n_mixed} mixed)"
        )


def _build_junctions(polylines, config):
    clusters = _cluster_cross_polyline_points(polylines, config["junction_tol"])
    # Merge pass. When two clusters represent fragments of one logical
    # junction (typically created by close-running Bresenham lines from
    # resolve_crossings), strict spatial clustering leaves them
    # separate; this stitches them back together. Disabled by setting
    # cluster_merge_centroid_distance to 0.
    merge_dist = float(config.get("cluster_merge_centroid_distance", 0.0))
    if merge_dist > 0 and len(clusters) > 1:
        clusters = _merge_clusters_by_polyline_proximity(
            clusters,
            polylines,
            max_centroid_distance=merge_dist,
            max_index_gap=int(config.get("cluster_merge_index_gap", 10)),
        )
    junctions = []
    for cluster in clusters:
        cluster_xy = np.array([polylines[pi][ii] for pi, ii in cluster])
        centroid = cluster_xy.mean(axis=0)
        per_poly_rep = {}
        per_poly_rep_is_endpoint = {}
        for pi, ii in cluster:
            n = len(polylines[pi])
            is_endpoint = ii == 0 or ii == n - 1
            d2 = float(np.sum((polylines[pi][ii] - centroid) ** 2))
            if pi not in per_poly_rep:
                per_poly_rep[pi] = ii
                per_poly_rep_is_endpoint[pi] = is_endpoint
                continue
            cur_is_endpoint = per_poly_rep_is_endpoint[pi]
            cur = per_poly_rep[pi]
            cur_d2 = float(np.sum((polylines[pi][cur] - centroid) ** 2))
            if is_endpoint and not cur_is_endpoint:
                per_poly_rep[pi] = ii
                per_poly_rep_is_endpoint[pi] = True
            elif is_endpoint == cur_is_endpoint and d2 < cur_d2:
                per_poly_rep[pi] = ii
        if len(per_poly_rep) < 2:
            continue

        participants = [
            _make_participation(polylines[pi], pi, ii, config)
            for pi, ii in per_poly_rep.items()
        ]
        rep_xy = np.array(
            [polylines[p.polyline_index][p.point_index] for p in participants]
        )
        location = rep_xy.mean(axis=0)
        junctions.append(Junction(location=location, participants=participants))
    return junctions


def _index_polyline_to_junctions(junctions):
    p2j = {}
    for ji, j in enumerate(junctions):
        for pi_local, part in enumerate(j.participants):
            p2j.setdefault(part.polyline_index, []).append((ji, pi_local))
    return p2j


def _make_participation(poly, polyline_index, point_index, config):
    n = len(poly)
    is_closed = _is_closed_polyline(poly)
    is_start = point_index == 0
    is_end = point_index == n - 1
    is_endpoint_index = is_start or is_end

    if is_endpoint_index and not is_closed:
        terminal_end = 0 if is_start else 1
        tangent_in = _terminal_tangent(
            poly, terminal_end, config["terminal_tangent_window"]
        )
        return Participation(
            polyline_index=polyline_index,
            point_index=point_index,
            role=Role.TERMINAL,
            terminal_end=terminal_end,
            arc_length_t=0.0 if is_start else 1.0,
            tangent_in=tangent_in,
            tangent_out=None,
        )

    skip = config.get("crossing_tangent_skip", 0)
    if is_endpoint_index and is_closed:
        tangent_in, tangent_out = _closed_loop_tangents(
            poly, skip, config["crossing_tangent_half_window"]
        )
        arc_length_t = 0.0
    else:
        tangent_in, tangent_out = _crossing_tangents(
            poly, point_index, skip, config["crossing_tangent_half_window"]
        )
        arc_length_t = _arc_length_t(poly, point_index)

    if _is_cusp(tangent_in, tangent_out, config["cusp_angle_threshold_deg"]):
        role = Role.CUSP
    else:
        role = Role.CROSSING

    return Participation(
        polyline_index=polyline_index,
        point_index=point_index,
        role=role,
        terminal_end=None,
        arc_length_t=arc_length_t,
        tangent_in=tangent_in,
        tangent_out=tangent_out,
    )


def _is_cusp(tangent_in, tangent_out, threshold_deg):
    incoming_dir = -tangent_in
    cos = float(np.clip(np.dot(incoming_dir, tangent_out), -1.0, 1.0))
    deflection_deg = float(np.degrees(np.arccos(cos)))
    return deflection_deg >= threshold_deg


def _cluster_cross_polyline_points(polylines, tol):
    entries = []
    coords_list = []
    for pi, poly in enumerate(polylines):
        for ii in range(len(poly)):
            entries.append((pi, ii))
            coords_list.append(poly[ii])
    if not coords_list:
        return []
    coords = np.asarray(coords_list, dtype=float)

    cell = max(tol, 1e-9)
    grid = {}
    for idx in range(len(coords)):
        key = (int(coords[idx, 0] // cell), int(coords[idx, 1] // cell))
        grid.setdefault(key, []).append(idx)

    parent = list(range(len(coords)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    tol2 = tol * tol
    for (cx, cy), idxs in list(grid.items()):
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (cx + dx, cy + dy)
                if key in grid:
                    candidates.extend(grid[key])
        for i in idxs:
            pi_i = entries[i][0]
            xi, yi = coords[i, 0], coords[i, 1]
            for j in candidates:
                if j <= i:
                    continue
                if entries[j][0] == pi_i:
                    continue
                dxv = xi - coords[j, 0]
                dyv = yi - coords[j, 1]
                if dxv * dxv + dyv * dyv < tol2:
                    union(i, j)

    groups = {}
    for idx in range(len(coords)):
        groups.setdefault(find(idx), []).append(idx)
    out = []
    for members in groups.values():
        cluster = [entries[i] for i in members]
        polys_in_cluster = {pi for pi, _ in cluster}
        if len(polys_in_cluster) >= 2:
            out.append(cluster)
    return out


_CLOSED_TOL = 1.5


def _merge_clusters_by_polyline_proximity(
    clusters,
    polylines,
    max_centroid_distance,
    max_index_gap,
):
    """Union spatial clusters that fragment a single logical junction.

    A single X-crossing whose polylines briefly diverge to more than
    ``junction_tol`` in the middle (typical of resolve_crossings'
    close-running Bresenham lines through a shallow-angle ribbon
    collapse) ends up split into two clusters by strict spatial
    clustering. This pass identifies such pairs by three checks and
    unions them:

      1. They share ≥2 polylines (so it's the same pair of strokes
         in both, not unrelated junctions).
      2. Their centroids are within ``max_centroid_distance``.
      3. For EVERY shared polyline, the polyline-index gap between
         the two clusters' ranges in that polyline is within
         ``max_index_gap``. This is the test that distinguishes
         fragmented one-junction (small gaps) from genuinely two
         junctions involving the same pair of polylines (large
         gaps; the polyline traverses real geometry between them).

    Clusters that don't merge with anything pass through unchanged.
    """
    if len(clusters) <= 1:
        return clusters

    # Pre-compute centroid and per-polyline (min_idx, max_idx) ranges.
    centroids = []
    ranges = []
    for cl in clusters:
        xy = np.array([polylines[pi][ii] for pi, ii in cl])
        centroids.append(xy.mean(axis=0))
        per_poly = {}
        for pi, ii in cl:
            per_poly.setdefault(pi, []).append(ii)
        ranges.append({pi: (min(ix), max(ix)) for pi, ix in per_poly.items()})

    parent = list(range(len(clusters)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    max_dist_sq = max_centroid_distance * max_centroid_distance
    n = len(clusters)
    for i in range(n):
        for j in range(i + 1, n):
            shared = set(ranges[i].keys()) & set(ranges[j].keys())
            if len(shared) < 2:
                continue
            dx = centroids[i][0] - centroids[j][0]
            dy = centroids[i][1] - centroids[j][1]
            if dx * dx + dy * dy > max_dist_sq:
                continue
            ok = True
            for pi in shared:
                lo_i, hi_i = ranges[i][pi]
                lo_j, hi_j = ranges[j][pi]
                if lo_i > hi_j:
                    gap = lo_i - hi_j
                elif lo_j > hi_i:
                    gap = lo_j - hi_i
                else:
                    gap = 0  # ranges overlap
                if gap > max_index_gap:
                    ok = False
                    break
            if ok:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for indices in groups.values():
        combined = []
        for i in indices:
            combined.extend(clusters[i])
        merged.append(combined)
    return merged


def _is_closed_polyline(poly):
    if len(poly) < 3:
        return False
    return float(np.linalg.norm(poly[0] - poly[-1])) < _CLOSED_TOL


def _arc_length_t(polyline, idx):
    if len(polyline) <= 1:
        return 0.0
    diffs = np.diff(polyline, axis=0)
    seglens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglens)])
    total = float(cum[-1])
    if total <= 1e-9:
        return 0.0
    return float(cum[idx] / total)


def _pca_unit_tangent(samples, orient_toward):
    if len(samples) < 2:
        return np.array([1.0, 0.0])
    centered = samples - samples.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    t = Vt[0]
    n = float(np.linalg.norm(t))
    if n < 1e-9:
        return np.array([1.0, 0.0])
    t = t / n
    if float(np.dot(t, orient_toward)) < 0:
        t = -t
    return t


def _terminal_tangent(polyline, terminal_end, window):
    n = len(polyline)
    if n < 2:
        return np.array([1.0, 0.0])
    if terminal_end == 0:
        samples = polyline[: min(n, window + 1)]
        orient = samples[-1] - samples[0]
    else:
        samples = polyline[max(0, n - window - 1) :]
        orient = samples[0] - samples[-1]
    return _pca_unit_tangent(samples, orient)


def _is_integer_point(p, tol: float = 0.05) -> bool:
    """True if a polyline point's coords are at integer pixel positions.

    Polylines coming out of ``trace_skeleton`` sit at integer pixel
    coords (the skeleton is a boolean image). ``repair_junctions``
    REPLACES the strict spatial-cluster region of each polyline with
    a single LS-solved jp (fractional, since it's the algebraic
    intersection of approach lines) plus 1px-spaced linear
    interpolation back to the polyline's real interior (also
    fractional). So "is this point at integer coords" is a reliable
    test for "is this point original skeleton geometry" vs "is this
    point repair-imposed".
    """
    return abs(p[0] - round(p[0])) < tol and abs(p[1] - round(p[1])) < tol


def _crossing_tangents(polyline, idx, skip, half_window):
    """Outward unit tangents on either side of an interior crossing point.

    Returns ``(tangent_in, tangent_out)`` — both pointing AWAY from
    ``polyline[idx]``; ``tangent_in`` toward lower indices,
    ``tangent_out`` toward higher indices.

    Bridge handling (this is most of the function's reason to exist):
    after ``Segment.Config.Repair`` runs, polyline indices near a
    junction can be bridge interpolation between the polyline's real
    interior and the LS-solved jp — they are 1px-spaced and nearly
    colinear with the repair-imposed direction, so a PCA window that
    includes them is biased away from the polyline's natural geometry.
    A cascade of nearby repairs can produce a long stretch of bridge
    points on either side of an interior rep (up to 20+ points per
    side), so a fixed ``skip`` is unreliable. Instead, this function:

      1. Steps ``skip`` points outward from the rep on each side (the
         user-configured baseline to clear short bridges and any
         immediate noise).
      2. Then keeps walking outward past any further bridge points,
         detected by their fractional coordinates (real skeleton
         pixels are at integer positions, bridge interpolation is
         not — see ``_is_integer_point``).
      3. Samples up to ``half_window + 1`` points beyond that.

    The ``skip`` argument is still useful for skipping past
    NON-bridge local noise (e.g. small skeletonization jogs at an X
    crossing). For bridge-only contamination ``skip=0`` is enough
    because the dynamic walk handles the bridge.
    """
    n = len(polyline)
    if n < 2:
        return np.array([1.0, 0.0]), np.array([1.0, 0.0])
    here = polyline[idx]

    def find_real_sample_anchor(direction: int) -> int:
        """Walk outward from idx in `direction` (+1 or -1): apply the
        baseline skip, then continue walking past any bridge (non-
        integer) points until a real skeleton point is found or the
        polyline boundary is reached."""
        cur = idx + direction * skip
        # Clip into polyline bounds.
        cur = max(0, min(n - 1, cur))
        # Skip further bridge points (non-integer).
        while 0 < cur < n - 1 and not _is_integer_point(polyline[cur]):
            nxt = cur + direction
            if nxt < 0 or nxt >= n:
                break
            cur = nxt
        return cur

    # In side (toward lower indices).
    in_hi = find_real_sample_anchor(-1)
    in_lo = max(0, in_hi - half_window)
    if in_hi > in_lo:
        samples_in = polyline[in_lo : in_hi + 1]
        orient_in = polyline[in_lo] - here
        tangent_in = _pca_unit_tangent(samples_in, orient_in)
    else:
        tangent_in = np.array([1.0, 0.0])

    # Out side (toward higher indices).
    out_lo = find_real_sample_anchor(+1)
    out_hi = min(n - 1, out_lo + half_window)
    if out_hi > out_lo:
        samples_out = polyline[out_lo : out_hi + 1]
        orient_out = polyline[out_hi] - here
        tangent_out = _pca_unit_tangent(samples_out, orient_out)
    else:
        tangent_out = np.array([1.0, 0.0])

    return tangent_in, tangent_out


def _closed_loop_tangents(polyline, skip, half_window):
    """Tangents at the closure point of a closed polyline.

    Same intent and bridge-handling logic as ``_crossing_tangents``,
    but the "in" side wraps around to the END of the polyline (the
    duplicate of index 0 lives at index n-1) so we walk backward from
    n-2 rather than from idx-1.
    """
    n = len(polyline)
    if n < 4:
        return np.array([1.0, 0.0]), np.array([1.0, 0.0])
    here = polyline[0]

    # Forward side: from index 0 going up, stop short of n-1.
    fwd_anchor = min(max(0, n - 2), skip + 1)
    while fwd_anchor < n - 2 and not _is_integer_point(polyline[fwd_anchor]):
        fwd_anchor += 1
    fwd_hi = min(n - 2, fwd_anchor + half_window)
    if fwd_hi > fwd_anchor:
        fwd_samples = polyline[fwd_anchor : fwd_hi + 1]
        orient_fwd = fwd_samples[-1] - here
        tangent_out = _pca_unit_tangent(fwd_samples, orient_fwd)
    else:
        tangent_out = np.array([1.0, 0.0])

    # Backward side: from index n-2 going down.
    bwd_anchor = max(1, n - 2 - skip)
    while bwd_anchor > 1 and not _is_integer_point(polyline[bwd_anchor]):
        bwd_anchor -= 1
    bwd_lo = max(1, bwd_anchor - half_window)
    if bwd_anchor > bwd_lo:
        bwd_samples = polyline[bwd_lo : bwd_anchor + 1]
        orient_bwd = bwd_samples[0] - here
        tangent_in = _pca_unit_tangent(bwd_samples, orient_bwd)
    else:
        tangent_in = np.array([1.0, 0.0])

    return tangent_in, tangent_out
