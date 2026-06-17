"""Shared spatial helpers for segment processing modules.

Used by:

- ``repair.py``: cluster ALL polyline points to discover junctions,
  including the interior points where a polyline passes smoothly
  through another polyline's endpoint.

- ``chromosome.py``: cluster polyline ENDPOINTS to find the
  multi-polyline junction clusters that bracket a candidate
  centromere.

- ``fusion.py`` (post-repair pass): cluster polyline ENDPOINTS to
  find polylines that share a clean post-repair junction location,
  which are then matched pairwise by tangent alignment.

The leading underscore in the module name indicates these are package-
private; downstream consumers should not import from this module.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

# ============================================================================
# Point gathering
# ============================================================================


def gather_points(
    polylines: List[NDArray[np.float64]],
) -> Tuple[List[Tuple[int, int]], NDArray[np.float64]]:
    """Flatten every point of every polyline into one coords array,
    paired with metadata of ``(poly_idx, point_idx)`` tuples.

    Used by junction repair, which has to discover crossings where the
    junction sits at an INTERIOR index of one or more polylines (a
    whisker piercing a head outline; the head outline is a long
    polyline whose junction point isn't an endpoint).
    """
    meta: List[Tuple[int, int]] = []
    coords_list: List[NDArray[np.float64]] = []
    for pi, poly in enumerate(polylines):
        for ii in range(len(poly)):
            meta.append((pi, ii))
            coords_list.append(poly[ii])
    if not coords_list:
        return meta, np.empty((0, 2), dtype=float)
    return meta, np.asarray(coords_list, dtype=float)


def gather_endpoints(
    polylines: List[NDArray[np.float64]],
) -> Tuple[List[Tuple[int, int]], NDArray[np.float64]]:
    """Flatten polyline endpoints (start and end of each polyline) with
    parallel ``(poly_idx, end_side)`` metadata. ``end_side`` is 0 for
    the polyline's first point, 1 for its last.

    Polylines with fewer than 2 points are skipped (no meaningful
    endpoint pair).

    Used by post-repair fusion and chromosome detection: after repair,
    polylines that share a junction have exact (or near-exact) shared
    endpoint coordinates, so clustering JUST the endpoints is both
    sufficient and significantly cheaper than clustering all points.
    """
    meta: List[Tuple[int, int]] = []
    coords_list: List[NDArray[np.float64]] = []
    for pi, poly in enumerate(polylines):
        if len(poly) < 2:
            continue
        meta.append((pi, 0))
        coords_list.append(poly[0])
        meta.append((pi, 1))
        coords_list.append(poly[-1])
    if not coords_list:
        return meta, np.empty((0, 2), dtype=float)
    return meta, np.asarray(coords_list, dtype=float)


# ============================================================================
# Spatial clustering
# ============================================================================


def spatial_clusters(
    coords: NDArray[np.float64],
    tol: float,
    point_poly_idx: List[int],
) -> List[List[int]]:
    """Union-find clustering of 2D points with a cross-polyline rule.

    Two points are unioned IFF (a) they belong to DIFFERENT polylines
    AND (b) they are within ``tol`` of each other. The cross-polyline
    restriction is exactly what makes the result correspond to
    JUNCTIONS: a long polyline that folds back close to itself is not
    unioned to itself, so it doesn't spuriously generate a junction at
    every fold.

    Uses a hashed grid of cell size ``tol`` so each point only checks
    points in its own 3x3 neighborhood of grid cells, keeping the cost
    near-linear in the number of input points.

    Returns a list of clusters, each a list of indices into ``coords``.
    Singleton clusters (one point) ARE returned; callers that want only
    multi-polyline clusters must filter themselves.
    """
    n = len(coords)
    if n == 0:
        return []
    cell_size = max(tol, 1e-9)
    grid: Dict[Tuple[int, int], List[int]] = {}
    for idx in range(n):
        c = (int(coords[idx, 0] // cell_size), int(coords[idx, 1] // cell_size))
        grid.setdefault(c, []).append(idx)
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

    tol_sq = tol * tol
    for (cx, cy), idxs in list(grid.items()):
        candidates: List[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (cx + dx, cy + dy)
                if key in grid:
                    candidates.extend(grid[key])
        for i in idxs:
            xi, yi = coords[i, 0], coords[i, 1]
            pi_i = point_poly_idx[i]
            for j in candidates:
                if j <= i:
                    continue
                if point_poly_idx[j] == pi_i:
                    continue
                dx_ = xi - coords[j, 0]
                dy_ = yi - coords[j, 1]
                if dx_ * dx_ + dy_ * dy_ < tol_sq:
                    union(i, j)
    groups: Dict[int, List[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)
    return list(groups.values())


# ============================================================================
# Stable endpoint tangent (post-repair-aware)
# ============================================================================


def stable_endpoint_tangent_and_curvature(
    polyline: NDArray[np.float64],
    end_side: int,
    skip: int,
    sample: int,
) -> Tuple[NDArray[np.float64], float]:
    """Outward unit tangent and unsigned local curvature at one of a
    polyline's endpoints, estimated from a window of stable interior
    points past any post-repair "bridge".

    Background: ``repair_junctions`` replaces each junction-affected
    polyline region with a single LS-solved junction point, then
    interpolates 1-pixel-spaced points between that single point and
    the polyline's stable interior. Those bridge points are nearly
    colinear and lie BETWEEN the polyline's natural geometry and the
    LS junction location - so PCA over a window that's mostly bridge
    points returns a tangent pointing at the OLD junction location,
    not in the polyline's true direction. For post-repair fusion and
    chromosome detection we need a tangent estimate that ignores the
    bridge.

    Strategy: skip the first ``skip`` points from the endpoint, then
    PCA-fit the next ``sample`` points. With ``skip ~ 5`` (typical
    bridge length is 4-6 points for the repair interp_max_spacing of
    1.0 px) and ``sample ~ 10``, the window is genuinely stable.

    Outward orientation: the returned tangent points TOWARD the chosen
    endpoint (i.e. in the direction the stroke "leaves" the polyline
    body, away from the interior), which is the convention fusion uses
    for scoring anti-parallel matches.

    Curvature via the small-arc sagitta-to-curvature relation:
    kappa ~ 8 * s / L^2 where s is the maximum perpendicular deviation
    of the sample window from its PCA principal axis and L is the
    chord length of the window. Returns 0 for windows too short or
    degenerate to support a meaningful curvature estimate.
    """
    n = len(polyline)
    if n < 2:
        return np.array([1.0, 0.0]), 0.0

    if end_side == 0:
        lo = min(skip, max(0, n - 2))
        hi = min(lo + sample, n)
    else:
        hi = max(n - skip, 2)
        lo = max(hi - sample, 0)

    pts = polyline[lo:hi]
    if len(pts) < 2:
        # Fallback: full-polyline chord.
        if end_side == 0:
            d = polyline[-1] - polyline[0]
        else:
            d = polyline[0] - polyline[-1]
        norm = float(np.linalg.norm(d))
        return (d / norm if norm > 1e-9 else np.array([1.0, 0.0])), 0.0

    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    tangent = Vt[0]

    endpoint = polyline[0] if end_side == 0 else polyline[-1]
    ref = endpoint - centroid
    if float(np.dot(tangent, ref)) < 0:
        tangent = -tangent

    if len(pts) < 3:
        return tangent, 0.0
    perp = Vt[1]
    perp_dists = np.abs(centered @ perp)
    max_perp = float(perp_dists.max())
    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    if chord < 1.0:
        return tangent, 0.0
    curvature = 8.0 * max_perp / (chord * chord)
    return tangent, curvature
