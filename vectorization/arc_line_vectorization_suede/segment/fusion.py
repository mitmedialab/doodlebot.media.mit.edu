"""Fuse adjacent skeleton segments into longer continuous strokes.

After skeletonization and segmentation we have a list of polylines, each
running between two node pixels. At a branch where multiple segments meet,
the original drawing was likely a single continuous pen stroke that crossed
through; the skeletonizer broke it because the local topology is a 'star'.

This module re-fuses such segments. The matching rule has two parts:

  (1) Reachability in the *original binary* (not just the skeleton): two
      endpoints are candidates only if a short 8-connected path through
      ink connects them. This handles "noisy" junctions where the two
      segments have endpoints separated by a thick blob of original ink
      that the skeletonizer chewed up.

  (2) Tangent alignment at the endpoints: of all candidates, the best
      match is the one whose outward tangent is most anti-parallel to
      the target's outward tangent, i.e. the one that looks like the
      smoothest continuation of the same stroke.

A greedy maximum-weight matching picks pairs starting from the highest
score, so at a 4-way junction the two best-aligned segments fuse first
and the remaining two fuse second (or stay separate).

This module also exposes ``fuse_post_repair``, a second-pass fusion that
runs AFTER junction repair (and after chromosome resolution). It drops
the A*-through-binary path-finding because by then the polylines that
should join already share a clean endpoint coordinate; the only question
is which pairs at each shared location are the best-aligned continuation.
See its docstring for details.
"""

from __future__ import annotations
import heapq
import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, TypedDict

import numpy as np
from numpy.typing import NDArray

from ._helpers import (
    gather_endpoints,
    spatial_clusters,
    stable_endpoint_tangent_and_curvature,
)

# ============================================================================
# Types
# ============================================================================


@dataclass
class FusionCandidate:
    """A potential fusion between two segment endpoints.

    Attributes:
        seg_a, end_a: index of the segment and which side (0 = polyline
            start, 1 = polyline end) of its two endpoints is involved.
        seg_b, end_b: same for the other segment.
        connecting_path: (K, 2) (x, y) pixels running from endpoint A to
            endpoint B through the original binary ink. Inclusive of both
            endpoints. 8-connected pixel-by-pixel. EMPTY for post-repair
            fusion candidates (which join already-co-located endpoints).
        path_length: number of pixels in `connecting_path`. 0 for
            post-repair candidates.
        tangent_score: in [-1, 1]. +1 = perfectly smooth continuation
            (outward tangents anti-parallel); 0 = perpendicular;
            -1 = the two segments fold back the same way at the junction.
        curvature_a, curvature_b: local curvature magnitudes at each
            endpoint (1/pixel units). Used by the score to penalize
            matches between segments of very different curvature.
        score: combined sort key for the greedy matcher.
    """

    seg_a: int
    end_a: int
    seg_b: int
    end_b: int
    connecting_path: NDArray[np.float64]
    path_length: float
    tangent_score: float
    curvature_a: float
    curvature_b: float
    score: float


# ============================================================================
# Helpers
# ============================================================================

NEIGHBOR_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _shortest_path_through_binary(
    binary: NDArray[np.bool_],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    max_cost: float,
) -> Optional[List[Tuple[int, int]]]:
    """Optimal 8-connected shortest path through True pixels.

    Uses A* with Euclidean step costs (1.0 for orthogonal moves, sqrt(2)
    for diagonals) and the octile-distance heuristic — the tightest
    admissible heuristic for this cost structure, which guarantees A*
    returns a true shortest-Euclidean-distance path while pruning the
    search aggressively.

    Returns the path as a list of (y, x) tuples (inclusive of both
    endpoints), or None if `goal` is not reachable from `start` through
    True pixels within `max_cost` total Euclidean distance.
    """
    H, W = binary.shape
    if not (0 <= start[0] < H and 0 <= start[1] < W and binary[start]):
        return None
    if not (0 <= goal[0] < H and 0 <= goal[1] < W and binary[goal]):
        return None
    if start == goal:
        return [start]

    SQRT2 = math.sqrt(2.0)
    # (dy, dx, step_cost)
    moves = (
        (-1, -1, SQRT2),
        (-1, 0, 1.0),
        (-1, 1, SQRT2),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, SQRT2),
        (1, 0, 1.0),
        (1, 1, SQRT2),
    )

    def octile(p: Tuple[int, int]) -> float:
        dy = abs(p[0] - goal[0])
        dx = abs(p[1] - goal[1])
        # octile distance = max + (sqrt(2) - 1) * min
        if dy > dx:
            return dy + (SQRT2 - 1.0) * dx
        return dx + (SQRT2 - 1.0) * dy

    # Priority queue items: (f_score, tiebreak_counter, node)
    open_pq: List[Tuple[float, int, Tuple[int, int]]] = []
    counter = 0
    g_score: dict = {start: 0.0}
    parent: dict = {start: None}
    heapq.heappush(open_pq, (octile(start), counter, start))

    while open_pq:
        _, _, cur = heapq.heappop(open_pq)
        if cur == goal:
            path: List[Tuple[int, int]] = [goal]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            path.reverse()
            return path
        cur_g = g_score[cur]
        # Stale entry (re-inserted with a worse g)? skip.
        # (We don't need an explicit closed set because we only relax when
        # we strictly improve g_score below.)
        cy, cx = cur
        for dy, dx, step_cost in moves:
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < H and 0 <= nx < W):
                continue
            if not binary[ny, nx]:
                continue
            tentative_g = cur_g + step_cost
            if tentative_g > max_cost:
                continue
            nbr = (ny, nx)
            if nbr in g_score and tentative_g >= g_score[nbr]:
                continue
            g_score[nbr] = tentative_g
            parent[nbr] = cur
            f_new = tentative_g + octile(nbr)
            counter += 1
            heapq.heappush(open_pq, (f_new, counter, nbr))
    return None


def _local_tangent_and_curvature(
    polyline: NDArray[np.float64],
    at_end: bool,
    lookback: int,
) -> Tuple[NDArray[np.float64], float]:
    """Estimate outward tangent and unsigned local curvature near an endpoint.

    Uses PCA on the last (or first) `lookback` pixels rather than a simple
    two-point difference, which makes the tangent much more robust to
    single-pixel "jog" artifacts that skeletonization sometimes introduces
    near junctions. The principal axis dominates as long as the bulk of the
    pixels follow the segment's true direction, even if the very last one
    or two pixels drift sideways.

    Curvature is approximated by the small-arc sagitta-to-curvature
    relation: for a circular arc of chord length L and maximum sagitta s,
    kappa = 1/R ~= 8 s / L^2. We measure s as the largest perpendicular
    deviation of the lookback pixels from their PCA principal axis.
    Returns 0 when the polyline is too short to support a meaningful
    estimate.
    """
    n = min(lookback, len(polyline))
    if n < 3:
        if at_end:
            d = polyline[-1] - polyline[0]
        else:
            d = polyline[0] - polyline[-1]
        norm = float(np.linalg.norm(d))
        return (d / norm if norm > 1e-9 else np.zeros(2)), 0.0

    pts = polyline[-n:] if at_end else polyline[:n]

    # PCA on the lookback window. The principal axis is the tangent
    # direction (unsigned); the perpendicular axis is what we measure
    # sagitta against.
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    tangent = Vt[0]
    perp = Vt[1]

    # Orient the tangent outward (i.e. pointing away from the segment's
    # interior, toward the chosen endpoint).
    ref = (pts[-1] - pts[0]) if at_end else (pts[0] - pts[-1])
    if float(np.dot(tangent, ref)) < 0:
        tangent = -tangent

    # Curvature magnitude proxy
    perp_dists = np.abs(centered @ perp)
    max_perp = float(perp_dists.max())
    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    if chord < 1.0:
        return tangent, 0.0
    curvature = 8.0 * max_perp / (chord * chord)
    return tangent, curvature


# ============================================================================
# Candidate enumeration & matching (FIRST PASS)
# ============================================================================


class FuseConfig(TypedDict):
    max_path_length: int
    lookback: int
    gap_penalty: float
    min_tangent_score: float
    curvature_penalty: float


def find_fusion_candidates(
    segments: List[NDArray[np.float64]],
    binary: NDArray[np.bool_],
    max_path_length: int,
    lookback: int,
    gap_penalty: float,
    curvature_penalty: float,
) -> List[FusionCandidate]:
    """Enumerate every (endpoint, endpoint) pair that is:
        (1) within 2 * max_path_length Euclidean distance, AND
        (2) reachable through ink in <= max_path_length BFS steps.
    Each pair becomes a FusionCandidate scored by tangent alignment minus
    a small penalty for gap length and a curvature-mismatch penalty.

    Tangents and curvatures are estimated by PCA over the last `lookback`
    pixels at each endpoint (more robust to small skeletonization jogs
    than a two-point difference). The curvature term penalizes matches
    between segments of very different local curvature — e.g. a straight
    line meeting a tight curve.

    score = tangent_score
            - gap_penalty       * path_length
            - curvature_penalty * |curvature_a - curvature_b|
    """
    candidates: List[FusionCandidate] = []
    endpoints: List[Tuple[int, int, NDArray]] = []
    for i, seg in enumerate(segments):
        if len(seg) < 2:
            continue
        endpoints.append((i, 0, seg[0]))
        endpoints.append((i, 1, seg[-1]))

    for ai in range(len(endpoints)):
        seg_a, end_a, pt_a = endpoints[ai]
        for bi in range(ai + 1, len(endpoints)):
            seg_b, end_b, pt_b = endpoints[bi]
            if seg_a == seg_b:
                continue
            d_eucl = float(np.linalg.norm(pt_a - pt_b))
            if d_eucl > 2 * max_path_length:
                continue
            ya, xa = int(round(pt_a[1])), int(round(pt_a[0]))
            yb, xb = int(round(pt_b[1])), int(round(pt_b[0]))
            path_yx = _shortest_path_through_binary(
                binary, (ya, xa), (yb, xb), max_path_length
            )
            if path_yx is None:
                continue
            path_xy = np.array([(x, y) for y, x in path_yx], dtype=float)
            path_length = float(len(path_yx))

            tan_a, curv_a = _local_tangent_and_curvature(
                segments[seg_a], at_end=(end_a == 1), lookback=lookback
            )
            tan_b, curv_b = _local_tangent_and_curvature(
                segments[seg_b], at_end=(end_b == 1), lookback=lookback
            )
            tangent_score = float(-np.dot(tan_a, tan_b))
            curvature_diff = abs(curv_a - curv_b)
            score = (
                tangent_score
                - gap_penalty * path_length
                - curvature_penalty * curvature_diff
            )
            candidates.append(
                FusionCandidate(
                    seg_a=seg_a,
                    end_a=end_a,
                    seg_b=seg_b,
                    end_b=end_b,
                    connecting_path=path_xy,
                    path_length=path_length,
                    tangent_score=tangent_score,
                    curvature_a=curv_a,
                    curvature_b=curv_b,
                    score=score,
                )
            )
    return candidates


def fuse_segments(
    segments: List[NDArray[np.float64]],
    binary: NDArray[np.bool_],
    config: FuseConfig,
) -> Tuple[List[NDArray[np.float64]], List[FusionCandidate], List[FusionCandidate]]:
    """Fuse segments by greedy maximum-weight matching of endpoints.

    Returns:
        (fused_segments, accepted, all_candidates)
            fused_segments:  list of (M, 2) polylines after fusion
            accepted:        candidates used in the final matching
            all_candidates:  every candidate considered, accepted or not
    """
    all_cands = find_fusion_candidates(
        segments,
        binary,
        max_path_length=config["max_path_length"],
        lookback=config["lookback"],
        gap_penalty=config["gap_penalty"],
        curvature_penalty=config["curvature_penalty"],
    )
    cands = [c for c in all_cands if c.tangent_score >= config["min_tangent_score"]]
    cands.sort(key=lambda c: c.score, reverse=True)

    n = len(segments)
    used: set = set()
    accepted: List[FusionCandidate] = []
    # link[seg][end] = (other_seg, other_end, connecting_path)  or  None
    link: List[List[Optional[Tuple[int, int, NDArray]]]] = [
        [None, None] for _ in range(n)
    ]

    for c in cands:
        ka = (c.seg_a, c.end_a)
        kb = (c.seg_b, c.end_b)
        if ka in used or kb in used:
            continue
        used.add(ka)
        used.add(kb)
        link[c.seg_a][c.end_a] = (c.seg_b, c.end_b, c.connecting_path)
        link[c.seg_b][c.end_b] = (c.seg_a, c.end_a, c.connecting_path[::-1])
        accepted.append(c)

    visited = [False] * n
    fused: List[NDArray[np.float64]] = []

    def walk(start_seg: int, start_end: int) -> NDArray[np.float64]:
        """Walk the chain starting at segments[start_seg] from endpoint
        `start_end`, emitting a single concatenated polyline."""
        chain: List[NDArray] = []
        cur_seg, cur_end = start_seg, start_end
        while not visited[cur_seg]:
            visited[cur_seg] = True
            poly = segments[cur_seg]
            pts = poly if cur_end == 0 else poly[::-1]
            if chain and np.allclose(chain[-1], pts[0], atol=0.5):
                chain.extend(pts[1:])
            else:
                chain.extend(pts)
            exit_end = 1 - cur_end
            link_info = link[cur_seg][exit_end]
            if link_info is None:
                break
            next_seg, next_end, conn_path = link_info
            if visited[next_seg]:
                break  # cycle
            # Insert the *interior* of the connecting path; the two endpoints
            # are already in `chain` (last) and will be the first pixel of
            # the next segment.
            if len(conn_path) > 2:
                interior = conn_path[1:-1]
                chain.extend(interior)
            cur_seg, cur_end = next_seg, next_end
        return np.array(chain, dtype=float)

    # Walk from free (unlinked) endpoints first.
    for i in range(n):
        if visited[i]:
            continue
        start_end = None
        if link[i][0] is None:
            start_end = 0
        elif link[i][1] is None:
            start_end = 1
        if start_end is not None:
            fused.append(walk(i, start_end))
    # Whatever remains is part of a fully-linked cycle.
    for i in range(n):
        if visited[i]:
            continue
        fused.append(walk(i, 0))

    return fused, accepted, all_cands


# ============================================================================
# Post-repair fusion (SECOND PASS)
# ============================================================================


class PostRepairFuseConfig(TypedDict):
    junction_tol: float
    """Max distance between endpoints (from different polylines) to be
    considered as meeting at the same point. Should match (or be a
    touch larger than) the value used in repair."""

    tangent_skip: int
    """Number of points to skip from each endpoint before sampling for
    tangent estimation. Skipping past repair's bridge points avoids a
    PCA tangent biased toward the OLD junction location. ~5 is
    reasonable when ``repair`` ran with ``interp_max_spacing=1.0``."""

    tangent_sample: int
    """Number of points to PCA-fit for tangent estimation, starting
    ``tangent_skip`` points in from the endpoint. ~10 is reasonable."""

    min_tangent_score: float
    """Threshold on ``-dot(tan_a, tan_b)`` below which a pair is not
    considered a fusion candidate at all. 0.6-0.7 is reasonable; the
    first pass accepts down to about 0.6 in typical configs and the
    second pass can afford to be a touch stricter since tangents are
    cleaner."""

    curvature_penalty: float
    """Penalty per unit local-curvature difference between the two
    endpoints being joined. Discourages joining a tight curve to a
    straight section. Keep small (~1.0); curvature estimates from a
    short window are noisy and shouldn't dominate the score."""


def fuse_post_repair(
    polylines: List[NDArray[np.float64]],
    config: PostRepairFuseConfig,
) -> Tuple[List[NDArray[np.float64]], List[FusionCandidate]]:
    """Second-pass fusion for polylines that share endpoint coordinates
    after junction repair (and optionally chromosome resolution).

    Unlike ``fuse_segments``, this doesn't do A*-through-binary path
    finding. Post-repair, polylines that should be one stroke already
    share an endpoint coordinate (the LS-cleaned junction point, or the
    chromosome midpoint), so the only question is which pairs at each
    shared location are the smoothest continuation.

    Tangents at endpoints are estimated by ``stable_endpoint_tangent_
    and_curvature``, which samples past repair's bridge points. Without
    that, the PCA tangent would be biased by the bridge's near-colinear
    points and could point at the OLD junction location instead of in
    the polyline's true direction.

    Matching is greedy max-weight, same as the first pass. Walking the
    chain is simpler because there are no connecting paths to splice
    in: the two ends of every accepted link are already at the same
    coordinate, so we just dedupe the seam.

    Returns:
        ``(joined_polylines, accepted)``:

        - ``joined_polylines``: the new polyline list with accepted
          pairs concatenated.

        - ``accepted``: the candidates that were used. Their
          ``connecting_path`` arrays are empty since there's no path
          to bridge; ``path_length`` is 0.
    """
    polylines = [np.asarray(p, dtype=float) for p in polylines]
    n = len(polylines)
    if n == 0:
        return [], []

    meta, coords = gather_endpoints(polylines)
    if len(coords) == 0:
        return list(polylines), []
    point_poly_idx = [m[0] for m in meta]
    raw_clusters = spatial_clusters(coords, config["junction_tol"], point_poly_idx)

    # Build candidates.
    candidates: List[FusionCandidate] = []
    skip = config["tangent_skip"]
    sample = config["tangent_sample"]
    min_score = config["min_tangent_score"]
    curv_pen = config["curvature_penalty"]
    for cl in raw_clusters:
        members = [meta[i] for i in cl]
        if len({pi for pi, _ in members}) < 2:
            continue
        # Cache tangents within the cluster to avoid recomputing per pair.
        member_tans: List[Tuple[int, int, NDArray, float]] = []
        for pi, end in members:
            tan, curv = stable_endpoint_tangent_and_curvature(
                polylines[pi], end, skip, sample
            )
            member_tans.append((pi, end, tan, curv))
        m = len(member_tans)
        for i in range(m):
            pi_a, end_a, tan_a, curv_a = member_tans[i]
            for j in range(i + 1, m):
                pi_b, end_b, tan_b, curv_b = member_tans[j]
                if pi_a == pi_b:
                    continue
                tan_score = float(-np.dot(tan_a, tan_b))
                if tan_score < min_score:
                    continue
                score = tan_score - curv_pen * abs(curv_a - curv_b)
                candidates.append(
                    FusionCandidate(
                        seg_a=pi_a,
                        end_a=end_a,
                        seg_b=pi_b,
                        end_b=end_b,
                        connecting_path=np.empty((0, 2), dtype=float),
                        path_length=0.0,
                        tangent_score=tan_score,
                        curvature_a=curv_a,
                        curvature_b=curv_b,
                        score=score,
                    )
                )

    # Greedy max-weight matching.
    candidates.sort(key=lambda c: c.score, reverse=True)
    used: Set[Tuple[int, int]] = set()
    accepted: List[FusionCandidate] = []
    link: List[List[Optional[Tuple[int, int]]]] = [[None, None] for _ in range(n)]
    for c in candidates:
        ka = (c.seg_a, c.end_a)
        kb = (c.seg_b, c.end_b)
        if ka in used or kb in used:
            continue
        used.add(ka)
        used.add(kb)
        link[c.seg_a][c.end_a] = (c.seg_b, c.end_b)
        link[c.seg_b][c.end_b] = (c.seg_a, c.end_a)
        accepted.append(c)

    # Walk chains and concatenate.
    visited = [False] * n
    fused: List[NDArray[np.float64]] = []

    def walk(start_seg: int, start_end: int) -> NDArray[np.float64]:
        """Walk the chain starting at polylines[start_seg] from endpoint
        side `start_end`. The seam between successive polylines is at
        the same coordinate by construction, so the duplicate point at
        the join is dropped by an ``allclose`` check."""
        chain: List[NDArray[np.float64]] = []
        cur_seg, cur_end = start_seg, start_end
        while not visited[cur_seg]:
            visited[cur_seg] = True
            poly = polylines[cur_seg]
            pts = poly if cur_end == 0 else poly[::-1]
            if chain and np.allclose(chain[-1], pts[0], atol=0.5):
                chain.extend(pts[1:])
            else:
                chain.extend(pts)
            exit_end = 1 - cur_end
            link_info = link[cur_seg][exit_end]
            if link_info is None:
                break
            next_seg, next_end = link_info
            if visited[next_seg]:
                break  # cycle
            cur_seg, cur_end = next_seg, next_end
        return np.array(chain, dtype=float)

    # Walk from free (unlinked) endpoints first.
    for i in range(n):
        if visited[i]:
            continue
        start_end: Optional[int] = None
        if link[i][0] is None:
            start_end = 0
        elif link[i][1] is None:
            start_end = 1
        if start_end is not None:
            fused.append(walk(i, start_end))
    # Whatever remains is in a fully-linked cycle.
    for i in range(n):
        if visited[i]:
            continue
        fused.append(walk(i, 0))

    return fused, accepted
