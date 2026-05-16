"""Per-segment fitting and chain subdivision.

Two layers:

* Single-primitive fitters: ``fit_line``, ``fit_circle``, ``fit_arc``.
  These are the building blocks both for terminal classification and as
  the per-window cost function in chain subdivision.
* ``fit_segment_chain`` / ``fit_segment_topdown``: split a long polyline
  into a sequence of primitives that share endpoints, trading off
  fidelity against complexity.

Single-primitive fits return a ``(primitive, rms)`` pair so the caller
can decide whether the fit was good enough to commit to.

Chain subdivision: a single segment from the upstream pipeline may be a
long fluid stroke that needs multiple primitives. The DP variant
minimizes ``Σ SSE_i + λ * n_primitives`` over all chain decompositions,
which is the MDL-style "fidelity vs simplicity" tradeoff from Favreau
et al. Top-down recursive split (``fit_segment_topdown``) is the
simpler, near-linear fallback that often produces cleaner splits at
high-curvature points because the split location IS the worst-fit
point.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from .primitives import Arc, Circle, Line, Primitive


_EPS = 1e-9


# ---------------------------------------------------------------------------
# Polyline utilities


def polyline_length(pts: NDArray[np.float64]) -> float:
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def is_closed_polyline(pts: NDArray[np.float64], tol: float = 1.5) -> bool:
    if len(pts) < 4:
        return False
    return float(np.linalg.norm(pts[0] - pts[-1])) < tol


def is_near_closed_polyline(
    pts: NDArray[np.float64],
    gap_ratio: float = 0.10,
    abs_tol: float = 10.0,
) -> bool:
    """Like ``is_closed_polyline`` but also catches polylines where the
    endpoints have a small gap *relative to* the polyline's arc length.
    Hand-drawn wheels and other closed shapes often have a 5-30 px gap
    where the stroke didn't quite meet itself; we want to recognize these
    as Circles, not as 350° arcs.

    A polyline is "near-closed" if EITHER:
      * absolute endpoint gap < ``abs_tol`` (covers small drawings), OR
      * gap / polyline_length < ``gap_ratio`` (covers larger drawings).
    """
    if len(pts) < 4:
        return False
    gap = float(np.linalg.norm(pts[0] - pts[-1]))
    if gap < abs_tol:
        return True
    L = polyline_length(pts)
    if L < _EPS:
        return False
    return gap / L < gap_ratio


def find_corners(
    pts: NDArray[np.float64],
    sharp_threshold_deg: float = 80.0,
    soft_threshold_deg: float = 50.0,
    sustain_indices: int = 3,
    min_separation: Optional[int] = None,
    window: Optional[int] = None,
) -> List[int]:
    """Like ``count_corners`` but returns the *indices* of detected
    corners (into ``pts``). See ``count_corners`` for the algorithm.

    A corner is reported when EITHER:
      * the turn at this index exceeds ``sharp_threshold_deg`` (a
        clear, sharp corner that doesn't need any neighborhood
        check — e.g., a cat-ear apex at ~130°); OR
      * the turn exceeds ``soft_threshold_deg`` AND every turn
        within ``sustain_indices`` on each side ALSO exceeds
        ``soft_threshold_deg`` (a gentle but sustained corner — a
        house corner at ~80° has neighbors also at ~75-83° for many
        indices, whereas a noise spike on a circle has a high turn
        at one index with neighbors at <25°).

    The neighbor-sustained check is the key discriminator between
    real corners (which are wide in turn-angle-space because the
    polyline has a sustained direction change over several samples)
    and per-pixel noise on a hand-drawn curve (which produces
    isolated spikes whose neighbors drop back to baseline).

    The tangent window scales with polyline length:
    ``w = max(8, min(30, n // 30))``. Short polylines use ``w=8``
    (avoids chord-to-chord false positives on small clean circles);
    long polylines use up to ``w=30`` to recover corners the
    upstream skeletonization rounded over ~10 pixels.
    """
    n = len(pts)
    if window is None:
        w = max(8, min(30, n // 30))
    else:
        w = window
    if min_separation is None:
        min_separation = max(12, w + 4)
    if n < 2 * w + 1:
        return []

    left = pts[w:n - w] - pts[: n - 2 * w]
    right = pts[2 * w:] - pts[w:n - w]
    ll = np.linalg.norm(left, axis=1)
    rl = np.linalg.norm(right, axis=1)
    valid = (ll > _EPS) & (rl > _EPS)
    left = left.copy()
    right = right.copy()
    left[valid] = left[valid] / ll[valid, None]
    right[valid] = right[valid] / rl[valid, None]

    dots = np.einsum("ij,ij->i", left, right)
    dots = np.clip(dots, -1.0, 1.0)
    turns = np.arccos(dots)  # radians

    sharp_thresh = math.radians(sharp_threshold_deg)
    soft_thresh = math.radians(soft_threshold_deg)
    n_turns = len(turns)

    corners: List[int] = []
    i = 0
    while i < n_turns:
        t = turns[i]
        is_sharp = t > sharp_thresh
        # Sustained-soft: turn is above soft threshold AND so are its
        # immediate neighbors at +/-sustain_indices.
        is_sustained = False
        if t > soft_thresh:
            lo_idx = i - sustain_indices
            hi_idx = i + sustain_indices
            if (lo_idx >= 0 and hi_idx < n_turns
                    and turns[lo_idx] > soft_thresh
                    and turns[hi_idx] > soft_thresh):
                is_sustained = True
        if is_sharp or is_sustained:
            # Non-max suppression in [i, i+min_separation).
            j_end = min(n_turns, i + min_separation)
            best = i
            for j in range(i + 1, j_end):
                if turns[j] > turns[best]:
                    best = j
            corners.append(best + w)
            i = best + min_separation
        else:
            i += 1
    return corners


def count_corners(
    pts: NDArray[np.float64],
    sharp_threshold_deg: float = 80.0,
    soft_threshold_deg: float = 50.0,
    sustain_indices: int = 3,
    min_separation: Optional[int] = None,
    window: Optional[int] = None,
) -> int:
    """Count the number of corners along a polyline. See
    ``find_corners`` for the algorithm.
    """
    return len(find_corners(
        pts, sharp_threshold_deg, soft_threshold_deg, sustain_indices,
        min_separation, window,
    ))


# ---------------------------------------------------------------------------
# Single-primitive fitters


def fit_line(pts: NDArray[np.float64]) -> Tuple[Line, float]:
    """ODR line fit. Returns (Line, rms_perp_residual).

    Math: minimize Σ ‖(pᵢ - μ) - ((pᵢ - μ)·d) d‖² over unit d. Closed-form
    via SVD of the centered point matrix; the leading right singular
    vector is the optimal direction.
    """
    if len(pts) < 2:
        return Line(pts[0].copy(), pts[0].copy()), 0.0
    mu = pts.mean(axis=0)
    centered = pts - mu
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    d = vh[0]
    t = centered @ d
    p0 = mu + t.min() * d
    p1 = mu + t.max() * d
    perp = centered - np.outer(t, d)
    rms = float(np.sqrt((perp ** 2).sum(axis=1).mean()))
    return Line(p0, p1), rms


def fit_circle_kasa(
    pts: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], float]:
    """Algebraic circle fit (Kasa). Fast initializer for geometric refine.

    Minimizes Σ (xᵢ² + yᵢ² + a xᵢ + b yᵢ + c)² with
    a = -2 cx, b = -2 cy, c = cx² + cy² - r². Linear in (a, b, c).
    """
    x, y = pts[:, 0], pts[:, 1]
    M = np.column_stack([x, y, np.ones_like(x)])
    rhs = -(x * x + y * y)
    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    a, b, c = sol
    cx, cy = -a / 2.0, -b / 2.0
    r_sq = cx * cx + cy * cy - c
    if r_sq < 0:
        # Numerical degeneracy (collinear points); return huge radius.
        return np.array([cx, cy]), 1e9
    return np.array([cx, cy]), float(np.sqrt(r_sq))


def fit_circle_geometric(
    pts: NDArray[np.float64],
    c0: NDArray[np.float64],
    r0: float,
) -> Tuple[NDArray[np.float64], float, float]:
    """Geometric circle refine. Returns (center, radius, rms).

    Minimizes Σ (‖pᵢ - c‖ - r)² which is the true geometric residual.
    Kasa's bias (underestimates radius for short arcs — see Chernov) is
    fixed up by this stage.
    """

    def residuals(params):
        return np.linalg.norm(pts - params[:2], axis=1) - params[2]

    x0 = np.array([c0[0], c0[1], r0])
    try:
        sol = least_squares(residuals, x0, method="lm", max_nfev=50)
        c = sol.x[:2]
        r = float(sol.x[2])
    except Exception:
        c, r = c0, r0
    res = np.linalg.norm(pts - c, axis=1) - r
    rms = float(np.sqrt((res ** 2).mean()))
    return c, r, rms


def fit_circle(
    pts: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], float, float]:
    """Combined Kasa + geometric refine. Returns (center, radius, rms)."""
    if len(pts) < 3:
        # Degenerate; can't fit a circle to <3 points.
        return np.array([0.0, 0.0]), 0.0, float("inf")
    c0, r0 = fit_circle_kasa(pts)
    return fit_circle_geometric(pts, c0, r0)


def fit_arc(pts: NDArray[np.float64]) -> Tuple[Optional[Arc], float]:
    """Fit a circle to the points, then build an arc using the first and
    last points as endpoints. Returns (Arc, rms) or (None, inf) if fit
    fails.

    Sweep direction is disambiguated by looking at the midpoint sample's
    side of the chord. Sweep magnitude assumes the MINOR arc; if the
    sample falls on the wrong side we flip to the major arc.
    """
    if len(pts) < 3:
        return None, float("inf")
    c, r, rms = fit_circle(pts)
    if not np.isfinite(r) or r < _EPS:
        return None, float("inf")

    p0 = pts[0]
    p1 = pts[-1]
    chord = p1 - p0
    L = float(np.linalg.norm(chord))
    if L < _EPS:
        # Endpoints coincide — treat as full circle, not an arc.
        return None, float("inf")

    # Determine sweep direction. Cross product chord × (mid - p0):
    # if positive (in y-down) and the geometric center is on the same
    # side as the mid sample, sweep is positive (CCW in image coords).
    mid_sample_idx = len(pts) // 2
    mid_sample = pts[mid_sample_idx]
    side_mid = chord[0] * (mid_sample[1] - p0[1]) - chord[1] * (mid_sample[0] - p0[0])
    side_center = chord[0] * (c[1] - p0[1]) - chord[1] * (c[0] - p0[0])

    # half-chord angle: sin(θ) = (L/2) / r. Clamp for numerical safety.
    half_chord_over_r = min(1.0, max(-1.0, L / (2.0 * r)))
    half_sweep_minor = float(np.arcsin(half_chord_over_r))

    # If the mid sample is on the OPPOSITE side of the chord from the
    # center, the arc passes around the far side -> major arc.
    is_major = (side_mid * side_center > 0.0)
    sweep_mag = (
        2.0 * (np.pi - half_sweep_minor) if is_major else 2.0 * half_sweep_minor
    )

    # An arc with |sweep| > ~240° means the polyline traces a large
    # majority of a circle but doesn't quite close. Fitting such a
    # polyline as a single Arc produces a near-degenerate bulge
    # magnitude (the endpoints are close together, so a tiny
    # perturbation flips the arc through 180°). The joint solver
    # tends to "fix" these by collapsing the major arc into a much
    # smaller minor arc with a different radius, which radically
    # changes the shape (the tree in treecar1 went from a round
    # outline to a narrow teardrop because a -264° arc fitting the
    # tree's bottom collapsed to a -72° arc). Better to let the
    # polyline either be treated as a Circle (handled in
    # fit_polyline's near-closed branch) or be split by chain
    # subdivision into multiple smaller arcs that ARE stable.
    if sweep_mag > math.radians(240.0):
        return None, float("inf")

    # Direction: traversal goes p0 -> mid -> p1. The arc bulges toward
    # the side where ``mid_sample`` lies. With our convention
    # ``normal = (-chord_y, chord_x)``, the Arc's ``center()`` puts the
    # center on the POSITIVE-cross side of the chord, so the arc
    # midpoint is on the NEGATIVE-cross side. So:
    #   side_mid < 0  → arc midpoint on negative-cross side → bulge > 0
    #   side_mid > 0  → arc midpoint on positive-cross side → bulge < 0
    sweep_sign = -1.0 if side_mid > 0 else 1.0
    sweep = sweep_sign * sweep_mag
    bulge = float(np.tan(sweep / 4.0))

    arc = Arc(p0.copy(), p1.copy(), bulge)
    return arc, rms


def fit_full_circle(
    pts: NDArray[np.float64],
) -> Tuple[Optional[Circle], float]:
    """Fit a Circle (not Arc) to a closed polyline. Returns (Circle, rms)."""
    if len(pts) < 3:
        return None, float("inf")
    c, r, rms = fit_circle(pts)
    if not np.isfinite(r) or r < _EPS:
        return None, float("inf")
    return Circle(c, r), rms


# ---------------------------------------------------------------------------
# Chain subdivision


@dataclass(frozen=True)
class ChainPiece:
    start_idx: int  # inclusive
    end_idx: int  # exclusive
    primitive: Primitive


class FitFailure(RuntimeError):
    pass


def _segment_extent(pts: NDArray[np.float64]) -> float:
    """A length scale to normalize tolerances by. Use bounding-box
    diagonal because it's stable for both straight and curvy strokes —
    arc length over-rewards wiggly fits.
    """
    if len(pts) < 2:
        return 1.0
    span = pts.max(axis=0) - pts.min(axis=0)
    diag = float(np.linalg.norm(span))
    return max(diag, 1.0)


def fit_single_primitive(
    pts: NDArray[np.float64],
    line_tol_abs: float,
    arc_tol_abs: float,
) -> Tuple[Optional[Primitive], float]:
    """Try line and arc; return whichever fits within tolerance with the
    smaller SSE. Returns ``(primitive, sse)`` or ``(None, inf)``.

    Tolerances are absolute pixel RMS thresholds.
    """
    n = len(pts)
    if n < 2:
        return None, float("inf")

    line, line_rms = fit_line(pts)
    line_ok = line_rms < line_tol_abs
    line_sse = (line_rms ** 2) * n

    if n < 3:
        if line_ok:
            return line, line_sse
        return None, float("inf")

    arc, arc_rms = fit_arc(pts)
    arc_ok = arc is not None and arc_rms < arc_tol_abs
    arc_sse = (arc_rms ** 2) * n if arc is not None else float("inf")

    # Prefer line when both work and line SSE isn't much worse — fewer
    # parameters, simpler downstream routing, doesn't suffer from bulge
    # numerics. The "1.5x" gives arcs a fair shot when the curve is real.
    if line_ok and (not arc_ok or line_sse <= 1.5 * arc_sse):
        return line, line_sse
    if arc_ok:
        return arc, arc_sse
    if line_ok:
        return line, line_sse
    return None, float("inf")


def fit_segment_topdown(
    pts: NDArray[np.float64],
    line_tol_abs: float,
    arc_tol_abs: float,
    min_len: int = 5,
    max_depth: int = 12,
) -> List[ChainPiece]:
    """Recursive split-and-fit.

    Fit one primitive to the whole window. If it fits within tolerance,
    accept. Otherwise split at the index of maximum residual and
    recurse on each half. O(N log N) typical.

    This is the plan's recommended starting point — it tends to put
    splits at semantically meaningful places (the worst-fit point
    *is* the corner) and avoids DP's pathological ties between
    near-equivalent splits.
    """
    pieces: List[ChainPiece] = []

    def recurse(lo: int, hi: int, depth: int) -> None:
        if hi - lo < min_len or depth >= max_depth:
            # Forced terminal — fit whatever single primitive we can.
            prim, _ = fit_single_primitive(
                pts[lo:hi], line_tol_abs * 4, arc_tol_abs * 4
            )
            if prim is None:
                # Last-ditch: straight line through endpoints.
                prim = Line(pts[lo].copy(), pts[hi - 1].copy())
            pieces.append(ChainPiece(lo, hi, prim))
            return

        prim, _ = fit_single_primitive(pts[lo:hi], line_tol_abs, arc_tol_abs)
        if prim is not None:
            pieces.append(ChainPiece(lo, hi, prim))
            return

        # Find split point: index of maximum residual from the best of
        # line / arc fits even though neither was good enough.
        line, _ = fit_line(pts[lo:hi])
        line_res = np.abs(line.perpendicular_distance(pts[lo:hi]))
        if hi - lo >= 3:
            c, r, _ = fit_circle(pts[lo:hi])
            arc_res = np.abs(np.linalg.norm(pts[lo:hi] - c, axis=1) - r)
            res = np.minimum(line_res, arc_res)
        else:
            res = line_res

        # Split at the worst-fit interior index; clamp to keep both
        # children at least ``min_len`` long.
        worst = int(np.argmax(res))
        worst += lo
        worst = max(lo + min_len, min(hi - min_len, worst))
        if worst <= lo or worst >= hi:
            # Couldn't find a valid split; accept the whole window as-is.
            prim = line
            pieces.append(ChainPiece(lo, hi, prim))
            return

        recurse(lo, worst + 1, depth + 1)  # include the split point in both
        recurse(worst, hi, depth + 1)

    recurse(0, len(pts), 0)
    # Stitch: sort + ensure indices form a contiguous chain
    pieces.sort(key=lambda p: p.start_idx)
    return pieces


def fit_segment_dp(
    pts: NDArray[np.float64],
    line_tol_abs: float,
    arc_tol_abs: float,
    lam: float,
    min_len: int = 5,
    max_window: Optional[int] = None,
) -> List[ChainPiece]:
    """Global DP chain subdivision.

    ``cost(i, j)`` = best single-primitive SSE for ``pts[i:j]``,
    or ``+inf`` if neither line nor arc fits within tolerance.
    ``best(j)`` = min total cost to fit ``pts[0:j]``.

    Recurrence: ``best(j) = min over i < j of [best(i) + cost(i,j) + lam]``.

    ``lam`` is the MDL penalty per primitive — larger ``lam`` → fewer,
    looser primitives. Start around ``2 * line_tol_abs²`` and tune.

    ``max_window`` caps the maximum (j - i) considered, which makes the
    inner loop O(N · max_window) instead of O(N²). For most real
    drawings a window of 200-500 source points is plenty.
    """
    N = len(pts)
    if N < min_len:
        prim, _ = fit_single_primitive(pts, line_tol_abs * 4, arc_tol_abs * 4)
        if prim is None:
            prim = Line(pts[0].copy(), pts[-1].copy())
        return [ChainPiece(0, N, prim)]

    cache: dict = {}

    def get_cost(i: int, j: int) -> Tuple[float, Optional[Primitive]]:
        if (i, j) in cache:
            return cache[(i, j)]
        if j - i < min_len:
            cache[(i, j)] = (float("inf"), None)
            return cache[(i, j)]
        prim, sse = fit_single_primitive(pts[i:j], line_tol_abs, arc_tol_abs)
        cache[(i, j)] = (sse, prim)
        return cache[(i, j)]

    best = [0.0] + [float("inf")] * N
    split = [-1] * (N + 1)

    if max_window is None:
        max_window = N

    for j in range(min_len, N + 1):
        i_lo = max(0, j - max_window)
        for i in range(i_lo, j - min_len + 1):
            if not np.isfinite(best[i]):
                continue
            sse, _ = get_cost(i, j)
            if not np.isfinite(sse):
                continue
            total = best[i] + sse + lam
            if total < best[j]:
                best[j] = total
                split[j] = i

    # Reconstruct the chain by backtracking from N.
    chain: List[ChainPiece] = []
    j = N
    while j > 0:
        i = split[j]
        if i < 0:
            # No valid split found — usually means the tolerance is too
            # tight for this stroke. Fall back to top-down on the
            # unfit suffix.
            fallback = fit_segment_topdown(
                pts[:j], line_tol_abs, arc_tol_abs, min_len
            )
            chain = fallback + chain
            return chain
        _, prim = get_cost(i, j)
        if prim is None:
            raise FitFailure(
                f"DP picked split ({i}, {j}) with no fitted primitive"
            )
        chain.append(ChainPiece(i, j, prim))
        j = i
    chain.reverse()
    return chain


def find_closed_subloop(
    pts: NDArray[np.float64],
    min_size: int = 100,
    closure_threshold_rel: float = 0.06,
    min_extent_rel: float = 0.20,
    max_circle_rms_rel: float = 0.06,
) -> Optional[Tuple[int, int]]:
    """Find a near-closed sub-loop within an open polyline that
    ALSO fits a single Circle well.

    Looks for (i, j) with ``j - i >= min_size`` such that:
      1. ``||pts[j] - pts[i]||`` < ``closure_threshold_rel * polyline_extent``
         (the endpoints meet)
      2. ``extent(pts[i:j+1])`` >= ``min_extent_rel * polyline_extent``
         (the loop has meaningful spatial size; rules out tiny
         self-crossings)
      3. The sub-loop fits a circle with RMS below
         ``max_circle_rms_rel * loop_extent`` (the loop is actually
         circular — rules out V-shapes or U-shapes whose endpoints
         happen to be close but whose interior path isn't a circle)

    Returns ``(i, j)`` for the LARGEST qualifying sub-loop, or
    ``None``. The largest is preferred because outer/wider loops
    are usually the intended shape (a wheel rim, not a small
    embedded swirl).

    Example: bikelove's right-wheel polyline poly[14] traces the
    rim for ~650 indices and then continues into the bottom
    squiggle. Detecting [0, 650] as a circular sub-loop lets the
    rim become a Circle while the squiggle gets fit separately.
    """
    n = len(pts)
    if n < 2 * min_size:
        return None
    extent = _segment_extent(pts)
    if extent < 1.0:
        return None
    closure_thresh = closure_threshold_rel * extent
    extent_thresh = min_extent_rel * extent

    # Coarse search on a downsampled grid.
    stride = max(1, n // 200)
    candidates: List[Tuple[int, int, int]] = []  # (size, i, j)
    for i in range(0, n - min_size, stride):
        for j in range(n - 1, i + min_size, -stride):
            d = float(np.linalg.norm(pts[j] - pts[i]))
            if d < closure_thresh:
                sub = pts[i:j + 1]
                sub_ext = _segment_extent(sub)
                if sub_ext >= extent_thresh:
                    candidates.append((j - i, i, j))
                break  # take largest j for this i

    if not candidates:
        return None

    # Try the largest candidates first; require a clean circle fit
    # AND a reasonable aspect ratio (a true wheel/sun is roughly
    # circular, not stretched).
    candidates.sort(reverse=True)
    for size, i, j in candidates:
        sub = pts[i:j + 1]
        circle, rms = fit_full_circle(sub)
        if circle is None:
            continue
        loop_ext = _segment_extent(sub)
        if rms >= max_circle_rms_rel * loop_ext:
            continue
        xs = sub[:, 0]
        ys = sub[:, 1]
        bbox_x = float(xs.max() - xs.min())
        bbox_y = float(ys.max() - ys.min())
        if min(bbox_x, bbox_y) < _EPS:
            continue
        aspect = max(bbox_x, bbox_y) / min(bbox_x, bbox_y)
        if aspect >= 1.25:
            continue
        # Refine: shift i and j by +/-stride to find the exact
        # closest pair.
        i0, j0 = i, j
        best_d = float(np.linalg.norm(pts[j0] - pts[i0]))
        refined_i, refined_j = i0, j0
        for di in range(-stride, stride + 1):
            ii = i0 + di
            if ii < 0 or ii >= n:
                continue
            for dj in range(-stride, stride + 1):
                jj = j0 + dj
                if jj < 0 or jj >= n or jj - ii < min_size:
                    continue
                d = float(np.linalg.norm(pts[jj] - pts[ii]))
                if d < best_d:
                    best_d = d
                    refined_i, refined_j = ii, jj
        return (refined_i, refined_j)
    return None


def fit_polyline(
    pts: NDArray[np.float64],
    line_tol: float = 0.005,
    arc_tol: float = 0.01,
    lam_rel: float = 4.0,
    min_len: int = 5,
    closed_tol: float = 1.5,
    use_dp: bool = True,
    max_window: Optional[int] = 256,
    circle_rms_rel: float = 0.06,
) -> List[ChainPiece]:
    """Top-level: fit a polyline into a chain of primitives.

    * If the polyline is closed and fits well as a single circle,
      shortcut to a one-piece chain. The "well" tolerance here is
      ``circle_rms_rel * extent`` — much looser than ``arc_tol``
      because hand-drawn closed loops (wheels, balloons, hearts)
      are usually meant to read as a single circle even when they
      wobble by several percent of the polyline's extent. Trying to
      preserve the wobble by chain-subdividing produces 10-piece
      fragmentations that look much worse than a clean circle.
    * Otherwise, try ``fit_single_primitive`` on the whole stroke
      first — most segments are single primitives once chromosomes
      and crossings have been removed upstream.
    * Otherwise, run chain subdivision (DP by default, top-down as
      fallback).

    ``line_tol`` and ``arc_tol`` are RELATIVE to the bounding-box
    diagonal, so the same parameters work across drawings of
    different scale. ``lam_rel`` is a multiplier on
    ``(line_tol * extent)²`` to set the per-primitive MDL penalty.
    """
    n = len(pts)
    if n < 2:
        return []
    extent = _segment_extent(pts)
    line_tol_abs = line_tol * extent
    arc_tol_abs = arc_tol * extent
    lam = lam_rel * (line_tol_abs ** 2)
    circle_rms_abs = circle_rms_rel * extent

    # Closed-loop fast path: try fitting as a single circle. Use the
    # near-closed detector (gap small relative to arc length OR
    # absolutely small) so hand-drawn wheels with a 10-30 px gap also
    # get caught — fit_arc would otherwise produce a near-360° arc
    # which is much worse visually than a Circle.
    #
    # BUT: only use the Circle shortcut if the polyline is genuinely
    # corner-free. A cat-head outline that includes ears is closed AND
    # the rms-of-circle-fit is moderate (the ears are short relative
    # to the head circumference), so the RMS gate alone would happily
    # erase the ears. Counting corners catches this — ear tips show up
    # as 80°+ tangent-direction changes that a true circle never has.
    corners = find_corners(pts)
    if is_near_closed_polyline(pts) and not corners:
        circle, rms = fit_full_circle(pts)
        if circle is not None and rms < circle_rms_abs:
            # Additionally check the polyline's aspect ratio. A closed
            # polyline that traces an oval (e.g., the catcar cat-head,
            # 444x339 pixels = aspect 1.31) can still fit a circle with
            # low rms (3.5% in that case, below the 6% gate), but
            # rendering it as a perfect circle is visually wrong — the
            # circle's radius is the AVERAGE of the oval's axes, so the
            # circle extends past the polyline in the shorter dimension.
            # In catcar this pushes the cat-head circle down through the
            # roof of the car (a clear visual overlap). Reject the
            # Circle shortcut when the bounding-box aspect ratio is too
            # far from 1, and let chain subdivision fit the oval with
            # multiple arcs instead.
            xs = pts[:, 0]
            ys = pts[:, 1]
            bbox_x = float(xs.max() - xs.min())
            bbox_y = float(ys.max() - ys.min())
            if min(bbox_x, bbox_y) > _EPS:
                aspect = max(bbox_x, bbox_y) / min(bbox_x, bbox_y)
            else:
                aspect = 1.0
            if aspect < 1.25:
                return [ChainPiece(0, n, circle)]
        # Otherwise fall through; the DP will handle rounded rectangles.

    # If the polyline HAS corners (ear apexes, heart cusps), split
    # explicitly at each corner index before running chain
    # subdivision. The chain subdivider only minimizes residuals — it
    # doesn't know about corners. A corner spans 2-3 indices in the
    # polyline and is essentially unfit-able by any single primitive,
    # so without an explicit split, top-down/DP both end up producing
    # 10+ tiny pieces around the corner trying to thread the needle.
    # Splitting first lets each side of the corner be fit cleanly as
    # a single primitive.
    #
    # ALSO: look for a near-closed sub-loop within the polyline (e.g.,
    # the bikelove right wheel rim is the first ~650 indices of a
    # 994-pt polyline that continues into the bottom squiggle). If a
    # sub-loop exists, split at its start and end indices so the loop
    # portion is processed as its own near-closed sub-polyline (which
    # will then hit the closed-circle shortcut).
    subloop = find_closed_subloop(pts) if not is_near_closed_polyline(pts) else None
    splits: List[int] = list(corners)
    if subloop is not None:
        splits.extend(subloop)
        splits = sorted(set(splits))
    if splits:
        pieces: List[ChainPiece] = []
        split_idxs = [0] + splits + [n]
        # Remove duplicates while preserving order
        split_idxs = sorted(set(split_idxs))
        for lo, hi in zip(split_idxs[:-1], split_idxs[1:]):
            sub = pts[lo:hi + 1] if hi < n else pts[lo:]
            if len(sub) < 2:
                continue
            sub_chain = fit_polyline(
                sub, line_tol=line_tol, arc_tol=arc_tol,
                lam_rel=lam_rel, min_len=min_len, closed_tol=closed_tol,
                use_dp=use_dp, max_window=max_window,
                circle_rms_rel=circle_rms_rel,
            )
            for cp in sub_chain:
                pieces.append(
                    ChainPiece(cp.start_idx + lo, cp.end_idx + lo, cp.primitive)
                )
        return pieces

    # Corner-free, non-closed polyline: try fitting it as a single Arc
    # with a relaxed tolerance, similar to the closed-loop circle
    # shortcut. The threshold is intentionally TIGHTER than the
    # closed-loop circle shortcut because most non-circle polylines
    # (heart halves, bike-frame contours, leaf outlines) fit a single
    # arc within 5-10% rms but aren't really arcs — they have
    # systematic curvature variation that a single arc averages away.
    # If we accept those as one arc we lose the heart cusps,
    # rectangle corners, leaf points, etc. Only accept the shortcut
    # when the fit is so close to a real arc that chain subdivision
    # couldn't do meaningfully better.
    #
    # Critically, the threshold is BOTH 4% of extent AND capped at an
    # absolute pixel ceiling. A pure relative threshold relaxes
    # linearly with extent, so a 700-pixel polyline gets a 28-px
    # tolerance — that's enough to swallow significant shape detail
    # (heartman's body sub-segments were 706 pts with arc-fit rms of
    # 23 px, just under 28 px, so the whole body collapsed to one
    # arc with 23 px of accumulated deviation). The absolute cap of
    # ~12 px keeps the shortcut "this is essentially noise on a clean
    # arc" for polylines of any length.
    arc_rms_abs = min(0.04 * extent, 12.0)
    if not corners:
        arc, arc_rms = fit_arc(pts)
        if (arc is not None and arc_rms < arc_rms_abs and
                arc.chord() > line_tol_abs * 2):
            return [ChainPiece(0, n, arc)]

    # Single-primitive shortcut (tight tolerance, line OR arc).
    single, _ = fit_single_primitive(pts, line_tol_abs, arc_tol_abs)
    if single is not None:
        return [ChainPiece(0, n, single)]

    # Chain subdivision.
    if use_dp:
        try:
            chain = fit_segment_dp(
                pts, line_tol_abs, arc_tol_abs, lam,
                min_len=min_len, max_window=max_window,
            )
        except FitFailure:
            chain = fit_segment_topdown(pts, line_tol_abs, arc_tol_abs, min_len)
    else:
        chain = fit_segment_topdown(pts, line_tol_abs, arc_tol_abs, min_len)

    return chain
