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


def find_inflections(
    pts: NDArray[np.float64],
    smooth_window: int = 4,
    sign_threshold: float = 0.03,
    min_sustain: int = 3,
    min_separation: int = 12,
) -> List[int]:
    """Return polyline indices at which the curvature reverses sign in
    a sustained way — the stroke smoothly switches from turning one
    direction to turning the other (a classic S-curve).

    These are NOT corners. The polyline is tangent-continuous at the
    inflection (no kink), but no single Line or Arc can represent the
    span on both sides, so chain subdivision has to split there. The
    existing ``find_corners`` only fires on tangent-direction
    discontinuities (≥50° turn), so smooth inflections slip through —
    the topdown recursion is then left to find the inflection via
    residual-argmax, which doesn't reliably pick the right index
    (angel poly 2's wing-bottom: a 58-point S-curve where residual
    argmax stayed near the boundary, the recursion nibbled 4 pts
    per level, hit max_depth, and the forced-terminal fell back to a
    chord-line that visibly cut through the angel's face).

    Algorithm: smooth the tangent direction, compute signed cross
    products of consecutive smoothed tangents (which is signed turn
    rate), and report an inflection when ``min_sustain`` samples of
    one sign appear immediately before ``min_sustain`` samples of the
    other sign (with magnitude above ``sign_threshold`` to ignore
    noise). Non-max suppression: merge inflections closer than
    ``min_separation``.

    Indices returned are positions in the input ``pts`` array.
    """
    n = len(pts)
    if n < 2 * (smooth_window + min_sustain) + 2:
        return []
    diffs = np.diff(pts, axis=0)
    if len(diffs) < smooth_window + 1:
        return []
    smoothed = np.empty((len(diffs) - smooth_window + 1, 2), dtype=np.float64)
    for k in range(len(smoothed)):
        v = diffs[k:k + smooth_window].sum(axis=0)
        nrm = float(np.linalg.norm(v))
        smoothed[k] = (0.0, 0.0) if nrm < _EPS else v / nrm
    if len(smoothed) < 2:
        return []
    cross = (
        smoothed[:-1, 0] * smoothed[1:, 1]
        - smoothed[:-1, 1] * smoothed[1:, 0]
    )

    # Walk the cross array looking for a stretch of "all positive
    # above threshold" immediately followed by "all negative above
    # threshold" (or vice versa).
    inflections: List[int] = []
    last_inflection_pos = -min_separation
    half_offset = smooth_window // 2 + 1  # map cross-index back to pts-index
    nc = len(cross)
    i = min_sustain
    while i < nc - min_sustain:
        left = cross[i - min_sustain:i]
        right = cross[i:i + min_sustain]
        if (
            np.all(left > sign_threshold) and np.all(right < -sign_threshold)
        ) or (
            np.all(left < -sign_threshold) and np.all(right > sign_threshold)
        ):
            pos = i + half_offset
            if pos - last_inflection_pos >= min_separation:
                inflections.append(pos)
                last_inflection_pos = pos
            i += min_separation
        else:
            i += 1
    return inflections


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


_MAX_RADIUS_FACTOR = 8.0

# Minimum arc sagitta (max perpendicular bow of the arc away from its
# chord) for a fitted arc to be kept as an Arc rather than degraded to a
# Line. An arc whose bow is below ~1 px is visually a straight line: the
# pen draws the same picture either way, but as an Arc it carries a
# radius/sweep the firmware estimator and the joint solver both treat as
# real curvature. ``routing.py`` already degrades sub-0.5 px arcs at
# emit time; applying the (slightly stricter) test here in ``fit_arc``
# means the primitive is a Line from the start, so the solver,
# beautification and routing all see a consistent shape. The threshold
# is an absolute pixel value on purpose — "sub-pixel bow" is a
# scale-invariant notion of "indistinguishable from a line".
_MIN_ARC_SAGITTA_PX = 1.0

# Absolute pixel ceiling on the RMS tolerance of ``fit_polyline``'s
# "accept the whole corner-free polyline as one Arc" shortcut. On the
# order of a stroke width: large enough that genuine pen tremor on a
# clean arc still shortcuts, small enough that a deliberately wavy edge
# (cheese-block scallops) is rejected and sent to chain subdivision so
# the waviness is preserved. See the shortcut site in ``fit_polyline``.
_SINGLE_ARC_SHORTCUT_RMS_CAP = 6.0

# Looser companion ceiling. Between the strict cap and this value the
# arc fit is "borderline": the shortcut is taken only if the polyline
# is too SPARSE to subdivide reliably (see ``_SUBDIVISION_MIN_POINTS``).
_SINGLE_ARC_SHORTCUT_RMS_LOOSE_CAP = 12.0

# Minimum point count for a borderline-rms polyline to be sent to chain
# subdivision rather than kept as one arc. Below this, subdivision
# would split too few points among the resulting pieces, leaving each
# piece's fit (and the joint solve) under-constrained. Densely sampled
# polylines above this threshold subdivide reliably.
_SUBDIVISION_MIN_POINTS = 40


def _has_curvature_reversal(
    pts: NDArray[np.float64],
    smooth_window: int = 4,
    min_turns_each_sign: int = 2,
    sign_threshold: float = 0.05,
) -> bool:
    """Detect whether a polyline turns in BOTH directions (an S-curve or
    zigzag), which means a single Line or Arc would mis-represent it.

    Computes a smoothed tangent at each point, then signs of consecutive
    smoothed-tangent cross products. If at least ``min_turns_each_sign``
    samples turn each way (after thresholding small noise via
    ``sign_threshold``), it's a reversal. Used inside ``fuse_chain`` to
    refuse fusing across inflections that would otherwise average out.
    """
    n = len(pts)
    if n < 2 * smooth_window + 2:
        return False
    diffs = np.diff(pts, axis=0)
    nd = len(diffs)
    if nd < smooth_window + 1:
        return False
    # Smooth the tangent by summing windows of consecutive segments,
    # then normalize. This filters out per-pixel jitter.
    smoothed = np.empty((nd - smooth_window + 1, 2), dtype=np.float64)
    for k in range(nd - smooth_window + 1):
        v = diffs[k:k + smooth_window].sum(axis=0)
        nrm = float(np.linalg.norm(v))
        if nrm < _EPS:
            smoothed[k] = np.array([0.0, 0.0])
        else:
            smoothed[k] = v / nrm
    if len(smoothed) < 2:
        return False
    # Cross product of consecutive smoothed tangents: sign indicates
    # turn direction. Positive = CCW in image coords (=CW on screen).
    cross = (
        smoothed[:-1, 0] * smoothed[1:, 1]
        - smoothed[:-1, 1] * smoothed[1:, 0]
    )
    pos = int(np.sum(cross > sign_threshold))
    neg = int(np.sum(cross < -sign_threshold))
    return pos >= min_turns_each_sign and neg >= min_turns_each_sign


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
    # Reject pathologically large radii. A "real" arc spans at least
    # some non-trivial fraction of its circle, so its radius is bounded
    # by ``radius <= _MAX_RADIUS_FACTOR * extent``. Beyond that, the fit
    # is a near-collinear point set that the algebraic solver "rescued"
    # by placing the center thousands of pixels away — the resulting
    # arc is visually indistinguishable from a line, but the firmware
    # time estimator and the downstream joint solver both treat it as
    # a real arc with a huge wheelbase swing.
    if r > _MAX_RADIUS_FACTOR * _segment_extent(pts):
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

    # Sagitta gate: reject arcs whose perpendicular bow is sub-pixel.
    # sagitta = r * (1 - cos(sweep / 2)). Such an "arc" is visually a
    # straight line; returning None lets ``fit_single_primitive`` /
    # ``fuse_chain`` pick the Line instead, which is cheaper for the
    # solver and avoids feeding the firmware estimator a spurious
    # radius. (A genuine gentle arc — e.g. 10 deg over a 150 px chord —
    # has several px of sagitta and is unaffected.)
    sagitta = r * (1.0 - math.cos(0.5 * sweep_mag))
    if sagitta < _MIN_ARC_SAGITTA_PX:
        return None, float("inf")

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
    # Reject pathological fits where the algebraic solver placed the
    # circle center far outside the points themselves — see fit_arc for
    # the same guard. The classic trigger is a tiny near-collinear
    # patch (e.g. a 4-point horizontal stub) where any Circle is a
    # perfect fit, including ones with the center thousands of pixels
    # away. Without this gate, the alien example produced two Circle
    # primitives with r=40,000 from a 6x1 px patch.
    if r > _MAX_RADIUS_FACTOR * _segment_extent(pts):
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

    # Prefer Arc when both fit AND the arc has visually meaningful
    # curvature. Without this, the chain subdivider would emit a Line
    # for any stroke whose sagitta happens to be below line_tol — even
    # if the underlying curve sweeps tens of degrees. Each subsequent
    # output line+spin pair then represents geometry that one arc
    # command could draw faster (and visually correctly). 10° is the
    # threshold at which sagitta/chord ≈ 0.022, i.e. about 2% of chord
    # — large enough to be visually clear, small enough not to misfire
    # on near-straight noise. See angel poly 2's wing and birdlove
    # poly 2's heart top.
    arc_min_sweep_rad = math.radians(10.0)
    if (
        arc_ok
        and arc is not None
        and abs(arc.sweep()) >= arc_min_sweep_rad
    ):
        return arc, arc_sse

    # Otherwise prefer Line when both work and line SSE isn't much
    # worse than arc SSE — fewer parameters, simpler downstream
    # routing, no bulge numerics. The "1.5x" gives arcs a fair shot
    # when the curve is real but below the meaningful-sweep gate.
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
        # BOTH children must have a strictly smaller window than this
        # one, otherwise we'd recurse on the same range until ``depth``
        # hits ``max_depth`` and the forced-terminal branch fires —
        # producing a chain of degenerate L(0) pieces, one per depth
        # level (angel poly 10 produced 10+ identical 1-point pieces
        # this way). The first child is ``recurse(lo, worst + 1)``, so
        # ``worst + 1 < hi`` keeps it strictly smaller. ``worst > lo``
        # keeps the second child strictly smaller.
        if worst <= lo or worst + 1 >= hi:
            # Couldn't find a valid split; accept the whole window
            # as-is. Use ``fit_single_primitive`` with relaxed
            # tolerance so the same Arc-vs-Line preference fires
            # here as elsewhere — defaulting to Line means windows
            # of real curvature get linified at exactly the points
            # the algorithm has the least information (the angel
            # wing's bottom edge ended up here, at 19.8° of sweep,
            # but the bail-out picked Line so a slanted edge cut
            # straight through the angel's face).
            prim, _ = fit_single_primitive(
                pts[lo:hi], line_tol_abs * 4, arc_tol_abs * 4
            )
            if prim is None:
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


def _validate_subloop_candidate(
    pts: NDArray[np.float64],
    i: int,
    j: int,
    stride: int,
    min_size: int,
    max_circle_rms_rel: float,
) -> Optional[Tuple[int, int]]:
    """Check one ``(i, j)`` sub-loop candidate: it must fit a clean
    circle, be roughly circular in aspect, and is then refined to the
    exact closest endpoint pair. Returns the refined ``(i, j)`` or
    ``None`` if the candidate fails any gate.
    """
    n = len(pts)
    sub = pts[i:j + 1]
    circle, rms = fit_full_circle(sub)
    if circle is None:
        return None
    loop_ext = _segment_extent(sub)
    if rms >= max_circle_rms_rel * loop_ext:
        return None
    xs = sub[:, 0]
    ys = sub[:, 1]
    bbox_x = float(xs.max() - xs.min())
    bbox_y = float(ys.max() - ys.min())
    if min(bbox_x, bbox_y) < _EPS:
        return None
    aspect = max(bbox_x, bbox_y) / min(bbox_x, bbox_y)
    if aspect >= 1.25:
        return None
    # Refine: shift i and j by +/-stride to find the exact closest pair.
    best_d = float(np.linalg.norm(pts[j] - pts[i]))
    refined_i, refined_j = i, j
    for di in range(-stride, stride + 1):
        ii = i + di
        if ii < 0 or ii >= n:
            continue
        for dj in range(-stride, stride + 1):
            jj = j + dj
            if jj < 0 or jj >= n or jj - ii < min_size:
                continue
            d = float(np.linalg.norm(pts[jj] - pts[ii]))
            if d < best_d:
                best_d = d
                refined_i, refined_j = ii, jj
    return (refined_i, refined_j)


def find_closed_subloops(
    pts: NDArray[np.float64],
    min_size: int = 100,
    closure_threshold_rel: float = 0.06,
    min_extent_rel: float = 0.20,
    max_circle_rms_rel: float = 0.06,
) -> List[Tuple[int, int]]:
    """Find ALL near-closed circular sub-loops within an open polyline.

    Generalizes ``find_closed_subloop``: a single hand-drawn stroke can
    thread more than one loop (a figure-8, a stack of bubbles drawn
    without lifting the pen, a flower drawn as several petals off one
    contour). Each qualifying sub-loop is split out so it can hit the
    closed-circle shortcut independently.

    Each returned ``(i, j)`` satisfies the same three gates as the
    single-loop version (endpoints meet, meaningful spatial extent,
    fits a circle with low RMS and near-unit aspect). Returned loops
    are pairwise NON-OVERLAPPING — when candidates overlap, the larger
    is kept (outer/wider loops are usually the intended shape). The
    list is sorted by start index so callers can use it directly as
    split points.
    """
    n = len(pts)
    if n < 2 * min_size:
        return []
    extent = _segment_extent(pts)
    if extent < 1.0:
        return []
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
        return []

    # Largest first; accept a candidate only if it passes the circle /
    # aspect gates AND does not overlap an already-accepted loop.
    candidates.sort(reverse=True)
    accepted: List[Tuple[int, int]] = []
    for _size, i, j in candidates:
        if any(not (j < ai or i > aj) for ai, aj in accepted):
            continue  # overlaps a kept loop
        refined = _validate_subloop_candidate(
            pts, i, j, stride, min_size, max_circle_rms_rel
        )
        if refined is None:
            continue
        ri, rj = refined
        # Re-check overlap after refinement.
        if any(not (rj < ai or ri > aj) for ai, aj in accepted):
            continue
        accepted.append((ri, rj))

    accepted.sort()
    return accepted


def find_closed_subloop(
    pts: NDArray[np.float64],
    min_size: int = 100,
    closure_threshold_rel: float = 0.06,
    min_extent_rel: float = 0.20,
    max_circle_rms_rel: float = 0.06,
) -> Optional[Tuple[int, int]]:
    """Find the single LARGEST near-closed circular sub-loop within an
    open polyline (thin wrapper over ``find_closed_subloops``, kept for
    callers that only want one).

    See ``find_closed_subloops`` for the gate criteria. The largest is
    preferred because outer/wider loops are usually the intended shape
    (a wheel rim, not a small embedded swirl).

    Example: bikelove's right-wheel polyline poly[14] traces the rim
    for ~650 indices and then continues into the bottom squiggle.
    Detecting [0, 650] as a circular sub-loop lets the rim become a
    Circle while the squiggle gets fit separately.
    """
    loops = find_closed_subloops(
        pts,
        min_size=min_size,
        closure_threshold_rel=closure_threshold_rel,
        min_extent_rel=min_extent_rel,
        max_circle_rms_rel=max_circle_rms_rel,
    )
    if not loops:
        return None
    return max(loops, key=lambda ij: ij[1] - ij[0])


def _line_collapse(
    pieces: List[ChainPiece],
    source_points: List[NDArray[np.float64]],
    line_tol_abs: float,
) -> Tuple[List[ChainPiece], List[NDArray[np.float64]]]:
    """Collapse runs of consecutive Lines (and small Arcs) whose union
    of source points fits a single Line within ``line_tol_abs``.

    Why this exists separately from the main greedy fusion: the main
    pass uses ``_has_curvature_reversal`` to refuse fusing across S-
    curves, which is necessary when the candidate is an Arc (the fit
    averages opposing curvatures to a near-zero sweep that wrong-
    renders the geometry). For a Line candidate that protection is
    moot — ``line_rms`` already rejects S-curves (the centerline is
    far from both halves) and accepts noise around a straight stretch
    (rms stays small). Dropping the guard lets the line-collapse find
    cases like angel poly 13's `[L(8), L(10), L(7)]` tail and angel
    poly 14's `[L(10), L(7), L(4), L(8)]` tail — runs that the chain
    subdivider broke up but which are visually one straight stroke.

    Arcs are also eligible. A small Arc bracketed by Lines whose union
    still fits a single Line (e.g., 5° sweep in the middle of a 100 px
    near-straight stretch) gets folded in. A large Arc fails the line
    fit naturally because its sagitta dominates rms. ``Circle`` is
    always skipped (a closed loop can never project onto a line).

    Single-piece "windows" are NOT modified — replacing a standalone
    Arc with its chord-line would silently throw away legitimate
    curvature. Collapse only happens when we successfully extend past
    the first piece.
    """
    n = len(pieces)
    if n < 2:
        return pieces, source_points

    # Pre-compute the sweep sign of each piece (0 for lines / tiny arcs).
    # Used to detect when extending the window would span an S-curve —
    # an arc of one sign followed by an arc of the opposite sign means
    # the underlying stroke turns both ways. ``line_rms`` happily
    # averages opposing curvatures (the deviations cancel), so it can
    # not flag the S-curve on its own; without this guard,
    # birdlove's heart-top S-curves got flattened into a polyline.
    arc_sign_threshold_rad = math.radians(8.0)
    sweep_signs: List[int] = []
    for c in pieces:
        p = c.primitive
        if isinstance(p, Arc) and abs(p.sweep()) >= arc_sign_threshold_rad:
            sweep_signs.append(1 if p.sweep() > 0 else -1)
        else:
            sweep_signs.append(0)

    out_pieces: List[ChainPiece] = []
    out_src: List[NDArray[np.float64]] = []
    i = 0
    while i < n:
        prim_i = pieces[i].primitive
        if isinstance(prim_i, Circle):
            out_pieces.append(pieces[i])
            out_src.append(source_points[i])
            i += 1
            continue

        best_j = i + 1
        best_line: Optional[Line] = None
        best_pts: Optional[NDArray[np.float64]] = None

        # Sign of the first signed-arc we've encountered in the window
        # (0 means we haven't seen one yet). Once set, any new piece
        # with the OPPOSITE sign breaks the window.
        seen_sign = sweep_signs[i]

        for j in range(i + 2, n + 1):
            if isinstance(pieces[j - 1].primitive, Circle):
                break
            nxt_sign = sweep_signs[j - 1]
            if nxt_sign != 0:
                if seen_sign != 0 and nxt_sign != seen_sign:
                    # Opposite-sign arcs in the same window — don't
                    # collapse to a line, that would silently
                    # straighten out an S-curve.
                    break
                seen_sign = nxt_sign
            concat = np.vstack(source_points[i:j])
            if len(concat) < 2:
                break
            cand_line, cand_rms = fit_line(concat)
            if cand_rms >= line_tol_abs:
                break
            best_j = j
            best_line = cand_line
            best_pts = concat

        if best_line is not None and best_pts is not None:
            out_pieces.append(
                ChainPiece(
                    pieces[i].start_idx,
                    pieces[best_j - 1].end_idx,
                    best_line,
                )
            )
            out_src.append(best_pts)
        else:
            out_pieces.append(pieces[i])
            out_src.append(source_points[i])
        i = best_j

    return out_pieces, out_src


def _is_degenerate(prim: Primitive) -> bool:
    """A primitive is degenerate if it represents essentially no
    geometry — a near-zero-length Line, or an Arc with near-zero sweep.
    Emitting one as a command costs an alignment spin and a
    line/arc op for no visible drawing.
    """
    if isinstance(prim, Line):
        return prim.length() < 0.5
    if isinstance(prim, Arc):
        return abs(prim.sweep()) < math.radians(0.5)
    return False


def _drop_degenerate_pieces(
    pieces: List[ChainPiece],
    source_points: List[NDArray[np.float64]],
    primitives: List[Primitive],
) -> Tuple[List[ChainPiece], List[NDArray[np.float64]], List[Primitive]]:
    """Strip degenerate primitives from a chain, folding their source
    points into the nearest live neighbor's source-point bag so the
    polyline coverage is preserved (and the subsequent greedy fusion
    can refit the neighbor against the full data).

    A degenerate at chain index ``k`` is folded into the previous live
    piece when one exists, otherwise into the next live piece. If the
    entire chain is degenerate (unlikely), an empty chain is returned;
    the caller treats that as "nothing to route".
    """
    out_pieces: List[ChainPiece] = []
    out_src: List[NDArray[np.float64]] = []
    out_prims: List[Primitive] = []
    pending_src: List[NDArray[np.float64]] = []  # for degenerates at chain start
    pending_start: int = -1

    for k, prim in enumerate(primitives):
        if _is_degenerate(prim):
            if out_pieces:
                # Fold into the previous live piece by extending its
                # end index and concatenating source points.
                last = out_pieces[-1]
                out_pieces[-1] = ChainPiece(
                    last.start_idx, pieces[k].end_idx, last.primitive
                )
                out_src[-1] = np.vstack([out_src[-1], source_points[k]])
            else:
                # No previous live piece yet — buffer for the next one.
                if pending_start < 0:
                    pending_start = pieces[k].start_idx
                pending_src.append(source_points[k])
            continue

        # Live piece: absorb any pending pre-chain degenerates.
        if pending_src:
            src = np.vstack(pending_src + [source_points[k]])
            new_piece = ChainPiece(
                pending_start, pieces[k].end_idx, prim
            )
            pending_src = []
            pending_start = -1
        else:
            src = source_points[k]
            new_piece = pieces[k]
        out_pieces.append(new_piece)
        out_src.append(src)
        out_prims.append(prim)

    # If the chain ended in degenerates with no prior live piece (and
    # nothing followed), we drop them entirely — there's no valid
    # primitive to attach them to.
    return out_pieces, out_src, out_prims


def fuse_chain(
    pieces: List[ChainPiece],
    source_points: List[NDArray[np.float64]],
    primitives: List[Primitive],
    line_tol_rel: float = 0.006,
    arc_tol_rel: float = 0.025,
    line_tol_abs_min: float = 1.0,
    arc_tol_abs_min: float = 2.5,
) -> Tuple[List[ChainPiece], List[NDArray[np.float64]]]:
    """Greedy within-chain fusion: merge runs of consecutive primitives
    whose union of source points still fits a single Line or Arc within
    tolerance.

    This catches the case where chain subdivision over-segments a gentle
    curve: each small piece individually fits a Line within ``line_tol``,
    but stitched together they're really a single Arc. The user-level
    output is then ``line spin line spin line spin …`` instead of one
    arc command — visually similar but draws much slower.

    Boundaries: this function operates within ONE chain (one
    ``FittedSegment``) at a time, so it never crosses StrokeGraph
    junctions — graph topology is preserved. Circles are skipped (they
    represent closed loops and aren't fusable with adjacent open
    primitives).

    Tolerances scale with the chain's bounding-box diagonal, with an
    absolute floor so very short chains still get meaningful tolerance.
    The defaults are ~3x looser than the chain-subdivision tolerance,
    because the hypothesis here is "these were fit as separate pieces
    only because the noise-floor per piece is below tolerance" — the
    union still has the same noise floor, so the fit RMS for the union
    is roughly the RMS of any one piece. Without some headroom over the
    per-piece tolerance, nothing ever fuses.

    Returns the new ``(pieces, source_points)`` lists. Caller is
    responsible for rebuilding the global primitive list and primitive-
    id mapping (typically via ``assign_global_ids``).
    """
    # Phase 0: swallow degenerate pieces. A "Line" with chord < 0.5 px
    # or an "Arc" with sweep < 0.5 degrees is a fit to ~1 source point
    # — it represents no real geometry but costs a spin and a
    # `{"distance": 0, "penDown": true}` command in the output stream
    # (the angel example had 16 such pieces from a recursion bug in
    # chain subdivision; even after fixing that, the joint solver and
    # other paths can still produce occasional degenerates, so we
    # filter here defensively). Each degenerate's source points are
    # appended to the previous live piece's source points so the
    # subsequent greedy fusion has the full polyline context.
    pieces, source_points, primitives = _drop_degenerate_pieces(
        pieces, source_points, primitives
    )
    n = len(pieces)
    if n <= 1:
        # Rewrap the single piece around its current primitive (which
        # may have been updated by the solver since pieces was built).
        if n == 0:
            return [], []
        return (
            [ChainPiece(pieces[0].start_idx, pieces[0].end_idx, primitives[0])],
            [source_points[0]],
        )

    # Chain-wide extent for tolerance scaling.
    all_pts = np.vstack(source_points)
    extent = _segment_extent(all_pts)
    line_tol_abs = max(line_tol_rel * extent, line_tol_abs_min)
    arc_tol_abs = max(arc_tol_rel * extent, arc_tol_abs_min)

    new_pieces: List[ChainPiece] = []
    new_src: List[NDArray[np.float64]] = []

    i = 0
    while i < n:
        # Circles aren't fusable — emit as-is.
        if isinstance(primitives[i], Circle):
            new_pieces.append(
                ChainPiece(pieces[i].start_idx, pieces[i].end_idx, primitives[i])
            )
            new_src.append(source_points[i])
            i += 1
            continue

        # Walk j outward from i+1 and keep the largest window for which
        # the union of source points still fits a single Line or Arc.
        # Including j=i+1 here also "re-fits" single pieces: the joint
        # solver can pull an Arc's radius outward to satisfy adjacent
        # constraints, leaving us with a near-line that ``fit_arc``
        # would now reject via its radius cap. Refitting forces those
        # cases to come back as Lines.
        best_j = i
        best_prim: Optional[Primitive] = None
        best_pts: Optional[NDArray[np.float64]] = None

        for j in range(i + 1, n + 1):
            # Bail when the next primitive is a Circle.
            if j > i + 1 and isinstance(primitives[j - 1], Circle):
                break

            concat = np.vstack(source_points[i:j])
            if len(concat) < 2:
                break

            # Curvature-reversal guard. If the concatenated points turn
            # in BOTH directions, this span is an S-curve / zigzag.
            # Fitting a single Line averages the wiggles flat; fitting a
            # single Arc averages the opposite-curvature halves into a
            # near-zero sweep. Either way the visual detail is lost, so
            # refuse to extend the window across the reversal. Cheap
            # checks (sign on primitives' sweeps alone) miss the
            # frequent case where chain subdivision split an S-curve
            # into short Lines that have no curvature signal.
            if j > i + 1 and _has_curvature_reversal(concat):
                break

            line, line_rms = fit_line(concat)
            arc: Optional[Arc] = None
            arc_rms = float("inf")
            if len(concat) >= 3:
                arc, arc_rms = fit_arc(concat)

            # Same Arc-vs-Line preference as ``fit_single_primitive``:
            # when both fit, an Arc with visually meaningful sweep
            # wins over a Line. This is what prevents the main
            # fusion from converting a run of small same-direction
            # arcs (birdlove heart top) into a polyline, just
            # because the polyline approximation happens to fit
            # within line_tol. Below 10° sweep the arc is too flat
            # to read as curved, so we still prefer the simpler
            # Line.
            candidate: Optional[Primitive] = None
            arc_min_sweep_rad = math.radians(10.0)
            arc_ok = (
                arc is not None
                and arc_rms < arc_tol_abs
            )
            arc_meaningful = (
                arc_ok
                and arc is not None
                and abs(arc.sweep()) >= arc_min_sweep_rad
            )
            if arc_meaningful:
                candidate = arc
            elif line_rms < line_tol_abs:
                candidate = line
            elif arc_ok:
                candidate = arc

            if candidate is None:
                break

            best_j = j
            best_prim = candidate
            best_pts = concat

        if best_prim is not None and best_pts is not None:
            start = pieces[i].start_idx
            end = pieces[best_j - 1].end_idx
            new_pieces.append(ChainPiece(start, end, best_prim))
            new_src.append(best_pts)
            i = best_j
        else:
            # Couldn't fit even the single piece (rare: very short
            # piece or pathological data). Keep the upstream primitive.
            new_pieces.append(
                ChainPiece(pieces[i].start_idx, pieces[i].end_idx, primitives[i])
            )
            new_src.append(source_points[i])
            i += 1

    # Phase 2: line-collapse. The main pass above uses the curvature-
    # reversal guard which is needed for the Arc-candidate branch but
    # over-rejects sequences of small Lines (and Lines + small Arcs)
    # whose union would happily fit a single Line. Run a second pass
    # without the guard, using ``fit_line``'s RMS as the only gate —
    # which naturally rejects S-curves (high centerline rms) while
    # accepting noise around a near-straight stroke.
    new_pieces, new_src = _line_collapse(new_pieces, new_src, line_tol_abs)

    return new_pieces, new_src


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
                # One axis is degenerate (zero spread) — the polyline is
                # effectively collinear. Treat as infinite aspect so the
                # Circle shortcut is rejected; the caller will fall through
                # to chain subdivision, which will fit a Line.
                aspect = float("inf")
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
    # ALSO: look for near-closed sub-loops within the polyline (e.g.,
    # the bikelove right wheel rim is the first ~650 indices of a
    # 994-pt polyline that continues into the bottom squiggle). A
    # single stroke can thread several loops (figure-8s, stacked
    # bubbles), so every qualifying sub-loop is split out; each loop
    # region is then processed as its own near-closed sub-polyline
    # (which hits the closed-circle shortcut).
    subloops = (
        find_closed_subloops(pts) if not is_near_closed_polyline(pts) else []
    )
    splits: List[int] = list(corners)
    for si, sj in subloops:
        splits.append(si)
        splits.append(sj)
    # Also split at smooth curvature reversals (S-curve inflections).
    # ``find_corners`` only fires on tangent-direction discontinuities
    # (kinks ≥50°); a polyline that smoothly switches from CCW to CW
    # turning has no kink and slips through. Without an explicit split,
    # the topdown recursion can't fit a single primitive across the
    # reversal (an S-curve isn't an arc), and its forced-terminal
    # branch falls back to a chord-line that visually misrepresents
    # the geometry — see angel poly 2's wing-bottom. Splitting at the
    # inflection turns the S-curve into two single-direction halves
    # that the recursive fit handles cleanly.
    inflections = find_inflections(pts)
    if inflections:
        splits.extend(inflections)
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
    # Critically, the threshold is BOTH a fraction of extent AND capped
    # at an absolute pixel ceiling. A pure relative threshold relaxes
    # linearly with extent, so a 700-pixel polyline gets a large
    # tolerance — enough to swallow significant shape detail (heartman's
    # body sub-segments were 706 pts with arc-fit rms of 23 px, so the
    # whole body collapsed to one arc with 23 px of accumulated
    # deviation). The absolute cap keeps the shortcut meaning "this is
    # essentially noise on a clean arc" for polylines of any length.
    #
    # The cap is two-tier. ``_SINGLE_ARC_SHORTCUT_RMS_CAP`` (~1 stroke
    # width) is the "definitely one clean arc" tolerance. Between that
    # and ``_SINGLE_ARC_SHORTCUT_RMS_LOOSE_CAP`` the fit is borderline,
    # and the deciding question is whether chain subdivision can be
    # *trusted* to do better. Subdivision splits the points among 2-3
    # pieces and re-fits each; that is only reliable when the polyline
    # is densely enough sampled that each resulting piece still has
    # enough points to constrain its fit and the joint solve. A sparse
    # polyline (few points spread over a large extent) subdivides into
    # under-constrained pieces that the solve pulls *off* the data —
    # this is what regressed the smile mouth (a 22-point segment).
    #
    # So: a borderline-rms polyline keeps the one-arc shortcut when it
    # is too sparse to subdivide safely (``n`` below
    # ``_SUBDIVISION_MIN_POINTS``); a densely sampled borderline
    # polyline falls through to chain subdivision, which tracks the
    # real shape detail (the cheese block's irregular edges, ~135
    # points each, fit a single arc at ~9 px rms but are visibly
    # better as a short arc chain). RMS magnitude alone cannot tell a
    # subdivide-worthy edge from clean tremor; sample density can.
    arc_rms_strict = min(0.04 * extent, _SINGLE_ARC_SHORTCUT_RMS_CAP)
    arc_rms_loose = min(0.04 * extent, _SINGLE_ARC_SHORTCUT_RMS_LOOSE_CAP)
    if not corners:
        arc, arc_rms = fit_arc(pts)
        if arc is not None and arc.chord() > line_tol_abs * 2:
            if arc_rms < arc_rms_strict:
                return [ChainPiece(0, n, arc)]
            if arc_rms < arc_rms_loose and n < _SUBDIVISION_MIN_POINTS:
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
