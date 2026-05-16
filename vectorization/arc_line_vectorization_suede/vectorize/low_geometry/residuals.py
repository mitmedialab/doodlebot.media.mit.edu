"""Residual assembly for the joint least-squares solve.

``scipy.optimize.least_squares`` minimizes ``½ Σ rᵢ²`` over a flat
parameter vector. We produce the residual vector as a concatenation of:

1. **Data residuals.** Each fitted primitive sees the source points that
   were assigned to it during chain subdivision. For lines the residual
   is perpendicular distance; for arcs/circles, signed radial distance.

2. **Coincidence residuals.** Two endpoints that should coincide
   contribute ``(p_a - p_b)`` (two scalars). High weight = exact in
   the limit.

3. **On-curve residuals.** An endpoint of one primitive lying on another
   contributes one scalar — perpendicular distance for a line host,
   radial distance for an arc/circle host.

4. **G1 residuals.** Cross product of unit tangents at the meeting
   endpoint. ``sin(angle)`` between them — smooth, no branch cut,
   exactly the right "small when aligned" quantity. Using ``arctan2``
   differences would introduce wrap-around discontinuities.

5. **Beautification residuals.** Parallel = cross of directions = 0.
   Perpendicular = dot of directions = 0. Equal radius =
   difference of radii. Concentric = difference of centers.

6. **Regularization.** Small log-radius pull-back to the initial fit so
   that an underdetermined arc doesn't drift to a giant near-line.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

from .manifest import (
    Coincide, Concentric, EqualRadius, G1Smooth, OnCurve, Parallel,
    Perpendicular, SoftConstraints, unpack, endpoint_position,
)
from .primitives import Arc, Circle, Line, Primitive


_EPS = 1e-9


@dataclass
class Weights:
    """Tunable weight scaling per residual category.

    Defaults reflect the protocol in the implementation recommendations:
    coincidence is essentially-hard (very large), G1 is moderate, data
    is unit, beautification is mid, regularization is small.
    """

    data: float = 1.0
    coincide: float = 200.0
    on_curve: float = 200.0
    g1: float = 10.0
    parallel: float = 5.0
    perpendicular: float = 5.0
    equal_radius: float = 2.0
    concentric: float = 5.0
    radius_reg: float = 0.05
    # Endpoint anchor: pull prim.p0 toward source_pts[0] and prim.p1
    # toward source_pts[-1]. Data residuals only constrain points to
    # lie on the underlying circle/line — they don't fix angular
    # position on a circle or position along a line. Without this
    # term, Coincide can slide endpoints along the fitted curve as
    # long as the chain still closes. Anchor weight should be small
    # relative to data (so the fit isn't dragged off the source) but
    # strong enough to lock angular position.
    endpoint_anchor: float = 5.0
    # Bulge magnitude penalty: prevents an arc with few source points
    # from running away to a near-complete circle (|bulge| → many,
    # |sweep| → 360°). With the residual = |bulge|, a typical fitted
    # bulge of 0.3 contributes ~(0.3)² to the cost, which is
    # negligible next to data residuals. But |bulge|=10 contributes
    # 100, which the optimizer feels strongly. Below |bulge|=1
    # (sweep < 180°) the penalty is essentially free.
    bulge_reg: float = 0.5


# ---------------------------------------------------------------------------
# Per-primitive endpoint and tangent helpers (numerical-stability layer)
#
# We compute these inline rather than calling primitive methods so that
# the residual function works with whatever `unpack` produced — the
# bulge value in the parameter vector may be transiently near zero
# mid-iteration even when the fit ultimately resolves to a curve.


def _endpoint(prim: Primitive, end: str) -> NDArray[np.float64]:
    if isinstance(prim, (Line, Arc)):
        return prim.p0 if end == "start" else prim.p1
    # Circle: no real endpoint; pick theta=0 by convention.
    if isinstance(prim, Circle):
        return prim.center + np.array([prim.radius, 0.0])
    raise TypeError(type(prim))


def _tangent_at(prim: Primitive, alpha: float) -> NDArray[np.float64]:
    """Unit tangent at parameter alpha ∈ [0, 1].

    Arc.tangent_at can numerically degenerate for very small bulge; in
    that case we fall back to the chord direction (the limit as
    sweep → 0). This keeps the cross-product G1 residual smooth across
    the bulge=0 transition.
    """
    if isinstance(prim, Line):
        return prim.direction()
    if isinstance(prim, Arc):
        if abs(prim.bulge) < 1e-3:
            d = prim.p1 - prim.p0
            n = float(np.linalg.norm(d))
            if n > _EPS:
                return d / n
            return np.array([1.0, 0.0])
        return prim.tangent_at(alpha)
    if isinstance(prim, Circle):
        return prim.tangent_at(alpha)
    raise TypeError(type(prim))


def _data_residual(
    prim: Primitive, pts: NDArray[np.float64]
) -> NDArray[np.float64]:
    if isinstance(prim, Line):
        return prim.perpendicular_distance(pts)
    if isinstance(prim, Arc):
        return prim.signed_distance(pts)
    if isinstance(prim, Circle):
        return prim.signed_distance(pts)
    raise TypeError(type(prim))


def _on_curve_residual(
    p: NDArray[np.float64], host: Primitive
) -> NDArray[np.float64]:
    """Single scalar: distance from point p to the host curve."""
    if isinstance(host, Line):
        return np.array([host.perpendicular_distance(p[None, :])[0]])
    if isinstance(host, Arc):
        # Use radial distance to the host's circle. (We don't constrain
        # to the arc's sweep range; OnCurve constraints come from the
        # junction graph which has already verified the point lies in
        # the arc's source-pixel range.)
        c = host.center()
        r = host.radius()
        return np.array([float(np.linalg.norm(p - c) - r)])
    if isinstance(host, Circle):
        return np.array(
            [float(np.linalg.norm(p - host.center) - host.radius)]
        )
    raise TypeError(type(host))


def _radius(prim: Primitive) -> float:
    if isinstance(prim, Circle):
        return float(prim.radius)
    if isinstance(prim, Arc):
        return float(prim.radius())
    raise TypeError(type(prim))


def _center(prim: Primitive) -> NDArray[np.float64]:
    if isinstance(prim, Circle):
        return prim.center
    if isinstance(prim, Arc):
        return prim.center()
    raise TypeError(type(prim))


# ---------------------------------------------------------------------------
# Residual assembly


def assemble_residuals(
    x: NDArray[np.float64],
    template: List[Primitive],
    source_points: Dict[int, NDArray[np.float64]],
    soft: SoftConstraints,
    weights: Weights,
    initial_radii: Dict[int, float],
) -> NDArray[np.float64]:
    """Build the full residual vector.

    Args:
        x: parameter vector to evaluate at.
        template: primitives, in manifest order, used to know which
            shape to rebuild at each slot.
        source_points: ``{prim_id: pts}`` mapping each primitive to
            the source points it should fit. Primitives without source
            points (e.g. those derived from elsewhere) can be omitted.
        soft: bundle of soft constraints.
        weights: per-category multiplicative weights.
        initial_radii: ``{prim_id: r0}`` for arcs/circles — used by
            the log-radius regularization term.
    """
    prims = unpack(x, template)
    parts: List[NDArray[np.float64]] = []

    # 1. Data residuals.
    for pid, pts in source_points.items():
        if len(pts) == 0:
            continue
        r = _data_residual(prims[pid], pts)
        parts.append(weights.data * r)

    # 1b. Endpoint anchors — pull prim.p0/p1 toward source_pts[0]/[-1].
    # Two 2-vector residuals per primitive (Circle has no endpoints
    # and is skipped). This is what gives the optimizer something to
    # work against when it would otherwise rotate an arc's endpoints
    # along the fitted circle.
    for pid, pts in source_points.items():
        if len(pts) == 0:
            continue
        prim = prims[pid]
        if isinstance(prim, (Line, Arc)):
            parts.append(weights.endpoint_anchor * (prim.p0 - pts[0]))
            parts.append(weights.endpoint_anchor * (prim.p1 - pts[-1]))

    # 2. Coincidence residuals — 2 scalars per pair.
    for c in soft.coincide:
        pa = _endpoint(prims[c.a[0]], c.a[1])
        pb = _endpoint(prims[c.b[0]], c.b[1])
        parts.append(weights.coincide * (pa - pb))

    # 3. On-curve residuals — 1 scalar per constraint.
    for c in soft.on_curve:
        p = _endpoint(prims[c.terminating[0]], c.terminating[1])
        parts.append(weights.on_curve * _on_curve_residual(p, prims[c.host]))

    # 4. G1 residuals — 1 scalar per constraint (cross product).
    for c in soft.g1:
        ta = _tangent_at(prims[c.a], c.alpha_a)
        tb = _tangent_at(prims[c.b], c.alpha_b)
        # If alpha_a = 1.0 we're at the "outgoing" end of A pointing
        # away from the join; if alpha_b = 0.0 we're at the "incoming"
        # end of B pointing into B. They should be parallel (sin = 0).
        cross = float(ta[0] * tb[1] - ta[1] * tb[0])
        parts.append(np.array([weights.g1 * cross]))

    # 5. Beautification residuals.
    for c in soft.parallel:
        a = prims[c.a]
        b = prims[c.b]
        if not isinstance(a, Line) or not isinstance(b, Line):
            continue
        da = a.direction()
        db = b.direction()
        cross = float(da[0] * db[1] - da[1] * db[0])
        parts.append(np.array([weights.parallel * cross]))

    for c in soft.perpendicular:
        a = prims[c.a]
        b = prims[c.b]
        if not isinstance(a, Line) or not isinstance(b, Line):
            continue
        da = a.direction()
        db = b.direction()
        dot = float(da @ db)
        parts.append(np.array([weights.perpendicular * dot]))

    for c in soft.equal_radius:
        ra = _radius(prims[c.a])
        rb = _radius(prims[c.b])
        if not np.isfinite(ra) or not np.isfinite(rb):
            continue
        parts.append(np.array([weights.equal_radius * (ra - rb)]))

    for c in soft.concentric:
        ca = _center(prims[c.a])
        cb = _center(prims[c.b])
        parts.append(weights.concentric * (ca - cb))

    # 6. Regularization — log-radius pull-back to the initial fit.
    # Log so that the residual scales proportionally with relative
    # radius change. Without this an arc can drift to absurd radii
    # when its source points are nearly colinear.
    for pid, r0 in initial_radii.items():
        if r0 <= _EPS:
            continue
        prim = prims[pid]
        if isinstance(prim, (Arc, Circle)):
            r = _radius(prim)
            if r <= _EPS or not np.isfinite(r):
                continue
            parts.append(np.array([weights.radius_reg * np.log(r / r0)]))

    # 7. Bulge-magnitude regularization — one residual per Arc.
    # Prevents an under-constrained arc (few or near-colinear source
    # points) from running away to |bulge| → many (sweep → 360°),
    # which the optimizer otherwise has no reason to avoid because
    # the data residuals only enforce "on circle", not "shortest arc".
    # We use residual = bulge² so the *cost* contribution is ∝ bulge⁴:
    # at |bulge| ≤ 1 (sweep ≤ 180°) the penalty is negligible (cost
    # ≤ 0.5 * w²), but at |bulge| = 5 it jumps to 0.5*w²*625 and at
    # |bulge| = 55 it dominates everything. The Jacobian
    # d(b²)/db = 2b pulls back to zero smoothly from either sign.
    # Order matters: this MUST appear last so the sparsity in
    # solve.py matches.
    for pid, prim in enumerate(prims):
        if isinstance(prim, Arc):
            b = float(prim.bulge)
            parts.append(np.array([weights.bulge_reg * b * b]))

    if not parts:
        return np.zeros(0)
    return np.concatenate(parts)
