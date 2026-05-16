"""Variable manifest — the parameter vector layout.

We follow the plan's recommended fallback ("Replace topo-sorted manifest
with implicit constraints"): every primitive owns its own parameters,
and high-weight soft constraints enforce coincidence and on-curve
relationships. This is dramatically simpler to implement and debug
than the handle/topo-sort approach, at the cost of needing larger
weights for the coincidence terms.

Parameter layout per primitive:

* ``Line``: 4 params — (x0, y0, x1, y1)
* ``Arc``: 5 params — (x0, y0, x1, y1, bulge)
* ``Circle``: 3 params — (cx, cy, radius)

The manifest exposes:

* ``pack(prims)`` / ``unpack(x)`` — convert between primitive list and
  the flat parameter vector ``scipy.optimize.least_squares`` consumes.
* ``parameter_scales(prims)`` — per-parameter typical magnitudes, used
  to construct an ``x_scale`` array. The optimizer's ``x_scale='jac'``
  works well most of the time but fails on the rare unused parameter,
  so we keep an explicit fallback.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .primitives import Arc, Circle, Line, Primitive


# ---------------------------------------------------------------------------
# Constraint dataclasses
#
# Each constraint refers to primitives by their index in the manifest's
# ``primitives`` list. Endpoint identifiers are the string "start" or
# "end" (a Circle has neither — junctions onto a circle use OnCircle,
# which doesn't pick an endpoint on the host).

EndPoint = Tuple[int, str]  # (prim_id, "start"|"end")


@dataclass
class Coincide:
    """Two endpoints must coincide. Used at terminate-into-each-other
    junctions and at internal chain joints."""

    a: EndPoint
    b: EndPoint


@dataclass
class OnCurve:
    """An endpoint of ``terminating`` lies on the curve ``host``.

    For a Line host, this is the perpendicular distance to the line.
    For an Arc/Circle host, this is the radial distance to the circle.
    """

    terminating: EndPoint
    host: int


@dataclass
class G1Smooth:
    """Two primitives meeting at an endpoint should have parallel
    tangents at that endpoint.

    ``alpha_a`` ∈ {0.0, 1.0} picks which end of primitive ``a`` the
    constraint applies to; same for ``b``.
    """

    a: int
    alpha_a: float
    b: int
    alpha_b: float


@dataclass
class Parallel:
    a: int
    b: int


@dataclass
class Perpendicular:
    a: int
    b: int


@dataclass
class EqualRadius:
    a: int
    b: int


@dataclass
class Concentric:
    a: int
    b: int


@dataclass
class SoftConstraints:
    """Bundle of all soft constraints applied at solve time. Mutable;
    beautification mutates the bundle and re-solves."""

    coincide: List[Coincide]
    on_curve: List[OnCurve]
    g1: List[G1Smooth]
    parallel: List[Parallel]
    perpendicular: List[Perpendicular]
    equal_radius: List[EqualRadius]
    concentric: List[Concentric]

    @classmethod
    def empty(cls) -> "SoftConstraints":
        return cls([], [], [], [], [], [], [])


# ---------------------------------------------------------------------------
# Parameter packing


def _param_count(prim: Primitive) -> int:
    if isinstance(prim, Line):
        return 4
    if isinstance(prim, Arc):
        return 5
    if isinstance(prim, Circle):
        return 3
    raise TypeError(f"unknown primitive {type(prim)!r}")


def parameter_count(prims: List[Primitive]) -> int:
    return sum(_param_count(p) for p in prims)


def _offsets(prims: List[Primitive]) -> List[int]:
    offs: List[int] = []
    cur = 0
    for p in prims:
        offs.append(cur)
        cur += _param_count(p)
    return offs


def pack(prims: List[Primitive]) -> NDArray[np.float64]:
    """Flatten primitives into a single parameter vector."""
    out = np.empty(parameter_count(prims), dtype=np.float64)
    cur = 0
    for p in prims:
        if isinstance(p, Line):
            out[cur:cur + 2] = p.p0
            out[cur + 2:cur + 4] = p.p1
            cur += 4
        elif isinstance(p, Arc):
            out[cur:cur + 2] = p.p0
            out[cur + 2:cur + 4] = p.p1
            out[cur + 4] = p.bulge
            cur += 5
        elif isinstance(p, Circle):
            out[cur:cur + 2] = p.center
            out[cur + 2] = p.radius
            cur += 3
        else:
            raise TypeError(type(p))
    return out


def unpack(
    x: NDArray[np.float64],
    template: List[Primitive],
) -> List[Primitive]:
    """Reconstruct primitives from a parameter vector. ``template`` is
    the original primitive list — used to pick which class to build at
    each slot. Returns a fresh list of new primitives.
    """
    out: List[Primitive] = []
    cur = 0
    for p in template:
        if isinstance(p, Line):
            out.append(Line(x[cur:cur + 2].copy(), x[cur + 2:cur + 4].copy()))
            cur += 4
        elif isinstance(p, Arc):
            out.append(
                Arc(x[cur:cur + 2].copy(), x[cur + 2:cur + 4].copy(),
                    float(x[cur + 4]))
            )
            cur += 5
        elif isinstance(p, Circle):
            out.append(Circle(x[cur:cur + 2].copy(), float(x[cur + 2])))
            cur += 3
        else:
            raise TypeError(type(p))
    return out


def parameter_scales(
    prims: List[Primitive],
    pos_scale: float,
) -> NDArray[np.float64]:
    """Per-parameter typical magnitude. ``pos_scale`` is e.g. the
    bounding-box diagonal of the drawing.

    * Positions: ``pos_scale``
    * Radius: ``pos_scale`` (same units as positions)
    * Bulge: 1.0 (dimensionless; ranges roughly [-1, 1] for sweeps ≤π)
    """
    out = np.empty(parameter_count(prims), dtype=np.float64)
    cur = 0
    for p in prims:
        if isinstance(p, Line):
            out[cur:cur + 4] = pos_scale
            cur += 4
        elif isinstance(p, Arc):
            out[cur:cur + 4] = pos_scale
            out[cur + 4] = 1.0
            cur += 5
        elif isinstance(p, Circle):
            out[cur:cur + 2] = pos_scale
            out[cur + 2] = pos_scale
            cur += 3
        else:
            raise TypeError(type(p))
    return out


def endpoint_position(prim: Primitive, end: str) -> NDArray[np.float64]:
    """For Line / Arc only. For Circle, callers should use OnCurve
    constraints instead."""
    if isinstance(prim, (Line, Arc)):
        return prim.p0 if end == "start" else prim.p1
    raise TypeError(
        f"endpoint_position not defined for {type(prim).__name__}"
    )
