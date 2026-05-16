"""Geometric primitives used during vectorization.

Coordinate convention: image coordinates with x right, y down.
Cross product ``a × b = a[0]*b[1] - a[1]*b[0]``. Positive cross in this
y-down frame corresponds to a clockwise rotation when rendered to a
screen (matching SVG).

Internal parameterization:

* ``Line`` — two endpoints.
* ``Arc`` — two endpoints + a scalar ``bulge = tan(sweep / 4)``. Sign of
  bulge encodes direction; bulge → 0 degenerates gracefully to a line.
  Choice of bulge keeps endpoints first-class which matches the
  endpoint-coincidence constraints we add later.
* ``Circle`` — center + radius. Has its own type because
  arc-with-coincident-endpoints is singular.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import NDArray


_EPS = 1e-9


@dataclass
class Line:
    p0: NDArray[np.float64]  # shape (2,)
    p1: NDArray[np.float64]  # shape (2,)

    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    def direction(self) -> NDArray[np.float64]:
        d = self.p1 - self.p0
        n = float(np.linalg.norm(d))
        if n < _EPS:
            return np.array([1.0, 0.0])
        return d / n

    def normal(self) -> NDArray[np.float64]:
        d = self.direction()
        return np.array([-d[1], d[0]])

    def point_at(self, t: float) -> NDArray[np.float64]:
        return (1.0 - t) * self.p0 + t * self.p1

    def tangent_at(self, t: float) -> NDArray[np.float64]:
        return self.direction()

    def perpendicular_distance(
        self, pts: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Signed perpendicular distance from each point to the line."""
        n = self.normal()
        return (pts - self.p0) @ n


@dataclass
class Arc:
    p0: NDArray[np.float64]
    p1: NDArray[np.float64]
    bulge: float

    def chord(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    def sweep(self) -> float:
        """Signed sweep angle, in radians, from p0 to p1."""
        return 4.0 * float(np.arctan(self.bulge))

    def radius(self) -> float:
        L = self.chord()
        if abs(self.bulge) < _EPS:
            return float("inf")
        return L * (1.0 + self.bulge * self.bulge) / (4.0 * abs(self.bulge))

    def center(self) -> NDArray[np.float64]:
        L = self.chord()
        if L < _EPS or abs(self.bulge) < _EPS:
            # Degenerate; return midpoint to avoid divide-by-zero.
            return 0.5 * (self.p0 + self.p1)
        mid = 0.5 * (self.p0 + self.p1)
        chord_dir = (self.p1 - self.p0) / L
        # Normal pointing 90° CCW in y-down frame (i.e. visually 90° CW
        # when rendered to screen). The signed `h` selects the side.
        normal = np.array([-chord_dir[1], chord_dir[0]])
        h = L * (1.0 - self.bulge * self.bulge) / (4.0 * self.bulge)
        return mid + h * normal

    def theta0(self) -> float:
        c = self.center()
        return float(np.arctan2(self.p0[1] - c[1], self.p0[0] - c[0]))

    def theta1(self) -> float:
        return self.theta0() + self.sweep()

    def point_at(self, alpha: float) -> NDArray[np.float64]:
        """Point on arc at parameter alpha ∈ [0, 1]."""
        c = self.center()
        r = self.radius()
        if not np.isfinite(r):
            return (1.0 - alpha) * self.p0 + alpha * self.p1
        theta = self.theta0() + alpha * self.sweep()
        return c + r * np.array([np.cos(theta), np.sin(theta)])

    def tangent_at(self, alpha: float) -> NDArray[np.float64]:
        """Unit tangent at parameter alpha ∈ [0, 1], pointing along
        traversal direction (p0 → p1).
        """
        sweep = self.sweep()
        if abs(sweep) < _EPS:
            # Effectively a line.
            d = self.p1 - self.p0
            n = float(np.linalg.norm(d))
            return d / n if n > _EPS else np.array([1.0, 0.0])
        c = self.center()
        theta = self.theta0() + alpha * sweep
        sign = 1.0 if sweep > 0 else -1.0
        return sign * np.array([-np.sin(theta), np.cos(theta)])

    def ccw(self) -> bool:
        return self.sweep() > 0

    def signed_distance(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        """For each point, ‖pt − center‖ − radius."""
        c = self.center()
        r = self.radius()
        if not np.isfinite(r):
            # Degenerate: fall back to line perpendicular distance.
            return _line_perp(self.p0, self.p1, pts)
        return np.linalg.norm(pts - c, axis=1) - r

    def is_full_circle(self) -> bool:
        return abs(self.sweep()) > 2.0 * np.pi - 1e-3


@dataclass
class Circle:
    center: NDArray[np.float64]
    radius: float

    def point_at(self, alpha: float) -> NDArray[np.float64]:
        """Point on circle, traversed CCW starting at angle 0."""
        theta = 2.0 * np.pi * alpha
        return self.center + self.radius * np.array(
            [np.cos(theta), np.sin(theta)]
        )

    def tangent_at(self, alpha: float) -> NDArray[np.float64]:
        theta = 2.0 * np.pi * alpha
        return np.array([-np.sin(theta), np.cos(theta)])

    def signed_distance(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.linalg.norm(pts - self.center, axis=1) - self.radius


Primitive = Union[Line, Arc, Circle]


# ---------------------------------------------------------------------------


def _line_perp(
    p0: NDArray[np.float64],
    p1: NDArray[np.float64],
    pts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Signed perpendicular distance — module-private helper used to keep
    ``Arc.signed_distance`` working in the degenerate (zero-bulge) case.
    """
    d = p1 - p0
    n = float(np.linalg.norm(d))
    if n < _EPS:
        return np.linalg.norm(pts - p0, axis=1)
    u = d / n
    perp = np.array([-u[1], u[0]])
    return (pts - p0) @ perp


def endpoint(prim: Primitive, end: str) -> NDArray[np.float64]:
    """Return the start or end position of a primitive.

    For ``Circle`` (full circle) the choice is arbitrary — start and end
    coincide. We return the rightmost point on the circle (theta = 0) for
    both 'start' and 'end' so routing can detect a closed-loop and emit
    a single arc-360° command.
    """
    if isinstance(prim, Line):
        return prim.p0 if end == "start" else prim.p1
    if isinstance(prim, Arc):
        return prim.p0 if end == "start" else prim.p1
    if isinstance(prim, Circle):
        return prim.point_at(0.0)
    raise TypeError(f"unknown primitive type {type(prim)!r}")


def tangent_at_end(prim: Primitive, end: str) -> NDArray[np.float64]:
    """Unit tangent at the start or end of a primitive, pointing in the
    traversal direction.
    """
    if isinstance(prim, Line):
        return prim.direction()
    if isinstance(prim, Arc):
        return prim.tangent_at(0.0 if end == "start" else 1.0)
    if isinstance(prim, Circle):
        return prim.tangent_at(0.0)
    raise TypeError(f"unknown primitive type {type(prim)!r}")
