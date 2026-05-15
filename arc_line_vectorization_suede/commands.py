"""Type definitions for the drawing-robot pipeline.

The output JSON shape mirrors the user's TypeScript discriminated union:

    type DrawingCommand =
        | { kind: "line", distance: number, penDown: boolean }
        | { kind: "spin", degrees: number }
        | { kind: "arc",  radius: number, degrees: number }

Conventions:
- Coordinates are image-space pixel coordinates (x right, y down).
- Angles are measured in this frame: positive = counter-clockwise in the
  (x-right, y-down) frame, which is clockwise when rendered to a screen.
  This is consistent with SVG, so the renderer round-trips correctly.
- For arcs, `radius` is always positive; the sign of `degrees` indicates
  direction (positive = curve to the left of current heading).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, TypedDict, Union

import numpy as np


# ---- Output JSON command types (match the TS discriminated union) -----------

class LineCommand(TypedDict):
    kind: Literal["line"]
    distance: float
    penDown: bool


class SpinCommand(TypedDict):
    kind: Literal["spin"]
    degrees: float


class ArcCommand(TypedDict):
    kind: Literal["arc"]
    radius: float
    degrees: float


DrawingCommand = Union[LineCommand, SpinCommand, ArcCommand]


# ---- Internal geometric primitives (used during fitting) --------------------

@dataclass
class LinePrimitive:
    start: np.ndarray  # shape (2,)
    end: np.ndarray    # shape (2,)


@dataclass
class ArcPrimitive:
    center: np.ndarray  # shape (2,)
    radius: float
    start: np.ndarray   # shape (2,) — point on the circle
    end: np.ndarray     # shape (2,) — point on the circle
    ccw: bool           # direction of traversal from start to end


Primitive = Union[LinePrimitive, ArcPrimitive]
Stroke = List[Primitive]  # connected chain of primitives sharing endpoints
