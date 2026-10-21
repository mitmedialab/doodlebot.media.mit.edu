"""Render a DrawingCommand list to SVG.

Kept from the retired Python vectorizer (``arc_line_vectorization_suede``):
the vectorizer itself now lives in the ``arcline_vectorize`` Rust extension,
which renders its own SVGs, but admin-edited commands still need re-rendering
server-side (see ``v2._render_commands_svg``).

We simulate the robot to recover the drawn primitives (line segments and
arcs) in order, then write an SVG with each *drawn* primitive coloured by
its index in the draw sequence (rainbow / hue ramp). Pen-up traversals are
shown as faint dashed grey lines so you can sanity-check ordering.

Command conventions (mirroring the bot's TypeScript discriminated union):

    type DrawingCommand =
        | { kind: "line", distance: number, penDown: boolean }
        | { kind: "spin", degrees: number }
        | { kind: "arc",  radius: number, degrees: number }

- Coordinates are image-space pixel coordinates (x right, y down).
- Angles are measured in this frame: positive = counter-clockwise in the
  (x-right, y-down) frame, which is clockwise when rendered to a screen.
  This is consistent with SVG, so the renderer round-trips correctly.
- For arcs, `radius` is always positive; the sign of `degrees` indicates
  direction (positive = curve to the left of current heading).
"""

from __future__ import annotations
import math
from typing import Literal, Optional, Tuple, Sequence, TypedDict, Union

import numpy as np


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


def _hsl(hue_deg: float, sat: float = 80.0, light: float = 50.0) -> str:
    return f"hsl({hue_deg:.1f}, {sat:.0f}%, {light:.0f}%)"


def _simulate(
    commands: Sequence[DrawingCommand],
    start_pos: Tuple[float, float],
    start_heading: float,
):
    """Replay commands; collect drawn segments, pen-up segments, and bounds."""
    pos = np.array(start_pos, dtype=float)
    heading = float(start_heading)
    drawn = []  # list of dicts {kind, ...} in draw order
    pen_up = []  # list of (p0, p1) tuples
    xs, ys = [pos[0]], [pos[1]]

    def update_bounds(px, py):
        xs.append(px)
        ys.append(py)

    for cmd in commands:
        if cmd["kind"] == "spin":
            heading += math.radians(cmd["degrees"])
        elif cmd["kind"] == "line":
            new_pos = pos + cmd["distance"] * np.array(
                [math.cos(heading), math.sin(heading)]
            )
            if cmd["penDown"]:
                drawn.append(
                    {
                        "kind": "line",
                        "p0": pos.copy(),
                        "p1": new_pos.copy(),
                    }
                )
            else:
                pen_up.append((pos.copy(), new_pos.copy()))
            update_bounds(new_pos[0], new_pos[1])
            pos = new_pos
        elif cmd["kind"] == "arc":
            r = float(cmd["radius"])
            sweep = math.radians(cmd["degrees"])
            ccw = sweep > 0
            # Centre is 90deg to the left of heading for CCW, right for CW.
            normal_angle = heading + (math.pi / 2 if ccw else -math.pi / 2)
            center = pos + r * np.array(
                [math.cos(normal_angle), math.sin(normal_angle)]
            )
            start_a = math.atan2(pos[1] - center[1], pos[0] - center[0])
            end_a = start_a + sweep
            new_pos = center + r * np.array([math.cos(end_a), math.sin(end_a)])
            drawn.append(
                {
                    "kind": "arc",
                    "p0": pos.copy(),
                    "p1": new_pos.copy(),
                    "center": center.copy(),
                    "radius": r,
                    "sweep": sweep,
                }
            )
            # Sample arc for bounds
            n_samp = max(2, int(abs(sweep) * 8))
            for k in range(n_samp + 1):
                t = k / n_samp
                a = start_a + t * sweep
                update_bounds(
                    center[0] + r * math.cos(a),
                    center[1] + r * math.sin(a),
                )
            pos = new_pos
            heading += sweep
        else:
            raise ValueError(f"Unknown command kind: {cmd!r}")

    return drawn, pen_up, (min(xs), min(ys), max(xs), max(ys))


def _render_drawing_parts(
    drawn,
    pen_up,
    stroke_width,
    pen_up_stroke_width,
    show_pen_up,
    stroke=None,
    show_endpoints=True,
):
    """SVG fragments for one drawing -- pen-up dashes, drawn primitives
    colored by execution order, start dot and end ring -- in the drawing's
    native coordinate system. Caller is responsible for the outer <svg>, the
    background <rect>, and any wrapping <g transform> that places the drawing
    in the final layout.

    ``stroke`` overrides the per-primitive color: pass a CSS color (e.g.
    ``"black"``) to draw every primitive in that single color, or leave it
    ``None`` to keep the default rainbow / hue ramp by execution order.

    ``show_endpoints`` draws the start dot + end ring that mark draw order;
    set it ``False`` when the SVG represents the finished artwork the user
    sees (those markers aren't part of the drawing).
    """
    parts = []

    if show_pen_up:
        parts.append('<g stroke="#bbb" stroke-dasharray="2,2" fill="none">')
        for p0, p1 in pen_up:
            parts.append(
                f'  <line x1="{p0[0]:.2f}" y1="{p0[1]:.2f}" '
                f'x2="{p1[0]:.2f}" y2="{p1[1]:.2f}" '
                f'stroke-width="{pen_up_stroke_width}" />'
            )
        parts.append("</g>")

    n = len(drawn)
    parts.append('<g fill="none" stroke-linecap="round">')
    for i, d in enumerate(drawn):
        color = stroke if stroke is not None else _hsl(360.0 * i / max(n, 1))
        if d["kind"] == "line":
            p0, p1 = d["p0"], d["p1"]
            parts.append(
                f'  <line x1="{p0[0]:.2f}" y1="{p0[1]:.2f}" '
                f'x2="{p1[0]:.2f}" y2="{p1[1]:.2f}" '
                f'stroke="{color}" stroke-width="{stroke_width}" />'
            )
        else:  # arc
            p0, p1 = d["p0"], d["p1"]
            r = d["radius"]
            sweep = d["sweep"]
            if abs(sweep) >= 2 * math.pi - 1e-3:
                center = d["center"]
                start_a = math.atan2(p0[1] - center[1], p0[0] - center[0])
                mid_a = start_a + sweep / 2.0
                pmx = center[0] + r * math.cos(mid_a)
                pmy = center[1] + r * math.sin(mid_a)
                sweep_flag = 1 if sweep > 0 else 0
                parts.append(
                    f'  <path d="M {p0[0]:.2f} {p0[1]:.2f} '
                    f"A {r:.2f} {r:.2f} 0 0 {sweep_flag} {pmx:.2f} {pmy:.2f} "
                    f'A {r:.2f} {r:.2f} 0 0 {sweep_flag} {p1[0]:.2f} {p1[1]:.2f}" '
                    f'stroke="{color}" stroke-width="{stroke_width}" />'
                )
            else:
                large_arc = 1 if abs(sweep) > math.pi else 0
                sweep_flag = 1 if sweep > 0 else 0
                parts.append(
                    f'  <path d="M {p0[0]:.2f} {p0[1]:.2f} '
                    f"A {r:.2f} {r:.2f} 0 {large_arc} {sweep_flag} "
                    f'{p1[0]:.2f} {p1[1]:.2f}" '
                    f'stroke="{color}" stroke-width="{stroke_width}" />'
                )
    parts.append("</g>")

    if drawn and show_endpoints:
        first_p0 = drawn[0]["p0"]
        last_p1 = drawn[-1]["p1"]
        parts.append(
            f'<circle cx="{first_p0[0]:.2f}" cy="{first_p0[1]:.2f}" '
            f'r="2" fill="black" />'
        )
        parts.append(
            f'<circle cx="{last_p1[0]:.2f}" cy="{last_p1[1]:.2f}" '
            f'r="3" fill="none" stroke="black" stroke-width="1" />'
        )

    return parts


def commands_to_svg(
    commands,
    output_path: Optional[str] = None,
    start_pos=(0.0, 0.0),
    start_heading=0.0,
    stroke_width=1.5,
    pen_up_stroke_width=0.5,
    padding=8.0,
    show_pen_up=True,
    stroke: Optional[str] = None,
    show_endpoints: bool = True,
):
    """Render a command list to an SVG file. Returns the SVG string.

    ``stroke`` forces a single stroke color for every drawn primitive (e.g.
    ``"black"``); left ``None`` it keeps the default rainbow coloring by
    execution order. ``show_endpoints`` toggles the start-dot / end-ring
    draw-order markers — set ``False`` for a clean finished-artwork render.
    """
    drawn, pen_up, (minx, miny, maxx, maxy) = _simulate(
        commands, start_pos, start_heading
    )
    minx -= padding
    miny -= padding
    maxx += padding
    maxy += padding
    width = maxx - minx
    height = maxy - miny

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{minx:.2f} {miny:.2f} {width:.2f} {height:.2f}" '
        f'width="{width:.0f}" height="{height:.0f}">',
        '<rect width="100%" height="100%" fill="white" '
        f'x="{minx:.2f}" y="{miny:.2f}" />',
    ]
    parts.extend(
        _render_drawing_parts(
            drawn,
            pen_up,
            stroke_width,
            pen_up_stroke_width,
            show_pen_up,
            stroke=stroke,
            show_endpoints=show_endpoints,
        )
    )
    parts.append("</svg>")

    svg = "\n".join(parts)
    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(svg)
    return svg

