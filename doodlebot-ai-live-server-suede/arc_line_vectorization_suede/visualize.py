"""Render a DrawingCommand list to SVG.

We simulate the robot to recover the drawn primitives (line segments and
arcs) in order, then write an SVG with each *drawn* primitive coloured by
its index in the draw sequence (rainbow / hue ramp). Pen-up traversals are
shown as faint dashed grey lines so you can sanity-check ordering.
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from .commands import DrawingCommand


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


def commands_to_svg_compare(
    commands_a: Sequence[DrawingCommand],
    commands_b: Sequence[DrawingCommand],
    output_path: Optional[str] = None,
    label_a="A",
    label_b="B",
    start_pos=(0.0, 0.0),
    start_heading=0.0,
    stroke_width=1.5,
    pen_up_stroke_width=0.5,
    padding=8.0,
    panel_gap=24.0,
    label_height=28.0,
    show_pen_up=True,
):
    """Render two command lists side by side at the same scale.

    Both panels share a unified bounding box (the union of each
    drawing's padded bbox), so a primitive at drawing-space (X, Y) in
    `commands_a` appears at exactly the same panel-relative position
    as a primitive at (X, Y) in `commands_b`. That equivalence is what
    makes "spot the difference" actually work -- if you rendered each
    panel to its own bbox, drawings of slightly different extent would
    end up at different scales and the visual diff would be muddled.
    Each panel still gets its own rainbow over its own primitives.
    """
    drawn_a, pen_up_a, bbox_a = _simulate(commands_a, start_pos, start_heading)
    drawn_b, pen_up_b, bbox_b = _simulate(commands_b, start_pos, start_heading)

    minx = min(bbox_a[0], bbox_b[0]) - padding
    miny = min(bbox_a[1], bbox_b[1]) - padding
    maxx = max(bbox_a[2], bbox_b[2]) + padding
    maxy = max(bbox_a[3], bbox_b[3]) + padding
    panel_w = maxx - minx
    panel_h = maxy - miny

    total_w = 2 * panel_w + panel_gap
    total_h = panel_h + label_height
    label_baseline = label_height * 0.7
    font_size = label_height * 0.5

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {total_w:.2f} {total_h:.2f}" '
        f'width="{total_w:.0f}" height="{total_h:.0f}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<line x1="{panel_w + panel_gap / 2:.2f}" '
        f'y1="{label_height:.2f}" '
        f'x2="{panel_w + panel_gap / 2:.2f}" '
        f'y2="{total_h:.2f}" '
        f'stroke="#e0e0e0" stroke-width="0.5" />',
        f'<text x="{panel_w / 2:.2f}" y="{label_baseline:.2f}" '
        f'text-anchor="middle" font-family="sans-serif" '
        f'font-size="{font_size:.2f}" fill="#333">{label_a}</text>',
        f'<text x="{panel_w + panel_gap + panel_w / 2:.2f}" '
        f'y="{label_baseline:.2f}" '
        f'text-anchor="middle" font-family="sans-serif" '
        f'font-size="{font_size:.2f}" fill="#333">{label_b}</text>',
    ]

    parts.append(f'<g transform="translate({-minx:.2f}, {label_height - miny:.2f})">')
    parts.extend(
        _render_drawing_parts(
            drawn_a,
            pen_up_a,
            stroke_width,
            pen_up_stroke_width,
            show_pen_up,
        )
    )
    parts.append("</g>")

    parts.append(
        f'<g transform="translate({panel_w + panel_gap - minx:.2f}, '
        f'{label_height - miny:.2f})">'
    )
    parts.extend(
        _render_drawing_parts(
            drawn_b,
            pen_up_b,
            stroke_width,
            pen_up_stroke_width,
            show_pen_up,
        )
    )
    parts.append("</g>")

    parts.append("</svg>")

    svg = "\n".join(parts)
    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(svg)
    return svg


def commands_to_svg_compare_n(
    panels: Sequence[Tuple[Sequence[DrawingCommand], str]],
    output_path: Optional[str] = None,
    start_pos: Tuple[float, float] = (0.0, 0.0),
    start_heading: float = 0.0,
    stroke_width: float = 1.5,
    pen_up_stroke_width: float = 0.5,
    padding: float = 8.0,
    panel_gap: float = 24.0,
    label_height: float = 28.0,
    show_pen_up: bool = True,
):
    """Render N command lists side by side at the same scale, with a
    text label drawn above each.

    Generalises ``commands_to_svg_compare`` from two panels to an
    arbitrary number. Like the two-panel version, every panel shares a
    single unified bounding box (the union of each drawing's padded
    bbox) so a primitive at world-space (X, Y) appears at the same
    panel-relative pixel in every panel — the visual diff between
    drawings stays meaningful even when their extents differ slightly.

    Args:
        panels: list of ``(commands, label)`` pairs in left-to-right
            order. ``label`` is the text drawn centred above the
            panel — callers typically encode an identifier and the
            estimated firmware drawing time, e.g.
            ``"optimized (1062.71s)"``.
    """
    if not panels:
        raise ValueError("panels must be non-empty")

    sim = []
    for commands, label in panels:
        drawn, pen_up, bbox = _simulate(commands, start_pos, start_heading)
        sim.append((label, drawn, pen_up, bbox))

    minx = min(s[3][0] for s in sim) - padding
    miny = min(s[3][1] for s in sim) - padding
    maxx = max(s[3][2] for s in sim) + padding
    maxy = max(s[3][3] for s in sim) + padding
    panel_w = maxx - minx
    panel_h = maxy - miny

    n = len(sim)
    total_w = n * panel_w + (n - 1) * panel_gap
    total_h = panel_h + label_height
    label_baseline = label_height * 0.7
    font_size = label_height * 0.5

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {total_w:.2f} {total_h:.2f}" '
        f'width="{total_w:.0f}" height="{total_h:.0f}">',
        '<rect width="100%" height="100%" fill="white" />',
    ]
    # Vertical separators between panels.
    for i in range(1, n):
        sx = i * panel_w + (i - 1) * panel_gap + panel_gap / 2.0
        parts.append(
            f'<line x1="{sx:.2f}" y1="{label_height:.2f}" '
            f'x2="{sx:.2f}" y2="{total_h:.2f}" '
            f'stroke="#e0e0e0" stroke-width="0.5" />'
        )
    # Centred label above each panel.
    for i, (label, _drawn, _pu, _bbox) in enumerate(sim):
        cx = i * panel_w + i * panel_gap + panel_w / 2.0
        parts.append(
            f'<text x="{cx:.2f}" y="{label_baseline:.2f}" '
            f'text-anchor="middle" font-family="sans-serif" '
            f'font-size="{font_size:.2f}" fill="#333">{label}</text>'
        )
    # Each panel's drawing in its own translated group.
    for i, (_label, drawn, pen_up, _bbox) in enumerate(sim):
        x_offset = i * panel_w + i * panel_gap - minx
        parts.append(
            f'<g transform="translate({x_offset:.2f}, {label_height - miny:.2f})">'
        )
        parts.extend(
            _render_drawing_parts(
                drawn,
                pen_up,
                stroke_width,
                pen_up_stroke_width,
                show_pen_up,
            )
        )
        parts.append("</g>")
    parts.append("</svg>")

    svg = "\n".join(parts)
    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(svg)
    return svg


def _primitive_length(primitive: dict) -> float:
    if primitive["kind"] == "line":
        return float(np.linalg.norm(primitive["p1"] - primitive["p0"]))
    return float(abs(primitive["sweep"]) * primitive["radius"])


def _allocate_frames_by_length(
    lengths: List[float],
    total_frames: int,
) -> List[int]:
    if not lengths:
        return []
    if total_frames <= 0:
        total_frames = len(lengths)
    sum_len = float(sum(lengths))
    if sum_len <= 1e-9:
        return [max(1, total_frames // len(lengths))] * len(lengths)

    alloc = [max(1, int(round(total_frames * (L / sum_len)))) for L in lengths]
    cur = sum(alloc)
    if cur == total_frames:
        return alloc

    # Adjust allocation to hit the target frame count exactly.
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    if cur < total_frames:
        k = 0
        while cur < total_frames:
            alloc[order[k % len(order)]] += 1
            cur += 1
            k += 1
    else:
        k = 0
        while cur > total_frames:
            idx = order[k % len(order)]
            if alloc[idx] > 1:
                alloc[idx] -= 1
                cur -= 1
            k += 1
    return alloc


def _draw_partial_primitive(
    draw: ImageDraw.ImageDraw,
    primitive: dict,
    t: float,
    color: str,
    stroke_width: int,
    map_pt,
) -> None:
    t = float(max(0.0, min(1.0, t)))
    if primitive["kind"] == "line":
        p0 = primitive["p0"]
        p1 = primitive["p1"]
        p = p0 + t * (p1 - p0)
        draw.line([map_pt(p0), map_pt(p)], fill=color, width=stroke_width)
        return

    center = primitive["center"]
    radius = float(primitive["radius"])
    sweep = float(primitive["sweep"]) * t
    p0 = primitive["p0"]
    start_a = math.atan2(p0[1] - center[1], p0[0] - center[0])
    n_samp = max(2, int(abs(sweep) * radius * 0.8))
    pts = []
    for k in range(n_samp + 1):
        a = start_a + sweep * (k / n_samp)
        p = np.array(
            [center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)]
        )
        pts.append(map_pt(p))
    draw.line(pts, fill=color, width=stroke_width)


def commands_to_svg_gif(
    commands: Sequence[DrawingCommand],
    output_path: str,
    start_pos: Tuple[float, float] = (0.0, 0.0),
    start_heading: float = 0.0,
    stroke_width: int = 2,
    padding: float = 8.0,
    scale: float = 4.0,
    fps: int = 24,
    duration_s: Optional[float] = None,
    units_per_second: float = 60.0,
    max_total_frames: int = 240,
    max_pixels_per_frame: int = 400_000,
    show_pen_up: bool = False,
    pen_up_stroke_width: int = 1,
) -> str:
    """Create an animated GIF of the drawing process in primitive order.

    The geometry and ordering match the SVG simulator: each line/arc is
    animated progressively, then the next primitive starts.
    """
    drawn, pen_up, (minx, miny, maxx, maxy) = _simulate(
        commands, start_pos, start_heading
    )

    minx -= padding
    miny -= padding
    maxx += padding
    maxy += padding
    width_px = max(1, int(math.ceil((maxx - minx) * scale)))
    height_px = max(1, int(math.ceil((maxy - miny) * scale)))
    px_count = width_px * height_px
    if px_count > max_pixels_per_frame > 0:
        shrink = math.sqrt(max_pixels_per_frame / float(px_count))
        scale *= shrink
        width_px = max(1, int(math.ceil((maxx - minx) * scale)))
        height_px = max(1, int(math.ceil((maxy - miny) * scale)))

    def map_pt(p: np.ndarray) -> Tuple[float, float]:
        return ((float(p[0]) - minx) * scale, (float(p[1]) - miny) * scale)

    lengths = [_primitive_length(d) for d in drawn]
    total_length = float(sum(lengths))
    if duration_s is None:
        duration_s = max(1.0, total_length / max(1e-6, units_per_second))
    total_frames = max(1, int(round(duration_s * max(1, fps))))
    if max_total_frames > 0:
        total_frames = min(total_frames, max_total_frames)
    frames_per_primitive = _allocate_frames_by_length(lengths, total_frames)

    base = Image.new("RGB", (width_px, height_px), "white")
    base_draw = ImageDraw.Draw(base)
    if show_pen_up:
        for p0, p1 in pen_up:
            base_draw.line(
                [map_pt(p0), map_pt(p1)],
                fill=(190, 190, 190),
                width=pen_up_stroke_width,
            )

    frames: List[Image.Image] = []
    n = len(drawn)
    for i, primitive in enumerate(drawn):
        hue = 360.0 * i / max(n, 1)
        color = _hsl(hue)
        n_frames = frames_per_primitive[i] if i < len(frames_per_primitive) else 1
        for k in range(1, n_frames + 1):
            frame = base.copy()
            draw = ImageDraw.Draw(frame)
            _draw_partial_primitive(
                draw,
                primitive,
                t=k / n_frames,
                color=color,
                stroke_width=stroke_width,
                map_pt=map_pt,
            )
            frames.append(
                frame.convert(
                    "P",
                    palette=Image.Palette.ADAPTIVE,
                    colors=256,
                    dither=Image.Dither.NONE,
                )
            )

        _draw_partial_primitive(
            base_draw,
            primitive,
            t=1.0,
            color=color,
            stroke_width=stroke_width,
            map_pt=map_pt,
        )

    if not frames:
        frames = [
            base.convert(
                "P",
                palette=Image.Palette.ADAPTIVE,
                colors=256,
                dither=Image.Dither.NONE,
            )
        ]

    frame_ms = max(1, int(round(1000 / max(1, fps))))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=frame_ms,
        loop=0,
    )
    return output_path


def commands_to_svg_gif_compare_n(
    panels: Sequence[Tuple[Sequence[DrawingCommand], str]],
    output_path: str,
    start_pos: Tuple[float, float] = (0.0, 0.0),
    start_heading: float = 0.0,
    stroke_width: int = 2,
    padding: float = 8.0,
    scale: float = 4.0,
    fps: int = 24,
    duration_s: Optional[float] = None,
    units_per_second: float = 60.0,
    max_total_frames: int = 240,
    max_pixels_per_frame: int = 400_000,
    show_pen_up: bool = True,
    pen_up_stroke_width: int = 1,
    panel_gap_px: int = 12,
    label_height_px: int = 32,
) -> str:
    """Animated GIF comparing N command lists side by side.

    Animation timing: each panel advances proportionally to the length
    drawn so far (same heuristic as ``commands_to_svg_gif``). The total
    animation duration is set by the panel with the longest total
    drawn length, so faster panels finish first — visually you can see
    which sequence draws more efficiently. Panels that finish early
    just hold on their final frame for the rest of the animation.

    Each panel's image shares the SAME world-space bounding box (union
    of all panel bboxes + padding) so a primitive at (X, Y) sits at
    the same pixel position in every panel; the visual diff stays
    meaningful even when the panels' extents differ.

    Args:
        panels: list of ``(commands, label)`` in left-to-right order.
            ``label`` is centred above the panel.
    """
    if not panels:
        raise ValueError("panels must be non-empty")

    panel_data = []
    for commands, label in panels:
        drawn, pen_up, bbox = _simulate(commands, start_pos, start_heading)
        lengths = [_primitive_length(d) for d in drawn]
        total_length = float(sum(lengths))
        panel_data.append((label, drawn, pen_up, bbox, lengths, total_length))

    minx = min(p[3][0] for p in panel_data) - padding
    miny = min(p[3][1] for p in panel_data) - padding
    maxx = max(p[3][2] for p in panel_data) + padding
    maxy = max(p[3][3] for p in panel_data) + padding
    panel_w_units = max(1e-6, maxx - minx)
    panel_h_units = max(1e-6, maxy - miny)
    n_panels = len(panel_data)

    panel_w_px = max(1, int(math.ceil(panel_w_units * scale)))
    panel_h_px = max(1, int(math.ceil(panel_h_units * scale)))
    total_w_px = n_panels * panel_w_px + (n_panels - 1) * panel_gap_px
    total_h_px = panel_h_px + label_height_px

    # Pixel budget per frame — if we're over, shrink scale to fit.
    if max_pixels_per_frame > 0 and total_w_px * total_h_px > max_pixels_per_frame:
        shrink = math.sqrt(
            max_pixels_per_frame / float(total_w_px * total_h_px)
        )
        scale *= shrink
        panel_w_px = max(1, int(math.ceil(panel_w_units * scale)))
        panel_h_px = max(1, int(math.ceil(panel_h_units * scale)))
        total_w_px = n_panels * panel_w_px + (n_panels - 1) * panel_gap_px
        total_h_px = panel_h_px + label_height_px

    def map_pt(p) -> Tuple[float, float]:
        return ((float(p[0]) - minx) * scale, (float(p[1]) - miny) * scale)

    # Pick the panel with the longest total draw to set the absolute
    # frame budget. Every panel gets a frame budget proportional to
    # its own length, so shorter panels finish before the GIF ends.
    max_total_length = max(p[5] for p in panel_data)
    if duration_s is None:
        duration_s = max(1.0, max_total_length / max(1e-6, units_per_second))
    total_frames = max(1, int(round(duration_s * max(1, fps))))
    if max_total_frames > 0:
        total_frames = min(total_frames, max_total_frames)

    # Per-panel: build base (pen-up dashes) and produce a stream of
    # per-frame images. Pad with the final base for any panel that
    # finishes before ``total_frames``.
    per_panel_frame_imgs: List[List[Image.Image]] = []
    for label, drawn, pen_up, _bbox, lengths, total_length in panel_data:
        base = Image.new("RGB", (panel_w_px, panel_h_px), "white")
        base_draw = ImageDraw.Draw(base)
        if show_pen_up:
            for p0, p1 in pen_up:
                base_draw.line(
                    [map_pt(p0), map_pt(p1)],
                    fill=(190, 190, 190),
                    width=pen_up_stroke_width,
                )

        frames_for_panel = max(
            1,
            int(round(total_frames * (total_length / max(1e-6, max_total_length)))),
        )
        per_prim = _allocate_frames_by_length(lengths, frames_for_panel)
        n_drawn = len(drawn)
        panel_frames: List[Image.Image] = []
        for prim_idx, primitive in enumerate(drawn):
            n_frames_p = per_prim[prim_idx] if prim_idx < len(per_prim) else 1
            hue = 360.0 * prim_idx / max(n_drawn, 1)
            color = _hsl(hue)
            for k in range(1, n_frames_p + 1):
                frame = base.copy()
                d = ImageDraw.Draw(frame)
                _draw_partial_primitive(
                    d,
                    primitive,
                    t=k / n_frames_p,
                    color=color,
                    stroke_width=stroke_width,
                    map_pt=map_pt,
                )
                panel_frames.append(frame)
            _draw_partial_primitive(
                base_draw,
                primitive,
                t=1.0,
                color=color,
                stroke_width=stroke_width,
                map_pt=map_pt,
            )
        # Hold on final base for any frames remaining.
        while len(panel_frames) < total_frames:
            panel_frames.append(base.copy())
        per_panel_frame_imgs.append(panel_frames[:total_frames])

    # Font for labels. Reuse the truetype-or-bitmap-fallback dance
    # from the overlay renderer.
    from PIL import ImageFont
    font = None
    for path in (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            font = ImageFont.truetype(path, max(8, int(label_height_px * 0.45)))
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    combined_frames: List[Image.Image] = []
    for f_i in range(total_frames):
        frame = Image.new("RGB", (total_w_px, total_h_px), "white")
        fdraw = ImageDraw.Draw(frame)
        for i, panel_imgs in enumerate(per_panel_frame_imgs):
            x_offset = i * (panel_w_px + panel_gap_px)
            frame.paste(panel_imgs[f_i], (x_offset, label_height_px))
            if i > 0:
                sx = x_offset - panel_gap_px // 2
                fdraw.line(
                    [(sx, label_height_px), (sx, total_h_px)],
                    fill=(220, 220, 220),
                    width=1,
                )
            label = panel_data[i][0]
            tb = font.getbbox(label)
            text_w = tb[2] - tb[0]
            text_h = tb[3] - tb[1]
            cx = x_offset + panel_w_px // 2 - text_w // 2
            ty = max(0, (label_height_px - text_h) // 2 - 2)
            fdraw.text((cx, ty), label, fill=(40, 40, 40), font=font)
        combined_frames.append(
            frame.convert(
                "P",
                palette=Image.Palette.ADAPTIVE,
                colors=256,
                dither=Image.Dither.NONE,
            )
        )

    if not combined_frames:
        combined_frames = [
            Image.new("RGB", (total_w_px, total_h_px), "white").convert(
                "P", palette=Image.Palette.ADAPTIVE, colors=256, dither=Image.Dither.NONE,
            )
        ]

    frame_ms = max(1, int(round(1000 / max(1, fps))))
    combined_frames[0].save(
        output_path,
        save_all=True,
        append_images=combined_frames[1:],
        optimize=False,
        duration=frame_ms,
        loop=0,
    )
    return output_path


# ---------------------------------------------------------------------------
# Heat map: estimated firmware time per spatial cell
# ---------------------------------------------------------------------------

# Inferno-style colormap, evaluated at five keypoints and linearly
# interpolated between them. Inlined so this module stays matplotlib-free
# (matplotlib is a dev-only dependency).
_HEAT_COLORMAP_KEYS: List[Tuple[float, Tuple[int, int, int]]] = [
    (0.00, (0, 0, 4)),
    (0.25, (87, 16, 110)),
    (0.50, (188, 55, 84)),
    (0.75, (249, 142, 9)),
    (1.00, (252, 255, 164)),
]


def _heat_color(t: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, float(t)))
    keys = _HEAT_COLORMAP_KEYS
    for i in range(len(keys) - 1):
        t0, c0 = keys[i]
        t1, c1 = keys[i + 1]
        if t <= t1:
            f = 0.0 if t1 - t0 < 1e-9 else (t - t0) / (t1 - t0)
            return (
                int(round(c0[0] + f * (c1[0] - c0[0]))),
                int(round(c0[1] + f * (c1[1] - c0[1]))),
                int(round(c0[2] + f * (c1[2] - c0[2]))),
            )
    return keys[-1][1]


def _simulate_with_time(
    commands: Sequence[DrawingCommand],
    start_pos: Tuple[float, float],
    start_heading: float,
    pixels_per_inch: float,
    sample_step_px: float,
):
    """Walk ``commands`` and produce a list of ``(xy, dt, kind)`` samples,
    where each sample covers an equal slice of the parent command's
    firmware-model time and lives at the pen's spatial position during
    that slice.

    * Spins put all their time at the current pen position (the pen
      doesn't move while spinning, so the time accumulates in one cell).
    * Lines and arcs sample uniformly along the path, with at least 2
      samples and an upper bound determined by ``sample_step_px``.
    * ``kind`` is one of ``"draw"``, ``"penup"``, ``"spin"`` so callers
      can distinguish productive time from overhead.

    Also returns ``(min_x, min_y, max_x, max_y)`` covering every
    sampled position so the caller can size a grid that contains the
    full trajectory (including pen-up jumps).
    """
    # Local imports keep ``release.visualize`` matplotlib-free at import
    # time. ``optimize`` is part of the same package, so importing it
    # here is just resolving a sibling module on demand.
    from .optimize import (
        estimate_arc_time,
        estimate_line_time,
        estimate_spin_time,
        _INCHES_PER_MICROSTEP,
    )

    pos = np.array(start_pos, dtype=float)
    heading = float(start_heading)
    samples: List[Tuple[NDArrayLike, float, str]] = []
    xs = [pos[0]]
    ys = [pos[1]]

    def _bump(x: float, y: float) -> None:
        xs.append(x)
        ys.append(y)

    for cmd in commands:
        if cmd["kind"] == "spin":
            t = estimate_spin_time(cmd["degrees"])
            samples.append((pos.copy(), float(t), "spin"))
            heading += math.radians(cmd["degrees"])
        elif cmd["kind"] == "line":
            distance_px = float(cmd["distance"])
            distance_inches = distance_px / pixels_per_inch
            distance_microsteps = distance_inches / _INCHES_PER_MICROSTEP
            t = estimate_line_time(distance_microsteps)
            direction = np.array([math.cos(heading), math.sin(heading)])
            new_pos = pos + distance_px * direction
            n = max(2, int(distance_px / max(sample_step_px, 1e-6)) + 1)
            dt = t / n
            kind = "draw" if cmd["penDown"] else "penup"
            for k in range(n):
                f = (k + 0.5) / n
                xy = pos + f * distance_px * direction
                samples.append((xy, dt, kind))
                _bump(xy[0], xy[1])
            pos = new_pos
        elif cmd["kind"] == "arc":
            r = float(cmd["radius"])
            sweep = math.radians(cmd["degrees"])
            radius_inches = r / pixels_per_inch
            t = estimate_arc_time(radius_inches, cmd["degrees"])
            ccw = sweep > 0.0
            normal_angle = heading + (math.pi / 2.0 if ccw else -math.pi / 2.0)
            center = pos + r * np.array(
                [math.cos(normal_angle), math.sin(normal_angle)]
            )
            start_a = math.atan2(pos[1] - center[1], pos[0] - center[0])
            arc_len_px = abs(sweep) * r
            n = max(2, int(arc_len_px / max(sample_step_px, 1e-6)) + 1)
            dt = t / n
            for k in range(n):
                f = (k + 0.5) / n
                a = start_a + f * sweep
                xy = center + r * np.array([math.cos(a), math.sin(a)])
                samples.append((xy, dt, "draw"))
                _bump(xy[0], xy[1])
            end_a = start_a + sweep
            pos = center + r * np.array([math.cos(end_a), math.sin(end_a)])
            heading += sweep
        else:
            raise ValueError(f"Unknown command kind: {cmd!r}")

    if not samples:
        return [], (float(pos[0]), float(pos[1]), float(pos[0]), float(pos[1]))
    return samples, (min(xs), min(ys), max(xs), max(ys))


# NDArray alias kept loose because we don't import the typing helper here.
NDArrayLike = np.ndarray


def commands_to_heatmap(
    commands: Sequence[DrawingCommand],
    output_path: Optional[str] = None,
    start_pos: Tuple[float, float] = (0.0, 0.0),
    start_heading: float = 0.0,
    pixels_per_inch: float = 1.0,
    cell_size: float = 4.0,
    padding: float = 16.0,
    include_pen_up: bool = True,
    include_spin: bool = True,
    overlay_drawing: bool = True,
    overlay_color: Tuple[int, int, int, int] = (255, 255, 255, 110),
    upscale: int = 2,
    gamma: float = 0.5,
    saturation_percentile: float = 99.0,
):
    """Render a heat map of estimated firmware drawing time per spatial
    cell, returned as a PIL ``Image`` and optionally written to
    ``output_path``.

    Each command's time (computed by the same estimator that
    ``OptimizeRoute`` uses, see ``release.optimize``) is binned into a
    2D grid by its spatial trajectory:

    * **Lines / arcs** distribute their time uniformly across samples
      taken at ``sample_step = cell_size / 2`` along the path.
    * **Spins** dump all of their time at the current pen position,
      because the pen doesn't move while spinning. A run of "small
      line + spin + small line + spin" therefore lights up a single
      cell with the cumulative cost of the spins on top of the
      cumulative line draw time, which is exactly the pattern we
      want to surface.
    * **Pen-up moves** are included by default and shaded the same way
      lines are, so wasted backtracks (e.g., a 3 px reposition between
      two chain ends that should have coincided) appear as a bright
      streak between the disjoint pieces.

    Each command's time (computed by the same estimator that
    ``OptimizeRoute`` uses, see ``release.optimize``) is binned into a
    2D grid by its spatial trajectory:

    Args:
        commands: a sequence of ``DrawingCommand`` (typically
            ``LowGeometryVectorize(...).commands_consolidated`` or
            ``OptimizeRoute(...).commands``).
        output_path: optional path to save a PNG.
        start_pos / start_heading: robot's starting pose; MUST match
            the pose used when generating ``commands`` so the
            simulator replays the same trajectory.
        pixels_per_inch: passed through to the time estimator. Same
            value you'd use with ``OptimizeRoute``.
        cell_size: heat-map cell size in *drawing pixels*. Smaller =
            sharper localization, larger = more thermal smoothing.
        padding: extra drawing-pixel margin around the trajectory's
            bounding box.
        include_pen_up / include_spin: include those command kinds in
            the heat (default True; set False to focus on pen-down
            drawing only).
        overlay_drawing: superimpose a thin translucent outline of the
            pen-down strokes so it's clear what each hot region maps
            to. Spins/pen-ups deliberately don't appear in the overlay
            — they're the things we're trying to *find*.
        overlay_color: RGBA for the overlay outline.
        upscale: integer scale factor for the output image; the heat
            grid is built at ``cell_size`` resolution and nearest-
            neighbor upscaled by this factor so the cells remain
            crisp at viewing size.
        gamma: brightness curve applied after normalization. The raw
            time distribution is dominated by a small number of very
            hot cells (a spin that dumps 4 s into one pixel sits next
            to thousands of pen-down cells with 0.01 s each), so a
            linear ramp pushes most of the drawing into the darkest
            color. ``gamma = 0.5`` (sqrt) lifts the mid-tones into a
            visible range while still letting hot spots saturate.
        saturation_percentile: per-cell time at this percentile maps
            to the brightest color; anything brighter saturates. 99
            means the top 1% of cells (typically spins and overlapping
            samples at corners) sit at the colormap's top end and the
            rest of the drawing reads against the contrast.
    """
    sample_step_px = max(cell_size * 0.5, 0.5)
    samples, (minx, miny, maxx, maxy) = _simulate_with_time(
        commands,
        start_pos,
        start_heading,
        pixels_per_inch,
        sample_step_px,
    )
    minx -= padding
    miny -= padding
    maxx += padding
    maxy += padding
    width_px = max(1.0, maxx - minx)
    height_px = max(1.0, maxy - miny)
    nx = max(1, int(math.ceil(width_px / cell_size)))
    ny = max(1, int(math.ceil(height_px / cell_size)))

    grid = np.zeros((ny, nx), dtype=np.float64)
    for xy, dt, kind in samples:
        if kind == "penup" and not include_pen_up:
            continue
        if kind == "spin" and not include_spin:
            continue
        cx = int((xy[0] - minx) / cell_size)
        cy = int((xy[1] - miny) / cell_size)
        if 0 <= cx < nx and 0 <= cy < ny:
            grid[cy, cx] += dt

    # Percentile-based saturation: use the high percentile of nonzero
    # cells (rather than the absolute max) so a single spin pixel
    # doesn't waste the entire dynamic range. Anything above the
    # percentile saturates to the brightest color.
    nonzero = grid[grid > 0.0]
    if nonzero.size == 0:
        # All zero — render a blank image and bail.
        img = Image.new("RGB", (nx * upscale, ny * upscale), color=(0, 0, 4))
        if output_path is not None:
            img.save(output_path)
        return img
    vmax = float(np.percentile(nonzero, saturation_percentile))
    if vmax <= 0.0:
        vmax = float(nonzero.max())

    # Build the RGB heat image. Gamma lifts mid-tones; clip saturates
    # the top end at the chosen percentile.
    normalized = np.clip(grid / vmax, 0.0, 1.0)
    if gamma != 1.0:
        normalized = np.power(normalized, float(gamma))
    rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
    # Sample colormap on a 256-entry lookup, then index in.
    lut = np.array(
        [_heat_color(i / 255.0) for i in range(256)], dtype=np.uint8
    )
    idx = np.clip((normalized * 255.0).astype(np.int32), 0, 255)
    rgb[:] = lut[idx]

    img = Image.fromarray(rgb, mode="RGB")
    if upscale > 1:
        img = img.resize(
            (nx * upscale, ny * upscale), resample=Image.Resampling.NEAREST
        )

    if overlay_drawing:
        # Replay just the pen-down primitives at the final image's
        # pixel resolution and stroke them in a translucent color so
        # they read as a faint outline behind the heat.
        drawn, _pen_up, _ = _simulate(commands, start_pos, start_heading)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)

        def _to_image(p) -> Tuple[float, float]:
            px = (p[0] - minx) / cell_size * upscale
            py = (p[1] - miny) / cell_size * upscale
            return (float(px), float(py))

        for d in drawn:
            if d["kind"] == "line":
                odraw.line(
                    [_to_image(d["p0"]), _to_image(d["p1"])],
                    fill=overlay_color,
                    width=max(1, upscale // 2),
                )
            else:
                center = d["center"]
                r = float(d["radius"])
                sweep = float(d["sweep"])
                start_a = math.atan2(
                    d["p0"][1] - center[1], d["p0"][0] - center[0]
                )
                n_samp = max(8, int(abs(sweep) * r * 0.5))
                pts = []
                for k in range(n_samp + 1):
                    a = start_a + sweep * (k / n_samp)
                    p = np.array(
                        [
                            center[0] + r * math.cos(a),
                            center[1] + r * math.sin(a),
                        ]
                    )
                    pts.append(_to_image(p))
                odraw.line(pts, fill=overlay_color, width=max(1, upscale // 2))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    if output_path is not None:
        img.save(output_path)
    return img


# ---------------------------------------------------------------------------
# Overlay: drawn-stroke geometry on top of the source image
# ---------------------------------------------------------------------------


def _format_command_label(index_1: int, cmd: DrawingCommand) -> str:
    """One-line text label for a command. Matches the ``"1. Line (4)"``
    / ``"2. Spin (40deg)"`` shape the README/tests show.

    Pen-up moves are labelled as "Drive" (vs "Line" for pen-down) so
    transit segments are unambiguously distinguishable from drawn
    strokes even when reading the label list on its own.
    """
    kind = cmd["kind"]
    if kind == "spin":
        return f"{index_1}. Spin ({cmd['degrees']:.1f}deg)"
    if kind == "line":
        head = "Line" if cmd.get("penDown", False) else "Drive"
        return f"{index_1}. {head} ({cmd['distance']:.1f})"
    if kind == "arc":
        return (
            f"{index_1}. Arc ({cmd['degrees']:.1f}deg, "
            f"r={cmd['radius']:.1f})"
        )
    return f"{index_1}. {kind}"


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    fill: Tuple[int, int, int, int],
    width: int,
    dash_length: float = 6.0,
    gap_length: float = 4.0,
) -> None:
    """Draw a dashed line from ``p0`` to ``p1`` by stepping along the
    segment and emitting alternating dash/gap pieces. PIL's ImageDraw
    has no built-in dash pattern; rolling our own keeps the dependency
    surface unchanged.
    """
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-3:
        return
    ux = dx / length
    uy = dy / length
    step = dash_length + gap_length
    t = 0.0
    while t < length:
        end_t = min(t + dash_length, length)
        sx = x0 + ux * t
        sy = y0 + uy * t
        ex = x0 + ux * end_t
        ey = y0 + uy * end_t
        draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)
        t += step


def commands_to_overlay(
    commands: Sequence[DrawingCommand],
    source: "str | Image.Image | np.ndarray",
    output_path: Optional[str] = None,
    start_pos: Tuple[float, float] = (0.0, 0.0),
    start_heading: float = 0.0,
    stroke_color: Tuple[int, int, int, int] = (220, 40, 40, 230),
    transit_color: Tuple[int, int, int, int] = (140, 140, 140, 200),
    stroke_width: int = 2,
    transit_width: int = 1,
    source_alpha: float = 0.35,
    show_labels: bool = True,
    label_font_size: int = 14,
    label_color: Tuple[int, int, int, int] = (30, 60, 160, 255),
    leader_color: Tuple[int, int, int, int] = (120, 120, 120, 180),
    label_dot_color: Tuple[int, int, int, int] = (30, 60, 160, 255),
    label_dot_radius: int = 3,
):
    """Render the pen-down geometry of ``commands`` on top of the source
    image, so you can eyeball how faithfully the vectorization tracks
    the original strokes.

    The source image is blended toward white at ``source_alpha`` (0 =
    pure white, 1 = source unchanged) so the colored overlay reads
    clearly against the faded background. Pen-up moves are not drawn
    — the overlay is only what physically gets inked.

    When ``show_labels`` is set, each command's starting point is
    marked and labelled (``"1. Line (4)"``, ``"2. Spin (40deg)"``,
    …). Labels are stacked in a column to the right of the source
    image so they can never overlap each other or the geometry,
    and each label is connected by a thin leader line back to its
    command's start position in the image.

    Coordinate alignment: the simulator replays the command sequence
    from ``start_pos`` in the same image coordinates the source image
    uses, so as long as the same ``start_pos`` / ``start_heading``
    were passed to the vectorizer (the pipeline defaults to (0, 0) and
    0 rad), drawn primitives land at the same pixel positions they
    would in the original source — no scaling or recentering.

    Args:
        commands: the optimized command sequence (e.g.
            ``OptimizeRoute(...).commands`` or
            ``LowGeometryVectorize(...).commands_consolidated``).
        source: file path / PIL image / numpy array of the source.
        output_path: optional PNG path to write to.
        stroke_color: RGBA for the overlay strokes.
        stroke_width: pen-down line width in source-image pixels.
        source_alpha: 0–1, how much of the source remains visible. 0.35
            is enough for the source to read as a faint trace under
            the overlay without competing for attention.
        show_labels: include numbered command labels and leader lines.
        label_font_size: font size for command labels.
        label_color / leader_color / label_dot_color / label_dot_radius:
            colors and dot size for the label callouts. Defaults are
            tuned to read clearly against both the red overlay and
            the faded source.
    """
    drawn, pen_up, _bbox = _simulate(commands, start_pos, start_heading)

    if isinstance(source, str):
        src = Image.open(source).convert("RGBA")
    elif isinstance(source, np.ndarray):
        arr = source
        if arr.dtype == np.bool_:
            arr = (arr.astype(np.uint8) * 255)
        elif arr.dtype != np.uint8:
            scaled = arr.astype(np.float64)
            if scaled.max() <= 1.0 + 1e-6:
                scaled = scaled * 255.0
            arr = np.clip(scaled, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            src = Image.fromarray(arr).convert("RGBA")
        else:
            src = Image.fromarray(arr).convert("RGBA")
    elif isinstance(source, Image.Image):
        src = source.convert("RGBA")
    else:
        raise TypeError(f"unsupported source type {type(source)!r}")

    # Fade by blending toward white. ``source_alpha`` is the weight on
    # the source side; the rest is white, so the source visibly fades
    # while preserving its hue.
    source_alpha = max(0.0, min(1.0, float(source_alpha)))
    white = Image.new("RGBA", src.size, (255, 255, 255, 255))
    base = Image.blend(white, src, source_alpha)

    overlay = Image.new("RGBA", src.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def _xy(p) -> Tuple[float, float]:
        return (float(p[0]), float(p[1]))

    # Pen-up transits first, so the pen-down strokes layer cleanly
    # over them.
    for p0, p1 in pen_up:
        _draw_dashed_line(
            draw,
            _xy(p0),
            _xy(p1),
            fill=transit_color,
            width=transit_width,
        )

    for d in drawn:
        if d["kind"] == "line":
            draw.line(
                [_xy(d["p0"]), _xy(d["p1"])],
                fill=stroke_color,
                width=stroke_width,
            )
        else:  # arc
            center = d["center"]
            r = float(d["radius"])
            sweep = float(d["sweep"])
            start_a = math.atan2(
                d["p0"][1] - center[1], d["p0"][0] - center[0]
            )
            # Sample density tracks arc length so longer arcs stay
            # smooth without blowing up cost for tiny ones.
            n_samp = max(8, int(abs(sweep) * r * 0.5))
            pts = []
            for k in range(n_samp + 1):
                a = start_a + sweep * (k / n_samp)
                pts.append((
                    center[0] + r * math.cos(a),
                    center[1] + r * math.sin(a),
                ))
            draw.line(pts, fill=stroke_color, width=stroke_width)

    if not show_labels or not commands:
        # Channel-separated rendering for the clean (unlabelled) view.
        # Each input layer is encoded into one RGB channel of the
        # output image, so you can grab a single layer by reading
        # that channel — handy for downstream tooling or diffs.
        #
        # Channel layout:
        #   R = rendered strokes  (dark where the bot drew with pen down)
        #   G = source image      (dark where the source has ink)
        #   B = transit (pen-up)  (dotted where the bot drove pen-up)
        #
        # Combined-visual semantics (handy for spotting where the
        # vectorization matches or diverges from the source):
        #   white   = no layer present
        #   blue    = source AND rendered overlap (✓ faithful)
        #   magenta = source only (rendered missed this stroke)
        #   cyan    = rendered only (rendered drew something extra)
        #   yellow  = transit only
        #   black   = all three layers overlap
        src_gray = src.convert("L")
        src_w, src_h = src.size

        r_channel = Image.new("L", (src_w, src_h), 255)
        rdraw = ImageDraw.Draw(r_channel)
        for d in drawn:
            if d["kind"] == "line":
                rdraw.line(
                    [_xy(d["p0"]), _xy(d["p1"])],
                    fill=0,
                    width=stroke_width,
                )
            else:  # arc
                center = d["center"]
                r_ = float(d["radius"])
                sweep = float(d["sweep"])
                start_a = math.atan2(
                    d["p0"][1] - center[1], d["p0"][0] - center[0]
                )
                n_samp = max(8, int(abs(sweep) * r_ * 0.5))
                pts = []
                for k in range(n_samp + 1):
                    a = start_a + sweep * (k / n_samp)
                    pts.append((
                        center[0] + r_ * math.cos(a),
                        center[1] + r_ * math.sin(a),
                    ))
                rdraw.line(pts, fill=0, width=stroke_width)

        b_channel = Image.new("L", (src_w, src_h), 255)
        bdraw = ImageDraw.Draw(b_channel)
        for p0, p1 in pen_up:
            _draw_dashed_line(
                bdraw, _xy(p0), _xy(p1), fill=0, width=transit_width,
            )

        result = Image.merge("RGB", (r_channel, src_gray, b_channel))
        if output_path is not None:
            result.save(output_path)
        return result

    # ----------------------------------------------------------------
    # Labels: walk the commands a second time and capture each one's
    # starting (pos, heading) state, then place the labels by
    # force-directed repulsion so each label ends up as close as
    # possible to its anchor while not overlapping any other label.
    # A short leader line ties each label back to a dot drawn at the
    # command's exact starting point.

    # Walk all commands and label each one (including pen-up
    # transits, which appear as "Drive (...)" so they're
    # distinguishable from drawn lines).
    cmd_starts: List[Tuple[int, str, Tuple[float, float]]] = []
    pos = np.asarray(start_pos, dtype=float)
    heading = float(start_heading)
    for k, cmd in enumerate(commands):
        cmd_starts.append(
            (k, _format_command_label(k + 1, cmd), (float(pos[0]), float(pos[1])))
        )
        kind = cmd["kind"]
        if kind == "spin":
            heading += math.radians(cmd["degrees"])
        elif kind == "line":
            d = float(cmd["distance"])
            pos = pos + d * np.array([math.cos(heading), math.sin(heading)])
        elif kind == "arc":
            r = float(cmd["radius"])
            sweep = math.radians(cmd["degrees"])
            ccw = sweep > 0.0
            normal_angle = heading + (math.pi / 2.0 if ccw else -math.pi / 2.0)
            center = pos + r * np.array([math.cos(normal_angle), math.sin(normal_angle)])
            start_a = math.atan2(pos[1] - center[1], pos[0] - center[0])
            end_a = start_a + sweep
            pos = center + r * np.array([math.cos(end_a), math.sin(end_a)])
            heading += sweep

    if not cmd_starts:
        result = Image.alpha_composite(base, overlay).convert("RGB")
        if output_path is not None:
            result.save(output_path)
        return result

    # Try to load a real font so the labels look like text (the PIL
    # default bitmap font is tiny and unreadable). Fall back if the
    # truetype isn't available.
    from PIL import ImageFont
    font = None
    for path in (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            font = ImageFont.truetype(path, label_font_size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    label_sizes = np.zeros((len(cmd_starts), 2), dtype=np.float64)
    for i, (_k, text, _sp) in enumerate(cmd_starts):
        bbox = font.getbbox(text)
        label_sizes[i, 0] = (bbox[2] - bbox[0]) + 4  # small horizontal padding
        label_sizes[i, 1] = (bbox[3] - bbox[1]) + 4

    src_w, src_h = src.size
    # Extend the canvas so labels in dense areas can spill into a
    # margin around the source rather than crowd the geometry.
    label_margin = int(max(label_sizes.max(axis=0)) + 30)
    canvas_w = src_w + 2 * label_margin
    canvas_h = src_h + 2 * label_margin
    offset_x = label_margin
    offset_y = label_margin

    anchors = np.array(
        [[sp[0] + offset_x, sp[1] + offset_y] for _k, _t, sp in cmd_starts],
        dtype=np.float64,
    )

    # Greedy spiral placement with an outside-the-source preference.
    # For each label:
    #   1. Sweep spiral rings around the anchor.
    #   2. Score each non-overlapping candidate as
    #        leader_distance + INSIDE_PENALTY (if the label box still
    #        overlaps the original source image rectangle).
    #   3. Pick the lowest-scored candidate among a budget of
    #      ~30 valid candidates found. The penalty is sized so a
    #      label can drift up to ~``inside_penalty_px`` further
    #      from its anchor to escape the source image; further than
    #      that, the leader gets uncomfortably long and we accept
    #      the inside placement instead.
    n = len(cmd_starts)
    ideal_dx = 14.0
    placed_boxes: List[Tuple[float, float, float, float]] = []
    final_pos = np.zeros((n, 2), dtype=np.float64)

    def _overlaps_any(x0: float, y0: float, w: float, h: float) -> bool:
        x1 = x0 + w
        y1 = y0 + h
        for bx0, by0, bx1, by1 in placed_boxes:
            if x1 <= bx0 or x0 >= bx1 or y1 <= by0 or y0 >= by1:
                continue
            return True
        return False

    # Source-image bounds in canvas coordinates (we offset everything
    # by ``label_margin`` when building the canvas, so the original
    # image lives inside [margin, margin + src_w/h)).
    img_left = float(offset_x)
    img_top = float(offset_y)
    img_right = float(offset_x + src_w)
    img_bottom = float(offset_y + src_h)

    # How much extra leader length we'll pay to get a label out of
    # the source image. ~half the smaller image dimension is a good
    # balance: anchors near the image's center end up labelled in
    # the margin (worth the longer leader to keep the source clean),
    # but anchors deep enough that the *nearest* margin is more than
    # this many pixels away fall back to an inside placement rather
    # than dangling a giant leader.
    inside_penalty_px = max(80.0, 0.45 * min(src_w, src_h))

    def _overlaps_image(x0: float, y0: float, w: float, h: float) -> bool:
        x1 = x0 + w
        y1 = y0 + h
        return not (x1 <= img_left or x0 >= img_right or y1 <= img_top or y0 >= img_bottom)

    margin = 4.0
    spiral_angles_deg = [0, -25, 25, -55, 55, -90, 90, -120, 120, 180, -150, 150]
    radius_step = 6.0
    max_radius = float(max(src_w, src_h)) + 2 * label_margin

    for i, (_k, _text, _sp) in enumerate(cmd_starts):
        w, h = label_sizes[i]
        ax, ay = anchors[i]
        best_xy: Optional[Tuple[float, float]] = None
        best_score = float("inf")
        best_is_outside = False
        radius = 0.0
        while radius <= max_radius:
            for ang in spiral_angles_deg:
                a = math.radians(ang)
                cx = ax + ideal_dx + radius * math.cos(a)
                cy = ay + radius * math.sin(a)
                lx = cx
                ly = cy - h / 2.0
                # Clamp into canvas.
                lx = min(max(margin, lx), canvas_w - w - margin)
                ly = min(max(margin, ly), canvas_h - h - margin)
                if _overlaps_any(lx, ly, w, h):
                    continue
                box_cx = lx + w / 2.0
                box_cy = ly + h / 2.0
                leader = math.hypot(box_cx - ax, box_cy - ay)
                outside = not _overlaps_image(lx, ly, w, h)
                score = leader + (0.0 if outside else inside_penalty_px)
                if score < best_score:
                    best_score = score
                    best_xy = (lx, ly)
                    best_is_outside = outside
            # Stop once we have an outside-image candidate AND we've
            # spiralled out far enough that any further candidate is
            # guaranteed to be worse (its leader length alone now
            # exceeds the current best score).
            if best_is_outside and radius > best_score:
                break
            radius += radius_step
        if best_xy is None:
            best_xy = (ax + ideal_dx, ay - h / 2.0)
        lx, ly = best_xy
        final_pos[i] = (lx, ly)
        placed_boxes.append((lx, ly, lx + w, ly + h))

    label_pos = final_pos

    # Render order (matters):
    #   1. white canvas
    #   2. faded source pasted at the source-image area
    #   3. leader lines drawn on the canvas. They cross into the
    #      source area, so we want them DRAWN BEFORE the stroke /
    #      transit overlay so the overlay can paint over them.
    #      That keeps the dotted transit lines unobscured even when
    #      a leader runs through the same pixels.
    #   4. overlay (pen-up dotted transits + pen-down strokes)
    #      composited on top — covers leaders only where the
    #      overlay is actually opaque.
    #   5. translucent label backplates, then text + anchor dots.
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    canvas.paste(base, (offset_x, offset_y))

    cdraw = ImageDraw.Draw(canvas)
    for i in range(len(cmd_starts)):
        ax, ay = anchors[i]
        lx, ly = label_pos[i]
        lw, lh = label_sizes[i]
        leader_target = (lx, ly + lh / 2.0)
        cdraw.line([(ax, ay), leader_target], fill=leader_color, width=1)

    overlay_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    overlay_canvas.paste(overlay, (offset_x, offset_y))
    canvas = Image.alpha_composite(canvas, overlay_canvas)

    for i, (_k, text, _sp) in enumerate(cmd_starts):
        lx, ly = label_pos[i]
        lw, lh = label_sizes[i]
        # Translucent white plate so the label reads cleanly when it
        # ends up on top of the source or the red overlay.
        plate = Image.new(
            "RGBA", (int(lw), int(lh)), (255, 255, 255, 215)
        )
        canvas.alpha_composite(plate, dest=(int(lx), int(ly)))

    cdraw = ImageDraw.Draw(canvas)
    for i, (_k, text, _sp) in enumerate(cmd_starts):
        ax, ay = anchors[i]
        lx, ly = label_pos[i]
        cdraw.ellipse(
            [
                ax - label_dot_radius, ay - label_dot_radius,
                ax + label_dot_radius, ay + label_dot_radius,
            ],
            fill=label_dot_color,
        )
        cdraw.text((lx + 2, ly + 2), text, fill=label_color, font=font)

    result = canvas.convert("RGB")
    if output_path is not None:
        result.save(output_path)
    return result


# ---------------------------------------------------------------------------
# Labeling debug visualizations
#
# These verify the per-pixel/per-segment back-pointers that the
# skeletonize, segment, and vectorize stages now carry. Each picks a
# palette with one colour per label and renders every pixel in the
# colour of its label. The palette uses golden-ratio hue spacing
# (the same trick visualize/segment.py already uses for segment
# colours) so adjacent labels — which are renumbered every time —
# come out with maximally distinct hues. That way you can eyeball
# the partition at a junction and see immediately whether two
# adjacent regions were collapsed into one or split correctly.
# ---------------------------------------------------------------------------


def _golden_ratio_palette(
    n: int,
    *,
    sat: float = 0.85,
    light: float = 0.58,
    seed_hue: float = 0.07,
) -> NDArray[np.uint8]:
    """Return an ``(n, 3)`` uint8 palette with golden-ratio hue spacing.

    Adjacent indices land far apart in hue regardless of ``n`` — what
    we want for label-based visualizations, where the label IDs that
    sit next to each other spatially also sit next to each other in
    the palette index.
    """
    import colorsys

    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    phi = 0.618033988749895
    out = np.zeros((n, 3), dtype=np.uint8)
    h = seed_hue % 1.0
    for i in range(n):
        r, g, b = colorsys.hls_to_rgb(h, light, sat)
        out[i] = (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
        h = (h + phi) % 1.0
    return out


def _renumber_for_adjacency_contrast(
    labeling: NDArray[np.int32],
    background: int = -1,
) -> Tuple[NDArray[np.int32], int]:
    """Permute label IDs so adjacent labels (in 8-connectivity) get
    indices that are far apart modulo n, maximising the visual
    contrast a golden-ratio palette gives them.

    The naive remap is identity, which leaves the input's label IDs
    intact: those IDs are correlated with spatial position (watershed
    seeds were assigned in row-major scan order), so neighbours often
    pick up consecutive palette entries. The remap below scans the
    labels in the order they first appear in a row-major walk and
    assigns them sequential new IDs interleaved by a stride relatively
    prime to ``n`` — which lands neighbouring labels at well-separated
    palette positions.

    Returns ``(renumbered, n_labels)`` where ``renumbered`` has the
    same shape as ``labeling`` with non-background values in
    ``[0, n_labels)`` and the background unchanged.
    """
    flat = labeling.ravel()
    unique_in_order: list = []
    seen = {}
    for v in flat:
        if v == background or v in seen:
            continue
        seen[v] = len(unique_in_order)
        unique_in_order.append(int(v))
    n = len(unique_in_order)
    if n == 0:
        return labeling.copy(), 0

    # Stride: small odd number coprime with n that scatters consecutive
    # input IDs across the palette. 7 is a fine default for any n not
    # divisible by 7; fall back to 1 if it ever isn't coprime.
    stride = 7 if (n % 7 != 0) else (5 if (n % 5 != 0) else 3 if (n % 3 != 0) else 1)
    new_id_for_old = np.full(max(unique_in_order) + 1, -1, dtype=np.int32)
    for k, old in enumerate(unique_in_order):
        new_id_for_old[old] = (k * stride) % n

    out = np.full_like(labeling, background, dtype=np.int32)
    mask = labeling != background
    out[mask] = new_id_for_old[labeling[mask]]
    return out, n


def visualize_command_labeling(
    binary: NDArray[np.bool_],
    raw_segments: Sequence[NDArray[np.float64]],
    labeled_commands: Sequence,
    start_pos: Tuple[float, float] = (0.0, 0.0),
    start_heading: float = 0.0,
    *,
    background: Tuple[int, int, int] = (255, 255, 255),
    binary_color: Tuple[int, int, int] = (235, 235, 235),
    output_path: Optional[str] = None,
) -> Image.Image:
    """Render the per-drawing-command raw-segment back-pointers from
    ``LowGeometryVectorize.labeled_commands_consolidated``.

    For each drawing command (pen-down line / arc / circle):
    * sample the command's geometry between every consecutive pair of
      ``command_start_ratio`` / ``command_end_ratio`` boundaries,
    * paint the sampled stretch in the colour of its ``raw_segment_id``,
    * paint the contributing raw-segment range in the same colour as a
      faint underlay.

    Adjacent spans get well-separated palette entries (golden-ratio
    hue spacing after stride-renumbering), so an over-merged span (one
    colour where two would be expected) or an over-split span (a hue
    jump inside one continuous run) is visually obvious.

    Spins / pen-up transit commands carry no labels and are skipped.
    """
    H, W = binary.shape

    # Renumber raw-segment ids for adjacency contrast — same trick as
    # the segment-stage viz.
    ids_in_order: List[int] = []
    seen = {}
    for lc in labeled_commands:
        for span in lc.spans:
            rid = span.raw_segment_id
            if rid in seen:
                continue
            seen[rid] = len(ids_in_order)
            ids_in_order.append(rid)
    n_distinct = len(ids_in_order)
    stride = (
        7 if (n_distinct > 0 and n_distinct % 7 != 0)
        else 5 if (n_distinct > 0 and n_distinct % 5 != 0)
        else 3 if (n_distinct > 0 and n_distinct % 3 != 0)
        else 1
    )
    new_index_for_raw_id = {
        rid: (k * stride) % max(n_distinct, 1)
        for k, rid in enumerate(ids_in_order)
    }
    palette = _golden_ratio_palette(max(n_distinct, 1))

    canvas = np.full((H, W, 3), background, dtype=np.uint8)
    if binary.any():
        canvas[binary] = binary_color

    def _paint(points: NDArray[np.float64], color: Tuple[int, int, int]) -> None:
        if len(points) == 0:
            return
        ix = np.clip(np.round(points[:, 0]).astype(int), 0, W - 1)
        iy = np.clip(np.round(points[:, 1]).astype(int), 0, H - 1)
        canvas[iy, ix] = color

    # Underlay: raw segments faintly painted in their assigned colour
    # (dim grey for raw segments no command claimed).
    for raw_id, poly in enumerate(raw_segments):
        if raw_id in new_index_for_raw_id:
            color = tuple(palette[new_index_for_raw_id[raw_id]].tolist())
        else:
            color = (180, 180, 180)
        _paint(np.asarray(poly, dtype=float), color)  # type: ignore[arg-type]

    # Walk the command stream geometrically, paint each drawing
    # primitive's span ranges in their raw-segment colour. Use the same
    # simulator as ``_simulate`` to recover each drawn primitive's
    # spatial geometry; then sample within ratio windows.
    drawn, _pen_up, _bbox = _simulate(labeled_commands_to_commands(labeled_commands),
                                        start_pos, start_heading)

    n_samples_per_ratio = 64
    drawing_iter = iter(drawn)
    for lc in labeled_commands:
        if lc.primitive_id is None or not lc.spans:
            continue
        try:
            d = next(drawing_iter)
        except StopIteration:
            break
        # Sample the command geometry uniformly in parameter t.
        ts = np.linspace(0.0, 1.0, n_samples_per_ratio * max(len(lc.spans), 1) + 1)
        if d["kind"] == "line":
            p0, p1 = d["p0"], d["p1"]
            pts = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
        else:  # arc
            center = d["center"]
            r = float(d["radius"])
            sweep = float(d["sweep"])
            start_a = math.atan2(
                d["p0"][1] - center[1], d["p0"][0] - center[0]
            )
            angles = start_a + ts * sweep
            pts = center[None, :] + r * np.stack(
                [np.cos(angles), np.sin(angles)], axis=1
            )
        # For each span, paint pixels whose t falls within
        # [start_ratio, end_ratio].
        for span in lc.spans:
            lo = min(span.command_start_ratio, span.command_end_ratio)
            hi = max(span.command_start_ratio, span.command_end_ratio)
            mask = (ts >= lo) & (ts <= hi)
            if not mask.any():
                continue
            color_idx = new_index_for_raw_id[span.raw_segment_id]
            color = tuple(palette[color_idx].tolist())
            _paint(pts[mask], color)  # type: ignore[arg-type]

    img = Image.fromarray(canvas, mode="RGB")
    if output_path is not None:
        img.save(output_path)
    return img


def labeled_commands_to_commands(labeled_commands: Sequence) -> List[DrawingCommand]:
    """Helper to strip ``LabeledCommand`` wrappers back to a plain
    command list (the original sequence ``_simulate`` consumes)."""
    return [lc.command for lc in labeled_commands]


def visualize_segment_labeling(
    binary: NDArray[np.bool_],
    raw_segments: Sequence[NDArray[np.float64]],
    labeled_segments: Sequence,
    *,
    background: Tuple[int, int, int] = (255, 255, 255),
    binary_color: Tuple[int, int, int] = (235, 235, 235),
    output_path: Optional[str] = None,
) -> Image.Image:
    """Render ``Segment.labeled_segments`` so each raw-segment span is
    drawn in a distinct colour, with the underlying raw segments drawn
    in the SAME colour as the spans that claim them.

    Each raw-segment id gets a colour from a golden-ratio palette
    (after renumbering for adjacency contrast across the spans that
    appear in the final segments). The final polylines are drawn
    pixel-by-pixel in their span's colour; the raw segments are drawn
    as a faint underlay in their assigned colour too, so you can see
    at a glance which raw segments contributed to which final spans
    (and which raw segments got dropped along the way).

    Adjacent spans land at well-separated palette positions, so an
    over-merged span (two spans collapsed into one) shows as a single
    colour where two would be expected, and an over-split span shows
    as a hue change inside what should be one continuous run.
    """
    H, W = binary.shape

    # The "labels" we colour against are the raw-segment ids that
    # actually show up in any final span (raw segments that got
    # filtered out or never matched are still drawn faintly, but they
    # don't drive the adjacency-contrast renumbering).
    spans_in_order: List[int] = []
    for ls in labeled_segments:
        for span in ls.spans:
            spans_in_order.append(span.raw_segment_id)

    # Build a renumbering that scatters adjacent span ids across the
    # palette (the dense-1D variant of the trick used by
    # ``_renumber_for_adjacency_contrast``).
    seen = {}
    unique_in_order: List[int] = []
    for raw_id in spans_in_order:
        if raw_id in seen:
            continue
        seen[raw_id] = len(unique_in_order)
        unique_in_order.append(raw_id)
    n_distinct = len(unique_in_order)
    stride = (
        7 if (n_distinct > 0 and n_distinct % 7 != 0)
        else 5 if (n_distinct > 0 and n_distinct % 5 != 0)
        else 3 if (n_distinct > 0 and n_distinct % 3 != 0)
        else 1
    )
    new_index_for_raw_id = {
        raw_id: (k * stride) % max(n_distinct, 1)
        for k, raw_id in enumerate(unique_in_order)
    }
    palette = _golden_ratio_palette(max(n_distinct, 1))

    canvas = np.full((H, W, 3), background, dtype=np.uint8)
    if binary.any():
        canvas[binary] = binary_color

    # Underlay: every raw segment in the colour its id would map to if
    # it survives in any final span (dim grey for raw segments that
    # never made it through).
    def _paint(points: NDArray[np.float64], color: Tuple[int, int, int]) -> None:
        if len(points) == 0:
            return
        ix = np.clip(np.round(points[:, 0]).astype(int), 0, W - 1)
        iy = np.clip(np.round(points[:, 1]).astype(int), 0, H - 1)
        canvas[iy, ix] = color

    for raw_id, poly in enumerate(raw_segments):
        if raw_id in new_index_for_raw_id:
            color = tuple(palette[new_index_for_raw_id[raw_id]].tolist())
        else:
            color = (180, 180, 180)
        _paint(np.asarray(poly, dtype=float), color)  # type: ignore[arg-type]

    # Overlay: every final-polyline span in its span colour, on top of
    # the binary underlay. Adjacent spans have contrasting colours by
    # construction, so a missing or merged span boundary is obvious.
    for ls in labeled_segments:
        for span in ls.spans:
            color_idx = new_index_for_raw_id[span.raw_segment_id]
            color = tuple(palette[color_idx].tolist())
            seg = ls.points[span.start : span.end]
            _paint(seg, color)  # type: ignore[arg-type]

    img = Image.fromarray(canvas, mode="RGB")
    if output_path is not None:
        img.save(output_path)
    return img


def visualize_skeleton_labeling(
    binary: NDArray[np.bool_],
    skel: NDArray[np.bool_],
    labeling: NDArray[np.int32],
    *,
    background: Tuple[int, int, int] = (255, 255, 255),
    skel_dot_color: Tuple[int, int, int] = (0, 0, 0),
    output_path: Optional[str] = None,
) -> Image.Image:
    """Render ``Skeletonize.labeling`` so each binary pixel takes the
    colour of its assigned skeleton pixel.

    Each skeleton pixel gets a distinct colour (golden-ratio hue
    spacing after renumbering for adjacency contrast); every binary
    pixel takes the colour of its associated skeleton pixel; the
    skeleton itself is overlaid in black so you can see where the
    seeds sit relative to their assigned regions.

    Adjacent skeleton labels come out in different colours, so you
    can verify at a glance that the partition at a junction respects
    the actual stroke topology (two arms shouldn't share a colour
    where they meet) and that no two neighbouring skeleton pixels
    got merged into the same region.
    """
    H, W = binary.shape
    renumbered, n_labels = _renumber_for_adjacency_contrast(labeling)
    palette = _golden_ratio_palette(max(n_labels, 1))

    canvas = np.full((H, W, 3), background, dtype=np.uint8)
    mask = renumbered >= 0
    if mask.any():
        canvas[mask] = palette[renumbered[mask]]
    # Black skeleton overlay so each seed is visible against its region.
    sy, sx = np.where(skel)
    canvas[sy, sx] = skel_dot_color

    img = Image.fromarray(canvas, mode="RGB")
    if output_path is not None:
        img.save(output_path)
    return img
