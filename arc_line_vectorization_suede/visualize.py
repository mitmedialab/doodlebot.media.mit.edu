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
):
    """SVG fragments for one drawing -- pen-up dashes, drawn primitives
    rainbow-colored by execution order, start dot and end ring -- in the
    drawing's native coordinate system. Caller is responsible for the
    outer <svg>, the background <rect>, and any wrapping <g transform>
    that places the drawing in the final layout.
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
        hue = 360.0 * i / max(n, 1)
        color = _hsl(hue)
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

    if drawn:
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
):
    """Render a command list to an SVG file. Returns the SVG string."""
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
