"""PNG -> drawing-robot commands pipeline.

Pipeline stages (matching the recipe):

    1. skeletonize the binary image (skimage)
    2. trace the skeleton into polylines using a graph walk
    3. resample, smooth, and segment each polyline at curvature peaks (corners)
    4. classify each segment as line or arc (LS line + algebraic circle fit)
    5. order strokes greedily to minimize pen-up travel
    6. emit a list of DrawingCommand dicts matching the TS type
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from ...commands import (
    ArcCommand,
    ArcPrimitive,
    DrawingCommand,
    LineCommand,
    LinePrimitive,
    Primitive,
    SpinCommand,
    Stroke,
)
from ..labels_common import LabeledCommand, label_commands_geometric

# ============================================================================
# Stage 3: curvature-based segmentation
# ============================================================================


def _resample_polyline(poly: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a polyline to `n` uniformly spaced points by arc length.

    Returns (resampled_points, original_arc_length_at_each_resample).
    """
    diffs = np.diff(poly, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = s[-1]
    if total < 1e-9:
        return poly.copy(), s
    s_new = np.linspace(0.0, total, n)
    x_new = np.interp(s_new, s, poly[:, 0])
    y_new = np.interp(s_new, s, poly[:, 1])
    return np.column_stack([x_new, y_new]), s_new


def _curvature(xy: np.ndarray, sigma: float) -> np.ndarray:
    """Discrete signed curvature along a uniformly-spaced polyline."""
    xs = gaussian_filter1d(xy[:, 0], sigma)
    ys = gaussian_filter1d(xy[:, 1], sigma)
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5
    denom = np.where(denom < 1e-10, 1e-10, denom)
    return (dx * ddy - dy * ddx) / denom


def segment_at_corners(
    poly: np.ndarray,
    sigma: float = 2.0,
    corner_threshold: float = 0.25,
    min_segment_length: int = 4,
) -> List[np.ndarray]:
    """Cut a polyline at peaks of |curvature|.

    Args:
        poly:                ordered (N, 2) points.
        sigma:               Gaussian smoothing scale (in resampled units).
        corner_threshold:    minimum |curvature| to be considered a corner.
        min_segment_length:  drop segments shorter than this many points.
    """
    if len(poly) < 2 * min_segment_length:
        return [poly]

    diffs = np.diff(poly, axis=0)
    total_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    if total_len < 4.0:
        return [poly]

    n_samples = max(int(total_len), 16)
    resampled, s_resampled = _resample_polyline(poly, n_samples)
    kappa = _curvature(resampled, sigma)
    abs_k = np.abs(kappa)

    peaks, _ = find_peaks(
        abs_k,
        height=corner_threshold,
        distance=max(int(sigma * 3), 3),
    )

    # Map each peak (in resampled-index space) back to an index in `poly`.
    if len(peaks) == 0:
        return [poly]

    # Cumulative arc length along the original polyline:
    s_orig = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    cut_points = []
    for p in peaks:
        s_at_peak = s_resampled[p]
        idx = int(np.searchsorted(s_orig, s_at_peak))
        idx = max(1, min(len(poly) - 2, idx))
        cut_points.append(idx)
    cut_points = sorted(set(cut_points))

    # Build segments, dropping those that are too short.
    segments: List[np.ndarray] = []
    prev = 0
    for ci in cut_points:
        if ci - prev >= min_segment_length:
            # +1 to include the cut point as the segment endpoint
            segments.append(poly[prev : ci + 1].copy())
            prev = ci
    if len(poly) - prev >= min_segment_length:
        segments.append(poly[prev:].copy())

    return segments if segments else [poly]


# ============================================================================
# Stage 4: line vs arc fitting
# ============================================================================


def fit_line(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Total-least-squares line fit; returns (start, end, max_perp_residual).

    `start` and `end` are the projections of the first/last points onto the
    fitted axis.
    """
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]
    perp = Vt[1]
    projections = centered @ direction
    perp_dists = np.abs(centered @ perp)
    # Use the projection of the first/last polyline point so endpoints
    # match the original ordering.
    t_first = projections[0]
    t_last = projections[-1]
    start = centroid + t_first * direction
    end = centroid + t_last * direction
    return start, end, float(perp_dists.max())


def fit_circle_algebraic(pts: np.ndarray):
    """Algebraic circle fit (a refined version of Kåsa's method).

    Returns (cx, cy, r) or None if the system is degenerate.
    """
    xs = pts[:, 0]
    ys = pts[:, 1]
    x_m = xs.mean()
    y_m = ys.mean()
    u = xs - x_m
    v = ys - y_m
    Suu = float((u * u).sum())
    Svv = float((v * v).sum())
    Suv = float((u * v).sum())
    Suuu = float((u * u * u).sum())
    Svvv = float((v * v * v).sum())
    Suvv = float((u * v * v).sum())
    Svuu = float((v * u * u).sum())
    A = np.array([[Suu, Suv], [Suv, Svv]])
    b = 0.5 * np.array([Suuu + Suvv, Svvv + Svuu])
    try:
        uc, vc = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    cx = uc + x_m
    cy = vc + y_m
    r = math.sqrt(uc * uc + vc * vc + (Suu + Svv) / len(pts))
    return cx, cy, r


def _arc_traversal_ccw(pts: np.ndarray, cx: float, cy: float) -> bool:
    """True if the polyline traverses around (cx, cy) counter-clockwise."""
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    unwrapped = np.unwrap(angles)
    return float(unwrapped[-1] - unwrapped[0]) > 0.0


class Arc(TypedDict):
    center: np.ndarray
    radius: float
    residual: float
    ccw: bool


def fit_arc(pts: np.ndarray) -> Arc | None:
    """Fit a circular arc; returns dict with center/radius/residual or None."""
    if len(pts) < 3:
        return None
    fit = fit_circle_algebraic(pts)
    if fit is None:
        return None
    cx, cy, r = fit
    if not math.isfinite(r) or r < 0.5:
        return None
    dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    residual = float(np.abs(dists - r).max())
    return {
        "center": np.array([cx, cy]),
        "radius": r,
        "ccw": _arc_traversal_ccw(pts, cx, cy),
        "residual": residual,
    }


def _segment_scale(pts: np.ndarray) -> float:
    """A characteristic length for a segment, robust to closed paths.

    For an open polyline, this is close to the chord length. For a closed
    loop (where start == end and the chord length is zero), this returns
    the bounding-box diagonal, which is a meaningful spatial scale.
    """
    bbox = pts.max(axis=0) - pts.min(axis=0)
    diag = float(np.linalg.norm(bbox))
    return max(diag, 1.0)


def classify_segment(
    pts: np.ndarray,
    arc_advantage: float = 1.5,
    max_radius_factor: float = 8.0,
    min_arc_radius: float = 1.5,
) -> Primitive:
    """Choose between a line and an arc primitive for a single segment.

    The arc is preferred only if its residual is meaningfully better than the
    line residual AND its radius is reasonable (extremely large radii are
    indistinguishable from lines and should be drawn as lines).
    """
    line_start, line_end, line_res = fit_line(pts)
    scale = _segment_scale(pts)
    arc = fit_arc(pts)

    use_arc = (
        arc is not None
        and arc["radius"] >= min_arc_radius
        and arc["radius"] <= max_radius_factor * scale
        and arc["residual"] * arc_advantage < max(line_res, 1e-6)
    )

    if use_arc and arc is not None:
        return ArcPrimitive(
            center=arc["center"],
            radius=arc["radius"],
            # Snap to the actual polyline endpoints so adjacent primitives
            # remain connected. Reproject onto the fitted circle for sanity.
            start=_project_to_circle(pts[0], arc["center"], arc["radius"]),
            end=_project_to_circle(pts[-1], arc["center"], arc["radius"]),
            ccw=arc["ccw"],
        )

    return LinePrimitive(start=pts[0].copy(), end=pts[-1].copy())


def _project_to_circle(p: np.ndarray, center: np.ndarray, r: float) -> np.ndarray:
    d = p - center
    n = np.linalg.norm(d)
    if n < 1e-9:
        return p.copy()
    return center + (r / n) * d


def _best_fit_with_residual(
    pts: np.ndarray,
    arc_advantage: float = 1.5,
    max_radius_factor: float = 8.0,
    min_arc_radius: float = 1.5,
) -> Tuple[Primitive, float]:
    """Same logic as `classify_segment` but also returns the absolute
    residual (in pixels) of the chosen fit. Used by the residual-based
    recursive subdivision in `polyline_to_stroke`.
    """
    line_start, line_end, line_res = fit_line(pts)
    scale = _segment_scale(pts)
    arc = fit_arc(pts)

    use_arc = (
        arc is not None
        and arc["radius"] >= min_arc_radius
        and arc["radius"] <= max_radius_factor * scale
        and arc["residual"] * arc_advantage < max(line_res, 1e-6)
    )
    if use_arc and arc is not None:
        prim = ArcPrimitive(
            center=arc["center"],
            radius=arc["radius"],
            start=_project_to_circle(pts[0], arc["center"], arc["radius"]),
            end=_project_to_circle(pts[-1], arc["center"], arc["radius"]),
            ccw=arc["ccw"],
        )
        return prim, float(arc["residual"])
    return LinePrimitive(start=pts[0].copy(), end=pts[-1].copy()), float(line_res)


# ============================================================================
# Stage 5: build strokes and order them
# ============================================================================


def polyline_to_stroke(
    poly: np.ndarray,
    sigma: float = 2.0,
    corner_threshold: float = 0.25,
    max_fit_residual: float | None = None,
    min_split_length: int = 4,
) -> Stroke:
    """Segment a polyline at corners and fit a primitive per segment.

    If `max_fit_residual` is set (in absolute pixel units), every segment
    whose best line/arc fit has residual greater than that threshold is
    recursively split at its midpoint until each piece fits within
    tolerance or becomes shorter than `min_split_length` points. This is
    the right knob for raising fidelity: corner-based segmentation alone
    doesn't help when a polyline has *gradually* varying curvature (like
    a heart or S-curve) — there's no curvature peak to cut at, and a
    single line/arc cannot capture the shape. Residual-based splitting
    forces extra cuts wherever a curve isn't well approximated by one
    primitive, regardless of whether there's a "corner" there.

    Set `max_fit_residual=None` (the default) for original behaviour:
    only corner-based segmentation, no residual cap.
    """
    segments = segment_at_corners(poly, sigma=sigma, corner_threshold=corner_threshold)
    if max_fit_residual is None:
        return [classify_segment(s) for s in segments if len(s) >= 2]

    primitives: Stroke = []
    # DFS via stack: segments processed left-to-right after splits.
    stack = list(reversed([s for s in segments if len(s) >= 2]))
    while stack:
        seg = stack.pop()
        prim, residual = _best_fit_with_residual(seg)
        if residual <= max_fit_residual or len(seg) < 2 * min_split_length:
            primitives.append(prim)
            continue
        mid = len(seg) // 2
        # Both halves share the midpoint so adjacent primitives connect.
        left = seg[: mid + 1]
        right = seg[mid:]
        # Push right first so left is processed next.
        stack.append(right)
        stack.append(left)
    return primitives


def _stroke_endpoints(stroke: Stroke) -> Tuple[np.ndarray, np.ndarray]:
    return stroke[0].start, stroke[-1].end


def _reverse_primitive(p: Primitive) -> Primitive:
    if isinstance(p, LinePrimitive):
        return LinePrimitive(start=p.end.copy(), end=p.start.copy())
    return ArcPrimitive(
        center=p.center.copy(),
        radius=p.radius,
        start=p.end.copy(),
        end=p.start.copy(),
        ccw=not p.ccw,
    )


def _reverse_stroke(stroke: Stroke) -> Stroke:
    return [_reverse_primitive(p) for p in reversed(stroke)]


def order_strokes(strokes: List[Stroke], origin: np.ndarray) -> List[Stroke]:
    """Greedy nearest-neighbour ordering; reverses strokes when their far end
    is closer to the cursor."""
    remaining = list(range(len(strokes)))
    cursor = origin.astype(float).copy()
    ordered: List[Stroke] = []
    while remaining:
        best_i = remaining[0]
        best_dist = math.inf
        best_reverse = False
        for i in remaining:
            s, e = _stroke_endpoints(strokes[i])
            ds = float(np.linalg.norm(cursor - s))
            de = float(np.linalg.norm(cursor - e))
            if ds < best_dist:
                best_dist, best_i, best_reverse = ds, i, False
            if de < best_dist:
                best_dist, best_i, best_reverse = de, i, True
        chosen = strokes[best_i]
        if best_reverse:
            chosen = _reverse_stroke(chosen)
        ordered.append(chosen)
        cursor = chosen[-1].end.copy()
        remaining.remove(best_i)
    return ordered


# ============================================================================
# Stage 6: ordered strokes -> DrawingCommand list
# ============================================================================


def _wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return ((a + math.pi) % (2 * math.pi)) - math.pi


def _emit_spin_to(target_heading: float, heading: float, eps: float = 1e-4):
    """Yield a SpinCommand if needed to align `heading` with `target_heading`.

    Returns (cmd_or_None, new_heading).
    """
    delta = _wrap_pi(target_heading - heading)
    if abs(delta) < eps:
        return None, heading
    return SpinCommand(kind="spin", degrees=math.degrees(delta)), target_heading


def _emit_primitive(prim: Primitive, pos: np.ndarray, heading: float):
    """Convert a single primitive into commands.

    Returns (cmds_list, new_pos, new_heading).
    """
    cmds: List[DrawingCommand] = []

    if isinstance(prim, LinePrimitive):
        delta = prim.end - pos
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            return cmds, pos, heading
        target_heading = math.atan2(delta[1], delta[0])
        spin_cmd, heading = _emit_spin_to(target_heading, heading)
        if spin_cmd is not None:
            cmds.append(spin_cmd)
        cmds.append(LineCommand(kind="line", distance=dist, penDown=True))
        return cmds, prim.end.copy(), heading

    # ArcPrimitive
    radial = prim.start - prim.center
    # Tangent: 90deg CCW from radial if traversing CCW, else 90deg CW.
    if prim.ccw:
        tangent = np.array([-radial[1], radial[0]])
    else:
        tangent = np.array([radial[1], -radial[0]])
    target_heading = math.atan2(tangent[1], tangent[0])
    spin_cmd, heading = _emit_spin_to(target_heading, heading)
    if spin_cmd is not None:
        cmds.append(spin_cmd)

    start_a = math.atan2(prim.start[1] - prim.center[1], prim.start[0] - prim.center[0])
    end_a = math.atan2(prim.end[1] - prim.center[1], prim.end[0] - prim.center[0])
    if prim.ccw:
        sweep = end_a - start_a
        if sweep <= 0:
            sweep += 2 * math.pi
    else:
        sweep = end_a - start_a
        if sweep >= 0:
            sweep -= 2 * math.pi

    cmds.append(
        ArcCommand(
            kind="arc",
            radius=float(prim.radius),
            degrees=math.degrees(sweep),
        )
    )
    new_heading = _wrap_pi(heading + sweep)
    return cmds, prim.end.copy(), new_heading


def strokes_to_commands(
    strokes: List[Stroke],
    start_pos: np.ndarray | None = None,
    start_heading: float = 0.0,
) -> List[DrawingCommand]:
    """Walk an ordered list of strokes and produce a flat command sequence.

    Between strokes, emits (spin, line penDown=False) to traverse pen-up.
    Within a stroke, emits a spin to align the heading with each primitive's
    starting tangent, then the primitive itself.
    """
    if start_pos is None:
        start_pos = np.zeros(2)
    pos = start_pos.astype(float).copy()
    heading = float(start_heading)
    out: List[DrawingCommand] = []

    for stroke in strokes:
        if not stroke:
            continue

        # Pen-up traversal to the start of this stroke.
        target = stroke[0].start
        delta = target - pos
        dist = float(np.linalg.norm(delta))
        if dist > 1e-6:
            target_heading = math.atan2(delta[1], delta[0])
            spin_cmd, heading = _emit_spin_to(target_heading, heading)
            if spin_cmd is not None:
                out.append(spin_cmd)
            out.append(LineCommand(kind="line", distance=dist, penDown=False))
            pos = target.copy()

        # Draw the stroke.
        for prim in stroke:
            cmds, pos, heading = _emit_primitive(prim, pos, heading)
            out.extend(cmds)

    return out


# ============================================================================
# Top-level driver
# ============================================================================


def polylines_to_commands(
    polylines: List[np.ndarray],
    sigma: float = 2.0,
    corner_threshold: float = 0.25,
    max_fit_residual: float | None = None,
    start_pos: np.ndarray | None = None,
    start_heading: float = 0.0,
) -> List[DrawingCommand]:
    """Run the polyline -> commands half of the pipeline.

    Each polyline is segmented at curvature peaks, every segment is
    classified as a line or an arc, the resulting strokes are ordered
    greedily for minimum pen-up travel, and the whole thing is emitted
    as a flat command list. This is the part of the pipeline that does
    not care where the polylines came from — raw skeleton tracing, the
    fusion stage, or anything else that produces (N, 2) pixel polylines
    is fine input.

    If `max_fit_residual` is set, segments are recursively subdivided
    until each one fits a single line or arc with residual at or below
    that pixel threshold. Useful when corner-based segmentation alone
    does not produce enough cuts on smoothly-varying curves.
    """
    strokes = [
        polyline_to_stroke(
            p,
            sigma=sigma,
            corner_threshold=corner_threshold,
            max_fit_residual=max_fit_residual,
        )
        for p in polylines
    ]
    strokes = [s for s in strokes if s]
    if start_pos is None:
        start_pos = np.zeros(2)
    ordered = order_strokes(strokes, origin=np.asarray(start_pos, dtype=float))
    return strokes_to_commands(
        ordered,
        start_pos=start_pos,
        start_heading=start_heading,
    )


class Vectorize:
    class Config:
        class ToCommands(TypedDict):
            sigma: float
            corner_threshold: float
            max_fit_residual: float | None

    def __init__(
        self,
        polylines: List[NDArray[np.float64]],
        start_pos: NDArray[np.float64],
        start_heading: float,
        commands: Config.ToCommands,
        raw_segments: List[NDArray[np.float64]] | None = None,
    ):
        # NOTE: the high-geometry path is a deliberately *naive* baseline
        # used only for comparison against the low-geometry pipeline. It
        # segments at curvature peaks, classifies each segment as a
        # line/arc, and greedily orders strokes — there is no joint
        # constraint solve and no command consolidation. A
        # command-consolidation pass once lived here but was always worse
        # than the low-geometry solver, so it was removed; consolidate
        # high-geometry output (if ever needed) by running the polylines
        # through ``LowGeometryVectorize`` instead.
        self.start_pos = np.asarray(start_pos, dtype=float)
        self.start_heading = float(start_heading)
        self.commands = polylines_to_commands(
            polylines,
            sigma=commands["sigma"],
            corner_threshold=commands["corner_threshold"],
            max_fit_residual=commands["max_fit_residual"],
            start_pos=start_pos,
            start_heading=start_heading,
        )

        # Optional per-command raw-segment labels. Because this pipeline
        # tracks no point-range provenance through its frozen
        # segment/classify/order logic, the labels are recovered
        # GEOMETRICALLY: each drawing command's path is sampled and each
        # sample matched to the nearest raw-segment pixel (see
        # ``release/vectorize/labels_common.py``). When the caller passes
        # ``raw_segments`` (``Segment.segments``), this populates
        # ``labeled_commands`` parallel to ``commands``; otherwise it
        # stays ``None``.
        self.labeled_commands: Optional[List[LabeledCommand]] = None
        if raw_segments is not None:
            self.labeled_commands = label_commands_geometric(
                self.commands,
                raw_segments,
                start_pos=(float(self.start_pos[0]), float(self.start_pos[1])),
                start_heading=self.start_heading,
            )
