"""Objective fidelity metric for vectorized output.

The pipeline trades fidelity for drawing speed (fewer commands = fewer
spins = faster, but a coarser reproduction). Without a number for the
fidelity side of that trade, "did this change make the output better or
worse" is a subjective judgement made by squinting at overlay PNGs.

This module rasterizes the drawn command centerlines and compares them
against the source image with distance transforms, yielding:

  * precision  — fraction of output centerline pixels lying within
                 ``tol`` px of source ink. Low precision = the output
                 draws strokes that aren't in the source.
  * recall     — fraction of source skeleton pixels lying within
                 ``tol`` px of an output centerline pixel. Low recall =
                 the output missed strokes / cut corners.
  * f1         — harmonic mean of the two.
  * chamfer_px — symmetric mean nearest-neighbour distance (px); a
                 scale-sensitive companion to the tolerance-thresholded
                 coverage numbers.

Precision is measured against the *binary* (filled stroke ribbon): an
output centerline that sits anywhere inside the real stroke is correct,
so its distance to ink is ~0. Recall is measured against the 1-px
*skeleton*: centerline-to-centerline, which is the honest "did we go
where the stroke goes" question.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np
from scipy import ndimage

from .commands import DrawingCommand
from .visualize import _simulate


def _estimate_stroke_width(binary: np.ndarray) -> float:
    """Stroke width estimate (px): 2x the median distance-transform
    value over ink pixels. Same estimator ``auto_config`` uses."""
    if not binary.any():
        return 3.0
    dt = ndimage.distance_transform_edt(binary)
    vals = dt[binary]
    return max(1.0, 2.0 * float(np.median(vals)))


def _rasterize_commands(
    commands: Sequence[DrawingCommand],
    shape: tuple,
    start_pos,
    start_heading: float,
) -> np.ndarray:
    """Replay the commands and rasterize every pen-down primitive as a
    1-px centerline mask of the given ``shape`` (h, w).

    Primitives are sampled at ~0.5 px spacing so the mask is
    well-connected even on diagonals (the distance-transform metric is
    forgiving of small gaps, but a connected mask keeps recall honest).
    """
    drawn, _pen_up, _bounds = _simulate(commands, start_pos, start_heading)
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)

    def plot(x: float, y: float) -> None:
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= yi < h and 0 <= xi < w:
            mask[yi, xi] = True

    for d in drawn:
        if d["kind"] == "line":
            p0 = np.asarray(d["p0"], dtype=float)
            p1 = np.asarray(d["p1"], dtype=float)
            length = float(np.linalg.norm(p1 - p0))
            n = max(2, int(length * 2.0))
            for k in range(n + 1):
                t = k / n
                p = p0 + t * (p1 - p0)
                plot(p[0], p[1])
        else:  # arc
            center = np.asarray(d["center"], dtype=float)
            r = float(d["radius"])
            sweep = float(d["sweep"])
            p0 = np.asarray(d["p0"], dtype=float)
            start_a = math.atan2(p0[1] - center[1], p0[0] - center[0])
            arc_len = abs(sweep) * r
            n = max(2, int(arc_len * 2.0))
            for k in range(n + 1):
                a = start_a + sweep * (k / n)
                plot(center[0] + r * math.cos(a), center[1] + r * math.sin(a))

    return mask


def coverage_metrics(
    commands: Sequence[DrawingCommand],
    source_binary: np.ndarray,
    source_skeleton: np.ndarray,
    start_pos,
    start_heading: float,
    tol_px: float | None = None,
) -> Dict[str, float]:
    """Compare a command stream against the source image.

    ``source_binary``   — filled stroke mask (bool, h x w).
    ``source_skeleton`` — 1-px skeleton of the source (bool, h x w).
    ``tol_px``          — coverage tolerance; defaults to the estimated
                          stroke width (so "within a stroke width"
                          counts as on-target).
    """
    if tol_px is None:
        tol_px = _estimate_stroke_width(source_binary)

    out = _rasterize_commands(
        commands, source_binary.shape, start_pos, start_heading
    )

    n_out = int(out.sum())
    n_skel = int(source_skeleton.sum())
    if n_out == 0 or n_skel == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "chamfer_px": float("inf"),
            "out_pixels": float(n_out),
            "tol_px": float(tol_px),
        }

    # dist_to_ink[y, x] = distance from (x, y) to the nearest ink pixel.
    dist_to_ink = ndimage.distance_transform_edt(~source_binary.astype(bool))
    # dist_to_out[y, x] = distance from (x, y) to the nearest output px.
    dist_to_out = ndimage.distance_transform_edt(~out)

    out_to_ink = dist_to_ink[out]              # per output pixel
    skel_to_out = dist_to_out[source_skeleton.astype(bool)]  # per skel pixel

    precision = float(np.mean(out_to_ink <= tol_px))
    recall = float(np.mean(skel_to_out <= tol_px))
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
    chamfer = 0.5 * (float(np.mean(out_to_ink)) + float(np.mean(skel_to_out)))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "chamfer_px": chamfer,
        "out_pixels": float(n_out),
        "tol_px": float(tol_px),
    }


def format_metrics(m: Dict[str, float]) -> str:
    """One-line human-readable summary of ``coverage_metrics`` output."""
    return (
        f"fidelity: f1={m['f1']:.3f} "
        f"(precision={m['precision']:.3f}, recall={m['recall']:.3f}), "
        f"chamfer={m['chamfer_px']:.2f}px @tol={m['tol_px']:.1f}px"
    )
