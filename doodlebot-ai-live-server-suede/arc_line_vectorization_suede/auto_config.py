"""Image-derived numerical thresholds for the pipeline.

Every stage of the pipeline (skeletonize, segment, graph, vectorize,
optimize) accepts a constellation of numerical tolerances. Most of
them are *pixel distances* tuned for a "typical" hand-drawn stroke
on a roughly-1k canvas — concretely, the test images here have
stroke widths around 3 px, and the configs in ``default_pipeline``
were tuned against that. When the input is at a different scale
(a tiny 256×256 sketch with 1 px strokes, a high-res 4096×4096 sketch
with 12 px strokes), those tolerances need to scale with it or the
pipeline misfires — junctions don't merge, holes don't fill, the
solver over- or under-snaps endpoints, etc.

This module computes one number from the binary image — the
characteristic stroke width — and emits a full config tree with every
pixel-distance threshold scaled relative to that. The reference
stroke width is 3.0 px (the median of the bundled examples), so for
images at that scale every derived value equals the hand-tuned
default.

Knobs deliberately NOT scaled by stroke width:

* Indices and counts (``lookback``, ``tangent_sample``,
  ``crossing_tangent_skip``, …) — these are along-polyline window
  sizes that depend on skeleton sampling density, which is one
  sample per pixel regardless of stroke width.
* Angle thresholds in degrees (``cusp_angle_threshold_deg``,
  ``min_tangent_spread_deg``, ``line_angle_tol_deg``) — these
  encode perceptual judgments about corners/curves and are
  scale-invariant.
* Ratios and dimensionless weights (``fat_ratio``,
  ``min_tangent_score``, ``corner_threshold``, ``gap_penalty``,
  ``curvature_penalty``, ``pixels_per_inch``, beautify ``_rel``
  tolerances, …) — these are unitless and don't depend on image
  scale.
* Algorithm-choice flags (``method``, ``merge_arcs``,
  ``reskeletonize``) — these aren't numerical thresholds.

If you find a parameter here that should also vary with image scale
but doesn't, that's a bug — fix the derivation rather than re-tuning
all the call sites.
"""

from __future__ import annotations
from typing import Any, Dict, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .skeletonize import _to_grayscale_float

# Reference stroke width the hand-tuned defaults assume. Computed as
# the rough median of stroke widths in the bundled examples — 2.8 px
# for the smallest, 4.5 px for the largest, with most at ~3 px.
_REF_STROKE_WIDTH_PX = 3.0


def _load_binary(
    source: Union[str, NDArray[np.uint8], NDArray[np.float64]],
) -> NDArray[np.bool_]:
    """Binarize an image source the same way ``Skeletonize`` does so the
    stroke-width estimate matches what the rest of the pipeline sees.

    The pipeline's ``Skeletonize.Config.Binarize(threshold=0.5)`` is
    the standard — values below 0.5 are strokes, above are background.

    Accepts the same source shapes ``Skeletonize`` does: a file path, a
    2-D grayscale array (bool / integer / float), or a 3-D RGB/RGBA
    array. A multi-channel array is collapsed to a single grayscale
    channel via the exact converter the pipeline uses, so passing a raw
    ``np.asarray(rgb_image)`` to ``default_pipeline`` works rather than
    raising deep inside config derivation.
    """
    if isinstance(source, str):
        arr = np.asarray(Image.open(source).convert("L"), dtype=np.float64) / 255.0
    elif isinstance(source, np.ndarray):
        if source.dtype == np.bool_:
            return source
        if source.ndim == 3:
            # RGB/RGBA: composite-over-white (RGBA) then luma-reduce,
            # exactly as ``Skeletonize`` does for array sources. The
            # result is already a float grayscale in [0, 1].
            arr = _to_grayscale_float(source)
        else:
            arr = source.astype(np.float64)
            if arr.max() > 1.0 + 1e-6:
                arr = arr / 255.0
    else:
        raise TypeError(f"unsupported source type {type(source)!r}")
    return arr < 0.5


def estimate_stroke_width(binary: NDArray[np.bool_]) -> float:
    """Estimate the characteristic stroke width of a binarized drawing
    in pixels.

    Mechanism: the Euclidean distance transform of the stroke mask
    gives, at every stroke pixel, the distance to the nearest
    background pixel. Half the stroke width at that pixel.

    Using the *median* across stroke pixels (rather than the max or a
    high percentile) is the most stable estimator across hand-drawn
    inputs:

    * The max is dominated by any pathologically fat blob — a single
      filled-in cell or a thick joint inflates the estimate.
    * High percentiles (p80, p95) overweight the medial-axis pixels
      and yield numbers ~1.5× the actual stroke width.
    * The median takes the central pixel of the typical stroke cross-
      section, which is consistently ~half the stroke width across
      most of the drawing. For a uniform stroke of width W, half the
      stroke's pixels have dt ≥ W/4, so 2 × median(dt) ≈ W/2 — close
      to the perceived stroke width.

    Hand-drawn examples in this repo come out at 2.8–4.5 px with this
    estimator, matching the visual stroke width and the regime the
    hand-tuned defaults were calibrated against.
    """
    if not binary.any():
        return _REF_STROKE_WIDTH_PX
    dt = distance_transform_edt(binary)
    skel = dt[binary]
    if len(skel) == 0:
        return _REF_STROKE_WIDTH_PX
    return float(2.0 * np.median(skel))


def _round_int(value: float, lo: int = 1) -> int:
    return max(lo, int(round(value)))


def derive_configs(
    source: Union[str, NDArray[np.uint8], NDArray[np.float64], NDArray[np.bool_]],
) -> Dict[str, Any]:
    """Return a full set of pipeline configs scaled to the input image.

    Pass the same source you'd hand to ``Skeletonize`` (a file path,
    a uint8 array, a float [0, 1] array, or a precomputed bool mask).
    The returned dict's keys are stage names; each value is the
    ``dict`` (or kwargs) you can hand to that stage's ``Config``
    constructor.
    """
    binary = _load_binary(source)
    sw = estimate_stroke_width(binary)
    h, w = binary.shape
    diag = float(np.hypot(w, h))

    # ``s`` is the linear scale relative to the reference stroke width.
    # ``s2`` scales areas.
    s = sw / _REF_STROKE_WIDTH_PX
    s2 = s * s

    return {
        # Diagnostics — handy for callers to log / report.
        "stroke_width_px": sw,
        "image_diagonal_px": diag,
        "scale": s,
        # --- Skeletonize ----------------------------------------------
        "binarize": {
            "threshold": 0.5,
        },
        "skeletonize": {
            "method": "zhang",
        },
        "collapse": {
            "skeletonize_method": "lee",
            "max_hole_area": _round_int(10 * s2),
            "max_thin_thickness": 3.0 * s,
            "reskeletonize": True,
        },
        "eyes": {
            # Filled-shape detector. A pupil / drawn-dot fills a small
            # connected component whose medial-axis skeleton has
            # collapsed to a handful of pixels. We accept the component
            # as a "fill" when its skeleton is small AND its area-to-
            # skeleton ratio is well above what a normal stroke
            # produces (a stroke has ``area/skel ~ stroke_width``).
            #
            # * ``max_skeleton_pixels`` scales linearly with stroke
            #   width — a slightly-noisy filled disk at sw=3 collapses
            #   to ~1-5 pixels; at sw=6 it can collapse to ~10.
            # * ``min_area`` scales with area — too small and we'd
            #   trigger on 2x2 speckles; too large and we'd miss small
            #   pupils.
            # * ``min_fill_ratio`` is dimensionless. 4.0 sits cleanly
            #   above any reasonable stroke (which lands at ~stroke
            #   width, i.e. ~3) and below every filled disk we've
            #   measured (which lands at 10x+).
            "max_skeleton_pixels": _round_int(8 * s),
            "min_fill_ratio": 4.0,
            "min_area": _round_int(9 * s2),
        },
        "detect": {
            # NOTE: the chromosome / crossing detector's parameters are
            # deliberately NOT scaled by stroke width, unlike the rest
            # of this file. Two reasons:
            #
            #  1. Chicken-and-egg. A "chromosome" is a region where two
            #     strokes overlap into wide ink. That wide ink inflates
            #     ``estimate_stroke_width`` (it is a median over the
            #     distance transform), so scaling the crossing detector
            #     by that estimate feeds the detector's own input
            #     pathology back into its tuning. Measured: angryhashtag
            #     estimates 8.25 px and beachnugget 6.0 px vs a ~2.8 px
            #     true width — scaling by s=2-2.75x then mis-sized every
            #     threshold here and the detector missed real crossings
            #     (scribble 2 -> 1) and hallucinated false ones
            #     (bikelove 0 -> 1).
            #  2. Redundant. The detector already adapts to local stroke
            #     width itself via ``local_tau`` (the radius below sets
            #     the neighborhood for that local estimate). A global
            #     stroke-width multiplier on top of a locally adaptive
            #     detector buys nothing.
            #
            # If genuine multi-resolution support is needed later, the
            # right approach is to derive these distances from the
            # detector's robust internal ``local_tau`` — not from the
            # global median estimate.
            "local_tau_radius": 40,
            "fat_ratio": 1.3,
            "min_fat_area": 8,
            "group_dilate": 15,
            "skel_ring_dilate": 5,
            "pairing_tangent_steps": 8,
            "pairing_threshold": 1.2,
            "min_chromosome_skel_length": 15,
        },
        # --- Segment --------------------------------------------------
        "segment": {
            "min_length": 10.0 * s,
        },
        "fuse": {
            "max_path_length": _round_int(20 * s, lo=2),
            "lookback": 10,
            "min_tangent_score": 0.5,
            "gap_penalty": 0.05,
            "curvature_penalty": 3.0,
        },
        "repair": {
            "junction_tol": 2.5 * s,
            "stable_skip": 2,
            "stable_sample": 6,
            "max_junction_region_length": _round_int(20 * s, lo=2),
            "min_output_polyline_length": 2,
            "min_tangent_spread_deg": 15.0,
            "interp_max_spacing": 1.0 * s,
            "min_curvature_spike_ratio": 2.0,
            "curvature_context_window": 8,
            # Index gap (count) — not scaled by stroke width.
            "cascade_gap": 3,
            # Pixel distance — scaled.
            "cascade_max_jp_distance": 3.0 * s,
        },
        "post_repair_fuse": {
            "junction_tol": 2.5 * s,
            "tangent_skip": 2,
            "tangent_sample": 10,
            "min_tangent_score": 0.6,
            "curvature_penalty": 1.0,
        },
        # --- StrokeGraph ----------------------------------------------
        "graph_build": {
            "junction_tol": 2.5 * s,
            "terminal_tangent_window": 10,
            "crossing_tangent_skip": 2,
            "crossing_tangent_half_window": 6,
            "cusp_angle_threshold_deg": 50.0,
            "cluster_merge_centroid_distance": 10.0 * s,
            "cluster_merge_index_gap": 10,
        },
        # --- HighGeometry vectorize -----------------------------------
        "high_geometry_commands": {
            "sigma": 2.0,
            "corner_threshold": 0.25,
            "max_fit_residual": 5.0 * s,
        },
        # --- OptimizeRoute --------------------------------------------
        "optimize_route": {
            "pixels_per_inch": 1.0,
            "pen_up_join_tol": 0.5 * s,
            "two_opt_passes": 16,
            "or_opt_passes": 8,
            "or_opt_max_segment_len": 3,
        },
    }
