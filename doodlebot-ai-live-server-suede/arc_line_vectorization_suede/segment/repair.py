"""Local junction repair for skeleton-derived polylines.

After ``fuse_segments``, polylines still pass through (or terminate at)
skeleton-level junctions. The handful of pixels nearest a Y, T, or X
junction is typically distorted: skeletonization commits each pixel to
exactly one stroke, so at a crossing it has to make local compromises
that look like small jogs, multi-pixel-thick "horizontal blobs", or
kinks in the polyline path. These don't match the actual stroke
geometry on either side of the junction, and they later confuse the
line/arc fitter.

The fix: trust each segment's behaviour OUTSIDE the junction (its
tangent in the stable region a few pixels away), not the distorted
local geometry. For every junction, solve for the single 2D point
that all stable approaches meet at (a 2x2 least-squares problem), then
replace each polyline's junction-affected indices with that single
point. Polylines that pass through a junction now do so without the
jog; polylines that terminate at one now end at the same shared point
as their neighbours.

Junction validity and region extent are decided GEOMETRICALLY, from
the polyline's smoothed second-derivative magnitude |r''|:

  (a) A cluster of close points across polylines is only treated as a
      real junction if at least one polyline has a |r''| spike in a
      padded window around the cluster. A spike means the polyline
      bends sharply somewhere near the cluster -- exactly what the
      skeletonizer's local "decide where to put pixels" compromises
      produce at a real X/Y/T crossing. Two smooth polylines that just
      happen to run within tol of each other (parallel coincidence,
      tangent kiss) have no such spike and are correctly skipped.
      Searching a few pixels OUTSIDE the strict cluster matters for
      terminating-polyline junctions: there the cluster captures only
      the polyline's first or last few points, and the actual kink
      sits 1-5 pixels in from the endpoint.

  (b) For each polyline in a real junction, the region replaced by
      the junction point is grown outward from the strict spatial
      cluster while |r''| stays elevated above the polyline's local
      baseline. Without this extension, skeleton noise that spills a
      few pixels past the close-points zone survives into the output
      as a small "shelf" or jog next to the cleaned junction.

  (c) Tangents for the LS solve are taken from samples just OUTSIDE
      the extended region, so the meeting point is constrained by
      genuinely smooth polyline geometry rather than by pixels that
      are still inside the noisy region.

Algorithm:

1. Build a junction graph over polyline points. Two points are linked
   if they lie within ``junction_tol`` of each other AND belong to
   different polylines. Connected components are junction clusters.

2. For each cluster spanning >= 2 polylines:
     - Reject if any polyline contributes more than
       ``max_junction_region_length`` consecutive points to the
       cluster (a sanity net against pathologically long runs --
       typically closed-loop polylines whose start and end coincide).
     - Reject unless at least one polyline shows a |r''| spike in a
       padded window around its cluster indices (the kink test).
     - Extend each polyline's region by walking outward through
       elevated-|r''| pixels.
     - Build approaches from samples beyond each extended region.
     - Reject if the approach tangents are all nearly parallel
       (tangent kiss).
     - Solve the 2x2 LS problem for the junction point.

3. For each polyline involved in the junction, splice out the
   junction-affected indices (extended) and replace them with the
   single junction point. Adjacent emitted segments are bridged with
   1-pixel-spaced interpolated points so the polyline stays visually
   continuous and downstream fitters don't see a gap.

References (close in spirit, none are exactly this algorithm):
  - Hilaire & Tombre 2006, "Robust and Accurate Vectorization of Line
    Drawings".
  - Favreau, Lafarge, Bousseau 2016, "Fidelity vs. Simplicity".
  - Bessmeltsev & Solomon 2019, "Vectorization of Line Drawings via
    Polyvector Fields".
  - Bao & Fu 2023, "Joint Curve Network Optimization for Drawing
    Vectorization" (does this jointly with global primitive alignment).
"""

from __future__ import annotations
from typing import List, Tuple, TypedDict

import numpy as np
from numpy.typing import NDArray

from ._helpers import gather_points, spatial_clusters


def _build_approaches(polyline, junction_indices, stable_skip, stable_sample):
    n = len(polyline)
    if not junction_indices:
        return []
    region_start = min(junction_indices)
    region_end = max(junction_indices)
    arms = []
    if region_start > 0:
        inner_idx = region_start - 1
        sample_near = inner_idx - stable_skip
        sample_far = inner_idx - stable_skip - stable_sample + 1
        sample_near = max(0, sample_near)
        sample_far = max(0, sample_far)
        if sample_near < sample_far:
            sample_near, sample_far = sample_far, sample_near
        if sample_near - sample_far < 1:
            sample_far = 0
            sample_near = inner_idx
        if sample_near > sample_far:
            samples = polyline[sample_far : sample_near + 1]
            anchor = samples.mean(axis=0)
            v = samples[0] - samples[-1]
            norm = float(np.linalg.norm(v))
            if norm > 1e-9:
                arms.append((inner_idx, -1, anchor, v / norm))
    if region_end < n - 1:
        inner_idx = region_end + 1
        sample_near = inner_idx + stable_skip
        sample_far = inner_idx + stable_skip + stable_sample - 1
        sample_near = min(n - 1, sample_near)
        sample_far = min(n - 1, sample_far)
        if sample_far < sample_near:
            sample_near, sample_far = sample_far, sample_near
        if sample_far - sample_near < 1:
            sample_near = inner_idx
            sample_far = n - 1
        if sample_far > sample_near:
            samples = polyline[sample_near : sample_far + 1]
            anchor = samples.mean(axis=0)
            v = samples[-1] - samples[0]
            norm = float(np.linalg.norm(v))
            if norm > 1e-9:
                arms.append((inner_idx, +1, anchor, v / norm))
    return arms


def _tangent_spread_deg(tangents):
    if len(tangents) < 2:
        return 180.0
    angles = np.mod(np.array([np.arctan2(t[1], t[0]) for t in tangents]), np.pi)
    sorted_a = np.sort(angles)
    gaps = np.append(np.diff(sorted_a), np.pi - sorted_a[-1] + sorted_a[0])
    max_gap = float(np.max(gaps))
    return float(np.degrees(np.pi - max_gap))


def _solve_junction_point(anchors, outward_tangents, fallback=None):
    if not anchors:
        if fallback is None:
            raise ValueError("No approaches and no fallback")
        return fallback.copy()
    if len(anchors) == 1:
        return anchors[0].copy()
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for a, t in zip(anchors, outward_tangents):
        M = np.eye(2) - np.outer(t, t)
        A += M
        b += M @ a
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.mean(anchors, axis=0) if fallback is None else fallback.copy()


def _merge_cascade_repairs(
    repairs: List[Tuple[int, int, NDArray]],
    cascade_gap: int = 3,
    max_jp_distance: float = 3.0,
) -> List[Tuple[int, int, NDArray]]:
    """Merge repairs whose junction regions are within ``cascade_gap``
    pixels on the same polyline AND whose junction points are within
    ``max_jp_distance`` pixels of each other.

    The original case this targets is one physical junction that
    spatial clustering split into two adjacent clusters with very
    slightly different centroids (e.g., P2 meets P4 at one centroid
    and P5 at a slightly different centroid two pixels away). Applied
    independently, the two repairs would chew through the polyline in
    series and remove a much larger chunk than either junction
    actually warrants; merging them keeps the chunk small.

    The JP-distance guard prevents merging when a single polyline
    threads through two PHYSICALLY DISTINCT junctions whose extended
    regions happen to overlap. Without the guard, both junction
    points get averaged into a single point in between, and the
    polyline's path no longer reaches either junction's true meeting
    point -- a clearly visible disconnect.
    """
    if len(repairs) <= 1:
        return list(repairs)
    repairs = sorted(repairs, key=lambda r: r[0])
    merged: List[Tuple[int, int, NDArray]] = [repairs[0]]
    for js, je, jp in repairs[1:]:
        last_js, last_je, last_jp = merged[-1]
        close_in_index = js - last_je <= cascade_gap
        close_in_space = (
            float(np.linalg.norm(np.asarray(jp) - np.asarray(last_jp)))
            <= max_jp_distance
        )
        if close_in_index and close_in_space:
            new_jp = (
                np.asarray(last_jp, dtype=float) + np.asarray(jp, dtype=float)
            ) / 2.0
            merged[-1] = (last_js, max(last_je, je), new_jp)
        else:
            merged.append((js, je, jp))
    return merged


def _polyline_d2_mag(polyline, smoothing_window=3):
    """Smoothed |2nd derivative| magnitude per polyline point.

    Used to detect skeleton-level kinks. For a polyline parameterized
    at ~1px spacing, |r''| ~ 1/R for a circular section (radius R), so
    a smooth curve has a stable low baseline, while a sharp local
    distortion (e.g. the multi-pixel "horizontal blob" left by skeleton-
    izing an X-crossing) spikes well above that baseline.
    """
    n = len(polyline)
    if n < 3:
        return np.zeros(n)
    dx = np.gradient(polyline[:, 0])
    dy = np.gradient(polyline[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    mag = np.sqrt(ddx * ddx + ddy * ddy)
    if smoothing_window > 1:
        k = min(smoothing_window, n)
        kernel = np.ones(k, dtype=float) / float(k)
        mag = np.convolve(mag, kernel, mode="same")
    return mag


def _kink_present(d2_mag, rs, re, context_window, spike_ratio, search_pad=4):
    """Test if |r''| spikes near the cluster vs the baseline further out.

    Looks for the spike in ``[rs - search_pad, re + search_pad]`` rather
    than just the strict cluster: at terminating-polyline junctions, the
    cluster captures only the few pixels nearest the meeting point, but
    the kink itself (the bend where the polyline turns to fit the
    skeleton's junction blob) usually sits 1-5 pixels INSIDE from the
    endpoint. Searching a slightly padded window catches that. Baseline
    is measured from the pixels just OUTSIDE this padded window.
    """
    n = len(d2_mag)
    if n == 0 or rs > re:
        return False
    rs_c = max(0, rs)
    re_c = min(n - 1, re)
    search_start = max(0, rs_c - search_pad)
    search_end = min(n - 1, re_c + search_pad)
    in_search = d2_mag[search_start : search_end + 1]
    if len(in_search) == 0:
        return False
    base_start = max(0, search_start - context_window)
    base_end = min(n, search_end + 1 + context_window)
    outside_pieces = []
    if base_start < search_start:
        outside_pieces.append(d2_mag[base_start:search_start])
    if search_end + 1 < base_end:
        outside_pieces.append(d2_mag[search_end + 1 : base_end])
    if not outside_pieces:
        return True
    outside = np.concatenate(outside_pieces)
    if len(outside) < 2:
        return True
    baseline = max(float(np.median(outside)), 1e-3)
    spike = float(np.max(in_search))
    return spike >= spike_ratio * baseline


def _extend_region_by_d2(d2_mag, rs, re, max_extend, baseline_factor=1.5):
    """Extend [rs, re] outward as long as |r''| stays elevated.

    Skeleton-level distortion often extends a few pixels BEYOND the
    strict spatial cluster: the polyline is no longer within tolerance
    of the other polyline out there, but its local geometry is still
    bent. Without this extension, those out-of-cluster noisy pixels
    survive into the output as a small "shelf" or jog next to the
    cleaned junction. We walk outward from the strict region on each
    side and grow it while |r''| exceeds ``baseline_factor`` x the
    baseline (median of values further outside).
    """
    n = len(d2_mag)
    if n == 0 or rs > re:
        return rs, re
    rs_c = max(0, rs)
    re_c = min(n - 1, re)
    # Baseline measured FAR from both the strict region and the
    # potential extension zone, so the extension can't bias it.
    far_left_end = max(0, rs_c - max_extend)
    far_right_start = min(n, re_c + 1 + max_extend)
    context = 8
    far_left_start = max(0, far_left_end - context)
    far_right_end = min(n, far_right_start + context)
    pieces = []
    if far_left_start < far_left_end:
        pieces.append(d2_mag[far_left_start:far_left_end])
    if far_right_start < far_right_end:
        pieces.append(d2_mag[far_right_start:far_right_end])
    if not pieces:
        return rs_c, re_c
    outside = np.concatenate(pieces)
    if len(outside) < 2:
        return rs_c, re_c
    baseline = max(float(np.median(outside)), 1e-3)
    threshold = baseline * baseline_factor
    rs_ext = rs_c
    for off in range(1, max_extend + 1):
        idx = rs_c - off
        if idx < 0:
            break
        if d2_mag[idx] > threshold:
            rs_ext = idx
        else:
            break
    re_ext = re_c
    for off in range(1, max_extend + 1):
        idx = re_c + off
        if idx >= n:
            break
        if d2_mag[idx] > threshold:
            re_ext = idx
        else:
            break
    return rs_ext, re_ext


def _interpolate(start, end, max_spacing):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    d = float(np.linalg.norm(end - start))
    if d <= max_spacing or max_spacing <= 0:
        return np.empty((0, 2), dtype=float)
    n_segments = int(np.ceil(d / max_spacing))
    t = np.linspace(0.0, 1.0, n_segments + 1)[1:-1]
    return start[None, :] + t[:, None] * (end - start)[None, :]


def _apply_repairs(polyline, repairs, interp_max_spacing=1.0):
    if not repairs:
        return polyline.copy()
    repairs = sorted(repairs, key=lambda r: r[0])
    pieces: List[NDArray] = []
    cursor = 0
    last_emitted: NDArray | None = None

    def push(arr):
        nonlocal last_emitted
        if len(arr) == 0:
            return
        if last_emitted is not None:
            bridge = _interpolate(last_emitted, arr[0], interp_max_spacing)
            if len(bridge) > 0:
                pieces.append(bridge)
        pieces.append(arr)
        last_emitted = arr[-1]

    for js, je, jp in repairs:
        if js > cursor:
            push(polyline[cursor:js])
        push(np.asarray(jp, dtype=float).reshape(1, 2))
        cursor = je + 1
    if cursor < len(polyline):
        push(polyline[cursor:])
    if not pieces:
        return np.empty((0, 2), dtype=polyline.dtype)
    return np.concatenate(pieces, axis=0).astype(polyline.dtype, copy=False)


class RepairConfig(TypedDict):

    junction_tol: float
    """
    Max distance for two polylines' points to be considered part of
    the same junction. ~2.5 px works for typical 8-connected skeletons.
    """

    stable_skip: int
    """
    Pixels to skip BEYOND the extended junction region before sampling
    for the tangent. Buffers against any residual distortion just
    outside the noisy zone. ~2 is usually right.
    """

    stable_sample: int
    """
    Number of pixels to sample for tangent estimation once past the
    skip buffer. Larger = more robust but smooths away genuine
    curvature in short approaches.
    """

    max_junction_region_length: int
    """
    Reject junctions where any polyline contributes more than this
    many consecutive points to the strict spatial cluster. Now a
    sanity net rather than the primary discriminator: the kink test
    below is what distinguishes real crossings from parallel
    coincidences. Useful default ~20-30 to catch closed-loop polylines
    whose start/end coincidence makes their cluster span the entire
    polyline.
    """

    min_tangent_spread_deg: float
    """
    Reject junctions where all approach tangents point in nearly the
    same direction (mod 180 deg). Backstop against tangent kisses
    that slip past the kink test. Default 15 deg accepts oblique X
    crossings (down to ~15 deg between strokes) while rejecting
    tangent kisses and parallel coincidences.
    """

    interp_max_spacing: float
    """
    After replacing a junction region with the junction point, the
    polyline is filled with interpolated points so that consecutive
    points are at most this far apart. Default 1.0 (one pixel) matches
    the skeleton spacing, so the polyline remains visually continuous
    and the downstream fitter doesn't see a gap.
    """

    min_output_polyline_length: int
    """
    After repair, drop polylines shorter than this (rare; happens if a
    polyline was almost entirely consumed by junctions).
    """

    min_curvature_spike_ratio: float
    """
    A cluster is treated as a real junction only if some polyline has
    ``max(|r''|)`` in a padded window around the cluster that is at
    least this many times the median |r''| baseline measured just
    OUTSIDE that window. Default 2.0 cleanly separates skeleton-level
    kinks (ratios of ~5-20x) from parallel coincidences (~1x).
    """

    curvature_context_window: int
    """
    Half-width (in pixels) of the |r''| context window used both for
    the kink test (where it sets the baseline measurement zone) and
    for the region-extension walk (where it caps how far the
    junction-affected region can grow on each side). Default 8.
    """

    cascade_gap: int
    """
    Two repairs on the same polyline are merged into one when their
    junction-affected index ranges are within this many polyline
    indices of each other (and their junction points are within
    ``cascade_max_jp_distance`` — see ``_merge_cascade_repairs``).
    This is an index gap, not a pixel distance, so it does NOT scale
    with stroke width. Default 3.
    """

    cascade_max_jp_distance: float
    """
    Distance guard (pixels) for cascade merging: two adjacent repairs
    are only merged when their solved junction points are within this
    distance, so a polyline threading two physically distinct
    junctions is not collapsed into one averaged point. A pixel
    distance — scales with stroke width. Default ~3 px at the
    reference scale.
    """


def repair_junctions(polylines, config: RepairConfig):
    if not polylines:
        return []
    polylines = [np.asarray(p, dtype=float) for p in polylines]
    meta, coords = gather_points(polylines)
    if len(coords) == 0:
        return list(polylines)
    point_poly_idx = [m[0] for m in meta]
    raw_clusters = spatial_clusters(coords, config["junction_tol"], point_poly_idx)
    clusters = []
    for c in raw_clusters:
        polys = {meta[i][0] for i in c}
        if len(polys) >= 2:
            clusters.append(c)
    per_poly_repairs = {pi: [] for pi in range(len(polylines))}

    # Pre-compute |r''| profile per polyline for the kink-presence test.
    d2_profiles = [_polyline_d2_mag(p, smoothing_window=3) for p in polylines]

    for cluster in clusters:
        poly_to_indices = {}
        for global_idx in cluster:
            pi, ii = meta[global_idx]
            poly_to_indices.setdefault(pi, []).append(ii)
        too_long = False
        for indices in poly_to_indices.values():
            if max(indices) - min(indices) + 1 > config["max_junction_region_length"]:
                too_long = True
                break
        if too_long:
            continue
        # KINK CHECK: distinguishes real crossings (sharp local turn in
        # at least one polyline) from "parallel coincidences" where two
        # smooth polylines happen to run within tol of each other.
        # Without this, raising max_junction_region_length to cover
        # legitimate X-crossings with multi-pixel-thick skeleton blobs
        # would also accept parallel runs.
        n_kinky = 0
        for pi, indices in poly_to_indices.items():
            rs_i, re_i = min(indices), max(indices)
            if _kink_present(
                d2_profiles[pi],
                rs_i,
                re_i,
                context_window=config["curvature_context_window"],
                spike_ratio=config["min_curvature_spike_ratio"],
            ):
                n_kinky += 1
        if n_kinky < 1:
            continue
        # Extend each polyline's junction-affected region by walking
        # outward as long as |r''| stays elevated. The skeleton-level
        # distortion at an X-crossing often spills a few pixels past
        # the strict spatial cluster, and leaving them in the output
        # produces a visible "shelf" or jog next to the cleaned point.
        extended_poly_indices = {}
        for pi, indices in poly_to_indices.items():
            rs_i, re_i = min(indices), max(indices)
            rs_ext, re_ext = _extend_region_by_d2(
                d2_profiles[pi],
                rs_i,
                re_i,
                max_extend=config["curvature_context_window"],
                baseline_factor=1.5,
            )
            extended_poly_indices[pi] = (rs_ext, re_ext)
        all_anchors = []
        all_tangents = []
        poly_to_region = {}
        for pi, indices in poly_to_indices.items():
            rs_ext, re_ext = extended_poly_indices[pi]
            # Build approaches from samples beyond the EXTENDED region,
            # so the tangent estimate is taken from genuinely smooth
            # polyline (not from pixels that are still inside the noise).
            ext_indices = list(range(rs_ext, re_ext + 1))
            arms = _build_approaches(
                polylines[pi],
                ext_indices,
                config["stable_skip"],
                config["stable_sample"],
            )
            if not arms:
                continue
            for _inner, _dir, anchor, tangent in arms:
                all_anchors.append(anchor)
                all_tangents.append(tangent)
            poly_to_region[pi] = (rs_ext, re_ext)
        if len(poly_to_region) < 2 or len(all_anchors) < 2:
            continue
        spread = _tangent_spread_deg(all_tangents)
        if spread < config["min_tangent_spread_deg"]:
            continue
        cluster_centroid = coords[cluster].mean(axis=0)
        jp = _solve_junction_point(all_anchors, all_tangents, fallback=cluster_centroid)
        max_dev = 5.0 * config["junction_tol"]
        if np.linalg.norm(jp - cluster_centroid) > max_dev:
            jp = cluster_centroid
        for pi, (rs, re) in poly_to_region.items():
            per_poly_repairs[pi].append((rs, re, jp))
    repaired = []
    cascade_gap = int(config.get("cascade_gap", 3))
    cascade_max_jp = float(config.get("cascade_max_jp_distance", 3.0))
    for pi, poly in enumerate(polylines):
        raw = per_poly_repairs[pi]
        merged = (
            _merge_cascade_repairs(
                raw,
                cascade_gap=cascade_gap,
                max_jp_distance=cascade_max_jp,
            )
            if len(raw) >= 2
            else raw
        )
        new_poly = _apply_repairs(
            poly, merged, interp_max_spacing=config["interp_max_spacing"]
        )
        if len(new_poly) >= config["min_output_polyline_length"]:
            repaired.append(new_poly)
    return repaired
