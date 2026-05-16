"""Detect ribbon-collapse artifacts in a binarized line drawing.

Two strokes that briefly share ink — by truly crossing over, by being
tangent, or by aligning for a short segment — produce a characteristic
skeletonization artifact: the merged region of ink is wide enough that
the thinning algorithm is forced to draw the skeleton as a single line
through the middle, even though the original drawing has two distinct
strokes there. After the merged region ends, the skeleton splits back
into two lines via a Y-junction on each side. The whole pattern looks
like ``)―(`` in the skeleton: a single connecting segment between two
Y-junctions. The literature calls this a "ribbon collapse" (Bessmeltsev
et al., PolyVector Fields, SIGGRAPH 2019), a "stroke merge" (Favreau et
al., SIGGRAPH 2016), or informally a "chromosome".

This module detects those artifacts so downstream code can resolve them
(pair the four arms, rewrite the skeleton as two distinct strokes).

Why this approach instead of the obvious ones
---------------------------------------------
Global stroke-half-width thresholding does not work when a drawing
contains both thin and thick strokes: the threshold is set by the
thicker strokes and the thinner-stroke ribbon collapses are too narrow
to qualify. The fix is a per-pixel LOCAL stroke half-width estimated
from nearby skeleton pixels.

Counting how many non-fat ink components touch a fat region also does
not work: the rest of the drawing's ink is one globally connected
blob, so the count is always 1. The fix is to count skeleton-arm
exits via 8-connected components of the skeleton in a ring around the
fat region, since the skeleton's 1-pixel arms are naturally separated
by background.

A degree-4 fat region is necessary but not sufficient: four strokes
meeting at a single point look identical at the topology level. Two
further checks distinguish actual ribbon collapses from point
junctions:

- **Stroke pairing**: the four arm tangents must pair into two
  anti-parallel pairs (two strokes continuing through). Wheel-spoke
  junctions fail this because the arms radiate in 4 directions.
- **Merged segment length**: the skeleton trapped inside the fat
  region must be a meaningful segment, not just a junction vertex.
  This is the most direct test for the ``)―(`` pattern: a ribbon
  collapse has a long single-line segment inside the fat region; a
  4-way junction has only a junction node and its immediate
  neighbors there.

Pipeline
--------
1. Compute `dt` (distance transform) and `skel` (skeleton).
2. Compute LOCAL tau per pixel: mean DT over skeleton pixels in a
   window. Adapts to varying stroke widths.
3. Threshold fat pixels: `dt > fat_ratio * local_tau` AND ink.
4. GROUP nearby fat fragments by dilation-then-cc-label, so a single
   ribbon collapse whose fat region has split into two disconnected
   peaks counts as one logical region.
5. For each group, count SKELETON-EXIT degree (8-connected components
   of skeleton in a ring outside the dilated group).
6. For degree-4 groups, compute the best STROKE PAIRING score.
7. Apply chromosome filters: pairing score above threshold AND
   skeleton-length-inside-fat above threshold.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    label as _ndi_label,
    uniform_filter,
)

_CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)


# ============================================================================
# Config & output types
# ============================================================================


class DetectConfig(TypedDict):
    local_tau_radius: int
    """Window half-size for local tau. ~40 px works on 1024-px sketches."""

    fat_ratio: float
    """Pixel is fat iff dt > fat_ratio * local_tau. ~1.3 is permissive."""

    min_fat_area: int
    """Drop fat groups below this combined pixel count."""

    group_dilate: int
    """Dilation iterations applied to fat_mask before grouping. Fat
    fragments whose dilated extents touch are merged into one logical
    region. Should be at least the maximum distance you expect between
    fragments of one crossing. ~12-20 px works."""

    skel_ring_dilate: int
    """Additional dilation past `group_dilate` when looking for
    skeleton arms in the exit ring."""

    pairing_tangent_steps: int
    """How far along each arm to look for its outward tangent. Larger
    is more robust to noise but less local. 6-10 works."""

    pairing_threshold: float
    """Minimum allowed sum of (-dot products) over the best pairing
    of 4 arms into 2 pairs. Both pairs perfectly anti-parallel sum
    to 2.0; 1.0 means an average pair angle of 120 deg. ~1.2 is a
    reasonable cutoff for 'the arms continue rather than meet'."""

    min_chromosome_skel_length: int
    """Minimum number of skeleton pixels that must sit INSIDE the
    fat region for the detection to be reported as a ribbon-collapse
    crossing. A true ribbon collapse forces the skeleton through the
    merged region as a single line whose length is proportional to
    the chromosome's extent; values of 15+ pixels on a 1024-px sketch
    cleanly separate real ribbon collapses from 4-way point junctions
    (where only the junction vertex itself sits in the fat region)."""


class Crossing(NamedTuple):
    fat_pixels: NDArray[np.int32]
    """(N, 2) integer (y, x) coordinates of the fat pixels in this group."""

    group_id: int
    """The group's label in ``DetectResult.group_labels``. Lets the
    resolver retrieve the dilated group region (the area inside which
    the merged skeleton segment lives) without recomputing the
    dilation."""

    degree: int
    """Number of distinct skeleton arms exiting the group."""

    peak_dt: float
    """Maximum DT in the fat pixels."""

    centroid: Tuple[float, float]
    """(y, x) centroid."""

    pairing_score: Optional[float]
    """Best pairing score; only computed for degree-4 groups. None
    otherwise."""

    chromosome_skel_length: int
    """Number of skeleton pixels that fall inside the fat region.
    A measure of the merged-segment length. Tiny for 4-way junctions
    (just the junction vertex), large for ribbon collapses."""

    arm_endpoints: NDArray[np.int32]
    """(degree, 2) array of (y, x) coordinates: each arm's proximal
    end (the skeleton pixel closest to the dilated group region).
    These pixels lie OUTSIDE the dilated group, so they survive when
    the resolver erases the group's interior, and new replacement
    lines can be drawn between paired endpoints to restore the two
    crossing strokes."""

    arm_tangents: NDArray[np.float64]
    """(degree, 2) array of outward unit tangents, one per arm,
    pointing from the group's centroid toward the arm."""

    arm_pairing: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]
    """The best pairing of the 4 arms into 2 anti-parallel pairs,
    given as a pair of (arm_index, arm_index) tuples (e.g.
    ``((0, 2), (1, 3))``). The resolver draws one replacement line
    per pair, between the two arms' endpoints. Only set for degree-4
    groups; ``None`` otherwise."""


class DetectResult(NamedTuple):
    local_tau: NDArray[np.float64]
    dt: NDArray[np.float64]
    skel: NDArray[np.bool_]
    fat_mask: NDArray[np.bool_]
    group_labels: NDArray[np.intp]
    all_regions: List[Crossing]
    crossings: List[Crossing]
    """Degree-4 groups that pass the stroke-pairing check."""


# ============================================================================
# Internals
# ============================================================================


def _label_tuple(
    inp: NDArray,
    structure: Optional[NDArray] = None,
) -> Tuple[NDArray[np.intp], int]:
    result = _ndi_label(inp, structure=structure, output=None)
    labels, n = result  # type: ignore[misc]
    return np.asarray(labels, dtype=np.intp), int(n)


def _local_tau(
    skel: NDArray[np.bool_],
    dt: NDArray[np.float64],
    radius: int,
) -> NDArray[np.float64]:
    """Per-pixel mean DT over skeleton pixels in a (2r+1) window.

    Implemented as the ratio of two `uniform_filter` outputs: one over
    skel*dt (numerator), one over skel (denominator). Where the window
    contains no skeleton pixels, falls back to the global median.
    """
    skel_f = skel.astype(np.float64)
    window = 2 * radius + 1
    weighted = uniform_filter(skel_f * dt, size=window, mode="constant", cval=0.0)
    count = uniform_filter(skel_f, size=window, mode="constant", cval=0.0)
    fallback = float(np.median(dt[skel])) if skel.any() else 1.0
    return np.where(count > 0, weighted / np.maximum(count, 1e-12), fallback)


def _arm_tangents_and_endpoints(
    arm_labels: NDArray[np.intp],
    n_arms: int,
    centroid: Tuple[float, float],
    steps: int,
) -> Tuple[NDArray[np.float64], NDArray[np.int32]]:
    """For each skeleton arm, compute the outward tangent (unit vector)
    AND the proximal endpoint (the arm's skeleton pixel closest to the
    group centroid).

    Returns:
        tangents: ``(n_arms, 2)`` float array of unit vectors pointing
            from centroid toward the arm.
        endpoints: ``(n_arms, 2)`` integer array of (y, x) coordinates
            of each arm's proximal end. These pixels lie just outside
            the dilated group region, on the original skeleton, and
            serve as connection points for resolver-drawn lines.

    The tangent for each arm is the direction from the group centroid
    to the centroid of the arm's first ``steps`` skeleton pixels
    (ordered by distance to the group centroid). The endpoint is the
    single closest pixel.
    """
    cy, cx = centroid
    tangents = np.zeros((n_arms, 2), dtype=np.float64)
    endpoints = np.zeros((n_arms, 2), dtype=np.int32)
    for arm_id in range(1, n_arms + 1):
        ys, xs = np.where(arm_labels == arm_id)
        idx = arm_id - 1
        if len(ys) == 0:
            continue
        d2 = (ys - cy) ** 2 + (xs - cx) ** 2
        order = np.argsort(d2)
        head = order[: max(1, min(steps, len(order)))]
        head_y = float(ys[head].mean())
        head_x = float(xs[head].mean())
        vec = np.array([head_y - cy, head_x - cx], dtype=np.float64)
        n = float(np.linalg.norm(vec))
        if n >= 1e-9:
            tangents[idx] = vec / n
        # The proximal endpoint is the single closest pixel to the
        # centroid. It is guaranteed to be in `arm_skel`, i.e. on the
        # original skeleton and outside the dilated group region.
        endpoints[idx, 0] = int(ys[order[0]])
        endpoints[idx, 1] = int(xs[order[0]])
    return tangents, endpoints


def _best_pairing(
    tangents: NDArray[np.float64],
) -> Tuple[float, Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Best score AND the winning pairing over the 3 ways to pair 4
    unit vectors into 2 pairs.

    Each pair contributes ``-dot(t_i, t_j)``: 1.0 if anti-parallel
    (the arms continue across the group as one stroke), -1.0 if
    parallel (same direction; bad), 0.0 if perpendicular. Score is in
    ``[-2, 2]``; ``>= 1.5`` ≈ both pairs at least 135 deg apart;
    ``>= 1.2`` ≈ both pairs at least 120 deg.
    """
    assert tangents.shape == (4, 2)
    matchings = [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    ]
    best_score = -np.inf
    best_match = matchings[0]
    for match in matchings:
        (a, b), (c, d) = match
        s = -float(np.dot(tangents[a], tangents[b])) - float(
            np.dot(tangents[c], tangents[d])
        )
        if s > best_score:
            best_score = s
            best_match = match
    return float(best_score), best_match


# ============================================================================
# Detection
# ============================================================================


def detect_crossings(
    binary: NDArray[np.bool_],
    config: DetectConfig,
    skel: NDArray[np.bool_],
) -> DetectResult:
    """Detect ribbon-collapse crossings in ``binary``.

    See module docstring for the algorithm.

    Arguments:
        binary: the binarized image (True where ink is present).
        config: detection parameters; see ``DetectConfig`` for fields.
        skel: pre-computed skeleton to use instead of running
            ``skimage.morphology.skeletonize(binary)`` internally. Pass
            this when you have a CLEANED skeleton (e.g. after
            ``collapse_small_holes``) so that the arm endpoints land
            on actual cleaned-skeleton pixels — important if the
            resolver will then run on the cleaned skeleton.
    """
    binary_b = binary.astype(bool)
    dt = cast(NDArray[np.float64], distance_transform_edt(binary_b))

    local_tau = _local_tau(skel, dt, config["local_tau_radius"])
    fat_mask = binary_b & (dt > config["fat_ratio"] * local_tau)

    # GROUPING: dilate fat_mask, then connected components on the
    # dilation. Nearby fragments collapse into one group.
    group_dilate = int(config["group_dilate"])
    dilated_fat = binary_dilation(fat_mask, iterations=group_dilate)
    group_labels, n_groups = _label_tuple(dilated_fat, structure=_CONNECTIVITY_8)

    min_area = int(config["min_fat_area"])
    skel_ring_dilate = int(config["skel_ring_dilate"])
    ring_dilate = group_dilate + skel_ring_dilate

    pairing_steps = int(config["pairing_tangent_steps"])
    pairing_threshold = float(config["pairing_threshold"])
    min_chrom_length = int(config["min_chromosome_skel_length"])

    all_regions: List[Crossing] = []
    crossings: List[Crossing] = []

    for gid in range(1, n_groups + 1):
        group_dilated_mask = group_labels == gid
        fat_in_group = fat_mask & group_dilated_mask
        area = int(fat_in_group.sum())
        if area < min_area:
            continue

        # Skeleton arms in the exit ring. The ring is a band outside
        # the dilated group region. Skel pixels inside the dilated
        # region are excluded (they're either fat or within the fat's
        # blur halo). Result: each clean arm appears as one 8-connected
        # component, since arms are 1px wide and separated by background.
        arm_region = binary_dilation(fat_in_group, iterations=ring_dilate)
        arm_skel = arm_region & ~group_dilated_mask & skel
        arm_labels, n_arms = _label_tuple(arm_skel, structure=_CONNECTIVITY_8)

        ys, xs = np.where(fat_in_group)
        pixels = np.stack([ys, xs], axis=1).astype(np.int32)
        peak_dt = float(dt[fat_in_group].max())
        centroid = (float(ys.mean()), float(xs.mean()))

        # Length of the merged segment: skeleton pixels INSIDE the fat
        # region. Distinguishes ribbon collapses (long single-line
        # segment forced through merged ink) from 4-way point junctions
        # (just the junction vertex inside fat).
        chrom_skel_len = int((fat_in_group & skel).sum())

        # Compute arm tangents and endpoints for any group with >=1
        # detected arm; needed for visualization and (when degree==4)
        # for the resolver to draw replacement line segments.
        if n_arms > 0:
            arm_tangents, arm_endpoints = _arm_tangents_and_endpoints(
                arm_labels,
                n_arms,
                centroid,
                pairing_steps,
            )
        else:
            arm_tangents = np.zeros((0, 2), dtype=np.float64)
            arm_endpoints = np.zeros((0, 2), dtype=np.int32)

        pairing_score: Optional[float] = None
        arm_pairing: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        passed_pairing = False
        if n_arms == 4:
            pairing_score, arm_pairing = _best_pairing(arm_tangents)
            passed_pairing = pairing_score >= pairing_threshold

        region = Crossing(
            fat_pixels=pixels,
            group_id=gid,
            degree=n_arms,
            peak_dt=peak_dt,
            centroid=centroid,
            pairing_score=pairing_score,
            chromosome_skel_length=chrom_skel_len,
            arm_endpoints=arm_endpoints,
            arm_tangents=arm_tangents,
            arm_pairing=arm_pairing,
        )
        all_regions.append(region)
        if passed_pairing and chrom_skel_len >= min_chrom_length:
            crossings.append(region)

    return DetectResult(
        local_tau=local_tau,
        dt=dt,
        skel=skel,
        fat_mask=fat_mask,
        group_labels=group_labels,
        all_regions=all_regions,
        crossings=crossings,
    )


# ============================================================================
# Resolution
# ============================================================================


def resolve_crossings(
    skel: NDArray[np.bool_],
    detection: DetectResult,
) -> NDArray[np.bool_]:
    """Rewrite the skeleton at each detected crossing to undo the
    ribbon-collapse artifact.

    For each detected crossing, the merged skeleton segment that the
    thinning algorithm wrote through the fat region is erased and
    replaced by two new straight-line segments — one per paired pair of
    arms. The new lines start from the original arm endpoints (which
    lie just OUTSIDE the dilated group region and so are preserved
    when the group interior is erased), so the surviving arm skeleton
    connects to the replacement lines seamlessly.

    The result has two non-intersecting topological paths through each
    crossing region, recovering the original two-stroke topology that
    skeletonization had collapsed. Downstream segment tracing will now
    follow each stroke through the crossing as a single curve.

    Arguments:
        skel: the cleaned skeleton (one-pixel-wide ink). Modified
            crossings will be written into a copy; ``skel`` is not
            mutated.
        detection: the output of ``detect_crossings``. The
            ``crossings`` list provides the fat pixels, arm endpoints,
            and pairings used by the resolver.

    Returns:
        A new boolean array of the same shape as ``skel`` with each
        detected crossing rewritten. Crossings with no pairing
        (``arm_pairing is None``) are left alone.

    Notes:
        - Straight Bresenham lines are used between paired endpoints.
          For most crossings this is geometrically faithful (the two
          strokes either cross at the centroid or run nearly parallel
          through it). For strongly curved strokes a Bezier or
          spline-based connection would be more accurate, but the
          downstream pipeline typically re-fits curves to the resolved
          skeleton, so the straight-line approximation is sufficient
          here.
        - Erasing the dilated group region (rather than just the fat
          pixels) is important: the merged skeleton segment can extend
          slightly beyond the fat-thresholded pixels into the
          "shoulders" where the DT dips just below the fat threshold
          but the strokes are still merged. The dilated group region
          captures that shoulder zone.
    """
    # Local imports keep the heavy skimage.draw module optional for
    # users who only need detection.
    from skimage.draw import line as _bresenham

    result = skel.copy()
    group_labels = detection.group_labels

    for crossing in detection.crossings:
        if crossing.arm_pairing is None:
            continue

        # Erase the merged skeleton segment(s). The dilated group
        # region (from group_labels) covers every pixel within
        # group_dilate of any fat pixel in this crossing — which is
        # exactly the area inside which the thinning algorithm wrote
        # the merged centerline.
        group_region = group_labels == crossing.group_id
        result[group_region] = False

        # Draw the two replacement segments, one per paired pair.
        H, W = result.shape
        for i, j in crossing.arm_pairing:
            y0, x0 = int(crossing.arm_endpoints[i, 0]), int(
                crossing.arm_endpoints[i, 1]
            )
            y1, x1 = int(crossing.arm_endpoints[j, 0]), int(
                crossing.arm_endpoints[j, 1]
            )
            rr, cc = _bresenham(y0, x0, y1, x1)
            # Defensive: clip out anything that would land out of bounds.
            valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            result[rr[valid], cc[valid]] = True

    return result
