"""Detect filled-in shapes (eyes, pupils, dots) and resolve them by
swapping their tiny medial-axis skeleton for the shape's boundary.

Why
---
Morphological thinning of a filled-in circle collapses the whole disk
to a single pixel — or a tight 2-5 pixel cluster on noisy hand-drawn
fills. Downstream segmentation either drops it as a sub-minimum-length
polyline, or hands it on as a degenerate "stroke" that the vectorizer
can't reasonably fit. The visual intent (a drawn dot or pupil) is lost.

This module identifies those filled regions in the binarized image and
rewrites the skeleton inside each one with the region's 1-pixel-wide
outer boundary. Segmentation then sees a clean closed loop where the
filled disk used to be, and the vectorizer fits it as a `Circle` (or a
short arc chain on irregular fills).

Detection
---------
Two signals combined, applied to every 8-connected component of the
binary mask:

* ``skeleton_pixels <= max_skeleton_pixels`` — the component's skeleton
  must be a small cluster (a real filled blob's medial axis collapses
  to a few pixels; a normal stroke skeletonizes to dozens or more,
  one per pixel of length).

* ``area / skeleton_pixels >= min_fill_ratio`` — the component must be
  substantially "filler" relative to its skeleton. A regular stroke
  has ``area / skel ≈ stroke_width`` (each skeleton pixel covers one
  column across the stroke), so a ratio threshold a few times larger
  than the stroke width cleanly distinguishes a fill from a stroke.

Both signals are needed: ``max_skeleton_pixels`` alone would catch
very short stroke fragments (a 3-pixel tick mark); ``min_fill_ratio``
alone would catch any sufficiently fat blob even if its skeleton was
long enough to be a real stroke through it.

Resolution
----------
For each accepted region, the boundary is computed as
``region & ~erode(region)`` — the outer ring of pixels one wide. The
old skeleton pixels falling inside the region are cleared, and the
boundary pixels are set. Downstream tracing then walks the ring as a
closed-loop polyline.
"""

from __future__ import annotations

from typing import List, NamedTuple, Tuple, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion, label as _ndi_label
from skimage.morphology import skeletonize as _skeletonize

_CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)


class EyeDetectConfig(TypedDict):
    max_skeleton_pixels: int
    """Upper bound on skeleton pixels falling inside the region. A
    filled circle/disk collapses to 1-5 skeleton pixels; a regular
    stroke has one skeleton pixel per pixel of length, so even short
    strokes exceed this. Scales linearly with stroke width."""

    min_fill_ratio: float
    """Region's pixel area must be at least this many times its
    contained skeleton-pixel count. A regular stroke has
    ``area / skeleton ~ stroke_width`` (each skeleton pixel covers
    one column across the stroke), so values above the stroke width
    cleanly separate fills (ratio 10-50x) from strokes (~3-5x).
    Dimensionless — does NOT scale with stroke width."""

    min_area: int
    """Reject regions smaller than this. Filters out single-pixel
    speckles and isolated 2x2 blobs that look filled but are too small
    to carry visual meaning. Scales with stroke width squared."""


class FilledRegion(NamedTuple):
    component_id: int
    """The component's id in ``EyeDetectResult.components`` (the
    connected-components label image)."""

    mask: NDArray[np.bool_]
    """Full pixel mask of this filled region (same shape as binary)."""

    boundary: NDArray[np.bool_]
    """The 1-pixel outer boundary, computed as
    ``mask & ~binary_erosion(mask)``."""

    area: int
    """Number of foreground pixels in the region."""

    skeleton_pixel_count: int
    """Number of skeleton pixels falling inside the region. The reason
    the region was even considered: small absolute count + high
    area-to-skeleton ratio is the fingerprint of a filled disk."""

    centroid: Tuple[float, float]
    """(y, x) centroid of the region. Used purely for diagnostics /
    visualization."""


class EyeDetectResult(NamedTuple):
    components: NDArray[np.intp]
    """Connected-components label image of ``binary``. ``components ==
    region.component_id`` recovers a specific region's mask."""

    n_components: int
    filled_regions: List[FilledRegion]


def detect_filled_regions(
    binary: NDArray[np.bool_],
    skel: NDArray[np.bool_],
    config: EyeDetectConfig,
) -> EyeDetectResult:
    """Find filled-in shapes in ``binary`` that should be re-skeletonized
    as their boundary rather than as their collapsed medial axis.

    Arguments:
        binary: foreground mask. Each 8-connected component is tested.
        skel: pre-cleaned 1-px skeleton aligned to ``binary``. Used
            only to count skeleton pixels per region (the area/skeleton
            ratio test).
        config: detection parameters; see ``EyeDetectConfig`` for the
            meaning of each field.
    """
    binary_b = binary.astype(bool)
    skel_b = skel.astype(bool)
    components, n = _ndi_label(binary_b, structure=_CONNECTIVITY_8)
    components = np.asarray(components, dtype=np.intp)
    n = int(n)

    max_skel = int(config["max_skeleton_pixels"])
    min_ratio = float(config["min_fill_ratio"])
    min_area = int(config["min_area"])

    filled_regions: List[FilledRegion] = []
    for cid in range(1, n + 1):
        mask = components == cid
        area = int(mask.sum())
        if area < min_area:
            continue
        n_skel = int((mask & skel_b).sum())
        # n_skel == 0 means the skeletonizer dropped every interior
        # pixel — extremely unusual but defensive. Treat as "not a
        # fill we can characterize" and skip.
        if n_skel == 0:
            continue
        if n_skel > max_skel:
            continue
        if area / float(n_skel) < min_ratio:
            continue
        # 1-pixel boundary. First take the morphological-gradient ring
        # (``mask & ~erode(mask)``), which is one-pixel-thick on convex
        # convex shapes but can pick up 2-pixel shoulders on irregular
        # hand-drawn fills — and those shoulders create degree-3 nodes
        # downstream that fragment the trace into tiny pieces filtered
        # out by ``min_length``. Re-skeletonizing the band collapses
        # any wide stretches back to one pixel so trace sees a single
        # clean closed loop.
        eroded = binary_erosion(mask, structure=_CONNECTIVITY_8)
        boundary_band = mask & ~eroded
        boundary = _skeletonize(boundary_band).astype(bool)
        ys, xs = np.where(mask)
        centroid = (float(ys.mean()), float(xs.mean()))
        filled_regions.append(
            FilledRegion(
                component_id=cid,
                mask=mask,
                boundary=boundary,
                area=area,
                skeleton_pixel_count=n_skel,
                centroid=centroid,
            )
        )

    return EyeDetectResult(
        components=components,
        n_components=n,
        filled_regions=filled_regions,
    )


def resolve_filled_regions(
    skel: NDArray[np.bool_],
    detection: EyeDetectResult,
) -> NDArray[np.bool_]:
    """Replace each detected filled region's interior skeleton with its
    1-pixel outer boundary.

    Returns a new boolean array; ``skel`` is not mutated.
    """
    result = skel.astype(bool).copy()
    for region in detection.filled_regions:
        # Wipe the collapsed-medial-axis skeleton inside the fill.
        result[region.mask] = False
        # Write in the boundary ring. Downstream tracing now sees a
        # clean closed loop where the disk used to be.
        result[region.boundary] = True
    return result
