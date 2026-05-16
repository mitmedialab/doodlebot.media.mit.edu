"""Post-process a skeleton to collapse small enclosed regions.

Some skeletonization implementations — notably `method="lee"` on certain
line shapes — leave behind tiny closed loops where the line failed to
collapse to 1-pixel width. Two common artifact shapes:

  1. "Double diagonal": two parallel diagonal runs of pixels share corner-
     only contact, sandwiching 1-pixel-wide enclosed regions between them.
  2. "Long thin bulge": a stretch where the line briefly thickens to 2
     pixels, producing a long but narrow enclosed region (high area, but
     never more than a couple of pixels thick anywhere).

This module provides a function that finds those enclosed regions and
fills them, then thins the result so the line ends up 1 pixel wide.

A region is collapsed if EITHER criterion is met:
  (a) its pixel area is below `max_hole_area` (catches tiny holes
      regardless of shape), OR
  (b) its maximum local thickness is at most `max_thin_thickness`
      (catches long-and-thin artifacts that pass the area test).

Local thickness is measured via the Euclidean distance transform:
`distance_transform_edt(region)` gives, for each interior pixel, the
distance to the nearest non-region pixel. The maximum of that field is
the radius of the largest disk that fits inside the region; we treat
that as half the local thickness. For a region that is k pixels thick
everywhere, the max distance-transform value is approximately k/2 (so
max_dt <= 1.0 catches up to ~2-pixel-thick regions).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as cc_label
from skimage.morphology import skeletonize

from typing import Literal, TypedDict

# 4-connectivity: only orthogonal neighbours count as connected. Crucially,
# this is the right choice for hole detection on a skeleton — a pair of
# skeleton pixels that meet only at a corner are 8-connected to each other,
# but the background pixels they sandwich are 4-isolated from the outside.
_CROSS_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def _enclosed_region_masks(skel: NDArray[np.bool_]) -> list[NDArray[np.bool_]]:
    """Return a list of boolean masks, one per enclosed background region.

    A region is "enclosed" when it is a 4-connected component of background
    pixels that does not touch the image border.
    """
    H, W = skel.shape
    if H == 0 or W == 0:
        return []
    background = ~skel
    labeled, n = cc_label(background, structure=_CROSS_4)
    if n == 0:
        return []
    border_labels: set[int] = set()
    border_labels.update(np.unique(labeled[0, :]).tolist())
    border_labels.update(np.unique(labeled[-1, :]).tolist())
    border_labels.update(np.unique(labeled[:, 0]).tolist())
    border_labels.update(np.unique(labeled[:, -1]).tolist())
    border_labels.discard(0)  # 0 is the foreground; not a real component
    return [labeled == lid for lid in range(1, n + 1) if lid not in border_labels]


def _max_thickness(mask: NDArray[np.bool_]) -> float:
    """Return the maximum local thickness of a region (in pixel units).

    Uses the Euclidean distance transform: for each interior pixel, the
    distance to the nearest non-region pixel; the maximum of that field
    is half the local thickness at the deepest interior point.
    """
    if not mask.any():
        return 0.0
    dt = distance_transform_edt(mask)
    return float(dt.max())


def find_enclosed_regions(
    skel: NDArray[np.bool_],
    max_area: int | None = None,
    max_thickness: float | None = None,
) -> list[NDArray[np.bool_]]:
    """Return masks of enclosed background regions matching the filters.

    A region passes if EITHER of the supplied filters matches:
      - `max_area`: area of the region in pixels is <= max_area
      - `max_thickness`: the region's maximum local thickness (via distance
        transform) is <= max_thickness

    If both are None, returns every enclosed region. If both are given,
    a region is included if either criterion matches (logical OR).
    """
    skel = skel.astype(bool)
    masks = _enclosed_region_masks(skel)
    if max_area is None and max_thickness is None:
        return masks

    out = []
    for m in masks:
        passes = False
        if max_area is not None and int(m.sum()) <= max_area:
            passes = True
        if not passes and max_thickness is not None:
            if _max_thickness(m) <= max_thickness:
                passes = True
        if passes:
            out.append(m)
    return out


class CollapseConfig(TypedDict):
    max_hole_area: int | None
    max_thin_thickness: float | None
    reskeletonize: bool
    skeletonize_method: Literal["lee", "zhang"]


def collapse_small_holes(
    skel: NDArray[np.bool_],
    config: CollapseConfig,
) -> NDArray[np.bool_]:
    """Fill collapsible enclosed background regions and (optionally) re-thin.

    A region is filled if EITHER of the following holds:
      (a) its area in pixels is <= `max_hole_area`, OR
      (b) its maximum local thickness is <= `max_thin_thickness`.

    Defaults catch the common Lee-method artifacts:
      - `max_hole_area=4`: handles "double diagonal" 1-pixel holes and
        small elbow artifacts.
      - `max_thin_thickness=1.0`: handles long-and-thin bulges where the
        line briefly doubles up; this catches anything up to ~2 pixels
        thick everywhere, regardless of length.

    The skeletonizer's topology preservation means strokes that ran
    through a filled region remain connected, so the line continues
    correctly through the patch.

    Set either parameter to 0 / `None` to disable that criterion.
    """
    skel = skel.astype(bool)
    if skel.size == 0:
        return skel.copy()

    masks = find_enclosed_regions(
        skel,
        max_area=config.get("max_hole_area"),
        max_thickness=config.get("max_thin_thickness"),
    )
    filled = skel.copy()
    for mask in masks:
        filled[mask] = True

    if config.get("reskeletonize") and masks:
        filled = skeletonize(filled, method=config.get("skeletonize_method"))

    return filled
