"""Per-pixel labeling: map every binary stroke pixel to a skeleton pixel.

For downstream tracing — "this pixel of the original image is what
became this vectorized command" — we need a complete forward map from
binary pixels to the final skeleton. ``skimage.skeletonize`` doesn't
expose which pixels it absorbed into which surviving pixel during
thinning (the Zhang-Suen and Lee variants iterate to a fixed point
and report only the result), so we recover the mapping after the fact
by geodesic flooding through the stroke mask.

The mechanism is one call to ``skimage.segmentation.watershed`` with:

* a *flat* elevation map (zero everywhere) — we don't want intensity
  gradients to bias the partition; we want pure geodesic distance,
* one *marker* per skeleton pixel,
* the binary stroke mask as the watershed mask, so the flood is
  confined to the stroke.

With a flat elevation, watershed degenerates to a multi-source BFS
that assigns every reachable pixel to its nearest marker by
in-stroke geodesic distance. Two arms meeting at a junction therefore
partition the surrounding ink along the actual stroke topology rather
than along a Euclidean perpendicular bisector that can leak across
junctions on thick ink.

Pixels outside the binary mask get label ``-1``. Stroke pixels in
binary components with no skeleton pixel at all (rare — a small
fragment that the thinner emptied entirely) also stay ``-1``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage.segmentation import watershed


def label_skeleton_pixels(
    binary: NDArray[np.bool_],
    skel: NDArray[np.bool_],
) -> NDArray[np.int32]:
    """Return an (H, W) int32 map where ``labeling[y, x]`` is the
    flat-index of the skeleton pixel that ``(y, x)`` belongs to.

    Flat-index ordering follows ``np.where(skel)`` row-major: i.e.
    ``label = k`` corresponds to the k-th True pixel of ``skel``
    when scanned in row-major order. ``label = -1`` means "no
    skeleton pixel was reachable from this position through the
    binary mask" (always the case for pixels outside ``binary``).

    Arguments:
        binary: foreground stroke mask. The flood is confined to
            True pixels of this mask.
        skel: final skeleton (e.g. ``Skeletonize.uncrossed``). Each
            True pixel becomes a watershed seed.
    """
    binary_b = binary.astype(bool)
    skel_b = skel.astype(bool)

    # Seeds: one positive int per skeleton pixel, in row-major order.
    sy, sx = np.where(skel_b)
    markers = np.zeros(binary_b.shape, dtype=np.int32)
    # watershed uses 0 = background / unmarked, so seed IDs start at 1.
    markers[sy, sx] = np.arange(1, len(sy) + 1, dtype=np.int32)

    # Skeleton seeds that sit outside the binary mask (shouldn't happen
    # for a well-formed pipeline, but defensively defendable) would
    # never be reachable; watershed handles that fine but downstream
    # callers may assume skel ⊆ binary. We don't enforce that here.
    mask = binary_b

    # Flat elevation -> pure geodesic BFS from each marker; 8-connected
    # so diagonal neighbours count as adjacent (matches the rest of the
    # pipeline's skeleton-graph topology).
    flooded = watershed(
        image=np.zeros(binary_b.shape, dtype=np.uint8),
        markers=markers,
        mask=mask,
        connectivity=2,
    )

    # Map watershed output (1-based marker IDs, 0 = unreached) back to
    # the 0-based flat-index convention promised in the docstring.
    out = flooded.astype(np.int32) - 1  # so seed-1 -> 0, ..., unreached -> -1
    return out
