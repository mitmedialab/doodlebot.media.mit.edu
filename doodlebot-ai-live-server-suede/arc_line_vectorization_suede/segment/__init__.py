"""Split a skeleton into segments at branch points and visualize them.

A "segment" is a maximal run of degree-2 skeleton pixels bounded by node
pixels at each end. A node pixel is one of:
  - an endpoint: a skeleton pixel with exactly one 8-connected skeleton
    neighbour (degree 1).
  - a branch: a skeleton pixel with three or more 8-connected skeleton
    neighbours (degree >= 3). Adjacent branch pixels are merged into a
    single super-junction so that thick junctions in the skeleton don't
    spawn spurious tiny segments.

``segment_skeleton`` is a thin wrapper around ``trace_skeleton`` that also
applies an optional minimum-length filter. ``visualize_segments`` produces
a PNG where each segment is drawn in a distinct colour using golden-ratio
hue spacing so adjacent segments are easy to tell apart visually.

The ``Segment`` class runs the full pipeline:

    skeleton + binary
       |
       v
    [trace]          self.segments
       |
       v
    [fuse 1st pass]  self.fused        (with self.accepted, self.all)
       |             A* through binary; bridges noisy junctions.
       v
    [repair]         self.repaired
       |             Per-junction LS solve, replaces noisy regions.
       v
    [resolve         self.cleaned      (with self.collapsed_chromosomes)
     chromosomes]    Collapses skeletonization "centromere" artifacts
       |             where two strokes overlapped along a length.
       v
    [fuse 2nd pass]  self.joined       (with self.joined_accepted)
                     Pairwise tangent matching at clean junctions;
                     no path-finding, just cluster-and-match.
"""

from __future__ import annotations

from typing import TypedDict, List

import numpy as np
from numpy.typing import NDArray

from .trace import trace_skeleton
from .fusion import (
    fuse_segments,
    FuseConfig,
    fuse_post_repair,
    PostRepairFuseConfig,
)
from .repair import repair_junctions, RepairConfig
from .labels import (
    LabeledSegment,
    RawSegmentSpan,
    assign_raw_segment_labels,
)


def filter_short_polylines(
    polylines: List[np.ndarray],
    min_length: float = 3.0,
) -> List[np.ndarray]:
    """Drop polylines whose total arc length is below `min_length` pixels.

    These are almost always skeletonization artifacts at junctions.
    """
    out = []
    for p in polylines:
        if len(p) < 2:
            continue
        diffs = np.diff(p, axis=0)
        length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        if length >= min_length:
            out.append(p)
    return out


def segment_skeleton(
    skel: NDArray[np.bool_],
    min_length: float = 0.0,
) -> List[NDArray[np.float64]]:
    """Split a skeleton into polyline segments at every node (branch / endpoint).

    Each returned segment is an (N, 2) float array of (x, y) pixel
    coordinates. The first and last point of every segment is a node
    pixel; every interior point is degree-2 (exactly two 8-connected
    skeleton neighbours). Segments share their endpoint pixels with the
    other segments that meet at the same node — i.e. the segments form
    the edge set of a graph whose vertices are the node pixels.

    Args:
        skel: boolean skeleton image (True = ink).
        min_length: drop segments whose total arc length is below this
            value in pixels. Useful for filtering occasional junction
            noise. Default 0 keeps everything.
    """
    polys = trace_skeleton(skel.astype(bool))
    return (
        filter_short_polylines(polys, min_length=min_length)
        if min_length > 0
        else polys
    )


class Segment:
    class Config:
        class Segment(TypedDict):
            min_length: float

        class Fuse(FuseConfig):
            pass

        class Repair(RepairConfig):
            pass

        class PostRepairFuse(PostRepairFuseConfig):
            pass

    def __init__(
        self,
        skel: NDArray[np.bool_],
        binary: NDArray[np.bool_],
        segment: Config.Segment,
        fuse: Config.Fuse,
        repair: Config.Repair,
        post_repair_fuse: Config.PostRepairFuse,
    ):
        self.segments = segment_skeleton(skel, min_length=segment["min_length"])

        self.fused_pre_repair, self.accepted, self.all = fuse_segments(
            self.segments, binary=binary, config=fuse
        )
        self.repaired = repair_junctions(self.fused_pre_repair, config=repair)

        self.fused_post_repair, self.joined_accepted = fuse_post_repair(
            self.repaired, config=post_repair_fuse
        )

        # Back-pointer: every pixel of every final polyline gets the id
        # of the raw (trace.py output, post-min-length filter) segment
        # it came from, packed into ``RawSegmentSpan`` runs that tile
        # the final polyline. ``self.labeled_segments[k]`` corresponds
        # to ``self.fused_post_repair[k]``.
        self.labeled_segments = assign_raw_segment_labels(
            self.segments, self.fused_post_repair
        )
