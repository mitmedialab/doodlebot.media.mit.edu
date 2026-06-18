"""Map every pixel of the final segmentation back to a raw segment.

A "raw segment" here is one polyline emitted by ``trace_skeleton``
after the short-polyline filter — i.e. the first thing the segment
pipeline produces, before fusion, repair, and the second fusion pass.

The downstream stages need to be able to answer "this pixel of the
final polyline came from which raw segment?". Maintaining that mapping
through the fuse/repair/fuse cascade as an explicit per-pixel label
array carried alongside each polyline is feasible but invasive — every
helper that concatenates, splices, or interpolates polyline pixels
would need a parallel branch for the label array.

We do it after the fact instead, by nearest-neighbour lookup against
the raw segments. Every final-polyline pixel takes the raw-segment id
of its spatially closest raw pixel; consecutive pixels with the same
id are then compressed into ``RawSegmentSpan`` runs that tile the
final polyline (no gaps, no overlap by construction).

This works because the transformations that produce the final
polylines move pixels by at most O(stroke-width):

* Fusion concatenates raw polylines and splices in a thin connecting
  path through ink — the bridge pixels sit between the two adjoining
  raw segments' endpoints, so each bridge pixel is closer to its
  side's raw segment than to the other.
* Repair replaces a junction-affected index range with the LS-solved
  junction point plus interpolated bridge points — those new pixels
  sit near the original junction cluster, so the nearest raw pixel is
  in the raw segment that contributed the cluster.
* Post-repair fusion is again a concatenation, with no new pixels.

Bridge pixels that fall exactly on a perpendicular bisector between
two raw segments tip arbitrarily but consistently, which is fine —
the resulting span boundary lands a pixel either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree


@dataclass(frozen=True)
class RawSegmentSpan:
    """A contiguous run of final-polyline pixels that all map to the
    same raw segment.

    ``start`` is inclusive and ``end`` is exclusive (Python-slice
    convention): ``final_polyline[span.start:span.end]`` is the range
    of points credited to ``raw_segment_id``.

    The spans for a single final polyline tile ``[0, len(polyline))``
    with no gaps and no overlap — together they reconstruct the
    polyline exactly.

    ``raw_start`` and ``raw_end`` are INCLUSIVE on both ends — the
    first and last raw-segment indices the span touches, walking the
    final polyline in its native order. If ``raw_end >= raw_start``
    the contribution walks the raw segment forward; if
    ``raw_end < raw_start`` it walks it backward (fusion can splice a
    raw segment in reverse). To extract the raw-segment slice
    regardless of direction, use
    ``raw_segments[id][min(raw_start, raw_end) : max(raw_start, raw_end) + 1]``.
    Keeping these inclusive avoids the ``-1`` edge case a half-open
    reverse interval would land at when ending at raw index 0.
    """

    raw_segment_id: int
    start: int
    end: int
    raw_start: int
    raw_end: int


@dataclass
class LabeledSegment:
    """A final polyline plus its decomposition into raw-segment spans.

    ``raw_ids[i]`` and ``raw_indices[i]`` give the raw segment and the
    index within that raw segment that final-polyline pixel ``i`` was
    assigned to. They're published alongside the compressed ``spans``
    because downstream stages (command labeling) need to slice partial
    raw-segment ranges that don't align to span boundaries.
    """

    points: NDArray[np.float64]
    spans: List[RawSegmentSpan]
    raw_ids: NDArray[np.int32]
    raw_indices: NDArray[np.int32]


def assign_raw_segment_labels(
    raw_segments: List[NDArray[np.float64]],
    final_segments: List[NDArray[np.float64]],
) -> List[LabeledSegment]:
    """For each final segment, decide which raw segment each of its
    pixels came from, and compress consecutive same-id pixels into
    spans.

    Arguments:
        raw_segments: ``trace_skeleton`` output (the polylines the rest
            of the segment pipeline consumes as input). The ``i``-th
            entry's id is ``i``.
        final_segments: the post-fuse-post-repair-post-fuse polylines
            ``Segment.fused_post_repair`` exposes.

    Returns: one ``LabeledSegment`` per entry in ``final_segments``,
    in the same order. A final segment with zero points yields a
    ``LabeledSegment`` with an empty ``spans`` list.
    """
    # Gather every raw-segment pixel into a single KDTree, remembering
    # which raw segment each pixel came from AND that pixel's local
    # index in its raw segment. cKDTree's query is what turns
    # "nearest raw pixel" into a single vectorised call per final
    # polyline.
    all_points: List[NDArray[np.float64]] = []
    all_ids: List[NDArray[np.int32]] = []
    all_local_indices: List[NDArray[np.int32]] = []
    for raw_id, poly in enumerate(raw_segments):
        arr = np.asarray(poly, dtype=float)
        if len(arr) == 0:
            continue
        all_points.append(arr)
        all_ids.append(np.full(len(arr), raw_id, dtype=np.int32))
        all_local_indices.append(np.arange(len(arr), dtype=np.int32))

    if not all_points:
        # Nothing to label against — emit empty spans for each final
        # segment so callers can iterate without a None check.
        return [
            LabeledSegment(
                points=np.asarray(p, dtype=float),
                spans=[],
                raw_ids=np.full(len(p), -1, dtype=np.int32),
                raw_indices=np.zeros(len(p), dtype=np.int32),
            )
            for p in final_segments
        ]

    raw_pixels = np.concatenate(all_points, axis=0)
    raw_ids = np.concatenate(all_ids, axis=0)
    raw_local_indices = np.concatenate(all_local_indices, axis=0)
    tree = KDTree(raw_pixels)

    out: List[LabeledSegment] = []
    for fp in final_segments:
        fp = np.asarray(fp, dtype=float)
        if len(fp) == 0:
            out.append(
                LabeledSegment(
                    points=fp,
                    spans=[],
                    raw_ids=np.zeros(0, dtype=np.int32),
                    raw_indices=np.zeros(0, dtype=np.int32),
                )
            )
            continue
        _, idx = tree.query(fp, k=1)
        per_pixel_id = raw_ids[idx]
        per_pixel_local = raw_local_indices[idx].astype(np.int32)

        spans: List[RawSegmentSpan] = []
        run_start = 0
        for i in range(1, len(fp)):
            if per_pixel_id[i] != per_pixel_id[i - 1]:
                spans.append(
                    RawSegmentSpan(
                        raw_segment_id=int(per_pixel_id[run_start]),
                        start=run_start,
                        end=i,
                        raw_start=int(per_pixel_local[run_start]),
                        raw_end=int(per_pixel_local[i - 1]),
                    )
                )
                run_start = i
        spans.append(
            RawSegmentSpan(
                raw_segment_id=int(per_pixel_id[run_start]),
                start=run_start,
                end=len(fp),
                raw_start=int(per_pixel_local[run_start]),
                raw_end=int(per_pixel_local[len(fp) - 1]),
            )
        )
        out.append(
            LabeledSegment(
                points=fp,
                spans=spans,
                raw_ids=per_pixel_id.astype(np.int32),
                raw_indices=per_pixel_local,
            )
        )
    return out
