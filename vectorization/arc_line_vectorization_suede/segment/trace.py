import numpy as np
from scipy.ndimage import label as cc_label
from typing import List, cast
from numpy.typing import NDArray

NEIGHBOR_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _skeleton_neighbors(skel: np.ndarray, y: int, x: int):
    H, W = skel.shape
    for dy, dx in NEIGHBOR_OFFSETS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
            yield ny, nx


def _compute_degree(skel: np.ndarray) -> np.ndarray:
    """Per-pixel count of 8-connected ON neighbours."""
    deg = np.zeros_like(skel, dtype=np.int8)
    ys, xs = np.where(skel)
    for y, x in zip(ys, xs):
        deg[y, x] = sum(1 for _ in _skeleton_neighbors(skel, y, x))
    return deg


def _label_node_components(skel: np.ndarray, deg: np.ndarray):
    """Assign a unique non-zero id to every "node pixel" in the skeleton.

    A node pixel is either:
      - an endpoint (degree == 1), or
      - a member of a super-junction: a connected component (8-connected) of
        pixels with degree >= 3. All pixels in one component share the same id.

    Returns (node_id, n_super_junctions). node_id is an int array with the same
    shape as skel; 0 means "not a node" (degree-2 interior).
    """

    structure = np.ones((3, 3), dtype=bool)
    junction_mask = deg >= 3
    sj_labels, n_sj = cast(
        tuple[NDArray[np.int32], int],
        cc_label(junction_mask, structure=structure),
    )

    node_id = np.zeros_like(skel, dtype=np.int32)
    node_id[junction_mask] = sj_labels[junction_mask]  # ids 1..n_sj

    endpoint_ys, endpoint_xs = np.where(deg == 1)
    next_id = n_sj + 1
    for y, x in zip(endpoint_ys, endpoint_xs):
        node_id[y, x] = next_id
        next_id += 1

    return node_id, n_sj


def trace_skeleton(skel: np.ndarray) -> List[NDArray[np.float64]]:
    """Trace a skeleton into a list of polylines.

    Returns a list of (N, 2) float arrays of (`x, y) pixel coordinates.

    Endpoints and junction-clusters anchor the polylines. Multiple adjacent
    junction pixels are treated as a single node (super-junction), so the
    walk does not spawn spurious 2-pixel polylines inside thick junctions.

    Closed loops with no endpoints/junctions are broken at an arbitrary pixel.
    """
    skel = skel.astype(bool)
    deg = _compute_degree(skel)
    node_id, _ = _label_node_components(skel, deg)

    polylines: List[NDArray[np.float64]] = []
    visited_starts = set()  # (start_pixel, first_step_pixel)

    def trace(start_pixel, start_id, first_step):
        """Walk from start_pixel via first_step until reaching a different
        node. Returns the path or None if first_step is in the same node."""
        if node_id[first_step] == start_id:
            return None  # hopping inside the same super-junction
        path = [start_pixel, first_step]
        prev = start_pixel
        cur = first_step
        while node_id[cur] == 0:
            next_p = None
            for nny, nnx in _skeleton_neighbors(skel, cur[0], cur[1]):
                cand = (nny, nnx)
                if cand == prev:
                    continue
                # Prefer non-node neighbours; otherwise accept the first node.
                if node_id[cand] == 0:
                    next_p = cand
                    break
                if next_p is None:
                    next_p = cand
            if next_p is None:
                break
            prev = cur
            cur = next_p
            path.append(cur)
        return path

    # All node pixels (members of super-junctions + endpoints)
    node_ys, node_xs = np.where(node_id != 0)
    for sy, sx in zip(node_ys, node_xs):
        start_pixel = (sy, sx)
        start_id = int(node_id[sy, sx])
        for ny, nx in _skeleton_neighbors(skel, sy, sx):
            first_step = (ny, nx)
            key = (start_pixel, first_step)
            if key in visited_starts:
                continue
            path = trace(start_pixel, start_id, first_step)
            if path is None:
                continue
            visited_starts.add(key)
            # Also mark the reverse direction so we don't re-trace the same edge
            if len(path) >= 2:
                visited_starts.add((path[-1], path[-2]))
            polylines.append(np.array([(x, y) for y, x in path], dtype=float))

    # Closed loops: connected components of skel pixels that contain no node.
    seen = np.zeros_like(skel, dtype=bool)
    for poly in polylines:
        for x, y in poly:
            seen[int(round(y)), int(round(x))] = True
    for y, x in zip(*np.where(skel & (node_id == 0) & ~seen)):
        path = [(y, x)]
        seen[y, x] = True
        prev = None
        cur = (y, x)
        while True:
            next_p = None
            for ny, nx in _skeleton_neighbors(skel, cur[0], cur[1]):
                if (ny, nx) != prev and not seen[ny, nx]:
                    next_p = (ny, nx)
                    break
            if next_p is None:
                break
            seen[next_p] = True
            path.append(next_p)
            prev = cur
            cur = next_p
        if len(path) >= 3:
            path.append(path[0])  # close the loop
            polylines.append(np.array([(x, y) for y, x in path], dtype=float))

    return polylines
