"""Track 2D spatial coincidence of primitive endpoints, so that any
operation that moves one endpoint can propagate the move to all
endpoints that originally coincided with it.

The motivating problem is "mismatched junctions across strokes". In
the input drawing, two strokes (say, a wheel rim and a spoke) often
share a junction at a point. After segmentation/fusion they end up in
different strokes -- the rim is one stroke and each spoke is another,
with pen-up traversals in between. The shared junction now exists as
TWO independent endpoints, one in each stroke. Later, when the rim
gets snapped onto a consensus circle by the consolidator, ONLY the
rim's endpoint moves; the spoke's endpoint stays where it was drawn,
producing a visible mismatch (spoke overshoots or falls short of the
new rim).

The junction graph clusters all primitive endpoints by 2D proximity
once at the start. Subsequent endpoint moves go through ``move()``,
which propagates to every endpoint sharing the same junction. The
cluster membership is built from ORIGINAL positions and is immutable
afterward -- endpoints that originally coincided always travel
together, even after they've drifted through several operations.

Typical usage during consolidation::

    jg = JunctionGraph(prims, epsilon=3.0)
    moved_others = jg.move(prims, idx, "start", new_pos)
    for (oi, _) in moved_others:
        if prims[oi].kind == "arc":
            refit_arc_through_endpoints(prims[oi])

The graph itself does not know how to refit arcs after their endpoints
move; the caller is responsible for that. This keeps the graph generic
and its responsibilities focused.
"""

from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


class JunctionGraph:
    """Spatial-coincidence map for primitive endpoints.

    Endpoints are identified by ``(prim_idx, "start"|"end")`` tuples.
    Two endpoints share a junction iff their L2 distance was below
    ``epsilon`` at construction time. Junction membership is fixed
    after construction; it does not depend on current positions.
    """

    def __init__(self, prims: Sequence[Any], epsilon: float = 3.0):
        self._epsilon = float(epsilon)

        endpoints: List[Tuple[int, str, NDArray]] = []
        for i, p in enumerate(prims):
            endpoints.append((i, "start", np.asarray(p.start, dtype=float)))
            endpoints.append((i, "end", np.asarray(p.end, dtype=float)))

        n = len(endpoints)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # O(n^2) pairwise distance check. n is twice the number of
        # primitives -- usually a few hundred, never thousands. A spatial
        # hash would speed this up but isn't worth the code.
        eps_sq = self._epsilon * self._epsilon
        for ii in range(n):
            pi = endpoints[ii][2]
            pi_x, pi_y = float(pi[0]), float(pi[1])
            for jj in range(ii + 1, n):
                pj = endpoints[jj][2]
                dx = pi_x - float(pj[0])
                dy = pi_y - float(pj[1])
                if dx * dx + dy * dy < eps_sq:
                    union(ii, jj)

        # Compress find paths and build the (junction_id -> members) and
        # (endpoint -> junction_id) tables.
        self._junction_of: Dict[Tuple[int, str], int] = {}
        self._members: Dict[int, List[Tuple[int, str]]] = {}
        for ii, (i, attr, _) in enumerate(endpoints):
            root = find(ii)
            self._junction_of[(i, attr)] = root
            self._members.setdefault(root, []).append((i, attr))

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def n_junctions(self) -> int:
        """Total number of distinct junctions, shared or not."""
        return len(self._members)

    def n_shared_junctions(self) -> int:
        """Number of junctions where two or more endpoints meet. This is
        the count of "real" junctions in the drawing -- single-membership
        junctions are just stroke endpoints that don't touch anything."""
        return sum(1 for v in self._members.values() if len(v) >= 2)

    def shared_junction_sizes(self) -> List[int]:
        """Member counts of every shared junction, sorted descending. A
        size-3 entry means three endpoints meet at one point in the
        original drawing."""
        sizes = [len(v) for v in self._members.values() if len(v) >= 2]
        sizes.sort(reverse=True)
        return sizes

    def members(self, prim_idx: int, attr: str) -> List[Tuple[int, str]]:
        """All endpoints sharing the junction with the given one,
        including itself. If the queried endpoint is not in the graph
        (which shouldn't happen for a graph built from the same prims),
        returns just the queried endpoint."""
        j = self._junction_of.get((prim_idx, attr))
        if j is None:
            return [(prim_idx, attr)]
        return list(self._members[j])

    def has_neighbours(self, prim_idx: int, attr: str) -> bool:
        """True iff this endpoint shares its junction with at least one
        other primitive's endpoint."""
        j = self._junction_of.get((prim_idx, attr))
        return j is not None and len(self._members[j]) >= 2

    def move(
        self,
        prims: Sequence[Any],
        prim_idx: int,
        attr: str,
        new_pos: NDArray,
    ) -> List[Tuple[int, str]]:
        """Set the position of ``(prim_idx, attr)`` and every endpoint
        sharing its junction to ``new_pos``. Returns the list of
        ``(other_idx, other_attr)`` that were ALSO moved -- i.e., the
        propagation set, not including the requested endpoint itself.

        After this call, every junction-mate of ``(prim_idx, attr)`` has
        ``new_pos`` as its position. The caller is expected to refit any
        arc primitives in the propagation set whose center+radius are no
        longer consistent with their new endpoints.
        """
        np_pos = np.asarray(new_pos, dtype=float)
        moved_others: List[Tuple[int, str]] = []
        for other_idx, other_attr in self.members(prim_idx, attr):
            p = prims[other_idx]
            if other_attr == "start":
                p.start = np_pos.copy()
            else:
                p.end = np_pos.copy()
            if (other_idx, other_attr) != (prim_idx, attr):
                moved_others.append((other_idx, other_attr))
        return moved_others
