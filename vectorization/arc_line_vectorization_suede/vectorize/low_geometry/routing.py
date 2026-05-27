"""Sequencing primitives into a pen-up-minimizing tour and emitting
robot commands.

Graph problem:

* **Vertices** = endpoint locations (with tolerance-based clustering).
  A Circle has no endpoints — it's a "loop edge" attached to a single
  virtual vertex at its rightmost point.
* **Edges** = primitives. Each edge is undirected (we can traverse a
  Line/Arc in either direction; the bulge / start-end pair gets
  flipped if we go end → start).
* **Goal** = minimum-pen-up tour. This is an Eulerian path if the
  graph has 0 or 2 odd-degree vertices; otherwise it's the Chinese
  Postman problem. NetworkX's ``eulerize`` adds duplicate edges to
  fix the parity, which translates back to extra pen-up moves.

Emitted commands match the robot's interface in ``commands.py``:

* ``{"kind": "line", "distance": …, "penDown": …}`` — drive forward.
* ``{"kind": "spin", "degrees": …}`` — rotate in place by signed degrees.
* ``{"kind": "arc",  "radius": …, "degrees": …}`` — drive an arc.

Heading convention (per ``commands.py``): positive degrees = CCW in
image coords (which renders as CW on screen). For arcs, ``radius`` is
always positive; the sign of ``degrees`` selects direction.

Scope — this stage minimizes PEN-UPS, not total turning. The Eulerian
/ Chinese-Postman tour picks *an* order that visits every primitive
with the fewest pen-up jumps, but among the many valid Eulerian paths
it does not choose the one with the least in-place spinning, and spins
are a large part of the firmware draw time. Closing that gap is the
job of ``OptimizeRoute`` (see ``release/optimize.py``), which re-orders
and re-directs the pen-down primitives against the full firmware
motion model. ``OptimizeRoute`` is therefore a REQUIRED final stage:
the command streams produced here (and exposed as
``LowGeometryVectorize.commands_consolidated``) are pen-up-optimal but
not time-optimal, and on some images are slower than the naive
high-geometry baseline until ``OptimizeRoute`` has run.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
import networkx as nx
from numpy.typing import NDArray

from ...commands import DrawingCommand
from .primitives import Arc, Circle, Line, Primitive, endpoint, tangent_at_end

# ---------------------------------------------------------------------------
# Endpoint clustering


def _endpoint_locations(
    prims: Sequence[Primitive], snap_tol: float
) -> Tuple[Dict[int, Tuple[int, int]], List[NDArray[np.float64]]]:
    """Cluster all primitive endpoints into a small set of canonical
    vertex locations.

    Returns:
        edge_to_verts: ``{prim_id: (start_vert, end_vert)}``
        vert_positions: list of (x, y) centroids, indexed by vertex id

    Circles get a synthetic vertex at their theta=0 point on both ends
    so they show up as a self-loop in the multigraph.
    """
    pts: List[NDArray[np.float64]] = []
    refs: List[Tuple[int, str]] = []
    for pid, p in enumerate(prims):
        if isinstance(p, Circle):
            v = endpoint(p, "start")  # theta = 0
            pts.append(v)
            refs.append((pid, "start"))
            pts.append(v)
            refs.append((pid, "end"))
        else:
            pts.append(endpoint(p, "start"))
            refs.append((pid, "start"))
            pts.append(endpoint(p, "end"))
            refs.append((pid, "end"))

    # Simple greedy clustering: O(N²) but N is small (<2*n_primitives).
    cluster: List[int] = [-1] * len(pts)
    centroids: List[List[NDArray[np.float64]]] = []
    for i, p in enumerate(pts):
        best = -1
        best_d = snap_tol
        for ci, ps in enumerate(centroids):
            mean = np.mean(np.stack(ps, axis=0), axis=0)
            d = float(np.linalg.norm(p - mean))
            if d < best_d:
                best = ci
                best_d = d
        if best >= 0:
            cluster[i] = best
            centroids[best].append(p)
        else:
            cluster[i] = len(centroids)
            centroids.append([p])

    vert_positions = [np.mean(np.stack(ps, axis=0), axis=0) for ps in centroids]
    edge_to_verts: Dict[int, Tuple[int, int]] = {}
    for i, (pid, end) in enumerate(refs):
        vs = edge_to_verts.get(pid, (-1, -1))
        if end == "start":
            edge_to_verts[pid] = (cluster[i], vs[1])
        else:
            edge_to_verts[pid] = (vs[0], cluster[i])
    return edge_to_verts, vert_positions


def _build_multigraph(
    edge_to_verts: Dict[int, Tuple[int, int]],
    prims: Sequence[Primitive],
) -> nx.MultiGraph:
    G = nx.MultiGraph()
    for pid, (u, v) in edge_to_verts.items():
        # Include all vertices, even if isolated, so eulerize() works.
        G.add_node(u)
        G.add_node(v)
        # Edge weight = arc-length-ish, useful for eulerize() to prefer
        # short duplications.
        p = prims[pid]
        if isinstance(p, Line):
            w = p.length()
        elif isinstance(p, Arc):
            w = abs(p.sweep()) * (
                p.radius() if math.isfinite(p.radius()) else p.chord()
            )
        elif isinstance(p, Circle):
            w = 2.0 * math.pi * p.radius
        else:
            w = 0.0
        # Prefix the multigraph key with 'p' so original edges are
        # string-keyed. eulerize() auto-assigns INTEGER keys to the
        # duplicate edges it adds for Chinese-Postman parity fixing; if
        # we used raw integer pids, those auto-assigned ints would
        # collide with primitive ids and the post-tour filter (which
        # tries to drop synthetic edges) would keep them — causing the
        # same primitive to be drawn multiple times in a row.
        G.add_edge(u, v, key=f"p{pid}", weight=float(w))
    return G


def _pick_start_vertex(
    G: nx.MultiGraph,
    vert_positions: List[NDArray[np.float64]],
    start_pos: NDArray[np.float64],
) -> int:
    """For an Eulerian *path* graph (exactly two odd vertices), the path
    must begin at one of them. Pick the odd vertex closer to the robot's
    starting position; otherwise pick the vertex closest to start_pos.
    """
    odd = [v for v in G.nodes if G.degree(v) % 2 == 1]
    candidates = odd if odd else list(G.nodes)
    if not candidates:
        return 0
    best = candidates[0]
    best_d = float(np.linalg.norm(vert_positions[best] - start_pos))
    for v in candidates[1:]:
        d = float(np.linalg.norm(vert_positions[v] - start_pos))
        if d < best_d:
            best = v
            best_d = d
    return best


def _eulerize_per_component(G: nx.MultiGraph) -> nx.MultiGraph:
    """Apply ``nx.eulerize`` per connected component, then return the
    union. NetworkX 3.x's ``eulerize`` requires a connected graph; for
    drawings with multiple disconnected groups we must split first.
    """
    out = nx.MultiGraph()
    for comp_nodes in nx.connected_components(G):
        sub = G.subgraph(comp_nodes).copy()
        if sub.number_of_edges() == 0:
            out.add_nodes_from(sub.nodes())
            continue
        if not nx.is_eulerian(sub) and not nx.has_eulerian_path(sub):
            sub = nx.eulerize(sub)
        out = nx.compose(out, sub)
    return out


def order_primitives(
    prims: Sequence[Primitive],
    start_pos: NDArray[np.float64],
    snap_tol: float = 1.0,
) -> List[Tuple[int, bool]]:
    """Return a tour ``[(prim_id, reverse), ...]`` that visits every
    primitive exactly once, with pen-up jumps inserted between
    disconnected components and parity-fix duplications when needed.

    ``reverse`` is True if the primitive should be traversed end →
    start. Lines and arcs reverse cleanly; circles ignore the flag.
    """
    if not prims:
        return []

    edge_to_verts, vert_positions = _endpoint_locations(prims, snap_tol)
    G = _build_multigraph(edge_to_verts, prims)
    G2 = _eulerize_per_component(G)

    # Within each connected component, run an Eulerian path / circuit.
    tour: List[Tuple[int, int, int]] = []  # (u, v, key)

    # Process components in an order that minimizes pen-up travel from
    # the current position.
    cur_pos = np.asarray(start_pos, dtype=float)
    components = list(nx.connected_components(G2))
    used_components: List[bool] = [False] * len(components)
    while True:
        # Pick the unused component whose closest vertex is nearest to
        # cur_pos.
        best_ci = -1
        best_d = float("inf")
        best_start = -1
        for ci, comp in enumerate(components):
            if used_components[ci]:
                continue
            sub = G2.subgraph(comp).copy()
            if sub.number_of_edges() == 0:
                continue
            start_v = _pick_start_vertex(sub, vert_positions, cur_pos)
            d = float(np.linalg.norm(vert_positions[start_v] - cur_pos))
            if d < best_d:
                best_d = d
                best_ci = ci
                best_start = start_v
        if best_ci < 0:
            break

        sub = G2.subgraph(components[best_ci]).copy()
        used_components[best_ci] = True

        # Eulerian path / circuit on this component.
        if nx.is_eulerian(sub):
            path = list(nx.eulerian_circuit(sub, source=best_start, keys=True))
        else:
            path = list(nx.eulerian_path(sub, source=best_start, keys=True))

        for u, v, key in path:
            tour.append((u, v, key))
            cur_pos = vert_positions[v]

    # Resolve direction per edge. Original edges are string-keyed
    # ("p{pid}") in _build_multigraph; eulerize-added duplicates use
    # auto-assigned integer keys. Drop anything that isn't string-keyed
    # — the pen-up handling in to_commands() will emit a pen-up + move
    # for the resulting gap.
    out: List[Tuple[int, bool]] = []
    for u, v, key in tour:
        if not isinstance(key, str) or not key.startswith("p"):
            continue
        pid = int(key[1:])
        orig_u, orig_v = edge_to_verts[pid]
        reverse = u == orig_v and v == orig_u and orig_u != orig_v
        out.append((pid, reverse))
    return out


# ---------------------------------------------------------------------------
# Command emission


def _wrap_to_pi(angle: float) -> float:
    """Wrap a radian angle to (-π, π]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _reverse_primitive(prim: Primitive) -> Primitive:
    """Return a primitive traversed in the opposite direction."""
    if isinstance(prim, Line):
        return Line(prim.p1.copy(), prim.p0.copy())
    if isinstance(prim, Arc):
        # Flip endpoints AND bulge sign — same arc, reversed traversal.
        return Arc(prim.p1.copy(), prim.p0.copy(), -prim.bulge)
    if isinstance(prim, Circle):
        # Circles are loops; "reverse" doesn't change anything.
        return prim
    raise TypeError(type(prim))


def _heading_change_deg(from_heading: float, to_heading: float) -> float:
    """Signed degrees needed to spin from one heading to another, picking
    the shortest rotation.
    """
    delta = _wrap_to_pi(to_heading - from_heading)
    return float(math.degrees(delta))


def _heading_of(vec: NDArray[np.float64]) -> float:
    return float(math.atan2(vec[1], vec[0]))


def to_commands(
    prims: Sequence[Primitive],
    tour: List[Tuple[int, bool]],
    start_pos: NDArray[np.float64],
    start_heading: float,  # radians
    pen_up_join_tol: float = 0.5,
) -> Sequence[DrawingCommand]:
    """Emit a robot command sequence for the ordered tour.

    A pen-down move is emitted as a sequence: optional spin to align
    with the primitive's start tangent, then either a line (with pen
    down) or an arc command. Pen-up jumps between disconnected
    components are emitted as: pen-up line + spin/arc to align with
    the next primitive's start.
    """
    cmds: Sequence[DrawingCommand] = []
    cur_pos = np.asarray(start_pos, dtype=float)
    cur_heading = float(start_heading)

    for k, (pid, reverse) in enumerate(tour):
        prim = prims[pid]
        if reverse:
            prim = _reverse_primitive(prim)

        # Pen position at primitive start.
        if isinstance(prim, Circle):
            start = prim.point_at(0.0)
        else:
            start = endpoint(prim, "start")

        # If the pen isn't already at the primitive's start, jump
        # there with a pen-up line. Pen-up still requires correct
        # heading first.
        gap = float(np.linalg.norm(start - cur_pos))
        if gap > pen_up_join_tol:
            target_heading = _heading_of(start - cur_pos)
            spin_deg = _heading_change_deg(cur_heading, target_heading)
            if abs(spin_deg) > 1e-3:
                cmds.append({"kind": "spin", "degrees": float(spin_deg)})
                cur_heading = target_heading
            cmds.append({"kind": "line", "distance": gap, "penDown": False})
            cur_pos = start.copy()

        # Decide whether to emit this primitive as an arc or a line.
        # This MUST be settled before the alignment spin: a major arc
        # (sweep > 180°) has a start tangent pointing nearly opposite
        # its chord, so if we spin to the arc's tangent and then emit
        # a chord-line, the simulator drives backwards relative to the
        # actual chord direction. Pick the spin target based on what
        # we're actually emitting.
        emit_as_line: Optional[float] = None  # distance, if degraded to line
        if isinstance(prim, Arc):
            sweep_deg = float(math.degrees(prim.sweep()))
            r = prim.radius()
            sweep_rad = prim.sweep()
            if math.isfinite(r) and r > 0:
                sagitta = abs(r * (1.0 - math.cos(sweep_rad / 2.0)))
                chord_over_r = prim.chord() / r
            else:
                sagitta = 0.0
                chord_over_r = 0.0
            # An arc whose sweep approaches 360° with a chord far
            # smaller than the diameter is an optimizer runaway: the
            # primitive is trying to be a closed circle but Arc isn't
            # the right type for that. A legitimate 350° arc would
            # still have a chord of about 0.17·r; a runaway with
            # bulge=55 has chord/r ≈ 0.04. Only when BOTH conditions
            # hold do we degrade.
            degenerate = (
                not math.isfinite(r)
                or r > 1e6
                or sagitta < 0.5
                or (abs(sweep_deg) > 270.0 and chord_over_r < 0.15)
            )
            if degenerate:
                emit_as_line = float(prim.chord())

        # Pick the heading we should face before emitting. For a line
        # or a degraded arc, this is the chord (p1 - p0) direction.
        # For a true arc, it's the arc's tangent at p0.
        if isinstance(prim, Line) or emit_as_line is not None:
            chord_vec = prim.p1 - prim.p0
            primitive_start_heading = _heading_of(chord_vec)
        elif isinstance(prim, Arc):
            start_tan = tangent_at_end(prim, "start")
            primitive_start_heading = _heading_of(start_tan)
        else:  # Circle
            # The 360° arc command emitted below assumes the robot is
            # heading along the CIRCLE's tangent at point_at(0.0). The
            # simulator places the circle's center at heading ± π/2;
            # if the heading isn't perpendicular to the radius at the
            # start point, the simulator draws the circle in the wrong
            # place. Set the heading to the circle's tangent at the
            # starting point, going CCW (matches the +360° sweep we
            # emit, and the convention used by Arc.tangent_at).
            start_tan = prim.tangent_at(0.0)
            primitive_start_heading = _heading_of(start_tan)

        spin_deg = _heading_change_deg(cur_heading, primitive_start_heading)
        if abs(spin_deg) > 1e-3:
            cmds.append({"kind": "spin", "degrees": float(spin_deg)})
            cur_heading = primitive_start_heading

        # Emit the primitive.
        if isinstance(prim, Line):
            d = prim.length()
            if d > 1e-9:
                cmds.append({"kind": "line", "distance": float(d), "penDown": True})
            cur_pos = prim.p1.copy()
            cur_heading = primitive_start_heading
        elif isinstance(prim, Arc):
            if emit_as_line is not None:
                if emit_as_line > 1e-9:
                    cmds.append(
                        {
                            "kind": "line",
                            "distance": float(emit_as_line),
                            "penDown": True,
                        }
                    )
                cur_pos = prim.p1.copy()
                cur_heading = primitive_start_heading
            else:
                cmds.append(
                    {
                        "kind": "arc",
                        "radius": float(r),
                        "degrees": float(sweep_deg),
                    }
                )
                cur_pos = prim.p1.copy()
                # Heading at end of arc:
                end_tan = tangent_at_end(prim, "end")
                cur_heading = _heading_of(end_tan)
        elif isinstance(prim, Circle):
            # Full circle. Drive a 360° arc (always CCW in image, which
            # is visually CW; sign matches Arc convention). Heading
            # returns to start so we leave it alone.
            cmds.append(
                {
                    "kind": "arc",
                    "radius": float(prim.radius),
                    "degrees": 360.0,
                }
            )
            # Pen position is unchanged (full loop).
        else:
            raise TypeError(type(prim))

    return cmds
