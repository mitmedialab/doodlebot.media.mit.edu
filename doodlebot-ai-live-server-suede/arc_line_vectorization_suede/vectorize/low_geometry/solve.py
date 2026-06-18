"""Top-level orchestration: from polylines + junctions to fitted primitives.

The flow:

1. For each polyline, run ``fit_polyline`` to get a chain of primitives
   plus per-primitive source-point ranges.
2. Lay out a flat manifest indexing each primitive globally.
3. Build the constraint bundle from the StrokeGraph's junctions:
   * Internal chain joints → Coincide + G1
   * Terminate-into-each-other junctions → Coincide between participating endpoints
   * Terminate-on (mixed) junctions → OnCurve from terminating endpoint to host
   * Smooth (low-deflection) junctions where two strokes meet → G1
4. Solve once → ``primitives_fitted``.
5. Run beautification on the result, add candidates as soft constraints,
   solve again → ``primitives_consolidated``.

The build is incremental — each step's output is exposed on the result
dataclass so the caller can render diagnostics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from ...graph import Junction, Role, StrokeGraph

from .beautify import BeautifyTolerances, detect, merge_into
from .fitting import ChainPiece, fit_polyline, is_closed_polyline
from .manifest import (
    Coincide,
    G1Smooth,
    OnCurve,
    SoftConstraints,
    pack,
    parameter_count,
    parameter_scales,
    unpack,
)
from .primitives import Arc, Circle, Line, Primitive
from .residuals import Weights, assemble_residuals

# ---------------------------------------------------------------------------
# Result containers


@dataclass
class FittedSegment:
    """The chain output for one polyline, with global primitive IDs."""

    polyline_index: int
    pieces: List[ChainPiece]
    primitive_ids: List[int]  # global index in the manifest, per piece
    source_points: List[NDArray[np.float64]]  # one array per piece


@dataclass
class SolveResult:
    primitives: List[Primitive]
    fitted_segments: List[FittedSegment]
    soft: SoftConstraints
    converged: bool
    cost: float
    n_iters: int


# ---------------------------------------------------------------------------
# Configuration


@dataclass
class FitConfig:
    line_tol: float = 0.005  # relative to segment bbox diagonal
    arc_tol: float = 0.012
    lam_rel: float = 4.0  # MDL penalty multiplier
    min_len: int = 5
    use_dp: bool = False  # top-down is faster & usually cleaner
    max_window: int = 256
    closed_tol: float = 1.5
    # G1 auto-detection at smooth segment-to-segment junctions:
    smooth_junction_deg_threshold: float = 25.0
    # Source points are subsampled (uniformly along the chain piece)
    # before being handed to the joint solver. Skeletal polylines from
    # the upstream are pixel-dense (~1000+ pts/polyline) and that scale
    # is well beyond what the data residual needs to nail down a
    # primitive — it just makes the optimizer slow. ~30 pts per
    # primitive captures shape with sub-pixel residual mean.
    source_points_per_primitive: int = 30
    # Cap on the working polyline length passed to ``fit_polyline``.
    # Top-down's min_len is in raw indices; on a 1000+ pt polyline
    # that means it can fragment a 5 px wiggle into a tiny piece
    # while the rest of the stroke fits as one arc. Subsampling first
    # makes min_len correspond to a stroke-length fraction instead.
    polyline_subsample_cap: int = 200


@dataclass
class SolveConfig:
    weights: Weights = field(default_factory=Weights)
    # Diminishing returns past ~60-100 iters; capping here keeps the
    # worst-case drawing under ~30s for the joint solve.
    max_iters: int = 80
    method: str = "trf"  # 'trf' supports bounds; 'lm' is faster but unbounded
    f_scale: float = 1.0


# ---------------------------------------------------------------------------
# Pipeline


def build_chains(
    graph: StrokeGraph,
    config: FitConfig,
) -> List[FittedSegment]:
    """Fit each polyline independently into a chain of primitives."""
    out: List[FittedSegment] = []
    for pi, raw_poly in enumerate(graph.polylines):
        if len(raw_poly) < 2:
            continue
        # Subsample the polyline so that ``min_len`` (in indices) and
        # the recursion's "split at worst residual" both operate on a
        # stroke-length-fraction scale rather than raw pixel density.
        cap_poly = max(2, int(config.polyline_subsample_cap))
        if len(raw_poly) > cap_poly:
            sub_idx = np.linspace(0, len(raw_poly) - 1, cap_poly, dtype=int)
            poly = raw_poly[sub_idx].copy()
        else:
            poly = raw_poly
        chain = fit_polyline(
            poly,
            line_tol=config.line_tol,
            arc_tol=config.arc_tol,
            lam_rel=config.lam_rel,
            min_len=config.min_len,
            closed_tol=config.closed_tol,
            use_dp=config.use_dp,
            max_window=config.max_window,
        )
        if not chain:
            continue
        # Source points per piece: subsample further if the piece is
        # bigger than the per-primitive cap. The piece indices already
        # refer to the subsampled polyline.
        src: List[NDArray[np.float64]] = []
        cap = max(2, int(config.source_points_per_primitive))
        for c in chain:
            piece = poly[c.start_idx : c.end_idx]
            if len(piece) <= cap:
                src.append(piece.copy())
            else:
                idx = np.linspace(0, len(piece) - 1, cap, dtype=int)
                src.append(piece[idx].copy())
        out.append(
            FittedSegment(
                polyline_index=pi,
                pieces=chain,
                primitive_ids=[],  # filled in below by ``assign_global_ids``
                source_points=src,
            )
        )
    return out


def assign_global_ids(segments: List[FittedSegment]) -> List[Primitive]:
    """Lay out a flat primitive list with global IDs and populate
    ``segment.primitive_ids``. Returns the resulting primitive list.
    """
    prims: List[Primitive] = []
    for seg in segments:
        seg.primitive_ids = []
        for piece in seg.pieces:
            seg.primitive_ids.append(len(prims))
            prims.append(piece.primitive)
    return prims


def _chain_endpoint(seg: FittedSegment, end: str) -> Tuple[int, str]:
    """Return ``(global_prim_id, "start"|"end")`` for the segment's
    external start/end."""
    if end == "start":
        return seg.primitive_ids[0], "start"
    return seg.primitive_ids[-1], "end"


def _polyline_endpoint_to_chain_end(poly_n: int, point_index: int) -> Optional[str]:
    """Map a Participation's point_index to the chain end ("start" or
    "end") if it sits at one of the polyline endpoints; None otherwise.
    """
    if point_index == 0:
        return "start"
    if point_index == poly_n - 1:
        return "end"
    return None


def _resolve_host_subprimitive(
    junction_xy: NDArray[np.float64],
    seg: FittedSegment,
    polyline: NDArray[np.float64],
) -> int:
    """For a terminate-on junction landing on segment ``seg``, find which
    sub-primitive in the chain the junction sits closest to. Return the
    global primitive ID.

    Why we use source_points rather than the raw polyline: ``Vectorize``
    subsamples each polyline (default cap 200 pts) before running
    ``fit_polyline``, so ``seg.pieces``' ``start_idx``/``end_idx``
    are indices into the SUBSAMPLED polyline, not the raw polyline.
    Using the raw polyline's argmin would land us at an index in
    full-polyline space that doesn't correspond to chain-piece
    coordinates (the housesun apex junction landed at full-idx 183,
    which falls in chain-piece [138:200] in subsampled space and
    returns the FLOOR primitive instead of the left roof slope).
    ``source_points`` per piece are the actual data the primitive was
    fit to, so finding the piece whose source_points are closest to
    the junction location is robust to any subsampling layer.
    """
    # If source_points are available, use them — they live in the same
    # coordinate space as the junction location.
    if seg.source_points and all(len(sp) > 0 for sp in seg.source_points):
        best_pid = seg.primitive_ids[0]
        best_dist = float("inf")
        for k, src in enumerate(seg.source_points):
            d = float(np.min(np.linalg.norm(src - junction_xy, axis=1)))
            if d < best_dist:
                best_dist = d
                best_pid = seg.primitive_ids[k]
        return best_pid
    # Fallback: closest-source-point index into ``polyline``. Assumes
    # ``polyline``'s indexing matches ``seg.pieces``' indexing — only
    # safe when no subsampling happened upstream.
    dists = np.linalg.norm(polyline - junction_xy, axis=1)
    i_nearest = int(np.argmin(dists))
    for k, piece in enumerate(seg.pieces):
        if piece.start_idx <= i_nearest < piece.end_idx:
            return seg.primitive_ids[k]
    return seg.primitive_ids[-1]


def build_junction_constraints(
    graph: StrokeGraph,
    segments: List[FittedSegment],
    smooth_junction_deg: float,
    primitives: Optional[List[Primitive]] = None,
) -> SoftConstraints:
    """Translate the StrokeGraph's junctions into soft constraints.

    Rules:

    * If two or more **terminal** participants meet at one junction,
      their chain ends are made coincident (pairwise to first).
      Exception: when the "base" endpoint sits on a Circle primitive,
      use OnCurve for the other endpoints instead — a Circle has no
      meaningful endpoint (the convention theta=0 is arbitrary), and
      a hard Coincide on that arbitrary point pulls the entire circle
      to satisfy the constraint. Putting the other endpoints
      ON THE PERIMETER respects the geometry instead.
    * If a **terminal** participant meets one or more **crossing** /
      **cusp** (interior) participants, the terminal end lies on the
      host's sub-primitive (OnCurve).
    * If the junction's pair-deflection is below
      ``smooth_junction_deg``, a G1 constraint is added between the
      participating strokes.
    """
    seg_by_poly: Dict[int, FittedSegment] = {s.polyline_index: s for s in segments}
    out = SoftConstraints.empty()

    def is_circle(pid: int) -> bool:
        if primitives is None:
            return False
        return 0 <= pid < len(primitives) and isinstance(primitives[pid], Circle)

    for j in graph.junctions:
        # Filter participants whose polyline made it into our segment
        # set (a polyline could be too short for fit_polyline and got
        # dropped).
        live = [p for p in j.participants if p.polyline_index in seg_by_poly]
        if len(live) < 2:
            continue

        # Bucket by terminal-vs-interior.
        terminals = [p for p in live if p.role == Role.TERMINAL]
        interiors = [p for p in live if p.role != Role.TERMINAL]

        # ------------------------------------------------------------
        # Coincidence: all terminal participants at this junction.
        # When the base endpoint lies on a Circle primitive, use
        # OnCurve instead of Coincide for the other endpoints.
        # ------------------------------------------------------------
        if len(terminals) >= 2:
            base = terminals[0]
            base_seg = seg_by_poly[base.polyline_index]
            base_end = _polyline_endpoint_to_chain_end(
                len(graph.polylines[base.polyline_index]), base.point_index
            )
            if base_end is not None:
                base_ep = _chain_endpoint(base_seg, base_end)
                base_is_circle = is_circle(base_ep[0])
                for p in terminals[1:]:
                    seg = seg_by_poly[p.polyline_index]
                    end = _polyline_endpoint_to_chain_end(
                        len(graph.polylines[p.polyline_index]), p.point_index
                    )
                    if end is None:
                        continue
                    ep = _chain_endpoint(seg, end)
                    if base_is_circle:
                        # Other terminal must lie on the circle.
                        out.on_curve.append(OnCurve(terminating=ep, host=base_ep[0]))
                    elif is_circle(ep[0]):
                        # The "other" terminal is the circle; the base
                        # must lie on it.
                        out.on_curve.append(OnCurve(terminating=base_ep, host=ep[0]))
                    else:
                        out.coincide.append(Coincide(base_ep, ep))

        # ------------------------------------------------------------
        # On-curve: each terminal's endpoint lies on each interior's
        # host sub-primitive. We pick the FIRST interior host so we
        # don't over-constrain (the interior strokes are roughly
        # tangent to each other at the junction; one constraint is
        # enough to keep the terminal pinned).
        # ------------------------------------------------------------
        if terminals and interiors:
            host_part = interiors[0]
            host_seg = seg_by_poly[host_part.polyline_index]
            host_poly = graph.polylines[host_part.polyline_index]
            host_id = _resolve_host_subprimitive(j.location, host_seg, host_poly)
            for term in terminals:
                seg = seg_by_poly[term.polyline_index]
                end = _polyline_endpoint_to_chain_end(
                    len(graph.polylines[term.polyline_index]), term.point_index
                )
                if end is None:
                    continue
                ep = _chain_endpoint(seg, end)
                out.on_curve.append(OnCurve(terminating=ep, host=host_id))

        # ------------------------------------------------------------
        # G1: at pure-terminal junctions where exactly two strokes
        # meet with a small deflection, add a smoothing constraint
        # between their chain-end sub-primitives.
        # ------------------------------------------------------------
        if len(terminals) == 2 and not interiors:
            a, b = terminals
            # Pair deflection: arccos(-tangent_a · tangent_b) — both
            # tangent_ins point INTO the polyline body, so an aligned
            # "smooth corner" has tangent_a ≈ -tangent_b.
            cos = float(np.clip(-a.tangent_in @ b.tangent_in, -1.0, 1.0))
            defl = float(np.degrees(np.arccos(cos)))
            if defl < smooth_junction_deg:
                seg_a = seg_by_poly[a.polyline_index]
                seg_b = seg_by_poly[b.polyline_index]
                end_a = _polyline_endpoint_to_chain_end(
                    len(graph.polylines[a.polyline_index]), a.point_index
                )
                end_b = _polyline_endpoint_to_chain_end(
                    len(graph.polylines[b.polyline_index]), b.point_index
                )
                if end_a is not None and end_b is not None:
                    pid_a, _ = _chain_endpoint(seg_a, end_a)
                    pid_b, _ = _chain_endpoint(seg_b, end_b)
                    # alpha = 1 means "outgoing tangent at end" — but
                    # since one of these is a chain-start, we want to
                    # flip the sign convention by choosing alpha
                    # appropriately.
                    alpha_a = 1.0 if end_a == "end" else 0.0
                    alpha_b = 0.0 if end_b == "start" else 1.0
                    out.g1.append(
                        G1Smooth(a=pid_a, alpha_a=alpha_a, b=pid_b, alpha_b=alpha_b)
                    )

    return out


# ---------------------------------------------------------------------------
# Jacobian sparsity
#
# The residual function evaluates ~600-3000 residuals over ~100-300
# parameters. Each residual only depends on a handful of parameters
# (the params of one or two primitives). Without a sparsity pattern,
# scipy's TRF finite-difference Jacobian does N_params extra residual
# evaluations per iteration — a 10-50x speedup is on the table.


def _param_slice(template: List[Primitive], pid: int) -> Tuple[int, int]:
    """Return ``(start, stop)`` parameter indices for primitive ``pid``."""
    cur = 0
    for i, p in enumerate(template):
        n = (
            4
            if isinstance(p, Line)
            else 5 if isinstance(p, Arc) else 3 if isinstance(p, Circle) else 0
        )
        if i == pid:
            return cur, cur + n
        cur += n
    raise IndexError(pid)


def _build_jacobian_sparsity(
    template: List[Primitive],
    source_points: Dict[int, NDArray[np.float64]],
    soft: SoftConstraints,
    initial_radii: Dict[int, float],
) -> "lil_matrix":
    """Build the sparsity pattern matching ``assemble_residuals``.

    Layout must mirror the residual block order in residuals.py exactly:
        1. data (per primitive with source points, in dict-iteration order)
        2. coincide (2 scalars each)
        3. on_curve (1 scalar each)
        4. g1 (1 scalar each)
        5. parallel (1 scalar each)
        6. perpendicular (1 scalar each)
        7. equal_radius (1 scalar each)
        8. concentric (2 scalars each)
        9. radius regularization (1 scalar per arc/circle with init radius)
        10. bulge regularization (1 scalar per Arc)
    """
    n_params = parameter_count(template)
    slices = {i: _param_slice(template, i) for i in range(len(template))}

    rows: List[Tuple[int, int]] = []  # (n_rows, depends_on_primitive_id_or_pair)

    # 1. data
    data_meta: List[Tuple[int, int]] = []
    for pid, pts in source_points.items():
        if len(pts) == 0:
            continue
        data_meta.append((pid, len(pts)))

    # 1b. endpoint anchors — 4 scalar rows per Line/Arc with source pts
    anchor_meta: List[int] = []
    for pid, pts in source_points.items():
        if len(pts) == 0:
            continue
        if isinstance(template[pid], (Line, Arc)):
            anchor_meta.append(pid)

    # 10. bulge reg — one row per Arc
    bulge_arc_ids = [i for i, p in enumerate(template) if isinstance(p, Arc)]

    n_rows = (
        sum(k for _, k in data_meta)
        + 4 * len(anchor_meta)
        + 2 * len(soft.coincide)
        + len(soft.on_curve)
        + len(soft.g1)
        + len(soft.parallel)
        + len(soft.perpendicular)
        + len(soft.equal_radius)
        + 2 * len(soft.concentric)
        + len(initial_radii)
        + len(bulge_arc_ids)
    )

    J = lil_matrix((n_rows, n_params), dtype=np.float64)

    row = 0
    # 1. data residuals
    for pid, k in data_meta:
        s, e = slices[pid]
        for _ in range(k):
            J[row, s:e] = 1
            row += 1
    # 1b. anchor residuals — depend on prim's p0/p1 only (first 4 params
    # of Line/Arc). To keep it simple and correct, mark the whole slice.
    for pid in anchor_meta:
        s, e = slices[pid]
        for _ in range(4):
            J[row, s:e] = 1
            row += 1
    # 2. coincide (2 scalars per pair, depends on endpoints of a & b)
    for c in soft.coincide:
        sa, ea = slices[c.a[0]]
        sb, eb = slices[c.b[0]]
        for _ in range(2):
            J[row, sa:ea] = 1
            J[row, sb:eb] = 1
            row += 1
    # 3. on-curve (1 scalar per constraint, depends on point primitive + host)
    for c in soft.on_curve:
        sa, ea = slices[c.terminating[0]]
        sb, eb = slices[c.host]
        J[row, sa:ea] = 1
        J[row, sb:eb] = 1
        row += 1
    # 4. G1 (1 scalar per constraint, depends on params of both primitives)
    for c in soft.g1:
        sa, ea = slices[c.a]
        sb, eb = slices[c.b]
        J[row, sa:ea] = 1
        J[row, sb:eb] = 1
        row += 1
    # 5. parallel
    for c in soft.parallel:
        sa, ea = slices[c.a]
        sb, eb = slices[c.b]
        J[row, sa:ea] = 1
        J[row, sb:eb] = 1
        row += 1
    # 6. perpendicular
    for c in soft.perpendicular:
        sa, ea = slices[c.a]
        sb, eb = slices[c.b]
        J[row, sa:ea] = 1
        J[row, sb:eb] = 1
        row += 1
    # 7. equal radius
    for c in soft.equal_radius:
        sa, ea = slices[c.a]
        sb, eb = slices[c.b]
        J[row, sa:ea] = 1
        J[row, sb:eb] = 1
        row += 1
    # 8. concentric (2 rows per pair)
    for c in soft.concentric:
        sa, ea = slices[c.a]
        sb, eb = slices[c.b]
        for _ in range(2):
            J[row, sa:ea] = 1
            J[row, sb:eb] = 1
            row += 1
    # 9. radius regularization
    for pid in initial_radii.keys():
        s, e = slices[pid]
        J[row, s:e] = 1
        row += 1
    # 10. bulge regularization — depends only on the bulge param of the arc
    # (it's the 5th param in the arc's 5-param slot). For simplicity mark
    # the whole arc slice; over-marking is harmless, only under-marking
    # would break the optimizer.
    for pid in bulge_arc_ids:
        s, e = slices[pid]
        J[row, s:e] = 1
        row += 1

    assert row == n_rows, f"sparsity row count mismatch: {row} vs {n_rows}"
    return J


# ---------------------------------------------------------------------------


def solve_once(
    primitives: List[Primitive],
    source_points: Dict[int, NDArray[np.float64]],
    soft: SoftConstraints,
    weights: Weights,
    pos_scale: float,
    max_iters: int = 200,
    method: str = "trf",
) -> Tuple[List[Primitive], SolveResult]:
    """Run one least-squares solve. Returns the (primitives, result)."""
    if not primitives:
        return [], SolveResult([], [], soft, True, 0.0, 0)

    template = primitives
    x0 = pack(template)

    # Pre-compute initial radii so the log-radius regularization can
    # anchor the curved primitives to their starting fits.
    initial_radii: Dict[int, float] = {}
    for i, p in enumerate(template):
        if isinstance(p, Circle):
            initial_radii[i] = max(1e-3, float(p.radius))
        elif isinstance(p, Arc):
            r = p.radius()
            if np.isfinite(r):
                initial_radii[i] = max(1e-3, float(r))

    def f(x):
        return assemble_residuals(
            x, template, source_points, soft, weights, initial_radii
        )

    x_scale = parameter_scales(template, pos_scale=pos_scale)
    jac_sparsity = (
        _build_jacobian_sparsity(template, source_points, soft, initial_radii)
        if method == "trf"
        else None
    )

    sol = least_squares(
        f,
        x0,
        method=method,
        x_scale=x_scale,
        max_nfev=max_iters,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
        jac_sparsity=jac_sparsity,
    )

    new_prims = unpack(sol.x, template)
    res = SolveResult(
        primitives=new_prims,
        fitted_segments=[],  # filled in by caller
        soft=soft,
        converged=bool(sol.success),
        cost=float(sol.cost),
        n_iters=int(sol.nfev),
    )
    return new_prims, res


def _bbox_diag(graph: StrokeGraph) -> float:
    pts = np.concatenate(graph.polylines, axis=0) if graph.polylines else None
    if pts is None or len(pts) == 0:
        return 100.0
    span = pts.max(axis=0) - pts.min(axis=0)
    return max(float(np.linalg.norm(span)), 10.0)
