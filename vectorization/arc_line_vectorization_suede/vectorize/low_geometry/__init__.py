"""Vectorize: convert a StrokeGraph into a sequence of robot drawing
commands.

Top-level pipeline (see ``solve.py`` for orchestration details):

1. Chain subdivision per polyline (``fit_polyline``).
2. Flat parameter manifest assembly.
3. First joint solve with junction-derived hard-ish constraints.
4. Beautification detection + second solve.
5. Eulerian routing → robot commands.

The ``Vectorize`` class is a one-shot pipeline: instantiate it with
inputs and read the result fields. Intermediate state is exposed so
the caller can render diagnostics or feed individual phases into
external visualization.

Result fields:

* ``self.chains`` — initial per-segment chains (pre-solve).
* ``self.primitives_initial`` — flat primitive list before any solve.
* ``self.soft_initial`` — soft constraints from junction translation.
* ``self.primitives_fitted`` — primitives after the first solve.
* ``self.soft_beautified`` — constraints augmented with beautification.
* ``self.primitives_consolidated`` — primitives after second solve.
* ``self.commands_fitted`` — robot commands from ``primitives_fitted``.
* ``self.commands_consolidated`` — robot commands from
  ``primitives_consolidated``.

The two command snapshots and the routing that feeds each:

    snapshot                primitives                tour
    ----------------------  ------------------------  --------------------
    commands_fitted         primitives_fitted         tour
    commands_consolidated   primitives_consolidated   tour_consolidated

``commands_consolidated`` is the intended pipeline output (it reflects
the beautification re-solve and arc-pair merging). ``commands_fitted``
is the pre-beautification snapshot, kept for diagnostics. Neither is
route-time-optimised — that is ``OptimizeRoute``'s job and is a
required final stage (see ``routing.py``: the Eulerian router
minimises pen-ups, not total turning).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, TypedDict

import numpy as np
from numpy.typing import NDArray

# Relative imports from the same subpackage.
from ...commands import DrawingCommand
from ...graph import StrokeGraph

from .beautify import BeautifyTolerances, detect, merge_arc_pairs, merge_into
from .fitting import ChainPiece, fit_polyline, fuse_chain, is_near_closed_polyline
from .labels import CommandSpan, LabeledCommand, label_commands
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
from .primitives import Arc, Circle, Line, Primitive, tangent_at_end
from .residuals import Weights, assemble_residuals
from .routing import order_primitives, to_commands
from .solve import (
    FitConfig,
    FittedSegment,
    SolveConfig,
    SolveResult,
    assign_global_ids,
    build_chains,
    build_junction_constraints,
    solve_once,
    _bbox_diag,
)
from ...segment.labels import LabeledSegment

# ---------------------------------------------------------------------------
# Public configuration (typed-dict form for parity with the rest of the
# codebase's Config namespaces)


class FitDict(TypedDict, total=False):
    line_tol: float
    arc_tol: float
    lam_rel: float
    min_len: int
    use_dp: bool
    max_window: int
    closed_tol: float
    smooth_junction_deg_threshold: float


class SolveDict(TypedDict, total=False):
    weights_data: float
    weights_coincide: float
    weights_on_curve: float
    weights_g1: float
    weights_parallel: float
    weights_perpendicular: float
    weights_equal_radius: float
    weights_concentric: float
    weights_radius_reg: float
    max_iters: int
    method: str  # 'trf' or 'lm'


class BeautifyDict(TypedDict, total=False):
    enabled: bool
    parallel_rad: float
    perp_rad: float
    radius_rel: float
    center_abs: float
    min_radius: float
    min_line_length: float


class RouteDict(TypedDict, total=False):
    snap_tol: float
    pen_up_join_tol: float


# ---------------------------------------------------------------------------


def _fit_config_from(d: Optional[dict]) -> FitConfig:
    cfg = FitConfig()
    if d:
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def _solve_config_from(d: Optional[dict]) -> SolveConfig:
    cfg = SolveConfig()
    if d is None:
        return cfg
    w = Weights()
    name_map = {
        "data": "data",
        "coincide": "coincide",
        "on_curve": "on_curve",
        "g1": "g1",
        "parallel": "parallel",
        "perpendicular": "perpendicular",
        "equal_radius": "equal_radius",
        "concentric": "concentric",
        "radius_reg": "radius_reg",
    }
    for key, attr in name_map.items():
        full = f"weights_{key}"
        if full in d:
            setattr(w, attr, d[full])
    cfg.weights = w
    if "max_iters" in d:
        cfg.max_iters = d["max_iters"]
    if "method" in d:
        cfg.method = d["method"]
    return cfg


def _beautify_tols_from(d: Optional[dict]) -> BeautifyTolerances:
    tol = BeautifyTolerances()
    if d:
        for k, v in d.items():
            if hasattr(tol, k):
                setattr(tol, k, v)
    return tol


# ---------------------------------------------------------------------------


class Vectorize:
    """Orchestrate the full vectorization pipeline.

    Args:
        graph: a built StrokeGraph (polylines + junctions).
        start_pos: robot's starting position (image coordinates).
        start_heading: robot's starting heading, in radians, in the
            x-right / y-down frame (positive = CCW in image = CW on
            screen).
        fit: chain subdivision configuration.
        solve: optimization configuration (per-category weights, etc.).
        beautify: beautification tolerance settings. Set ``enabled``
            to False to skip the second solve.
        route: routing options (endpoint snap tolerance).
    """

    class Config:
        Fit = FitDict
        Solve = SolveDict
        Beautify = BeautifyDict
        Route = RouteDict

    def __init__(
        self,
        graph: StrokeGraph,
        start_pos: NDArray[np.float64],
        start_heading: float = 0.0,
        fit: Optional[FitDict] = None,
        solve: Optional[SolveDict] = None,
        beautify: Optional[BeautifyDict] = None,
        route: Optional[RouteDict] = None,
        labeled_segments: Optional[Sequence[LabeledSegment]] = None,
    ):
        self.graph = graph
        self.start_pos = np.asarray(start_pos, dtype=float)
        self.start_heading = float(start_heading)

        self.fit_config = _fit_config_from(fit)
        self.solve_config = _solve_config_from(solve)
        self.beautify_enabled = beautify is None or beautify.get("enabled", True)
        # Strip the 'enabled' flag before passing to tolerance ctor.
        beautify_clean = (
            {k: v for k, v in beautify.items() if k != "enabled"} if beautify else None
        )
        self.beautify_tols = _beautify_tols_from(beautify_clean)
        self.route_config = dict(route or {})
        # Optional per-pixel raw-segment labels for the polylines this
        # vectorizer was built against. When supplied, the command stream
        # is annotated post-routing with which raw segments each drawing
        # command pulled from — see ``labeled_commands_consolidated``.
        self.input_labeled_segments: Optional[Sequence[LabeledSegment]] = (
            labeled_segments
        )

        self._run()

    # ----------------------------------------------------------------

    def _run(self) -> None:
        # Phase 1: chain subdivision.
        self.fitted_segments: List[FittedSegment] = build_chains(
            self.graph, self.fit_config
        )
        self.primitives_initial: List[Primitive] = assign_global_ids(
            self.fitted_segments
        )

        # Per-primitive source-point map (global ID -> NDArray of source
        # pixels assigned to that primitive).
        source_points: Dict[int, NDArray] = {}
        for seg in self.fitted_segments:
            for pid, src in zip(seg.primitive_ids, seg.source_points):
                source_points[pid] = src
        self.source_points = source_points

        smooth_thresh_rad = math.radians(self.fit_config.smooth_junction_deg_threshold)

        def _build_constraints(
            segments: List[FittedSegment], prims: List[Primitive]
        ) -> SoftConstraints:
            """Junction + internal-chain-joint constraints for the
            current ``(segments, prims)`` state. Called once initially
            and again after the within-chain fusion pass (which
            renumbers primitive ids and changes which primitive ids are
            adjacent inside a chain).

            See the long-form notes baked into the original inline
            version for the Circle vs Arc/Line joint handling and the
            G1 deflection gate.
            """
            soft_ = build_junction_constraints(
                self.graph,
                segments,
                smooth_junction_deg=self.fit_config.smooth_junction_deg_threshold,
                primitives=prims,
            )
            for seg in segments:
                pids = seg.primitive_ids
                for k in range(len(pids) - 1):
                    a_pid = pids[k]
                    b_pid = pids[k + 1]
                    a_is_circle = isinstance(prims[a_pid], Circle)
                    b_is_circle = isinstance(prims[b_pid], Circle)
                    if a_is_circle and not b_is_circle:
                        soft_.on_curve.append(
                            OnCurve(terminating=(b_pid, "start"), host=a_pid)
                        )
                    elif b_is_circle and not a_is_circle:
                        soft_.on_curve.append(
                            OnCurve(terminating=(a_pid, "end"), host=b_pid)
                        )
                    else:
                        soft_.coincide.append(
                            Coincide((a_pid, "end"), (b_pid, "start"))
                        )
                    if a_is_circle or b_is_circle:
                        continue
                    t_end = tangent_at_end(prims[a_pid], "end")
                    t_start = tangent_at_end(prims[b_pid], "start")
                    dot = float(np.clip(np.dot(t_end, t_start), -1.0, 1.0))
                    deflection = math.acos(dot)
                    if deflection < smooth_thresh_rad:
                        soft_.g1.append(
                            G1Smooth(a=a_pid, alpha_a=1.0, b=b_pid, alpha_b=0.0)
                        )

                # Wrap-around joint for near-closed polylines. The graph
                # builder only marks polylines with endpoint gap < 1.5 px
                # as closed (and only those get a self-junction that
                # pulls both ends together). Hand-drawn loops often
                # close with a 2-5 px gap — graph treats one end as a
                # terminal that happens to share a junction with another
                # stroke, leaving the OTHER end floating. Routing then
                # sees the two chain ends as separate vertices and
                # emits a wasteful pen-up + back-track between them
                # (the ghostclock bottom-of-body junction was the
                # canonical case). Adding a Coincide between the
                # chain's first.start and last.end pulls them together
                # via the solver, so the routing's endpoint clustering
                # then collapses them into one vertex.
                if len(pids) >= 2:
                    poly = self.graph.polylines[seg.polyline_index]
                    # NOTE: ``abs_tol`` here is deliberately stricter
                    # (5 px) than ``is_near_closed_polyline``'s default
                    # (10 px, used by ``fit_polyline`` to decide "is
                    # this loop a Circle"). The two tests answer
                    # different questions. The fitting test can afford
                    # to be generous — mis-classifying a 10 px-gap loop
                    # as a Circle is visually fine. This test ADDS A
                    # HARD COINCIDE pulling the chain's two ends
                    # together; doing that on a stroke that wasn't
                    # really meant to close (a 6-10 px gap that is
                    # genuine open geometry) would visibly distort it.
                    # So the wrap-around joint only fires on near-exact
                    # closure.
                    if is_near_closed_polyline(poly, abs_tol=5.0):
                        first_pid = pids[0]
                        last_pid = pids[-1]
                        first_is_circle = isinstance(prims[first_pid], Circle)
                        last_is_circle = isinstance(prims[last_pid], Circle)
                        if not (first_is_circle or last_is_circle):
                            soft_.coincide.append(
                                Coincide((last_pid, "end"), (first_pid, "start"))
                            )
            return soft_

        # Phase 2: junction-derived constraints. Coincide ALWAYS at
        # internal joints — we want consecutive primitives to share an
        # endpoint. G1 ONLY when the joint is actually smooth: top-down
        # splitting puts chain breakpoints at sharp corners (the apex
        # of a cat ear, the cusp of a heart), where the two adjoining
        # primitives have materially different tangents. Adding G1
        # there would round the corner. So we measure tangent
        # deflection and skip G1 past ``smooth_junction_deg_threshold``.
        #
        # EXCEPTION: when one of the two consecutive primitives is a
        # Circle (chain produced by sub-loop extraction — the loop
        # part fits as a Circle, the rest fits as an Arc), Coincide
        # would force the Circle's theta=0 (an arbitrary convention
        # point) to match the Arc's endpoint, which pulls the entire
        # circle to satisfy it (vasesun's sun went from r=90 to r=1077
        # because the solver shifted the center 1500px away to put
        # theta=0 at the stem-top junction). Use OnCurve instead.
        self.soft_initial = _build_constraints(
            self.fitted_segments, self.primitives_initial
        )

        # Phase 3: first solve.
        pos_scale = _bbox_diag(self.graph)
        if self.primitives_initial:
            primitives_fitted, result = solve_once(
                self.primitives_initial,
                source_points,
                self.soft_initial,
                self.solve_config.weights,
                pos_scale=pos_scale,
                max_iters=self.solve_config.max_iters,
                method=self.solve_config.method,
            )
        else:
            primitives_fitted = []
            result = SolveResult([], [], self.soft_initial, True, 0.0, 0)
        self.primitives_fitted = primitives_fitted
        self.solve_result = result

        # Phase 3.5: within-chain fusion.
        #
        # The chain subdivider's per-piece tolerances are tight; on
        # hand-drawn curves with noisy strokes, a gentle arc can fail
        # the single-arc fit while each ~5px sub-window fits a line
        # well. The result is a "line spin line spin …" chain that
        # represents what should have been one arc command. We
        # post-process each chain by greedily fusing runs of
        # consecutive primitives whose union still fits a single
        # Line/Arc within a (looser) tolerance. Chain boundaries are
        # respected — graph junctions stay intact.
        if primitives_fitted:
            new_segments: List[FittedSegment] = []
            for seg in self.fitted_segments:
                chain_prims = [primitives_fitted[pid] for pid in seg.primitive_ids]
                fused_pieces, fused_src = fuse_chain(
                    seg.pieces, seg.source_points, chain_prims
                )
                new_segments.append(
                    FittedSegment(
                        polyline_index=seg.polyline_index,
                        pieces=fused_pieces,
                        primitive_ids=[],  # filled by assign_global_ids below
                        source_points=fused_src,
                    )
                )
            n_before = len(primitives_fitted)
            self.fitted_segments = new_segments
            self.primitives_fitted = assign_global_ids(self.fitted_segments)
            # Refresh the global source-points map with the new ids.
            new_source: Dict[int, NDArray] = {}
            for seg in self.fitted_segments:
                for pid, src in zip(seg.primitive_ids, seg.source_points):
                    new_source[pid] = src
            source_points = new_source
            self.source_points = source_points
            self.n_fused = n_before - len(self.primitives_fitted)
            # Regenerate constraints against the new primitive ids.
            self.soft_initial = _build_constraints(
                self.fitted_segments, self.primitives_fitted
            )
            primitives_fitted = self.primitives_fitted
        else:
            self.n_fused = 0

        # Phase 4: beautification + re-solve.
        if self.beautify_enabled and primitives_fitted:
            additions = detect(primitives_fitted, self.beautify_tols)
            soft_b = SoftConstraints(
                coincide=list(self.soft_initial.coincide),
                on_curve=list(self.soft_initial.on_curve),
                g1=list(self.soft_initial.g1),
                parallel=list(self.soft_initial.parallel),
                perpendicular=list(self.soft_initial.perpendicular),
                equal_radius=list(self.soft_initial.equal_radius),
                concentric=list(self.soft_initial.concentric),
            )
            merge_into(soft_b, additions)
            self.soft_beautified = soft_b
            primitives_consolidated, result_b = solve_once(
                primitives_fitted,
                source_points,
                soft_b,
                self.solve_config.weights,
                pos_scale=pos_scale,
                max_iters=self.solve_config.max_iters,
                method=self.solve_config.method,
            )
            self.primitives_consolidated = primitives_consolidated
            self.solve_result_consolidated = result_b
        else:
            self.soft_beautified = self.soft_initial
            self.primitives_consolidated = primitives_fitted
            self.solve_result_consolidated = self.solve_result

        # Phase 4.5: merge arc pairs that approximate a single circle.
        # The upstream segmenter sometimes breaks a closed shape into
        # two ~180° polylines (e.g. the bird's head outline in
        # birdlove), and each becomes a separate Arc with a slightly
        # different center and radius. Soft EqualRadius / Concentric
        # constraints during the solve nudge them toward agreement
        # but don't fully merge them. This pass actually consolidates
        # the pair into a single Circle when they together cover ≥320°
        # and their endpoints close up.
        consolidated_after_merge, merged_pairs = merge_arc_pairs(
            self.primitives_consolidated
        )
        self.merged_arc_pairs = merged_pairs
        if merged_pairs:
            self.primitives_consolidated = consolidated_after_merge

        # Phase 5: routing + command emission.
        snap_tol = float(self.route_config.get("snap_tol", 1.5))
        pen_up_join_tol = float(self.route_config.get("pen_up_join_tol", 0.5))

        self.tour = order_primitives(
            self.primitives_fitted, self.start_pos, snap_tol=snap_tol
        )
        self.commands_fitted: Sequence[DrawingCommand] = to_commands(
            self.primitives_fitted,
            self.tour,
            self.start_pos,
            self.start_heading,
            pen_up_join_tol=pen_up_join_tol,
        )

        self.tour_consolidated = order_primitives(
            self.primitives_consolidated, self.start_pos, snap_tol=snap_tol
        )
        self.commands_consolidated: Sequence[DrawingCommand] = to_commands(
            self.primitives_consolidated,
            self.tour_consolidated,
            self.start_pos,
            self.start_heading,
            pen_up_join_tol=pen_up_join_tol,
        )

        # Per-command back-pointers: every drawing command gets a list
        # of ``CommandSpan`` runs telling you which raw segments the
        # primitive it draws came from, with start/end indices in the
        # raw segment and start/end ratios along the command's
        # geometry. Non-drawing commands (spins, pen-ups) carry an
        # empty span list. Only produced when the caller supplied
        # ``labeled_segments``; otherwise stays ``None``.
        self.labeled_commands_consolidated: Optional[List[LabeledCommand]] = None
        if self.input_labeled_segments is not None:
            # ``primitives_consolidated`` references fitted_segments' piece
            # ranges, which index into the SUBSAMPLED polyline that
            # ``build_chains`` produced. ``polyline_subsample_cap`` is
            # the same number ``build_chains`` used.
            self.labeled_commands_consolidated = label_commands(
                self.commands_consolidated,
                self.tour_consolidated,
                self.primitives_consolidated,
                self.fitted_segments,
                self.graph.polylines,
                self.input_labeled_segments,
                polyline_subsample_cap=self.fit_config.polyline_subsample_cap,
                start_pos=self.start_pos,
                start_heading=self.start_heading,
                pen_up_join_tol=pen_up_join_tol,
            )

    # ----------------------------------------------------------------
    # Diagnostics

    def primitives_by_polyline(self, which: str = "fitted") -> List[List[Primitive]]:
        """Return per-polyline grouped primitives for diagnostic
        rendering.

        ``which`` ∈ {'initial', 'fitted', 'consolidated'} picks which
        primitive snapshot to slice.
        """
        if which == "initial":
            flat = self.primitives_initial
        elif which == "fitted":
            flat = self.primitives_fitted
        elif which == "consolidated":
            flat = self.primitives_consolidated
        else:
            raise ValueError(f"unknown snapshot {which!r}")

        out: List[List[Primitive]] = []
        for seg in self.fitted_segments:
            out.append([flat[i] for i in seg.primitive_ids])
        return out

    def stats(self) -> str:
        n_lines = sum(1 for p in self.primitives_fitted if isinstance(p, Line))
        n_arcs = sum(1 for p in self.primitives_fitted if isinstance(p, Arc))
        n_circles = sum(1 for p in self.primitives_fitted if isinstance(p, Circle))
        n_pen_ups = sum(
            1 for c in self.commands_fitted if c["kind"] == "line" and not c["penDown"]
        )
        return (
            f"{len(self.primitives_fitted)} primitives "
            f"({n_lines} lines, {n_arcs} arcs, {n_circles} circles) "
            f"in {len(self.fitted_segments)} chains, "
            f"{len(self.commands_fitted)} commands "
            f"({n_pen_ups} pen-ups), "
            f"first-solve cost={self.solve_result.cost:.2f}, "
            f"consolidated cost={self.solve_result_consolidated.cost:.2f}"
        )
