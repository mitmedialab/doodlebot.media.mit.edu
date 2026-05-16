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
* ``self.commands`` — robot commands from ``primitives_fitted``.
* ``self.consolidated`` — robot commands from ``primitives_consolidated``.
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
from .fitting import ChainPiece, fit_polyline
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

        # Phase 2: junction-derived constraints.
        soft = build_junction_constraints(
            self.graph,
            self.fitted_segments,
            smooth_junction_deg=self.fit_config.smooth_junction_deg_threshold,
            primitives=self.primitives_initial,
        )
        # Add internal chain joint constraints. Coincide ALWAYS — we
        # want consecutive primitives to share an endpoint. G1 ONLY
        # when the joint is actually smooth: top-down splitting puts
        # chain breakpoints at sharp corners (the apex of a cat ear,
        # the cusp of a heart), where the two adjoining primitives
        # have materially different tangents. Adding G1 there forces
        # the joint to smooth out — which rounds off the corner and
        # turns a triangular ear into a trapezoidal blob. So we
        # measure the tangent deflection at the joint and skip G1
        # past the threshold.
        #
        # EXCEPTION: when one of the two consecutive primitives is a
        # Circle (chain produced by sub-loop extraction — the loop
        # part fits as a Circle, the rest fits as an Arc), Coincide
        # would force the Circle's theta=0 to match the Arc's
        # endpoint. theta=0 is the arbitrary convention point on the
        # Circle (center + (r, 0)), so a hard match there pulls the
        # entire circle to satisfy it (vasesun's sun went from r=90
        # to r=1077 because the solver shifted the center 1500px
        # away to put theta=0 at the stem-top junction). Use
        # OnCurve instead — the Arc's endpoint must lie SOMEWHERE
        # on the Circle's perimeter, not at a specific point.
        smooth_thresh_rad = math.radians(self.fit_config.smooth_junction_deg_threshold)
        for seg in self.fitted_segments:
            pids = seg.primitive_ids
            for k in range(len(pids) - 1):
                a_pid = pids[k]
                b_pid = pids[k + 1]
                a_is_circle = isinstance(self.primitives_initial[a_pid], Circle)
                b_is_circle = isinstance(self.primitives_initial[b_pid], Circle)
                if a_is_circle and not b_is_circle:
                    soft.on_curve.append(
                        OnCurve(terminating=(b_pid, "start"), host=a_pid)
                    )
                elif b_is_circle and not a_is_circle:
                    soft.on_curve.append(
                        OnCurve(terminating=(a_pid, "end"), host=b_pid)
                    )
                elif a_is_circle and b_is_circle:
                    # Two adjacent Circles in a chain is unusual but
                    # not impossible. Treat as concentric (their centers
                    # coincide) rather than endpoint-coincide.
                    soft.coincide.append(Coincide((a_pid, "end"), (b_pid, "start")))
                else:
                    soft.coincide.append(Coincide((a_pid, "end"), (b_pid, "start")))
                # G1 only applies between primitives with meaningful
                # tangent endpoints (skip if either is a Circle, since
                # tangent_at_end on a Circle is at the arbitrary theta=0
                # point and isn't meaningful for chain joints).
                if a_is_circle or b_is_circle:
                    continue
                t_end = tangent_at_end(self.primitives_initial[a_pid], "end")
                t_start = tangent_at_end(self.primitives_initial[b_pid], "start")
                dot = float(np.clip(np.dot(t_end, t_start), -1.0, 1.0))
                deflection = math.acos(dot)
                if deflection < smooth_thresh_rad:
                    soft.g1.append(G1Smooth(a=a_pid, alpha_a=1.0, b=b_pid, alpha_b=0.0))
        self.soft_initial = soft

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
            result = SolveResult([], [], soft, True, 0.0, 0)
        self.primitives_fitted = primitives_fitted
        self.solve_result = result

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
        self.commands: Sequence[DrawingCommand] = to_commands(
            self.primitives_fitted,
            self.tour,
            self.start_pos,
            self.start_heading,
            pen_up_join_tol=pen_up_join_tol,
        )

        self.tour_consolidated = order_primitives(
            self.primitives_consolidated, self.start_pos, snap_tol=snap_tol
        )
        self.consolidated: Sequence[DrawingCommand] = to_commands(
            self.primitives_consolidated,
            self.tour_consolidated,
            self.start_pos,
            self.start_heading,
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
            1 for c in self.commands if c["kind"] == "line" and not c["penDown"]
        )
        return (
            f"{len(self.primitives_fitted)} primitives "
            f"({n_lines} lines, {n_arcs} arcs, {n_circles} circles) "
            f"in {len(self.fitted_segments)} chains, "
            f"{len(self.commands)} commands "
            f"({n_pen_ups} pen-ups), "
            f"first-solve cost={self.solve_result.cost:.2f}, "
            f"consolidated cost={self.solve_result_consolidated.cost:.2f}"
        )
