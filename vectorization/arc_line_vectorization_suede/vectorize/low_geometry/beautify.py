"""Beautification pass: detect candidate soft constraints.

After the first solve, scan all primitives for near-relationships and
add them as constraints so the second solve snaps them exact (or close
to it). The four relationships we look for:

* **Parallel.** Two lines whose direction angle differs by < ``parallel_tol`` rad.
* **Perpendicular.** Two lines whose directions differ by < ``perp_tol`` rad from 90°.
* **Equal radius.** Two arcs/circles whose radii differ by less than
  ``radius_tol`` fractionally.
* **Concentric.** Two arcs/circles whose centers are within
  ``center_tol`` absolute pixels.

Tolerances should be calibrated to the upstream noise of hand-drawn
strokes: roughly the natural mode of the residual histogram. Defaults
are starting points; tune on real input.
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple

import numpy as np

from .manifest import (
    Concentric, EqualRadius, Parallel, Perpendicular, SoftConstraints,
)
from .primitives import Arc, Circle, Line, Primitive


@dataclass
class BeautifyTolerances:
    parallel_rad: float = 0.10   # ~5.7°
    perp_rad: float = 0.10
    radius_rel: float = 0.07     # 7% relative radius difference
    center_abs: float = 4.0       # absolute pixels
    min_radius: float = 2.0       # don't beautify tiny arcs
    min_line_length: float = 5.0  # don't beautify tiny lines


def _line_angle(line: Line) -> float:
    """Direction angle in [-π/2, π/2). Lines have no head/tail; +π is
    the same direction as 0, so we fold to a half-circle for
    comparisons.
    """
    d = line.direction()
    return float(np.arctan2(d[1], d[0]))


def _radius(prim: Primitive) -> float:
    if isinstance(prim, Circle):
        return float(prim.radius)
    if isinstance(prim, Arc):
        return float(prim.radius())
    return float("inf")


def _center(prim: Primitive):
    if isinstance(prim, Circle):
        return prim.center
    if isinstance(prim, Arc):
        return prim.center()
    return None


def _arc_angular_coverage_deg(arc: Arc, center: np.ndarray) -> set:
    """The set of integer-degree angular bins (around ``center``) that
    ``arc`` sweeps through.

    Used to test whether a group of arcs *tiles* a full circle. Summing
    arc sweeps is not enough: a crescent's two arcs both sweep the same
    angular sector (just at two radii) so their sweeps sum past 320°
    while their angular *union* is far short of 360°. Measuring the
    union of covered angles rejects that case and generalizes cleanly
    to any number of arcs.

    The covered interval is computed exactly from the start angle and
    sweep (then every integer degree inside it is marked) rather than
    by point-sampling — point-sampling plus integer truncation leaves
    spurious 1° gaps that under-count a genuine full circle.
    """
    import math

    p0 = arc.p0
    a0 = math.degrees(math.atan2(p0[1] - center[1], p0[0] - center[0]))
    sweep = math.degrees(arc.sweep())
    lo = min(a0, a0 + sweep)
    hi = max(a0, a0 + sweep)
    bins = set()
    d = math.floor(lo)
    while d <= hi:
        bins.add(int(d % 360))
        d += 1.0
    return bins


def detect(
    prims: List[Primitive],
    tol: BeautifyTolerances,
) -> SoftConstraints:
    """Return a SoftConstraints bundle of detected candidates.

    Note: only returns the four beautification fields. Caller is
    expected to merge these into the existing constraint bundle.
    """
    out = SoftConstraints.empty()

    lines: List[Tuple[int, Line]] = [
        (i, p) for i, p in enumerate(prims)
        if isinstance(p, Line) and p.length() >= tol.min_line_length
    ]

    for (i, a), (j, b) in combinations(lines, 2):
        ai = _line_angle(a)
        aj = _line_angle(b)
        # Fold to the unsigned smallest angle between two undirected
        # lines (both ranges of length π).
        delta = (ai - aj + np.pi / 2.0) % np.pi - np.pi / 2.0
        absdelta = abs(delta)
        if absdelta < tol.parallel_rad:
            out.parallel.append(Parallel(i, j))
        elif abs(absdelta - np.pi / 2.0) < tol.perp_rad:
            out.perpendicular.append(Perpendicular(i, j))

    curved: List[Tuple[int, Primitive]] = [
        (i, p) for i, p in enumerate(prims)
        if isinstance(p, (Arc, Circle)) and _radius(p) >= tol.min_radius
        and np.isfinite(_radius(p))
    ]

    for (i, a), (j, b) in combinations(curved, 2):
        ra = _radius(a)
        rb = _radius(b)
        denom = max(ra, rb)
        if denom > 0 and abs(ra - rb) / denom < tol.radius_rel:
            out.equal_radius.append(EqualRadius(i, j))
        ca = _center(a)
        cb = _center(b)
        if ca is not None and cb is not None:
            if float(np.linalg.norm(ca - cb)) < tol.center_abs:
                out.concentric.append(Concentric(i, j))

    return out


def merge_into(target: SoftConstraints, additions: SoftConstraints) -> None:
    """Merge beautification additions into ``target`` in place."""
    target.parallel.extend(additions.parallel)
    target.perpendicular.extend(additions.perpendicular)
    target.equal_radius.extend(additions.equal_radius)
    target.concentric.extend(additions.concentric)


def merge_arc_pairs(
    prims: List[Primitive],
    center_tol_rel: float = 1.0,
    radius_tol_rel: float = 0.10,
    endpoint_tol_rel: float = 0.20,
    full_circle_sweep_thresh_deg: float = 320.0,
    sweep_match_tol_deg: float = 40.0,
) -> Tuple[List[Primitive], List[Tuple[int, int]]]:
    """Detect and merge pairs of Arc primitives that approximate two
    halves of the same circle.

    Two common forms appear in real data:

    1. **Chained halves.** Arc A runs from p0 to p1, arc B runs from
       p1' ≈ p1 back to p0' ≈ p0, going the other way around. The two
       chain end-to-end.
    2. **Parallel halves.** Both arcs share start and end endpoints,
       and bulge in opposite directions. This happens when upstream
       segmentation produces TWO polylines tracing the same loop in
       opposite directions (e.g. the bird's head outline in birdlove
       gets split into upper and lower halves, each starting at the
       chord endpoint nearest the beak and ending at the chord
       endpoint nearest the body).

    Either way, the criteria are:
      * radii within ``radius_tol_rel`` fractionally
      * centers within ``center_tol_rel * mean_radius`` (loose because
        each half-fit pulled its center toward its own data)
      * endpoints connect (one of the two patterns above)
      * combined sweep ≥ ``full_circle_sweep_thresh_deg``

    If all hold, the pair is replaced with a single Circle at the
    chord midpoint (which is closer to the true center than either
    individual fit when the two fits split the difference).

    A second pass then merges groups of THREE OR MORE arcs that
    together tile a circle (see the in-body comment): same concentric
    + equal-radius test, but qualification is by angular-union
    coverage rather than a pairwise sweep match.

    Returns ``(new_primitives, merged_pairs)``.
    """
    import math
    n = len(prims)
    if n < 2:
        return list(prims), []

    arc_pids = [
        i for i, p in enumerate(prims)
        if isinstance(p, Arc) and np.isfinite(_radius(p)) and _radius(p) > 2.0
    ]

    used = [False] * n
    merged_pairs: List[Tuple[int, int]] = []
    replacement: dict = {}

    for ii in range(len(arc_pids)):
        i = arc_pids[ii]
        if used[i]:
            continue
        a = prims[i]
        assert isinstance(a, Arc)
        ra = _radius(a)
        ca = _center(a)

        for jj in range(ii + 1, len(arc_pids)):
            j = arc_pids[jj]
            if used[j]:
                continue
            b = prims[j]
            assert isinstance(b, Arc)
            rb = _radius(b)
            cb = _center(b)
            r_mean = 0.5 * (ra + rb)

            if abs(ra - rb) / max(ra, rb) > radius_tol_rel:
                continue
            if float(np.linalg.norm(ca - cb)) > center_tol_rel * r_mean:
                continue

            tol_pt = max(endpoint_tol_rel * r_mean, 3.0)
            d_e_s = float(np.linalg.norm(a.p1 - b.p0))
            d_e_e = float(np.linalg.norm(a.p1 - b.p1))
            d_s_s = float(np.linalg.norm(a.p0 - b.p0))
            d_s_e = float(np.linalg.norm(a.p0 - b.p1))

            chained = (d_e_s < tol_pt and d_s_e < tol_pt) or \
                      (d_e_e < tol_pt and d_s_s < tol_pt)
            # Parallel halves: both endpoints coincide as a pair.
            parallel = (d_s_s < tol_pt and d_e_e < tol_pt) or \
                       (d_s_e < tol_pt and d_e_s < tol_pt)
            if not (chained or parallel):
                continue

            sweep_a_abs = abs(math.degrees(a.sweep()))
            sweep_b_abs = abs(math.degrees(b.sweep()))
            sweep_total = sweep_a_abs + sweep_b_abs
            if sweep_total < full_circle_sweep_thresh_deg:
                continue

            # CRUCIAL: a true circle traced as two halves has
            # |sweep_a| ≈ |sweep_b| ≈ 180° (each half covers half
            # the perimeter). A crescent moon's outer arc wraps more
            # than 180° and its inner arc is well under 180°, so
            # |sweep_a| - |sweep_b| can be 70-100° even when the
            # radii happen to be similar. The eggmoon crescent has
            # radii 103 vs 99 (4% diff, passes the radius gate) but
            # sweeps 224° vs 150° — clearly not the same circle.
            # Require the sweeps to be within sweep_match_tol_deg.
            if abs(sweep_a_abs - sweep_b_abs) > sweep_match_tol_deg:
                continue

            # Compute the geometrically-correct center from the chord
            # midpoint. For the parallel-halves case (both arcs share
            # endpoints), the two fits typically pull their centers
            # in opposite directions along the perpendicular bisector
            # of the chord. The true center lies on that bisector at
            # the right distance, and the simple average of the two
            # noisy fits is usually a much better estimate than
            # either fit alone.
            new_center = 0.5 * (ca + cb)
            new_radius = r_mean

            replacement[i] = Circle(center=new_center, radius=new_radius)
            replacement[j] = None
            used[i] = True
            used[j] = True
            merged_pairs.append((i, j))
            break

    # --- 3+ arc grouping pass ------------------------------------------
    # The loop above only joins arcs two at a time. A closed loop that
    # was chain-subdivided into THREE OR MORE arcs (a wheel split at
    # spurious corners, a circle broken at inflection false-positives)
    # needs a group merge. Generalized criteria, on the arcs the
    # pairwise pass left untouched:
    #   * mutually concentric (centers within center_tol) + equal radius
    #   * angular spans around the shared centre cover ~360° (the union,
    #     not the sum — this is what still rejects crescents)
    #   * every endpoint pairs with another arc's endpoint (a genuine
    #     connected loop, not co-incidentally co-circular arcs)
    remaining = [p for p in arc_pids if not used[p]]
    for ii in range(len(remaining)):
        i = remaining[ii]
        if used[i]:
            continue
        a = prims[i]
        assert isinstance(a, Arc)
        ra = _radius(a)
        ca = _center(a)
        group = [i]
        for j in remaining[ii + 1:]:
            if used[j]:
                continue
            b = prims[j]
            assert isinstance(b, Arc)
            rb = _radius(b)
            cb = _center(b)
            r_mean = 0.5 * (ra + rb)
            if abs(ra - rb) / max(ra, rb) > radius_tol_rel:
                continue
            if float(np.linalg.norm(ca - cb)) > center_tol_rel * r_mean:
                continue
            group.append(j)

        if len(group) < 3:
            continue

        arcs = [prims[g] for g in group]
        radii = [_radius(p) for p in arcs]
        if (max(radii) - min(radii)) / max(radii) > radius_tol_rel:
            continue
        centers = [_center(p) for p in arcs]
        mean_c = np.mean(np.array(centers), axis=0)
        mean_r = float(np.mean(radii))

        covered: set = set()
        total_sweep = 0.0
        for p in arcs:
            assert isinstance(p, Arc)
            covered |= _arc_angular_coverage_deg(p, mean_c)
            total_sweep += abs(math.degrees(p.sweep()))
        # Angular union must tile ~360° (1 bin == 1°), and the summed
        # sweep must stay within a single wrap (no double-covered loop).
        if len(covered) < 340:
            continue
        if total_sweep > 420.0:
            continue

        # Endpoint pairing: every one of the 2k endpoints must sit near
        # an endpoint of a DIFFERENT arc in the group.
        tol_pt = max(endpoint_tol_rel * mean_r, 3.0)
        endpoints = []
        for p in arcs:
            assert isinstance(p, Arc)
            endpoints.append(p.p0)
            endpoints.append(p.p1)
        connected = True
        for ei, e in enumerate(endpoints):
            partner = any(
                (eo // 2) != (ei // 2)
                and float(np.linalg.norm(e - o)) < tol_pt
                for eo, o in enumerate(endpoints)
            )
            if not partner:
                connected = False
                break
        if not connected:
            continue

        # All gates passed — replace the whole group with one Circle.
        replacement[group[0]] = Circle(center=mean_c, radius=mean_r)
        for g in group[1:]:
            replacement[g] = None
        for g in group:
            used[g] = True
        for g in group[1:]:
            merged_pairs.append((group[0], g))

    if not merged_pairs:
        return list(prims), []

    new_prims: List[Primitive] = []
    for pid, p in enumerate(prims):
        if pid in replacement:
            rep = replacement[pid]
            if rep is not None:
                new_prims.append(rep)
        else:
            new_prims.append(p)
    return new_prims, merged_pairs
