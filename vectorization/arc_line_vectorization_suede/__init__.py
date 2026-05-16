from .skeletonize import Skeletonize, ImageSource
from .segment import Segment
from .graph import StrokeGraph
from .vectorize.low_geometry import Vectorize as LowGeometryVectorize
from .vectorize.high_geometry import Vectorize as HighGeometryVectorize
from .commands import DrawingCommand

import numpy as np


def default_pipeline(source: ImageSource):
    skeleton = Skeletonize(
        source,
        Skeletonize.Config.Binarize(threshold=0.5),
        Skeletonize.Config.Skeletonize(method="zhang"),
        Skeletonize.Config.Collapse(
            skeletonize_method="lee",
            max_hole_area=10,
            max_thin_thickness=3.0,
            reskeletonize=True,
        ),
        detect_config={
            "local_tau_radius": 40,
            "fat_ratio": 1.3,
            "min_fat_area": 8,
            "group_dilate": 15,
            "skel_ring_dilate": 5,
            "pairing_tangent_steps": 8,
            "pairing_threshold": 1.2,
            "min_chromosome_skel_length": 15,
        },
    )

    junction_tol = 2.5
    tangent_sample = 10

    segment = Segment(
        skeleton.uncrossed,
        skeleton.binary,
        Segment.Config.Segment(min_length=10.0),
        Segment.Config.Fuse(
            max_path_length=20,
            lookback=10,
            min_tangent_score=0.5,
            gap_penalty=0.05,
            curvature_penalty=3.0,
        ),
        Segment.Config.Repair(
            junction_tol=junction_tol,
            stable_skip=2,
            stable_sample=6,
            max_junction_region_length=20,
            min_output_polyline_length=2,
            min_tangent_spread_deg=15.0,
            interp_max_spacing=1.0,
            min_curvature_spike_ratio=2.0,
            curvature_context_window=8,
        ),
        Segment.Config.PostRepairFuse(
            junction_tol=junction_tol,
            tangent_skip=2,
            tangent_sample=tangent_sample,
            min_tangent_score=0.6,
            curvature_penalty=1.0,
        ),
    )

    graph = StrokeGraph(
        segment.fused_post_repair,
        StrokeGraph.Config.Build(
            junction_tol=junction_tol,  # match Repair.junction_tol
            terminal_tangent_window=10,  # should match fuse.lookback
            crossing_tangent_skip=2,  # baseline; dynamic walk handles arbitrary bridges
            crossing_tangent_half_window=6,
            cusp_angle_threshold_deg=50.0,  # raise to ~50 to handle bikelove's cusp-like junction
            cluster_merge_centroid_distance=10.0,
            cluster_merge_index_gap=10,
        ),
    )

    start_pos = np.array([0.0, 0.0])
    start_heading = 0.0
    low_geometry = LowGeometryVectorize(
        graph,
        start_pos=start_pos,
        start_heading=start_heading,
    )
    high_geometry = HighGeometryVectorize(
        segment.fused_post_repair,
        start_pos=start_pos,
        start_heading=start_heading,
        commands=HighGeometryVectorize.Config.ToCommands(
            sigma=2.0,
            corner_threshold=0.25,
            max_fit_residual=5.0,
        ),
        consolidate=HighGeometryVectorize.Config.Consolidate(
            center_tol_rel=0.25,
            radius_tol_rel=0.25,
            center_tol_abs=3.0,
            radius_tol_abs=3.0,
            max_endpoint_snap_rel=0.15,
            max_endpoint_snap_abs=6.0,
            proximity_min_radius_ratio=0.4,
            line_angle_tol_deg=6.0,
            line_offset_tol_abs=5.0,
            min_line_length=5.0,
            max_line_endpoint_snap_abs=5.0,
            junction_epsilon=3.0,
            merge_arcs=True,
            merge_lines=True,
            return_report=False,
        ),
    )

    return skeleton, segment, graph, low_geometry, high_geometry
