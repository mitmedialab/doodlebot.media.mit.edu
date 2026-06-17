from .skeletonize import Skeletonize, ImageSource
from .segment import Segment
from .graph import StrokeGraph
from .vectorize.low_geometry import Vectorize as LowGeometryVectorize
from .vectorize.high_geometry import Vectorize as HighGeometryVectorize
from .optimize import OptimizeRoute
from .commands import DrawingCommand
from .auto_config import derive_configs, estimate_stroke_width

import numpy as np


def default_pipeline(source: ImageSource):
    # Every numerical threshold below is derived from the input image's
    # estimated stroke width. The hand-tuned defaults for this codebase
    # were calibrated against ~3 px strokes on ~1k canvases; ``derive_
    # configs`` scales pixel distances (and areas, where appropriate)
    # so a smaller or larger drawing gets the same *geometric* tolerance
    # rather than the same absolute pixel value. See ``auto_config.py``
    # for which params scale and which deliberately don't.
    cfg = derive_configs(source)

    skeleton = Skeletonize(
        source,
        Skeletonize.Config.Binarize(**cfg["binarize"]),
        Skeletonize.Config.Skeletonize(**cfg["skeletonize"]),
        Skeletonize.Config.Collapse(**cfg["collapse"]),
        Skeletonize.Config.Eyes(**cfg["eyes"]),
        detect_config=cfg["detect"],
    )

    segment = Segment(
        skeleton.uncrossed,
        skeleton.binary,
        Segment.Config.Segment(**cfg["segment"]),
        Segment.Config.Fuse(**cfg["fuse"]),
        Segment.Config.Repair(**cfg["repair"]),
        Segment.Config.PostRepairFuse(**cfg["post_repair_fuse"]),
    )

    graph = StrokeGraph(
        segment.fused_post_repair,
        StrokeGraph.Config.Build(**cfg["graph_build"]),
    )

    start_pos = np.array([0.0, 0.0])
    start_heading = 0.0
    low_geometry = LowGeometryVectorize(
        graph,
        start_pos=start_pos,
        start_heading=start_heading,
        labeled_segments=segment.labeled_segments,
    )
    high_geometry = HighGeometryVectorize(
        segment.fused_post_repair,
        start_pos=start_pos,
        start_heading=start_heading,
        commands=HighGeometryVectorize.Config.ToCommands(**cfg["high_geometry_commands"]),
        raw_segments=segment.segments,
    )

    optimized_low_geometry = OptimizeRoute(
        low_geometry.commands_consolidated,
        start_pos=start_pos,
        start_heading=start_heading,
        cfg=OptimizeRoute.Config.Optimize(**cfg["optimize_route"]),
    )

    optimized_high_geometry = OptimizeRoute(
        high_geometry.commands,
        start_pos=start_pos,
        start_heading=start_heading,
        cfg=OptimizeRoute.Config.Optimize(**cfg["optimize_route"]),
    )

    return (
        skeleton,
        segment,
        graph,
        low_geometry,
        high_geometry,
        optimized_low_geometry,
        optimized_high_geometry,
    )
