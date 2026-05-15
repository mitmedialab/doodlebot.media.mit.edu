import numpy as np

from typing import Sequence

from arc_line_vectorization_suede import default_pipeline, DrawingCommand
from arc_line_vectorization_suede.visualize import commands_to_svg_compare


def _commands_to_jsonable(commands: Sequence[DrawingCommand]):
    """Strip numpy scalar wrappers so DrawingCommand dicts serialize cleanly."""
    out = []
    for cmd in commands:
        item = {}
        for key, value in cmd.items():
            if isinstance(value, np.bool_):
                item[key] = bool(value)
            elif isinstance(value, np.integer):
                item[key] = int(value)
            elif isinstance(value, np.floating):
                item[key] = float(value)
            else:
                item[key] = value
        out.append(item)
    return out


def run_vectorization(image_array: np.ndarray):
    _, _, _, low_geometry, high_geometry = default_pipeline(image_array)
    svg = commands_to_svg_compare(
        low_geometry.consolidated,
        high_geometry.commands,
        label_a="low_geometry.consolidated",
        label_b="high_geometry.commands",
    )
    return {
        "low_geometry_consolidated": _commands_to_jsonable(low_geometry.consolidated),
        "high_geometry_commands": _commands_to_jsonable(high_geometry.commands),
        "svg": svg,
    }


class VectorizationError(Exception):
    """Custom exception for vectorization errors"""

    pass
