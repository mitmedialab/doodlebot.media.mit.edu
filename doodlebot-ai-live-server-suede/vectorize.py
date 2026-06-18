from .arc_line_vectorization_suede import default_pipeline, DrawingCommand
from .arc_line_vectorization_suede.visualize import commands_to_svg, commands_to_svg_compare

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from PIL import Image

import asyncio
from functools import wraps
import io
from typing import Sequence

router = APIRouter()


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
    _, _, _, _, _, low_geometry_optimized, high_geometry_optimized = default_pipeline(
        image_array
    )
    svg = commands_to_svg_compare(
        low_geometry_optimized.commands,
        high_geometry_optimized.commands,
        label_a="low_geometry_optimized.commands",
        label_b="high_geometry_optimized.commands",
    )
    return {
        "low_geometry": _commands_to_jsonable(low_geometry_optimized.commands),
        "high_geometry": _commands_to_jsonable(high_geometry_optimized.commands),
        "svg": svg,
        # The low-geometry commands are what we actually publish to a robot, so
        # render them on their own — this is what the gallery shows after the
        # eager vectorize completes.
        "low_geometry_svg": commands_to_svg(low_geometry_optimized.commands),
    }


class VectorizationError(Exception):
    """Custom exception for vectorization errors"""

    pass


def handle_errors(func):
    """Decorator for error handling"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except VectorizationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )

    return wrapper


@router.post("/vectorize")
@handle_errors
async def vectorize_endpoint(image_file: UploadFile = File(...)):
    """Vectorize an uploaded image into robot drawing commands.

    Returns the low-geometry consolidated commands, the high-geometry
    commands, and a side-by-side comparison SVG of the two.
    """
    image_bytes = await image_file.read()
    if not image_bytes:
        raise VectorizationError("Empty image upload")
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.load()
    except Exception as e:
        raise VectorizationError(f"Could not decode image: {e}")
    image_array = np.asarray(pil_image)
    return await asyncio.to_thread(run_vectorization, image_array)
