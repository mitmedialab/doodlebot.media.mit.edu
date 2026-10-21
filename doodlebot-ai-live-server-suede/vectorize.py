from .svg import DrawingCommand

# Rust port of the arc/line vectorizer (formerly the vendored
# arc_line_vectorization_suede package). Prebuilt wheels are pinned per
# platform in pyproject.toml; see
# https://github.com/pmalacho-mit/suede.doodlebot-vectorize
import arcline_vectorize

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from PIL import Image

import asyncio
from functools import wraps
import io
from typing import TypedDict, cast

router = APIRouter()


class VectorizationResult(TypedDict):
    """Return shape of :func:`run_vectorization`.

    ``low_geometry`` / ``high_geometry`` are the drawing commands as plain,
    JSON-able dicts (the vectorizer's ``DrawingCommand`` *TypedDict* form) —
    NOT the Pydantic ``DrawingCommand`` models used by ``robots.py``. Callers
    that dispatch to a robot must parse them into those models first (see
    ``robots.parse_commands``).

    ``svg`` is a side-by-side low/high comparison; ``low_geometry_svg`` renders
    the low-geometry commands on their own — those are what we actually
    publish to a robot, so it's what the gallery shows after the eager
    vectorize completes."""

    low_geometry: list[DrawingCommand]
    high_geometry: list[DrawingCommand]
    svg: str
    low_geometry_svg: str


def run_vectorization(image_array: np.ndarray) -> VectorizationResult:
    """Vectorize ``np.asarray(pil_image)`` (grayscale, RGB or RGBA, 8-bit).

    The extension takes a row-major RGBA8 buffer. Alpha is only honoured for
    real RGBA input (composited over white); RGB/grayscale get an opaque alpha
    channel, matching how the original Python pipeline treated them. The
    extension releases the GIL, so ``asyncio.to_thread`` runs it in parallel.
    """
    arr = np.ascontiguousarray(image_array)
    if arr.dtype != np.uint8:
        raise ValueError(f"expected an 8-bit image, got dtype {arr.dtype}")
    h, w = arr.shape[:2]
    opaque = np.full((h, w), 255, np.uint8)
    if arr.ndim == 2:
        rgba, has_alpha = np.dstack([arr, arr, arr, opaque]), False
    elif arr.ndim == 3 and arr.shape[2] == 3:
        rgba, has_alpha = np.dstack([arr, opaque]), False
    elif arr.ndim == 3 and arr.shape[2] == 4:
        rgba, has_alpha = arr, True
    else:
        raise ValueError(f"unsupported image shape {arr.shape}")
    return cast(
        VectorizationResult,
        arcline_vectorize.vectorize(rgba.tobytes(), w, h, has_alpha),
    )


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
