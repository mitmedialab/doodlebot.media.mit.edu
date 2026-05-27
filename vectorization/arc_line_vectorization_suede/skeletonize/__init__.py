import numpy as np
from numpy.typing import NDArray
from skimage import io
from skimage.color import rgb2gray, rgba2rgb
from skimage.morphology import skeletonize as _skeletonize

from .cleanup import collapse_small_holes, CollapseConfig
from .crossings import (
    DetectConfig,
    DetectResult,
    detect_crossings,
    resolve_crossings,
)
from .eyes import (
    EyeDetectConfig,
    EyeDetectResult,
    detect_filled_regions,
    resolve_filled_regions,
)
from .labeling import label_skeleton_pixels

from typing import Literal, TypedDict, NamedTuple, Union


class BinarizeConfig(TypedDict):
    threshold: float  # 0.0 to 1.0


ImageSource = Union[str, np.ndarray]


def _to_grayscale_float(img: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary image array to a float grayscale in [0, 1]."""
    if img.ndim == 3:
        if img.shape[-1] == 4:
            img = rgba2rgb(img)  # also yields float in [0, 1]
        if img.shape[-1] == 3:
            return rgb2gray(img)  # float in [0, 1]
        raise ValueError(
            f"Unsupported channel count {img.shape[-1]} for 3D image input; "
            "expected 3 (RGB) or 4 (RGBA)."
        )
    if img.ndim != 2:
        raise ValueError(
            f"Unsupported image ndim {img.ndim}; expected 2 (grayscale) or 3."
        )
    if img.dtype == np.bool_:
        return img.astype(np.float64)
    if np.issubdtype(img.dtype, np.integer):
        return img / np.float64(np.iinfo(img.dtype).max)
    return img.astype(np.float64, copy=False)


def to_binary(source: ImageSource, config: BinarizeConfig) -> NDArray[np.bool_]:
    if isinstance(source, np.ndarray):
        img = _to_grayscale_float(source)
    else:
        img = io.imread(source, as_gray=True)
        if img.dtype != np.float64 and img.dtype != np.float32:
            img = img / 255.0
    return img < config["threshold"]


class SkeletonizeConfig(TypedDict):
    method: Literal["lee", "zhang"]


def skeletonize(
    mask: NDArray[np.bool_], config: SkeletonizeConfig
) -> NDArray[np.bool_]:
    return _skeletonize(mask, method=config["method"])


class Skeletonize:
    """End-to-end skeletonization pipeline.

    Stages:
      1. binarize  -> self.binary
      2. skeletonize -> self.skeletonized
      3. collapse_small_holes -> self.collapsed
            (fixes thin double-pixel artifacts from Lee thinning)
      4. detect_filled_regions -> self.eye_detection
            (finds filled-in shapes whose medial-axis collapsed to a
            tiny pixel cluster — pupils, drawn dots — that segmentation
            would otherwise drop or mis-handle)
      5. resolve_filled_regions -> self.eyes_resolved
            (replaces each filled region's interior skeleton with its
            1-pixel outer boundary, so segmentation traces a clean loop)
      6. detect_crossings -> self.detection
            (identifies ribbon-collapse regions in the binary; uses
            the eye-resolved skeleton so arm endpoints land on actual
            surviving skeleton pixels for downstream resolution)
      7. resolve_crossings -> self.uncrossed
            (rewrites each detected ribbon collapse: erases the merged
            skeleton segment and replaces it with two straight lines
            between paired arm endpoints, restoring two non-intersecting
            paths through the crossing)
      8. label_skeleton_pixels -> self.labeling
            (assigns every binary pixel to the single most appropriate
            pixel of ``self.uncrossed`` via geodesic flooding through
            the stroke mask — the lookup that lets downstream stages
            trace each pixel of the original image to its skeleton
            ancestor; -1 for pixels outside the stroke mask)

    The final skeleton for downstream consumption is `self.uncrossed`.
    Earlier stages are kept on the instance so visualization / debug
    can compare them.
    """

    class Config:
        class Binarize(BinarizeConfig):
            pass

        class Skeletonize(SkeletonizeConfig):
            pass

        class Collapse(CollapseConfig):
            pass

        class Eyes(EyeDetectConfig):
            pass

        class Detect(DetectConfig):
            pass

    class Output(NamedTuple):
        binary: NDArray[np.bool_]
        skeletonized: NDArray[np.bool_]
        collapsed: NDArray[np.bool_]
        eye_detection: EyeDetectResult
        eyes_resolved: NDArray[np.bool_]
        detection: DetectResult
        uncrossed: NDArray[np.bool_]
        labeling: NDArray[np.int32]

    def __init__(
        self,
        source: ImageSource,
        binarize_config: Config.Binarize,
        skeletonize_config: Config.Skeletonize,
        collapse_config: Config.Collapse,
        eyes_config: Config.Eyes,
        detect_config: Config.Detect,
    ):
        self.binary = to_binary(source, binarize_config)
        self.skeletonized = skeletonize(self.binary, skeletonize_config)
        self.collapsed = collapse_small_holes(self.skeletonized, collapse_config)
        # Eye detection runs against the binary using the cleaned
        # skeleton as the "is there a real stroke here" reference.
        # We resolve detected fills BEFORE crossing detection so that
        # crossing detection sees the same skeleton the rest of the
        # pipeline does.
        self.eye_detection = detect_filled_regions(
            self.binary, self.collapsed, eyes_config
        )
        self.eyes_resolved = resolve_filled_regions(
            self.collapsed, self.eye_detection
        )
        # Detection uses the BINARY for its distance-transform analysis
        # but is given the EYE-RESOLVED skeleton so arm endpoints land
        # on pixels that will survive the resolver's edits.
        self.detection = detect_crossings(
            self.binary, detect_config, skel=self.eyes_resolved
        )
        # Resolution rewrites the eye-resolved skeleton in place at
        # every detected crossing.
        self.uncrossed = resolve_crossings(self.eyes_resolved, self.detection)
        # Per-pixel labeling — every binary pixel gets the flat index
        # of its associated final-skeleton pixel; -1 outside the stroke.
        self.labeling = label_skeleton_pixels(self.binary, self.uncrossed)

        self.output = self.Output(
            binary=self.binary,
            skeletonized=self.skeletonized,
            collapsed=self.collapsed,
            eye_detection=self.eye_detection,
            eyes_resolved=self.eyes_resolved,
            detection=self.detection,
            uncrossed=self.uncrossed,
            labeling=self.labeling,
        )
