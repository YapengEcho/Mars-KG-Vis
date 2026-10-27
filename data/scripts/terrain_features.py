# -*- coding: utf-8 -*-
"""CTX patch coordinate mapping and compact MOLA terrain features.

The module deliberately contains no YOLO or model code. It keeps the
geospatial conversion and terrain statistics independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np
from scipy import ndimage


PATCH_RE = re.compile(r"^(?P<tile>E-?\d{3}_N-?\d{2})_(?P<x>\d{5})_(?P<y>\d{5})$")
CENTER_RE = re.compile(
    r"^(?P<tile>E-?\d{3}_N-?\d{2})_big_(?P<sid>\d+)_(?P<x>\d{5})_(?P<y>\d{5})_(?P<size>\d+)$"
)


@dataclass(frozen=True)
class PatchMeta:
    tile: str
    kind: str
    x: int
    y: int
    source_size: int
    source_id: int | None = None


# Keep this order stable: it is stored in Stage2 dataset metadata and used by
# downstream reports to identify terrain-only and fused feature columns.
TERRAIN_FEATURE_NAMES = (
    "elevation_mean_m",
    "elevation_std_m",
    "slope_mean_deg",
    "slope_std_deg",
    "slope_p95_deg",
    "aspect_sin_mean",
    "aspect_cos_mean",
    "curvature_mean",
    "curvature_std",
    "relief_p50_m",
    "relief_p95_m",
    "roughness_mean_m",
    "roughness_p95_m",
    "nodata_fraction",
)


def parse_patch_stem(stem: str) -> PatchMeta:
    """Parse a generated patch name into its source CTX window."""
    match = CENTER_RE.fullmatch(stem)
    if match:
        return PatchMeta(
            tile=match.group("tile"),
            kind="center",
            x=int(match.group("x")),
            y=int(match.group("y")),
            source_size=int(match.group("size")),
            source_id=int(match.group("sid")),
        )
    match = PATCH_RE.fullmatch(stem)
    if match:
        return PatchMeta(
            tile=match.group("tile"),
            kind="win",
            x=int(match.group("x")),
            y=int(match.group("y")),
            source_size=1024,
        )
    raise ValueError(f"unrecognized generated patch filename: {stem}")


def patch_box_to_ctx(meta: PatchMeta, box, patch_px: int = 1024) -> tuple[float, float, float, float]:
    """Map a normalized YOLO box in a generated patch to CTX source pixels."""
    _cls, nx, ny, nw, nh = [float(value) for value in box]
    scale = float(meta.source_size) / float(patch_px)
    cx = float(meta.x) + nx * patch_px * scale
    cy = float(meta.y) + ny * patch_px * scale
    width = nw * patch_px * scale
    height = nh * patch_px * scale
    return cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0


def _world_from_pixel(x: float, y: float, gt: tuple[float, ...]) -> tuple[float, float]:
    return gt[0] + x * gt[1], gt[3] + y * gt[5]


def _pixel_from_world(x: float, y: float, gt: tuple[float, ...]) -> tuple[float, float]:
    return (x - gt[0]) / gt[1], (y - gt[3]) / gt[5]


def ctx_bbox_to_dem_window(
    ctx_bbox: tuple[float, float, float, float],
    ctx_gt: tuple[float, ...],
    dem_gt: tuple[float, ...],
    dem_shape: tuple[int, int],
    padding_px: int = 2,
    min_size: int = 5,
) -> tuple[int, int, int, int]:
    """Map a CTX pixel bbox to a clipped DEM window (left, top, right, bottom)."""
    x0, y0, x1, y1 = ctx_bbox
    world = [_world_from_pixel(x, y, ctx_gt) for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
    dem_pixels = [_pixel_from_world(x, y, dem_gt) for x, y in world]
    cols = [point[0] for point in dem_pixels]
    rows = [point[1] for point in dem_pixels]
    height, width = dem_shape
    left = math.floor(min(cols)) - padding_px
    top = math.floor(min(rows)) - padding_px
    right = math.ceil(max(cols)) + padding_px + 1
    bottom = math.ceil(max(rows)) + padding_px + 1
    if right - left < min_size:
        extra = min_size - (right - left)
        left -= extra // 2
        right += extra - extra // 2
    if bottom - top < min_size:
        extra = min_size - (bottom - top)
        top -= extra // 2
        bottom += extra - extra // 2
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    if right <= left or bottom <= top:
        raise ValueError("CTX bbox maps outside DEM extent")
    return int(left), int(top), int(right), int(bottom)


def terrain_features(elevation: np.ndarray, pixel_m: float, nodata: int | float | None = -32768) -> np.ndarray:
    """Compute finite, compact terrain statistics from a small DEM window."""
    values = np.asarray(elevation, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 3:
        raise ValueError("terrain window must be a 2-D array at least 3x3")
    valid = np.isfinite(values)
    if nodata is not None:
        valid &= values != float(nodata)
    if not np.any(valid):
        raise ValueError("NoData terrain window has no valid pixels")

    valid_values = values[valid]
    filled = values.copy()
    filled[~valid] = float(np.median(valid_values))
    spacing = float(pixel_m)
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("pixel_m must be positive")

    dz_dy, dz_dx = np.gradient(filled, spacing, spacing)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    aspect = np.arctan2(-dz_dx, dz_dy)
    d2x = np.gradient(np.gradient(filled, spacing, axis=1), spacing, axis=1)
    d2y = np.gradient(np.gradient(filled, spacing, axis=0), spacing, axis=0)
    curvature = d2x + d2y
    relief = ndimage.maximum_filter(filled, size=3, mode="nearest") - ndimage.minimum_filter(
        filled, size=3, mode="nearest"
    )
    local_mean = ndimage.uniform_filter(filled, size=3, mode="nearest")
    local_sq_mean = ndimage.uniform_filter(filled * filled, size=3, mode="nearest")
    roughness = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))

    def mean(array: np.ndarray) -> float:
        return float(np.mean(array[valid]))

    def std(array: np.ndarray) -> float:
        return float(np.std(array[valid]))

    def percentile(array: np.ndarray, q: float) -> float:
        return float(np.percentile(array[valid], q))

    result = np.asarray(
        [
            mean(filled),
            std(filled),
            mean(slope),
            std(slope),
            percentile(slope, 95),
            mean(np.sin(aspect)),
            mean(np.cos(aspect)),
            mean(curvature),
            std(curvature),
            percentile(relief, 50),
            percentile(relief, 95),
            mean(roughness),
            percentile(roughness, 95),
            float(np.mean(~valid)),
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("terrain features contain NaN or Inf")
    return result
