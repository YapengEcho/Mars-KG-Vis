# -*- coding: utf-8 -*-
"""Shared Stage2 feature composition and CTX-to-MOLA extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

import terrain_features as terrain


BASE = Path(__file__).resolve().parents[2]
CTX_ROOT = BASE / "data/raw/ctx_tiles"
FEATURE_MODES = ("ctx", "terrain", "ctx_terrain")


def read_geotransform(path: Path) -> tuple[float, float, float, float, float, float]:
    """Read the north-up GeoTIFF affine transform used by CTX and MOLA."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        scale = page.tags[33550].value
        tie = page.tags[33922].value
    return float(tie[3]), float(scale[0]), 0.0, float(tie[4]), 0.0, -float(scale[1])


def compose_features(backbone, geom, conf, terrain_values, mode: str) -> np.ndarray:
    """Compose one of the three frozen Stage2 feature variants."""
    if mode not in FEATURE_MODES:
        raise ValueError(f"unknown Stage2 feature mode: {mode}")
    geom = np.asarray(geom, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32).reshape(-1, 1)
    terrain_values = np.asarray(terrain_values, dtype=np.float32)
    if geom.ndim != 2 or geom.shape[1] != 5:
        raise ValueError("geom must have shape (N, 5)")
    if terrain_values.ndim != 2 or terrain_values.shape[1] != len(terrain.TERRAIN_FEATURE_NAMES):
        raise ValueError("terrain_values has an unexpected feature dimension")
    n = geom.shape[0]
    if conf.shape[0] != n or terrain_values.shape[0] != n:
        raise ValueError("feature row counts do not match")
    common = np.concatenate([geom, conf], axis=1)
    if mode == "terrain":
        return np.concatenate([common, terrain_values], axis=1).astype(np.float32)
    backbone = np.asarray(backbone, dtype=np.float32)
    if backbone.ndim != 2 or backbone.shape[0] != n:
        raise ValueError("backbone row count does not match")
    if mode == "ctx":
        return np.concatenate([backbone, common], axis=1).astype(np.float32)
    return np.concatenate([backbone, common, terrain_values], axis=1).astype(np.float32)


def feature_names(mode: str, backbone_dim: int = 448) -> tuple[str, ...]:
    """Return stable column names stored alongside each Stage2 dataset."""
    if mode not in FEATURE_MODES:
        raise ValueError(f"unknown Stage2 feature mode: {mode}")
    names = []
    if mode != "terrain":
        names.extend(f"backbone_{i:03d}" for i in range(backbone_dim))
    names.extend(("box_w_norm", "box_h_norm", "box_aspect", "box_log_area", "box_area_norm", "stage1_conf"))
    if mode != "ctx":
        names.extend(terrain.TERRAIN_FEATURE_NAMES)
    return tuple(names)


class TerrainExtractor:
    """Read MOLA once and map generated patch boxes back to terrain windows."""

    def __init__(self, dem_path: Path, ctx_root: Path = CTX_ROOT, padding_px: int = 2):
        self.dem_path = Path(dem_path)
        self.ctx_root = Path(ctx_root)
        self.padding_px = int(padding_px)
        if not self.dem_path.exists():
            raise FileNotFoundError(self.dem_path)
        with tifffile.TiffFile(self.dem_path) as tif:
            page = tif.pages[0]
            self.dem_shape = tuple(int(v) for v in tif.series[0].shape[:2])
            self.dem_pixel_m = float(page.tags[33550].value[0])
            self.dem_nodata = int(page.tags["GDAL_NODATA"].value) if page.tags.get("GDAL_NODATA") else None
        self.dem = tifffile.memmap(self.dem_path)
        self.dem_gt = read_geotransform(self.dem_path)
        self._ctx_gt: dict[str, tuple[float, ...]] = {}

    def _ctx_transform(self, tile: str) -> tuple[float, ...]:
        if tile not in self._ctx_gt:
            matches = sorted(self.ctx_root.glob(f"MurrayLab_GlobalCTXMosaic_V01_{tile}/*_Mosaic.tif"))
            if not matches:
                matches = sorted(self.ctx_root.glob(f"MurrayLab_GlobalCTXMosaic_V01_{tile}/**/*_Mosaic.tif"))
            if not matches:
                raise FileNotFoundError(f"CTX mosaic not found for {tile}")
            self._ctx_gt[tile] = read_geotransform(matches[0])
        return self._ctx_gt[tile]

    def for_patch(self, stem: str, box, patch_px: int = 1024) -> np.ndarray:
        meta = terrain.parse_patch_stem(stem)
        ctx_bbox = terrain.patch_box_to_ctx(meta, box, patch_px=patch_px)
        dem_window = terrain.ctx_bbox_to_dem_window(
            ctx_bbox,
            self._ctx_transform(meta.tile),
            self.dem_gt,
            self.dem_shape,
            padding_px=self.padding_px,
        )
        left, top, right, bottom = dem_window
        return terrain.terrain_features(
            np.asarray(self.dem[top:bottom, left:right]),
            pixel_m=self.dem_pixel_m,
            nodata=self.dem_nodata,
        )
