# -*- coding: utf-8 -*-
"""Generate centered SHP review crops for a selected set of CTX tiles.

This is a review-only artifact.  It creates two images for every
``tile/source_id`` pair:

* ``center_poly``: every intersecting SHP polygon boundary;
* ``center_aabb``: the same crop with every intersecting polygon's AABB.

The two views deliberately share the exact same crop window.  The crop size
is adaptive to the target AABB; it is not hard-coded to the 8 km benchmark
window.  No training manifest is modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import shapefile
import tifffile
from PIL import Image, ImageDraw, ImageFont


BASE = Path(__file__).resolve().parents[2]
RAW = BASE / "data" / "raw"
CTX_DIR = RAW / "ctx_tiles"
SHP = RAW / "mars_landslide_boundaries" / "landslide_boundaries" / "火星滑坡-带边界.shp"
PLAN = RAW / "tile_split_plan_today20_new_only.json"
OUT_DIR = BASE / "data" / "processed" / "centered_review_today20"
R_MARS = 3396190.0
DEG = math.pi / 180.0


def discover_tifs(root: Path) -> dict[str, Path]:
    """Return ``tile -> GeoTIFF`` for complete-looking downloaded tiles."""
    found: dict[str, Path] = {}
    for tif in root.glob("*/MurrayLab_CTX_V01_*.tif"):
        match = re.search(r"(E-?\d{3}_N-?\d{2})", tif.name)
        if match:
            found[match.group(1)] = tif
    return found


def read_geotransform(tif_path: Path) -> tuple[tuple[float, ...], int, int]:
    """Read the GeoTIFF affine transform and raster dimensions."""
    with tifffile.TiffFile(str(tif_path)) as tif:
        page = tif.pages[0]
        scale = page.tags[33550].value
        tie = page.tags[33922].value
        height, width = page.shape[:2]
    return (tie[3], scale[0], 0.0, tie[4], 0.0, -scale[1]), width, height


def point_to_px(lon: float, lat: float, gt: tuple[float, ...]) -> tuple[float, float]:
    """Map a SHP lon/lat point to CTX pixel coordinates."""
    k = R_MARS * DEG
    return (k * lon - gt[0]) / gt[1], (k * lat - gt[3]) / gt[5]


def polygon_parts(shape) -> list[list[tuple[float, float]]]:
    points = shape.points
    starts = list(shape.parts) + [len(points)]
    return [points[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]


def map_source_records(shape_records: Iterable) -> dict[int, object]:
    """Map the persisted SHP sequence field to records, never array indices."""
    mapped: dict[int, object] = {}
    for index, record in enumerate(shape_records):
        try:
            source_id = int(round(float(record.record[0])))
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"SHP record {index} has no numeric sequence field") from exc
        if source_id in mapped:
            raise ValueError(f"duplicate SHP sequence number: {source_id}")
        mapped[source_id] = record
    return mapped


def compute_window_px(max_side_px: float, min_window_px: int = 1024,
                      margin_factor: float = 1.5, max_window_px: int = 12000) -> int:
    """Choose a square window with proportional context around a target."""
    if max_side_px <= 0:
        raise ValueError("max_side_px must be positive")
    if min_window_px <= 0 or max_window_px < min_window_px:
        raise ValueError("invalid window limits")
    if margin_factor < 1.0:
        raise ValueError("margin_factor must be >= 1")
    return min(max_window_px, max(min_window_px, int(math.ceil(max_side_px * margin_factor))))


def centered_window(aabb_px: tuple[float, float, float, float] | list[float],
                    margin_factor: float, min_window_px: int = 1024,
                    max_window_px: int = 12000) -> dict:
    """Return one nominal centered square window for an AABB."""
    x0, y0, x1, y1 = map(float, aabb_px)
    window_px = compute_window_px(max(x1 - x0, y1 - y0), min_window_px,
                                  margin_factor, max_window_px)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    ox, oy = cx - window_px / 2.0, cy - window_px / 2.0
    return {
        "window_px": window_px,
        "center_px": [round(cx, 3), round(cy, 3)],
        "window_origin_nominal": [round(ox, 3), round(oy, 3)],
        "window_origin": [int(round(ox)), int(round(oy)),
                          int(round(ox)) + window_px, int(round(oy)) + window_px],
    }


def place_window(origin: tuple[int, int, int, int], window_px: int,
                 tile_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Clip a nominal window to the tile without pushing the target out.

    A fixed-size inward shift keeps the crop square, but a target centered
    near the CTX edge then loses the part that is still inside the tile.
    Shrink on the overflowing side instead, and keep the opposite edge.
    """
    width, height = tile_shape
    if width <= 0 or height <= 0:
        raise ValueError("tile dimensions must be positive")
    ox, oy, _ox1, _oy1 = origin
    ox1, oy1 = ox + window_px, oy + window_px
    ox, ox1 = max(0, ox), min(width, ox1)
    oy, oy1 = max(0, oy), min(height, oy1)
    if ox >= ox1 or oy >= oy1:
        raise ValueError("window does not intersect the tile")
    return ox, oy, ox1, oy1


def _intersects(a: tuple[float, float, float, float],
                b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def make_review_row(tile: str, target_source_id: int, records: list[dict],
                    tile_shape: tuple[int, int], margin_factor: float = 1.5,
                    min_window_px: int = 1024, max_window_px: int = 12000) -> dict:
    """Build the manifest row for one target and all polygons in its crop."""
    target = next(r for r in records if r["source_id"] == target_source_id)
    window = centered_window(target["aabb_px"], margin_factor, min_window_px, max_window_px)
    width, height = tile_shape
    ox, oy, ox1, oy1 = place_window(tuple(window["window_origin"]), window["window_px"], tile_shape)
    crop_box = (float(ox), float(oy), float(ox1), float(oy1))
    included = sorted(
        (r["source_id"] for r in records if _intersects(tuple(r["aabb_px"]), crop_box))
    )
    tx0, ty0, tx1, ty1 = map(float, target["aabb_px"])
    aabb_complete = tx0 >= 0 and ty0 >= 0 and tx1 <= width and ty1 <= height
    polygon_complete = all(
        0 <= x <= width and 0 <= y <= height
        for part in target["polygon_px"] for x, y in part
    )
    edge_touch = (tx0 <= crop_box[0] or ty0 <= crop_box[1]
                  or tx1 >= crop_box[2] or ty1 >= crop_box[3])
    stem = f"{tile}_src{target_source_id:05d}"
    return {
        "tile": tile,
        "source_id": target_source_id,
        "center_poly_file": f"center_poly/{stem}_center_poly.png",
        "center_aabb_file": f"center_aabb/{stem}_center_aabb.png",
        "window_origin": [ox, oy, ox1, oy1],
        "poly_window_origin": [ox, oy, ox1, oy1],
        "native_window_px": [ox1 - ox, oy1 - oy],
        "center_px": window["center_px"],
        "polygon_bbox_px": [round(v, 3) for v in target["aabb_px"]],
        "aabb_complete": bool(aabb_complete),
        "polygon_complete": bool(polygon_complete),
        "edge_touch": bool(edge_touch),
        "intersecting_source_ids": included,
        "review_status": "",
        "review_notes": "",
    }


def _tile_bounds(gt: tuple[float, ...], width: int, height: int) -> tuple[float, float, float, float]:
    k = R_MARS * DEG
    lon0, lon1 = gt[0] / k, (gt[0] + width * gt[1]) / k
    lat0, lat1 = gt[3] / k, (gt[3] + height * gt[5]) / k
    return min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1)


def _record_for_tile(source_id: int, record, gt: tuple[float, ...], tile_bounds: tuple[float, ...]) -> dict | None:
    bbox = tuple(map(float, record.shape.bbox))
    if not _intersects(bbox, tile_bounds):
        return None
    parts_px = []
    all_points = []
    for part in polygon_parts(record.shape):
        points = [point_to_px(float(lon), float(lat), gt) for lon, lat in part]
        parts_px.append(points)
        all_points.extend(points)
    if not all_points:
        return None
    xs, ys = zip(*all_points)
    return {
        "source_id": source_id,
        "raw_type": str(record.record[1]).strip() if len(record.record) > 1 else "",
        "aabb_px": [min(xs), min(ys), max(xs), max(ys)],
        "polygon_px": parts_px,
    }


def _draw_label(draw: ImageDraw.ImageDraw, text: str, x: float, y: float,
                color: tuple[int, int, int], image_size: tuple[int, int]) -> None:
    font = ImageFont.load_default()
    tx = max(2, min(int(x), image_size[0] - 110))
    ty = max(2, min(int(y), image_size[1] - 18))
    draw.rectangle((tx, ty, tx + max(48, len(text) * 7 + 8), ty + 16), fill=(0, 0, 0))
    draw.text((tx + 3, ty + 2), text, fill=color, font=font)


def render_view(crop: np.ndarray, records_by_id: dict[int, dict], row: dict,
                view: str, out_path: Path) -> None:
    image = Image.fromarray(crop.astype(np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    ox, oy, _x1, _y1 = row["window_origin"]
    target_id = row["source_id"]
    for source_id in row["intersecting_source_ids"]:
        rec = records_by_id[source_id]
        color = (255, 40, 40) if source_id == target_id else (255, 205, 0)
        width = 5 if source_id == target_id else 3
        if view == "poly":
            for part in rec["polygon_px"]:
                points = [(x - ox, y - oy) for x, y in part]
                if len(points) >= 2:
                    draw.line(points + [points[0]], fill=color, width=width, joint="curve")
            lx, ly = rec["aabb_px"][0] - ox, rec["aabb_px"][1] - oy
        else:
            x0, y0, x1, y1 = rec["aabb_px"]
            draw.rectangle((x0 - ox, y0 - oy, x1 - ox, y1 - oy), outline=color, width=width)
            lx, ly = x0 - ox, y0 - oy
        _draw_label(draw, str(source_id), lx, ly, color, image.size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, compress_level=1)


def build_pack(plan_path: Path = PLAN, ctx_dir: Path = CTX_DIR, shp_path: Path = SHP,
               out_dir: Path = OUT_DIR, margin_factor: float = 1.5,
               min_window_px: int = 1024, max_window_px: int = 12000) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tif_of = discover_tifs(ctx_dir)
    missing = sorted(tile for tile in plan if tile not in tif_of)
    if missing:
        raise FileNotFoundError(f"plan tiles missing from CTX: {missing}")
    sf = shapefile.Reader(str(shp_path))
    shape_records = map_source_records(sf.shapeRecords())
    rows: list[dict] = []
    for tile in sorted(plan):
        gt, width, height = read_geotransform(tif_of[tile])
        bounds = _tile_bounds(gt, width, height)
        records = []
        for source_id, record in shape_records.items():
            converted = _record_for_tile(source_id, record, gt, bounds)
            if converted:
                records.append(converted)
        records_by_id = {r["source_id"]: r for r in records}
        if not records:
            continue
        arr = tifffile.memmap(str(tif_of[tile]))
        for target in records:
            row = make_review_row(tile, target["source_id"], records,
                                  (width, height), margin_factor,
                                  min_window_px, max_window_px)
            x0, y0, x1, y1 = row["window_origin"]
            crop = np.asarray(arr[y0:y1, x0:x1])
            expected = (y1 - y0, x1 - x0)
            if crop.shape != expected:
                raise RuntimeError(f"unexpected crop shape for {tile}/{target['source_id']}: {crop.shape}")
            render_view(crop, records_by_id, row, "poly", out_dir / row["center_poly_file"])
            render_view(crop, records_by_id, row, "aabb", out_dir / row["center_aabb_file"])
            rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["tile", "source_id", "center_poly_file", "center_aabb_file", "window_origin",
              "native_window_px", "polygon_bbox_px", "aabb_complete", "polygon_complete",
              "edge_touch", "intersecting_source_ids", "review_status", "review_notes"]
    with (out_dir / "centered_review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], (list, dict)) else row[k]
                             for k in fields})
    manifest = {
        "purpose": "今天新增20幅CTX图幅的中心裁片人工审核；不改变训练集",
        "plan": str(plan_path),
        "window_policy": {
            "formula": "ceil(max(AABB width, AABB height) * margin_factor)",
            "margin_factor": margin_factor,
            "min_window_px": min_window_px,
            "max_window_px": max_window_px,
            "note": "不是固定8 km；center_poly与center_aabb严格使用同一窗口",
        },
        "count": len(rows),
        "rows": rows,
    }
    (out_dir / "centered_review_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "README.txt").write_text(
        "20幅新增CTX图幅中心审核包\n"
        "每个 tile/source_id 有两张完全同窗口图片：\n"
        "  center_poly = SHP真实多边形边界（红色目标，黄色相邻目标）\n"
        "  center_aabb = 同一窗口的外接矩形（红色目标，黄色相邻目标）\n"
        "窗口按目标AABB大边长自适应，最低1024像素，不是固定8 km。\n"
        "aabb_complete/polygon_complete=false 表示目标跨出当前CTX图幅，不能当作完整正样本。\n"
        "本包只用于人工审核；未审核前不得加入训练，也不得把未确认区域当负样本。\n",
        encoding="utf-8")
    return {"count": len(rows), "out_dir": str(out_dir), "manifest": str(out_dir / "centered_review_manifest.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--ctx", type=Path, default=CTX_DIR)
    parser.add_argument("--shp", type=Path, default=SHP)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--margin-factor", type=float, default=1.5)
    parser.add_argument("--min-window-px", type=int, default=1024)
    parser.add_argument("--max-window-px", type=int, default=12000)
    args = parser.parse_args()
    print(json.dumps(build_pack(args.plan, args.ctx, args.shp, args.out_dir,
                                args.margin_factor, args.min_window_px,
                                args.max_window_px), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
