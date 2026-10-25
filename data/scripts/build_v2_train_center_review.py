# -*- coding: utf-8 -*-
"""Build resumable center review packs from restored v2 training tiles only.

Produces per-tile full-resolution paired views and one combined review CSV.
Human review columns remain blank. Existing completed packs are not replaced.
No training labels, split plans, or original SHP are modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import time

from PIL import Image
import numpy as np
import shapefile
import tifffile

from check_tile_connectivity import tiles_of_bbox
from fetch_v2_train_tiles import BASE, CTX, FINAL, PLAN, existing_tif, train_tiles
from make_centered_review_pack import (SHP, map_source_records, discover_tifs, read_geotransform,
                                      _tile_bounds, _record_for_tile, make_review_row, render_view)

DEFAULT_OUT = BASE / "data/processed/centered_review_v2_train_20261006"


def build_complete_pack(tile, out_dir):
    """Keep full target context; large previews record exact sampling stride."""
    tif = discover_tifs(CTX)[tile]
    gt, width, height = read_geotransform(tif)
    bounds = _tile_bounds(gt, width, height)
    with shapefile.Reader(str(SHP)) as reader:
        sources = map_source_records(reader.shapeRecords())
    records = [rec for sid, source in sources.items()
               if (rec := _record_for_tile(sid, source, gt, bounds)) is not None]
    arr = tifffile.memmap(tif)
    rows = []
    for target in records:
        # Larger than a CTX tile: the crop can include every available pixel;
        # a source crossing the tile is explicitly partial, never auto-accepted.
        row = make_review_row(tile, target["source_id"], records, (width, height), max_window_px=72000)
        x0, y0, x1, y1 = row["window_origin"]
        longest = max(x1-x0, y1-y0)
        stride = max(1, (longest+4095)//4096) if longest > 12000 else 1
        crop = np.asarray(arr[y0:y1:stride, x0:x1:stride])
        row["preview_downsample_stride"] = stride
        row["preview_size_px"] = [crop.shape[1], crop.shape[0]]
        # Drawing coordinates are transformed for the preview, while the
        # persisted annotation coordinates remain original CTX tile pixels.
        preview_row = dict(row, window_origin=[0, 0, crop.shape[1], crop.shape[0]])
        preview_records = {}
        for rec in records:
            a, b, c, d = rec["aabb_px"]
            preview_records[rec["source_id"]] = dict(rec,
                aabb_px=[(a-x0)/stride, (b-y0)/stride, (c-x0)/stride, (d-y0)/stride],
                polygon_px=[[(float(x-x0)/stride, float(y-y0)/stride) for x,y in part]
                            for part in rec["polygon_px"]])
        render_view(crop, preview_records, preview_row, "poly", out_dir / row["center_poly_file"])
        render_view(crop, preview_records, preview_row, "aabb", out_dir / row["center_aabb_file"])
        rows.append(row)
    del arr
    fields = list(rows[0]) if rows else []
    with (out_dir / "centered_review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k,v in row.items()})
    save_json(out_dir / "centered_review_manifest.json", {"rows": rows, "count": len(rows),
        "purpose": "complete native center windows; preview sampling is explicitly recorded"})
    return {"count": len(rows), "out_dir": str(out_dir)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint():
    paths = [SHP, SHP.with_suffix(".shx"), SHP.with_suffix(".dbf"), PLAN, FINAL,
             Path(__file__).with_name("make_centered_review_pack.py")]
    return {str(p): sha(p) for p in paths}


def geometric_status(row):
    if str(row["aabb_complete"]).lower() != "true" or str(row["polygon_complete"]).lower() != "true":
        return "tile_truncated"
    window = json.loads(row["window_origin"])
    box = json.loads(row["polygon_bbox_px"])
    if not (window[0] <= box[0] and window[1] <= box[1] and window[2] >= box[2] and window[3] >= box[3]):
        return "window_truncated"
    return "complete"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def finalize(out, wanted, signatures):
    with shapefile.Reader(str(SHP)) as reader:
        records = map_source_records(reader.shapeRecords())
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    protected = {t for t, s in plan.items() if s in {"val", "test"}} | set(json.loads(FINAL.read_text(encoding="utf-8"))["tiles"])
    rows = []
    catalog_path = BASE / "data/raw/mars_landslides.json"
    catalog = {int(r["source_id"]): r for r in json.loads(catalog_path.read_text(encoding="utf-8"))}
    for tile in wanted:
        for row in read_csv(out / "tiles" / tile / "centered_review_manifest.csv"):
            sid = int(row["source_id"])
            row["shp_raw_type"] = str(records[sid].record[1]).strip()
            row["class_name"] = catalog[sid]["type"]
            row.setdefault("preview_downsample_stride", "1")
            row.setdefault("preview_size_px", row["native_window_px"])
            row.setdefault("poly_window_origin", row["window_origin"])
            row["geometry_status"] = geometric_status(row)
            hits = sorted(tiles_of_bbox(*records[sid].shape.bbox) & protected)
            row["protected_touch_tiles"] = json.dumps(hits)
            row["training_gate"] = ("hold_until_actual_overlap_audit" if hits else "pending_human_review")
            row["center_poly_file"] = f"tiles/{tile}/" + row["center_poly_file"]
            row["center_aabb_file"] = f"tiles/{tile}/" + row["center_aabb_file"]
            rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    master = out / "centered_review_manifest.csv"
    if not master.exists():
        with master.open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        previous = read_csv(master)
        if {(r["tile"], r["source_id"]) for r in previous} != {(r["tile"], r["source_id"]) for r in rows}:
            raise ValueError("existing master review differs; refusing to overwrite human work")
    unique = {int(r["source_id"]): r["class_name"] for r in rows}
    summary = {
        "status": "review_only_no_training", "tile_count": len(wanted), "row_count": len(rows),
        "unique_source_count": len(unique), "source_class_counts": dict(Counter(unique.values())),
        "row_class_counts": dict(Counter(r["class_name"] for r in rows)),
        "geometry_counts": dict(Counter(r["geometry_status"] for r in rows)),
        "possible_protected_touch_rows": sum(r["protected_touch_tiles"] != "[]" for r in rows),
        "note": "touch is conservative, not proof of pixel overlap; downloading is permitted but these targets remain held for training until actual overlap is resolved",
        "input_sha256": signatures,
        "catalog_sha256": sha(catalog_path),
        "window_policy": "full target-centered window (max72000), clipped only to actual CTX bounds; large images previewed with recorded stride",
        "rows": rows,
    }
    save_json(out / "centered_review_manifest.json", summary)
    # Small previews avoid loading hundreds of full native CTX crops at once.
    thumbs = out / "thumbs"
    thumbs.mkdir(exist_ok=True)
    cards = []
    for row in rows:
        images = []
        for view, key in (("poly", "center_poly_file"), ("aabb", "center_aabb_file")):
            thumb = thumbs / f"{row['tile']}_src{int(row['source_id']):05d}_{view}.jpg"
            if not thumb.exists():
                with Image.open(out / row[key]) as image:
                    image.thumbnail((480, 480))
                    image.convert("RGB").save(thumb, quality=85)
            images.append(f'<a href="{html.escape(row[key])}"><img loading="lazy" src="thumbs/{thumb.name}" alt="{view}"></a>')
        cards.append(f'<article data-tile="{row["tile"]}" data-geometry="{row["geometry_status"]}"><h3>{row["tile"]} · {row["source_id"]} · {row["class_name"]}</h3>'
                     f'<p>完整性：{row["geometry_status"]}；预览采样步长：{row["preview_downsample_stride"]}；跨保护区待查：{html.escape(row["protected_touch_tiles"])}</p><div class="views">{"".join(images)}</div></article>')
    options = ''.join(f'<option>{t}</option>' for t in wanted)
    index = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>v2训练图幅中心审核</title>
<style>body{font:16px system-ui;background:#f4f6f8;color:#16202b;margin:24px}header{position:sticky;top:0;background:#f4f6f8;padding:12px;z-index:1}article{background:white;padding:18px;margin:18px 0;border:1px solid #ddd;border-radius:8px}.views{display:flex;gap:16px;flex-wrap:wrap}.views img{max-width:100%;width:480px}select{font:inherit;padding:8px}h3{margin:0}</style>
<header><h1>v2 训练图幅 · 滑坡中心审核</h1><p>左：原始多边形；右：水平外接框。红色为中心目标，黄色为相邻目标。点击图片查看保存的预览尺寸；大图有采样，原始窗口坐标和步长见 CSV。审核结论填写在同目录 CSV，此页仅供查看。</p>
<label>图幅 <select id="tile"><option value="">全部</option>OPTIONS</select></label>
<label>完整性 <select id="geom"><option value="">全部</option><option>complete</option><option>tile_truncated</option><option>window_truncated</option></select></label></header>CARDS
<script>for(const id of ['tile','geom'])document.getElementById(id).addEventListener('change',()=>{const t=document.getElementById('tile').value,g=document.getElementById('geom').value;for(const a of document.querySelectorAll('article'))a.hidden=Boolean((t&&a.dataset.tile!==t)||(g&&a.dataset.geometry!==g))})</script></html>'''
    (out / "index.html").write_text(index.replace("OPTIONS", options).replace("CARDS", ''.join(cards)), encoding="utf-8")
    (out / "README.txt").write_text(
        "v2训练图幅中心审核包。原始SHP与旧训练集未改。\n"
        "打开index.html浏览成对图片，在centered_review_manifest.csv填写review_status/review_notes。\n"
        "中心目标与相邻目标都需检查；不能把暂扣/未审区域当背景。\n"
        "geometry_status是几何完整性，不是人工判断；protected_touch_tiles只是保守提示，不是泄漏定论。\n"
        "大窗口预览有采样：图幅坐标=窗口起点+预览坐标*preview_downsample_stride。\n"
        "本包重裁的是旧训练图幅，不能称为新独立滑坡，也不能直接拿新裁片评测分母与旧结果排名。\n"
        "先复审再生成独立候选版本；没有自动接受、导出训练标签或训练。\n", encoding="utf-8")
    return {k: v for k, v in summary.items() if k not in {"rows", "input_sha256"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--wait-for-download", action="store_true")
    parser.add_argument("--restore-report", type=Path, default=BASE / "runs/v2_train_restore_status.json")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    wanted, signatures = train_tiles(), fingerprint()
    completed = set()
    while len(completed) < len(wanted):
        advanced = False
        for tile in wanted:
            if tile in completed:
                continue
            tile_dir = out / "tiles" / tile
            ready = tile_dir / "_ready.json"
            if ready.exists():
                meta = json.loads(ready.read_text(encoding="utf-8"))
                if meta["input_sha256"] != signatures:
                    raise ValueError(f"input version changed for existing review pack {tile}")
                completed.add(tile)
                continue
            verified = existing_tif(CTX, tile)
            if verified is None:
                continue
            tile_dir.mkdir(parents=True, exist_ok=True)
            plan_file = tile_dir / "_train_tile_plan.json"
            plan_file.write_text(json.dumps({tile: "train"}), encoding="utf-8")
            print(f"REVIEW_BUILD {tile}", flush=True)
            result = build_complete_pack(tile, tile_dir)
            save_json(ready, {"input_sha256": signatures, "verified_tif": verified, "result": result})
            completed.add(tile)
            advanced = True
            save_json(out / "progress.json", {"status": "building", "completed": sorted(completed),
                                              "total": len(wanted), "updated_epoch": time.time()})
            print(f"REVIEW_READY {tile}: {result['count']} sources ({len(completed)}/{len(wanted)})", flush=True)
        if len(completed) == len(wanted):
            break
        if not args.wait_for_download:
            print(f"WAITING_TILES {len(wanted)-len(completed)}", flush=True)
            return
        if args.restore_report.exists():
            restore = json.loads(args.restore_report.read_text(encoding="utf-8"))
            if restore["status"] in {"failed", "complete"}:
                missing = set(wanted) - completed
                if missing and not any(existing_tif(CTX, tile) for tile in missing):
                    raise RuntimeError(f"download stopped with missing tiles: {sorted(missing)}")
        if not advanced:
            time.sleep(15)
    if fingerprint() != signatures:
        raise ValueError("input changed during review generation")
    summary = finalize(out, wanted, signatures)
    save_json(out / "progress.json", {"status": "complete", "completed": wanted,
                                      "total": len(wanted), "updated_epoch": time.time()})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
