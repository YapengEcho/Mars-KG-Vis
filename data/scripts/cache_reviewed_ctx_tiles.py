"""Preserve native unmarked context crops before retiring reviewed, noncrossing CTX TIFFs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import shapefile
import tifffile

from check_tile_connectivity import tiles_of_bbox
from fetch_v2_train_tiles import BASE, CTX, confined, verify_tif
from make_centered_review_pack import SHP, map_source_records, read_geotransform, _tile_bounds, _record_for_tile
from manual_box_review import context_window, has_complete_geometry, write_json_atomic


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def native_window(row, box, width, height):
    original = json.loads(row["window_origin"])
    union = [min(original[0], box[0]), min(original[1], box[1]),
             max(original[2], box[2]), max(original[3], box[3])]
    side = math.ceil(max(union[2]-union[0], union[3]-union[1]) * 4)
    window = context_window(union, side, width, height)
    if not (window[0] <= union[0] <= union[2] <= window[2] and
            window[1] <= union[1] <= union[3] <= window[3]):
        raise ValueError("native cache does not contain corrected box and original window")
    return window


def preservation_window(row, box, width, height, backgrounds):
    window = native_window(row, box, width, height)
    for background in backgrounds.values():
        bbox = background["aabb_px"]
        if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox):
            raise ValueError("invalid saved background box")
        if not (0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height):
            raise ValueError("saved background box outside original tile")
        window = [min(window[0], math.floor(bbox[0])), min(window[1], math.floor(bbox[1])),
                  max(window[2], math.ceil(bbox[2])), max(window[3], math.ceil(bbox[3]))]
    return window


def backgrounds_for_tile(folder, tile):
    path = folder / "manual_background_boxes.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {key: value for key, value in data.items() if value["tile"] == tile}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--tiles", nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--delete-originals", action="store_true")
    parser.add_argument("--protect-tiles", nargs="*", default=[],
                        help="Additional active background or mosaic source tiles to keep")
    args = parser.parse_args()
    with shapefile.Reader(str(SHP)) as reader:
        sources = map_source_records(reader.shapeRecords())
    touches = {sid: tiles_of_bbox(*r.shape.bbox) for sid, r in sources.items()}
    folder = args.pack / "manual_review"
    current = json.loads((folder / "current_batch.json").read_text(encoding="utf-8"))
    active_tiles = {i["tile"] for i in current["items"]} | set(args.protect_tiles)
    corrections = {(r["tile"], r["source_id"]): r
                   for r in read_csv(folder / "manual_boxes.reviewed.csv")}
    exclusions = json.loads((folder / "review_exclusions.json").read_text(encoding="utf-8"))
    report = {"status": "preserving", "tiles": {}, "updated_epoch": time.time(),
              "policy": "reviewed noncrossing inactive tiles only; preserve stride-1 context before removal"}
    write_json_atomic(args.report, report)
    for tile in args.tiles:
        if tile in active_tiles or any(tile in hits and len(hits) > 1 for hits in touches.values()):
            raise ValueError(f"active or crossing tile cannot be retired: {tile}")
        paths = list(CTX.glob(f"*/MurrayLab_CTX_V01_{tile}_Mosaic.tif"))
        if len(paths) != 1:
            raise ValueError(f"expected one original TIFF: {tile}")
        source_path = confined(CTX, paths[0])
        verified = verify_tif(source_path)
        gt, width, height = read_geotransform(source_path)
        bounds = _tile_bounds(gt, width, height)
        records = {sid: r for sid, s in sources.items()
                   if (r := _record_for_tile(sid, s, gt, bounds)) is not None}
        rows = read_csv(args.pack / "tiles" / tile / "centered_review_manifest.csv")
        if set(records) != {int(r["source_id"]) for r in rows}:
            raise ValueError(f"review pack does not cover every target: {tile}")
        backgrounds = backgrounds_for_tile(folder, tile)
        if any(str(b["source_review_id"]) not in {r["source_id"] for r in rows}
               for b in backgrounds.values()):
            raise ValueError(f"background is missing its source review context: {tile}")
        cache = args.pack / "native_context" / tile
        cache.mkdir(parents=True, exist_ok=True)
        arr = tifffile.memmap(source_path)
        entries = []
        for row in rows:
            sid = row["source_id"]; key = f"{tile}:{sid}"
            correction = corrections.get((tile, sid))
            if not has_complete_geometry(row) or not (key in exclusions or
                    correction and correction["confirmed"].lower() == "true"):
                raise ValueError(f"unreviewed or truncated target cannot be retired: {key}")
            box = ([float(correction[k]) for k in ("x0", "y0", "x1", "y1")]
                   if correction else json.loads(row["polygon_bbox_px"]))
            saved_backgrounds = {k: b for k, b in backgrounds.items()
                                 if str(b["source_review_id"]) == sid}
            window = preservation_window(row, box, width, height, saved_backgrounds)
            x0, y0, x1, y1 = window
            crop = np.asarray(arr[y0:y1, x0:x1])
            path = cache / f"{tile}_src{int(sid):05d}_native.tif"
            temp = path.with_suffix(".writing.tif")
            tifffile.imwrite(temp, crop, compression="deflate", rowsperstrip=256,
                             maxworkers=2, metadata=None)
            restored = tifffile.imread(temp)
            if not np.array_equal(restored, crop):
                raise ValueError(f"native pixel verification failed: {key}")
            del restored
            digest = hashlib.sha256()
            with temp.open("rb") as handle:
                for block in iter(lambda: handle.read(8 << 20), b""):
                    digest.update(block)
            os.replace(temp, path)
            entries.append({"tile": tile, "source_id": int(sid), "path": str(path.resolve()),
                            "window_origin": window, "stride": 1, "shape": list(crop.shape),
                            "sha256": digest.hexdigest(), "manual_correction": correction,
                            "exclusion": exclusions.get(key),
                            "saved_backgrounds": saved_backgrounds,
                            "intersecting_source_ids": [n for n, r in records.items()
                                if r["aabb_px"][0] < x1 and r["aabb_px"][2] > x0
                                and r["aabb_px"][1] < y1 and r["aabb_px"][3] > y0]})
            del crop
        del arr
        manifest = {"tile": tile, "original": verified, "geotransform": gt,
                    "tile_width": width, "tile_height": height, "entries": entries,
                    "purpose": "lossless unmarked native context; not training-ready labels",
                    "context_policy": "4x review/positive context, extended to contain every saved background rectangle",
                    "recovery": "larger context or full-tile backgrounds require re-downloading original CTX"}
        write_json_atomic(cache / "manifest.json", manifest)
        # Recheck only this tile's decisions: concurrent edits in other tiles are safe.
        latest = {(r["tile"], r["source_id"]): r for r in read_csv(folder / "manual_boxes.reviewed.csv")}
        latest_exclusions = json.loads((folder / "review_exclusions.json").read_text(encoding="utf-8"))
        if backgrounds_for_tile(folder, tile) != backgrounds:
            raise ValueError(f"background decisions changed during preservation: {tile}")
        for entry in entries:
            key = (tile, str(entry["source_id"]))
            if latest.get(key) != entry["manual_correction"] or latest_exclusions.get(f"{key[0]}:{key[1]}") != entry["exclusion"]:
                raise ValueError(f"review decisions changed during preservation: {key}")
        current = json.loads((folder / "current_batch.json").read_text(encoding="utf-8"))
        if tile in {i["tile"] for i in current["items"]}:
            raise ValueError(f"tile became active during preservation: {tile}")
        if verify_tif(source_path) != verified:
            raise ValueError(f"source changed during preservation: {tile}")
        cache_bytes = sum(Path(e["path"]).stat().st_size for e in entries)
        if cache_bytes >= verified["bytes"]:
            # These caches were created in this pass; the verified original is
            # still present, so retaining the duplicate pixels wastes storage.
            for entry in entries:
                confined(cache, Path(entry["path"])).unlink()
            confined(cache, cache / "manifest.json").unlink()
            report["tiles"][tile] = {"original_deleted": False,
                                     "original": verified,
                                     "reason": "preserved contexts do not save storage",
                                     "cancelled_native_context_bytes": cache_bytes,
                                     "redundant_caches_removed": True,
                                     "net_freed_bytes": 0}
            write_json_atomic(args.report, report)
            print(f"RETAINED {tile}: native contexts exceed original size", flush=True)
            continue
        if args.delete_originals:
            confined(CTX, source_path).unlink()
        report["tiles"][tile] = {"original": verified, "native_context_count": len(entries),
                                 "native_context_bytes": cache_bytes,
                                 "manifest": str((cache / "manifest.json").resolve()),
                                 "preserved_background_count": len(backgrounds),
                                 "original_deleted": args.delete_originals,
                                 "net_freed_bytes": verified["bytes"]-cache_bytes if args.delete_originals else 0}
        report["updated_epoch"] = time.time()
        write_json_atomic(args.report, report)
        print(f"PRESERVED {tile} {len(entries)} contexts; deleted={args.delete_originals}; net_freed={report['tiles'][tile]['net_freed_bytes']}", flush=True)
    report["status"] = "complete"
    report["deferred_non_crossing_tiles"] = sorted(active_tiles & {t for t in active_tiles
        if not any(t in hits and len(hits)>1 for hits in touches.values())})
    write_json_atomic(args.report, report)


if __name__ == "__main__":
    main()
