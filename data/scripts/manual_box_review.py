"""Local manual box editor over original CTX pixels; independent correction CSV."""
from __future__ import annotations

import argparse
import csv
import hashlib
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image
import shapefile
import tifffile

from build_center_candidate_manifest import OVERRIDE_FIELDS, validate_box
from make_centered_review_pack import (BASE, SHP, discover_tifs, read_geotransform,
    map_source_records, _tile_bounds, _record_for_tile, _intersects)

PACK = BASE / "data/processed/centered_review_v2_train_20261006"
HTML = Path(__file__).with_suffix(".html")
BACKGROUND_KINDS = {"plain_background", "crater_wall", "valley_wall", "dune",
                    "shadow", "eroded_slope", "other"}


def has_complete_geometry(row):
    """Original reference geometry must fit the available tile for this queue."""
    return all(str(row.get(field, "")).strip().lower() == "true"
               for field in ("aabb_complete", "polygon_complete"))


def write_json_atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def update_draft_status(pack, tile, sid, status, reason):
    path = pack / "centered_review_manifest.draft.csv"
    if not path.exists():
        return
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    matches = [r for r in rows if r["tile"] == tile and int(r["source_id"]) == sid]
    if len(matches) != 1:
        raise ValueError("审核草稿中的目标编号缺失或重复")
    matches[0]["review_status"], matches[0]["review_notes"] = status, reason
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def context_window(box, side, width, height):
    w, h = min(side, width), min(side, height)
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    x = max(0, min(round(cx-w/2), width-w))
    y = max(0, min(round(cy-h/2), height-h))
    return [x, y, x+w, y+h]


def validate_manual(payload, allowed, width, height):
    sid = int(payload["source_id"])
    if sid not in allowed:
        raise ValueError("目标编号不在本审核批次")
    box = validate_box(payload["box"])
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError("框超出原始图幅，缺失像素需要重裁或拼接")
    if not _intersects(tuple(box), tuple(allowed[sid]["aabb_px"])):
        raise ValueError("新框未覆盖原目标附近，请核对目标编号")
    if payload.get("confirmed") is not True:
        raise ValueError("请先确认标注范围")
    if payload.get("label_scope") not in {"full_landslide", "visible_core"}:
        raise ValueError("未知标注范围")
    if not str(payload.get("reviewer", "")).strip() or not str(payload.get("reason", "")).strip():
        raise ValueError("请填写审核人与修订原因")
    return sid, box


def write_correction(path, record):
    """Replace only this source's sidecar entry; preserve other manual work."""
    rows = []
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    key = (record["tile"], str(record["source_id"]))
    matching = [r for r in rows if (r["tile"], r["source_id"]) == key]
    if len(matching) > 1:
        raise ValueError("人工修订表存在重复编号，暂不覆盖")
    rows = [r for r in rows if (r["tile"], r["source_id"]) != key] + [record]
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OVERRIDE_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


class Review:
    def __init__(self, tile, ids, pack=PACK, ctx=None):
        self.tile, self.pack = tile, pack
        self.folder = pack / "manual_review"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.corrections = self.folder / "manual_boxes.reviewed.csv"
        self.lock = threading.Lock()
        source_csv = pack / "tiles" / tile / "centered_review_manifest.csv"
        with source_csv.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = {int(r["source_id"]): r for r in rows if int(r["source_id"]) in ids}
        if set(self.rows) != set(ids):
            raise ValueError("审核编号不在已生成的图幅审核包内")
        self.tif = discover_tifs(ctx or BASE / "data/raw/ctx_tiles").get(tile)
        self.cache_entries = {}
        if self.tif:
            gt, self.width, self.height = read_geotransform(self.tif)
        else:
            manifest = pack / "native_context" / tile / "manifest.json"
            cache = json.loads(manifest.read_text(encoding="utf-8"))
            gt = tuple(map(float, cache["geotransform"]))
            self.width, self.height = int(cache["tile_width"]), int(cache["tile_height"])
            if cache["tile"] != tile or len(gt) != 6 or not all(map(math.isfinite, gt)):
                raise ValueError("原始像素缓存坐标无效")
            self.cache_entries = {int(e["source_id"]): e for e in cache["entries"]}
            if any(sid not in self.cache_entries or self.cache_entries[sid]["stride"] != 1
                   or not Path(self.cache_entries[sid]["path"]).is_file() for sid in ids):
                raise ValueError("缺少该目标的原分辨率上下文缓存")
        bounds = _tile_bounds(gt, self.width, self.height)
        with shapefile.Reader(str(SHP)) as reader:
            sources = map_source_records(reader.shapeRecords())
        self.records = {sid: r for sid, source in sources.items()
            if (r := _record_for_tile(sid, source, gt, bounds)) is not None}
        self.allowed = {sid: self.records[sid] for sid in ids}

    def saved(self):
        if not self.corrections.exists():
            return {}
        with self.corrections.open(encoding="utf-8-sig", newline="") as handle:
            return {int(r["source_id"]): r for r in csv.DictReader(handle) if r["tile"] == self.tile}

    @lru_cache(maxsize=4)
    def view(self, sid, factor):
        if sid not in self.rows or factor not in {1, 4, 8, 16}:
            raise ValueError("未知目标或视野")
        original = json.loads(self.rows[sid]["window_origin"])
        if factor == 1:
            window = original
        else:
            side = max(original[2]-original[0], original[3]-original[1]) * factor
            window = context_window(self.allowed[sid]["aabb_px"], side, self.width, self.height)
        requested_window = list(window)
        available = [0, 0, self.width, self.height]
        if self.tif is None:
            entry = self.cache_entries[sid]
            available = entry["window_origin"]
            window = [max(window[0], available[0]), max(window[1], available[1]),
                      min(window[2], available[2]), min(window[3], available[3])]
            if window[0] >= window[2] or window[1] >= window[3]:
                raise ValueError("当前视野没有已保留的原始像素")
        x0, y0, x1, y1 = window
        stride = max(1, math.ceil(max(x1-x0, y1-y0)/4096))
        if self.tif:
            arr = tifffile.memmap(self.tif)
            crop = np.asarray(arr[y0:y1:stride, x0:x1:stride])
        else:
            path = Path(entry["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("原始像素缓存校验失败")
            arr = tifffile.imread(path)
            if list(arr.shape) != entry["shape"] or arr.dtype != np.uint8:
                raise ValueError("原始像素缓存尺寸无效")
            crop = arr[y0-available[1]:y1-available[1]:stride,
                       x0-available[0]:x1-available[0]:stride]
        preview = Image.fromarray(crop)
        buffer = io.BytesIO(); preview.save(buffer, format="PNG")
        polygons = [{"source_id": r["source_id"], "polygon_px": r["polygon_px"]}
            for r in self.records.values() if _intersects(tuple(r["aabb_px"]), tuple(window))]
        meta = {"tile": self.tile, "source_id": sid, "window": window, "stride": stride,
            "image_width": preview.width, "image_height": preview.height,
            "original_box": self.allowed[sid]["aabb_px"], "polygons": polygons,
            "tile_width": self.width, "tile_height": self.height,
            "geometry_complete": has_complete_geometry(self.rows[sid]),
            "available_window_px": available, "cached_context": self.tif is None,
            "context_limited": window != requested_window}
        return meta, buffer.getvalue()

    def save(self, payload):
        sid, box = validate_manual(payload, self.allowed, self.width, self.height)
        record = {"tile": self.tile, "source_id": str(sid),
            **dict(zip(("x0", "y0", "x1", "y1"), box)),
            "label_scope": payload["label_scope"], "reviewer": str(payload["reviewer"]).strip(),
            "confirmed": "true", "reason": str(payload["reason"]).strip()}
        with self.lock:
            with (self.folder / "manual_box_history.jsonl").open("a", encoding="utf-8") as history:
                history.write(json.dumps({"saved_epoch": time.time(), "record": record}, ensure_ascii=False)+"\n")
            write_correction(self.corrections, record)
        return record


class ReviewBatch:
    """Keep tile and source ID together when a batch spans several images."""
    def __init__(self, items, pack=PACK, ctx=None, batch_number=None, groups=None, background_only=False, dataset_role='training'):
        if dataset_role not in {'training', 'evaluation'}:
            raise ValueError('未知数据用途')
        self.dataset_role = dataset_role
        self.evaluation_splits = {}
        if dataset_role == 'evaluation':
            plan = json.loads((pack/'evaluation_plan.json').read_text(encoding='utf-8'))
            if plan.get('role') != 'evaluation_only_never_training':
                raise ValueError('缺少独立评估审核清单')
            self.evaluation_splits = plan['tile_splits']
            if any(self.evaluation_splits.get(i['tile']) not in {'val', 'test'} for i in items):
                raise ValueError('评估批次包含非验证或测试图幅')
        self.batch_number = batch_number
        self.groups, self.background_only = groups or [], background_only
        selections = {}
        seen = set()
        for item in items:
            tile, sid = item["tile"], int(item["source_id"])
            if (tile, sid) in seen:
                raise ValueError("审核批次存在重复目标")
            seen.add((tile, sid))
            selections.setdefault(tile, []).append(sid)
        if not selections:
            raise ValueError("审核批次为空")
        self.reviews = {tile: Review(tile, ids, pack, ctx) for tile, ids in selections.items()}
        shared_lock = threading.RLock()
        for review in self.reviews.values():
            review.lock = shared_lock
        first = next(iter(self.reviews.values()))
        self.folder, self.corrections = first.folder, first.corrections
        self.pack, self.lock = pack, shared_lock
        self.exclusion_path = self.folder / "review_exclusions.json"
        self.items = [{"tile": item["tile"], "source_id": int(item["source_id"])} for item in items]

    def select(self, tile):
        if tile not in self.reviews:
            raise ValueError("图幅不在本审核批次")
        return self.reviews[tile]

    def batch_info(self):
        exclusions = self.exclusions()
        saved = {f"{tile}:{sid}": record for tile, review in self.reviews.items()
                 for sid, record in review.saved().items() if sid in review.rows
                 and record["confirmed"].strip().lower() == "true"
                 and f"{tile}:{sid}" not in exclusions}
        selected = {f"{i['tile']}:{i['source_id']}" for i in self.items}
        return {"batch_number": self.batch_number, "items": self.items, "saved": saved,
                "groups": self.groups, "background_only": self.background_only, "dataset_role": self.dataset_role,
                "excluded": {key: record for key, record in exclusions.items() if key in selected},
                "backgrounds": {key: record for key, record in self.backgrounds().items()
                                if record["tile"] in self.reviews}}

    def backgrounds(self):
        path = self.folder / "manual_background_boxes.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def background_history(self, record):
        with (self.folder / "manual_background_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False)+"\n")

    def save_background(self, payload):
        """Save an independently reviewed empty crop region, never a YOLO object."""
        review = self.select(payload["tile"])
        sid = int(payload["source_id"])
        if sid not in review.rows:
            raise ValueError("目标编号不在本审核批次")
        box = validate_box(payload["box"])
        # The eventual crop uses whole pixels: validate that exact region now.
        box = [math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])]
        if not (0 <= box[0] < box[2] <= review.width and 0 <= box[1] < box[3] <= review.height):
            raise ValueError("背景框超出原始图幅")
        factor = payload.get("view_factor")
        if type(factor) is not int or factor not in {1, 4, 8, 16}:
            raise ValueError("请核对绘框视野")
        meta, _ = review.view(sid, factor)
        window = meta["window"]
        if not (window[0] <= box[0] and window[1] <= box[1]
                and box[2] <= window[2] and box[3] <= window[3]):
            raise ValueError("背景框必须位于当前已查看的影像范围内")
        reviewer, reason = str(payload.get("reviewer", "")).strip(), str(payload.get("reason", "")).strip()
        kind = payload.get("background_kind", "plain_background")
        if payload.get("confirmed") is not True or not reviewer or not reason:
            raise ValueError("请确认整个框内没有滑坡，并填写审核人和判断依据")
        if len(reviewer) > 100 or len(reason) > 1000 or kind not in BACKGROUND_KINDS:
            raise ValueError("审核信息或背景类型无效")
        from continue_ctx_review_download import protected_tiles
        if self.dataset_role == 'training' and review.tile in protected_tiles():
            raise ValueError("保护的验证或测试图幅不能加入训练背景")
        with self.lock:
            # All original SHP extents stay blocked, including rejected and partial slides.
            blockers = [r["aabb_px"] for r in review.records.values()]
            blockers.extend([float(r[k]) for k in ("x0", "y0", "x1", "y1")]
                            for r in review.saved().values())
            blockers.extend(r["exclusion_box_px"] for r in self.exclusions().values()
                            if r["tile"] == review.tile and r.get("exclusion_box_px"))
            if any(_intersects(tuple(box), tuple(b)) for b in blockers):
                raise ValueError("背景框与已知滑坡、人工修订或排除范围相交，请换一块确认没有滑坡的区域")
            identity = json.dumps([review.tile, box], separators=(",", ":"))
            background_id = "BG_"+hashlib.sha256(identity.encode()).hexdigest()[:20]
            record = {"background_id": background_id, "tile": review.tile,
                      "source_review_id": sid, "aabb_px": box, "view_window_px": window,
                      "view_factor": factor, "tile_size_px": [review.width, review.height],
                      "background_kind": kind, "reviewer": reviewer, "reason": reason,
                      "confirmed": True, "judgement": "certain_background",
                      "training_gate": "pending_recrop_and_overlap_recheck",
                      "saved_epoch": time.time()}
            if self.dataset_role == 'evaluation':
                record.update(dataset_role='evaluation', split=self.evaluation_splits[review.tile],
                              training_gate='forbidden_evaluation_only')
            backgrounds = self.backgrounds()
            self.background_history({"action": "save", "record": record,
                                     "previous": backgrounds.get(background_id)})
            backgrounds[background_id] = record
            write_json_atomic(self.folder / "manual_background_boxes.json", backgrounds)
            return record

    def remove_background(self, payload):
        review = self.select(payload["tile"])
        background_id = payload["background_id"]
        with self.lock:
            backgrounds = self.backgrounds()
            if background_id not in backgrounds or backgrounds[background_id]["tile"] != review.tile:
                raise ValueError("背景框不在当前图幅")
            previous = backgrounds.pop(background_id)
            self.background_history({"action": "remove", "saved_epoch": time.time(), "record": previous})
            write_json_atomic(self.folder / "manual_background_boxes.json", backgrounds)
        return background_id

    def exclusions(self):
        return (json.loads(self.exclusion_path.read_text(encoding="utf-8"))
                if self.exclusion_path.exists() else {})

    def decision_history(self, record):
        with (self.folder / "review_decision_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False)+"\n")

    def exclude(self, payload):
        if self.background_only:
            raise ValueError("当前为历史背景补审，已完成的正样本和排除决定保留")
        review = self.select(payload["tile"])
        sid = int(payload["source_id"])
        if sid not in review.rows:
            raise ValueError("目标编号不在本审核批次")
        reviewer, reason = str(payload.get("reviewer", "")).strip(), str(payload.get("reason", "")).strip()
        if not reviewer or not reason or len(reviewer) > 100 or len(reason) > 1000:
            raise ValueError("请填写审核人与排除原因")
        key = f"{review.tile}:{sid}"
        record = {"tile": review.tile, "source_id": sid, "review_status": "reject",
                  "training_use": "exclude_positive", "negative_eligible": False,
                  "reviewer": reviewer, "reason": reason,
                  "decision_basis": "explicit_human_review", "saved_epoch": time.time()}
        with self.lock:
            exclusions = self.exclusions()
            previous = review.saved().get(sid)
            if previous:
                record["exclusion_box_px"] = [float(previous[field]) for field in ("x0", "y0", "x1", "y1")]
            self.decision_history({**record, "previous_correction": previous})
            exclusions[key] = record
            write_json_atomic(self.exclusion_path, exclusions)
            if previous:
                write_correction(self.corrections, {**previous, "confirmed": "false", "reason": reason})
            update_draft_status(self.pack, review.tile, sid, "reject", reason)
        return record

    def save(self, payload):
        if self.background_only:
            raise ValueError("当前为历史背景补审，已完成的正样本框保留")
        if self.dataset_role == 'evaluation':
            review=self.select(payload['tile'])
            if payload.get('label_scope')!='full_landslide' or not has_complete_geometry(review.rows[int(payload['source_id'])]):
                raise ValueError('完整目标评估需完整范围；跨图幅或缺失上下文请先拼接再审核')
        with self.lock:
            record = self.select(payload["tile"]).save(payload)
            key = f"{record['tile']}:{record['source_id']}"
            exclusions = self.exclusions()
            if key in exclusions:
                self.decision_history({"action": "restore_positive", "saved_epoch": time.time(),
                                       "record": record, "previous_exclusion": exclusions[key]})
                del exclusions[key]
                write_json_atomic(self.exclusion_path, exclusions)
                update_draft_status(self.pack, record["tile"], int(record["source_id"]), "accept", record["reason"])
            return record


def handler_for(review):
    class Handler(BaseHTTPRequestHandler):
        def response(self, body, content_type="application/json; charset=utf-8", status=200):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            try:
                request = urlparse(self.path)
                if request.path == "/":
                    return self.response(HTML.read_bytes(), "text/html; charset=utf-8")
                if request.path == "/api/batch":
                    return self.response(review.batch_info())
                if request.path == "/api/export":
                    data = review.corrections.read_bytes() if review.corrections.exists() else ("\ufeff"+",".join(OVERRIDE_FIELDS)+"\n").encode("utf-8")
                    return self.response(data, "text/csv; charset=utf-8")
                if request.path in {"/api/view", "/api/image"}:
                    query = parse_qs(request.query)
                    selected = review.select(query["tile"][0])
                    meta, png = selected.view(int(query["id"][0]), int(query.get("factor", ["4"])[0]))
                    return self.response(meta) if request.path == "/api/view" else self.response(png, "image/png")
                self.response({"error": "not found"}, status=404)
            except (ValueError, KeyError) as exc:
                self.response({"error": str(exc)}, status=400)

        def do_POST(self):
            if self.path not in {"/api/save", "/api/exclude", "/api/background", "/api/background/remove"} or self.headers.get("X-Manual-Review") != "1":
                return self.response({"error": "invalid save request"}, status=403)
            origin = self.headers.get("Origin")
            if origin and origin != "http://"+self.headers.get("Host", ""):
                return self.response({"error": "origin mismatch"}, status=403)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(size))
                if self.path == "/api/exclude":
                    self.response({"excluded": review.exclude(payload)})
                elif self.path == "/api/background":
                    self.response({"background": review.save_background(payload)})
                elif self.path == "/api/background/remove":
                    self.response({"removed": review.remove_background(payload)})
                else:
                    self.response({"saved": review.save(payload)})
            except (ValueError, KeyError, TypeError) as exc:
                self.response({"error": str(exc)}, status=400)

        def log_message(self, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile", default="E-020_N00")
    parser.add_argument("--ids", nargs="+", type=int, default=list(range(2134, 2144)))
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address; default is local only. Use a LAN IPv4 address for trusted team access.")
    parser.add_argument("--batch-file", type=Path, help="JSON items with tile and source_id; supports several tiles")
    args = parser.parse_args()
    manifest = json.loads(args.batch_file.read_text(encoding="utf-8")) if args.batch_file else {}
    items = (manifest["items"] if args.batch_file
             else [{"tile": args.tile, "source_id": sid} for sid in args.ids])
    review = ReviewBatch(items, batch_number=manifest.get("batch"), groups=manifest.get("groups"),
                         background_only=manifest.get("review_mode") == "background_only")
    server = ThreadingHTTPServer((args.host, args.port), handler_for(review))
    print(f"MANUAL_REVIEW http://{args.host}:{args.port}/", flush=True)
    print(f"CORRECTIONS {review.corrections}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
