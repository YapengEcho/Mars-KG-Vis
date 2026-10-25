# -*- coding: utf-8 -*-
"""E0：SL/CL 边界标签核查（spec §6）。

做三件事：
  1. 在指定 split 上找"SL/CL 交界候选"：GT 类 ∈ {SL, CL} 且 (Stage1 argmax != GT 类
     或 最大类概率 < --max-conf)，按"最可疑优先"排序；
  2. 为每个候选出一张三联图（GT 框=红、Stage1 最高分框=绿、SHP 真实多边形=蓝），
     写 runs/label_audit/sheet_*.jpg 联表 + audit_manifest.csv（human_label 列留空待人工填）；
  3. `--mode score` 读回人工填好的 CSV，统计标签错误率并写 label_audit_report.json。

用法：
    python audit_cl_sl_labels.py --mode sample --limit 40
    python audit_cl_sl_labels.py --mode score --filled runs/label_audit/audit_manifest_filled.csv
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import shapefile
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parents[2]
DATA_YAML = BASE / "data" / "processed" / "yolo" / "data.yaml"
WEIGHTS = BASE / "runs" / "detect" / "yolov8n_mars_rebal_w2" / "weights" / "best.pt"
SHP = (BASE / "data" / "raw" / "mars_landslide_boundaries"
       / "landslide_boundaries" / "火星滑坡-带边界.shp")
OUT_DIR = BASE / "runs" / "label_audit"

PATCH_PX = 1024
CLASS_NAMES = ("SL", "SP", "FL", "CL")
BORDER_CLASSES = (0, 3)   # SL, CL


def parse_stem(stem: str):
    """patch 文件名 -> (tile_quad, 窗口左上角x, 窗口左上角y, 缩放比)。

    win:    f"{q}_{px0:05d}_{py0:05d}"                -> 原生 1024 窗口，缩放 1.0
    center: f"{q}_big_{seq}_{wx0:05d}_{wy0:05d}_{win}" -> win 缩放到 1024，缩放 1024/win
    q 形如 'E-088_N-08'，本身含下划线，故前两段合并为 q。
    """
    parts = stem.split("_")
    quad = "_".join(parts[:2])
    rest = parts[2:]
    if rest and rest[0] == "big":
        wx0, wy0, win = int(rest[2]), int(rest[3]), int(rest[4])
        return quad, wx0, wy0, PATCH_PX / float(win)
    return quad, int(rest[0]), int(rest[1]), 1.0


def load_shp_rings(seq: int):
    """SHP 中某个 source_id 的所有环（经纬度点列）。"""
    reader = shapefile.Reader(str(SHP))
    fields = [f[0] for f in reader.fields[1:]]
    i_seq = fields.index("序号")
    for sr in reader.iterShapeRecords():
        if int(sr.record[i_seq]) != seq:
            continue
        pts = [(float(p[0]), float(p[1])) for p in sr.shape.points]
        parts = list(sr.shape.parts) + [len(pts)]
        return [pts[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
    return []


def polygon_to_patch_px(rings, quad: str, wx0: int, wy0: int, scale: float):
    """经纬度环 -> patch 像素环（按图幅 geotransform 投影后减去窗口原点、乘缩放比）。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_yolo_dataset import CTX_DIR, discover_tifs, lonlat_to_pixel, read_geotransform

    tifs = discover_tifs(CTX_DIR)
    if quad not in tifs:
        return []
    gt = read_geotransform(tifs[quad])
    out = []
    for ring in rings:
        px = []
        for lon, lat in ring:
            x, y = lonlat_to_pixel(lon, lat, gt)
            px.append(((x - wx0) * scale, (y - wy0) * scale))
        out.append(px)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sample", choices=["sample", "score"])
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--data", default=str(DATA_YAML))
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-conf", type=float, default=0.35, help="低于此最大类概率即视为可疑")
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=40, help="最多出多少候选（按可疑度排序）")
    ap.add_argument("--per-sheet", type=int, default=8, help="每张联表放几个候选")
    ap.add_argument("--filled", default=str(OUT_DIR / "audit_manifest_filled.csv"))
    ap.add_argument("--hide-gt", action="store_true",
                    help="出盲判联表：图上只印行号、清单不写 gt/Stage1 类别，另存 *_blind 避免锚定")
    args = ap.parse_args()

    if args.mode == "score":
        score(args)
        return
    sample(args)


def sample(args):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_recall_report import (infer, load_ground_truth, load_images,
                                    resolve_dataset_root)

    root = resolve_dataset_root(Path(args.data))
    images = load_images(root, args.split)
    gt = load_ground_truth(root, args.split, images, patch_px=PATCH_PX)
    preds = infer(Path(args.weights), images, args.imgsz, args.conf, args.device, False)

    candidates = []
    for image in images:
        boxes, scores, classes = preds.get(
            image.stem,
            (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)),
        )
        for box_i, (cls, x0, y0, x1, y1) in enumerate(gt.get(image.stem, [])):
            if cls not in BORDER_CLASSES:
                continue
            if len(scores) == 0:
                top_cls, top_conf = -1, 0.0
            else:
                j = int(np.argmax(scores))
                top_cls, top_conf = int(classes[j]), float(scores[j])
            suspicious = (top_cls != cls) or (top_conf < args.max_conf)
            if not suspicious:
                continue
            candidates.append({
                "stem": image.stem, "path": str(image), "box_index": box_i,
                "gt_class": CLASS_NAMES[cls], "stage1_class": (
                    CLASS_NAMES[top_cls] if top_cls >= 0 else "none"),
                "stage1_conf": round(top_conf, 4),
                "box": [round(float(v), 2) for v in (x0, y0, x1, y1)],
            })

    candidates.sort(key=lambda c: c["stage1_conf"])
    candidates = candidates[:args.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per = args.per_sheet
    sheet_paths = []
    for sheet_i in range(0, max(len(candidates), 1), per):
        chunk = candidates[sheet_i:sheet_i + per]
        if not chunk:
            break
        tile = 320
        sheet = Image.new("RGB", (tile * 2, tile * math.ceil(len(chunk) / 2)), (20, 20, 20))
        for k, cand in enumerate(chunk):
            img = Image.open(cand["path"]).convert("RGB")
            x0, y0, x1, y1 = cand["box"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            side = max(x1 - x0, y1 - y0, 32.0) * 2.0
            left = int(max(0, min(cx - side / 2, PATCH_PX - side)))
            top = int(max(0, min(cy - side / 2, PATCH_PX - side)))
            crop = img.crop((left, top, int(min(PATCH_PX, left + side)),
                             int(min(PATCH_PX, top + side)))).resize((tile, tile))
            scale = tile / max(side, 1.0)
            draw = ImageDraw.Draw(crop)
            draw.rectangle([((x0 - left) * scale, (y0 - top) * scale),
                            ((x1 - left) * scale, (y1 - top) * scale)], outline=(255, 60, 60), width=2)
            quad, wx0, wy0, zoom = parse_stem(cand["stem"])
            for ring in polygon_to_patch_px(load_shp_rings(_seq_of(cand)), quad, wx0, wy0, zoom):
                pts = [((px - left) * scale, (py - top) * scale) for px, py in ring]
                if len(pts) >= 3:
                    draw.polygon(pts, outline=(80, 140, 255))
            draw.text((4, 4), f"#{sheet_i + k + 1}" if args.hide_gt else
                      f"gt={cand['gt_class']} s1={cand['stage1_class']} "
                      f"c={cand['stage1_conf']:.2f}", fill=(0, 255, 0))
            sheet.paste(crop, ((k % 2) * tile, (k // 2) * tile))
        sheet_path = OUT_DIR / f"sheet_{sheet_i // per:03d}{'_blind' if args.hide_gt else ''}.jpg"
        sheet.save(sheet_path, quality=88)
        sheet_paths.append(str(sheet_path))

    manifest = OUT_DIR / f"audit_manifest{'_blind' if args.hide_gt else ''}.csv"
    fields = ["sheet", "stem", "box_index", "human_label", "note"]
    if not args.hide_gt:
        fields[3:3] = ["gt_class", "stage1_class", "stage1_conf"]
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, cand in enumerate(candidates):
            row = dict(cand)
            row.pop("path", None)
            row.pop("box", None)
            if args.hide_gt:
                row.pop("gt_class", None)
                row.pop("stage1_class", None)
                row.pop("stage1_conf", None)
            row["sheet"] = sheet_paths[i // per] if per else ""
            row["human_label"] = ""
            row["note"] = ""
            writer.writerow(row)

    print(f"候选 {len(candidates)} 条 -> {manifest}")
    print(f"联表 {len(sheet_paths)} 张 -> {OUT_DIR}")
    print("人工判读：在 audit_manifest.csv 的 human_label 列填 SL/SP/FL/CL 或 wrong，"
          "存为 audit_manifest_filled.csv 后运行 --mode score")


def _seq_of(cand):
    """从文件名解析 source_id：center patch 名里带 seq，win patch 名里没有（返回 -1）。"""
    parts = cand["stem"].split("_")
    if len(parts) > 2 and parts[2] == "big":
        return int(parts[3])
    return -1


def score(args):
    filled = Path(args.filled)
    if not filled.exists():
        print(f"未找到人工标注文件 {filled}")
        print("先运行 --mode sample 生成 audit_manifest.csv，人工填写 human_label 后另存为该文件名")
        return
    rows = list(csv.DictReader(filled.open(encoding="utf-8-sig")))
    rows = [r for r in rows if (r.get("human_label") or "").strip()]
    if not rows:
        print("human_label 列全为空，无统计意义")
        return
    if not (rows[0].get("gt_class") or "").strip():
        base = OUT_DIR / "audit_manifest.csv"
        key2gt = {(r["stem"], r["box_index"]): r["gt_class"]
                  for r in csv.DictReader(base.open(encoding="utf-8-sig"))}
        for r in rows:
            r["gt_class"] = key2gt.get((r["stem"], r["box_index"]), "")
        rows = [r for r in rows if (r.get("gt_class") or "").strip()]

    per_gt = {}
    for r in rows:
        gt = r["gt_class"].strip()
        human = r["human_label"].strip().upper()
        bucket = per_gt.setdefault(gt, {"n": 0, "changed": 0, "to": {}})
        bucket["n"] += 1
        if human != gt:
            bucket["changed"] += 1
            bucket["to"][human] = bucket["to"].get(human, 0) + 1

    report = {
        "audited_total": len(rows),
        "filled_csv": str(filled),
        "per_gt_class": {
            gt: {"n": v["n"], "changed": v["changed"],
                 "error_rate": round(v["changed"] / v["n"], 4), "changed_to": v["to"]}
            for gt, v in sorted(per_gt.items())
        },
    }
    sl_cl = sum(v["changed"] for gt, v in per_gt.items() if gt in ("SL", "CL"))
    report["sl_cl_boundary_error_rate"] = round(
        sl_cl / max(sum(v["n"] for gt, v in per_gt.items() if gt in ("SL", "CL")), 1), 4)
    report["decision_hint"] = (
        "SL/CL 边界错误率 >=0.15 时，建议 SL↔CL 不硬分、直接降级为人工复核；"
        "否则可保留硬分并只对概率差 < δ 的样本复核。"
    )
    out = OUT_DIR / "label_audit_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
