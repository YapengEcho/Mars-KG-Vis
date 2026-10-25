# -*- coding: utf-8 -*-
"""固定尺寸滑动窗口切图 + 生成 YOLO 检测数据集（SHP 驱动框 + geotransform 映射）。

把 SHP 多边形边界（唯一真值）转成 YOLOv8 可直接训练的数据集。核心逻辑：

  1. 每个滑坡的检测框 = 多边形在火星等距圆柱投影下的四至（AABB），
     经 TIFF 的 geotransform 映射到像素空间（读 ModelTiepoint/ModelPixelScale，
     不用 4°/宽高硬算）。bbox 跨图幅的滑坡在每个覆盖图幅各登记一次。
  2. 滑坡按框尺度分两档，避开「大滑坡被滑动窗口切碎 → 满屏框」的硬伤：
     - 中小滑坡（框长边 <= --max-side）：走固定 patch 滑动窗口求交；
     - 大滑坡（框长边 > --max-side）：以框中心按自身尺度裁窗口，缩放到 patch。
  3. 正样本 patch 分配 + 负样本（无滑坡背景）抽样 + 空间隔离划分：
     - 默认（不传 --split-plan）：正样本按滑坡连通分量、负样本按空间网格块，
       确定性划分，同一滑坡不跨 split；
     - --split-plan <json>：整图幅分组划分（tile -> train/val/test）。同一张
       4°×4° 图幅只进一个集合，跨图幅相连的图幅整组同侧；该 JSON 同时充当
       图幅白名单（未下载完成的图幅天然被排除）。任一出框滑坡若涉及分属不同
       split 的图幅，直接报错终止——那说明图幅分组没绑干净，仍是泄漏。
  4. 输出 images/、labels/（YOLO txt 归一化格式）与 data.yaml。

类别：SL=0, SP=1, FL=2, CL=3（对齐论文四分类）。

用法：
    python make_yolo_dataset.py
    python make_yolo_dataset.py --patch 1024 --overlap 128 --neg-ratio 1 --max-side 819
    python make_yolo_dataset.py --split-plan data/raw/tile_split_plan.json
    python make_yolo_dataset.py --split-plan data/raw/tile_split_plan.json --out data/processed/yolo_tile
"""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
import random
import re
from pathlib import Path

import numpy as np
import shapefile
import tifffile
from PIL import Image

BASE = Path(__file__).resolve().parents[2]
RAW = BASE / "data" / "raw"
CTX_DIR = RAW / "ctx_tiles"
OUT = BASE / "data" / "processed" / "yolo"
SHP = RAW / "mars_landslide_boundaries" / "landslide_boundaries" / "火星滑坡-带边界.shp"

CLASSES = {"SL": 0, "SP": 1, "FL": 2, "CL": 3}
CLASS_NAMES = ["SL", "SP", "FL", "CL"]

# 火星等距圆柱投影（GeoKey 1026 = Mars_2015_Ocentric_Equirectangular_clon_0，
# GeoKey 2057 = 半径 R=3396190m，中央经线 0°）。整个世界坐标由经纬度唯一确定：
#   x = R * lon * pi/180,  y = R * lat * pi/180  （单位米）
R_MARS = 3396190.0
DEG = math.pi / 180.0

MAX_WIN = 12000  # centered crop 最大窗口边长（px），避免单窗口内存过大


def discover_tifs(root: Path):
    found = {}
    for tif in root.glob("*/MurrayLab_CTX_V01_*.tif"):
        m = re.search(r"(E-?\d{3}_N-?\d{2})", tif.name)
        if m:
            found[m.group(1)] = tif
    return found


def read_geotransform(tif_path):
    """读 GeoTIFF ModelTiepoint + ModelPixelScale -> GDAL 六参 geotransform。

    返回 (origin_x, pixel_w, 0, origin_y, 0, pixel_h)，pixel_h 取负（北在顶）。
    """
    with tifffile.TiffFile(str(tif_path)) as t:
        page = t.pages[0]
        scale = page.tags[33550].value   # ModelPixelScale (sx, sy, sz)
        tie = page.tags[33922].value     # ModelTiepoint (i, j, k, X, Y, Z)
    return (tie[3], scale[0], 0.0, tie[4], 0.0, -scale[1])


def lonlat_to_pixel(lon, lat, gt):
    """经纬度（度）-> 像素坐标，走火星等距圆柱世界坐标 + geotransform。"""
    px = (R_MARS * lon * DEG - gt[0]) / gt[1]
    py = (R_MARS * lat * DEG - gt[3]) / gt[5]
    return px, py


def aabb_to_pixel(aabb, gt, W, H):
    """多边形四至 (lon_min, lat_min, lon_max, lat_max) -> 像素四至，裁剪到 [0,W]×[0,H]。"""
    lon_min, lat_min, lon_max, lat_max = aabb
    px0 = (R_MARS * lon_min * DEG - gt[0]) / gt[1]
    px1 = (R_MARS * lon_max * DEG - gt[0]) / gt[1]
    py0 = (R_MARS * lat_max * DEG - gt[3]) / gt[5]  # 北边（lat_max → py 小）
    py1 = (R_MARS * lat_min * DEG - gt[3]) / gt[5]  # 南边（lat_min → py 大）
    if px0 > px1:
        px0, px1 = px1, px0
    if py0 > py1:
        py0, py1 = py1, py0
    x0 = max(0.0, min(px0, W))
    x1 = max(0.0, min(px1, W))
    y0 = max(0.0, min(py0, H))
    y1 = max(0.0, min(py1, H))
    return x0, y0, x1, y1


def load_shp_polygons():
    """读 SHP，返回 {source_id: (lon_min, lat_min, lon_max, lat_max)}。

    用 shape.bbox 直接取多边形四至（孔洞在外环内，不影响四至）。
    """
    sf = shapefile.Reader(str(SHP))
    out = {}
    for sr in sf.iterShapeRecords():
        sid = int(sr.record["序号"])
        out[sid] = tuple(sr.shape.bbox)
    return out


def _stable_hash(*parts):
    """跨进程稳定的确定性哈希。

    python 内置 hash() 受 PYTHONHASHSEED 影响、跨进程不可复现，不能用于划分。
    这里用 sha256（取前 4 字节）保证同一输入永远得到同一 split。
    """
    s = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest()[:4], "big")


def split_of_seq(seq, val_ratio, test_ratio=0.0):
    """滑坡 source_id -> split（确定性、可复现），同一滑坡所有 patch 必同侧。

    三档：train / val / test，依稳定哈希落在 [0,100) 内判定。
    """
    r = _stable_hash("seq", seq) % 100
    if r < round(test_ratio * 100):
        return "test"
    if r < round((test_ratio + val_ratio) * 100):
        return "val"
    return "train"


def split_of_grid(quad, gx, gy, val_ratio, test_ratio=0.0):
    """负样本按空间网格块分组 -> split（确定性）。"""
    r = _stable_hash("grid", quad, gx, gy) % 100
    if r < round(test_ratio * 100):
        return "test"
    if r < round((test_ratio + val_ratio) * 100):
        return "val"
    return "train"


def split_random_connected_component(pos_patches, neg_patches, val_ratio, test_ratio=0.0):
    """默认划分：正样本按滑坡连通分量、负样本按空间网格块 -> split（确定性）。

    反泄漏：若某 patch 同时含 train 滑坡与 val 滑坡，仅强制归一边会把同一滑坡的
    影像泄到两侧。故先用并查集把「任一 patch 中共现的滑坡」连成连通分量，整个
    连通分量只进一个 split，任何滑坡所有 patch 必同侧。
    """
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for p in pos_patches:
        seqs = {b[5] for b in p[3]}
        s0 = min(seqs)
        for s in seqs:
            union(s0, s)

    comp_split = {}
    tagged_pos = []
    for p in pos_patches:
        comp = find(min(b[5] for b in p[3]))
        if comp not in comp_split:
            comp_split[comp] = split_of_seq(comp, val_ratio, test_ratio)
        tagged_pos.append(p + (comp_split[comp],))
    tagged_neg = [p + (split_of_grid(p[0], p[2][0] // 4096, p[2][1] // 4096,
                                     val_ratio, test_ratio),)
                  for p in neg_patches]
    return tagged_pos, tagged_neg


def load_split_plan(path: Path, discovered: dict) -> dict:
    """读整图幅划分方案（tile -> split），校验取值合法且图幅确实在磁盘上。

    返回的 plan 同时充当图幅白名单：不在其中的图幅（例如仍在下载的半成品）一律
    不参与切图，避免半截 Mosaic.tif 产出垃圾切片。
    """
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    bad = {t: s for t, s in plan.items() if s not in ("train", "val", "test")}
    if bad:
        raise SystemExit(f"[图幅划分] split 取值非法（只允许 train/val/test）: {bad}")
    missing = sorted(t for t in plan if t not in discovered)
    if missing:
        raise SystemExit(f"[图幅划分] 方案内图幅在磁盘上不存在（未解压或名字不符）: {missing}")
    skipped = sorted(t for t in discovered if t not in plan)
    if skipped:
        print(f"[图幅划分] {len(skipped)} 个已解压图幅不在方案内，本轮跳过: {skipped}")
    return plan


def collect_tile_boxes(tifs, records, polys, max_side):
    """逐图幅登记检测框（供划分一致性校验与实际切图共用一次扫描）。

    返回 {tile: {"W", "H", "small", "big"}}。只读 TIFF 头（memmap 不加载像素）。
    """
    out = {}
    for q, tif_path in tifs.items():
        arr = tifffile.memmap(tif_path)
        H, W = arr.shape
        gt = read_geotransform(tif_path)

        # 由图幅 geotransform 反推经纬范围（不靠 4°/宽高硬算）
        k = R_MARS * DEG
        lon_west = gt[0] / k
        lon_east = (gt[0] + W * gt[1]) / k
        lat_north = gt[3] / k
        lat_south = (gt[3] + H * gt[5]) / k

        small, big = [], []
        for sid, x in records.items():
            if x["type"] not in CLASSES:
                continue
            aabb = polys.get(sid)
            if aabb is None:
                continue
            # bbox 覆盖图幅才登记（bbox 跨图幅的在每个覆盖图幅各计一次）
            if (aabb[2] <= lon_west or aabb[0] >= lon_east
                    or aabb[3] <= lat_south or aabb[1] >= lat_north):
                continue
            x0, y0, x1, y1 = aabb_to_pixel(aabb, gt, W, H)
            w = x1 - x0
            h = y1 - y0
            if w <= 1 or h <= 1:
                continue
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            box = (CLASSES[x["type"]], cx, cy, w, h, sid)
            (small if max(w, h) <= max_side else big).append(box)
        out[q] = {"W": W, "H": H, "small": small, "big": big}
    return out


def assert_tile_split_consistency(tile_boxes, plan):
    """任一出框滑坡涉及的图幅必须全部同 split，否则拒绝生成数据集。

    图幅分组的前提是：同一滑坡只在同 split 的图幅里出现。若某滑坡在 val 图幅与
    train 图幅上都出框，它自己就是跨 split 的泄漏通道，必须先把这些图幅并入同一
    组再切图，而不是照常输出。
    """
    seq_tiles = defaultdict(set)
    for q, tb in tile_boxes.items():
        for box in tb["small"] + tb["big"]:
            seq_tiles[box[5]].add(q)

    conflicts = []
    for seq, tiles in seq_tiles.items():
        splits = {plan[t] for t in tiles}
        if len(splits) > 1:
            conflicts.append((seq, sorted(tiles), sorted(splits)))
    if conflicts:
        for seq, tiles, splits in conflicts[:20]:
            print(f"  泄漏: source_id={seq} 出框图幅 {tiles} 分属 {splits}")
        raise SystemExit(
            f"[图幅划分] {len(conflicts)} 条滑坡跨越不同 split 的图幅，"
            f"请把这些图幅并入同一组后再切图")
    return seq_tiles


def slide_name(q2, kind, seq, params, patch):
    """生成唯一且稳定的 patch 文件名（含 source_id 兜底，杜绝覆盖）。

    - win:   按窗口左上角 (px0,py0) 定位，本图幅内唯一；
    - center: 窗口坐标随图幅唯一定位，但不同大滑坡可能落到同窗口，
              追加 seq 保证每个滑坡一条独立文件，绝不互相覆盖。
    """
    if kind == "win":
        px0, py0 = params
        return f"{q2}_{px0:05d}_{py0:05d}"
    # center: 窗口左上角 + 滑坡序号，保证唯一
    wx0, wy0, win = params
    return f"{q2}_big_{seq}_{wx0:05d}_{wy0:05d}_{win}"


def make_starts(size, patch, stride):
    """滑动窗口起点列表，保证右/下边缘也覆盖到。"""
    starts = list(range(0, size - patch + 1, stride))
    if not starts or starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def intersect(x0, y0, x1, y1, px0, py0, ps):
    return x0 < px0 + ps and x1 > px0 and y0 < py0 + ps and y1 > py0


def window_intersects_any_box(boxes, px0, py0, patch):
    """Return whether a window overlaps any source bounding box.

    Large slides are emitted as centered crops rather than regular sliding-window
    labels. Their overlapping regular windows must therefore be excluded from
    both positive and negative candidates; otherwise an unlabeled large slide can
    become a false-negative training sample.
    """
    for _cls, cx, cy, w, h, _seq in boxes:
        if intersect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
                     px0, py0, patch):
            return True
    return False


def clip_box_to_window(box, px0, py0, patch):
    """Clip one pixel-space box to a window and return YOLO-normalized values."""
    cls, cx, cy, w, h, seq = box
    bx0, by0 = cx - w / 2, cy - h / 2
    bx1, by1 = cx + w / 2, cy + h / 2
    if not intersect(bx0, by0, bx1, by1, px0, py0, patch):
        return None
    ix0, iy0 = max(bx0, px0), max(by0, py0)
    ix1, iy1 = min(bx1, px0 + patch), min(by1, py0 + patch)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return (cls, ((ix0 + ix1) / 2 - px0) / patch,
            ((iy0 + iy1) / 2 - py0) / patch,
            (ix1 - ix0) / patch, (iy1 - iy0) / patch, seq)


def collect_center_crop_hits(primary_box, other_boxes, px0, py0, patch):
    """Collect every labeled box intersecting a centered crop.

    Keep the crop's primary slide first so its source ID remains stable in the
    generated filename. Other boxes are clipped to the same crop and retained
    instead of being silently treated as background.
    """
    primary = clip_box_to_window(primary_box, px0, py0, patch)
    if primary is None:
        return []
    hits = [primary]
    primary_seq = primary_box[5]
    for box in other_boxes:
        if box[5] == primary_seq:
            continue
        clipped = clip_box_to_window(box, px0, py0, patch)
        if clipped is not None:
            hits.append(clipped)
    return hits


def is_usable_mixed_window(small_hit, big_hit, max_fraction=0.9):
    """Whether a window with large-slide overlap is safe to keep.

    A large slide alone is represented by a centered crop. In a mixed window,
    keep it only when a small slide is also labeled and the clipped large box
    does not dominate the whole patch.
    """
    if not big_hit:
        return True
    if not small_hit:
        return False
    return all(box[3] <= max_fraction and box[4] <= max_fraction
               for box in big_hit)


def clear_output_dirs(out: Path):
    """Remove stale generated images/labels before writing a fresh dataset."""
    for split in ("train", "val", "test"):
        for kind, suffix in (("images", ".png"), ("labels", ".txt")):
            directory = out / kind / split
            directory.mkdir(parents=True, exist_ok=True)
            for path in directory.glob(f"*{suffix}"):
                path.unlink()

    # Ultralytics caches label scans beside the split directories. A stale
    # cache can make a freshly regenerated dataset appear to have old labels.
    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for cache in labels_dir.glob("*.cache"):
        cache.unlink()


def stretch(crop: np.ndarray) -> np.ndarray:
    """2%~98% 百分位直方图拉伸，NoData(0) 保持 0。"""
    v = crop[crop > 0]
    lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if v.size else (0, 255)
    if hi <= lo:
        hi = lo + 1
    st = np.clip((crop.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    st[crop == 0] = 0
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--neg-ratio", type=float, default=1.0, help="负样本 = 正样本 * ratio")
    ap.add_argument("--max-side", type=float, default=None,
                    help="框边长阈值(px)，超过走 centered crop；默认 0.8*patch")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--test-ratio", type=float, default=0.15,
                    help="独立 test 集比例（空间隔离，默认 0.15）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-plan", default=None,
                    help="整图幅分组划分 JSON（tile -> train/val/test），同时充当图幅白名单")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    stride = args.patch - args.overlap
    max_side = args.max_side if args.max_side else 0.8 * args.patch

    tifs = discover_tifs(CTX_DIR)
    if not tifs:
        print("未发现解压图幅，请先下载解压到", CTX_DIR)
        return

    plan = None
    if args.split_plan:
        plan = load_split_plan(Path(args.split_plan), tifs)
        tifs = {t: p for t, p in tifs.items() if t in plan}

    clear_output_dirs(out)

    data = json.load(open(RAW / "mars_landslides.json", encoding="utf-8"))
    records = {int(x["source_id"]): x for x in data}
    polys = load_shp_polygons()
    missing = [s for s in records if s not in polys]
    if missing:
        print(f"[警告] {len(missing)} 条 JSON 记录在 SHP 中无对应多边形，已跳过: {sorted(missing)[:10]}")

    random.seed(args.seed)

    tile_boxes = collect_tile_boxes(tifs, records, polys, max_side)
    if plan is not None:
        seq_tiles = assert_tile_split_consistency(tile_boxes, plan)
        per_split = Counter(plan[q] for q in tile_boxes)
        print("[图幅划分] 图幅数 " + " / ".join(
            f"{s} {per_split[s]}" for s in ("train", "val", "test")))
        print(f"[图幅划分] 出框滑坡 {len(seq_tiles)} 条，全部只落在单一 split（无跨幅泄漏）")

    # pos_patches: (quad, kind, params, boxes_norm)
    #   kind="win"    params=(px0, py0)
    #   kind="center" params=(wx0, wy0, win)
    # neg_patches: (quad, kind, params)
    pos_patches = []
    neg_patches = []

    for q, tb in tile_boxes.items():
        W, H = tb["W"], tb["H"]
        small, big = tb["small"], tb["big"]

        # 1) 中小滑坡：滑动窗口求交
        xs = make_starts(W, args.patch, stride)
        ys = make_starts(H, args.patch, stride)
        for py0 in ys:
            for px0 in xs:
                hit = []
                for cls, cx, cy, w, h, seq in small:
                    clipped = clip_box_to_window((cls, cx, cy, w, h, seq),
                                                 px0, py0, args.patch)
                    if clipped is not None:
                        hit.append(clipped)
                big_hit = [clip_box_to_window(box, px0, py0, args.patch)
                           for box in big]
                big_hit = [box for box in big_hit if box is not None]
                if big_hit:
                    # A window containing only a large slide is represented by
                    # its centered crop. If it also contains a small slide,
                    # retain both labels only when the large box is not nearly
                    # the full patch; otherwise discard the ambiguous window.
                    if not is_usable_mixed_window(hit, big_hit):
                        continue
                    hit.extend(big_hit)
                if hit:
                    pos_patches.append((q, "win", (px0, py0), hit))
                else:
                    neg_patches.append((q, "win", (px0, py0)))

        # 2) 大滑坡：centered crop 全貌
        for cls, cx, cy, w, h, seq in big:
            side = max(w, h)
            # 窗口必须完整包住框：至少 side，最好 1.25 倍留边。
            # 旧版 min(side*1.25, MAX_WIN) 会把 >MAX_WIN 的框截断，
            # 这里只用图像边界兜底，保证不截断。内存上限由 MAX_WIN 拆块兜底（见下）。
            win = int(min(max(side * 1.25, side), W, H))
            # 超大框超单窗口内存上限时，退化为多窗口拆分（保底 min_win）
            if win > MAX_WIN:
                win = int(min(MAX_WIN, W, H))
                # 多窗口覆盖整个框：以 stride=win 横向/纵向平铺
                xs0 = make_starts(W, win, win)
                ys0 = make_starts(H, win, win)
                xs0 = [x for x in xs0 if x < cx + w / 2 and x + win > cx - w / 2]
                ys0 = [y for y in ys0 if y < cy + h / 2 and y + win > cy - h / 2]
                for y0w in ys0:
                    for x0w in xs0:
                        hits = collect_center_crop_hits(
                            (cls, cx, cy, w, h, seq),
                            small + big,
                            x0w,
                            y0w,
                            win,
                        )
                        if not hits:
                            continue
                        pos_patches.append((q, "center", (x0w, y0w, win),
                                            hits))
                continue
            wx0 = int(max(0, min(cx - win / 2, W - win)))
            wy0 = int(max(0, min(cy - win / 2, H - win)))
            hits = collect_center_crop_hits(
                (cls, cx, cy, w, h, seq),
                small + big,
                wx0,
                wy0,
                win,
            )
            if not hits:
                continue
            pos_patches.append((q, "center", (wx0, wy0, win),
                                hits))

    # 满屏框统计（修复效果验证）
    all_boxes = [b for p in pos_patches for b in p[3]]
    full = sum(1 for b in all_boxes if b[3] > 0.9 or b[4] > 0.9)
    print(f"正样本 patch: {len(pos_patches)}（滑动窗口 {len([p for p in pos_patches if p[1]=='win'])} "
          f"+ centered crop {len([p for p in pos_patches if p[1]=='center'])}）")
    if all_boxes:
        print(f"总框 {len(all_boxes)}，满屏框 {full}（{full / len(all_boxes) * 100:.1f}%）")

    # 抽样负样本
    n_neg = int(len(pos_patches) * args.neg_ratio)
    random.shuffle(neg_patches)
    neg_patches = neg_patches[:n_neg]

    # 划分：plan 模式下 split 完全由 patch 所属图幅决定（跨图幅相连的图幅已整组
    # 同侧，一致性由 assert_tile_split_consistency 校验）；默认模式走滑坡连通
    # 分量 + 负样本空间网格块划分。
    if plan is not None:
        pos_patches = [p + (plan[p[0]],) for p in pos_patches]
        neg_patches = [p + (plan[p[0]],) for p in neg_patches]
        mode = "整图幅分组划分"
    else:
        pos_patches, neg_patches = split_random_connected_component(
            pos_patches, neg_patches, args.val_ratio, args.test_ratio)
        mode = "滑坡连通分量空间隔离划分"

    n_train = sum(1 for p in pos_patches if p[4] == "train") + \
              sum(1 for p in neg_patches if p[3] == "train")
    n_val = sum(1 for p in pos_patches if p[4] == "val") + \
            sum(1 for p in neg_patches if p[3] == "val")
    n_test = sum(1 for p in pos_patches if p[4] == "test") + \
             sum(1 for p in neg_patches if p[3] == "test")
    print(f"{mode}：正样本 {len(pos_patches)}、负样本 {len(neg_patches)}，"
          f"train {n_train} / val {n_val} / test {n_test}")

    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    tif_cache = {}
    count_pos = count_neg = 0
    for kind_pl, p in [("pos", q) for q in pos_patches] + [("neg", q) for q in neg_patches]:
        split = p[4] if kind_pl == "pos" else p[3]
        q2 = p[0]
        if q2 not in tif_cache:
            tif_cache[q2] = tifffile.memmap(tifs[q2])
        arr = tif_cache[q2]

        if p[1] == "win":
            px0, py0 = p[2]
            crop = arr[py0:py0 + args.patch, px0:px0 + args.patch].copy()
            img = Image.fromarray(stretch(crop))
            fname = slide_name(q2, "win", 0, (px0, py0), args.patch)
        else:  # center
            wx0, wy0, win = p[2]
            crop = arr[wy0:wy0 + win, wx0:wx0 + win].copy()
            img = Image.fromarray(stretch(crop))
            img = img.resize((args.patch, args.patch), Image.LANCZOS)
            seq = p[3][0][5]  # 该 centered patch 唯一滑坡的 source_id
            fname = slide_name(q2, "center", seq, (wx0, wy0, win), args.patch)
        img.save(out / "images" / split / f"{fname}.png")

        if kind_pl == "pos":
            with (out / "labels" / split / f"{fname}.txt").open("w") as f:
                for cls, nx, ny, nw, nh, _seq in p[3]:
                    f.write(f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
            count_pos += 1
        else:
            (out / "labels" / split / f"{fname}.txt").touch()
            count_neg += 1

    yaml_lines = [
        f"path: {out.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for i, n in enumerate(CLASS_NAMES):
        yaml_lines.append(f"  {i}: {n}")
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    # 质量报告：类别/框分布、划分计数、满屏比、文件名唯一性、跨幅登记
    n_total = len(pos_patches) + len(neg_patches)
    per_class = Counter()
    for p in pos_patches:
        for b in p[3]:
            per_class[CLASS_NAMES[b[0]]] += 1
    # 唯一性校验：所有写出的文件名不得重复（覆盖会静默丢数据）
    fnames = []
    for kind_pl, p in [("pos", q) for q in pos_patches] + [("neg", q) for q in neg_patches]:
        sp = p[4] if kind_pl == "pos" else p[3]
        q2, kind = p[0], p[1]
        args_b = p[2]
        if kind == "win":
            fnames.append(slide_name(q2, "win", 0, args_b, args.patch))
        else:
            seq = p[3][0][5]
            fnames.append(slide_name(q2, "center", seq, args_b, args.patch))
    dup = len(fnames) - len(set(fnames))
    report = {
        "patch": args.patch, "overlap": args.overlap, "max_side": max_side,
        "val_ratio": args.val_ratio, "test_ratio": args.test_ratio,
        "split_mode": "tile" if plan is not None else "connected_component",
        "split_plan": str(args.split_plan) if plan is not None else None,
        "tiles_per_split": {s: sorted(t for t, v in plan.items() if v == s)
                            for s in ("train", "val", "test")} if plan is not None else None,
        "quad_count": len(tifs),
        "pos_patches": len(pos_patches), "neg_patches": len(neg_patches),
        "split": {"train": n_train, "val": n_val, "test": n_test},
        "landslides_per_split": {s: len({b[5] for p in pos_patches if p[4] == s for b in p[3]})
                                 for s in ("train", "val", "test")},
        "boxes_total": len(all_boxes), "boxes_full_frame_ratio": round(full / len(all_boxes), 4) if all_boxes else 0.0,
        "boxes_per_class": dict(per_class),
        "boxes_per_class_per_split": {
            s: dict(Counter(CLASS_NAMES[b[0]] for p in pos_patches if p[4] == s for b in p[3]))
            for s in ("train", "val", "test")},
        "duplicate_filenames": dup,
        "landslides_covered_seqs": len({b[5] for p in pos_patches for b in p[3]}),
    }
    with (out / "quality_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    (out / "quality_report.txt").write_text(
        f"patch={args.patch} overlap={args.overlap} max_side={max_side:.0f}\n"
        f"划分模式 {report['split_mode']}（plan={report['split_plan']}）\n"
        f"图幅 {len(tifs)} 个；正样本 {len(pos_patches)} 负样本 {len(neg_patches)}\n"
        f"划分 train={n_train} val={n_val} test={n_test}\n"
        f"出框滑坡数 train/val/test = "
        f"{report['landslides_per_split']['train']}/{report['landslides_per_split']['val']}"
        f"/{report['landslides_per_split']['test']}\n"
        f"框总数 {len(all_boxes)}，满屏框占比 {report['boxes_full_frame_ratio']*100:.1f}%，"
        f"重复文件名 {dup}（应 0）\n"
        f"按类框数 {dict(per_class)}\n"
        + "".join(f"  按类框数[{s}] {report['boxes_per_class_per_split'][s]}\n"
                  for s in ("train", "val", "test")),
        encoding="utf-8")

    print(f"写入：正样本 {count_pos}，负样本 {count_neg}，"
          f"train {n_train} / val {n_val} / test {n_test}，重复文件名 {dup}（应 0）")
    print("输出目录:", out)


if __name__ == "__main__":
    main()
