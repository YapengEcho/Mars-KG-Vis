# -*- coding: utf-8 -*-

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
import random
import re
from pathlib import Path
from typing import TypedDict

import numpy as np
import shapefile
import tifffile
from PIL import Image

if __package__:
    from .check_tile_connectivity import build_components
else:
    from check_tile_connectivity import build_components

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

GeoTransform = tuple[float, float, float, float, float, float]
Bounds = tuple[float, float, float, float]
LandslideBox = tuple[int, float, float, float, float, int]
CropCoordinates = tuple[int, int] | tuple[int, int, int]
PositivePatch = tuple[str, str, CropCoordinates, list[LandslideBox]]
NegativePatch = tuple[str, str, tuple[int, int]]
AssignedPatch = tuple[str, str, CropCoordinates, list[LandslideBox], str]
AssignedNegativePatch = tuple[str, str, tuple[int, int], str]


class TileBoxes(TypedDict):
    W: int
    H: int
    bounds: Bounds
    small: list[LandslideBox]
    big: list[LandslideBox]


def discover_tifs(root: Path):
    found = {}
    for tif in root.glob("*/MurrayLab_CTX_V01_*.tif"):
        m = re.search(r"(E-?\d{3}_N-?\d{2})", tif.name)
        if m:
            found[m.group(1)] = tif
    return found


def read_geotransform(tif_path: str | Path) -> GeoTransform:
    """从 GeoTIFF 读取图幅起点和像元大小，按 GDAL 六参数格式返回。"""
    with tifffile.TiffFile(str(tif_path)) as t:
        page = t.pages[0]
        if isinstance(page, tifffile.TiffFrame):
            page = page.aspage()
        scale = page.tags[33550].value   # 记录像元大小。
        tie = page.tags[33922].value     # 记录影像和地图的对应点。
    return tie[3], scale[0], 0.0, tie[4], 0.0, -scale[1]


def lonlat_to_pixel(lon: float, lat: float, gt: GeoTransform) -> tuple[float, float]:
    """用火星等距圆柱投影把经纬度转到平面，再换算成图像列号和行号。"""
    px = (R_MARS * lon * DEG - gt[0]) / gt[1]
    py = (R_MARS * lat * DEG - gt[3]) / gt[5]
    return px, py


def aabb_to_pixel(aabb: Bounds, gt: GeoTransform, width: int, height: int) -> Bounds:
    """把多边形的外接范围换成像素位置，超出图幅的部分截到图像边缘。"""
    lon_min, lat_min, lon_max, lat_max = aabb
    # 投影坐标减去图幅起点，再除以每个像元对应的距离。
    px0 = (R_MARS * lon_min * DEG - gt[0]) / gt[1]
    px1 = (R_MARS * lon_max * DEG - gt[0]) / gt[1]
    # 北边在图像上方，对应较小的行号。
    py0 = (R_MARS * lat_max * DEG - gt[3]) / gt[5]
    py1 = (R_MARS * lat_min * DEG - gt[3]) / gt[5]
    if px0 > px1:
        px0, px1 = px1, px0
    if py0 > py1:
        py0, py1 = py1, py0
    x0 = max(0.0, min(px0, width))
    x1 = max(0.0, min(px1, width))
    y0 = max(0.0, min(py0, height))
    y1 = max(0.0, min(py1, height))
    return x0, y0, x1, y1


def load_shp_polygons() -> dict[int, Bounds]:
    """从 SHP 取每个滑坡的外接经纬度范围，用原编号索引。

    这里用 shape.bbox 取最外侧的边界，内部孔洞不影响这个范围。
    """
    out: dict[int, Bounds] = {}
    with shapefile.Reader(str(SHP)) as sf:
        for sr in sf.iterShapeRecords():
            record, shape = sr.record, sr.shape
            if record is None or shape is None:
                raise ValueError("滑坡 SHP 条目缺少属性或几何。")
            source_id = record["序号"]
            if not isinstance(source_id, (int, float, str)):
                raise ValueError(f"滑坡 SHP 的序号无效：{source_id!r}")
            lon_min, lat_min, lon_max, lat_max = shape.bbox
            out[int(source_id)] = lon_min, lat_min, lon_max, lat_max
    return out


def load_split_plan(path: Path, discovered: dict[str, Path]) -> dict[str, str]:
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


def collect_tile_boxes(
    tifs: dict[str, Path],
    records: dict[int, dict],
    polys: dict[int, Bounds],
    max_side: float,
) -> dict[str, TileBoxes]:
    """逐图幅登记检测框（供划分一致性校验与实际切图共用一次扫描）。

    返回 {tile: {"W", "H", "small", "big"}}。只读 TIFF 头（memmap 不加载像素）。
    """
    out: dict[str, TileBoxes] = {}
    for q, tif_path in tifs.items():
        arr = tifffile.memmap(tif_path)
        height, width = arr.shape
        gt = read_geotransform(tif_path)

        # 由图幅 geotransform 反推经纬范围（不靠 4°/宽高硬算）
        k = R_MARS * DEG
        lon_west = gt[0] / k
        lon_east = (gt[0] + width * gt[1]) / k
        lat_north = gt[3] / k
        lat_south = (gt[3] + height * gt[5]) / k

        small: list[LandslideBox] = []
        big: list[LandslideBox] = []
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
            x0, y0, x1, y1 = aabb_to_pixel(aabb, gt, width, height)
            w = x1 - x0
            h = y1 - y0
            if w <= 1 or h <= 1:
                continue
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            box = (CLASSES[x["type"]], cx, cy, w, h, sid)
            (small if max(w, h) <= max_side else big).append(box)
        out[q] = {
            "W": width, "H": height, "bounds": (lon_west, lat_south, lon_east, lat_north),
            "small": small, "big": big,
        }
    return out


def tiles_by_source(tile_boxes: dict[str, TileBoxes]) -> dict[int, set[str]]:
    seq_tiles: dict[int, set[str]] = defaultdict(set)
    for q, tb in tile_boxes.items():
        for box in tb["small"] + tb["big"]:
            seq_tiles[box[5]].add(q)
    return seq_tiles


def overlapping_tile_pairs(tile_boxes: dict[str, TileBoxes]):
    # Ignore floating-point noise at a shared edge, far below one CTX pixel.
    tolerance = 1e-9
    footprints = [(name, tile["bounds"]) for name, tile in tile_boxes.items()]
    for (left, (w1, s1, e1, n1)), (right, (w2, s2, e2, n2)) in combinations(footprints, 2):
        if (min(e1, e2) - max(w1, w2) > tolerance
                and min(n1, n2) - max(s1, s2) > tolerance):
            yield left, right


def make_tile_split_plan(
    tile_boxes: dict[str, TileBoxes], positive_tiles: set[str],
    val_ratio: float, test_ratio: float, seed: int,
) -> dict[str, str]:
    """Assign entire groups of tiles, including shared landslides and overlapping footprints."""
    edges = list(tiles_by_source(tile_boxes).values())
    edges.extend({left, right} for left, right in overlapping_tile_pairs(tile_boxes))
    roots = build_components(edges)
    components: dict[str, set[str]] = defaultdict(set)
    for tile in tile_boxes:
        components[roots.get(tile, tile)].add(tile)

    positive_groups, background_groups = [], []
    for group in sorted(components.values(), key=min):
        if group & positive_tiles:
            positive_groups.append(group)
        else:
            background_groups.append(group)

    group_count = len(positive_groups)
    required = 1 + int(val_ratio > 0) + int(test_ratio > 0)
    if group_count < required:
        raise SystemExit(f"[图幅划分] 只有 {group_count} 个含滑坡的独立图幅组，需要 {required} 个。"
                         "请补充独立图幅；只导出训练集可设置 --val-ratio 0 --test-ratio 0。")

    rng = random.Random(seed)
    rng.shuffle(positive_groups)
    plan: dict[str, str] = {}
    for split, ratio, reserve in (("test", test_ratio, 1 + int(val_ratio > 0)), ("val", val_ratio, 1)):
        if ratio:
            count = min(max(1, round(group_count * ratio)), len(positive_groups) - reserve)
            for group in positive_groups[:count]:
                plan.update(dict.fromkeys(group, split))
            positive_groups = positive_groups[count:]
    for group in positive_groups:
        plan.update(dict.fromkeys(group, "train"))
    for group in background_groups:
        split = rng.choices(("train", "val", "test"), (1 - val_ratio - test_ratio, val_ratio, test_ratio))[0]
        plan.update(dict.fromkeys(group, split))
    return plan


def assert_tile_split_consistency(tile_boxes: dict[str, TileBoxes], plan: dict[str, str]):
    """拒绝相互重叠或含有同一滑坡的图幅进入不同集合。"""
    seq_tiles = tiles_by_source(tile_boxes)

    conflicts = []
    for seq, tiles in seq_tiles.items():
        splits = {plan[t] for t in tiles}
        if len(splits) > 1:
            conflicts.append((seq, sorted(tiles), sorted(splits)))
    if conflicts:
        for seq, conflict_tiles, conflict_splits in conflicts[:20]:
            print(f"  泄漏: source_id={seq} 出框图幅 {conflict_tiles} 分属 {conflict_splits}")
        raise SystemExit(
            f"[图幅划分] {len(conflicts)} 条滑坡跨越不同 split 的图幅，"
            f"请把这些图幅并入同一组后再切图")
    for left, right in overlapping_tile_pairs(tile_boxes):
        if plan[left] != plan[right]:
            raise SystemExit(f"[图幅划分] 图幅 {left} 和 {right} 的实际覆盖范围重叠，"
                             "请把这两个图幅分到同一集合。")
    return seq_tiles


def slide_name(q2, kind, seq, params, patch):
    """生成唯一且稳定的 patch 文件名（含 source_id 兜底，杜绝覆盖）。

    - win:   按窗口左上角 (px0,py0) 定位，本图幅内唯一；
    - center: 窗口坐标随图幅唯一定位，但不同大滑坡可能落到同窗口，
              追加 seq 保证每个滑坡一条独立文件，绝不互相覆盖。

    patch 参数保留给已有调用。
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


def clip_box_to_window(
    box: LandslideBox, px0: int, py0: int, patch: int,
) -> LandslideBox | None:
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


def collect_center_crop_hits(
    primary_box: LandslideBox, other_boxes: list[LandslideBox],
    px0: int, py0: int, patch: int,
) -> list[LandslideBox]:
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
    ap.add_argument("--val-ratio", type=float, default=0.2,
                    help="独立验证图幅组比例（默认 0.2）")
    ap.add_argument("--test-ratio", type=float, default=0.15,
                    help="独立测试图幅组比例（默认 0.15）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-plan", default=None,
                    help="整图幅分组划分 JSON（tile -> train/val/test），同时充当图幅白名单")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if not args.split_plan and not (0 <= args.val_ratio and 0 <= args.test_ratio
                                    and args.val_ratio + args.test_ratio < 1):
        ap.error("验证、测试比例须为非负数且相加小于 1。")

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

    data = json.loads((RAW / "mars_landslides.json").read_text(encoding="utf-8"))
    records = {int(x["source_id"]): x for x in data}
    polys = load_shp_polygons()
    missing = [s for s in records if s not in polys]
    if missing:
        print(f"[警告] {len(missing)} 条 JSON 记录在 SHP 中无对应多边形，已跳过: {sorted(missing)[:10]}")

    random.seed(args.seed)

    tile_boxes = collect_tile_boxes(tifs, records, polys, max_side)
    if plan is not None:
        assert_tile_split_consistency(tile_boxes, plan)

    # pos_patches: (quad, kind, params, boxes_norm)
    #   kind="win"    params=(px0, py0)
    #   kind="center" params=(wx0, wy0, win)
    # neg_patches: (quad, kind, params)
    pos_patches: list[PositivePatch] = []
    neg_patches: list[NegativePatch] = []

    for q, tb in tile_boxes.items():
        width, height = tb["W"], tb["H"]
        small, big = tb["small"], tb["big"]

        # 1) 中小滑坡：滑动窗口求交
        xs = make_starts(width, args.patch, stride)
        ys = make_starts(height, args.patch, stride)
        for py0 in ys:
            for px0 in xs:
                hit = []
                for box in small:
                    clipped = clip_box_to_window(box, px0, py0, args.patch)
                    if clipped is not None:
                        hit.append(clipped)
                big_hit = [
                    clipped_box for box in big
                    if (clipped_box := clip_box_to_window(box, px0, py0, args.patch)) is not None
                ]
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
            win = int(min(max(side * 1.25, side), width, height))
            # 超大框超单窗口内存上限时，退化为多窗口拆分（保底 min_win）
            if win > MAX_WIN:
                win = int(min(MAX_WIN, width, height))
                # 多窗口覆盖整个框：以 stride=win 横向/纵向平铺
                xs0 = make_starts(width, win, win)
                ys0 = make_starts(height, win, win)
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
            wx0 = int(max(0, min(cx - win / 2, width - win)))
            wy0 = int(max(0, min(cy - win / 2, height - win)))
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

    split_mode = "tile" if plan is not None else "tile_auto"
    if plan is None:
        plan = make_tile_split_plan(tile_boxes, {p[0] for p in pos_patches},
                                    args.val_ratio, args.test_ratio, args.seed)

    split_positive: list[AssignedPatch] = [p + (plan[p[0]],) for p in pos_patches]
    split_negative: list[AssignedNegativePatch] = [p + (plan[p[0]],) for p in neg_patches]
    per_split = Counter(plan.values())
    print("[图幅划分] 图幅数 " + " / ".join(
        f"{s} {per_split[s]}" for s in ("train", "val", "test")))

    # 空框列表表示背景裁片，写图和标签时可以使用同一种记录格式。
    patches_to_write = list(split_positive)
    patches_to_write.extend(
        (tile, kind, coords, [], split) for tile, kind, coords, split in split_negative
    )
    patch_counts = Counter(p[-1] for p in patches_to_write)
    n_train, n_val, n_test = patch_counts["train"], patch_counts["val"], patch_counts["test"]
    print(f"整图幅分组：正样本 {len(pos_patches)}、负样本 {len(neg_patches)}，"
          f"train {n_train} / val {n_val} / test {n_test}")

    clear_output_dirs(out)

    tif_cache = {}
    fnames = []
    count_pos = count_neg = 0
    for q2, kind, coords, boxes, split in patches_to_write:
        if q2 not in tif_cache:
            tif_cache[q2] = tifffile.memmap(tifs[q2])
        arr = tif_cache[q2]

        px0, py0 = coords[0], coords[1]
        win = coords[2] if len(coords) == 3 else args.patch
        crop = arr[py0:py0 + win, px0:px0 + win].copy()
        img = Image.fromarray(stretch(crop))
        if kind == "center":
            img = img.resize((args.patch, args.patch), Image.Resampling.LANCZOS)
        seq = boxes[0][5] if kind == "center" else 0  # 中心裁片用主滑坡编号命名。
        fname = slide_name(q2, kind, seq, coords, args.patch)
        img.save(out / "images" / split / f"{fname}.png")
        fnames.append(fname)

        if boxes:
            with (out / "labels" / split / f"{fname}.txt").open("w") as f:
                for cls, nx, ny, nw, nh, _seq in boxes:
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
    per_class = Counter(CLASS_NAMES[box[0]] for box in all_boxes)
    dup = len(fnames) - len(set(fnames))
    split_plan = args.split_plan
    landslides_per_split = {
        s: len({b[5] for p in split_positive if p[4] == s for b in p[3]})
        for s in ("train", "val", "test")
    }
    boxes_per_class_per_split = {
        s: dict(Counter(CLASS_NAMES[b[0]] for p in split_positive if p[4] == s for b in p[3]))
        for s in ("train", "val", "test")
    }
    boxes_full_frame_ratio = round(full / len(all_boxes), 4) if all_boxes else 0.0
    report = {
        "patch": args.patch, "overlap": args.overlap, "max_side": max_side,
        "val_ratio": args.val_ratio, "test_ratio": args.test_ratio,
        "seed": args.seed,
        "split_mode": split_mode,
        "split_plan": split_plan,
        "tiles_per_split": {s: sorted(t for t, v in plan.items() if v == s)
                            for s in ("train", "val", "test")},
        "quad_count": len(tifs),
        "pos_patches": len(pos_patches), "neg_patches": len(neg_patches),
        "split": {"train": n_train, "val": n_val, "test": n_test},
        "landslides_per_split": landslides_per_split,
        "boxes_total": len(all_boxes), "boxes_full_frame_ratio": boxes_full_frame_ratio,
        "boxes_per_class": dict(per_class),
        "boxes_per_class_per_split": boxes_per_class_per_split,
        "duplicate_filenames": dup,
        "landslides_covered_seqs": len({b[5] for p in pos_patches for b in p[3]}),
    }
    with (out / "quality_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    (out / "quality_report.txt").write_text(
        f"patch={args.patch} overlap={args.overlap} max_side={max_side:.0f}\n"
        f"划分模式 {split_mode}（plan={split_plan}）\n"
        f"图幅 {len(tifs)} 个；正样本 {len(pos_patches)} 负样本 {len(neg_patches)}\n"
        f"划分 train={n_train} val={n_val} test={n_test}\n"
        f"出框滑坡数 train/val/test = "
        f"{landslides_per_split['train']}/{landslides_per_split['val']}"
        f"/{landslides_per_split['test']}\n"
        f"框总数 {len(all_boxes)}，满屏框占比 {boxes_full_frame_ratio*100:.1f}%，"
        f"重复文件名 {dup}（应 0）\n"
        f"按类框数 {dict(per_class)}\n"
        + "".join(f"  按类框数[{s}] {boxes_per_class_per_split[s]}\n"
                  for s in ("train", "val", "test")),
        encoding="utf-8")

    print(f"写入：正样本 {count_pos}，负样本 {count_neg}，"
          f"train {n_train} / val {n_val} / test {n_test}，重复文件名 {dup}（应 0）")
    print("输出目录:", out)


if __name__ == "__main__":
    main()
