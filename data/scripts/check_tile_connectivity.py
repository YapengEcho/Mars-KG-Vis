# -*- coding: utf-8 -*-
"""只读图幅连通分量检查：任何改用整图幅划分方案前先跑，避免跨 split 泄漏。

连接判据（两种，含义不同，别混用）：

  overlap（默认，泄漏判据）
      一条滑坡的 SHP 外接框在某图幅内产生**大于 1px** 的真实重叠，才算连通。
      与 make_yolo_dataset.py 的登记逻辑（collect_tile_boxes）一致：零面积相切、
      不足 1px 的贴边都不会产出检测框，因此不构成泄漏通道。需要读 TIFF 头
      （geotransform），只对方案内的图幅求解。

  touch（下载/盘点判据）
      外接框经 floor 取整覆盖到的全部图幅（含正好落在边界的相邻图幅）。
      与 fl_tile_analysis.py / tile_download_plan 的口径一致，偏保守，
      适合决定「要下载哪些图幅」，不适合判断泄漏。

连通图幅必须整组进同一个 split，否则跨幅滑坡会把影像泄漏到另一侧。

用法：
    python data/scripts/check_tile_connectivity.py --tile E-084_N-08 --tile E-076_N-08
    python data/scripts/check_tile_connectivity.py --mode touch
    python data/scripts/check_tile_connectivity.py --plan data/raw/tile_split_plan_flval.json
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import shapefile

BASE = Path(__file__).resolve().parents[2]
SHP = BASE / "data/raw/mars_landslide_boundaries/landslide_boundaries/火星滑坡-带边界.shp"
DEFAULT_PLAN = BASE / "data/raw/tile_split_plan_flval.json"
DEFAULT_CTX = BASE / "data/raw/ctx_tiles"
QUAD_DEG = 4
R_MARS = 3396190.0
DEG = math.pi / 180.0
MIN_OVERLAP_PX = 1.0


def fmt(tx: int, ty: int) -> str:
    """图幅名：tx/ty 为该图幅的西边/南边（度）。负值带连字符，非负不带。"""
    e = f"E-{abs(tx):03d}" if tx < 0 else f"E{tx:03d}"
    n = f"N-{abs(ty):02d}" if ty < 0 else f"N{ty:02d}"
    return f"{e}_{n}"


def tiles_of_bbox(lo: float, la: float, hi: float, ha: float) -> set[str]:
    """touch 口径：外接框经 floor 覆盖的全部 4°×4° 图幅（含边界相切）。"""
    t0, t1 = math.floor(lo / QUAD_DEG), math.floor(hi / QUAD_DEG)
    u0, u1 = math.floor(la / QUAD_DEG), math.floor(ha / QUAD_DEG)
    return {fmt(x * QUAD_DEG, y * QUAD_DEG)
            for x in range(t0, t1 + 1) for y in range(u0, u1 + 1)}


def load_tile_geo(ctx: Path, names: set[str]) -> dict[str, tuple]:
    """读方案内图幅的 (geotransform, W, H)；只读 TIFF 头，不加载像素。"""
    import re  # pylint: disable=import-outside-toplevel

    import tifffile  # pylint: disable=import-outside-toplevel

    found: dict[str, tuple] = {}
    for tif in ctx.glob("*/MurrayLab_CTX_V01_*.tif"):
        m = re.search(r"(E-?\d{3}_N-?\d{2})", tif.name)
        if not m or m.group(1) not in names:
            continue
        with tifffile.TiffFile(str(tif)) as t:
            page = t.pages[0]
            scale = page.tags[33550].value
            tie = page.tags[33922].value
            shape = page.shape
        gt = (tie[3], scale[0], 0.0, tie[4], 0.0, -scale[1])
        found[m.group(1)] = (gt, shape[1], shape[0])
    return found


def tiles_of_bbox_overlap(aabb, tile_geo: dict[str, tuple]) -> set[str]:
    """overlap 口径：外接框在某图幅内像素重叠 > 1px 才计入（与登记逻辑一致）。"""
    lon_min, lat_min, lon_max, lat_max = aabb
    k = R_MARS * DEG
    hits: set[str] = set()
    for q, (gt, width, height) in tile_geo.items():
        lon_west = gt[0] / k
        lon_east = (gt[0] + width * gt[1]) / k
        lat_north = gt[3] / k
        lat_south = (gt[3] + height * gt[5]) / k
        if (lon_max <= lon_west or lon_min >= lon_east
                or lat_max <= lat_south or lat_min >= lat_north):
            continue
        px0 = (k * lon_min - gt[0]) / gt[1]
        px1 = (k * lon_max - gt[0]) / gt[1]
        py0 = (k * lat_max - gt[3]) / gt[5]
        py1 = (k * lat_min - gt[3]) / gt[5]
        if px0 > px1:
            px0, px1 = px1, px0
        if py0 > py1:
            py0, py1 = py1, py0
        w = min(px1, width) - max(px0, 0.0)
        h = min(py1, height) - max(py0, 0.0)
        if w > MIN_OVERLAP_PX and h > MIN_OVERLAP_PX:
            hits.add(q)
    return hits


def build_components(edges: list[set[str]]) -> dict[str, str]:
    """并查集：返回 {tile: 分量根}。edges 为若干「同一滑坡连通到的图幅集合」。"""
    parent: dict[str, str] = {}

    def find(a: str) -> str:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for group in edges:
        it = iter(sorted(group))
        first = next(it)
        for other in it:
            ra, rb = find(first), find(other)
            if ra != rb:
                parent[ra] = rb
    return {t: find(t) for t in parent}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shp", type=Path, default=SHP)
    ap.add_argument("--plan", type=Path, default=DEFAULT_PLAN,
                    help="外接划分方案 JSON（tile -> train/val/test）")
    ap.add_argument("--mode", choices=("overlap", "touch"), default="overlap",
                    help="overlap=真实像素重叠（泄漏判据，默认）；touch=floor 覆盖（保守盘点）")
    ap.add_argument("--ctx", type=Path, default=DEFAULT_CTX, help="overlap 模式下的解压图幅目录")
    ap.add_argument("--tile", action="append", default=[],
                    help="只报告这些图幅所在的分量，可重复")
    ap.add_argument("--min-size", type=int, default=2, help="只列出规模不小于该值的分量")
    args = ap.parse_args()

    plan: dict[str, str] = {}
    if args.plan.exists():
        plan = json.loads(args.plan.read_text(encoding="utf-8"))

    tile_geo: dict[str, tuple] = {}
    if args.mode == "overlap":
        tile_geo = load_tile_geo(args.ctx, set(plan))
        absent = sorted(t for t in plan if t not in tile_geo)
        if absent:
            print(f"[警告] 方案内 {len(absent)} 张图幅未在 {args.ctx} 找到，无法参与 overlap 求解: {absent}")
        if not tile_geo:
            raise SystemExit("overlap 模式需要解压图幅，请检查 --ctx")

    sf = shapefile.Reader(str(args.shp))
    edges: list[set[str]] = []
    cross = 0
    for sr in sf.iterShapeRecords():
        ts = (tiles_of_bbox_overlap(sr.shape.bbox, tile_geo) if args.mode == "overlap"
              else tiles_of_bbox(*sr.shape.bbox))
        if args.mode == "overlap":
            ts &= tile_geo.keys()
        if len(ts) > 1:
            cross += 1
            edges.append(ts)
    comp = build_components(edges)

    members: dict[str, list[str]] = defaultdict(list)
    for tile, root in comp.items():
        members[root].append(tile)

    print(f"[模式 {args.mode}] SHP 多边形 {len(sf.shapes())} 条，跨图幅 {cross} 条，"
          f"连通图幅 {len(comp)} 张，分量 {len(members)} 个")
    if plan:
        print(f"划分方案 {args.plan.name}: {len(plan)} 张图幅")
        solo = sorted(t for t in plan if t not in comp)
        if solo:
            print(f"方案中 {len(solo)} 张图幅不与任何图幅连通（可单独挪）: {solo}")

    seen: set[str] = set()
    for tile in args.tile:
        if tile not in comp:
            print(f"\n[!] {tile} 不参与任何跨幅连接（独立图幅，可单独挪）")
            continue
        root = comp[tile]
        if root in seen:
            continue
        seen.add(root)
        group = sorted(members[root])
        by_split: dict[str, list[str]] = defaultdict(list)
        for t in group:
            by_split[plan.get(t, "<不在方案内>")].append(t)
        print(f"\n=== {tile} 所在分量：{len(group)} 张图幅 ===")
        for sp in ("train", "val", "test", "<不在方案内>"):
            if by_split.get(sp):
                print(f"  {sp:<12} {by_split[sp]}")

    print(f"\n=== 方案内分量一览（规模 >= {args.min_size}）===")
    plan_roots = {comp[t] for t in plan if t in comp}
    for root in sorted(plan_roots, key=lambda r: -len(members[r])):
        group = [t for t in sorted(members[root]) if t in plan]
        if len(group) < args.min_size:
            continue
        splits = sorted({plan[t] for t in group})
        flag = "  <== 跨 split 泄漏" if len(splits) > 1 else ""
        print(f"  [{len(group)}] {group} splits={splits}{flag}")


if __name__ == "__main__":
    main()