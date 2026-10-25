# -*- coding: utf-8 -*-
"""跨切片碎片核查：判断"切片内无输出"的 GT 是否真的是被切片切断所致。

动机：`runs/stage1_unmatched_diag.json` 把 cause=patch_no_detection 的 GT 记为
"整片 conf>=0.001 都没有框"。但 1024 滑窗（overlap 128，stride 896）会把跨缝的
滑坡切成碎片，碎片本身可能只是滑坡的一角——若邻片存在同一滑坡更完整的实例，
那么"这片没框"就跟切片覆盖无关，不能归因于重叠率不足。

做法：把每条 GT 的局部框换算成绝对坐标，在所有常规滑窗切片里找**包含它且更大**
的同类 GT。这里用包含度（inter / area(fragment)）而不是 IoU：碎片被完整视图
包住时 IoU 会因面积悬殊而变得很小（例如 446x112 的碎片落在 446x815 的完整框里，
IoU 仅 0.138，包含度却是 1.00），IoU 会漏判。

只读 labels，不跑推理。输出每条 GT 的最近邻片完整视图、面积比与包含度。
"""
import argparse
import json
from pathlib import Path

PATCH = 1024
CLASSES = ("SL", "SP", "FL", "CL")
BASE = Path(__file__).resolve().parents[2]


def parse_tile(stem: str) -> tuple[str, int, int] | None:
    """常规滑窗切片名形如 <图幅>_<x0>_<y0>；大滑坡居中裁剪（含 _big_）返回 None。"""
    parts = stem.split("_")
    if len(parts) >= 2 and len(parts[-1]) == 5 and len(parts[-2]) == 5 \
            and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2]), int(parts[-2]), int(parts[-1])
    return None


def load_abs(stem: str, ox: int, oy: int, label_dir: Path) -> list[tuple]:
    out = []
    path = label_dir / f"{stem}.txt"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        f = line.split()
        cls, cx, cy, w, h = int(f[0]), *(float(x) for x in f[1:5])
        out.append((cls, stem,
                    (cx - w / 2) * PATCH + ox, (cy - h / 2) * PATCH + oy,
                    (cx + w / 2) * PATCH + ox, (cy + h / 2) * PATCH + oy))
    return out


def containment(frag: tuple, cand: tuple) -> float:
    """frag 被 cand 覆盖的面积占比。"""
    ix0, iy0 = max(frag[2], cand[2]), max(frag[3], cand[3])
    ix1, iy1 = min(frag[4], cand[4]), min(frag[5], cand[5])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = (frag[4] - frag[2]) * (frag[5] - frag[3])
    return inter / area if area > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag", type=Path,
                        default=BASE / "runs" / "stage1_unmatched_diag.json")
    parser.add_argument("--labels", type=Path,
                        default=BASE / "data" / "processed" / "yolo" / "labels" / "val")
    parser.add_argument("--cause", default="patch_no_detection",
                        help="要核查的归因类别")
    parser.add_argument("--min-containment", type=float, default=0.8)
    args = parser.parse_args()

    diag = json.loads(args.diag.read_text(encoding="utf-8"))
    recs = [r for r in diag["records"] if r["cause"] == args.cause]

    all_gt, tiles = [], {}
    for path in sorted(args.labels.glob("*.txt")):
        parsed = parse_tile(path.stem)
        if not parsed:
            continue
        _prefix, ox, oy = parsed
        all_gt.extend(load_abs(path.stem, ox, oy, args.labels))
        tiles[path.stem] = (ox, oy)
    print(f"常规滑窗切片 {len(tiles)} 个，GT 实例 {len(all_gt)} 条，"
          f"待核查 {len(recs)} 条（cause={args.cause}）")

    for r in recs:
        parsed = parse_tile(r["stem"])
        if not parsed:
            print(f"\n[{r['stem']}] 非常规滑窗切片，跳过")
            continue
        ox, oy = parsed[1], parsed[2]
        cls_id = CLASSES.index(r["cls"])
        mine = [g for g in all_gt if g[1] == r["stem"] and g[0] == cls_id]
        print(f"\n[{r['stem']}] 目标类 {r['cls']} size={r['size_px']}px  "
              f"该片同类 GT {len(mine)} 条")
        for g in mine:
            best = None
            for h in all_gt:
                if h[0] != cls_id or h[1] == g[1]:
                    continue
                c = containment(g, h)
                if best is None or c > best[0]:
                    best = (c, h)
            loc = f"local=({g[2]-ox:.0f},{g[3]-oy:.0f},{g[4]-ox:.0f},{g[5]-oy:.0f})"
            size = f"{g[4]-g[2]:.0f}x{g[5]-g[3]:.0f}"
            g_area = (g[4] - g[2]) * (g[5] - g[3])
            if best and best[0] >= args.min_containment:
                c, h = best
                hx, hy = tiles[h[1]]
                h_area = (h[4] - h[2]) * (h[5] - h[3])
                print(f"  {loc} 尺寸 {size} -> 被邻片 {h[1]} 包含 {c:.2f}，"
                      f"邻片内框=({h[2]-hx:.0f},{h[3]-hy:.0f},{h[4]-hx:.0f},{h[5]-hy:.0f}) "
                      f"尺寸 {h[4]-h[2]:.0f}x{h[5]-h[3]:.0f} 面积比 {h_area/g_area:.2f}x")
            else:
                extra = f"（best 包含度={best[0]:.3f}）" if best else ""
                print(f"  {loc} 尺寸 {size} -> 邻片无同类完整视图{extra}")


if __name__ == "__main__":
    main()
