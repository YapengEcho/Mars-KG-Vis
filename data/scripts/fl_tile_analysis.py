# -*- coding: utf-8 -*-
"""按图幅统计各类滑坡 bbox 覆盖数，找出 FL 密集图幅（含下载状态）。"""
import os
import json
import math
import collections

import shapefile

RAW = r"d:\Mars-KG-Vis\data\raw"
SHP = RAW + r"\mars_landslide_boundaries\landslide_boundaries\火星滑坡-带边界.shp"


def fmt(tx, ty):
    e = ("E-" + f"{abs(tx):03d}") if tx < 0 else ("E" + f"{tx:03d}")
    n = ("N-" + f"{abs(ty):02d}") if ty < 0 else ("N" + f"{ty:02d}")
    return f"{e}_{n}"


def tiles(lo, la, hi, ha):
    Q = 4
    t0, t1 = math.floor(lo / Q), math.floor(hi / Q)
    u0, u1 = math.floor(la / Q), math.floor(ha / Q)
    s = set()
    for tx in range(t0, t1 + 1):
        for ty in range(u0, u1 + 1):
            s.add(fmt(tx * Q, ty * Q))
    return s


def main():
    data = json.load(open(RAW + r"\mars_landslides.json", encoding="utf-8"))
    recs = {int(x["source_id"]): x for x in data}
    sf = shapefile.Reader(SHP)

    bytile = collections.defaultdict(collections.Counter)
    total_fl = 0
    for sr in sf.iterShapeRecords():
        sid = int(sr.record["序号"])
        if sid not in recs:
            continue
        t = recs[sid]["type"]
        if t not in ("FL", "SL", "SP", "CL"):
            continue
        for tl in tiles(*sr.shape.bbox):
            bytile[tl][t] += 1
        if t == "FL":
            total_fl += 1

    dirs = os.listdir(RAW + r"\ctx_tiles")

    print("FL 总数(bbox 覆盖口径) =", total_fl)
    print("含 FL 图幅数 =", sum(1 for tl in bytile if bytile[tl]["FL"] > 0))
    print("=== FL >= 3 的图幅及各类 bbox 覆盖 ===")
    cands = [(tl, by) for tl, by in bytile.items() if by["FL"] >= 3]
    cands.sort(key=lambda r: -r[1]["FL"])
    for tl, by in cands:
        mark = "已下载" if any(tl in d for d in dirs) else "待下载"
        print(f'{tl} [{mark}] FL={by["FL"]} SL={by["SL"]} SP={by["SP"]} CL={by["CL"]} 总={sum(by.values())}')


if __name__ == "__main__":
    main()