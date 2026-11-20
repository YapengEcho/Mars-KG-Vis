# -*- coding: utf-8 -*-
"""从成理工 GeoMars 系统公开前端资源下载并提取火星滑坡编目数据。

数据来源：https://cdut-mars.top/
成都理工大学地质灾害防治与地质环境保护全国重点实验室
「GeoMars 火星滑坡展示系统」。

从首页查找火星地图脚本，再读取地图引用的 project-catalog-*.js。
旧版把编目直接打包在地图脚本中。两种格式均可导出 JSON 和 CSV。
默认将网站原始数据写入 data/raw/geomars_download，供后续清洗与入库。

仅用于《Python 程序设计》课程项目；数据版权属于原作者，报告需注明来源。
"""

import argparse
import csv
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

BASE_URL = "https://cdut-mars.top"
BASE_DIR = Path(__file__).resolve().parents[2]  # data/scripts -> data -> 项目根
RAW_DIR = BASE_DIR / "data" / "raw"

# 类型代码 -> (英文全称, 中文)。对应论文《火星全球滑坡编目及分类初探》四分类。
# 保留编目的原始代码，并附上论文使用的中英文类别名称。
TYPE_MAP = {
    "SL": ("Sliding", "滑移型"),
    "SP": ("Slumping", "滑塌型"),
    "FL": ("Flowing", "流动型"),
    "CL": ("Composite", "复合型"),
}

# 数字（兼容 .7837 这种省略前导 0 的写法，如面积/周长）
_NUM = r'-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?'

# 匹配数组里的一条对象；字段顺序与前端打包格式完全一致。
OBJ_RE = re.compile(
    r'\{id:(\d+(?:[eE][+-]?\d+)?),name:"([^"]*)",type:"([A-Za-z]+)",'
    r'area:(' + _NUM + r'),Perimeter:(' + _NUM + r'),'
    r'lon_degree:"([^"]*)",lat_degree:"([^"]*)",'
    r'lon:(' + _NUM + r'),lat:(' + _NUM + r'),'
    r'desc:"([^"]*)"\}'
)


def http_get(url: str, *, label: str | None = None) -> str:
    """用标准库请求一个文本资源（零第三方依赖）。"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (course project)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read()
        if label is not None:
            print(f"{req.get_method()} {url}  |  HTTP {resp.status}  |  {len(body):,} B  |  {label}", flush=True)
        return body.decode("utf-8")


def _fetch_resource(path: str, label: str, verbose: bool) -> str:
    return http_get(BASE_URL + path, label=label if verbose else None)


def fetch_viewer_js(*, verbose: bool = False) -> str:
    """从 GeoMars 首页定位并读取火星地图脚本。"""
    html = _fetch_resource("/", "GeoMars 首页", verbose)
    main = re.search(r"/assets/index-[A-Za-z0-9_-]+\.js", html)
    if not main:
        raise RuntimeError("未在首页找到主 JS bundle")

    main_js = _fetch_resource(main.group(), "主脚本", verbose)
    viewer = re.search(r"""import\(["']\./(MarsViewer-[A-Za-z0-9_-]+\.js)["']\)""", main_js)
    if not viewer:
        raise RuntimeError("未在主 bundle 找到 MarsViewer 动态 chunk")

    return _fetch_resource("/assets/" + viewer.group(1), "火星地图脚本", verbose)


def extract_records(js_text: str) -> list[dict]:
    """从 JS 文本里逐条提取滑坡记录，并按项目约定补全字段与类型映射。"""
    records = []
    for g in OBJ_RE.finditer(js_text):
        raw_id, name, code, area, perimeter, lon_deg, lat_deg, lon, lat, desc = g.groups()
        type_en, type_cn = TYPE_MAP.get(code, (code, code))
        ordinal = int(float(raw_id))  # id 可能写成 1e3 这类科学计数，先转 float
        records.append({
            "id": f"ls_{ordinal:04d}",   # 项目约定的稳定主键
            "source_id": ordinal,         # 前端原始数字 id，保留以便追溯
            "name": name,
            "type": code,
            "type_en": type_en,
            "type_cn": type_cn,
            "area": float(area),
            "perimeter": float(perimeter),
            "lon": float(lon),
            "lat": float(lat),
            "lon_degree": lon_deg,
            "lat_degree": lat_deg,
            "desc": desc,
        })
    return records


def fetch_catalog_records(*, verbose: bool = False) -> list[dict]:
    viewer_js = fetch_viewer_js(verbose=verbose)
    catalog = re.search(r"""["']\./(project-catalog-[A-Za-z0-9_-]+\.js)["']""", viewer_js)
    js_text = (
        _fetch_resource("/assets/" + catalog.group(1), "滑坡编目文件", verbose)
        if catalog else viewer_js
    )

    records = extract_records(js_text)
    if not records:
        raise RuntimeError("未提取到任何滑坡记录，请检查正则与源页面结构")
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="从 GeoMars 网站采集滑坡编目并导出 JSON 和 CSV。")
    parser.add_argument("--source-id", type=int, help="只导出指定的网站原始编号，例如 493")
    parser.add_argument(
        "--output-dir", type=Path, default=RAW_DIR / "geomars_download",
        help="导出目录，默认 data/raw/geomars_download",
    )
    args = parser.parse_args(argv)

    try:
        records = fetch_catalog_records(verbose=True)
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"采集失败：{exc}\n")
    total = len(records)
    stem = "mars_landslides"
    if args.source_id is not None:
        records = [record for record in records if record["source_id"] == args.source_id]
        if len(records) != 1:
            parser.error(f"编号 {args.source_id} 应匹配一条记录，实际匹配 {len(records)} 条")
        stem = f"mars_landslide_{args.source_id:04d}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / (stem + ".json")
    csv_path = args.output_dir / (stem + ".csv")

    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fields = list(records[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    print(f"解析 {total} 条编目记录，导出 {len(records)} 条。")
    if args.source_id is None:
        by_type = Counter(record["type_cn"] for record in records)
        print("类型分布  " + "，".join(f"{kind} {count} 处" for kind, count in by_type.items()))
    print(f"CSV   {csv_path.resolve()}")
    print(f"JSON  {json_path.resolve()}")


if __name__ == "__main__":
    main()
