# -*- coding: utf-8 -*-
"""从成理工 GeoMars 系统公开前端资源下载并提取火星滑坡编目数据。

数据来源：https://cdut-mars.top/
成都理工大学地质灾害防治与地质环境保护全国重点实验室
「GeoMars 火星滑坡展示系统」。

关键事实：该系统的 2971 处火星滑坡数据不是通过后端接口下发，而是直接
打包在前端 JS 资源 /assets/MarsViewer-*.js 中（形如
`const Sa=[{id:1, ...}, ...]`）。本脚本下载该文件并把数组提取为 JSON / CSV，
作为项目「数据获取」环节的主数据，供后续清洗与入库使用。

仅用于《Python 程序设计》课程项目；数据版权属于原作者，报告需注明来源。
"""

import csv
import json
import re
import urllib.request
from pathlib import Path

BASE_URL = "https://cdut-mars.top"
BASE_DIR = Path(__file__).resolve().parents[2]  # data/scripts -> data -> 项目根
RAW_DIR = BASE_DIR / "data" / "raw"

# 类型代码 -> (英文全称, 中文)。对应论文《火星全球滑坡编目及分类初探》四分类。
# 代码与类型的对应关系以论文披露的各类型数量为准（Flowing=50、Composite=813、
# Sliding=1787、Slumping=312），而非缩写猜测。
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


def http_get(url: str) -> str:
    """用标准库请求一个文本资源（零第三方依赖）。"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (course project)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def fetch_viewer_js() -> str:
    """下载包含 2971 条滑坡数据的前端 JS 文本。

    前端是 Vue 单页应用，主 bundle 按 hash 命名，MarsViewer 又是主 bundle
    动态 import 的 chunk，文件名同样带 hash。因此分三步解析出真实文件名，
    避免硬编码 hash，保证脚本在对方版本升级后依然可重复运行。
    """
    # 1) 首页 -> 主 bundle 路径
    html = http_get(BASE_URL + "/")
    m = re.search(r'/assets/index-[^"]+\.js', html)
    if not m:
        raise RuntimeError("未在首页找到主 JS bundle")

    # 2) 主 bundle -> MarsViewer 动态 chunk 文件名
    main_js = http_get(BASE_URL + m.group())
    m = re.search(r'import\("\./(MarsViewer-[^"]+\.js)"\)', main_js)
    if not m:
        raise RuntimeError("未在主 bundle 找到 MarsViewer 动态 chunk")

    # 3) 下载 MarsViewer chunk
    return http_get(BASE_URL + "/assets/" + m.group(1))


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


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    js_text = fetch_viewer_js()

    records = extract_records(js_text)
    if not records:
        raise RuntimeError("未提取到任何滑坡记录，请检查正则与源页面结构")

    json_path = RAW_DIR / "mars_landslides.json"
    csv_path = RAW_DIR / "mars_landslides.csv"

    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fields = list(records[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    from collections import Counter
    by_type = Counter(r["type_cn"] for r in records)
    print(f"提取成功：共 {len(records)} 条滑坡")
    print("类型分布：" + "，".join(f"{k} {v} 处" for k, v in by_type.items()))
    print(f"已写入：\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()