"""Build the week-1 delivery artifacts from the read-only handoff package.

Outputs, all deterministic from the handoff:

* ``records20.json`` - the 20-record real sample in the APP contract shape, with
  the handoff block that keeps the coordinate status visible;
* ``cases12.json`` - the 12 case candidates, each with the crop, source tiles,
  measured area and an explicit gap list;
* ``tables.md`` - the same two tables in Markdown for review.

The script never invents names, coordinates or citations. Descriptions are
composed from catalog facts (id, type, position, area, CTX tile) and point at
the inventory paper by page; whether a candidate is *typical* for its class
stays an open expert decision.

Usage:
    python scripts/build_week1_artifacts.py --handoff PATH --out PATH
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TYPE_CN = {"SL": "滑移型", "SP": "滑塌型", "FL": "流动型", "CL": "复合型"}
TYPE_EN = {"SL": "slide", "SP": "slump", "FL": "flow", "CL": "complex"}
PAPER = {
    "id": "inventory_paper",
    "title": "火星全球滑坡编目及分类初探",
    "doi": "https://doi.org/10.13203/j.whugis20250300",
    "pages": {"types": "PDF 第8—9页（表2）", "model": "PDF 第9页（图3）", "counts": "PDF 第7页（3.2节）"},
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ew(value: float) -> str:
    return f"{abs(value):.4f}°{'W' if value < 0 else 'E'}"


def ns(value: float) -> str:
    return f"{abs(value):.4f}°{'S' if value < 0 else 'N'}"


def build_records20(root: Path) -> dict[str, Any]:
    draft = read_json(root / "01_catalog" / "records20.draft.json")
    catalog = {record["source_id"]: record for record in read_json(root / "01_catalog" / "mars_landslides.json")}
    index = {entry["source_id"]: entry for entry in read_json(root / "03_images" / "image_index.json")}

    records = []
    for record in draft["records"]:
        source_id = record["sourceId"]
        original = catalog[source_id]
        image = index[source_id]
        image_path = root / image["image_path"]
        if not image_path.is_file():
            raise SystemExit(f"missing crop for source_id {source_id}: {image['image_path']}")
        if sha256_of(image_path) != image["image_sha256"]:
            raise SystemExit(f"checksum mismatch for source_id {source_id}")
        if abs(original["lon"] - record["longitude"]) > 5e-6 or abs(original["lat"] - record["latitude"]) > 5e-6:
            raise SystemExit(f"coordinate mismatch for source_id {source_id}")

        records.append(
            {
                "id": record["id"],
                "sourceId": source_id,
                "name": record["name"],
                "type": record["type"],
                "longitude": original["lon"],
                "latitude": original["lat"],
                "areaKm2": original["area"],
                "summary": "",
                "imageUrl": None,
                "source": {
                    "title": "张乐飞等《火星全球滑坡编目及分类初探》（老师提供的编目与矢量数据）",
                    "url": PAPER["doi"],
                },
                "isDemo": False,
                "handoff": {
                    "localImagePath": image["image_path"],
                    "rawType": image["raw_type"],
                    "sourceTiles": image["source_tiles"],
                    "imageSha256": image["image_sha256"],
                    "coordinateStatus": "latitude_convention_pending_verification",
                    "caseSelectionStatus": "not_a_publishable_catalog",
                },
            }
        )

    return {
        "status": "verified_traceable_but_not_publishable",
        "reason": "原始纬度约定尚未核准；图片仅为 224 像素预览，imageUrl 未接入",
        "target_contract": "contracts/catalog.md",
        "coordinate_verification_required": True,
        "counts": {kind: sum(1 for record in records if record["type"] == kind) for kind in TYPE_CN},
        "records": records,
    }


def build_cases12(root: Path) -> dict[str, Any]:
    cases = read_json(root / "06_templates" / "cases12.todo.json")
    catalog = {record["source_id"]: record for record in read_json(root / "01_catalog" / "mars_landslides.json")}
    index = {entry["source_id"]: entry for entry in read_json(root / "03_images" / "image_index.json")}

    selected = []
    for case in cases:
        source_id = case["source_id"]
        original = catalog[source_id]
        image = index[source_id]
        area = original["area"]
        tiles = ", ".join(image["source_tiles"])
        # The crop window is a square of 1.5x the manual box's long edge, so the
        # ground scale differs per target; always derive it from this image.
        x0, y0, x1, y1 = image["native_window_world_m"]
        window_m = x1 - x0
        meters_per_pixel = window_m / 224
        selected.append(
            {
                "source_id": source_id,
                "catalog_id": case["catalog_id"],
                "type": case["type"],
                "type_cn": TYPE_CN[case["type"]],
                "type_en": TYPE_EN[case["type"]],
                "title": f"滑坡 {case['catalog_id']}（{TYPE_CN[case['type']]}）",
                "description": (
                    f"编号 {case['catalog_id']}，{case['type']}（{TYPE_CN[case['type']]}）。"
                    f"中心位置 {ew(original['lon'])}，{ns(original['lat'])}；编目面积 {area:.2f} km²；"
                    f"预览裁片取自 CTX 图幅 {tiles}，窗口 {window_m / 1000:.1f} km，"
                    f"约 {meters_per_pixel:.1f} m/像素。"
                    f"类型定义与判别依据见《火星全球滑坡编目及分类初探》表2（PDF 第8—9页）与图3（第9页）；"
                    "本条是否作为该类的典型展示案例，仍待专家确认。"
                ),
                "longitude": original["lon"],
                "latitude": original["lat"],
                "areaKm2": area,
                "preview_path": image["image_path"],
                "preview_sha256": image["image_sha256"],
                "source_tiles": image["source_tiles"],
                "crop_window_m": round(window_m, 1),
                "meters_per_pixel": round(meters_per_pixel, 2),
                "point_record_index0": case["point_record_index0"],
                "boundary_record_index0": case["boundary_record_index0"],
                "geometry": {
                    "point_shp": "02_vectors/landslide_points/火星滑坡点.shp",
                    "boundary_shp": "02_vectors/landslide_boundaries/火星滑坡-带边界.shp",
                    "status": "available_as_shapefile_only",
                },
                "references": [PAPER],
                "gaps": {
                    "typicality": "候选编号未认证为该类典型案例，需滕佳杰/老师确认，可替换",
                    "image": (
                        f"仅有 224×224 预览裁片（窗口 {window_m / 1000:.1f} km，约 "
                        f"{meters_per_pixel:.1f} m/像素；窗口随目标大小变化），展示需按编号申请高清 CTX 图"
                    ),
                    "geometry": "边界为编目多边形（PolygonZ），需转 GeoJSON/WKT 并确认火星坐标语义后才能上球",
                    "text": "标题与说明目前只写数据事实，地质描述与出处待补",
                    "coordinate": "纬度约定（planetocentric/planetographic）待核准",
                },
                "status": "candidate_pending_expert_confirmation",
            }
        )

    return {
        "status": "candidates_not_final",
        "selection_basis": "每类 3 条，均来自 Valles Marineris 区域、有 224 裁片且有对应点位与边界记录；未按模型分数筛选，未重判类型",
        "paper": PAPER,
        "cases": selected,
    }


def render_tables(records20: dict[str, Any], cases12: dict[str, Any]) -> str:
    lines = [
        "# 第 1 周交付表（自动生成）",
        "",
        "由 `scripts/build_week1_artifacts.py` 生成；数值来自交接包，未做任何坐标或类型改写。",
        "",
        "## 20 条真实样例",
        "",
        "| # | id | 类型 | 经度 | 纬度 | 面积 km² | 裁片 |",
        "|---|---|---|---|---|---|---|",
    ]
    for position, record in enumerate(records20["records"], start=1):
        lines.append(
            f"| {position} | `{record['id']}` | {record['type']} | {ew(record['longitude'])} | "
            f"{ns(record['latitude'])} | {record['areaKm2']:.2f} | `{record['handoff']['localImagePath']}` |"
        )
    lines += [
        "",
        "## 12 个案例候选",
        "",
        "| # | id | 类型 | 经度 | 纬度 | 面积 km² | CTX 图幅 | 窗口 km | m/像素 | 状态 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for position, case in enumerate(cases12["cases"], start=1):
        lines.append(
            f"| {position} | `{case['catalog_id']}` | {case['type']} {case['type_cn']} | {ew(case['longitude'])} | "
            f"{ns(case['latitude'])} | {case['areaKm2']:.2f} | {', '.join(case['source_tiles'])} | "
            f"{case['crop_window_m'] / 1000:.1f} | {case['meters_per_pixel']:.1f} | 候选中 |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    root = args.handoff.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    records20 = build_records20(root)
    cases12 = build_cases12(root)
    (out / "records20.json").write_text(json.dumps(records20, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "cases12.json").write_text(json.dumps(cases12, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "tables.md").write_text(render_tables(records20, cases12), encoding="utf-8")
    print(f"wrote {len(records20['records'])} sample records and {len(cases12['cases'])} cases to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
