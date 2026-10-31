"""Build the first batch of case resource mappings (week 2).

Maps each suggested case (``06_templates/cases12.todo.json``) to every resource
the app needs:

* catalog record (``landslides.id``)
* preview crop in ``train224`` plus its SHA-256
* point and boundary geometry (shapefile record index, raw type, area check)
* citation document and the paper locator for the class
* a higher-resolution image request (window in degrees, CTX / HiRISE pixel size)

Outputs ``cases12.mapping.json``, ``cases12.highres.csv`` (UTF-8 with BOM) and
``cases12.table.md``. With ``--sqlite`` the cases, geometry links, documents and
case-document rows are inserted into an existing database built by
``scripts/export_checked_catalog.py``.

Nothing is inferred: titles stay provisional, states stay ``draft`` and the
case descriptions are left empty for week 3.

Usage
-----
    python scripts/build_case_mapping.py --handoff PATH --out DIR
    python scripts/build_case_mapping.py --handoff PATH --out DIR --sqlite path.db
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

MARS_RADIUS_M = 3396190.0
DEG_PER_M = 180.0 / (math.pi * MARS_RADIUS_M)
PAPER_URL = "https://doi.org/10.13203/j.whugis20250300"
PAPER_TITLE = "火星全球滑坡编目及分类初探"
DOCUMENT_ID = "doc-inventory-paper"
SECTION = {
    "SL": ("PDF 第 8 页 3.2.1 节；表 2（第 8—9 页）", "图 4（第 10 页）"),
    "SP": ("PDF 第 10 页 3.2.2 节；表 2（第 8—9 页）", "图 5（第 11 页）"),
    "FL": ("PDF 第 11—12 页 3.2.3 节；表 2（第 8—9 页）", "图 6（第 12 页）"),
    "CL": ("PDF 第 12—13 页 3.2.4 节；表 2（第 8—9 页）", "图 3（第 9 页）、图 7"),
}
AREA_TOLERANCE = 0.005


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def geo_bounds(native_window: list[float]) -> dict[str, float]:
    """Convert a metre window in the crop index to degrees.

    ``image_index.json`` stores the window in an equirectangular plane where
    ``x = lon·R`` but ``y`` grows southwards (screen coordinates), so
    ``lat = −y·DEG_PER_M``. Verified on all 1083 crops: median |Δlat| 0.002°
    versus 15.2° if the sign is dropped, and median |Δlon| 0.002°.
    """
    x0, y0, x1, y1 = native_window
    return {
        "lonMin": round(x0 * DEG_PER_M, 6),
        "lonMax": round(x1 * DEG_PER_M, 6),
        "latMin": round(-y1 * DEG_PER_M, 6),
        "latMax": round(-y0 * DEG_PER_M, 6),
    }


def load_ctx_tiles(directory: Path) -> dict[str, tuple[float, float, float, float]]:
    """Return ``{quad name: bbox}`` for the 4°x4° CTX mosaic quads.

    The grid ships as a single shapefile whose file name is mojibake (GBK bytes
    decoded as cp437), so the quad names must be read from the ``Name`` field
    instead of the file stem.
    """
    import shapefile

    path = next(directory.glob("*.shp"))
    reader = shapefile.Reader(str(path), encoding="utf-8")
    return {
        record["Name"]: tuple(shape.bbox)
        for record, shape in zip(reader.records(), reader.shapes())
    }


def build_mapping(handoff: Path, cases_file: Path) -> dict[str, Any]:
    import shapefile

    catalog = {int(record["source_id"]): record for record in read_json(handoff / "01_catalog" / "mars_landslides.json")}
    index = {int(entry["source_id"]): entry for entry in read_json(handoff / "01_catalog" / "source_id_index.json")}
    images = {int(entry["source_id"]): entry for entry in read_json(handoff / "03_images" / "image_index.json")}
    suggestions = read_json(cases_file)

    tiles = load_ctx_tiles(handoff / "02_vectors" / "ctx_grid")
    points = shapefile.Reader(str(handoff / "02_vectors" / "landslide_points" / "火星滑坡点"), encoding="utf-8")
    boundaries = shapefile.Reader(str(handoff / "02_vectors" / "landslide_boundaries" / "火星滑坡-带边界"), encoding="utf-8")

    cases: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for suggestion in suggestions:
        source_id = int(suggestion["source_id"])
        record = catalog.get(source_id)
        entry = images.get(source_id)
        problem: list[str] = []
        if record is None:
            problem.append("目录中无此编号")
        if entry is None:
            problem.append("无预览裁片记录")
        if problem:
            issues.append({"source_id": source_id, "problems": problem})
            continue

        image_path = handoff / entry["image_path"]
        if not image_path.is_file():
            problem.append("裁片文件缺失")

        window = entry["native_window_world_m"]
        window_m = window[2] - window[0]
        meters_per_pixel = window_m / entry["image_size_px"][0]
        bounds = geo_bounds(window)
        center_lon = (bounds["lonMin"] + bounds["lonMax"]) / 2
        center_lat = (bounds["latMin"] + bounds["latMax"]) / 2

        tile = (entry.get("source_tiles") or [""])[0]
        quad = tiles.get(tile.upper())
        if not tile:
            problem.append("裁片未记录 CTX 图幅")
        elif quad is None:
            problem.append(f"CTX 网格中无 {tile}")
        elif not (quad[0] <= center_lon <= quad[2] and quad[1] <= center_lat <= quad[3]):
            problem.append(f"中心点不在图幅 {tile} 内")

        boundary = boundaries.shape(index[source_id]["boundary_record_index0"])
        point = points.shape(index[source_id]["point_record_index0"])
        computed_area = spherical_area_km2(boundary.points)
        stored_area = record.get("area") or 0
        if stored_area and abs(computed_area - stored_area) / stored_area > AREA_TOLERANCE:
            problem.append("面积与边界不一致")

        section, figure = SECTION[record["type"]]

        case = {
            "id": f"case-ls-{source_id:04d}",
            "landslideId": record["id"],
            "sourceId": source_id,
            "title": f"{record.get('type_cn') or record['type']}案例 · {record['id']}",
            "description": "",
            "status": "draft",
            "imageAssetId": f"train224-{source_id}",
            "preview": {
                "path": entry["image_path"],
                "sha256": entry["image_sha256"],
                "sizePx": entry["image_size_px"],
                "windowM": round(window_m, 3),
                "metersPerPixel": round(meters_per_pixel, 3),
                "tiles": entry.get("source_tiles") or [],
            },
            "geometry": {
                "point": {
                    "assetId": f"geometry-point-{source_id:04d}",
                    "recordIndex0": index[source_id]["point_record_index0"],
                    "rawType": index[source_id]["point_type"],
                    "longitude": round(point.points[0][0], 7),
                    "latitude": round(point.points[0][1], 7),
                },
                "polygon": {
                    "assetId": f"geometry-polygon-{source_id:04d}",
                    "recordIndex0": index[source_id]["boundary_record_index0"],
                    "rawType": index[source_id]["raw_type"],
                    "areaKm2": round(computed_area, 3),
                    "vertexCount": len(boundary.points),
                },
            },
            "citation": [
                {
                    "documentId": DOCUMENT_ID,
                    "locator": f"{section}；{figure}",
                }
            ],
            "highResolution": {
                "lonMin": bounds["lonMin"],
                "lonMax": bounds["lonMax"],
                "latMin": bounds["latMin"],
                "latMax": bounds["latMax"],
                "windowM": round(window_m, 1),
                "ctxWidthPx": round(window_m / 6.0),
                "hiriseWidthPx": round(window_m / 0.3),
                "requestedDisplayPx": 1024,
                "status": "pending_request",
            },
            "needsExpertConfirmation": True,
            "notes": [
                "案例标签来自老师给出的编目，本包未做独立判别",
                "标题为占位，说明留待第 3 周撰写",
            ],
        }
        if problem:
            case["problems"] = problem
            issues.append({"source_id": source_id, "problems": problem})
        cases.append(case)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "catalog": str(handoff / "01_catalog" / "mars_landslides.json"),
            "casesTemplate": str(cases_file),
            "paper": {"title": PAPER_TITLE, "url": PAPER_URL},
        },
        "documents": [
            {
                "id": DOCUMENT_ID,
                "title": PAPER_TITLE,
                "url": PAPER_URL,
                "localPath": "04_references/火星全球滑坡编目及分类初探.pdf",
                "citation": "火星全球滑坡编目及分类初探. 武汉大学学报（信息科学版）, 2025. doi:10.13203/j.whugis20250300（作者信息待补）",
            }
        ],
        "byType": {
            kind: sum(1 for case in cases if catalog[case["sourceId"]]["type"] == kind)
            for kind in ("SL", "SP", "FL", "CL")
        },
        "cases": cases,
        "issues": issues,
    }


def spherical_area_km2(points: list[list[float]]) -> float:
    total = 0.0
    for index in range(len(points) - 1):
        lon1, lat1 = math.radians(points[index][0]), math.radians(points[index][1])
        lon2, lat2 = math.radians(points[index + 1][0]), math.radians(points[index + 1][1])
        total += (lon2 - lon1) * (math.sin(lat1) + math.sin(lat2))
    return abs(total) / 2 * MARS_RADIUS_M * MARS_RADIUS_M / 1_000_000


def write_csv(mapping: dict[str, Any], path: Path) -> None:
    fields = [
        "case_id", "source_id", "landslide_id", "type", "title",
        "center_lon", "center_lat", "lon_min", "lon_max", "lat_min", "lat_max",
        "window_m", "preview_m_per_px", "ctx_width_px_6m", "hirise_width_px_0_3m",
        "requested_display_px", "tile", "request_status", "note",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in mapping["cases"]:
            high = case["highResolution"]
            writer.writerow(
                {
                    "case_id": case["id"],
                    "source_id": case["sourceId"],
                    "landslide_id": case["landslideId"],
                    "type": case["geometry"]["point"]["rawType"],
                    "title": case["title"],
                    "center_lon": round((high["lonMin"] + high["lonMax"]) / 2, 6),
                    "center_lat": round((high["latMin"] + high["latMax"]) / 2, 6),
                    "lon_min": high["lonMin"],
                    "lon_max": high["lonMax"],
                    "lat_min": high["latMin"],
                    "lat_max": high["latMax"],
                    "window_m": high["windowM"],
                    "preview_m_per_px": case["preview"]["metersPerPixel"],
                    "ctx_width_px_6m": high["ctxWidthPx"],
                    "hirise_width_px_0_3m": high["hiriseWidthPx"],
                    "requested_display_px": high["requestedDisplayPx"],
                    "tile": " ".join(case["preview"]["tiles"]),
                    "request_status": high["status"],
                    "note": "请按框内范围提供 1024 像素宽的展示图；CTX 6 m/px 或 HiRISE 0.3 m/px",
                }
            )


def write_table(mapping: dict[str, Any], path: Path) -> None:
    lines = [
        "| 案例 | 编号 | 类别 | 中心坐标 | 窗口 | 预览 m/px | CTX 宽/px | 图幅 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for case in mapping["cases"]:
        high = case["highResolution"]
        lines.append(
            f"| {case['id']} | {case['landslideId']} | {case['geometry']['point']['rawType']} | "
            f"{(high['lonMin'] + high['lonMax']) / 2:.4f}, {(high['latMin'] + high['latMax']) / 2:.4f} | "
            f"{high['windowM'] / 1000:.2f} km | {case['preview']['metersPerPixel']:.1f} | "
            f"{high['ctxWidthPx']} | {' '.join(case['preview']['tiles'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_into_sqlite(path: Path, mapping: dict[str, Any], handoff: Path) -> dict[str, int]:
    import sqlite3

    point_shp = "02_vectors/landslide_points/火星滑坡点.shp"
    boundary_shp = "02_vectors/landslide_boundaries/火星滑坡-带边界.shp"
    with sqlite3.connect(path) as db:
        for document in mapping["documents"]:
            db.execute(
                "INSERT OR REPLACE INTO documents(id,title,url,local_path,citation) VALUES (?,?,?,?,?)",
                (document["id"], document["title"], document["url"], document["localPath"], document["citation"]),
            )
        for case in mapping["cases"]:
            source_id = case["sourceId"]
            for kind, shp, geometry in (
                ("point", point_shp, case["geometry"]["point"]),
                ("polygon", boundary_shp, case["geometry"]["polygon"]),
            ):
                db.execute(
                    "INSERT OR REPLACE INTO assets(id,landslide_id,kind,path,mime_type,sha256,width,height,metadata_json)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        f"geometry-{kind}-{source_id:04d}",
                        case["landslideId"],
                        kind,
                        str(handoff / shp),
                        "application/x-shapefile",
                        None,
                        None,
                        None,
                        json.dumps({"recordIndex0": geometry["recordIndex0"], "rawType": geometry["rawType"]}, ensure_ascii=False),
                    ),
                )
                db.execute(
                    "INSERT OR REPLACE INTO geometry_links(landslide_id,kind,asset_id,record_index0,raw_type) VALUES (?,?,?,?,?)",
                    (case["landslideId"], kind, f"geometry-{kind}-{source_id:04d}", geometry["recordIndex0"], geometry["rawType"]),
                )
            db.execute(
                "INSERT OR REPLACE INTO cases(id,landslide_id,title,description,image_asset_id,status) VALUES (?,?,?,?,?,?)",
                (case["id"], case["landslideId"], case["title"], case["description"], case["imageAssetId"], case["status"]),
            )
            for citation in case["citation"]:
                db.execute(
                    "INSERT OR REPLACE INTO case_documents(case_id,document_id,locator) VALUES (?,?,?)",
                    (case["id"], citation["documentId"], citation["locator"]),
                )
        counts = {
            table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("cases", "documents", "case_documents", "geometry_links")
        }
    return counts


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=Path, help="案例建议 JSON；默认交接包 06_templates/cases12.todo.json")
    parser.add_argument("--sqlite", type=Path, help="把案例、几何链接、文献关系写入已存在的数据库")
    args = parser.parse_args(argv)

    handoff = args.handoff.expanduser().resolve()
    out = args.out.expanduser().resolve()
    cases_file = (args.cases or handoff / "06_templates" / "cases12.todo.json").expanduser().resolve()
    if not (handoff / "01_catalog" / "mars_landslides.json").is_file():
        parser.error(f"{handoff} 不是交接包目录")
    if not cases_file.is_file():
        parser.error(f"找不到案例清单 {cases_file}")

    mapping = build_mapping(handoff, cases_file)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases12.mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(mapping, out / "cases12.highres.csv")
    write_table(mapping, out / "cases12.table.md")

    if args.sqlite:
        database = args.sqlite.expanduser().resolve()
        if not database.is_file():
            parser.error(f"数据库不存在：{database}")
        counts = load_into_sqlite(database, mapping, handoff)
        print(f"写入数据库：{counts}")

    print(f"案例映射：{len(mapping['cases'])} 条，问题 {len(mapping['issues'])} 条")
    for issue in mapping["issues"]:
        print(f"  问题 {issue['source_id']}: {'；'.join(issue['problems'])}")
    return 1 if mapping["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
