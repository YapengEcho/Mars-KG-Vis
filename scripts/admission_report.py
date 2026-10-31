"""Admission checks for the real Mars landslide catalog.

Week 2 asks for a checked catalog: identifiers, types, coordinates and missing
values, with an exportable report. This script reads the read-only handoff
package, runs the checks below and writes a JSON + Markdown report. It never
rewrites source data and never embeds full records in the report.

Checks
------
identifiers  unique ids, ``ls_{source_id:04d}`` pattern, contiguous source_id
types        enum, counts against the inventory paper, alias mapping RL->SL /
             DF->FL and agreement between catalog, point and boundary layers
coordinates  range, duplicate points, DMS sign loss, point geometry equality,
             point-in-polygon, polygon area against the catalog area
missing      empty name/summary, null area/perimeter, image coverage
images       file presence, 224x224 size, SHA-256, crop scale statistics
geometry     2971 points / 2971 polygons and a complete record index

Verdict: ``blocked`` when any check fails, ``ready_for_import`` only when the
caller supplies coordinate evidence, ``conditional_pending_coordinate_verification``
otherwise.

Usage
-----
    python scripts/admission_report.py --handoff PATH --json report.json --md report.md
    python scripts/admission_report.py --handoff PATH --coordinate-evidence "老师确认..."
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from handoff_check import parse_dms, sha256_of  # noqa: E402

SLIDE_TYPES = ("SL", "SP", "FL", "CL")
TYPE_ALIASES = {"RL": "SL", "DF": "FL"}
PAPER_COUNTS = {"SL": 1787, "SP": 321, "FL": 50, "CL": 813}
MARS_RADIUS_M = 3396190.0
COORD_TOLERANCE_DEG = 5e-6
AREA_TOLERANCE = 0.005
CROP_RATIO_MIN, CROP_RATIO_MAX = 1.45, 1.55


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


class Admission:
    """Collect checks and decide whether the catalog may be imported."""

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, identifier: str, title: str, status: str, count: int, detail: str) -> None:
        self.checks.append(
            {"id": identifier, "title": title, "status": status, "count": count, "detail": detail}
        )

    def summary(self) -> dict[str, int]:
        return dict(Counter(check["status"] for check in self.checks))

    @property
    def blockers(self) -> list[dict[str, Any]]:
        return [check for check in self.checks if check["status"] == "fail"]

    def verdict(self, coordinate_evidence: str | None) -> str:
        if self.blockers:
            return "blocked"
        return "ready_for_import" if coordinate_evidence else "conditional_pending_coordinate_verification"


def check_identifiers(catalog: list[dict[str, Any]], report: Admission) -> None:
    ids = [record["id"] for record in catalog]
    report.add(
        "ids.unique", "编号唯一", "pass" if len(set(ids)) == len(ids) else "fail",
        len(ids) - len(set(ids)), f"{len(set(ids))} 个唯一编号 / {len(ids)} 条记录",
    )

    malformed = [
        record["id"]
        for record in catalog
        if record["id"] != f"ls_{int(record['source_id']):04d}"
    ]
    report.add(
        "ids.pattern", "编号与序号对应", "pass" if not malformed else "fail",
        len(malformed), "编号均为 ls_{source_id:04d}" if not malformed else f"不符合：{malformed[:5]}",
    )

    source_ids = sorted(int(record["source_id"]) for record in catalog)
    expected = list(range(1, len(catalog) + 1))
    report.add(
        "ids.contiguous", "原始序号连续", "pass" if source_ids == expected else "fail",
        0 if source_ids == expected else len(source_ids),
        f"1..{len(catalog)} 连续" if source_ids == expected else "存在缺号或重复",
    )


def check_types(
    catalog: list[dict[str, Any]],
    index: list[dict[str, Any]],
    report: Admission,
) -> None:
    counts = Counter(record["type"] for record in catalog)
    unknown = sorted(set(counts) - set(SLIDE_TYPES))
    report.add(
        "types.enum", "类别枚举", "pass" if not unknown else "fail",
        len(unknown), f"{dict(counts)}" if not unknown else f"未知类别：{unknown}",
    )

    mismatch = {kind: counts.get(kind, 0) - PAPER_COUNTS[kind] for kind in SLIDE_TYPES}
    agree = all(value == 0 for value in mismatch.values())
    report.add(
        "types.counts_vs_paper", "类别计数与论文一致", "pass" if agree else "fail",
        sum(1 for value in mismatch.values() if value),
        f"目录 {dict(counts)}；论文 {PAPER_COUNTS}" if agree else f"差异 {mismatch}",
    )

    raw_counts = Counter(entry["raw_type"] for entry in index)
    unexpected = sorted(set(raw_counts) - set(SLIDE_TYPES) - set(TYPE_ALIASES))
    alias_records = [entry for entry in index if entry["raw_type"] in TYPE_ALIASES]
    alias_ok = all(entry["display_type"] == TYPE_ALIASES[entry["raw_type"]] for entry in alias_records)
    report.add(
        "types.alias_mapping", "别名映射", "pass" if not unexpected and alias_ok else "fail",
        len(unexpected), f"映射 {TYPE_ALIASES}，命中 {len(alias_records)} 条；原始计数 {dict(raw_counts)}",
    )

    inconsistent = [
        entry["source_id"]
        for entry in index
        if entry["display_type"] != entry["point_type"]
    ]
    report.add(
        "types.point_agreement", "点位与目录类别一致", "pass" if not inconsistent else "fail",
        len(inconsistent),
        "2971 条点位类别与目录一致" if not inconsistent else f"不一致：{inconsistent[:5]}",
    )


def check_coordinates(
    root: Path,
    catalog: list[dict[str, Any]],
    index: list[dict[str, Any]],
    report: Admission,
) -> None:
    out_of_range = [
        record["id"]
        for record in catalog
        if not (-180 <= record["lon"] <= 180 and -90 <= record["lat"] <= 90)
    ]
    report.add(
        "coords.range", "经纬度范围", "pass" if not out_of_range else "fail",
        len(out_of_range),
        f"经度 {min(r['lon'] for r in catalog):.4f}..{max(r['lon'] for r in catalog):.4f}，"
        f"纬度 {min(r['lat'] for r in catalog):.4f}..{max(r['lat'] for r in catalog):.4f}"
        if not out_of_range
        else f"越界：{out_of_range[:5]}",
    )

    pairs = Counter((round(record["lon"], 6), round(record["lat"], 6)) for record in catalog)
    duplicates = {pair: count for pair, count in pairs.items() if count > 1}
    report.add(
        "coords.duplicates", "重复点位", "warn" if duplicates else "pass",
        sum(count - 1 for count in duplicates.values()),
        f"{len(duplicates)} 组重复坐标" if duplicates else "无重复坐标",
    )

    try:
        import shapefile  # pyshp
    except ImportError:
        report.add("coords.geometry", "点位/边界核对", "warn", 0, "未安装 pyshp，跳过几何核对")
        return

    points = shapefile.Reader(str(root / "02_vectors" / "landslide_points" / "火星滑坡点"), encoding="utf-8")
    boundaries = shapefile.Reader(str(root / "02_vectors" / "landslide_boundaries" / "火星滑坡-带边界"), encoding="utf-8")
    point_by_source = {int(round(rec.record[0])): rec for rec in points.iterShapeRecords()}
    boundary_by_source = {int(round(rec.record[0])): rec for rec in boundaries.iterShapeRecords()}

    sign_lost = 0
    value_bad: list[int] = []
    mismatch = 0
    outside: list[tuple[int, float]] = []
    area_bad: list[int] = []
    area_diffs: list[float] = []

    for record in catalog:
        source_id = int(record["source_id"])
        point = point_by_source[source_id]
        boundary = boundary_by_source[source_id]

        lon, lat = point.shape.points[0][:2]
        dms_lon, dms_lat = parse_dms(point.record[4]), parse_dms(point.record[5])
        if dms_lon is not None and dms_lat is not None:
            if abs(dms_lon - lon) > COORD_TOLERANCE_DEG or abs(dms_lat - lat) > COORD_TOLERANCE_DEG:
                if (
                    abs(abs(dms_lon) - abs(lon)) <= COORD_TOLERANCE_DEG
                    and abs(abs(dms_lat) - abs(lat)) <= COORD_TOLERANCE_DEG
                ):
                    sign_lost += 1
                else:
                    value_bad.append(source_id)

        if abs(lon - record["lon"]) > COORD_TOLERANCE_DEG or abs(lat - record["lat"]) > COORD_TOLERANCE_DEG:
            mismatch += 1

        parts = list(boundary.shape.parts) + [len(boundary.shape.points)]
        if not point_in_polygon(lon, lat, boundary.shape.points, parts):
            outside.append((source_id, distance_to_rings_deg(lon, lat, boundary.shape.points, parts)))

        stored = record.get("area")
        if stored and stored > 0:
            computed = polygon_area_km2(boundary.shape.points, parts)
            difference = abs(computed - stored) / stored
            area_diffs.append(difference)
            if difference > AREA_TOLERANCE:
                area_bad.append(source_id)

    report.add(
        "coords.dms", "度分秒文本与数值一致", "fail" if value_bad else ("warn" if sign_lost else "pass"),
        len(value_bad) or sign_lost,
        f"数值错误 {len(value_bad)} 条；度为 0 时丢负号 {sign_lost} 条（数值字段正确）"
        if value_bad or sign_lost
        else "2971 条一致",
    )
    report.add(
        "coords.point_vs_catalog", "点位几何与目录坐标一致", "pass" if mismatch == 0 else "fail",
        mismatch, "2971 条一致（容差 0.000005 度）" if mismatch == 0 else f"{mismatch} 条不一致",
    )
    outside_detail = ""
    if outside:
        shown = "，".join(f"{sid}（偏离 {dist * 111.32:.0f} km）" for sid, dist in outside[:5])
        outside_detail = f"越界 {len(outside)} 条：{shown}"
    outside_status = "pass" if not outside else ("warn" if len(outside) <= len(catalog) * 0.005 else "fail")
    report.add(
        "coords.point_in_polygon", "点位落在所属边界内", outside_status,
        len(outside), "2971 条全部落在对应多边形内" if not outside else outside_detail,
    )
    median_diff = sorted(area_diffs)[len(area_diffs) // 2] if area_diffs else 0.0
    report.add(
        "coords.area_check", "面积与多边形一致", "pass" if not area_bad else "warn",
        len(area_bad),
        f"{len(area_diffs)} 条比对，中位相对偏差 {median_diff:.6%}"
        + (f"；超过 {AREA_TOLERANCE:.1%} 的 {len(area_bad)} 条：{area_bad[:5]}" if area_bad else ""),
    )


def point_in_polygon(x: float, y: float, points: Iterable[tuple[float, float]], parts: list[int]) -> bool:
    """Ray casting over every ring; a point inside any ring counts as inside."""
    coords = list(points)
    for start, end in zip(parts, parts[1:]):
        ring = [(coords[i][0], coords[i][1]) for i in range(start, end)]
        inside = False
        for index in range(len(ring) - 1):
            x1, y1 = ring[index]
            x2, y2 = ring[index + 1]
            if (y1 > y) != (y2 > y):
                crossing = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < crossing:
                    inside = not inside
        if inside:
            return True
    return False


def polygon_area_km2(points: Iterable[tuple[float, float, float]], parts: list[int] | None = None) -> float:
    """Spherical polygon area with R = 3396190 m, the formula ArcGIS used here.

    Area = R^2/2 * |sum (lon2-lon1)(sin lat1 + sin lat2)| over every ring; holes
    and islands are summed with their vertex order, matching the stored values to
    about 1e-6 relative.
    """
    coords = list(points)
    rings = list(zip(parts, parts[1:])) if parts else [(0, len(coords))]
    total = 0.0
    for start, end in rings:
        for index in range(start, end - 1):
            lon1, lat1 = math.radians(coords[index][0]), math.radians(coords[index][1])
            lon2, lat2 = math.radians(coords[index + 1][0]), math.radians(coords[index + 1][1])
            total += (lon2 - lon1) * (math.sin(lat1) + math.sin(lat2))
    return abs(total) / 2 * MARS_RADIUS_M * MARS_RADIUS_M / 1_000_000


def distance_to_rings_deg(
    x: float,
    y: float,
    points: Iterable[tuple[float, float, float]],
    parts: list[int],
) -> float:
    """Shortest distance from a point to any ring edge, in degrees (lon scaled by cos lat)."""
    coords = list(points)
    scale = math.cos(math.radians(y)) or 1.0
    best = float("inf")
    for start, end in zip(parts, parts[1:]):
        for index in range(start, end - 1):
            x1, y1 = coords[index][:2]
            x2, y2 = coords[index + 1][:2]
            dx, dy = (x2 - x1) * scale, y2 - y1
            px, py = (x - x1) * scale, y - y1
            length_squared = dx * dx + dy * dy
            ratio = 0.0 if length_squared == 0 else max(0.0, min(1.0, (px * dx + py * dy) / length_squared))
            best = min(best, math.hypot(px - ratio * dx, py - ratio * dy))
    return best


def check_missing(
    catalog: list[dict[str, Any]],
    images: list[dict[str, Any]],
    report: Admission,
) -> None:
    empty_name = sum(1 for record in catalog if not str(record.get("name") or "").strip())
    empty_desc = sum(1 for record in catalog if not str(record.get("desc") or "").strip())
    null_area = sum(1 for record in catalog if record.get("area") is None)
    null_perimeter = sum(1 for record in catalog if record.get("perimeter") is None)
    report.add(
        "missing.text", "名称与说明缺失", "warn" if empty_name or empty_desc else "pass",
        empty_name + empty_desc,
        f"名称空 {empty_name}/{len(catalog)}，说明空 {empty_desc}/{len(catalog)}；按契约回填“滑坡 + 编号”",
    )
    report.add(
        "missing.numeric", "面积与周长缺失", "warn" if null_area or null_perimeter else "pass",
        null_area + null_perimeter,
        f"面积空 {null_area}，周长空 {null_perimeter}",
    )

    covered = {entry["source_id"] for entry in images}
    by_type = Counter(record["type"] for record in catalog)
    covered_type = Counter(record["type"] for record in catalog if record["source_id"] in covered)
    detail = "，".join(f"{kind} {covered_type[kind]}/{by_type[kind]}" for kind in SLIDE_TYPES)
    report.add(
        "missing.images", "图片覆盖", "warn" if covered else "fail",
        len(catalog) - len(covered),
        f"{len(covered)}/{len(catalog)} 有裁片（{len(covered) / len(catalog):.1%}）；{detail}",
    )


def check_images(
    root: Path,
    images: list[dict[str, Any]],
    report: Admission,
    check_sha: bool,
) -> None:
    missing = [entry["source_id"] for entry in images if not (root / entry["image_path"]).is_file()]
    report.add(
        "images.files", "裁片文件存在", "pass" if not missing else "fail",
        len(missing), f"{len(images)} 张全部存在" if not missing else f"缺失：{missing[:5]}",
    )
    if check_sha:
        bad = [
            entry["source_id"]
            for entry in images
            if (root / entry["image_path"]).is_file()
            and sha256_of(root / entry["image_path"]) != entry["image_sha256"]
        ]
        report.add(
            "images.hash", "裁片哈希", "pass" if not bad else "fail",
            len(bad), "全部匹配" if not bad else f"不匹配：{bad[:5]}",
        )

    windows = [(entry["native_window_world_m"][2] - entry["native_window_world_m"][0]) for entry in images]
    scales = [window / 224 for window in windows]
    ratios = []
    for entry in images:
        x0, y0, x1, y1 = entry["native_window_world_m"]
        bx0, by0, bx1, by1 = entry["manual_box_world_m"]
        long_edge = max(bx1 - bx0, by1 - by0)
        if long_edge > 0:
            ratios.append((x1 - x0) / long_edge)
    off_ratio = sum(1 for ratio in ratios if not (CROP_RATIO_MIN <= ratio <= CROP_RATIO_MAX))
    report.add(
        "images.scale", "裁片尺度随目标变化", "pass" if off_ratio == 0 else "warn",
        off_ratio,
        f"窗口 {min(windows) / 1000:.2f}–{max(windows) / 1000:.2f} km（中位 {sorted(windows)[len(windows) // 2] / 1000:.2f}），"
        f"{min(scales):.1f}–{max(scales):.1f} m/像素（中位 {sorted(scales)[len(scales) // 2]:.1f}）；"
        f"窗口/框 1.5× 的有 {len(ratios) - off_ratio}/{len(ratios)}",
    )


def check_geometry(root: Path, catalog: list[dict[str, Any]], index: list[dict[str, Any]], report: Admission) -> None:
    try:
        import shapefile
    except ImportError:
        report.add("geometry.reader", "几何文件", "warn", 0, "未安装 pyshp，跳过")
        return
    points = shapefile.Reader(str(root / "02_vectors" / "landslide_points" / "火星滑坡点"), encoding="utf-8")
    boundaries = shapefile.Reader(str(root / "02_vectors" / "landslide_boundaries" / "火星滑坡-带边界"), encoding="utf-8")
    report.add(
        "geometry.counts", "点位/边界条数", "pass" if len(points) == len(boundaries) == len(catalog) else "fail",
        0, f"点位 {len(points)}，边界 {len(boundaries)}，目录 {len(catalog)}",
    )
    point_indices = sorted(entry["point_record_index0"] for entry in index)
    boundary_indices = sorted(entry["boundary_record_index0"] for entry in index)
    complete = point_indices == list(range(len(index))) and boundary_indices == list(range(len(index)))
    report.add(
        "geometry.index", "记录下标完整", "pass" if complete else "fail",
        0, "点位与边界下标覆盖 0..n-1" if complete else "下标有缺口",
    )


def run(root: Path, *, check_sha: bool = True, coordinate_evidence: str | None = None) -> dict[str, Any]:
    catalog = read_json(root / "01_catalog" / "mars_landslides.json")
    index = read_json(root / "01_catalog" / "source_id_index.json")
    images = read_json(root / "03_images" / "image_index.json")

    report = Admission()
    check_identifiers(catalog, report)
    check_types(catalog, index, report)
    check_coordinates(root, catalog, index, report)
    check_missing(catalog, images, report)
    check_images(root, images, report, check_sha)
    check_geometry(root, catalog, index, report)

    return {
        "dataset": {
            "records": len(catalog),
            "with_image": len(images),
            "types": dict(Counter(record["type"] for record in catalog)),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "coordinate_evidence": coordinate_evidence,
        "summary": report.summary(),
        "verdict": report.verdict(coordinate_evidence),
        "blockers": [check["id"] for check in report.blockers],
        "checks": report.checks,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# 目录准入报告",
        "",
        f"生成时间：{report['generated_at']}　数据：{report['dataset']['records']} 条记录，"
        f"{report['dataset']['with_image']} 张裁片　类别：{report['dataset']['types']}",
        "",
        f"**结论：`{report['verdict']}`**　",
        f"通过 {report['summary'].get('pass', 0)} 项，警告 {report['summary'].get('warn', 0)} 项，"
        f"失败 {report['summary'].get('fail', 0)} 项。",
        "",
    ]
    if report["blockers"]:
        lines += ["> 阻断项：" + "、".join(f"`{item}`" for item in report["blockers"]), ""]
    lines += ["| 检查项 | 结果 | 数量 | 说明 |", "|---|---|---|---|"]
    for check in report["checks"]:
        lines.append(f"| {check['title']} (`{check['id']}`) | {check['status']} | {check['count']} | {check['detail']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--md", type=Path)
    parser.add_argument("--coordinate-evidence", help="核准纬度约定的依据；提供后结论才可能是 ready_for_import")
    parser.add_argument("--skip-sha", action="store_true")
    args = parser.parse_args(argv)

    root = args.handoff.expanduser().resolve()
    if not (root / "01_catalog" / "mars_landslides.json").is_file():
        parser.error(f"{root} 不是交接包目录")

    report = run(root, check_sha=not args.skip_sha, coordinate_evidence=args.coordinate_evidence)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, args.md)

    for check in report["checks"]:
        print(f"{check['status'].upper():4} {check['id']}: {check['detail']}")
    print(f"VERDICT {report['verdict']} {report['summary']}")
    return 1 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
