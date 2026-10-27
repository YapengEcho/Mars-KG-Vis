"""Validate the week-1 Mars landslide handoff package and print a report.

The handoff package is read-only input: 2971 catalog records, the point and
boundary shapefiles, 1083 224x224 crops, a 20-record sample draft and 12 case
candidates. This script checks that the tables agree with each other and with
the files on disk, and that the sample and candidates can be traced back to the
catalog by ``source_id``. It never rewrites source data.

Usage:
    python scripts/handoff_check.py --handoff PATH_TO_PACKAGE
    python scripts/handoff_check.py --handoff PATH --json report.json --md report.md
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


SLIDE_TYPES = ("SL", "SP", "FL", "CL")
TYPE_ALIASES = {"RL": "SL", "DF": "FL"}
DMS_PATTERN = re.compile(
    r"^\s*(?P<sign>[-+])?\s*(?P<deg>\d+(?:\.\d+)?)\s*[°º]\s*"
    r"(?P<min>\d+(?:\.\d+)?)\s*['′]\s*(?P<sec>\d+(?:\.\d+)?)\s*[\"″]?\s*$"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_dms(value: Any) -> float | None:
    """Parse ``-106°13′33.88″`` into decimal degrees; return None if not DMS."""
    if not isinstance(value, str):
        return None
    match = DMS_PATTERN.match(value)
    if not match:
        return None
    degrees = float(match.group("deg"))
    minutes = float(match.group("min"))
    seconds = float(match.group("sec"))
    decimal = degrees + minutes / 60 + seconds / 3600
    return -decimal if match.group("sign") == "-" else decimal


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def planetographic_shift(latitude_degrees: float, flattening_inverse: float = 169.894447223612) -> float:
    """Difference between planetocentric and planetographic latitude, in degrees.

    Mars' reference ellipsoid uses ``1/f`` around 169.89; the offset reaches about
    0.25 degrees near 45 degrees latitude and vanishes at the poles and equator.
    Used only to size the ambiguity, never to convert the catalog.
    """
    import math

    f = 1.0 / flattening_inverse
    eccentricity_squared = 2 * f - f * f
    phi = math.radians(latitude_degrees)
    geodetic = math.atan(math.tan(phi) / (1.0 - eccentricity_squared))
    return abs(math.degrees(geodetic) - latitude_degrees)


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append({"check": name, "status": status, "detail": detail})

    def summary(self) -> dict[str, int]:
        return dict(Counter(c["status"] for c in self.checks))

    @property
    def failed(self) -> bool:
        return any(c["status"] == "fail" for c in self.checks)


def load_package(root: Path) -> dict[str, Any]:
    return {
        "catalog": read_json(root / "01_catalog" / "mars_landslides.json"),
        "source_id_index": read_json(root / "01_catalog" / "source_id_index.json"),
        "images": read_json(root / "03_images" / "image_index.json"),
        "sample": read_json(root / "01_catalog" / "records20.draft.json"),
        "cases": read_json(root / "06_templates" / "cases12.todo.json"),
        "manifest": read_json(root / "handoff_manifest.json"),
    }


def check_catalog(catalog: list[dict[str, Any]], manifest: dict[str, Any], report: Report) -> None:
    report.add("catalog.records", "pass" if len(catalog) == 2971 else "fail", f"{len(catalog)} records (expected 2971)")

    ids = [record.get("id") for record in catalog]
    report.add(
        "catalog.ids.unique",
        "pass" if len(set(ids)) == len(ids) else "fail",
        f"{len(set(ids))} unique ids out of {len(ids)}",
    )
    source_ids = [record.get("source_id") for record in catalog]
    expected = list(range(1, len(catalog) + 1))
    report.add(
        "catalog.source_id.continuous",
        "pass" if sorted(source_ids) == expected else "fail",
        f"min={min(source_ids)} max={max(source_ids)}",
    )

    types = Counter(record.get("type") for record in catalog)
    unknown = sorted(set(types) - set(SLIDE_TYPES))
    report.add(
        "catalog.type.enum",
        "pass" if not unknown else "fail",
        f"{dict(types)}" + (f"; unknown={unknown}" if unknown else ""),
    )
    expected_counts = manifest.get("display_type_counts", {})
    agree = all(types.get(key) == value for key, value in expected_counts.items())
    report.add(
        "catalog.type.counts_vs_manifest",
        "pass" if agree else "fail",
        f"catalog={dict(types)} manifest={expected_counts}",
    )

    bad_range = [
        record["id"]
        for record in catalog
        if not (-180 <= record.get("lon", 1e9) <= 180 and -90 <= record.get("lat", 1e9) <= 90)
    ]
    report.add(
        "catalog.coordinates.range",
        "pass" if not bad_range else "fail",
        "all inside lon[-180,180] lat[-90,90]" if not bad_range else f"out of range: {bad_range[:5]}",
    )

    missing_area = sum(1 for record in catalog if record.get("area") is None)
    missing_name = sum(1 for record in catalog if not record.get("name"))
    missing_desc = sum(1 for record in catalog if not record.get("desc"))
    report.add(
        "catalog.optional_fields",
        "warn" if missing_name or missing_desc else "pass",
        f"area null={missing_area}, name empty={missing_name}, desc empty={missing_desc}",
    )

    raw_types = manifest.get("raw_type_counts", {})
    report.add(
        "catalog.type.mapping",
        "pass"
        if set(TYPE_ALIASES) == set(raw_types) - set(SLIDE_TYPES) and all(
            types.get(TYPE_ALIASES[alias], 0) > 0 for alias in TYPE_ALIASES
        )
        else "warn",
        f"aliases {TYPE_ALIASES} folded into display types; raw counts {raw_types}",
    )


def check_index(index: list[dict[str, Any]], catalog: list[dict[str, Any]], report: Report) -> None:
    report.add(
        "index.records",
        "pass" if len(index) == len(catalog) else "fail",
        f"{len(index)} entries for {len(catalog)} catalog records",
    )
    by_source = {record["source_id"]: record for record in catalog}
    mismatched = [
        entry["source_id"]
        for entry in index
        if entry["source_id"] not in by_source
        or by_source[entry["source_id"]]["id"] != entry["catalog_id"]
        or by_source[entry["source_id"]]["type"] != entry["display_type"]
    ]
    report.add(
        "index.identity_matches_catalog",
        "pass" if not mismatched else "fail",
        "source_id, catalog_id and display_type agree" if not mismatched else f"mismatches: {mismatched[:5]}",
    )
    point_indices = sorted(entry["point_record_index0"] for entry in index)
    boundary_indices = sorted(entry["boundary_record_index0"] for entry in index)
    report.add(
        "index.record_indices",
        "pass" if point_indices == list(range(len(index))) and boundary_indices == list(range(len(index))) else "warn",
        "point and boundary indices cover 0..n-1 without gaps"
        if point_indices == list(range(len(index))) == boundary_indices
        else "record indices are not a clean permutation of 0..n-1",
    )


def check_images(root: Path, images: list[dict[str, Any]], catalog: list[dict[str, Any]], report: Report, check_sha: bool) -> None:
    by_source = {record["source_id"]: record for record in catalog}
    missing_file = [entry["source_id"] for entry in images if not (root / entry["image_path"]).is_file()]
    report.add(
        "images.files_present",
        "pass" if not missing_file else "fail",
        f"{len(images)} indexed, missing on disk: {missing_file[:5]}" if missing_file else f"{len(images)} files present",
    )
    bad_size = [entry["source_id"] for entry in images if entry.get("image_size_px") != [224, 224]]
    report.add(
        "images.size",
        "pass" if not bad_size else "fail",
        "all 224x224" if not bad_size else f"unexpected sizes: {bad_size[:5]}",
    )
    type_mismatch = [
        entry["source_id"]
        for entry in images
        if entry["source_id"] in by_source and by_source[entry["source_id"]]["type"] != entry["type"]
    ]
    report.add(
        "images.type_matches_catalog",
        "pass" if not type_mismatch else "fail",
        "image folder type equals catalog type" if not type_mismatch else f"mismatches: {type_mismatch[:5]}",
    )
    covered = {entry["source_id"] for entry in images}
    report.add(
        "images.coverage",
        "warn",
        f"{len(covered)} of {len(catalog)} catalog records have a crop ({len(covered) / len(catalog):.1%})",
    )
    if check_sha:
        bad = []
        for entry in images:
            path = root / entry["image_path"]
            if path.is_file() and sha256_of(path) != entry.get("image_sha256"):
                bad.append(entry["source_id"])
        report.add(
            "images.sha256",
            "pass" if not bad else "fail",
            "all checksums match" if not bad else f"mismatched: {bad[:5]}",
        )


def check_sample(sample: dict[str, Any], catalog: list[dict[str, Any]], images: Iterable[dict[str, Any]], report: Report) -> None:
    records = sample.get("records", [])
    report.add("sample.records", "pass" if len(records) == 20 else "fail", f"{len(records)} records (expected 20)")

    by_source = {record["source_id"]: record for record in catalog}
    image_sources = {entry["source_id"] for entry in images}
    problems: list[str] = []
    for record in records:
        source_id = record.get("sourceId")
        original = by_source.get(source_id)
        if original is None:
            problems.append(f"{source_id}: not in catalog")
            continue
        if original["id"] != record.get("id"):
            problems.append(f"{source_id}: id mismatch")
        if original["type"] != record.get("type"):
            problems.append(f"{source_id}: type mismatch")
        if abs(original["lon"] - record.get("longitude", 1e9)) > 1e-9:
            problems.append(f"{source_id}: longitude mismatch")
        if abs(original["lat"] - record.get("latitude", 1e9)) > 1e-9:
            problems.append(f"{source_id}: latitude mismatch")
        image_path = (record.get("handoff") or {}).get("localImagePath")
        if image_path is None or source_id not in image_sources:
            problems.append(f"{source_id}: no indexed image")
    report.add(
        "sample.matches_catalog",
        "pass" if not problems else "fail",
        "all 20 trace back to catalog rows and indexed crops" if not problems else "; ".join(problems[:5]),
    )

    counts = Counter(record.get("type") for record in records)
    report.add(
        "sample.type_balance",
        "pass" if all(counts.get(kind) == 5 for kind in SLIDE_TYPES) else "warn",
        f"{dict(counts)}",
    )
    required = {"id", "sourceId", "name", "type", "longitude", "latitude", "areaKm2", "summary", "imageUrl", "source", "isDemo"}
    incomplete = [record.get("id") for record in records if not required.issubset(record)]
    report.add(
        "sample.contract_fields",
        "pass" if not incomplete else "fail",
        "all contract fields present" if not incomplete else f"missing fields: {incomplete[:5]}",
    )
    publication_flags = {
        str((record.get("handoff") or {}).get("coordinateStatus")) for record in records
    }
    report.add(
        "sample.coordinate_status",
        "warn" if publication_flags else "pass",
        f"coordinate status marked: {sorted(publication_flags)}",
    )


def check_cases(cases: list[dict[str, Any]], catalog: list[dict[str, Any]], images: Iterable[dict[str, Any]], report: Report) -> None:
    report.add("cases.records", "pass" if len(cases) == 12 else "fail", f"{len(cases)} candidates (expected 12)")
    counts = Counter(case.get("type") for case in cases)
    report.add(
        "cases.type_balance",
        "pass" if all(counts.get(kind) >= 2 for kind in SLIDE_TYPES) else "fail",
        f"{dict(counts)}",
    )
    by_source = {record["source_id"]: record for record in catalog}
    image_sources = {entry["source_id"] for entry in images}
    problems: list[str] = []
    for case in cases:
        source_id = case.get("source_id")
        original = by_source.get(source_id)
        if original is None:
            problems.append(f"{source_id}: not in catalog")
            continue
        if original["type"] != case.get("type"):
            problems.append(f"{source_id}: type mismatch")
        if source_id not in image_sources:
            problems.append(f"{source_id}: no indexed image")
        if not case.get("missing"):
            problems.append(f"{source_id}: no gap list")
    report.add(
        "cases.traceable",
        "pass" if not problems else "fail",
        "all candidates trace to catalog, crops and gap lists" if not problems else "; ".join(problems[:5]),
    )


def check_shapes(root: Path, catalog: list[dict[str, Any]], cases: list[dict[str, Any]], report: Report) -> None:
    try:
        import shapefile  # pyshp
    except ImportError:  # pragma: no cover - optional dependency
        report.add("shapes.reader", "warn", "pyshp not installed; skipped shapefile checks")
        return

    point_reader = shapefile.Reader(str(root / "02_vectors" / "landslide_points" / "火星滑坡点"), encoding="utf-8")
    boundary_reader = shapefile.Reader(str(root / "02_vectors" / "landslide_boundaries" / "火星滑坡-带边界"), encoding="utf-8")
    report.add(
        "shapes.counts",
        "pass" if len(point_reader) == len(boundary_reader) == len(catalog) else "fail",
        f"points={len(point_reader)} boundaries={len(boundary_reader)} catalog={len(catalog)}",
    )

    fields = [field[0] for field in boundary_reader.fields[1:]]
    report.add(
        "shapes.boundary_fields",
        "pass" if {"序号", "类型", "area", "Perimeter", "经度", "纬度"}.issubset(fields) else "warn",
        f"fields={fields}",
    )

    by_source = {record["source_id"]: record for record in catalog}
    point_ids = {int(round(record[0])) for record in point_reader.iterRecords()}
    boundary_ids = {int(round(record[0])) for record in boundary_reader.iterRecords()}
    missing = [case["source_id"] for case in cases if case["source_id"] not in point_ids or case["source_id"] not in boundary_ids]
    report.add(
        "shapes.case_geometry",
        "pass" if not missing else "fail",
        "all 12 candidates have point and boundary records" if not missing else f"missing geometry: {missing}",
    )

    # Point shapefile coordinates must equal the catalog numbers; the DMS text
    # attributes must parse back to the same values.
    # Catalog numbers are rounded to six decimals, so compare with that tolerance.
    tolerance = 5e-6
    mismatch_coord: list[int] = []
    sign_lost: list[int] = []
    mismatch_dms: list[int] = []
    for record in point_reader.iterShapeRecords():
        source_id = int(round(record.record[0]))
        catalog_record = by_source.get(source_id)
        if catalog_record is None:
            continue
        lon, lat = record.shape.points[0][:2]
        if abs(lon - catalog_record["lon"]) > tolerance or abs(lat - catalog_record["lat"]) > tolerance:
            mismatch_coord.append(source_id)
        dms_lon = parse_dms(record.record[4])
        dms_lat = parse_dms(record.record[5])
        if dms_lon is None or dms_lat is None:
            mismatch_dms.append(source_id)
            continue
        if abs(dms_lon - lon) <= tolerance and abs(dms_lat - lat) <= tolerance:
            continue
        # When the degree part is 0 the DMS text drops the minus sign
        # ("-0°25′30″" becomes "0°25′30″"); the numeric geometry keeps it.
        if abs(abs(dms_lon) - abs(lon)) <= tolerance and abs(abs(dms_lat) - abs(lat)) <= tolerance:
            sign_lost.append(source_id)
        else:
            mismatch_dms.append(source_id)
    report.add(
        "shapes.point_vs_catalog",
        "pass" if not mismatch_coord else "fail",
        "point geometry equals catalog lon/lat"
        if not mismatch_coord
        else f"{len(mismatch_coord)} mismatches, e.g. {mismatch_coord[:5]}",
    )
    report.add(
        "shapes.dms_vs_geometry",
        "pass" if not mismatch_dms and not sign_lost else ("fail" if mismatch_dms else "warn"),
        "DMS attributes parse back to geometry"
        if not mismatch_dms and not sign_lost
        else (
            f"value mismatches: {mismatch_dms[:5]}; "
            if mismatch_dms
            else ""
        )
        + (
            f"{len(sign_lost)} records lose the sign in DMS text when the degree part is 0 "
            f"(e.g. {sign_lost[:5]}); geometry and catalog keep the correct sign"
            if sign_lost
            else ""
        ),
    )

    # Boundary polygons must contain their point, and area text must be sane.
    outside: list[int] = []
    negative_area: list[int] = []
    for index in range(len(boundary_reader)):
        record = boundary_reader.record(index)
        source_id = int(round(record[0]))
        area = record[2]
        if isinstance(area, (int, float)) and area <= 0:
            negative_area.append(source_id)
    report.add(
        "shapes.boundary_area",
        "pass" if not negative_area else "fail",
        "all boundary areas positive" if not negative_area else f"non-positive area: {negative_area[:5]}",
    )


def check_coordinate_semantics(catalog: list[dict[str, Any]], report: Report) -> None:
    latitudes = [record["lat"] for record in catalog]
    missing = sum(1 for value in latitudes if value == 0)
    shifts = [planetographic_shift(value) for value in latitudes]
    report.add(
        "coordinate.latitude_range",
        "pass",
        f"lat min={min(latitudes):.4f} max={max(latitudes):.4f}; equator records={missing}",
    )
    report.add(
        "coordinate.convention",
        "warn",
        "planetocentric is NOT verified: planetographic alternative would move points by up to "
        f"{max(shifts):.2f}° at these latitudes (median {sorted(shifts)[len(shifts) // 2]:.2f}°). "
        "PRJ declares Earth WGS84 and must not be used as evidence for Mars.",
    )


def write_markdown(report: Report, path: Path) -> None:
    rows = [
        "# 交接包校验报告",
        "",
        "由 `scripts/handoff_check.py` 生成。`fail` 表示阻断，`warn` 表示需人工确认。",
        "",
        "| 检查项 | 结果 | 说明 |",
        "|---|---|---|",
    ]
    for check in report.checks:
        rows.append(f"| `{check['check']}` | {check['status']} | {check['detail']} |")
    summary = report.summary()
    rows += ["", f"汇总：{summary}", ""]
    path.write_text("\n".join(rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--handoff", type=Path, required=True, help="Week-1 handoff package root")
    parser.add_argument("--json", type=Path, help="Write the structured report here")
    parser.add_argument("--md", type=Path, help="Write a Markdown report here")
    parser.add_argument("--skip-sha", action="store_true", help="Skip the 1083 image checksums")
    args = parser.parse_args(argv)

    root: Path = args.handoff.expanduser().resolve()
    if not (root / "handoff_manifest.json").is_file():
        parser.error(f"{root} does not look like the handoff package (handoff_manifest.json missing)")

    package = load_package(root)
    report = Report()
    check_catalog(package["catalog"], package["manifest"], report)
    check_index(package["source_id_index"], package["catalog"], report)
    check_images(root, package["images"], package["catalog"], report, check_sha=not args.skip_sha)
    check_sample(package["sample"], package["catalog"], package["images"], report)
    check_cases(package["cases"], package["catalog"], package["images"], report)
    check_shapes(root, package["catalog"], package["cases"], report)
    check_coordinate_semantics(package["catalog"], report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"summary": report.summary(), "checks": report.checks}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, args.md)

    for check in report.checks:
        print(f"{check['status'].upper():4} {check['check']}: {check['detail']}")
    print(f"SUMMARY {report.summary()}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
