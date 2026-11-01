"""Export the *checked* handoff catalog for the API database, with a manifest.

``scripts/export_catalog.py`` owns the mobile catalog contract (field checks,
name backfill, explicit planetocentric confirmation). This script adds the week-2
gate on top of it:

1. run ``admission_report`` over the read-only handoff package;
2. refuse to export when any admission check fails;
3. refuse ``ready`` output unless ``--coordinate-evidence`` records who approved
   the latitude convention *and* ``--latitude-convention`` names a verified one;
4. write ``manifest.json`` with SHA-256 of inputs and outputs, counts, the
   coordinate evidence and the admission summary.

Modes
-----
``staging``  raw inventory list for ``api.sqlite_store`` (``raw=True``), stays
             ``release_state = staging`` with latitude ``unverified``.
``ready``    ``api.schemas.Catalog`` payload, validated with pydantic.

Latitude
--------
The convention is one parameter for admission, export and case mapping, so the
report and the written numbers cannot disagree. Default is ``unverified``: the
values are copied as they are and ``ready`` is refused. Passing
``--latitude-convention planetographic`` additionally converts with
``scripts/mars_latitude.planetographic_to_planetocentric`` and records the shift
in ``manifest.json``. See ``mars_latitude`` for why the default is *not* to
convert: doing so moves 736 catalog points out of their manual crop frames.

Usage
-----
    python scripts/export_checked_catalog.py --handoff PATH --out DIR --mode staging
    python scripts/export_checked_catalog.py --handoff PATH --out DIR --mode ready \
        --coordinate-evidence "2026-11-03 老师确认纬度按行星面使用" \
        --latitude-convention planetographic --sqlite path.db
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

import admission_report  # noqa: E402
from export_catalog import build_catalog  # noqa: E402
import mars_latitude  # noqa: E402

SOURCE_TITLE = "老师提供的火星滑坡编目"
DEFAULT_VERSION = "mars-landslides-20261101-v3"
DEFAULT_LATITUDE_CONVENTION = mars_latitude.DEFAULT_CONVENTION
LATITUDE_TARGET = "planetocentric"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records_digest(records: list[dict[str, Any]]) -> str:
    import hashlib as _hashlib

    canonical = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def convert_latitudes(
    records: list[dict[str, Any]],
    convention: str = DEFAULT_LATITUDE_CONVENTION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the records to publish plus a latitude summary.

    ``planetographic`` runs the IAU-2000 conversion and reports how far each
    record moved; ``planetocentric`` and ``unverified`` keep the values as
    authored. Every record keeps its raw value in ``lat_source`` either way.
    """
    convention = mars_latitude.resolve_convention(convention)

    shifts: list[float] = []
    converted: list[dict[str, Any]] = []
    for record in records:
        latitude = float(record["lat"])
        target = mars_latitude.to_export_latitude(latitude, convention)
        shifts.append(abs(target - latitude))
        item = dict(record)
        item["lat"] = round(target, 9) if convention == "planetographic" else target
        item["lat_source"] = latitude
        converted.append(item)

    applied = convention == "planetographic"
    summary = {
        "from": convention,
        "to": LATITUDE_TARGET,
        "applied": applied,
        "label": mars_latitude.export_label(convention),
        "records": len(converted),
        "recordsShifted": sum(1 for shift in shifts if shift > 1e-9),
        "maxShiftDeg": round(max(shifts), 6) if shifts else 0.0,
        "maxShiftKm": round((max(shifts) if shifts else 0.0) * mars_latitude.KM_PER_DEGREE, 3),
        "meanShiftDeg": round(sum(shifts) / len(shifts), 6) if shifts else 0.0,
        "note": (
            "按 IAU 2000 椭球 (b/a)² 换算为行星心纬度"
            if applied
            else "未换算：纬度约定未核实（unverified）或源清单声明为行星心纬度，数值原样保留"
        ),
    }
    if applied:
        summary.update(
            {
                "ellipsoid": "IAU_2000_Mars",
                "axisRatio": round(mars_latitude.AXIS_RATIO, 9),
                "flattening": round(mars_latitude.FLATTENING, 9),
                "formula": "lat_c = atan((b/a)^2 * tan(lat_g))",
            }
        )
    return converted, summary


def build_staging_payload(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = []
    for record in records:
        payload.append(
            {
                "id": record["id"],
                "source_id": int(record["source_id"]),
                "name": record.get("name") or "",
                "type": record["type"],
                "lon": float(record["lon"]),
                "lat": float(record["lat"]),
                "area": record.get("area"),
                "perimeter": record.get("perimeter"),
                "desc": record.get("desc") or "",
            }
        )
    return payload


def validate_ready(payload: dict[str, Any]) -> None:
    try:
        from api.schemas import Catalog
    except ImportError as error:  # pragma: no cover - depends on the env
        print(f"提示：未安装 pydantic/fastapi（{error}），跳过 api.schemas 校验")
        return
    Catalog.model_validate(payload)


def build_sqlite(
    path: Path,
    payload: Any,
    handoff: Path,
    images: list[dict[str, Any]],
    *,
    raw: bool,
) -> dict[str, Any]:
    from api.sqlite_store import build_database

    source_index = json.loads((handoff / "01_catalog" / "source_id_index.json").read_text(encoding="utf-8-sig"))
    return build_database(
        path,
        payload,
        raw=raw,
        source_title=SOURCE_TITLE,
        source_index=source_index,
        image_index=images,
        asset_root=handoff,
    )


def export(
    handoff: Path,
    out: Path,
    *,
    mode: str,
    coordinate_evidence: str | None,
    version: str,
    sqlite_path: Path | None,
    check_sha: bool = True,
    latitude_convention: str = DEFAULT_LATITUDE_CONVENTION,
) -> dict[str, Any]:
    convention = mars_latitude.resolve_convention(latitude_convention)
    raw_records = json.loads((handoff / "01_catalog" / "mars_landslides.json").read_text(encoding="utf-8-sig"))
    records, conversion = convert_latitudes(raw_records, convention)
    images = json.loads((handoff / "03_images" / "image_index.json").read_text(encoding="utf-8-sig"))

    admission = admission_report.run(
        handoff,
        check_sha=check_sha,
        coordinate_evidence=coordinate_evidence,
        latitude_convention=convention,
    )
    if admission["verdict"] == "blocked":
        raise SystemExit("准入检查存在阻断项，导出中止：" + "、".join(admission["blockers"]))
    if mode == "ready":
        if not (coordinate_evidence or "").strip():
            raise SystemExit(
                "ready 导出需要 --coordinate-evidence：写明纬度约定由谁、何时核准。"
                f"当前结论为 {admission['verdict']}。"
            )
        if convention == "unverified":
            raise SystemExit(
                "ready 导出需要 --latitude-convention planetocentric 或 planetographic："
                "约定未核实时只能出 staging，避免把未核实的坐标标成已核准。"
            )

    if mode == "ready":
        payload: Any = build_catalog(
            records,
            source_title=SOURCE_TITLE,
            coordinate_system="planetocentric",
            version=version,
        )
        validate_ready(payload)
    else:
        payload = build_staging_payload(records)

    out.mkdir(parents=True, exist_ok=True)
    catalog_out = out / "catalog.json"
    catalog_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "admission.json").write_text(json.dumps(admission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    admission_report.write_markdown(admission, out / "admission.md")

    database = None
    if sqlite_path:
        database = build_sqlite(sqlite_path, payload, handoff, images, raw=(mode != "ready"))

    exported = payload["records"] if isinstance(payload, dict) else payload
    counts: dict[str, int] = {}
    for record in exported:
        counts[record["type"]] = counts.get(record["type"], 0) + 1

    manifest = {
        "version": version,
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "assembler": "scripts/export_checked_catalog.py",
        "handoff": str(handoff),
        "records": len(exported),
        "byType": counts,
        "withImage": len(images),
        "recordsSha256": records_digest(exported),
        "coordinateSystem": payload["coordinateSystem"] if isinstance(payload, dict) else {
            "body": "Mars",
            "longitude": "east-positive, -180..180",
            "latitude": mars_latitude.export_label(convention),
        },
        "coordinateEvidence": coordinate_evidence,
        "latitudeConversion": conversion,
        "admission": {
            "verdict": admission["verdict"],
            "summary": admission["summary"],
            "blockers": admission["blockers"],
        },
        "inputs": {
            name: {"sha256": sha256_of(handoff / name), "bytes": (handoff / name).stat().st_size}
            for name in (
                "01_catalog/mars_landslides.json",
                "01_catalog/source_id_index.json",
                "03_images/image_index.json",
            )
        },
        "outputs": {
            "catalog.json": {"sha256": sha256_of(catalog_out), "bytes": catalog_out.stat().st_size},
        },
        "sqlite": database,
        "notes": [
            "名称按“滑坡 + 编号”回填，说明沿用原始 desc（当前为空），案例文案另行撰写。",
            "imageUrl 留空；预览图通过 assets 表指向本地 train224 目录。",
            "ready 模式需要坐标依据 + 已核实的纬度约定；否则只输出 staging，坐标原样保留。",
            f"纬度约定：{convention}；{'已按 (b/a)^2 换算为行星心纬度，最大偏移 %.4f°（约 %.1f km）' % (conversion['maxShiftDeg'], conversion['maxShiftKm']) if conversion['applied'] else '未做换算，数值与源清单一致'}。",
            "不默认换算的原因：1083 个裁片的人工框在本地 CTX 配套投影（行星心）下生成，"
            "把编目按行星面→行星心换算会让 736 个原本在框内的点移出框外。",
            "staging 的 payload 与源清单同口径；api.sqlite_store 的 raw 模式把 metadata.latitude 记作 unverified，"
            "表示“未核准发布”，不是坐标系未知。",
        ],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("staging", "ready"), default="staging")
    parser.add_argument("--coordinate-evidence", help="核准纬度约定的依据；ready 模式必填")
    parser.add_argument(
        "--latitude-convention",
        choices=mars_latitude.CONVENTIONS,
        default=DEFAULT_LATITUDE_CONVENTION,
        help="源清单的纬度约定：unverified（默认，原样保留）/ planetocentric（原样保留）/ "
        "planetographic（按 IAU 2000 椭球换算为行星心纬度）",
    )
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--sqlite", type=Path, help="同时构建 SQLite；目标文件必须尚不存在")
    parser.add_argument("--skip-sha", action="store_true")
    args = parser.parse_args(argv)

    handoff = args.handoff.expanduser().resolve()
    if not (handoff / "01_catalog" / "mars_landslides.json").is_file():
        parser.error(f"{handoff} 不是交接包目录")

    manifest = export(
        handoff,
        args.out.expanduser().resolve(),
        mode=args.mode,
        coordinate_evidence=args.coordinate_evidence,
        version=args.version,
        sqlite_path=args.sqlite.expanduser().resolve() if args.sqlite else None,
        check_sha=not args.skip_sha,
        latitude_convention=args.latitude_convention,
    )
    print(
        f"导出完成：mode={manifest['mode']} records={manifest['records']} "
        f"byType={manifest['byType']} verdict={manifest['admission']['verdict']}"
    )
    conversion = manifest["latitudeConversion"]
    if conversion["applied"]:
        latitude_line = (
            f"纬度换算：{conversion['from']} -> {conversion['to']}，"
            f"移动 {conversion['recordsShifted']} 条，最大 {conversion['maxShiftDeg']:.4f}°"
            f"（约 {conversion['maxShiftKm']:.1f} km）"
        )
    else:
        latitude_line = f"纬度：{conversion['from']}，数值原样保留（未换算）"
    print(latitude_line)
    if manifest["sqlite"]:
        print(f"SQLite：{manifest['sqlite']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
