"""Export the *checked* handoff catalog for the API database, with a manifest.

``scripts/export_catalog.py`` owns the mobile catalog contract (field checks,
name backfill, explicit planetocentric confirmation). This script adds the week-2
gate on top of it:

1. run ``admission_report`` over the read-only handoff package;
2. refuse to export when any admission check fails;
3. refuse ``ready`` output unless ``--coordinate-evidence`` records who approved
   the latitude convention;
4. write ``manifest.json`` with SHA-256 of inputs and outputs, counts, the
   coordinate evidence and the admission summary.

Modes
-----
``staging``  raw inventory list for ``api.sqlite_store`` (``raw=True``), stays
             ``release_state = staging`` with latitude ``unverified``.
``ready``    ``api.schemas.Catalog`` payload, validated with pydantic.

Usage
-----
    python scripts/export_checked_catalog.py --handoff PATH --out DIR --mode staging
    python scripts/export_checked_catalog.py --handoff PATH --out DIR --mode ready \
        --coordinate-evidence "2026-10-31 老师确认纬度按行星中心纬度使用" --sqlite path.db
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

SOURCE_TITLE = "老师提供的火星滑坡编目"
DEFAULT_VERSION = "mars-landslides-20261031-v1"


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
) -> dict[str, Any]:
    records = json.loads((handoff / "01_catalog" / "mars_landslides.json").read_text(encoding="utf-8-sig"))
    images = json.loads((handoff / "03_images" / "image_index.json").read_text(encoding="utf-8-sig"))

    admission = admission_report.run(handoff, check_sha=check_sha, coordinate_evidence=coordinate_evidence)
    if admission["verdict"] == "blocked":
        raise SystemExit("准入检查存在阻断项，导出中止：" + "、".join(admission["blockers"]))
    if mode == "ready" and not (coordinate_evidence or "").strip():
        raise SystemExit(
            "ready 导出需要 --coordinate-evidence：写明纬度约定由谁、何时核准。"
            f"当前结论为 {admission['verdict']}。"
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
            "latitude": "unverified",
        },
        "coordinateEvidence": coordinate_evidence,
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
            "ready 模式只在提供坐标核准依据时可用；否则仅输出 staging。",
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
    )
    print(
        f"导出完成：mode={manifest['mode']} records={manifest['records']} "
        f"byType={manifest['byType']} verdict={manifest['admission']['verdict']}"
    )
    if manifest["sqlite"]:
        print(f"SQLite：{manifest['sqlite']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
