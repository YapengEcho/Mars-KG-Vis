"""Compare the two latitude conventions against the handoff geometry.

Prints, for the same catalog, how many points fall outside

1. their boundary polygon (``02_vectors/landslide_boundaries``), and
2. their manual crop box (``manual_box_world_m`` in ``03_images/image_index.json``,
   recorded in the local CTX mosaic projection);

first with the raw values, then after a planetographic -> planetocentric
conversion. This is the evidence behind the default ``unverified`` /
no-conversion behaviour: converting pushes 736 points out of frames they
currently sit in.

    python scripts/compare_latitude_conventions.py --handoff PATH
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import admission_report  # noqa: E402
import mars_latitude  # noqa: E402


def polygon_counts(root: Path, catalog: list[dict]) -> dict[str, int]:
    import shapefile

    reader = shapefile.Reader(str(root / "02_vectors" / "landslide_boundaries" / "火星滑坡-带边界"), encoding="utf-8")
    polygons = {int(round(record[0])): shape for record, shape in zip(reader.records(), reader.shapes())}
    raw = converted = 0
    for record in catalog:
        shape = polygons[int(record["source_id"])]
        parts = list(shape.parts) + [len(shape.points)]
        if not admission_report.point_in_polygon(record["lon"], record["lat"], shape.points, parts):
            raw += 1
        if not admission_report.point_in_polygon(
            record["lon"], mars_latitude.planetographic_to_planetocentric(record["lat"]), shape.points, parts
        ):
            converted += 1
    return {"records": len(catalog), "outsideRaw": raw, "outsideConverted": converted}


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="同时写出 JSON")
    args = parser.parse_args(argv)

    root = args.handoff.expanduser().resolve()
    catalog = admission_report.read_json(root / "01_catalog" / "mars_landslides.json")
    if isinstance(catalog, dict):
        catalog = catalog.get("records", catalog)

    polygons = polygon_counts(root, catalog)
    boxes = admission_report.manual_box_evidence(root, catalog, mars_latitude.DEFAULT_CONVENTION)
    result = {"polygons": polygons, "manualBoxes": boxes}

    print(f"记录数：{polygons['records']}")
    print(
        f"边界多边形：原值越界 {polygons['outsideRaw']} 条，换算后 {polygons['outsideConverted']} 条"
        f"（多移出 {polygons['outsideConverted'] - polygons['outsideRaw']} 条）"
    )
    if boxes:
        print(
            f"人工取图框（共 {boxes['boxes']} 个）：原值越界 {boxes['outsideNow']} 条，"
            f"换算后 {boxes['outsideIfConverted']} 条（多移出 {boxes['movedOutByConversion']} 条）"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
