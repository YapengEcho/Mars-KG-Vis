"""Export an explicitly selected source JSON to the mobile catalog contract."""

from __future__ import annotations

import argparse
from datetime import date
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


SLIDE_TYPES = frozenset({"SL", "SP", "FL", "CL"})
COORDINATE_SYSTEM = {
    "body": "Mars",
    "longitude": "east-positive, -180..180",
    "latitude": "planetocentric, degrees",
}


def number(value: Any, field: str, row: int) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Record {row}: {field} must be a finite number")
    return value


def build_catalog(
    rows: Any,
    *,
    source_title: str,
    coordinate_system: str,
    version: str,
    limit: int | None = None,
) -> dict[str, Any]:
    """Preserve source IDs and class codes without inferring geology or coordinates."""
    if coordinate_system != "planetocentric":
        raise ValueError("Explicit planetocentric coordinates are required; no conversion is performed")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input must be a non-empty JSON array")
    if not source_title.strip() or not version.strip():
        raise ValueError("Source title and version must not be empty")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("Limit must be a positive integer")

    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    source_ids: set[int] = set()
    # Validate every source row before applying the requested preview limit.
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Record {index}: expected an object")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"Record {index}: id must be a non-empty string")
        if identifier in ids:
            raise ValueError(f"Record {index}: duplicate id {identifier!r}")
        ids.add(identifier)

        source_id = row.get("source_id")
        if source_id is not None:
            if isinstance(source_id, bool) or not isinstance(source_id, int):
                raise ValueError(f"Record {index}: source_id must be an integer or null")
            if source_id in source_ids:
                raise ValueError(f"Record {index}: duplicate source_id {source_id!r}")
            source_ids.add(source_id)
        kind = row.get("type")
        if not isinstance(kind, str) or kind not in SLIDE_TYPES:
            raise ValueError(f"Record {index}: unknown type; expected SL, SP, FL, or CL")
        longitude = number(row.get("lon"), "lon", index)
        latitude = number(row.get("lat"), "lat", index)
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            raise ValueError(f"Record {index}: coordinates are outside the declared range")

        # The source catalog's area field is in square kilometres; no image-area inference.
        area = row.get("area")
        if area is None or area == "":
            area = None
        else:
            area = number(area, "area", index)
            if area < 0:
                raise ValueError(f"Record {index}: area must be non-negative or null")
        name = row.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError(f"Record {index}: name must be a string or null")
        if not name or not name.strip():
            name = f"滑坡 {source_id if source_id is not None else identifier}"
        summary = row.get("desc")
        if summary is None:
            summary = ""
        if not isinstance(summary, str):
            raise ValueError(f"Record {index}: desc must be a string or null")

        if limit is None or len(records) < limit:
            records.append({
                "id": identifier,
                "sourceId": source_id,
                "name": name,
                "type": kind,
                "longitude": longitude,
                "latitude": latitude,
                "areaKm2": area,
                "summary": summary,
                "imageUrl": None,
                "source": {"title": source_title.strip(), "url": None},
                "isDemo": False,
            })

    return {
        "version": version.strip(),
        "mode": "catalog",
        "coordinateSystem": dict(COORDINATE_SYSTEM),
        "records": records,
    }


def export_catalog(
    input_path: Path,
    output_path: Path,
    *,
    source_title: str,
    coordinate_system: str,
    version: str,
    limit: int | None = None,
) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if input_path == output_path or (output_path.exists() and input_path.samefile(output_path)):
        raise ValueError("Output must not overwrite the input file")
    rows = json.loads(input_path.read_text(encoding="utf-8-sig"))
    catalog = build_catalog(
        rows,
        source_title=source_title,
        coordinate_system=coordinate_system,
        version=version,
        limit=limit,
    )
    payload = json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source JSON array; never selected automatically")
    parser.add_argument("--output", type=Path, required=True, help="Destination catalog JSON")
    parser.add_argument("--source-title", required=True, help="Verified source name for exported records")
    parser.add_argument("--coordinate-system", choices=["planetocentric"], required=True,
                        help="Explicitly confirm east-positive longitude and planetocentric latitude in degrees")
    parser.add_argument("--limit", type=int, help="Maximum exported records, after validating the entire input")
    parser.add_argument("--version", default=date.today().isoformat(), help="Catalog version; defaults to today's date")
    args = parser.parse_args(argv)
    try:
        catalog = export_catalog(
            args.input, args.output, source_title=args.source_title,
            coordinate_system=args.coordinate_system, version=args.version, limit=args.limit,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Exported {len(catalog['records'])} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
