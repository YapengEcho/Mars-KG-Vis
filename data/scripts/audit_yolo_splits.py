# -*- coding: utf-8 -*-
"""Audit a generated YOLO dataset without modifying it.

The generated YOLO txt files intentionally omit source_id.  For the leakage
audit this module therefore reconstructs each patch's pixel footprint from its
filename and intersects it with the SHP AABBs used to build the dataset.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


CLASS_NAMES = ("SL", "SP", "FL", "CL")
PATCH_RE = re.compile(
    r"^(?P<tile>E-?\d{3}_N-?\d{2})_(?P<x>\d{5})_(?P<y>\d{5})$"
)
CENTER_RE = re.compile(
    r"^(?P<tile>E-?\d{3}_N-?\d{2})_big_(?P<sid>\d+)_(?P<x>\d{5})_(?P<y>\d{5})_(?P<size>\d+)$"
)


def parse_patch_name(stem: str) -> dict:
    """Parse a generated patch stem into tile, origin, size and source id."""
    match = CENTER_RE.fullmatch(stem)
    if match:
        return {
            "tile": match.group("tile"),
            "kind": "center",
            "x": int(match.group("x")),
            "y": int(match.group("y")),
            "size": int(match.group("size")),
            "source_id": int(match.group("sid")),
        }
    match = PATCH_RE.fullmatch(stem)
    if match:
        return {
            "tile": match.group("tile"),
            "kind": "win",
            "x": int(match.group("x")),
            "y": int(match.group("y")),
            "size": 1024,
            "source_id": None,
        }
    raise ValueError(f"unrecognized generated patch filename: {stem}")


def _size_bin(max_side_px: float) -> str:
    if max_side_px < 64:
        return "tiny_lt64"
    if max_side_px < 256:
        return "small_64_256"
    if max_side_px < 819:
        return "medium_256_819"
    return "large_ge819"


def _iter_labels(label_dir: Path) -> Iterable[tuple[str, list[tuple[int, float, float]]]]:
    for path in sorted(label_dir.glob("*.txt")):
        rows = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 5:
                raise ValueError(f"{path}:{line_no}: expected 5 YOLO fields")
            cls = int(fields[0])
            if not 0 <= cls < len(CLASS_NAMES):
                raise ValueError(f"{path}:{line_no}: invalid class {cls}")
            rows.append((cls, float(fields[3]), float(fields[4])))
        yield path.stem, rows


def summarize_split(image_dir: Path, label_dir: Path, patch_size: int = 1024) -> dict:
    """Return image, class, tile and object-size counts for one split."""
    images = sorted(image_dir.glob("*.png"))
    label_map = {stem: rows for stem, rows in _iter_labels(label_dir)}
    per_class = Counter()
    size_bins = Counter()
    tile_counts = Counter()
    positive_images = 0
    boxes = 0
    for image in images:
        meta = parse_patch_name(image.stem)
        tile_counts[meta["tile"]] += 1
        rows = label_map.get(image.stem, [])
        if rows:
            positive_images += 1
        for cls, width, height in rows:
            boxes += 1
            per_class[CLASS_NAMES[cls]] += 1
            size_bins[_size_bin(max(width, height) * patch_size)] += 1
    unknown_labels = sorted(set(label_map) - {p.stem for p in images})
    return {
        "images": len(images),
        "positive_images": positive_images,
        "negative_images": len(images) - positive_images,
        "boxes": boxes,
        "boxes_per_class": {name: per_class[name] for name in CLASS_NAMES},
        "size_bins": {
            name: size_bins[name]
            for name in ("tiny_lt64", "small_64_256", "medium_256_819", "large_ge819")
        },
        "tiles": dict(sorted(tile_counts.items())),
        "unknown_label_files": unknown_labels,
    }


def find_split_overlaps(source_splits: dict[int, set[str]]) -> dict[int, list[str]]:
    """Return source IDs assigned to more than one split."""
    return {
        int(source_id): sorted(splits)
        for source_id, splits in source_splits.items()
        if len(splits) > 1
    }


def choose_source(candidates: Iterable[tuple[int, float]]) -> int | None:
    """Choose the smallest containing bbox, with source_id as a tie-breaker."""
    candidates = list(candidates)
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[1], item[0]))[0]


def _intersects(box: tuple[float, float, float, float], x: int, y: int, size: int) -> bool:
    x0, y0, x1, y1 = box
    return x0 < x + size and x1 > x and y0 < y + size and y1 > y


def _load_tile_boxes(base: Path) -> tuple[dict, dict]:
    """Load source bbox pixel extents keyed by tile, using the production mapping."""
    scripts = base / "data" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import make_yolo_dataset as maker  # pylint: disable=import-outside-toplevel
    import tifffile  # pylint: disable=import-outside-toplevel

    raw = base / "data" / "raw"
    records = {
        int(row["source_id"]): row
        for row in json.loads((raw / "mars_landslides.json").read_text(encoding="utf-8"))
    }
    polygons = maker.load_shp_polygons()
    tifs = maker.discover_tifs(raw / "ctx_tiles")
    tile_boxes = {}
    for tile, tif_path in tifs.items():
        arr = tifffile.memmap(tif_path)
        height, width = arr.shape
        gt = maker.read_geotransform(tif_path)
        boxes = []
        for source_id, record in records.items():
            if record.get("type") not in maker.CLASSES or source_id not in polygons:
                continue
            px0, py0, px1, py1 = maker.aabb_to_pixel(polygons[source_id], gt, width, height)
            if px1 > px0 and py1 > py0:
                cls = maker.CLASSES[record["type"]]
                boxes.append((source_id, cls, (px0, py0, px1, py1),
                              max(0.0, px1 - px0) * max(0.0, py1 - py0)))
        tile_boxes[tile] = boxes
    return tile_boxes, tifs


def _read_label_rows(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) >= 5:
            rows.append(tuple([int(fields[0])] + [float(x) for x in fields[1:5]]))
    return rows


def _resolve_label_source(tile_boxes, meta: dict, row: tuple[int, float, float, float, float]) -> int | None:
    cls, nx, ny, _width, _height = row
    cx = meta["x"] + nx * meta["size"]
    cy = meta["y"] + ny * meta["size"]
    candidates = [
        (sid, area)
        for sid, source_cls, box, area in tile_boxes.get(meta["tile"], [])
        if source_cls == cls and box[0] <= cx <= box[2] and box[1] <= cy <= box[3]
    ]
    return choose_source(candidates)


def audit_source_provenance(dataset_root: Path, base: Path, splits=("train", "val", "test")) -> dict:
    """Map patch footprints back to SHP source IDs and report split leakage."""
    tile_boxes, _ = _load_tile_boxes(base)
    source_splits = defaultdict(set)
    source_tiles = defaultdict(set)
    tile_splits = defaultdict(set)
    negative_hits = []
    patch_count = 0
    for split in splits:
        image_dir = dataset_root / "images" / split
        for image in sorted(image_dir.glob("*.png")):
            patch_count += 1
            meta = parse_patch_name(image.stem)
            label_path = dataset_root / "labels" / split / f"{image.stem}.txt"
            rows = _read_label_rows(label_path)
            labeled_ids = set()
            if meta["source_id"] is not None:
                labeled_ids.add(meta["source_id"])
            else:
                for row in rows:
                    source_id = _resolve_label_source(tile_boxes, meta, row)
                    if source_id is not None:
                        labeled_ids.add(source_id)
            for sid in labeled_ids:
                source_splits[sid].add(split)
                source_tiles[sid].add(meta["tile"])
                tile_splits[meta["tile"]].add(split)
            hits = {
                sid
                for sid, _source_cls, box, _area in tile_boxes.get(meta["tile"], [])
                if _intersects(box, meta["x"], meta["y"], meta["size"])
            }
            if not rows and hits:
                negative_hits.append({"split": split, "image": image.name, "source_ids": sorted(hits)})
    return {
        "patches_checked": patch_count,
        "source_count": len(source_splits),
        "source_split_overlaps": find_split_overlaps(source_splits),
        "source_cross_tile": {
            str(sid): sorted(tiles)
            for sid, tiles in source_tiles.items()
            if len(tiles) > 1
        },
        "tile_split_membership": {
            tile: sorted(splits_seen) for tile, splits_seen in sorted(tile_splits.items())
        },
        "negative_patches_intersecting_source_bbox": negative_hits,
    }


def run_audit(dataset_root: Path, base: Path, patch_size: int = 1024, with_provenance: bool = True) -> dict:
    result = {
        "dataset": str(dataset_root),
        "splits": {
            split: summarize_split(
                dataset_root / "images" / split,
                dataset_root / "labels" / split,
                patch_size,
            )
            for split in ("train", "val", "test")
        },
    }
    if with_provenance:
        result["provenance"] = audit_source_provenance(dataset_root, base)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/yolo"))
    parser.add_argument("--base", type=Path, default=Path("."))
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-provenance", action="store_true")
    args = parser.parse_args()
    report = run_audit(
        args.dataset,
        args.base,
        patch_size=args.patch_size,
        with_provenance=not args.no_provenance,
    )
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.out:
        args.out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
