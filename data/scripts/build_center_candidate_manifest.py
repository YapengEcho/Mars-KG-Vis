# -*- coding: utf-8 -*-
"""Build a candidate-label manifest from the reviewed center crops.

Reads the review draft and writes one new JSON. It does not edit the SHP,
the review CSV, or any existing training set.

Included primaries are ``accept`` only. ``needs_expand`` is listed but not
labeled: the review says which side to grow, not by how many pixels.
``boundary_wrong``, ``partial`` and ``reject`` are counted and excluded.

Only accepted, complete neighbors are positive labels. Held or unreviewed
neighbors are explicit excluded regions and block ordinary YOLO export.
Boxes are axis-aligned, normalized ``cx cy w h``, same geometry as
``make_yolo_dataset.clip_box_to_window``. Classes use SL/SP/FL/CL=0/1/2/3.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from make_centered_review_pack import (
    BASE,
    OUT_DIR,
    _intersects,
    discover_tifs,
    map_source_records,
    read_geotransform,
    _record_for_tile,
    _tile_bounds,
)
import shapefile
from check_tile_connectivity import tiles_of_bbox


INCLUDE = {"accept"}
HOLD = {"needs_expand", "boundary_wrong", "partial", "reject"}
CLASSES = {"SL": 0, "SP": 1, "FL": 2, "CL": 3}
# Existing mars_landslides catalog normalizes SHP RL -> SL and DF -> FL.
TYPE_ALIASES = {"RL": "SL", "DF": "FL"}
OVERRIDE_FIELDS = ["tile", "source_id", "x0", "y0", "x1", "y1",
                   "label_scope", "reviewer", "confirmed", "reason"]


def validate_box(box) -> list[float]:
    if len(box) != 4:
        raise ValueError("box must contain x0,y0,x1,y1")
    box = list(map(float, box))
    if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"invalid box: {box}")
    return box


def contains(outer, inner) -> bool:
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3])


def load_overrides(path: Path | None, review_keys: set) -> dict:
    if path is None:
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result, seen = {}, set()
    for row in rows:
        key = (row["tile"], int(row["source_id"]))
        if key not in review_keys or key in seen:
            raise ValueError(f"unknown or duplicate override key: {key}")
        seen.add(key)
        confirmed = row["confirmed"].strip().lower()
        if confirmed not in {"", "false", "true"}:
            raise ValueError(f"confirmed must be true/false: {key}")
        if confirmed != "true":
            continue
        if row["label_scope"] not in {"full_landslide", "visible_core"}:
            raise ValueError(f"unknown label_scope: {key}")
        if not row["reviewer"].strip() or not row["reason"].strip():
            raise ValueError(f"confirmed override needs reviewer and reason: {key}")
        result[key] = {"aabb_px": validate_box([row[k] for k in ("x0", "y0", "x1", "y1")]),
                       "label_scope": row["label_scope"], "reviewer": row["reviewer"],
                       "reason": row["reason"]}
    return result


def label_decision(tile, record, reviews, overrides) -> tuple[list, str | None, str]:
    key = (tile, record["source_id"])
    override = overrides.get(key)
    review = reviews.get(key)
    if review and review.get("explicit_exclusion"):
        boxes = [record["aabb_px"]]
        if override:
            boxes.append(override["aabb_px"])
        if review.get("exclusion_box_px"):
            boxes.append(review["exclusion_box_px"])
        union = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                 max(b[2] for b in boxes), max(b[3] for b in boxes)]
        return union, "reject", "explicit_human_exclusion"
    if override:
        if override["label_scope"] == "full_landslide":
            return override["aabb_px"], None, "confirmed_manual_aabb"
        old, new = record["aabb_px"], override["aabb_px"]
        union = [min(old[0], new[0]), min(old[1], new[1]),
                 max(old[2], new[2]), max(old[3], new[3])]
        return union, "visible_core_separate_task", "manual_visible_core"
    review = reviews.get(key)
    if review is None:
        return record["aabb_px"], "unreviewed_neighbor", "shp_aabb"
    status = review["review_status"]
    return record["aabb_px"], None if status == "accept" else status, "shp_aabb"


def make_candidate(tile, primary, records, reviews, overrides) -> dict:
    row = reviews[(tile, primary["source_id"])]
    origin = json.loads(row["window_origin"])
    native = json.loads(row["native_window_px"])
    labels, excluded = [], []
    for other in sorted(records, key=lambda rec: rec["source_id"]):
        box, reason, provenance = label_decision(tile, other, reviews, overrides)
        if not _intersects(tuple(box), tuple(origin)):
            continue
        xywh = yolo_box(box, origin, native)
        if reason is None and not contains(origin, box):
            reason = "accepted_neighbor_truncated_in_this_crop"
        class_name = TYPE_ALIASES.get(other["raw_type"], other["raw_type"])
        item = {"source_id": other["source_id"], "class_id": CLASSES[class_name],
                "class_name": class_name, "shp_raw_type": other["raw_type"], "stage1_class_id": 0,
                "xywh": xywh, "aabb_tile_px": box, "label_origin": provenance,
                "review_status": reviews.get((tile, other["source_id"]), {}).get("review_status"),
                "review_notes": reviews.get((tile, other["source_id"]), {}).get("review_notes", ""),
                "manual_override": overrides.get((tile, other["source_id"]))}
        if reason is None:
            labels.append(item)
        else:
            item["reason"] = reason
            excluded.append(item)
    return {"tile": tile, "source_id": primary["source_id"], "split": "train",
            "window_origin": origin, "native_window_px": native,
            "labels": labels, "excluded_regions": excluded,
            "unreviewed_neighbor_ids": [r["source_id"] for r in excluded if r["reason"] == "unreviewed_neighbor"],
            "eligible_for_standard_yolo_export": not excluded}


def validate_plan(plan, protected_tiles) -> None:
    if not plan or set(plan.values()) != {"train"}:
        raise ValueError("candidate plan must be nonempty and train-only")
    overlap = sorted(set(plan) & protected_tiles)
    if overlap:
        raise ValueError(f"protected val/test/final-eval tiles in plan: {overlap}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_review(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    unknown = sorted({r["review_status"] for r in rows} - INCLUDE - HOLD)
    if unknown:
        raise ValueError(f"unknown review_status: {unknown}")
    if not rows:
        raise ValueError("review CSV is empty")
    seen = set()
    for row in rows:
        key = (row["tile"], int(row["source_id"]))
        if key in seen:
            raise ValueError(f"duplicate review key: {key}")
        seen.add(key)
        origin = validate_box(json.loads(row["window_origin"]))
        native = json.loads(row["native_window_px"])
        if native != [origin[2] - origin[0], origin[3] - origin[1]]:
            raise ValueError(f"window size mismatch: {key}")
        validate_box(json.loads(row["polygon_bbox_px"]))
    return rows


def yolo_box(aabb: list[float], origin: list[int], window: tuple[int, int]) -> list[float] | None:
    """Clip an AABB to the window and return normalized cx, cy, w, h."""
    aabb = validate_box(aabb)
    x0, y0, x1, y1 = validate_box(origin)
    width, height = window
    if [width, height] != [x1 - x0, y1 - y0]:
        raise ValueError("window dimensions do not match origin")
    ix0, iy0 = max(aabb[0], x0), max(aabb[1], y0)
    ix1, iy1 = min(aabb[2], x1), min(aabb[3], y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return [
        round(((ix0 + ix1) / 2 - x0) / width, 8),
        round(((iy0 + iy1) / 2 - y0) / height, 8),
        round((ix1 - ix0) / width, 8),
        round((iy1 - iy0) / height, 8),
    ]


def build(review_csv: Path, plan_path: Path, ctx_dir: Path, shp_path: Path,
          out_path: Path, overrides_path: Path | None = None,
          split_plan_path: Path = BASE / "data/raw/tile_split_plan_flval_v2.json",
          final_eval_path: Path = BASE / "data/raw/final_eval_tiles_frozen.json") -> dict:
    rows = load_review(review_csv)
    by_key = {(r["tile"], int(r["source_id"])): r for r in rows}
    overrides = load_overrides(overrides_path, set(by_key))
    exclusions_path = review_csv.parent / "manual_review/review_exclusions.json"
    if exclusions_path.exists():
        exclusions = json.loads(exclusions_path.read_text(encoding="utf-8"))
        for decision in exclusions.values():
            key = (decision["tile"], int(decision["source_id"]))
            if key in by_key:
                if decision["training_use"] != "exclude_positive" or decision["negative_eligible"] is not False:
                    raise ValueError("human exclusions cannot become negative samples")
                by_key[key].update(review_status="reject", review_notes=decision["reason"], explicit_exclusion=True)
                if decision.get("exclusion_box_px"):
                    by_key[key]["exclusion_box_px"] = validate_box(decision["exclusion_box_px"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    splits = json.loads(split_plan_path.read_text(encoding="utf-8"))
    frozen = json.loads(final_eval_path.read_text(encoding="utf-8"))
    protected = {t for t, split in splits.items() if split in {"val", "test"}} | set(frozen["tiles"])
    validate_plan(plan, protected)
    if set(r["tile"] for r in rows) - set(plan):
        raise ValueError("review contains tiles outside candidate plan")
    tif_of = discover_tifs(ctx_dir)
    cache_manifests = {}
    cache_geometry = {}
    for tile in sorted(set(plan) - set(tif_of)):
        path = review_csv.parent / "native_context" / tile / "manifest.json"
        if not path.exists():
            continue
        cache = json.loads(path.read_text(encoding="utf-8"))
        gt = tuple(map(float, cache["geotransform"]))
        width, height = int(cache["tile_width"]), int(cache["tile_height"])
        if (cache["tile"] != tile or len(gt) != 6 or not all(map(math.isfinite, gt))
                or gt[1] <= 0 or gt[5] >= 0 or min(width, height) <= 0
                or cache["original"]["shape"] != [height, width]):
            raise ValueError(f"invalid native cache geometry: {tile}")
        if not cache["entries"] or any(e["stride"] != 1 or not Path(e["path"]).is_file()
                                       for e in cache["entries"]):
            raise ValueError(f"missing native cache pixels: {tile}")
        cache_manifests[tile] = path
        cache_geometry[tile] = (gt, width, height)
    missing = sorted(set(plan) - set(tif_of) - set(cache_geometry))
    if missing:
        raise FileNotFoundError(f"missing CTX tiles: {missing}")
    with shapefile.Reader(str(shp_path)) as reader:
        shape_records = map_source_records(reader.shapeRecords())

    candidates = []
    held = []
    matched = set()
    for tile in sorted(plan):
        gt, width, height = (read_geotransform(tif_of[tile]) if tile in tif_of
                             else cache_geometry[tile])
        bounds = _tile_bounds(gt, width, height)
        records = []
        for source_id, record in shape_records.items():
            converted = _record_for_tile(source_id, record, gt, bounds)
            if converted:
                records.append(converted)
        for record in records:
            if TYPE_ALIASES.get(record["raw_type"], record["raw_type"]) not in CLASSES:
                raise ValueError(f"unknown SHP class: {record['raw_type']}")
            key = (tile, record["source_id"])
            hits = tiles_of_bbox(*shape_records[record["source_id"]].shape.bbox) & protected
            if hits:
                raise ValueError(f"source {key} connects to protected tiles: {sorted(hits)}")
            review = by_key.get(key)
            if review is None:
                continue
            matched.add(key)
            old_box = json.loads(review["polygon_bbox_px"])
            if any(abs(a - b) > 0.01 for a, b in zip(old_box, record["aabb_px"])):
                raise ValueError(f"review/SHP geometry changed: {key}")
            origin = json.loads(review["window_origin"])
            if not contains([0, 0, width, height], origin):
                raise ValueError(f"review window outside tile: {key}")
            if key in overrides and not contains([0, 0, width, height], overrides[key]["aabb_px"]):
                raise ValueError(f"manual box outside tile; mosaic/recrop required: {key}")
            box, reason, _ = label_decision(tile, record, by_key, overrides)
            if reason is None and not contains(origin, box):
                reason = "primary_requires_recrop"
            if reason is not None:
                held.append({"tile": tile, "source_id": record["source_id"],
                             "review_status": review["review_status"], "reason": reason,
                             "review_notes": review["review_notes"],
                             "manual_override": overrides.get(key)})
                continue
            candidate = make_candidate(tile, record, records, by_key, overrides)
            candidate["center_poly_file"] = str(review_csv.parent / review["center_poly_file"])
            candidate["center_aabb_file"] = str(review_csv.parent / review["center_aabb_file"])
            if tile in cache_manifests:
                candidate["native_context_manifest"] = str(cache_manifests[tile].resolve())
                candidate["pixel_source"] = "cached_native_context; full-tile context requires re-download"
            if not any(label["source_id"] == record["source_id"] for label in candidate["labels"]):
                raise ValueError(f"primary has no valid positive label: {key}")
            candidates.append(candidate)

    if matched != set(by_key):
        raise ValueError(f"review objects absent from SHP/tile: {sorted(set(by_key) - matched)}")
    inputs = {"review": review_csv, "plan": plan_path, "split_plan": split_plan_path,
              "final_eval": final_eval_path, "shp": shp_path,
              "shx": shp_path.with_suffix(".shx"), "dbf": shp_path.with_suffix(".dbf"),
              "generator": Path(__file__),
              "geometry_helper": Path(__file__).with_name("make_centered_review_pack.py"),
              "connectivity_helper": Path(__file__).with_name("check_tile_connectivity.py")}
    if overrides_path:
        inputs["overrides"] = overrides_path
    if exclusions_path.exists():
        inputs["exclusions"] = exclusions_path
    for tile, path in cache_manifests.items():
        inputs[f"native_context_{tile}"] = path
    if out_path.resolve() in {p.resolve() for p in inputs.values()}:
        raise ValueError("output must not overwrite an input")
    labels = [label for c in candidates for label in c["labels"]]

    payload = {
        "schema_version": 2,
        "status": "candidate_manifest_only",
        "purpose": "复审通过的中心样本候选标签；水平框；不修改 SHP 与旧训练集",
        "source_review": str(review_csv),
        "source_review_sha256": sha256_file(review_csv),
        "inputs": {k: {"path": str(p), "sha256": sha256_file(p)} for k, p in inputs.items()},
        "class_map": CLASSES, "stage1_class_map": {"LANDSLIDE": 0},
        "shp_type_aliases": TYPE_ALIASES,
        "box": "axis-aligned cx cy w h, normalized to the exact reviewed window",
        "label_scope": "full_landslide",
        "review_status_counts": dict(Counter(r["review_status"] for r in rows)),
        "included_status": sorted(INCLUDE),
        "candidate_count": len(candidates),
        "held_count": len(held),
        "standard_yolo_eligible_count": sum(c["eligible_for_standard_yolo_export"] for c in candidates),
        "positive_label_occurrences": len(labels),
        "unique_positive_source_count": len({label["source_id"] for label in labels}),
        "class_label_occurrences": dict(Counter(label["class_name"] for label in labels)),
        "excluded_region_occurrences": sum(len(c["excluded_regions"]) for c in candidates),
        "split_guard": {"train_only": True, "protected_tile_overlap": [],
                        "protected_source_overlap": [], "source_rule": "conservative SHP bbox touch"},
        "export_policy": "Standard YOLO export requires zero excluded regions. Otherwise hold the crop or use a separately validated ignore-loss path; merely omitting labels does not ignore pixels.",
        "held": held,
        "candidates": candidates,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_repair_template(payload: dict, path: Path) -> None:
    """Create a manual annotation sidecar; never overwrite human edits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OVERRIDE_FIELDS + ["review_status", "review_notes"])
        writer.writeheader()
        for item in payload["held"]:
            writer.writerow({"tile": item["tile"], "source_id": item["source_id"],
                             "label_scope": "full_landslide", "confirmed": "false",
                             "review_status": item["review_status"], "review_notes": item["review_notes"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path,
                        default=OUT_DIR / "centered_review_manifest.draft.csv")
    parser.add_argument("--plan", type=Path,
                        default=Path(__file__).resolve().parents[2]
                        / "data" / "raw" / "tile_split_plan_today20_new_only.json")
    parser.add_argument("--ctx", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "raw" / "ctx_tiles")
    parser.add_argument("--shp", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "raw"
                        / "mars_landslide_boundaries" / "landslide_boundaries" / "火星滑坡-带边界.shp")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "manifest"
                        / "center_review_accept_candidates.json")
    parser.add_argument("--overrides", type=Path,
                        help="Confirmed manual boxes in full CTX tile pixel coordinates")
    parser.add_argument("--repair-template", type=Path,
                        help="Create a new manual correction CSV; never overwrite an existing file")
    args = parser.parse_args()
    payload = build(args.review, args.plan, args.ctx, args.shp, args.out, args.overrides)
    if args.repair_template:
        write_repair_template(payload, args.repair_template)
    print(json.dumps({k: payload[k] for k in ("candidate_count", "held_count", "standard_yolo_eligible_count",
                     "positive_label_occurrences", "unique_positive_source_count", "class_label_occurrences",
                     "excluded_region_occurrences", "manifest_sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
