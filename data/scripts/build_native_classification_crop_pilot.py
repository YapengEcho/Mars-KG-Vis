"""Read three native CTX crops for a visual classification engineering pilot.

No encoder, inference, classifier fit, downloads, label edits or dataset changes.
The old padded crop and a direct-native 224 crop are displayed side by side.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import extract_manual_classification_features as C
from export_manual_detection_dataset import stretch_uint8
from make_centered_review_pack import read_geotransform
from recover_reviewed_mosaic_samples import exact_coverage, native_crop, native_hash, world_box

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "runs/research/classification_pixel_provenance_20261015_v2/plan.json"
EXPORT = ROOT / "data/processed/manual_yolo_20261009_v8/export_plan.json"
TRIAGE = ROOT / "runs/research/fourclass_error_triage_20261015_v1/report.json"
OUT = ROOT / "runs/research/native_classification_crop_pilot_20261015_v1"
IDS = ("590", "1928", "1358")
MARGIN = 0.25


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    C.write_json_atomic(path, value)


def contains(outer, inner):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def source_extent_in_anchor(gt, src):
    t = src["geotransform"]
    w, h = src["size_px"]
    return [(t[0] - gt[0]) / gt[1], (t[3] - gt[3]) / gt[5],
            (t[0] + w*t[1] - gt[0]) / gt[1], (t[3] + h*t[5] - gt[3]) / gt[5]]


def choose_window(box, gt, sources):
    width, height = box[2] - box[0], box[3] - box[1]
    expanded = [box[0]-MARGIN*width, box[1]-MARGIN*height,
                box[2]+MARGIN*width, box[3]+MARGIN*height]
    side = max(math.ceil(1.5*max(width, height)),
               math.ceil(expanded[2])-math.floor(expanded[0]),
               math.ceil(expanded[3])-math.floor(expanded[1]))
    nominal = [math.floor((box[0]+box[2]-side)/2), math.floor((box[1]+box[3]-side)/2)]
    # Subpixel outward rounding can add one native pixel. Constrain placement so
    # every side keeps at least 25% of its corresponding target dimension.
    low = [math.ceil(expanded[2]-side), math.ceil(expanded[3]-side)]
    high = [math.floor(expanded[0]), math.floor(expanded[1])]
    nominal = [max(low[i], min(high[i], nominal[i])) for i in range(2)]
    extents = [source_extent_in_anchor(gt, s) for s in sources]
    bounds = [math.ceil(min(e[0] for e in extents)-1e-6), math.ceil(min(e[1] for e in extents)-1e-6),
              math.floor(max(e[2] for e in extents)+1e-6), math.floor(max(e[3] for e in extents)+1e-6)]
    feasible_low = [max(low[i], bounds[i]) for i in range(2)]
    feasible_high = [min(high[i], bounds[i+2]-side) for i in range(2)]
    if any(a > b for a, b in zip(feasible_low, feasible_high)):
        raise ValueError("Existing same-fold donors cannot contain full square and 25% per-axis context")
    actual = [max(feasible_low[i], min(feasible_high[i], nominal[i])) for i in range(2)]
    window = [actual[0], actual[1], actual[0]+side, actual[1]+side]
    if not contains(window, expanded):
        raise RuntimeError("Bounded shift trimmed the human target or required context")
    return window, dict(nominal_window_px=[*nominal, nominal[0]+side, nominal[1]+side],
        actual_window_px=window, shift_px=[actual[i]-nominal[i] for i in range(2)],
        side_px=side, margin_per_target_axis=MARGIN, expanded_context_box_px=expanded,
        actual_margin_px=dict(left=box[0]-window[0], top=box[1]-window[1],
                              right=window[2]-box[2], bottom=window[3]-box[3]),
        donor_extent_bounding_rectangle_px=bounds,
        shift_policy="Minimum bounded shift preserving full human AABB and >=25% of target width/height on each corresponding side; square short-axis extra context may be asymmetric")


def entry_from_window(gt, window, sources):
    x0, y0, x1, y1 = window
    return dict(geotransform=[gt[0]+x0*gt[1], gt[1], 0, gt[3]+y0*gt[5], 0, gt[5]],
                size_px=[x1-x0, y1-y0], sources=sources)


def verify_sources(sources, fold, tile_folds):
    receipts = []
    for src in sources:
        path = Path(src["path"])
        if not path.is_file():
            raise FileNotFoundError("Native donor unavailable: " + str(path))
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (src["bytes"], src["mtime_ns"]):
            raise ValueError("Native donor metadata changed: " + str(path))
        if tile_folds.get(src["tile"]) != fold:
            raise ValueError("Native donor not isolated in this target's corrected fold")
        cache_sha = src.get("cache_sha256")
        if cache_sha and C.sha256(path) != cache_sha:
            raise ValueError("Cached native donor bytes changed: " + str(path))
        if not cache_sha:
            actual_gt, w, h = read_geotransform(path)
            if [w, h] != src["size_px"] or not np.allclose(actual_gt, src["geotransform"], atol=1e-7, rtol=0):
                raise ValueError("Raw donor geometry differs from frozen metadata")
        receipts.append(dict(tile=src["tile"], path=str(path), bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns, cache_sha256=cache_sha, fold_id=fold,
            verification="exact SHA256 for cache; raw header+size+mtime and frozen parent-native-region hash for large raw TIFF"))
    return receipts


def percentile_receipt(crop):
    counts = np.bincount(crop.ravel(), minlength=256).astype(np.int64)
    counts[0] = 0
    n = int(counts.sum())
    cumulative = np.cumsum(counts)
    def quantile(p):
        index = (n-1)*p
        a, b = math.floor(index), math.ceil(index)
        lo, hi = np.searchsorted(cumulative, a, side="right"), np.searchsorted(cumulative, b, side="right")
        return float(lo+(hi-lo)*(index-a))
    lo, hi = (quantile(.02), quantile(.98)) if n else (0., 255.)
    if hi <= lo:
        hi = lo+1
    return dict(policy="Historical shared uint8 stretch: nonzero-pixel 2nd/98th percentiles, linear interpolation, clip to0..255, uint8 truncation, raw zero remains0",
                p02=lo, p98=hi, nonzero_native_pixels=n, raw_zero_fraction=float(np.mean(crop == 0)))


def overlay(image, window, box, old_window):
    result = image.convert("RGB")
    draw = ImageDraw.Draw(result)
    side = window[2]-window[0]
    def scaled(rect):
        return [(rect[0]-window[0])*224/side, (rect[1]-window[1])*224/side,
                (rect[2]-window[0])*224/side, (rect[3]-window[1])*224/side]
    draw.rectangle(scaled(old_window), outline="#00dbff", width=2)
    draw.rectangle(scaled(box), outline="#ffd13b", width=2)
    draw.rectangle((0, 0, 223, 223), outline="#44ee87", width=2)
    return result


def build(out=OUT):
    plan = C.verify_plan(PLAN)
    exports = C.read_json(EXPORT)
    manifest = C.read_json(plan["source_manifest"])
    if manifest["source_plan_sha256"] != C.sha256(EXPORT):
        raise ValueError("Detector export plan changed from classifier crop provenance")
    triage = C.read_json(TRIAGE)
    highpad = {r["source_id"] for r in triage["confirmed_high_padding_examples"]}
    if set(IDS) != highpad:
        raise ValueError("Pilot target set differs from documented high-padding examples")
    rows = {r["source_id"]: r for r in plan["rows"]}
    samples = {s["stem"]: s for s in exports["samples"]}
    tile_folds = {}
    for row in plan["rows"]:
        for tile in row["source_tiles"]:
            previous = tile_folds.setdefault(tile, row["fold_id"])
            if previous != row["fold_id"]:
                raise ValueError("Corrected donor tile crosses classifier folds")
    out = C.new_directory(out)
    report = dict(status="running_visual_engineering_pilot", created_utc=utc(),
        scope="three_target_native_pixel_visual_pilot_not_model_scores_or_training_dataset",
        source_plan=str(PLAN), source_plan_sha256=C.sha256(PLAN),
        source_protocol_sha256=plan["source_protocol_sha256"], source_manifest_sha256=plan["source_manifest_sha256"],
        detector_export_plan=str(EXPORT), detector_export_plan_sha256=C.sha256(EXPORT),
        triage_report=str(TRIAGE), triage_report_sha256=C.sha256(TRIAGE),
        script_sha256=C.sha256(__file__), target_ids=list(IDS), records=[],
        inputs_modified=False, labels_modified=False, downloads=False, gpu_used=False,
        model_inference=False, classifiers_fitted=False,
        limitations=["Selected three severe-padding examples; no model score or general performance claim.",
            "Native contexts can contain neighboring geology and target subtypes remain unreviewed.",
            "Fresh context and brightness stretch differ as well as resampling: any later score comparison is a crop-pipeline comparison.",
            "Full1083 native recrop requires donor/context/cross-fold footprint verification before classifier evaluation.",
            "Geometric padding fraction is separate from black pixels from terrain/nodata/percentile clipping."])
    write(out / "report.json", report)
    sheet = Image.new("RGB", (3*244, 3*276+36), "#171717")
    draw = ImageDraw.Draw(sheet)
    for j, title in enumerate(("Old256 ->224", "Native ->224", "Native overlays")):
        draw.text((j*244+10, 10), title, fill="white")
    for index, identifier in enumerate(IDS):
        row = rows[identifier]
        sample = samples[row["source_stem"]]
        record = dict(source_id=identifier, catalog_class=row["catalog_class"],
            fold_id=row["fold_id"], spatial_group_id=row["spatial_group_id"],
            old_crop=row["crop"], old_crop_sha256=row["crop_sha256"],
            old_geometric_padding_fraction=row["padding_fraction"], source_stem=row["source_stem"])
        try:
            if C.sha256(row["crop"]) != row["crop_sha256"]:
                raise ValueError("Old classifier crop changed")
            source = sample["pixel_source"]
            if source["kind"] != "native_mosaic":
                raise ValueError("These three pilot examples require frozen native-mosaic provenance")
            labels = [r for r in sample["labels"] if str(r["source_id"]) == identifier]
            if len(labels) != 1 or labels[0]["class_name"] != row["catalog_class"]:
                raise ValueError("Human box/type provenance mismatch")
            box, gt = labels[0]["manual_aabb_px"], sample["geotransform"]
            donors = source["sources"]
            record["donors"] = verify_sources(donors, row["fold_id"], tile_folds)
            parent_crop = native_crop(source)
            parent_hash = native_hash(parent_crop)
            del parent_crop
            if parent_hash != sample["native_crop_sha256"]:
                raise ValueError("Frozen detector parent native-pixel region changed")
            window, geometry = choose_window(box, gt, donors)
            entry = entry_from_window(gt, window, donors)
            if not exact_coverage(entry):
                raise ValueError("Requested native context has a donor-coverage hole; no invented padding permitted")
            crop = native_crop(entry)
            normalization = percentile_receipt(crop)
            raw_hash = native_hash(crop)
            stretched = stretch_uint8(crop)
            native224 = Image.fromarray(stretched).resize((224, 224), Image.Resampling.BILINEAR)
            with Image.open(row["crop"]) as old:
                if old.size != (256, 256):
                    raise ValueError("Expected old frozen256×256 crop")
                old224 = old.convert("L").resize((224, 224), Image.Resampling.BILINEAR)
            folder = out / identifier
            folder.mkdir()
            native_path, old_path, overlay_path = folder / "native_direct224.png", folder / "old_256_to224.png", folder / "native_overlay224.png"
            native224.save(native_path)
            old224.save(old_path)
            marked = overlay(native224, window, box, sample["window_px"])
            marked.save(overlay_path)
            record.update(status="native_pixels_read_and_direct224_exported", manual_box_px=box,
                manual_box_world_m=world_box(gt, box), native_window_world_m=world_box(gt, window),
                anchor_geotransform=gt, native_output_geotransform=entry["geotransform"],
                native_shape=list(crop.shape), native_pixel_sha256=raw_hash,
                frozen_parent_native_pixel_sha256_verified=parent_hash, geometry=geometry,
                full_human_box_preserved=True, minimum_context_preserved=True,
                complete_native_coverage=True, all_donors_same_corrected_fold=True,
                geometric_padding_fraction=0.0, brightness_normalization=normalization,
                old_zero_pixel_fraction224=float(np.mean(np.asarray(old224)==0)),
                native_zero_pixel_fraction224=float(np.mean(np.asarray(native224)==0)),
                native_direct224=str(native_path), native_direct224_sha256=C.sha256(native_path),
                old_256_to224=str(old_path), old_256_to224_sha256=C.sha256(old_path),
                native_overlay224=str(overlay_path), native_overlay224_sha256=C.sha256(overlay_path),
                resampling="native uint8 ->historical crop-local nonzero2/98stretch ->PIL BILINEAR224; no detectorPNG or256 intermediary",
                overlay_legend=dict(yellow="unchanged human box", cyan="old detector parent window", green="new native square boundary"))
            y = 36+index*276
            for j, image in enumerate((old224, native224, marked)):
                sheet.paste(image, (10+j*244, y))
            draw.text((10, y+230), f"ID{identifier}/{row['catalog_class']} fold{row['fold_id']} old pad{row['padding_fraction']:.1%} ->0%", fill="white")
            del crop, stretched
        except (ValueError, FileNotFoundError) as error:
            record.update(status="held_native_io_or_coverage_limitation", error_type=type(error).__name__, error=str(error))
            draw.text((10, 36+index*276+30), f"ID{identifier}: held, inspect report", fill="#ffb96c")
        report["records"].append(record)
        write(out / "report.json", report)
    sheet.save(out / "comparison.png")
    successful = sum(r["status"] == "native_pixels_read_and_direct224_exported" for r in report["records"])
    report.update(status="completed_visual_engineering_pilot" if successful == 3 else "completed_with_native_io_limitations",
        finished_utc=utc(), exported_targets=successful, held_targets=3-successful,
        comparison=str(out / "comparison.png"), comparison_sha256=C.sha256(out / "comparison.png"))
    write(out / "report.json", report)
    (out / "README.md").write_text("""# 原始像素四分类裁片小试

仅检查590、1928、1358三个旧裁片填黑严重的目标，没有提特征、训练、改框或改类别。
comparison.png每行左边是原256裁片缩到224，中间是原始CTX像素直接缩到224，右边叠加方框。
黄色是完整人工框，青色是旧检测图取景范围，绿色是新的原始像素正方形边界。

新窗口长边为完整人工框长边的约1.5倍，至少保留每一方向对应框宽/高的25%上下文。
如窗口靠近原图边界，仅在不损失上述框和上下文的范围内平移；实际边距和位移记录在report.json。
所有供图图幅必须仍在该目标的纠正后CV折中；缺像素就暂扣，禁止伪造背景或拿预览图代替。
保持历史非零像素2/98百分位拉伸政策，原始窗口直接一次缩放到224，不经过检测PNG和256中间图。

表中旧/新几何填黑比例与零像素比例分别报告：原图nodata、亮度拉伸或暗地形也会产生黑像素。
这是三个典型问题的视觉工程小试，不能报告准确率提升；全1083重裁前还要重新核验像素供图、上下文及跨折重叠。
所有原文件均保留，完整来源、哈希、窗口、人工框、折和归一化记录在report.json。
""", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], out=str(out), exported_targets=successful,
        held_targets=3-successful), ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    destination = args.out.resolve()
    if not destination.is_relative_to(ROOT / "runs/research"):
        raise ValueError("Pilot must remain in project runs/research")
    build(destination)
