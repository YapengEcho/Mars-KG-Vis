"""Freeze native crop geometry, then resume an audited train-only 1083-crop build.

prepare is metadata-only. build requires the separately frozen spatial-audit
master plan. No downloads, GPU, encoders, subtype relabeling or cohort dropping.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import uuid
import zipfile

import numpy as np
from PIL import Image
import tifffile

import extract_manual_classification_features as C
import export_manual_detection_dataset as E
from build_native_classification_crop_pilot import choose_window, entry_from_window, source_extent_in_anchor
from make_centered_review_pack import discover_tifs, read_geotransform
from recover_reviewed_mosaic_samples import exact_coverage, world_box

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PLAN = ROOT / "runs/research/classification_pixel_provenance_20261015_v2/plan.json"
EXPORT = ROOT / "data/processed/manual_yolo_20261009_v8/export_plan.json"
CACHE = ROOT / "data/processed/centered_review_v2_train_20261006/native_context"
META = ROOT / "runs/research/native_crop_comparison_preparation_20261015_v1"
DATASET = ROOT / "data/processed/manual_stage2_native_20261015_v1"
MASTER = ROOT / "runs/research/native_crop_comparison_protocol_20261015_v1/master_plan.json"


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    C.write_json_atomic(path, value)


def intersects(a, b):
    return min(a[2], b[2])-max(a[0], b[0]) > 1e-5 and min(a[3], b[3])-max(a[1], b[1]) > 1e-5


def cache_source(entry, manifest, path):
    if entry["stride"] != 1:
        raise ValueError("Downsampled cache prohibited")
    gt = manifest["geotransform"]
    x, y = entry["window_origin"][:2]
    h, w = entry["shape"]
    p = Path(entry["path"])
    if not p.is_file():
        return None
    stat = p.stat()
    return dict(tile=manifest["tile"], path=str(p.resolve()), bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns, geotransform=[gt[0]+x*gt[1], gt[1], 0, gt[3]+y*gt[5], 0, gt[5]],
        size_px=[w, h], cache_sha256=entry["sha256"], source_kind="lossless_native_cache",
        cache_manifest=str(path.resolve()), cache_manifest_sha256=C.sha256(path))


def npz_payload(path, size):
    matches = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                shape, _, dtype = np.lib.format._read_array_header(stream,version)
            if dtype == np.uint8 and len(shape) == 2 and list(shape[::-1]) == size:
                matches.append(name[:-4])
    if len(matches) != 1:
        raise ValueError("Native NPZ must have one matching grayscale uint8 payload")
    return matches[0]


def available_pool(exports):
    pool, manifests = {}, []
    protected = set(exports["protected_tiles"])
    for tile, spec in exports["tile_sources"].items():
        if tile in protected:
            continue
        if spec["kind"] != "full_tile" or not Path(spec["path"]).is_file():
            continue
        stat = Path(spec["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (spec["bytes"], spec["mtime_ns"]):
            raise ValueError("Frozen raw native metadata changed: " + spec["path"])
        pool.setdefault(tile, []).append(dict(tile=tile, path=spec["path"], bytes=spec["bytes"],
            mtime_ns=spec["mtime_ns"], geotransform=spec["geotransform"],
            size_px=spec["tile_size_px"], source_kind="raw_full_tile"))
    for sample in exports["samples"]:
        if sample["pixel_source"]["kind"] != "native_mosaic":
            continue
        for src in sample["pixel_source"]["sources"]:
            if src["tile"] in protected:
                continue
            if not Path(src["path"]).is_file() or any(x["path"] == src["path"] for x in pool.get(src["tile"], [])):
                continue
            stat = Path(src["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (src["bytes"], src["mtime_ns"]):
                raise ValueError("Frozen mosaic donor metadata changed")
            pool.setdefault(src["tile"], []).append(dict(src,
                source_kind="lossless_native_cache" if src.get("cache_sha256") else "raw_full_tile"))
    # Existing training-region neighbors may not have appeared in the older
    # export. Admit their native TIFF headers explicitly and freeze receipts;
    # never download a missing neighbor or use protected evaluation imagery.
    for tile, path in sorted(discover_tifs(ROOT / "data/raw/ctx_tiles").items()):
        if tile in protected or any(x["path"] == str(path.resolve()) for x in pool.get(tile, [])):
            continue
        gt, w, h = read_geotransform(path)
        stat = path.stat()
        pool.setdefault(tile, []).append(dict(tile=tile, path=str(path.resolve()), bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns, geotransform=list(gt), size_px=[w, h],
            source_kind="existing_neighbor_raw_full_tile", admitted_from_current_native_header=True))
    for path in sorted(CACHE.glob("*/manifest.json")):
        manifest = C.read_json(path)
        if manifest["tile"] in protected:
            continue
        manifests.append(dict(path=str(path.resolve()), sha256=C.sha256(path)))
        for entry in manifest["entries"]:
            src = cache_source(entry, manifest, path)
            if src:
                pool.setdefault(src["tile"], []).append(src)
    return pool, manifests


def protected_rectangles(exports):
    scale = 3396190*math.pi/180
    rectangles = []
    for tile in exports["protected_tiles"]:
        match = re.fullmatch(r"E(-?\d+)_N(-?\d+)", tile)
        if not match:
            raise ValueError("Invalid protected tile identity")
        west, south = map(int, match.groups())
        rectangles.append(dict(tile=tile, world_m=[west*scale, -(south+4)*scale, (west+4)*scale, -south*scale],
                               geometry="Frozen nominal4degree CTX tile footprint; no external pixels read"))
    return rectangles


def donors_for(sample, anchor, pool, box):
    original = sample["pixel_source"]
    tiles = [s["tile"] for s in original["sources"]] if original["kind"] == "native_mosaic" else [sample["tile"]]
    preferred = [s["path"] for s in original["sources"]] if original["kind"] == "native_mosaic" else [original["path"]]
    side = math.ceil(1.5*max(box[2]-box[0], box[3]-box[1]))
    reach = [box[0]-side, box[1]-side, box[2]+side, box[3]+side]
    candidates = [s for sources in pool.values() for s in sources
                  if intersects(reach, source_extent_in_anchor(anchor, s))]
    # Frozen original donors first; replacement is allowed only from existing
    # stride1 lossless caches of the exact same donor tile, never a preview.
    candidates.sort(key=lambda s: (s["path"] not in preferred,
        preferred.index(s["path"]) if s["path"] in preferred else 0,
        s["source_kind"] != "raw_full_tile", s["path"]))
    return list({s["path"]: s for s in candidates}.values())


def covered_window(box, gt, donors, protected):
    choices = []
    for sources in [donors] + [[s] for s in donors]:
        try:
            window, geometry = choose_window(box, gt, sources)
            world = world_box(gt, window)
            if any(intersects(world, r["world_m"]) for r in protected):
                continue
            relevant = [s for s in donors if intersects(window, source_extent_in_anchor(gt, s))]
            if relevant and exact_coverage(entry_from_window(gt, window, relevant)):
                shift = geometry["shift_px"]
                choices.append((sum(x*x for x in shift), window, geometry, relevant))
                if not any(shift):
                    break
        except ValueError:
            continue
    if choices:
        _, window, geometry, sources = min(choices, key=lambda r: (r[0], r[1][1], r[1][0]))
        return window, geometry, sources
    # Uniform pre-score engineering rule: unavailable source edges can reduce
    # context, but must never cut the human box or invent/borrow held-out pixels.
    width, height = box[2]-box[0], box[3]-box[1]
    expanded = [box[0]-.25*width, box[1]-.25*height, box[2]+.25*width, box[3]+.25*height]
    side = max(math.ceil(1.5*max(width, height)), math.ceil(expanded[2])-math.floor(expanded[0]),
               math.ceil(expanded[3])-math.floor(expanded[1]))
    nominal = [math.floor((box[0]+box[2]-side)/2), math.floor((box[1]+box[3]-side)/2)]
    fallback = []
    for subset in [donors]+[[s] for s in donors]:
        extents = [source_extent_in_anchor(gt, s) for s in subset]
        if not extents:
            continue
        bounds = [math.ceil(min(e[0] for e in extents)-1e-6), math.ceil(min(e[1] for e in extents)-1e-6),
                  math.floor(max(e[2] for e in extents)+1e-6), math.floor(max(e[3] for e in extents)+1e-6)]
        low = [max(bounds[i], math.ceil(box[i+2]-side)) for i in range(2)]
        high = [min(bounds[i+2]-side, math.floor(box[i])) for i in range(2)]
        if any(a>b for a,b in zip(low,high)):
            continue
        actual = [max(low[i],min(high[i],nominal[i])) for i in range(2)]
        window = [actual[0],actual[1],actual[0]+side,actual[1]+side]
        if any(intersects(world_box(gt,window),r["world_m"]) for r in protected):
            continue
        sources = [s for s in donors if intersects(window,source_extent_in_anchor(gt,s))]
        if not exact_coverage(entry_from_window(gt,window,sources)):
            continue
        margins = dict(left=box[0]-window[0],top=box[1]-window[1],right=window[2]-box[2],bottom=window[3]-box[3])
        desired = dict(left=.25*width,right=.25*width,top=.25*height,bottom=.25*height)
        deficits = {k:max(0,desired[k]-margins[k]) for k in desired}
        shift = [actual[i]-nominal[i] for i in range(2)]
        geometry = dict(nominal_window_px=[*nominal,nominal[0]+side,nominal[1]+side],actual_window_px=window,
            shift_px=shift,side_px=side,margin_per_target_axis=.25,expanded_context_box_px=expanded,
            actual_margin_px=margins,donor_extent_bounding_rectangle_px=bounds,
            shift_policy="Uniform unavailable-source-edge fallback: preserve complete human box and fixed square scale, minimize context deficit then distance from nominal window")
        fallback.append((sum(deficits.values()),sum(s*s for s in shift),window,geometry,sources))
    if not fallback:
        raise ValueError("Even uniform context-edge fallback cannot cover full human box in fixed square without protected or missing pixels")
    _,_,window,geometry,sources = min(fallback,key=lambda r:(r[0],r[1],r[2][1],r[2][0]))
    return window, geometry, sources


def prepare(outmeta=META, dataset=DATASET):
    plan = C.verify_plan(SOURCE_PLAN)
    exports = C.read_json(EXPORT)
    manifest = C.read_json(plan["source_manifest"])
    if len(plan["rows"]) != 1083 or manifest["source_plan_sha256"] != C.sha256(EXPORT):
        raise ValueError("Require unchanged1083 cohort and frozen detector pixel provenance")
    outmeta = Path(outmeta).resolve()
    if outmeta.exists():
        previous = outmeta / "native_build_plan.json"
        if not previous.is_file() or C.read_json(previous)["status"] != "preflight_failed_no_pixel_build_allowed":
            raise FileExistsError("Preserve existing successful/frozen native plan")
        attempts = list(outmeta.glob("native_build_plan.failed_attempt_*.json"))
        archived = outmeta / f"native_build_plan.failed_attempt_{len(attempts)+1:02d}.json"
        shutil.copy2(previous, archived)
    else:
        outmeta = C.new_directory(outmeta)
    dataset = Path(dataset).resolve()
    if dataset.exists():
        raise FileExistsError("Prepare must not overwrite a dataset")
    pool, manifests = available_pool(exports)
    protected = protected_rectangles(exports)
    for sources in pool.values():
        for source in sources:
            path = Path(source["path"])
            if path.suffix.lower() == ".npz":
                source["npz_payload"] = npz_payload(path,source["size_px"])
                continue
            if path.suffix.lower() not in (".tif", ".tiff"):
                continue
            with tifffile.TiffFile(path) as tif:
                page = tif.pages[0]
                if page.dtype != np.uint8 or list(page.shape[::-1]) != source["size_px"]:
                    raise ValueError("Native donor header dtype/shape changed: " + str(path))
    samples = {s["stem"]: s for s in exports["samples"]}
    frozen = dict(schema_version=1, status="metadata_preflight_running_no_pixels_read", created_utc=utc(),
        dataset=str(dataset), source_plan=str(SOURCE_PLAN), source_plan_sha256=C.sha256(SOURCE_PLAN),
        source_protocol=plan["source_protocol"], source_protocol_sha256=plan["source_protocol_sha256"],
        source_manifest=plan["source_manifest"], source_manifest_sha256=plan["source_manifest_sha256"],
        export_plan=str(EXPORT), export_plan_sha256=C.sha256(EXPORT),
        script_sha256=C.sha256(__file__), geometry_helper_sha256=C.sha256(ROOT / "data/scripts/build_native_classification_crop_pilot.py"),
        historical_stretch_code_sha256=C.sha256(E.__file__), expected_master_plan=str(MASTER),
        cache_manifests=manifests, class_mapping=C.CLASS_TO_ID, unique_source_count=1083,
        protected_tiles=exports["protected_tiles"], protected_world_rectangles=protected,
        boundary_context_policy=dict(preferred="Complete human AABB plus >=25% of corresponding width/height each side in square approx1.5longside; minimum bounded shift",
            unavailable_source_edge="If preferred real context impossible, uniformly allow minimum-deficit bounded shift retaining ENTIRE human AABB and fixed square scale; explicit per-side deficit; no ID/class branch",
            never_allowed=["target trimming","invented background padding","protected evaluation source pixels","silent cohort drop"]),
        held_source_ids=plan["held_source_ids"], cohort_and_labels_unchanged=True,
        pixels_read=False, models_loaded=False, gpu_used=False, downloads=False,
        interpolation="direct full native square -> uint8 nonzero2/98 crop-local stretch ->224 PIL BILINEAR",
        rows=[], failures=[])
    write(outmeta / "native_build_plan.json", frozen)
    for i, old in enumerate(plan["rows"]):
        row = dict(source_id=old["source_id"], catalog_class=old["catalog_class"], class_id=old["class_id"],
            class_reviewed=old["class_reviewed"], old_crop=old["crop"], old_crop_sha256=old["crop_sha256"],
            old_padding_fraction=old["padding_fraction"], old_source_tiles=old["source_tiles"],
            old_fold_id=old["fold_id"], old_spatial_group_id=old["spatial_group_id"],
            source_stem=old["source_stem"], crop=str(dataset / "train" / old["catalog_class"] / f"slide_{int(old['source_id']):05d}.png"))
        try:
            sample = samples[old["source_stem"]]
            label = next(s for s in sample["labels"] if str(s["source_id"]) == old["source_id"])
            if label["class_name"] != old["catalog_class"]:
                raise ValueError("Subtype provenance differs")
            gt = sample["geotransform"] if "geotransform" in sample else exports["tile_sources"][sample["tile"]]["geotransform"]
            donors = donors_for(sample, gt, pool, label["manual_aabb_px"])
            window, geometry, sources = covered_window(label["manual_aabb_px"], gt, donors, protected)
            parent_sources = [s for s in donors if intersects(sample["window_px"], source_extent_in_anchor(gt, s))]
            parent = entry_from_window(gt, sample["window_px"], parent_sources)
            box = label["manual_aabb_px"]
            desired = dict(left=.25*(box[2]-box[0]),right=.25*(box[2]-box[0]),
                           top=.25*(box[3]-box[1]),bottom=.25*(box[3]-box[1]))
            actual = geometry["actual_margin_px"]
            deficit = {k:max(0,desired[k]-actual[k]) for k in desired}
            exception = any(x>1e-6 for x in deficit.values())
            row.update(status="complete_geometry_ready_for_spatial_audit", manual_aabb_px=label["manual_aabb_px"],
                manual_box_world_m=world_box(gt, label["manual_aabb_px"]), geotransform=gt,
                window_px=window, geometry=geometry, size_px=[window[2]-window[0], window[3]-window[1]],
                native_output_geotransform=entry_from_window(gt, window, sources)["geotransform"],
                native_window_world_m=world_box(gt, window), sources=sources,
                contributing_source_tiles=sorted({s["tile"] for s in sources}),
                same_old_fold_donors_only=False,
                new_donor_tiles_require_independent_spatial_audit=True,
                percentile_statistics_donor_scope="Entire new native square, including full context, not only target ROI",
                full_human_box_preserved=True, minimum_context_preserved=not exception,
                context_policy_exception=exception,
                context_exception_reason="unavailable_source_edge" if exception else None,
                context_deficit_px=deficit,actual_context_margin_px=actual,desired_context_margin_px=desired,
                complete_native_coverage=True, geometric_padding_fraction=0.0,
                parent_native_entry=parent, parent_native_sha256=sample.get("native_crop_sha256"),
                parent_native_verifiable=bool(sample.get("native_crop_sha256") and exact_coverage(parent)))
        except (ValueError, KeyError, StopIteration) as error:
            row.update(status="preflight_failed_do_not_drop_target", error=repr(error))
            frozen["failures"].append(dict(source_id=old["source_id"], error=repr(error)))
        frozen["rows"].append(row)
        if (i+1) % 50 == 0:
            write(outmeta / "native_build_plan.json", frozen)
            write(outmeta / "status.json", dict(status=frozen["status"], completed_metadata_rows=i+1, total=1083, updated_utc=utc()))
    frozen.update(status="metadata_ready_awaiting_frozen_spatial_audit" if not frozen["failures"] else "preflight_failed_no_pixel_build_allowed",
        finished_utc=utc(), bounded_shift_targets=sum(any(r.get("geometry", {}).get("shift_px", [])) for r in frozen["rows"]),
        context_policy_exception_targets=sum(r.get("context_policy_exception",False) for r in frozen["rows"]))
    path = outmeta / "native_build_plan.json"
    write(path, frozen)
    (outmeta / "native_build_plan.sha256").write_text(C.sha256(path)+"\n", encoding="utf-8")
    write(outmeta / "status.json", dict(status=frozen["status"], completed_metadata_rows=1083,
        failures=len(frozen["failures"]), bounded_shift_targets=frozen["bounded_shift_targets"], plan_sha256=C.sha256(path), updated_utc=utc()))
    print(json.dumps(dict(status=frozen["status"], plan=str(path), targets=1083, failures=len(frozen["failures"])), ensure_ascii=False), flush=True)
    return dict(plan=str(path), status=frozen["status"])


class NativeReader:
    """Retain memory-mapped raw TIFFs; LRU decoded caches have a512MiB budget."""
    def __init__(self, budget=512*1024*1024):
        self.cache = OrderedDict()
        self.budget = budget
        self.decoded_bytes = 0
        self.verified = {}

    def get(self, src):
        name = src["path"]
        if name in self.cache:
            arr, cost = self.cache.pop(name)
            self.cache[name] = arr, cost
            return arr
        path = Path(name)
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (src["bytes"], src["mtime_ns"]):
            raise ValueError("Native file changed: " + name)
        if name not in self.verified:
            if src.get("cache_sha256") and C.sha256(path) != src["cache_sha256"]:
                raise ValueError("Lossless cache SHA256 mismatch: " + name)
            if not src.get("cache_sha256"):
                gt, w, h = read_geotransform(path)
                if [w, h] != src["size_px"] or not np.allclose(gt, src["geotransform"], atol=1e-7, rtol=0):
                    raise ValueError("Raw native geometry mismatch")
            self.verified[name] = dict(path=name, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                cache_sha256=src.get("cache_sha256"), verified_utc=utc())
        try:
            arr = tifffile.memmap(path)
            cost = 0
        except (ValueError, tifffile.TiffFileError):
            wanted = src["size_px"][0]*src["size_px"][1]
            if wanted > self.budget:
                raise MemoryError("Single compressed native cache exceeds512MiB decode budget")
            while self.cache and self.decoded_bytes+wanted > self.budget:
                self.evict()
            if path.suffix.lower() == ".npz":
                with np.load(path, allow_pickle=False) as values:
                    arr = values[src.get("npz_payload") or npz_payload(path,src["size_px"])]
            else:
                arr = tifffile.imread(path)
            cost = arr.nbytes
        if arr.dtype != np.uint8 or list(arr.shape[::-1]) != src["size_px"]:
            raise ValueError("Native donor must retain exact uint8 grayscale source dimensions")
        while len(self.cache) >= 8:
            self.evict()
        self.cache[name] = arr, cost
        self.decoded_bytes += cost
        return arr

    def evict(self):
        _, (arr, cost) = self.cache.popitem(last=False)
        self.decoded_bytes -= cost
        if isinstance(arr, np.memmap):
            arr._mmap.close()

    def slabs(self, entry):
        w, h = entry["size_px"]
        gt = entry["geotransform"]
        xx = gt[0]+np.arange(w)*gt[1]
        for start in range(0, h, 128):
            end = min(start+128, h)
            yy = gt[3]+np.arange(start, end)*gt[5]
            block = np.zeros((end-start, w), dtype=np.uint8)
            coverage = np.zeros(block.shape, dtype=bool)
            used = {}
            for src in entry["sources"]:
                t = src["geotransform"]
                sw, sh = src["size_px"]
                sx, sy = np.rint((xx-t[0])/t[1]).astype(np.int64), np.rint((yy-t[3])/t[5]).astype(np.int64)
                ix, iy = np.flatnonzero((sx>=0)&(sx<sw)), np.flatnonzero((sy>=0)&(sy<sh))
                if not len(ix) or not len(iy):
                    continue
                region = np.ix_(iy, ix)
                missing = ~coverage[region]
                if not missing.any():
                    continue
                arr = self.get(src)
                pixels = np.asarray(arr[np.ix_(sy[iy], sx[ix])])
                old = block[region]
                old[missing] = pixels[missing]
                block[region] = old
                coverage[region] = True
                used[src["path"]] = int(missing.sum())
                if coverage.all():
                    break
            if not coverage.all():
                raise ValueError("Actual native pixels have a coverage hole")
            yield start, block, used

    def close(self):
        while self.cache:
            self.evict()


def histogram_lut(counts):
    # Same histogram, nonzero percentile interpolation and LUT as shared
    # export_manual_detection_dataset.stretch_uint8; applied slab-by-slab so
    # no full float64 native crop or redundant RAM canvas is allocated.
    counts = counts.copy()
    counts[0] = 0
    n = int(counts.sum())
    cumulative = np.cumsum(counts)
    def quantile(p):
        index = (n-1)*p
        low, high = math.floor(index), math.ceil(index)
        a = int(np.searchsorted(cumulative, low, side="right"))
        b = int(np.searchsorted(cumulative, high, side="right"))
        return a+(b-a)*(index-low)
    lo, hi = (quantile(.02), quantile(.98)) if n else (0, 255)
    if hi <= lo:
        hi = lo+1
    lut = np.clip((np.arange(256, dtype=np.float64)-lo)/(hi-lo)*255, 0, 255).astype(np.uint8)
    lut[0] = 0
    return lut, dict(p02=lo, p98=hi, nonzero_native_pixels=n,
        policy="historical shared crop-local nonzero2/98 linear-percentile uint8 LUT, rawzero retained")


def build(plan_path, master_plan_path=None):
    import msvcrt
    import psutil
    plan_path = Path(plan_path).resolve()
    frozen = C.read_json(plan_path)
    plan_sha = C.sha256(plan_path)
    if frozen["status"] != "metadata_ready_awaiting_frozen_spatial_audit" or frozen["failures"] or len(frozen["rows"]) != 1083:
        raise ValueError("Require complete1083 metadata; no silent cohort drop")
    master_path = Path(master_plan_path or frozen["expected_master_plan"])
    master = C.read_json(master_path)
    if master.get("native_build_plan_sha256") != plan_sha or master.get("native_spatial_audit_approved") is not True or master.get("status") != "locked_before_execution":
        raise ValueError("Await root frozen spatial audit bound to this exact native plan")
    for path, expected in [(Path(__file__), frozen["script_sha256"]),
        (Path(frozen["source_plan"]), frozen["source_plan_sha256"]),
        (Path(frozen["export_plan"]), frozen["export_plan_sha256"]),
        (Path(E.__file__), frozen["historical_stretch_code_sha256"]),
        (ROOT / "data/scripts/build_native_classification_crop_pilot.py", frozen["geometry_helper_sha256"])]:
        if C.sha256(path) != expected:
            raise ValueError("Frozen preparation code/source changed: " + str(path))
    for ref in frozen["cache_manifests"]:
        if C.sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("Frozen live cache metadata changed")
    dataset = Path(frozen["dataset"]).resolve()
    if not dataset.is_relative_to(ROOT / "data/processed"):
        raise ValueError("Native dataset must remain inside processed workspace")
    if dataset.exists():
        previous = C.read_json(dataset / "status.json")
        if previous["native_build_plan_sha256"] != plan_sha:
            raise FileExistsError("Existing native dataset belongs to another frozen plan")
        if previous["status"] == "completed1083_native_train_crops_same_ids_labels":
            complete_path = dataset / "manifest.json"
            complete = C.read_json(complete_path)
            if complete["native_build_plan_sha256"] != plan_sha or len(complete["samples"]) != 1083 or C.sha256(complete_path) != previous["manifest_sha256"]:
                raise ValueError("Completed native manifest differs; preserve outputs")
            for actual, expected in zip(complete["samples"],frozen["rows"]):
                if actual["source_id"] != expected["source_id"] or actual["catalog_class"] != expected["catalog_class"] or C.sha256(actual["crop"]) != actual["crop_sha256"]:
                    raise ValueError("Completed native crop cohort or bytes changed")
            return dict(status=complete["status"],manifest=str(complete_path),targets=1083)
    else:
        dataset.mkdir(parents=True)
    lock = (dataset / "build.lock").open("a+b")
    if lock.seek(0,2) == 0:
        lock.write(b"\0")
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        lock.close()
        raise RuntimeError("Another process already owns the native dataset build")
    status = dict(status="building_native_crops", started_utc=utc(), pid=os.getpid(), native_build_plan_sha256=plan_sha,
        master_plan_sha256=C.sha256(master_path), completed_targets=0, total_targets=1083,
        gpu_used=False, models_loaded=False, downloads=False, source_labels_modified=False)
    receipts_dir, scratch = dataset / "receipts", dataset / "_scratch"
    receipts_dir.mkdir(exist_ok=True)
    scratch.mkdir(exist_ok=True)
    reader, rows, parent_hashes = NativeReader(), [], {}
    try:
        write(dataset / "status.json", status)
        for i, row in enumerate(frozen["rows"]):
            dest = Path(row["crop"])
            receipt_path = receipts_dir / f"slide_{int(row['source_id']):05d}.json"
            if receipt_path.exists():
                receipt = C.read_json(receipt_path)
                if receipt["native_build_plan_sha256"] != plan_sha or receipt["row_sha256"] != hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest() or C.sha256(dest) != receipt["crop_sha256"]:
                    raise ValueError("Existing target receipt differs; preserve prior output")
                rows.append(receipt)
            else:
                if dest.exists():
                    raise FileExistsError("Unreceipted final PNG preserved; inspect before retry: " + str(dest))
                if row["parent_native_verifiable"]:
                    stem = row["source_stem"]
                    if stem not in parent_hashes:
                        digest = hashlib.sha256()
                        for _, block, _ in reader.slabs(row["parent_native_entry"]):
                            digest.update(block.tobytes())
                        parent_hashes[stem] = digest.hexdigest()
                    if parent_hashes[stem] != row["parent_native_sha256"]:
                        raise ValueError("Native replacement donors differ from frozen parent pixel hash: " + stem)
                w, h = row["size_px"]
                required_scratch = 2*w*h+256*1024*1024
                if shutil.disk_usage(dataset).free < required_scratch:
                    raise OSError("Insufficient free disk for two bounded native scratch maps plus256MiB reserve; outputs preserved")
                token = f"{row['source_id']}_{uuid.uuid4().hex}"
                raw_path, stretch_path = scratch / (token+"_raw.npy"), scratch / (token+"_stretched.npy")
                raw = stretched = image = None
                try:
                    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.uint8, shape=(h, w))
                    counts, digest, donor_pixels = np.zeros(256, dtype=np.int64), hashlib.sha256(), {}
                    entry = dict(geotransform=row["native_output_geotransform"], size_px=row["size_px"], sources=row["sources"])
                    for start, block, used in reader.slabs(entry):
                        raw[start:start+len(block)] = block
                        counts += np.bincount(block.ravel(), minlength=256)
                        digest.update(block.tobytes())
                        for path, n in used.items():
                            donor_pixels[path] = donor_pixels.get(path, 0)+n
                    raw.flush()
                    lut, brightness = histogram_lut(counts)
                    brightness["raw_zero_fraction"] = int(counts[0])/(w*h)
                    stretched = np.lib.format.open_memmap(stretch_path, mode="w+", dtype=np.uint8, shape=(h, w))
                    for start in range(0, h, 128):
                        stretched[start:start+128] = lut[raw[start:start+128]]
                    stretched.flush()
                    image = Image.frombuffer("L", (w, h), stretched, "raw", "L", 0, 1)
                    direct224 = image.resize((224, 224), Image.Resampling.BILINEAR)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp_png = dest.with_name(dest.stem+"_"+uuid.uuid4().hex+".tmp.png")
                    direct224.save(tmp_png)
                    tmp_png.replace(dest)
                    receipt = dict(row, status="native_direct224_complete", crop_sha256=C.sha256(dest),
                        native_build_plan_sha256=plan_sha, row_sha256=hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
                        native_crop_sha256=digest.hexdigest(), parent_native_hash_verified=row["parent_native_verifiable"],
                        brightness_normalization=brightness, actual_donor_pixel_counts=donor_pixels,
                        actual_donor_tiles=sorted({s["tile"] for s in row["sources"] if s["path"] in donor_pixels}),
                        source_file_verification=[reader.verified[p] for p in donor_pixels],
                        output_size=[224, 224], output_mode="L", geometric_padding_fraction=0.0,
                        output_black_pixel_fraction=float(np.mean(np.asarray(direct224)==0)), finished_utc=utc())
                    write(receipt_path, receipt)
                    rows.append(receipt)
                finally:
                    if image is not None:
                        image.close()
                    for array in (stretched, raw):
                        if array is not None:
                            array._mmap.close()
                    for path in (raw_path, stretch_path):
                        if path.exists() and path.resolve().parent == scratch.resolve():
                            path.unlink()
            status.update(completed_targets=i+1, current_source_id=row["source_id"], updated_utc=utc(),
                distinct_native_files_verified=len(reader.verified), decoded_cache_bytes=reader.decoded_bytes)
            memory = psutil.Process().memory_info()
            status.update(rss_bytes=memory.rss, peak_rss_bytes=getattr(memory, "peak_wset", memory.rss))
            write(dataset / "status.json", status)
            print(json.dumps(status), flush=True)
        manifest = dict(status="completed1083_native_train_crops_same_ids_labels", created_utc=utc(),
            native_build_plan=str(plan_path), native_build_plan_sha256=plan_sha,
            master_plan=str(master_path.resolve()), master_plan_sha256=C.sha256(master_path),
            source_plan_sha256=frozen["source_plan_sha256"], source_manifest_sha256=frozen["source_manifest_sha256"],
            source_ids=[r["source_id"] for r in rows], targets=1083, crop_size=224,
            classes=frozen["class_mapping"], held_source_ids=frozen["held_source_ids"],
            full_human_boxes_preserved=True, geometric_padding_fraction=0.0,
            label_source="unchanged unreviewed catalog; no subtype relabeling", samples=rows,
            native_file_verification=list({v["path"]:v for r in rows for v in r.get("source_file_verification",[])}.values()), gpu_used=False, models_loaded=False,
            downloads=False, independent_test_claim=False,
            comparison_scope="Native crop pipeline versus old padded detector-PNG pipeline: context/percentile/resampling jointly differ")
        write(dataset / "manifest.json", manifest)
        status.update(status="completed1083_native_train_crops_same_ids_labels", finished_utc=utc(), manifest_sha256=C.sha256(dataset / "manifest.json"))
        return dict(status=status["status"], manifest=str(dataset / "manifest.json"), targets=1083)
    except BaseException as error:
        status.update(status="failed_resumable_preserved_outputs", error_type=type(error).__name__, error=str(error), finished_utc=utc())
        raise
    finally:
        reader.close()
        write(dataset / "status.json", status)
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "build"))
    parser.add_argument("--out", type=Path, default=META)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--plan", type=Path, default=META / "native_build_plan.json")
    parser.add_argument("--master-plan", type=Path, default=MASTER)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.out.resolve(), args.dataset.resolve())
    else:
        print(json.dumps(build(args.plan, args.master_plan), ensure_ascii=False))
