"""Protect reviewed positives from held neighbors and annotation drift."""
import csv
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import shapefile
import tifffile

SCRIPTS = Path(__file__).resolve().parents[1] / "data/scripts"
sys.path.insert(0, str(SCRIPTS))
m = importlib.import_module("build_center_candidate_manifest")


def rec(sid, box, cls="SL"):
    return {"source_id": sid, "aabb_px": box, "raw_type": cls}


def review(sid, status="accept", box=(10, 10, 40, 40), origin=(0, 0, 100, 100)):
    return {"tile": "E000_N00", "source_id": str(sid), "review_status": status,
            "review_notes": "human review", "window_origin": json.dumps(origin),
            "native_window_px": json.dumps([origin[2]-origin[0], origin[3]-origin[1]]),
            "polygon_bbox_px": json.dumps(box), "center_poly_file": "poly.png",
            "center_aabb_file": "aabb.png"}


def write_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("status", sorted(m.HOLD))
def test_held_neighbor_is_never_a_positive_or_implicit_background(status):
    records = [rec(1, [10, 10, 40, 40]), rec(2, [50, 50, 80, 80], "CL")]
    reviews = {("E000_N00", 1): review(1), ("E000_N00", 2): review(2, status)}
    c = m.make_candidate("E000_N00", records[0], records, reviews, {})
    assert [label["source_id"] for label in c["labels"]] == [1]
    assert c["excluded_regions"][0]["reason"] == status
    assert not c["eligible_for_standard_yolo_export"]


def test_all_accepted_complete_neighbors_are_labeled_with_classes():
    records = [rec(1, [10, 10, 40, 40]), rec(2, [50, 50, 80, 80], "CL")]
    reviews = {("E000_N00", sid): review(sid) for sid in (1, 2)}
    c = m.make_candidate("E000_N00", records[0], records, reviews, {})
    assert [label["class_id"] for label in c["labels"]] == [0, 3]
    assert all(label["stage1_class_id"] == 0 for label in c["labels"])
    assert c["eligible_for_standard_yolo_export"]


def test_explicit_exclusion_overrides_saved_box_and_blocks_background_export():
    records = [rec(1, [10, 10, 40, 40]), rec(2, [50, 50, 80, 80])]
    reviews = {("E000_N00", 1): review(1), ("E000_N00", 2): {**review(2),
        "explicit_exclusion": True, "exclusion_box_px": [45, 45, 95, 95]}}
    overrides = {("E000_N00", 2): {"aabb_px": [45, 45, 85, 85], "label_scope": "full_landslide"}}
    box, reason, provenance = m.label_decision("E000_N00", records[1], reviews, overrides)
    assert box == [45, 45, 95, 95] and reason == "reject"
    assert provenance == "explicit_human_exclusion"
    candidate = m.make_candidate("E000_N00", records[0], records, reviews, overrides)
    assert [label["source_id"] for label in candidate["labels"]] == [1]
    assert not candidate["eligible_for_standard_yolo_export"]


def test_catalog_alias_is_normalized_without_losing_original_shp_type():
    records = [rec(1, [10, 10, 40, 40], "DF"), rec(2, [50, 50, 80, 80], "RL")]
    reviews = {("E000_N00", sid): review(sid) for sid in (1, 2)}
    c = m.make_candidate("E000_N00", records[0], records, reviews, {})
    assert [label["class_id"] for label in c["labels"]] == [2, 0]
    assert [label["class_name"] for label in c["labels"]] == ["FL", "SL"]
    assert [label["shp_raw_type"] for label in c["labels"]] == ["DF", "RL"]


def test_unreviewed_and_truncated_accepted_neighbors_block_export():
    records = [rec(1, [10, 10, 40, 40]), rec(2, [90, 40, 130, 80]),
               rec(3, [50, 50, 80, 80])]
    reviews = {("E000_N00", 1): review(1), ("E000_N00", 2): review(2)}
    c = m.make_candidate("E000_N00", records[0], records, reviews, {})
    assert [r["reason"] for r in c["excluded_regions"]] == [
        "accepted_neighbor_truncated_in_this_crop", "unreviewed_neighbor"]
    assert c["unreviewed_neighbor_ids"] == [3]
    assert not c["eligible_for_standard_yolo_export"]


def test_exact_review_rectangle_is_used_not_default_centered_window():
    record = rec(1, [60, 30, 100, 50])
    row = review(1, origin=(50, 20, 150, 70))
    c = m.make_candidate("E000_N00", record, [record], {("E000_N00", 1): row}, {})
    assert c["window_origin"] == [50, 20, 150, 70]
    assert c["labels"][0]["xywh"] == [0.3, 0.4, 0.4, 0.4]


def test_full_manual_box_replaces_aabb_but_visible_core_stays_separate():
    record = rec(1, [10, 10, 40, 40])
    reviews = {("E000_N00", 1): review(1, "needs_expand")}
    override = {"aabb_px": [5, 5, 45, 45], "label_scope": "full_landslide"}
    c = m.make_candidate("E000_N00", record, [record], reviews,
                         {("E000_N00", 1): override})
    assert c["labels"][0]["aabb_tile_px"] == [5, 5, 45, 45]
    assert c["labels"][0]["label_origin"] == "confirmed_manual_aabb"
    override["label_scope"] = "visible_core"
    c = m.make_candidate("E000_N00", record, [record], reviews,
                         {("E000_N00", 1): override})
    assert not c["labels"]
    assert c["excluded_regions"][0]["reason"] == "visible_core_separate_task"
    assert c["excluded_regions"][0]["aabb_tile_px"] == [5, 5, 45, 45]


@pytest.mark.parametrize("bad_box", [[0, 0, 0, 1], [2, 0, 1, 1], [0, 0, float("nan"), 1]])
def test_invalid_manual_geometry_fails(bad_box):
    with pytest.raises(ValueError, match="invalid box"):
        m.validate_box(bad_box)


def test_duplicate_reviews_and_window_mismatch_fail(tmp_path):
    path = tmp_path / "review.csv"
    write_csv(path, [review(1), review(1)])
    with pytest.raises(ValueError, match="duplicate"):
        m.load_review(path)
    row = review(1)
    row["native_window_px"] = "[99, 100]"
    write_csv(path, [row])
    with pytest.raises(ValueError, match="window size mismatch"):
        m.load_review(path)


def test_manual_confirmation_and_author_are_required(tmp_path):
    path = tmp_path / "override.csv"
    row = dict(zip(m.OVERRIDE_FIELDS, ["E000_N00", 1, 1, 1, 50, 50,
                                      "full_landslide", "", "false", "include headwall"]))
    write_csv(path, [row])
    assert m.load_overrides(path, {("E000_N00", 1)}) == {}
    row["confirmed"] = "true"
    write_csv(path, [row])
    with pytest.raises(ValueError, match="reviewer and reason"):
        m.load_overrides(path, {("E000_N00", 1)})
    row["reviewer"] = "owner"
    write_csv(path, [row])
    assert m.load_overrides(path, {("E000_N00", 1)})[("E000_N00", 1)]["aabb_px"] == [1, 1, 50, 50]


def test_protected_or_non_train_plan_fails():
    with pytest.raises(ValueError, match="protected"):
        m.validate_plan({"E000_N00": "train"}, {"E000_N00"})
    with pytest.raises(ValueError, match="train-only"):
        m.validate_plan({"E000_N00": "val"}, set())


@pytest.fixture
def dataset(tmp_path):
    shp = tmp_path / "slides.shp"
    k = 3396190 * np.pi / 180
    with shapefile.Writer(str(shp)) as writer:
        writer.field("source_id", "N")
        writer.field("type", "C")
        for sid, cls, box in [(1, "SL", [10, 10, 40, 40]), (2, "CL", [50, 50, 80, 80])]:
            x0, y0, x1, y1 = box
            points = [(x0/k, (100-y0)/k), (x1/k, (100-y0)/k),
                      (x1/k, (100-y1)/k), (x0/k, (100-y1)/k), (x0/k, (100-y0)/k)]
            writer.poly([points])
            writer.record(sid, cls)
    ctx = tmp_path / "ctx"
    tile = ctx / "E000_N00"
    tile.mkdir(parents=True)
    tifffile.imwrite(tile / "MurrayLab_CTX_V01_E000_N00.tif", np.zeros((100, 100), dtype=np.uint8),
                      extratags=[(33550, "d", 3, (1, 1, 0), False),
                                 (33922, "d", 6, (0, 0, 0, 0, 100, 0), False)])
    csv_path = tmp_path / "review.csv"
    write_csv(csv_path, [review(1), review(2, "boundary_wrong", box=[50, 50, 80, 80])])
    plan = tmp_path / "plan.json"
    plan.write_text('{"E000_N00":"train"}', encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text('{"E008_N08":"val"}', encoding="utf-8")
    final = tmp_path / "final.json"
    final.write_text('{"tiles":["E012_N12"]}', encoding="utf-8")
    return csv_path, plan, ctx, shp, tmp_path / "output.json", split, final


def run_build(dataset, overrides=None):
    csv_path, plan, ctx, shp, out, split, final = dataset
    return m.build(csv_path, plan, ctx, shp, out, overrides, split, final)


def test_end_to_end_preserves_sources_and_is_reproducible(dataset):
    csv_path, plan, ctx, shp, out, split, final = dataset
    inputs = [csv_path, plan, split, final, shp, shp.with_suffix(".shx"), shp.with_suffix(".dbf")]
    before = {path: m.sha256_file(path) for path in inputs}
    p = run_build(dataset)
    assert p["candidate_count"] == 1 and p["held_count"] == 1
    assert p["standard_yolo_eligible_count"] == 0
    assert p["positive_label_occurrences"] == 1
    assert p["candidates"][0]["excluded_regions"][0]["source_id"] == 2
    assert json.loads(out.read_text(encoding="utf-8"))["manifest_sha256"] == p["manifest_sha256"]
    assert {path: m.sha256_file(path) for path in inputs} == before
    assert run_build(dataset)["manifest_sha256"] == p["manifest_sha256"]


def test_annotation_drift_fails_before_output(dataset):
    csv_path, _, _, _, out, _, _ = dataset
    rows = m.load_review(csv_path)
    rows[0]["polygon_bbox_px"] = "[10, 10, 41, 40]"
    write_csv(csv_path, rows)
    with pytest.raises(ValueError, match="geometry changed"):
        run_build(dataset)
    assert not out.exists()


@pytest.mark.parametrize("remove_pixels", [False, True])
def test_retired_tile_uses_native_cache_and_keeps_neighbor_holds(dataset, remove_pixels):
    csv_path, _, ctx, _, _, _, _ = dataset
    cache = csv_path.parent / "native_context" / "E000_N00"
    cache.mkdir(parents=True)
    source = next(ctx.glob("*/*.tif"))
    native = cache / "native.tif"
    source.rename(native)
    manifest = {"tile": "E000_N00", "geotransform": [0, 1, 0, 100, 0, -1],
                "tile_width": 100, "tile_height": 100, "original": {"shape": [100, 100]},
                "entries": [{"source_id": sid, "stride": 1, "path": str(native),
                             "window_origin": [0, 0, 100, 100]} for sid in (1, 2)]}
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if remove_pixels:
        native.unlink()
        with pytest.raises(ValueError, match="missing native cache pixels"):
            run_build(dataset)
        return
    payload = run_build(dataset)
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["excluded_regions"][0]["source_id"] == 2
    assert not payload["candidates"][0]["eligible_for_standard_yolo_export"]
    assert "native_context_E000_N00" in payload["inputs"]


def test_repair_template_never_overwrites_human_edit(tmp_path):
    path = tmp_path / "manual.csv"
    payload = {"held": [{"tile": "E000_N00", "source_id": 1, "review_status": "needs_expand",
                         "review_notes": "headwall missing"}]}
    m.write_repair_template(payload, path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        m.write_repair_template(payload, path)
    assert path.read_bytes() == before
