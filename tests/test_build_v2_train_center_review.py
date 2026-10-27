import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data/scripts"))
m = importlib.import_module("build_v2_train_center_review")


def row(box, window, complete=True):
    return {"aabb_complete": str(complete), "polygon_complete": str(complete),
            "polygon_bbox_px": json.dumps(box), "window_origin": json.dumps(window)}


def test_complete_target_requires_full_window_coverage():
    assert m.geometric_status(row([10, 20, 100, 200], [0, 0, 120, 220])) == "complete"
    assert m.geometric_status(row([10, 20, 100, 200], [0, 0, 120, 150])) == "window_truncated"


def test_missing_tile_pixels_are_distinct_from_window_truncation():
    assert m.geometric_status(row([-10, 20, 100, 200], [0, 0, 120, 220], False)) == "tile_truncated"


def test_save_json_updates_only_its_output(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("human review", encoding="utf-8")
    output = tmp_path / "status.json"
    m.save_json(output, {"count": 1})
    m.save_json(output, {"count": 2})
    assert json.loads(output.read_text(encoding="utf-8")) == {"count": 2}
    assert source.read_text(encoding="utf-8") == "human review"


def test_large_center_target_is_preserved_while_preview_is_sampled(monkeypatch, tmp_path):
    record = {"source_id": 1, "raw_type": "SL", "aabb_px": [4000, 4000, 18000, 18000],
              "polygon_px": [[(4000, 4000), (18000, 4000), (18000, 18000), (4000, 18000)]]}
    class Reader:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def shapeRecords(self): return []
    class Raster:
        def __getitem__(self, slices):
            y, x = slices
            return np.zeros((len(range(y.start, y.stop, y.step)), len(range(x.start, x.stop, x.step))), dtype=np.uint8)
    rendered = []
    monkeypatch.setattr(m, "discover_tifs", lambda _: {"E000_N00": "fake.tif"})
    monkeypatch.setattr(m, "read_geotransform", lambda _: ((0, 5, 0, 110000, 0, -5), 22000, 22000))
    monkeypatch.setattr(m.shapefile, "Reader", lambda _: Reader())
    monkeypatch.setattr(m, "map_source_records", lambda _: {1: object()})
    monkeypatch.setattr(m, "_record_for_tile", lambda *args: record)
    monkeypatch.setattr(m.tifffile, "memmap", lambda _: Raster())
    monkeypatch.setattr(m, "render_view", lambda crop, recs, row, view, path: rendered.append((crop.shape, recs, row)))
    result = m.build_complete_pack("E000_N00", tmp_path)
    row = m.read_csv(tmp_path / "centered_review_manifest.csv")[0]
    assert result["count"] == 1
    assert json.loads(row["native_window_px"]) == [21000, 21000]
    assert m.geometric_status(row) == "complete"
    assert int(row["preview_downsample_stride"]) == 6
    assert all(shape == (3500, 3500) for shape, _, _ in rendered)
    assert row["review_status"] == ""


def test_finalize_merges_old_and_sampled_rows_without_overwriting_human_review(monkeypatch, tmp_path):
    import csv
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    (raw / "mars_landslides.json").write_text(json.dumps([
        {"source_id": 1, "type": "SL"}, {"source_id": 2, "type": "FL"}]), encoding="utf-8")
    plan, final = raw / "plan.json", raw / "final.json"
    plan.write_text('{"E000_N00":"train","E004_N00":"train"}', encoding="utf-8")
    final.write_text('{"tiles":[]}', encoding="utf-8")
    class Reader:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def shapeRecords(self): return []
    class Preview:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def thumbnail(self, *args): pass
        def convert(self, *args): return self
        def save(self, p, **kw): Path(p).write_bytes(b"preview")
    records = {1: SimpleNamespace(record=[1, "SL"], shape=SimpleNamespace(bbox=[0,0,0.1,0.1])),
               2: SimpleNamespace(record=[2, "DF"], shape=SimpleNamespace(bbox=[4,0,4.1,0.1]))}
    monkeypatch.setattr(m, "BASE", tmp_path)
    monkeypatch.setattr(m, "PLAN", plan)
    monkeypatch.setattr(m, "FINAL", final)
    monkeypatch.setattr(m.shapefile, "Reader", lambda _: Reader())
    monkeypatch.setattr(m, "map_source_records", lambda _: records)
    monkeypatch.setattr(m.Image, "open", lambda _: Preview())
    out = tmp_path / "review"
    for sid, tile in ((1,"E000_N00"), (2,"E004_N00")):
        folder = out / "tiles" / tile
        folder.mkdir(parents=True)
        data = {"tile":tile, "source_id":str(sid), "center_poly_file":"poly.png", "center_aabb_file":"aabb.png",
                "window_origin":"[0,0,100,100]", "native_window_px":"[100,100]", "polygon_bbox_px":"[10,10,90,90]",
                "aabb_complete":"True", "polygon_complete":"True", "review_status":"", "review_notes":""}
        if sid == 2: data.update(center_px="[50,50]", preview_downsample_stride="6", preview_size_px="[17,17]")
        with (folder / "centered_review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data))
            writer.writeheader(); writer.writerow(data)
    summary = m.finalize(out, ["E000_N00","E004_N00"], {})
    assert summary["source_class_counts"] == {"SL":1,"FL":1}
    master = out / "centered_review_manifest.csv"
    text = master.read_text(encoding="utf-8-sig") + "\n"
    master.write_text(text, encoding="utf-8-sig")
    before = master.read_bytes()
    m.finalize(out, ["E000_N00","E004_N00"], {})
    assert master.read_bytes() == before
