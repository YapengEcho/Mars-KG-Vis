import csv
import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data/scripts"))
m = importlib.import_module("manual_box_review")


@pytest.mark.parametrize("row, expected", [
    ({"aabb_complete": "True", "polygon_complete": "True"}, True),
    ({"aabb_complete": "False", "polygon_complete": "False"}, False),
    ({"aabb_complete": "True", "polygon_complete": "False"}, False),
    ({"aabb_complete": "False", "polygon_complete": "True"}, False),
    ({"aabb_complete": "True"}, False),
])
def test_full_geometry_queue_holds_truncated_or_unknown_references(row, expected):
    assert m.has_complete_geometry(row) is expected


def test_context_expansion_retains_available_target_near_tile_edge():
    assert m.context_window([900, 900, 990, 990], 400, 1000, 1000) == [600, 600, 1000, 1000]
    assert m.context_window([900, 900, 990, 990], 4000, 1000, 1000) == [0, 0, 1000, 1000]


def payload(**kw):
    return {"source_id": 1, "box": [10, 10, 60, 80], "confirmed": True,
            "label_scope": "full_landslide", "reviewer": "test", "reason": "bottom missing", **kw}


@pytest.mark.parametrize("bad", [payload(confirmed=False), payload(box=[-1, 10, 60, 80]),
    payload(box=[90, 90, 100, 100]), payload(source_id=2), payload(reviewer="")])
def test_unconfirmed_wrong_target_or_invalid_geometry_cannot_be_saved(bad):
    with pytest.raises(ValueError):
        m.validate_manual(bad, {1: {"aabb_px": [20, 20, 40, 40]}}, 100, 100)


def test_corrected_box_can_extend_beyond_old_review_crop():
    sid, box = m.validate_manual(payload(box=[10, 10, 60, 99]),
        {1: {"aabb_px": [20, 20, 40, 40]}}, 100, 100)
    assert sid == 1 and box == [10, 10, 60, 99]


def test_revisions_preserve_other_sources_and_csv_reader_compatibility(tmp_path):
    path = tmp_path / "manual.csv"
    def record(sid, bottom):
        return {"tile": "E000_N00", "source_id": str(sid), "x0": 10, "y0": 10,
            "x1": 60, "y1": bottom, "label_scope": "full_landslide",
            "reviewer": "test", "confirmed": "true", "reason": "bottom missing"}
    m.write_correction(path, record(1, 80)); m.write_correction(path, record(2, 90))
    m.write_correction(path, record(1, 99))
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {r["source_id"]: r["y1"] for r in rows} == {"1": "99", "2": "90"}
    from build_center_candidate_manifest import load_overrides
    overrides = load_overrides(path, {("E000_N00", 1), ("E000_N00", 2)})
    assert overrides[("E000_N00", 1)]["aabb_px"] == [10, 10, 60, 99]


def test_multi_tile_same_source_id_keeps_separate_saved_boxes(tmp_path, monkeypatch):
    class FakeReview(m.Review):
        def __init__(self, tile, ids, pack, ctx=None):
            self.tile = tile
            self.folder = pack
            self.corrections = pack / "manual.csv"
            self.rows = {sid: {} for sid in ids}
            self.allowed = {sid: {"aabb_px": [20, 20, 40, 40]} for sid in ids}
            self.width = self.height = 100

    monkeypatch.setattr(m, "Review", FakeReview)
    batch = m.ReviewBatch([{"tile": "A", "source_id": 1}, {"tile": "B", "source_id": 1}], tmp_path)
    batch.save(payload(tile="A", box=[10, 10, 60, 80]))
    batch.save(payload(tile="B", box=[15, 15, 65, 90]))
    saved = batch.batch_info()["saved"]
    assert set(saved) == {"A:1", "B:1"}
    assert float(saved["A:1"]["y1"]) == 80
    assert float(saved["B:1"]["y1"]) == 90
    assert batch.select("A").lock is batch.select("B").lock
    with pytest.raises(ValueError):
        batch.save(payload(tile="C"))
    assert batch.batch_info()["saved"] == saved


def test_multi_tile_batch_rejects_duplicate_or_empty_selections():
    with pytest.raises(ValueError):
        m.ReviewBatch([])
    with pytest.raises(ValueError):
        m.ReviewBatch([{"tile": "A", "source_id": 1}, {"tile": "A", "source_id": 1}])


@pytest.fixture
def decision_batch(tmp_path, monkeypatch):
    class FakeReview(m.Review):
        def __init__(self, tile, ids, pack, ctx=None):
            self.tile = tile
            self.folder = pack / "manual_review"
            self.folder.mkdir(exist_ok=True)
            self.corrections = self.folder / "manual_boxes.reviewed.csv"
            self.rows = {sid: {} for sid in ids}
            self.allowed = {sid: {"aabb_px": [20, 20, 40, 40]} for sid in ids}
            self.width = self.height = 100
    monkeypatch.setattr(m, "Review", FakeReview)
    items = [{"tile": "A", "source_id": 1}, {"tile": "B", "source_id": 1}]
    with (tmp_path / "centered_review_manifest.draft.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tile", "source_id", "review_status", "review_notes"])
        writer.writeheader()
        writer.writerows({**item, "review_status": "", "review_notes": ""} for item in items)
    return m.ReviewBatch(items, tmp_path)


def test_exclusion_without_box_is_persistent_and_never_a_negative(decision_batch):
    batch = decision_batch
    record = batch.exclude({"tile": "A", "source_id": 1, "reviewer": "human", "reason": "unclear"})
    assert record["negative_eligible"] is False
    assert record["training_use"] == "exclude_positive"
    assert batch.batch_info()["saved"] == {}
    assert set(batch.batch_info()["excluded"]) == {"A:1"}
    reloaded = m.ReviewBatch(batch.items, batch.pack)
    assert reloaded.batch_info()["excluded"]["A:1"]["reason"] == "unclear"
    with (batch.pack / "centered_review_manifest.draft.csv").open(encoding="utf-8-sig", newline="") as handle:
        draft = {r["tile"]: r for r in csv.DictReader(handle)}
    assert draft["A"]["review_status"] == "reject"
    assert draft["B"]["review_status"] == ""


def test_exclusion_invalidates_prior_box_and_confirmed_resave_can_restore(decision_batch):
    batch = decision_batch
    batch.save(payload(tile="A")); batch.save(payload(tile="B"))
    batch.exclude({"tile": "A", "source_id": 1, "reviewer": "human", "reason": "unclear"})
    assert set(batch.batch_info()["saved"]) == {"B:1"}
    from build_center_candidate_manifest import load_overrides
    assert set(load_overrides(batch.corrections, {("A", 1), ("B", 1)})) == {("B", 1)}
    assert batch.exclusions()["A:1"]["exclusion_box_px"] == [10, 10, 60, 80]
    batch.save(payload(tile="A", box=[15, 15, 60, 90]))
    assert batch.batch_info()["excluded"] == {}
    assert set(batch.batch_info()["saved"]) == {"A:1", "B:1"}
    history = [json.loads(line) for line in (batch.folder / "review_decision_history.jsonl").read_text().splitlines()]
    assert history[-1]["action"] == "restore_positive"


@pytest.mark.parametrize("bad", [
    {"tile": "C", "source_id": 1, "reviewer": "human", "reason": "unclear"},
    {"tile": "A", "source_id": 9, "reviewer": "human", "reason": "unclear"},
    {"tile": "A", "source_id": 1, "reviewer": "", "reason": "unclear"},
    {"tile": "A", "source_id": 1, "reviewer": "human", "reason": ""},
])
def test_invalid_exclusion_cannot_write_a_decision(decision_batch, bad):
    with pytest.raises(ValueError):
        decision_batch.exclude(bad)
    assert decision_batch.exclusions() == {}


def test_exclusion_http_endpoint_persists_decision(decision_batch):
    from urllib.request import Request, urlopen
    server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.handler_for(decision_batch))
    worker = m.threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(url+"/api/exclude", data=json.dumps({"tile": "A", "source_id": 1,
            "reviewer": "human", "reason": "unclear"}).encode(),
            headers={"Content-Type": "application/json", "X-Manual-Review": "1", "Origin": url}, method="POST")
        with urlopen(request, timeout=3) as response:
            assert json.load(response)["excluded"]["negative_eligible"] is False
        with urlopen(url+"/api/batch", timeout=3) as response:
            assert set(json.load(response)["excluded"]) == {"A:1"}
    finally:
        server.shutdown(); server.server_close(); worker.join(timeout=3)


@pytest.fixture
def background_batch(decision_batch, monkeypatch):
    import continue_ctx_review_download
    monkeypatch.setattr(continue_ctx_review_download, "protected_tiles", lambda: set())
    for review in decision_batch.reviews.values():
        review.records = {1: {"aabb_px": [20, 20, 40, 40]},
                          9: {"aabb_px": [80, 80, 100, 100]}}
        review.view = lambda sid, factor: ({"window": [0, 0, 100, 100]}, b"")
    return decision_batch


def background_payload(**kw):
    return {"tile": "A", "source_id": 1, "box": [45, 45, 65, 65],
            "view_factor": 4, "confirmed": True, "reviewer": "human",
            "reason": "entire region is dune field without landslide morphology",
            "background_kind": "dune", **kw}


def test_background_regions_persist_without_changing_positive_progress(background_batch):
    b = background_batch
    before = (b.pack / "centered_review_manifest.draft.csv").read_bytes()
    first = b.save_background(background_payload())
    second = b.save_background(background_payload(box=[45, 5, 65, 15]))
    assert first["background_id"] != second["background_id"]
    assert first["training_gate"] == "pending_recrop_and_overlap_recheck"
    assert first["judgement"] == "certain_background"
    assert b.batch_info()["saved"] == b.batch_info()["excluded"] == {}
    assert len(b.batch_info()["backgrounds"]) == 2
    assert (b.pack / "centered_review_manifest.draft.csv").read_bytes() == before
    assert not b.corrections.exists()
    again = m.ReviewBatch(b.items, b.pack)
    assert len(again.backgrounds()) == 2
    b.save_background(background_payload())
    assert len(b.backgrounds()) == 2  # retrying the same box never duplicates it
    b.remove_background({"tile": "A", "background_id": first["background_id"]})
    assert len(b.backgrounds()) == 1
    history = [json.loads(line) for line in (b.folder / "manual_background_history.jsonl").read_text().splitlines()]
    assert history[-1]["action"] == "remove"


@pytest.mark.parametrize("bad", [
    background_payload(box=[10, 10, 30, 30]),  # original target
    background_payload(box=[85, 85, 95, 95]),  # neighboring, unreviewed slide
    background_payload(box=[-1, 50, 5, 70]),
    background_payload(box=[45, 45, 45, 65]),
    background_payload(box=[45, 45, float("nan"), 65]),
    background_payload(confirmed=False),
    background_payload(reviewer=""),
    background_payload(reason=""),
    background_payload(background_kind="landslide"),
    background_payload(source_id=99),
    background_payload(tile="C"),
    background_payload(view_factor=2),
])
def test_invalid_or_landslide_overlapping_background_cannot_be_saved(background_batch, bad):
    with pytest.raises(ValueError):
        background_batch.save_background(bad)
    assert background_batch.backgrounds() == {}


def test_background_blocks_manual_extensions_even_after_positive_exclusion(background_batch):
    b = background_batch
    b.save(payload(tile="A", box=[10, 10, 70, 70]))
    with pytest.raises(ValueError):
        b.save_background(background_payload())
    b.exclude({"tile": "A", "source_id": 1, "reviewer": "human", "reason": "unclear"})
    with pytest.raises(ValueError):
        b.save_background(background_payload())
    assert b.backgrounds() == {}


def test_background_must_fit_view_and_unprotected_tile(background_batch, monkeypatch):
    b = background_batch
    b.select("A").view = lambda sid, factor: ({"window": [0, 0, 50, 50]}, b"")
    with pytest.raises(ValueError):
        b.save_background(background_payload())
    b.select("A").view = lambda sid, factor: ({"window": [0, 0, 100, 100]}, b"")
    import continue_ctx_review_download
    monkeypatch.setattr(continue_ctx_review_download, "protected_tiles", lambda: {"A"})
    with pytest.raises(ValueError):
        b.save_background(background_payload())
    assert b.backgrounds() == {}


def test_evaluation_background_stays_out_of_training(background_batch, monkeypatch):
    import continue_ctx_review_download
    b = background_batch
    monkeypatch.setattr(continue_ctx_review_download, 'protected_tiles', lambda: {'A'})
    with pytest.raises(ValueError, match='不能加入训练背景'):
        b.save_background(background_payload())
    (b.pack/'evaluation_plan.json').write_text(json.dumps(dict(
        role='evaluation_only_never_training', tile_splits={'A':'val','B':'test'})), encoding='utf-8')
    evaluation=m.ReviewBatch(b.items, b.pack, dataset_role='evaluation')
    for tile, review in evaluation.reviews.items():
        review.records=b.select(tile).records
        review.view=b.select(tile).view
    record=evaluation.save_background(background_payload())
    assert record['split']=='val'
    assert record['training_gate']=='forbidden_evaluation_only'
    assert record['dataset_role']=='evaluation'
    assert evaluation.batch_info()['dataset_role']=='evaluation'


def test_evaluation_requires_plan_and_complete_geometry(decision_batch):
    b=decision_batch
    with pytest.raises(FileNotFoundError):
        m.ReviewBatch(b.items, b.pack, dataset_role='evaluation')
    (b.pack/'evaluation_plan.json').write_text(json.dumps(dict(
        role='evaluation_only_never_training', tile_splits={'A':'val','B':'test'})), encoding='utf-8')
    evaluation=m.ReviewBatch(b.items, b.pack, dataset_role='evaluation')
    with pytest.raises(ValueError, match='拼接'):
        evaluation.save(payload(tile='A'))
    evaluation.select('A').rows[1]=dict(aabb_complete='True',polygon_complete='True')
    with pytest.raises(ValueError, match='完整范围'):
        evaluation.save(payload(tile='A',label_scope='visible_core'))
    assert evaluation.save(payload(tile='A'))['confirmed']=='true'


def test_background_pixel_rounding_cannot_include_fractional_landslide_edge(background_batch):
    background_batch.select("A").records[1]["aabb_px"] = [20, 20, 40.3, 40]
    with pytest.raises(ValueError):
        background_batch.save_background(background_payload(box=[40.4, 20, 50, 30]))
    assert background_batch.backgrounds() == {}


def test_background_type_is_optional_and_history_positive_decisions_are_read_only(background_batch):
    b = background_batch
    data = background_payload(); del data["background_kind"]
    assert b.save_background(data)["background_kind"] == "plain_background"
    b.background_only = True
    before = (b.pack / "centered_review_manifest.draft.csv").read_bytes()
    with pytest.raises(ValueError):
        b.save(payload(tile="A"))
    with pytest.raises(ValueError):
        b.exclude({"tile": "A", "source_id": 1, "reviewer": "human", "reason": "unclear"})
    assert (b.pack / "centered_review_manifest.draft.csv").read_bytes() == before
    assert not b.corrections.exists()


def test_retired_tile_uses_native_pixels_with_full_tile_coordinates_and_bounded_view(tmp_path):
    import hashlib
    import io
    import numpy as np
    import tifffile
    from PIL import Image
    pixels = np.arange(1600, dtype=np.uint16).reshape(40, 40).astype(np.uint8)
    path = tmp_path / "native.tif"
    tifffile.imwrite(path, pixels, compression="deflate")
    review = m.Review.__new__(m.Review)
    review.tile = "A"; review.width = review.height = 100; review.tif = None
    review.cache_entries = {1: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "window_origin": [20, 20, 60, 60], "shape": [40, 40], "stride": 1}}
    review.rows = {1: {"window_origin": "[25,25,45,45]", "aabb_complete": "True", "polygon_complete": "True"}}
    review.allowed = {1: {"aabb_px": [30,30,35,35]}}
    review.records = {1: {"source_id": 1, "aabb_px": [30,30,35,35], "polygon_px": []}}
    meta, png = review.view(1, 1)
    assert meta["window"] == [25,25,45,45] and meta["cached_context"] is True
    assert np.array_equal(np.array(Image.open(io.BytesIO(png))), pixels[5:25, 5:25])
    meta, png = review.view(1, 4)
    assert meta["window"] == [20,20,60,60] and meta["context_limited"] is True
    assert np.array_equal(np.array(Image.open(io.BytesIO(png))), pixels)


def test_background_http_save_and_remove_preserve_positive_decision(background_batch):
    from urllib.request import Request, urlopen
    b = background_batch
    b.save(payload(tile="B"))
    before = b.corrections.read_bytes()
    server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.handler_for(b))
    worker = m.threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    def post(path, data):
        request = Request(url+path, data=json.dumps(data).encode(),
                          headers={"Content-Type": "application/json", "X-Manual-Review": "1", "Origin": url}, method="POST")
        with urlopen(request, timeout=3) as response:
            return json.load(response)
    try:
        record = post("/api/background", background_payload())["background"]
        with pytest.raises(ValueError):
            b.remove_background({"tile": "B", "background_id": record["background_id"]})
        assert post("/api/background/remove", {"tile": "A", "background_id": record["background_id"]})["removed"] == record["background_id"]
        assert b.corrections.read_bytes() == before
        assert set(b.batch_info()["saved"]) == {"B:1"}
    finally:
        server.shutdown(); server.server_close(); worker.join(timeout=3)
