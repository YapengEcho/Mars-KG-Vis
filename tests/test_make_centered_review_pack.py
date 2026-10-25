import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "data" / "scripts" / "make_centered_review_pack.py"
spec = importlib.util.spec_from_file_location("make_centered_review_pack", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_window_is_adaptive_and_has_minimum_context():
    assert module.compute_window_px(400, min_window_px=1024, margin_factor=1.5) == 1024
    assert module.compute_window_px(2000, min_window_px=1024, margin_factor=1.5) == 3000


def test_centered_window_contains_target_aabb():
    window = module.centered_window((100.0, 200.0, 900.0, 1000.0), 1.5)
    x0, y0, x1, y1 = window["window_origin"]
    assert x0 <= 100.0 and y0 <= 200.0
    assert x1 >= 900.0 and y1 >= 1000.0
    assert window["window_px"] == 1200


def test_polygon_and_aabb_views_share_exact_window_and_all_intersections():
    records = [
        {"source_id": 1813, "aabb_px": [100, 100, 500, 500], "polygon_px": [[(100, 100), (500, 100), (500, 500)]]},
        {"source_id": 1812, "aabb_px": [450, 450, 700, 700], "polygon_px": [[(450, 450), (700, 450), (700, 700)]]},
    ]
    result = module.make_review_row("E-044_N08", 1813, records, (2000, 2000))
    assert result["center_poly_file"].endswith("_center_poly.png")
    assert result["center_aabb_file"].endswith("_center_aabb.png")
    assert result["window_origin"] == result["poly_window_origin"]
    assert result["intersecting_source_ids"] == [1812, 1813]


def test_source_id_uses_persisted_sequence_not_record_index():
    class Record:
        record = [1813, "SL"]

    mapped = module.map_source_records([Record()])
    assert list(mapped) == [1813]


def test_edge_touch_marks_target_reaching_clamped_crop_boundary():
    records = [
        {"source_id": 1, "aabb_px": [-100, 100, 700, 700], "polygon_px": [[(-100, 100), (700, 100), (700, 700)]]},
    ]
    result = module.make_review_row("E000_N00", 1, records, (2000, 2000))
    assert result["edge_touch"] is True


def test_edge_window_keeps_in_tile_target_instead_of_shifting_it_out():
    # Center is 80 px from the right edge. A fixed 1200 px shift would start
    # at x=800 and cut the in-tile half of the box.
    records = [
        {"source_id": 2572, "aabb_px": [1700, 400, 2300, 1000],
         "polygon_px": [[(1700, 400), (2300, 400), (2300, 1000), (1700, 1000)]]},
    ]
    result = module.make_review_row("E096_N04", 2572, records, (2000, 2000))
    ox, oy, ox1, oy1 = result["window_origin"]
    assert ox <= 1700 and oy <= 400
    assert ox1 == 2000 and oy1 >= 1000
    assert result["native_window_px"] == [ox1 - ox, oy1 - oy]
    assert ox1 - ox < 1200
