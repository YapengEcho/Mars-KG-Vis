from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data/scripts"))
from cache_reviewed_ctx_tiles import native_window
from continue_ctx_review_download import validate_queue


def catalog(tile="E-040_N-08"):
    archive = f"MurrayLab_GlobalCTXMosaic_V01_{tile}.zip"
    return {tile: {"zip": archive, "url": f"https://murray-lab.caltech.edu/CTX/V01/tiles/{archive}"}}


def test_queue_blocks_target_crossing_into_protected_tile():
    tile, protected = "E-040_N-08", "E-040_N-12"
    with pytest.raises(ValueError, match="protected"):
        validate_queue([tile], catalog(), {1: {tile, protected}}, {protected})


def test_queue_accepts_safe_target_and_does_not_assign_a_split():
    tile = "E-040_N-08"
    source = catalog()
    validate_queue([tile], source, {1: {tile}}, {"E-040_N-12"})
    assert "split" not in source[tile]


def test_queue_rejects_catalog_url_substitution():
    tile = "E-040_N-08"
    source = catalog()
    source[tile]["url"] = "https://example.com/tile.zip"
    with pytest.raises(ValueError, match="URL"):
        validate_queue([tile], source, {1: {tile}}, set())


def test_native_cache_contains_expanded_manual_bottom_and_reference_window():
    row = {"window_origin": json.dumps([100, 100, 1124, 1124])}
    corrected = [150, 500, 1100, 1800]
    x0, y0, x1, y1 = native_window(row, corrected, 10000, 10000)
    assert x0 <= 100 and y0 <= 100 and x1 >= 1124 and y1 >= 1800


def test_native_cache_preserves_tile_edge_pixels_without_overflow():
    row = {"window_origin": json.dumps([8000, 8000, 9500, 9500])}
    window = native_window(row, [8500, 8500, 10000, 10000], 10000, 10000)
    assert window[2:] == [10000, 10000]
    assert window[0] >= 0 and window[1] >= 0


def test_native_cache_rejects_missing_pixels():
    row = {"window_origin": json.dumps([8000, 8000, 9500, 9500])}
    with pytest.raises(ValueError, match="contain"):
        native_window(row, [8500, 8500, 10001, 10000], 10000, 10000)
