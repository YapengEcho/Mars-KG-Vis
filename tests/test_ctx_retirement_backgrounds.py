import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data/scripts"))
from cache_reviewed_ctx_tiles import preservation_window


def test_retirement_preserves_background_outside_original_context():
    row = {"window_origin": json.dumps([100, 100, 200, 200])}
    old = preservation_window(row, [110, 110, 180, 180], 1000, 1000, {})
    new = preservation_window(row, [110, 110, 180, 180], 1000, 1000,
                              {"far": {"aabb_px": [700.2, 750.1, 820.8, 880.9]}})
    assert new[0] <= old[0] and new[1] <= old[1]
    assert new[2] == 821 and new[3] == 881


@pytest.mark.parametrize("bbox", [[-1, 0, 1, 1], [0, 0, 1001, 2],
                                  [3, 3, 2, 4], [0, 0, float("nan"), 2]])
def test_invalid_background_prevents_retirement(bbox):
    with pytest.raises(ValueError):
        preservation_window({"window_origin": "[100,100,200,200]"},
                            [110, 110, 180, 180], 1000, 1000,
                            {"invalid": {"aabb_px": bbox}})
