import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data/scripts"))
from recover_held_manual_samples import verify_window
from cross_tile_neighbor_review import compose


def record(sid, bbox, hold=None):
    return dict(source_id=sid, support=bbox, hold_reason=hold)


def test_recovery_closes_over_all_complete_neighbors():
    records = {1: record(1, [1, 1, 3, 3]), 2: record(2, [3, 2, 5, 4])}
    assert [r['source_id'] for r in verify_window(records, [0, 0, 6, 6], 1)] == [1, 2]


@pytest.mark.parametrize('neighbor', [record(2, [3, 2, 8, 4]),
                                     record(2, [3, 2, 5, 4], 'human_excluded'),
                                     record(2, [3, 2, 5, 4], 'incomplete_reference')])
def test_recovery_rejects_held_or_clipped_neighbor(neighbor):
    with pytest.raises(ValueError):
        verify_window({1: record(1, [1, 1, 3, 3]), 2: neighbor}, [0, 0, 6, 6], 1)


def test_mosaic_has_exact_seam_and_marks_missing_pixels(tmp_path):
    sources = []
    for i in range(2):
        p = tmp_path / f'{i}.tif'
        tifffile.imwrite(p, np.full((4, 4), 30 + i * 70, dtype=np.uint8))
        sources.append(dict(path=str(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns,
                            geotransform=[i * 4, 1, 0, 0, 0, -1], size_px=[4, 4]))
    e = dict(size_px=[10, 4], geotransform=[0, 1, 0, 0, 0, -1], sources=sources)
    image, coverage, stride = compose(e)
    assert stride == 1 and np.all(image[:, :4] == 30) and np.all(image[:, 4:8] == 100)
    assert coverage[:, :8].all() and not coverage[:, 8:].any()
    sources[0]['bytes'] += 1
    with pytest.raises(ValueError, match='source changed'):
        compose(e)
