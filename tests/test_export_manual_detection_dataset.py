"""Guard label completeness and negative contamination in human dataset export."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data/scripts'))
from export_manual_detection_dataset import (positive_window,background_reason,
                                             labels_text,offset_window)


def slide(sid,box,manual=None,hold=None):
    from export_manual_detection_dataset import union
    manual=manual or box
    return dict(source_id=sid,aabb_px=box,manual_box=manual,
                support=union([box,manual]),hold_reason=hold)


def test_recrop_covers_expanded_manual_and_original_for_same_image_control():
    r=slide(1,[1200,1200,1600,1600],[1200,900,1900,1900])
    window,hits,reason=positive_window(1,{1:r},[0,0,6000,6000],'a')
    assert reason is None and len(hits)==1
    assert window[0]<=1200 and window[1]<=900 and window[2]>=1900 and window[3]>=1900


def test_neighbor_closure_includes_all_complete_labels():
    records={1:slide(1,[2000,2000,2400,2400]),2:slide(2,[2300,2200,3600,2700])}
    w,hits,reason=positive_window(1,records,[0,0,6000,6000],'a')
    assert reason is None and {r['source_id'] for r in hits}=={1,2}
    assert w[2]>=3600 and w[3]>=2700


@pytest.mark.parametrize('reason',['human_exclusion_not_background','missing_confirmed_manual_box','unreviewed_or_incomplete_geometry','source_touches_protected_tile'])
def test_unknown_or_excluded_neighbor_blocks_ordinary_yolo(reason):
    rs={1:slide(1,[2000,2000,2400,2400]),2:slide(2,[2350,2100,3000,2300],hold=reason)}
    w,hits,why=positive_window(1,rs,[0,0,6000,6000],'a')
    assert w is None and hits==[] and reason in why


def test_cache_does_not_silently_truncate_expanded_slide():
    rs={1:slide(1,[2000,2000,2400,2400]),2:slide(2,[2350,2100,4500,2300])}
    w,hits,why=positive_window(1,rs,[1000,1000,4000,4000],'a')
    assert w is None and why=='neighbor_requires_unavailable_native_context'


def test_recheck_background_against_new_manual_extension():
    bg=dict(confirmed=True,judgement='certain_background',reviewer='owner',reason='reviewed',
            aabb_px=[3000,2000,3200,2200],view_window_px=[0,0,5000,5000])
    assert background_reason(bg,[0,0,5000,5000],{1:slide(1,[2000,2000,2400,2400],[2000,2000,3100,2300])}).startswith('overlaps_known_or_manual')


def test_clean_confirmed_background_is_empty_detection_label():
    bg=dict(confirmed=True,judgement='certain_background',reviewer='owner',reason='reviewed',
            aabb_px=[3000,2000,3200,2200],view_window_px=[0,0,5000,5000])
    assert background_reason(bg,[0,0,5000,5000],{1:slide(1,[2000,2000,2400,2400])}) is None
    assert labels_text(dict(window_px=bg['aabb_px'],labels=[]),'manual_aabb_px')==''


def test_geometry_to_yolo_uses_native_rectangle_without_stretching():
    s=dict(window_px=[1000,2000,3000,3000],labels=[dict(class_name='CL',source_id=1,
        manual_aabb_px=[1500,2250,2500,2750],shp_aabb_px=[1400,2200,2400,2700])])
    assert labels_text(s,'manual_aabb_px').split()==['0','0.50000000','0.50000000','0.50000000','0.50000000']
    assert labels_text(s,'manual_aabb_px',True).startswith('3 ')


def test_context_offsets_repeat_and_vary_across_sources():
    args=([2000,2000,2400,2400],[0,0,6000,6000])
    assert offset_window(*args,'1')==offset_window(*args,'1')
    assert offset_window(*args,'1')!=offset_window(*args,'2')


def test_compressed_retired_tile_cache_uses_exact_native_offset(tmp_path):
    import numpy as np
    import tifffile
    from export_manual_detection_dataset import Pixels,sha256_file
    a=np.arange(80,dtype=np.uint8).reshape(8,10)
    path=tmp_path/'context.tif'
    tifffile.imwrite(path,a,compression='deflate')
    sample=dict(window_px=[102,203,107,207],pixel_source=dict(kind='cached_context',
        path=str(path),shape=[8,10],window_origin=[100,200,110,208],sha256=sha256_file(path)))
    assert np.array_equal(Pixels().crop(sample),a[3:7,2:7])
    sample['pixel_source']['sha256']='bad'
    with pytest.raises(ValueError,match='checksum'):
        Pixels().crop(sample)


def test_old_six_decimal_rounding_kept_but_real_label_overflow_rejected(tmp_path):
    from export_manual_detection_dataset import verify_labels
    path=tmp_path/'label.txt'
    path.write_text('0 0.920994 0.862240 0.158012 0.275521\n')
    verify_labels(path)
    path.write_text('0 0.95 0.5 0.2 0.3\n')
    with pytest.raises(ValueError,match='Out-of-frame'):
        verify_labels(path)


def test_chunked_stretch_matches_historical_numpy_percentiles():
    import numpy as np
    from export_manual_detection_dataset import stretch_uint8
    from make_yolo_dataset import stretch
    rng=np.random.default_rng(99)
    frames=[rng.integers(0,256,size=(147,333),dtype=np.uint8),
            np.zeros((3,7),dtype=np.uint8),np.full((120,100),100,dtype=np.uint8),
            np.array([[0,1,2,4,20,40,90,180,255]],dtype=np.uint8)]
    for frame in frames:
        assert np.array_equal(stretch_uint8(frame),stretch(frame))
