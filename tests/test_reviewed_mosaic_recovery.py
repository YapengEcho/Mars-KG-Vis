import sys
from pathlib import Path
import numpy as np
import tifffile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data/scripts'))
from recover_reviewed_mosaic_samples import world_box,pixel_box,exact_coverage,coverage_rows,native_crop,tight_closure,labels_equivalent,freeze,read_json,write_json,relocate_review_inputs
import csv
import threading
import pytest
import shutil
from types import SimpleNamespace
from cross_tile_neighbor_review import MosaicReview


def source(path,array,x):
    tifffile.imwrite(path,array)
    return dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,
                geotransform=[x,1,0,0,0,-1],size_px=[array.shape[1],array.shape[0]])


def test_world_coordinate_transform_preserves_off_tile_boxes():
    gt=[-100,5,0,300,0,-5];box=[-4.5,8,99.5,111]
    assert pixel_box(gt,world_box(gt,box))==box


def test_existing_source_identity_does_not_hide_a_revised_manual_box():
    a=[dict(source_id=7,class_name='SL',manual_aabb_px=[0,0,10,10],shp_aabb_px=[0,0,10,10])]
    b=[dict(a[0],manual_aabb_px=[0,0,12,10])]
    assert labels_equivalent(a,a)
    assert not labels_equivalent(a,b)


def test_native_mosaic_pixels_are_exact_across_seam(tmp_path):
    a=np.arange(12,dtype=np.uint8).reshape(3,4);b=a+40
    entry=dict(geotransform=[0,1,0,0,0,-1],size_px=[8,3],sources=[source(tmp_path/'a.tif',a,0),source(tmp_path/'b.tif',b,4)])
    assert exact_coverage(entry)
    assert np.array_equal(native_crop(entry),np.concatenate([a,b],axis=1))


def test_coverage_detects_one_pixel_gap(tmp_path):
    a=np.ones((3,4),dtype=np.uint8)
    entry=dict(geotransform=[0,1,0,0,0,-1],size_px=[9,3],sources=[source(tmp_path/'a.tif',a,0),source(tmp_path/'b.tif',a,5)])
    assert not exact_coverage(entry)
    assert exact_coverage(entry)==coverage_rows(entry,0,3).all()


def test_lossless_compressed_native_context_is_not_a_preview(tmp_path):
    a=np.arange(12,dtype=np.uint8).reshape(3,4)
    path=tmp_path/'cache.tif';tifffile.imwrite(path,a,compression='deflate')
    s=dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,
           geotransform=[0,1,0,0,0,-1],size_px=[4,3])
    entry=dict(geotransform=[0,1,0,0,0,-1],size_px=[4,3],sources=[s])
    assert np.array_equal(native_crop(entry),a)


def test_coverage_detects_uncovered_corner(tmp_path):
    a=np.ones((2,2),dtype=np.uint8)
    src1=source(tmp_path/'a.tif',a,0);src2=source(tmp_path/'b.tif',a,2);src2['geotransform'][3]=-2
    entry=dict(geotransform=[0,1,0,0,0,-1],size_px=[4,4],sources=[src1,src2])
    assert not exact_coverage(entry)


def test_closure_expands_approved_neighbor_but_holds_excluded_neighbor():
    records={1:dict(source_id=1,support=[.2,.2,4.2,4.2],hold_reason=None),2:dict(source_id=2,support=[3,3,7,7],hold_reason=None)}
    window,hits,reason=tight_closure(1,records)
    assert window==[0,0,7,7] and [r['source_id'] for r in hits]==[1,2] and reason is None
    records[2]['hold_reason']='human_excluded_not_background'
    window,hits,reason=tight_closure(1,records)
    assert window is None and 'human_excluded_not_background' in reason


def test_excluding_previously_saved_mosaic_removes_active_positive_and_keeps_history(tmp_path):
    review=MosaicReview.__new__(MosaicReview);review.folder=tmp_path;review.lock=threading.RLock()
    review.corrections=tmp_path/'cross_tile_manual_boxes.reviewed.csv';review.fields=['tile','source_id']
    with review.corrections.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=review.fields);writer.writeheader();writer.writerow(dict(tile='MOSAIC_00007',source_id=7))
    entry=dict(tile='MOSAIC_00007',source_id=7,geotransform=[0,1,0,0,0,-1],original_box=[0,0,10,10])
    review.select=lambda tile:SimpleNamespace(entry=entry)
    row=review.exclude(dict(tile=entry['tile'],source_id=7,reviewer='human',reason='unclear'))
    assert row['negative_eligible'] is False
    assert review.saved()=={}
    assert 'MOSAIC_00007:7' in review.exclusions()
    assert 'exclude' in (tmp_path/'cross_tile_history.jsonl').read_text(encoding='utf-8')


def review_pack(folder,sid,accepted=False,excluded=False):
    folder.mkdir();(folder/'manual_review').mkdir()
    write_json(folder/'mosaic_review_manifest.json',dict(entries=[dict(tile=f'MOSAIC_{sid:05d}',source_id=sid)],pending=[],ready_count=1,pending_count=0))
    if accepted:
        with (folder/'manual_review/cross_tile_manual_boxes.reviewed.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['tile','source_id','confirmed']);writer.writeheader();writer.writerow(dict(tile=f'MOSAIC_{sid:05d}',source_id=sid,confirmed='true'))
    if excluded:write_json(folder/'manual_review/cross_tile_exclusions.json',{f'MOSAIC_{sid:05d}:{sid}':dict(source_id=sid,negative_eligible=False)})


def test_empty_followup_is_visible_without_creating_training_approvals(tmp_path):
    primary=tmp_path/'primary';extra=tmp_path/'extra';review_pack(primary,7,accepted=True);review_pack(extra,8)
    write_json(primary/'review_followup_packs.json',dict(packs=[str(extra)]))
    index=freeze(primary,tmp_path/'inputs')
    assert read_json(Path(index['manifest']['snapshot']))['ready_count']==2
    with Path(index['manual']['snapshot']).open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert [row['source_id'] for row in rows]==['7']
    assert not Path(index['followup_1_manual']['source']).exists()


def test_saved_followup_and_exclusions_survive_nested_snapshot_relocation(tmp_path):
    primary=tmp_path/'primary';extra=tmp_path/'extra';review_pack(primary,7,excluded=True);review_pack(extra,8,accepted=True)
    write_json(primary/'review_followup_packs.json',dict(packs=[str(extra)]))
    inputs=tmp_path/'inputs';index=freeze(primary,inputs);target=tmp_path/'dataset_inputs';shutil.copytree(inputs,target)
    moved=relocate_review_inputs(index,inputs,target)
    assert Path(moved['followup_1_manifest']['snapshot'])==target/'followup_1/mosaic_review_manifest.json'
    assert read_json(Path(moved['primary_manifest']['snapshot']))['entries'][0]['source_id']==7
    assert read_json(Path(moved['followup_1_manifest']['snapshot']))['entries'][0]['source_id']==8
    assert 'MOSAIC_00007:7' in read_json(Path(moved['exclusions']['snapshot']))
    with Path(moved['manual']['snapshot']).open(encoding='utf-8-sig') as f:assert [r['source_id'] for r in csv.DictReader(f)]==['8']


def test_duplicate_confirmed_source_across_packs_is_rejected(tmp_path):
    primary=tmp_path/'primary';extra=tmp_path/'extra';review_pack(primary,7,accepted=True);review_pack(extra,7,accepted=True)
    write_json(primary/'review_followup_packs.json',dict(packs=[str(extra)]))
    with pytest.raises(ValueError,match='Duplicate manual source'):freeze(primary,tmp_path/'inputs')


def test_missing_followup_manifest_cannot_be_frozen_as_an_empty_review(tmp_path):
    primary=tmp_path/'primary';review_pack(primary,7)
    write_json(primary/'review_followup_packs.json',dict(packs=[str(tmp_path/'missing')]))
    with pytest.raises(FileNotFoundError):freeze(primary,tmp_path/'inputs')
