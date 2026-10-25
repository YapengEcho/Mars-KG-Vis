"""Freeze matched old/native crop groups before pixels; finalize after PNG build.

No native TIFF reads, encoder loads, model fits, downloads or scheduler changes.
The metadata master is intentionally not an executable feature-extraction plan.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from pathlib import Path
from collections import defaultdict

import numpy as np

import extract_manual_classification_features as C
from prepare_manual_classification_protocol import spatial_components
from recover_reviewed_mosaic_samples import exact_coverage

ROOT = Path(__file__).resolve().parents[2]
OLD_PLAN = ROOT/'runs/research/classification_pixel_provenance_20261015_v2/plan.json'
EXPORT = ROOT/'data/processed/manual_yolo_20261009_v8/export_plan.json'
NATIVE_METADATA = ROOT/'runs/research/native_crop_comparison_preparation_20261015_v1/native_build_plan.json'
OUT = ROOT/'runs/research/native_crop_comparison_protocol_20261015_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def utc():
    return datetime.now(timezone.utc).isoformat()


def world(gt, box):
    if gt[2] or gt[4] or gt[1] <= 0 or gt[5] >= 0:
        raise ValueError('Expected unrotated native grid')
    return [gt[0]+box[0]*gt[1], -gt[3]-box[1]*gt[5],
            gt[0]+box[2]*gt[1], -gt[3]-box[3]*gt[5]]


def intersect(a, b):
    rect=[max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])]
    return rect if rect[0]<rect[2] and rect[1]<rect[3] else None


def protected_geography(export):
    # Use the same spherical metre coordinates as the frozen CTX exporter.
    # No evaluation TIFF, header, pixels or prediction cache is accessed.
    k=3396190*math.pi/180
    result={}
    for tile in export['protected_tiles']:
        match=re.fullmatch(r'E(-?\d+)_N(-?\d+)',tile)
        if match is None:raise ValueError('Unrecognized protected tile ID')
        lon,lat=map(int,match.groups())
        result[tile]=[lon*k,-(lat+4)*k,(lon+4)*k,-lat*k]
    return result


def protected_hits(rect,export):
    hits=[]
    for tile,extent in protected_geography(export).items():
        hit=intersect(rect,extent)
        if hit and hit[2]-hit[0]>1e-6 and hit[3]-hit[1]>1e-6:hits.append(tile)
    return hits


def protected_source_geography(export):
    """Frozen SHP AABB footprints by persisted catalog sequence; no eval images."""
    import shapefile
    refs={name:export['inputs'][name] for name in ['shp','shx','dbf']}
    for reference in refs.values():
        if C.sha256(reference['snapshot'])!=reference['sha256']:
            raise ValueError('Frozen catalog boundary snapshot changed')
    protected={str(int(x)) for x in export['protected_sources']};found={};k=3396190*math.pi/180
    reader=shapefile.Reader(shp=refs['shp']['snapshot'],shx=refs['shx']['snapshot'],dbf=refs['dbf']['snapshot'])
    try:
        for record in reader.iterShapeRecords():
            identifier=str(int(round(float(record.record[0]))))
            if identifier not in protected:continue
            if identifier in found:raise ValueError('Duplicate protected catalog sequence')
            lon0,lat0,lon1,lat1=map(float,record.shape.bbox)
            found[identifier]=[lon0*k,-lat1*k,lon1*k,-lat0*k]
    finally:reader.close()
    if set(found)!=protected:raise ValueError('Missing protected catalog footprint')
    return found,refs


def protected_source_hits(rect,footprints):
    return sorted((sid for sid,box in footprints.items() if (hit:=intersect(rect,box)) and hit[2]-hit[0]>1e-6 and hit[3]-hit[1]>1e-6),key=int)


def old_normalization_footprint(row, sample, export):
    gt=sample.get('geotransform', export['tile_sources'].get(sample['tile'],{}).get('geotransform'))
    if gt is None:
        raise ValueError('Missing original anchor grid')
    rect=world(gt,sample['window_px'])
    pixel=sample['pixel_source']
    if pixel['kind']=='native_mosaic':
        sources=pixel['sources']
    elif pixel['kind']=='full_tile':
        tile=export['tile_sources'][sample['tile']]
        sources=[dict(tile=sample['tile'],path=pixel['path'],geotransform=tile['geotransform'],size_px=tile['tile_size_px'])]
    elif pixel['kind']=='cached_context':
        t=export['tile_sources'][sample['tile']]['geotransform'];x,y=pixel['window_origin'][:2];h,w=pixel['shape']
        sources=[dict(tile=sample['tile'],path=pixel['path'],size_px=[w,h],
                      geotransform=[t[0]+x*t[1],t[1],0,t[3]+y*t[5],0,t[5]])]
    else:
        raise ValueError('Unsupported old pixel provenance')
    tiles=sorted({s['tile'] for s in sources if intersect(rect,world(s['geotransform'],[0,0,*s['size_px']]))})
    if not tiles:
        raise ValueError('Missing original normalization donor')
    return dict(world_box_m=rect,contributing_source_tiles=tiles,
                source_stem=sample['stem'],native_parent_hash=sample['native_crop_sha256'],
                dependency='entire original native detector crop; its percentile stretch affects all classifier pixels')


def native_row_check(item, old_row, sample, export):
    if str(item['source_id'])!=old_row['source_id'] or item['class_id']!=old_row['class_id'] or item['catalog_class']!=old_row['catalog_class']:
        raise ValueError('Native metadata ID/class differs')
    if item['old_crop_sha256']!=old_row['crop_sha256']:
        raise ValueError('Native metadata old crop differs')
    label=[x for x in sample['labels'] if str(x['source_id'])==old_row['source_id']]
    if len(label)!=1 or not np.allclose(label[0]['manual_aabb_px'],item['manual_aabb_px'],rtol=0,atol=1e-7):
        raise ValueError('Reviewed human box changed')
    box=item['manual_aabb_px'];window=item['window_px'];side=window[2]-window[0]
    if side<=0 or side!=window[3]-window[1] or any(int(x)!=x for x in window):
        raise ValueError('Native window is not a whole-pixel square')
    width,height=box[2]-box[0],box[3]-box[1]
    if min(width,height)<=0 or side+1e-6<1.5*max(width,height):
        raise ValueError('Native square does not preserve requested target scale')
    margins=[box[0]-window[0],box[1]-window[1],window[2]-box[2],window[3]-box[3]]
    if min(margins)<-1e-6:
        raise ValueError('Bounded shift cut the reviewed human target')
    expected_deficit=dict(zip(['left','top','right','bottom'],[max(0,b-a) for a,b in zip(margins,[width*.25,height*.25,width*.25,height*.25])]))
    exception=any(v>1e-6 for v in expected_deficit.values())
    if bool(item.get('context_policy_exception',False))!=exception:
        raise ValueError('Context deficit flag differs from actual geometry')
    if exception:
        if item.get('context_exception_reason')!='unavailable_source_edge' or item.get('full_human_box_preserved') is not True:
            raise ValueError('Boundary fallback must explicitly preserve entire human box')
        if any(abs(item['context_deficit_px'][k]-v)>1e-6 for k,v in expected_deficit.items()):
            raise ValueError('Boundary context deficit metadata differs')
    sources=item['sources'];gt=item['native_output_geotransform'];size=item.get('size_px',[side,side])
    if size!=[side,side] or not np.allclose(world(gt,[0,0,*size]),item['native_window_world_m'],rtol=0,atol=1e-6):
        raise ValueError('Native window world transform mismatch')
    if not exact_coverage(dict(geotransform=gt,size_px=size,sources=sources)):
        raise ValueError('Native metadata has coverage hole')
    actual_tiles=set()
    for src in sources:
        hit=intersect(item['native_window_world_m'],world(src['geotransform'],[0,0,*src['size_px']]))
        if hit:
            actual_tiles.add(src['tile'])
            if src['tile'] in export['protected_tiles']:
                raise ValueError('Planned native pixels use protected evaluation donor: '+src['tile'])
        path=Path(src['path'])
        if not path.is_file():
            raise FileNotFoundError(path)
        stat=path.stat()
        if stat.st_size!=src['bytes'] or stat.st_mtime_ns!=src['mtime_ns']:
            raise ValueError('Planned native source metadata changed')
    if actual_tiles!=set(item['contributing_source_tiles']):
        raise ValueError('Native normalization donor tile set differs from whole window')
    protected=protected_hits(item['native_window_world_m'],export)
    if protected:
        raise ValueError('Full native context overlaps protected evaluation geography: '+str(protected))
    expected=Path(item['crop']).resolve()
    if not expected.is_relative_to(ROOT/'data/processed') or expected==Path(old_row['crop']).resolve():
        raise ValueError('Use a new native PNG path in processed data')
    return sorted(actual_tiles)


def cross_fold_geometry(rows):
    boxes=[];identifiers=[];arms=[];folds=[]
    for row in rows:
        for arm,key in [('old','old_parent_footprint_world_m'),('native','native_footprint_world_m')]:
            boxes.append(row[key]);identifiers.append(row['source_id']);arms.append(arm);folds.append(row['fold_id'])
    boxes=np.asarray(boxes);folds=np.asarray(folds);indices=np.arange(len(boxes));pairs=[]
    for i in range(len(boxes)):
        delta=np.minimum(boxes[:,2:],boxes[i,2:])-np.maximum(boxes[:,:2],boxes[i,:2])
        hit=(indices>i)&(folds!=folds[i])&(delta[:,0]>1e-6)&(delta[:,1]>1e-6)
        for j in np.flatnonzero(hit):
            pairs.append(dict(source_ids=[identifiers[i],identifiers[j]],arms=[arms[i],arms[j]]))
    return pairs


def prepare(native_metadata_path=NATIVE_METADATA, out=OUT):
    out=Path(out).resolve();metadata_path=Path(native_metadata_path).resolve()
    if out.exists():
        raise FileExistsError('Preserve earlier comparison master: '+str(out))
    old=C.verify_plan(OLD_PLAN);protocol=read(old['source_protocol']);export=read(EXPORT);metadata=read(metadata_path)
    source_samples={str(r['source_id']):r for r in read(old['source_manifest'])['samples']}
    protected_sources,protected_source_refs=protected_source_geography(export)
    items=metadata.get('rows',metadata.get('samples',[]))
    if len(items)!=1083 or metadata.get('failures') or metadata.get('status')!='metadata_ready_awaiting_frozen_spatial_audit':
        raise ValueError('Require terminal ready metadata for all 1083 targets')
    for path_key,hash_key in [('source_plan','source_plan_sha256'),('source_protocol','source_protocol_sha256'),('source_manifest','source_manifest_sha256'),('export_plan','export_plan_sha256')]:
        if C.sha256(metadata[path_key])!=metadata[hash_key]:raise ValueError('Native source pin differs')
    for path,key in [(ROOT/'data/scripts/build_native_classification_dataset.py','script_sha256'),
                     (ROOT/'data/scripts/build_native_classification_crop_pilot.py','geometry_helper_sha256'),
                     (ROOT/'data/scripts/export_manual_detection_dataset.py','historical_stretch_code_sha256')]:
        if C.sha256(path)!=metadata[key]:raise ValueError('Native generator code changed since preparation')
    for reference in metadata['cache_manifests']:
        if C.sha256(reference['path'])!=reference['sha256']:raise ValueError('Native cache metadata changed')
    by_id={str(x['source_id']):x for x in items}
    if len(by_id)!=1083 or set(by_id)!={r['source_id'] for r in old['rows']}:
        raise ValueError('Native metadata target cohort differs')
    original=copy.deepcopy(protocol['targets']);samples={s['stem']:s for s in export['samples']}
    old_rows={r['source_id']:r for r in old['rows']};rows_by_id={r['source_id']:r for r in original};footprints={};native_tiles={}
    for row in original:
        sample=samples[row['source_stem']]
        footprint=old_normalization_footprint(row,sample,export)
        footprints[row['source_id']]=footprint
        union=set(row['source_tiles'])|set(footprint['contributing_source_tiles'])
        if row['source_id'] in by_id:
            tiles=native_row_check(by_id[row['source_id']],old_rows[row['source_id']],sample,export)
            native_tiles[row['source_id']]=tiles;union.update(tiles)
        row['source_tiles']=sorted(union)
    groups=spatial_components(original)
    eligible=[r for r in original if r['exploratory_eligible']];eligible=sorted(eligible,key=lambda r:int(r['source_id']))
    if [r['source_id'] for r in eligible]!=[r['source_id'] for r in old['rows']]:
        raise ValueError('Ordered target cohort changed')
    old_partition={frozenset(r['source_id'] for r in old['rows'] if r['spatial_group_id']==g) for g in {r['spatial_group_id'] for r in old['rows']}}
    new_partition={frozenset(r['source_id'] for r in eligible if r['spatial_group_id']==g) for g in {r['spatial_group_id'] for r in eligible}}
    preserved=old_partition==new_partition
    if preserved:
        fold_ids=np.array([old_rows[r['source_id']]['fold_id'] for r in eligible])
        summaries=C.assert_fold_isolation(eligible,fold_ids,3)
        receipt=dict(copy.deepcopy(old['fold_assignment_receipt']),assignment_policy='preserved existing count-only 3folds because component membership is unchanged')
    else:
        fold_ids,summaries,receipt=C.make_folds(eligible,folds=3,seed=old['fold_assignment_receipt']['base_seed'],attempts=200)
        receipt['assignment_policy']='refrozen from original seed and counts only after donor component membership changed'
    rows=[]
    for row,fold in zip(eligible,fold_ids):
        identifier=row['source_id'];item=by_id[identifier];old_row=old_rows[identifier]
        rows.append(dict(source_id=identifier,class_id=row['class_id'],catalog_class=row['catalog_class'],
            spatial_group_id=row['spatial_group_id'],fold_id=int(fold),source_tiles=row['source_tiles'],
            old_crop=old_row['crop'],old_crop_sha256=old_row['crop_sha256'],
            old_target_box_image_px=source_samples[identifier]['target_box_image_px'],
            native_expected_crop=str(Path(item['crop']).resolve()),native_crop_sha256_status='pending_actual_build',
            old_parent_footprint_world_m=footprints[identifier]['world_box_m'],
            old_normalization_donor_tiles=footprints[identifier]['contributing_source_tiles'],
            native_footprint_world_m=item['native_window_world_m'],native_normalization_donor_tiles=native_tiles[identifier],
            manual_aabb_px=item['manual_aabb_px'],manual_box_world_m=item['manual_box_world_m'],
            native_window_px=item['window_px'],native_output_geotransform=item['native_output_geotransform'],
            native_size_px=item.get('size_px',[item['window_px'][2]-item['window_px'][0]]*2),
            context_policy_exception=bool(item.get('context_policy_exception',False)),
            context_exception_reason=item.get('context_exception_reason'),
            context_deficit_px=item.get('context_deficit_px',{}),
            actual_context_margin_px=item.get('actual_context_margin_px',{}),
            desired_context_margin_px=item.get('desired_context_margin_px',{})))
    geometry=cross_fold_geometry(rows)
    if geometry:
        raise ValueError('Potential old/native footprint overlap crosses folds: '+str(geometry[:5]))
    new_protected_hits=[];old_protected_hits=[]
    for row in rows:
        newhits=protected_source_hits(row['native_footprint_world_m'],protected_sources)
        oldhits=protected_source_hits(row['old_parent_footprint_world_m'],protected_sources)
        if newhits:new_protected_hits.append(dict(source_id=row['source_id'],protected_source_ids=newhits))
        if oldhits:old_protected_hits.append(dict(source_id=row['source_id'],protected_source_ids=oldhits))
    if new_protected_hits:
        raise ValueError('New native windows intersect protected source AABBs: '+str(new_protected_hits))
    for row in rows:
        if C.sha256(row['old_crop'])!=row['old_crop_sha256']:
            raise ValueError('Old image changed')
    master=dict(schema_version=1,status='locked_before_execution',created_utc=utc(),
        scope=C.POLICY,native_spatial_audit_approved=True,feature_plan_executable=False,
        source_plan=str(OLD_PLAN.resolve()),source_plan_sha256=C.sha256(OLD_PLAN),
        source_protocol=old['source_protocol'],source_protocol_sha256=old['source_protocol_sha256'],
        old_source_manifest=old['source_manifest'],old_source_manifest_sha256=old['source_manifest_sha256'],
        frozen_export_plan=str(EXPORT.resolve()),frozen_export_plan_sha256=C.sha256(EXPORT),
        native_build_plan=str(metadata_path),native_build_plan_sha256=C.sha256(metadata_path),
        native_builder_script=str(ROOT/'data/scripts/build_native_classification_dataset.py'),native_builder_script_sha256=metadata['script_sha256'],
        geometry_helper_script=str(ROOT/'data/scripts/build_native_classification_crop_pilot.py'),geometry_helper_script_sha256=metadata['geometry_helper_sha256'],
        historical_stretch_script=str(ROOT/'data/scripts/export_manual_detection_dataset.py'),historical_stretch_script_sha256=metadata['historical_stretch_code_sha256'],
        protocol_preparer_script=str(Path(__file__).resolve()),protocol_preparer_script_sha256=C.sha256(__file__),
        class_mapping=old['class_mapping'],class_names=old['class_names'],catalog_counts=old['catalog_counts'],
        unique_source_count=1083,held_source_ids=old['held_source_ids'],folds=3,
        original_folds_preserved=preserved,fold_assignment_receipt=receipt,fold_summaries=summaries,
        union_protocol_targets=original,union_groups=groups,rows=rows,
        group_policy=dict(method='components of retained semantic/identity-held relationships and old full normalization plus new full normalization pixel donor tiles',
                          holds_preserve_graph_connections=True,actual_pixel_donor_tiles_included=True,
                          scope='training-internal matched crop pipeline comparison; no independent final acceptance',
                          fold_assignment=receipt['assignment_policy'],
                          limitation='Geometry does not independently adjudicate types or prove geographic buffers.'),
        preprocessing=copy.deepcopy(old['preprocessing']),cv_comparison=copy.deepcopy(old['cv_comparison']),
        primary_configuration=dict(encoder='dinov2',head='logistic_regression',balance_mode='balanced',C=1.0,
                                   preselection_basis='previous corrected-group internal screening; declared before new crop scores'),
        crop_comparison=dict(old='unchanged frozen 256 PNG to RGB bilinear224',native='native uint8 window, own-window nonzero 2-98 stretch, direct bilinear224 grayscale PNG then RGB',
                             interpretation='crop pipeline changes include context, contrast statistics and resampling, not an isolated resolution factor'),
        boundary_context_policy=metadata['boundary_context_policy'],
        boundary_context_exception_source_ids=[r['source_id'] for r in rows if r['context_policy_exception']],
        protected_source_aabb_world_m=protected_sources,protected_source_snapshot_refs=protected_source_refs,
        verification=dict(actual_donor_tiles_cross_folds=False,cross_fold_old_native_normalization_footprint_overlap_pairs=geometry,
                          protected_evaluation_geography_overlap=False,protected_tile_ids=export['protected_tiles'],
                          protected_source_count=len(protected_sources),new_native_protected_source_aabb_hits=new_protected_hits,
                          old_historical_normalization_protected_source_aabb_hits=old_protected_hits,
                          old_ordered_crop_hashes_verified=True,new_crop_hashes_not_yet_available=True),
        resources=dict(native_tiff_pixels_read=False,images_modified=False,gpu_used=False,classifiers_fitted=False,downloads=False))
    out.mkdir(parents=True);write(out/'master_plan.json',master)
    verify_master(out/'master_plan.json')
    write(out/'verification.json',dict(status='passed_before_pixel_build',master_plan_sha256=C.sha256(out/'master_plan.json'),
                                     original_folds_preserved=preserved,targets=1083,held_source_ids=old['held_source_ids'],
                                     zero_cross_fold_normalization_footprint_overlap=True))
    return dict(master_plan=str(out/'master_plan.json'),original_folds_preserved=preserved,
                groups=len(groups),fold_counts=[s['held_counts'] for s in summaries])


def verify_master(master_path):
    master=read(master_path)
    if master['status']!='locked_before_execution' or master['native_spatial_audit_approved'] is not True or master['feature_plan_executable']:
        raise ValueError('Not an approved metadata-only master')
    for path_key,hash_key in [('source_plan','source_plan_sha256'),('source_protocol','source_protocol_sha256'),
        ('old_source_manifest','old_source_manifest_sha256'),('frozen_export_plan','frozen_export_plan_sha256'),
        ('native_build_plan','native_build_plan_sha256'),('protocol_preparer_script','protocol_preparer_script_sha256'),
        ('native_builder_script','native_builder_script_sha256'),('geometry_helper_script','geometry_helper_script_sha256'),
        ('historical_stretch_script','historical_stretch_script_sha256')]:
        if C.sha256(master[path_key])!=master[hash_key]:
            raise ValueError('Master pinned input changed: '+path_key)
    old=C.verify_plan(master['source_plan']);rows=master['rows'];metadata=read(master['native_build_plan']);export=read(master['frozen_export_plan']);items={str(r['source_id']):r for r in metadata.get('rows',metadata.get('samples',[]))}
    protected_sources,refs=protected_source_geography(export)
    if protected_sources!=master['protected_source_aabb_world_m'] or refs!=master['protected_source_snapshot_refs']:
        raise ValueError('Protected source footprint audit changed')
    if len(rows)!=1083 or [r['source_id'] for r in rows]!=[r['source_id'] for r in old['rows']]:
        raise ValueError('Ordered master cohort changed')
    if master['cv_comparison']!=old['cv_comparison'] or master['held_source_ids']!=old['held_source_ids']:
        raise ValueError('Head policy or identity holds changed')
    tile_folds=defaultdict(set)
    for row,previous in zip(rows,old['rows']):
        for key in ['source_id','class_id','catalog_class']:
            if row[key]!=previous[key]:raise ValueError('Master ordered type differs')
        if row['old_crop']!=previous['crop'] or row['old_crop_sha256']!=previous['crop_sha256']:
            raise ValueError('Master old crop changed')
        item=items[row['source_id']]
        if str(Path(item['crop']).resolve())!=row['native_expected_crop'] or item['native_window_world_m']!=row['native_footprint_world_m']:
            raise ValueError('Master native geometry/path changed')
        if bool(item.get('context_policy_exception',False))!=row['context_policy_exception'] or item.get('context_deficit_px',{})!=row['context_deficit_px']:
            raise ValueError('Master boundary exception diagnostics changed')
        if protected_hits(row['native_footprint_world_m'],export) or set(row['native_normalization_donor_tiles'])&set(export['protected_tiles']):
            raise ValueError('Protected evaluation geography in frozen native context')
        if protected_source_hits(row['native_footprint_world_m'],protected_sources):
            raise ValueError('Protected source AABB in frozen native context')
        for tile in row['source_tiles']:tile_folds[tile].add(row['fold_id'])
    if any(len(v)>1 for v in tile_folds.values()):raise ValueError('Donor tile crosses master folds')
    summaries=C.assert_fold_isolation(rows,[r['fold_id'] for r in rows],3)
    if summaries!=master['fold_summaries'] or cross_fold_geometry(rows):raise ValueError('Master fold/footprint audit changed')
    return master


def finalize(master_path,native_manifest_path,out_dir):
    master=verify_master(master_path);manifest_path=Path(native_manifest_path).resolve();manifest=read(manifest_path)
    if manifest.get('status')!='completed1083_native_train_crops_same_ids_labels':
        raise ValueError('Require completed native PNG manifest before feature extraction')
    metadata=read(master['native_build_plan']);frozen_by_id={str(r['source_id']):r for r in metadata['rows']}
    items=manifest.get('samples',manifest.get('rows',[]));by_id={str(x['source_id']):x for x in items}
    if len(items)!=1083 or len(by_id)!=1083 or set(by_id)!={r['source_id'] for r in master['rows']}:
        raise ValueError('Actual native manifest cohort differs')
    if manifest.get('master_plan_sha256')!=C.sha256(master_path) or manifest.get('native_build_plan_sha256')!=master['native_build_plan_sha256']:
        raise ValueError('Actual native manifest is not bound to this frozen master')
    native_hashes={};old_by_id={r['source_id']:r for r in master['union_protocol_targets']}
    from PIL import Image
    for row in master['rows']:
        item=by_id[row['source_id']];path=Path(item['crop']).resolve()
        frozen=frozen_by_id[row['source_id']]
        if item.get('status')!='native_direct224_complete' or item.get('native_build_plan_sha256')!=master['native_build_plan_sha256']:
            raise ValueError('Native row is incomplete or belongs to another build')
        if item.get('row_sha256')!=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest():
            raise ValueError('Native row receipt is not bound to frozen geometry')
        if any(item.get(k)!=v for k,v in frozen.items() if k!='status'):
            raise ValueError('Built native row geometry or source metadata changed')
        if item.get('full_human_box_preserved') is not True or item.get('complete_native_coverage') is not True or item.get('minimum_context_preserved')!=(not row['context_policy_exception']):
            raise ValueError('Built native row lost target/coverage/context policy')
        counts=item.get('actual_donor_pixel_counts',{})
        if not counts or any(not isinstance(n,int) or n<=0 for n in counts.values()) or sum(counts.values())!=np.prod(row['native_size_px']):
            raise ValueError('Actual native donor pixels do not cover the whole window exactly')
        sources={s['path']:s for s in frozen['sources']}
        if not set(counts)<=set(sources) or set(item.get('actual_donor_tiles',[]))!={sources[p]['tile'] for p in counts}:
            raise ValueError('Native actual donor paths/tiles differ from frozen sources')
        if not set(item['actual_donor_tiles'])<=set(row['native_normalization_donor_tiles']):
            raise ValueError('Native actual donor tile was not included in spatial audit')
        if str(path)!=row['native_expected_crop'] or item['catalog_class']!=row['catalog_class'] or item['class_id']!=row['class_id']:
            raise ValueError('Actual native path/type changed')
        digest=C.sha256(path)
        if digest!=item['crop_sha256']:raise ValueError('Built native PNG changed')
        with Image.open(path) as image:
            if image.size!=(224,224) or image.mode!='L':raise ValueError('Expected full224 native grayscale PNG')
        if C.sha256(row['old_crop'])!=row['old_crop_sha256']:raise ValueError('Old crop mutated after freeze')
        native_hashes[row['source_id']]=digest
    out=Path(out_dir).resolve()
    if not out.is_relative_to(ROOT/'runs'):raise ValueError('Use an experiment directory inside project runs')
    out.mkdir(parents=True,exist_ok=True)
    result={}
    for arm in ['old','native']:
        folder=out/arm
        if folder.exists():raise FileExistsError('Preserve previous finalized '+arm+' plan')
        folder.mkdir()
        protocol=copy.deepcopy(read(master['source_protocol']));protocol['targets']=copy.deepcopy(master['union_protocol_targets']);protocol['groups']=copy.deepcopy(master['union_groups'])
        protocol.update(created_utc=utc(),status='matched_crop_protocol_after_built_image_hash_verification',
                        spatial_group_count=len(protocol['groups']),group_policy=master['group_policy'])
        protocol['source_tile_node_count']=len({t for r in protocol['targets'] for t in r['source_tiles']})
        protocol['cross_tile_targets']=sum(len(r['source_tiles'])>1 for r in protocol['targets'])
        protocol['eligible_groups_per_catalog_class']={name:sum(g['eligible_catalog_class_counts'][name]>0 for g in protocol['groups']) for name in C.CLASS_NAMES}
        protocol['nominal_all_class_cv_fold_ceiling']=min(protocol['eligible_groups_per_catalog_class'].values())
        protocol['folds_assigned']=True
        group_folds={r['spatial_group_id']:r['fold_id'] for r in master['rows']}
        for group in protocol['groups']:group['assigned_fold']=group_folds.get(group['spatial_group_id'])
        if arm=='native':
            protocol['input_manifest']=str(manifest_path);protocol['input_manifest_sha256']=C.sha256(manifest_path)
            for target in protocol['targets']:
                if target['source_id'] in by_id:
                    item=by_id[target['source_id']];target['crop']=str(Path(item['crop']).resolve());target['crop_sha256']=native_hashes[target['source_id']]
                    target['crop_references']=[target['crop']];target['padding_fraction']=0.
            # Held-only nodes retain their original image references and remain excluded.
        protocol['pixel_crop_comparison']=dict(arm=arm,master_plan=str(Path(master_path).resolve()),master_plan_sha256=C.sha256(master_path),native_manifest=str(manifest_path),native_manifest_sha256=C.sha256(manifest_path))
        protocol_path=folder/'source_protocol.json';write(protocol_path,protocol)
        rows=C.eligible_rows(protocol,allow_catalog_labels=True);fold_by_id={r['source_id']:r['fold_id'] for r in master['rows']}
        rows=[dict(r,fold_id=fold_by_id[r['source_id']]) for r in rows]
        master_by_id={r['source_id']:r for r in master['rows']}
        for row in rows:
            source=master_by_id[row['source_id']]
            if arm=='old':row['target_box_image_px']=source['old_target_box_image_px']
            else:
                x,y,x1,y1=source['native_window_px'];box=source['manual_aabb_px'];side=x1-x
                row['target_box_image_px']=[(box[0]-x)*224/side,(box[1]-y)*224/side,(box[2]-x)*224/side,(box[3]-y)*224/side]
                row['longest_target_side_px']=max(row['target_box_image_px'][2]-row['target_box_image_px'][0],row['target_box_image_px'][3]-row['target_box_image_px'][1])
        plan=dict(schema_version=1,created_utc=utc(),status='matched_crop_arm_locked_before_feature_or_head_execution',scope=C.POLICY,
            allow_catalog_labels=True,source_protocol=str(protocol_path),source_protocol_sha256=C.sha256(protocol_path),
            source_manifest=protocol['input_manifest'],source_manifest_sha256=protocol['input_manifest_sha256'],
            class_names=master['class_names'],class_mapping=master['class_mapping'],unique_source_count=1083,catalog_counts=master['catalog_counts'],approved_subtype_count=0,
            folds=3,fold_assignment_receipt=master['fold_assignment_receipt'],fold_summaries=master['fold_summaries'],group_policy=master['group_policy'],
            preprocessing=master['preprocessing'],cv_comparison=master['cv_comparison'],held_source_ids=master['held_source_ids'],rows=rows,
            comparison_arm=arm,master_plan=str(Path(master_path).resolve()),master_plan_sha256=C.sha256(master_path),
            native_manifest=str(manifest_path),native_manifest_sha256=C.sha256(manifest_path),
            original_folds_preserved=master['original_folds_preserved'],
            boundary_context_policy=master['boundary_context_policy'],boundary_context_exception_source_ids=master['boundary_context_exception_source_ids'],
            resource_receipt=dict(gpu_used=False,features_extracted=False,classifiers_fitted=False,images_modified=False,source_manifest_modified=False,downloads=False))
        plan_path=folder/'plan.json';write(plan_path,plan);C.verify_plan(plan_path);result[arm+'_plan']=str(plan_path)
    write(out/'finalized_crop_plan_verification.json',dict(status='passed',master_plan_sha256=C.sha256(master_path),native_manifest_sha256=C.sha256(manifest_path),
        old_plan_sha256=C.sha256(result['old_plan']),native_plan_sha256=C.sha256(result['native_plan']),cohort=1083,held_source_ids=master['held_source_ids'],
        same_ordered_labels_groups_folds=True,new_png_hashes_verified=True,new_png_mode='L',new_png_size=[224,224]))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-metadata',type=Path,default=NATIVE_METADATA)
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    print(json.dumps(prepare(args.native_metadata,args.out),ensure_ascii=False))
