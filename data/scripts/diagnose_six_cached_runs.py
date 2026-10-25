"""Read-only, reproducible error analysis of six saved best-checkpoint test pools.

Counts reproduce the existing GT-order/confidence-order matcher. Independent
max-IoU diagnostics are separate from one-to-one TP counts. No threshold tuning,
new inference, training, or ground-truth rewriting occurs.
"""
from __future__ import annotations
import csv
import argparse
import hashlib
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import shapefile

from review_eval_labels import ROOT,read,sha,original_boxes,atomic
from audit_yolo_splits import parse_patch_name
from eval_recall_report import iou_matrix,match_image
from make_centered_review_pack import SHP,map_source_records,point_to_px,read_geotransform

FOLDER=ROOT/'runs/ablations/eval_label_review_20261009'
OUT=ROOT/'runs/ablations/six_run_error_diagnosis_20261011'
DATA=ROOT/'data/processed/manual_yolo_20261009_v8/manual'
REBUILD=ROOT/'data/processed/manual_eval_rebuild_20261011_v1'

def ratio(a,b):return float(a/b) if b else None

def area(box):return max(0,box[2]-box[0])*max(0,box[3]-box[1])

def max_iou(box,preds):
    if not len(preds):return 0.,-1
    vals=iou_matrix(np.array([box],np.float32),np.asarray(preds,np.float32).reshape(-1,4))[0]
    j=int(vals.argmax());return float(vals[j]),j

def geometry(gt,pred):
    gw,gh=gt[2]-gt[0],gt[3]-gt[1];pw,ph=pred[2]-pred[0],pred[3]-pred[1]
    gx,gy=(gt[0]+gt[2])/2,(gt[1]+gt[3])/2;px,py=(pred[0]+pred[2])/2,(pred[1]+pred[3])/2
    centered=[gx-pw/2,gy-ph/2,gx+pw/2,gy+ph/2]
    resized=[px-gw/2,py-gh/2,px+gw/2,py+gh/2]
    return dict(pred_area_ratio=area(pred)/area(gt),pred_width_ratio=pw/gw,pred_height_ratio=ph/gh,
        center_dx_fraction=(px-gx)/gw,center_dy_fraction=(py-gy)/gh,
        center_distance_fraction=math.hypot((px-gx)/gw,(py-gy)/gh),
        centered_iou=max_iou(gt,[centered])[0],resized_iou=max_iou(gt,[resized])[0])

def classify_miss(matched,kept_iou,all_iou):
    if matched:return 'matched'
    if kept_iou>=.5:return 'one_prediction_multiple_labels'
    if all_iou>=.5:return 'accurate_proposal_below_locked_confidence'
    if kept_iou>=.1:return 'overlap_but_iou_below_0.5'
    return 'no_nearby_proposal_at_locked_confidence'

def size_bin(side):
    return 'lt64' if side<64 else '64_256' if side<256 else '256_819' if side<819 else 'ge819'

def physical_bin(side):
    return 'lt1km' if side<1 else '1_5km' if side<5 else '5_15km' if side<15 else 'ge15km'

def crop_stats(gray,box):
    h,w=gray.shape;x0,y0,x1,y1=box
    a,b=max(0,math.floor(x0)),max(0,math.floor(y0));c,d=min(w,math.ceil(x1)),min(h,math.ceil(y1))
    pixels=gray[b:d,a:c]
    margin=max(8,round(max(x1-x0,y1-y0)*.1))
    l,t=max(0,a-margin),max(0,b-margin);r,u=min(w,c+margin),min(h,d+margin)
    frame=gray[t:u,l:r];mask=np.ones(frame.shape,bool);mask[b-t:d-t,a-l:c-l]=False
    ring=frame[mask]
    return dict(box_mean=float(pixels.mean()),box_std=float(pixels.std()),
        box_range_p90_p10=float(np.percentile(pixels,90)-np.percentile(pixels,10)),
        box_dark_fraction=float(np.mean(pixels<=40)),
        box_vs_ring_mean_difference=float(pixels.mean()-ring.mean()) if len(ring) else None)

def build_features():
    geo=read(FOLDER/'tile_georeferences.json')
    with shapefile.Reader(str(SHP)) as reader:sources=map_source_records(reader.shapeRecords())
    full={}
    for tile,v in geo.items():
        gt=v['geotransform'];entries=[]
        for sid,s in sources.items():
            pts=[point_to_px(x,y,gt) for x,y in s.shape.points]
            starts=list(s.shape.parts)+[len(pts)]
            for idx,p in enumerate([pts]+[pts[starts[k]:starts[k+1]] for k in range(len(starts)-1)]):
                xs,ys=zip(*p);box=[min(xs),min(ys),max(xs),max(ys)]
                if idx and box==entries[-1]['box']:continue
                entries.append(dict(sid=sid,box=box,part=idx))
        full[tile]=entries
    items=[];features=[];images={}
    for image in sorted((DATA/'images/test').glob('*.png')):
        stem=image.stem;meta=parse_patch_name(stem);tile=meta['tile']
        with Image.open(image) as im:gray=np.asarray(im.convert('L'))
        h,w=gray.shape;boxes=original_boxes(DATA/'labels/test'/f'{stem}.txt',w,h)
        scale=w/meta['size'] if meta['kind']=='center' else 1.
        candidates=[]
        for ref in full[tile]:
            box=[(ref['box'][i]-meta['x' if i%2==0 else 'y'])*scale for i in range(4)]
            if box[2]<=0 or box[3]<=0 or box[0]>=w or box[1]>=h:continue
            clipped=[max(0.,min(w if i%2==0 else h,box[i])) for i in range(4)]
            if area(clipped)>0:candidates.append(dict(**ref,mapped_box=box,clipped_box=clipped))
        centers=np.array([[(r['box'][0]+r['box'][2])/2,(r['box'][1]+r['box'][3])/2] for r in full[tile] if r['part']==0 and r['sid']!=455])
        center_ids=[r['sid'] for r in full[tile] if r['part']==0 and r['sid']!=455]
        images[stem]=dict(stem=stem,tile=tile,image=str(image),width=w,height=h,gt=len(boxes),kind=meta['kind'],
            image_mean=float(gray.mean()),image_std=float(gray.std()),image_dark_fraction=float(np.mean(gray<=40)),
            image_black_fraction=float(np.mean(gray<=5)),image_sha256=sha(image),label_sha256=sha(DATA/'labels/test'/f'{stem}.txt'))
        ious=iou_matrix(np.asarray(boxes,np.float32).reshape(-1,4),np.asarray([r['clipped_box'] for r in candidates],np.float32).reshape(-1,4))
        mutual=iou_matrix(np.asarray(boxes,np.float32).reshape(-1,4),np.asarray(boxes,np.float32).reshape(-1,4));np.fill_diagonal(mutual,0)
        for gi,box in enumerate(boxes):
            side=max(box[2]-box[0],box[3]-box[1]);cx,cy=(box[0]+box[2])/(2*w),(box[1]+box[3])/(2*h)
            row=dict(stem=stem,tile=tile,gt_index=gi,box=box,box_long_side_px=side,size_bin=size_bin(side),
                center_x=cx,center_y=cy,position='central' if .25<=cx<=.75 and .25<=cy<=.75 else 'outer',
                touching_border=min(box[0],box[1],w-box[2],h-box[3])<=1,
                image_gt_count=len(boxes),density_bin='1' if len(boxes)==1 else '2_4' if len(boxes)<=4 else 'ge5',
                max_other_gt_iou=float(mutual[gi].max()) if len(boxes)>1 else 0.,kind=meta['kind'],
                effective_mpp=5./scale,visible_box_long_side_km=side*5./scale/1000,
                source_id=None,source_map_iou=0.,clipped=None,visible_bbox_fraction=None,
                full_source_long_side_km=None,physical_bin='unknown',nearby_sources_5km=None)
            if candidates:
                ci=int(ious[gi].argmax());ref=candidates[ci];row['source_map_iou']=float(ious[gi,ci])
                if row['source_map_iou']>=.995:
                    sid=ref['sid'];mapped=ref['mapped_box'];native=ref['box']
                    row.update(source_id=sid,source_part=ref['part'],source_type=str(sources[sid].record[1]),
                        matching_source_ids=sorted({candidates[j]['sid'] for j in np.flatnonzero(ious[gi]>=.995)}),
                        clipped=min(mapped[0],mapped[1],w-mapped[2],h-mapped[3])<-.01,
                        visible_bbox_fraction=area(ref['clipped_box'])/area(mapped),
                        full_source_long_side_km=max(native[2]-native[0],native[3]-native[1])*5/1000)
                    row['physical_bin']=physical_bin(row['full_source_long_side_km'])
                    center=[(native[0]+native[2])/2,(native[1]+native[3])/2]
                    dist=np.sqrt(np.sum((centers-center)**2,axis=1))*5/1000
                    keep=np.array([i not in {sid,454 if sid==455 else sid} for i in center_ids])
                    row['nearby_sources_5km']=int(np.sum((dist<=5)&keep))
            row.update(crop_stats(gray,box));features.append(row)
        items.append(dict(stem=stem,tile=tile,box_list=boxes,meta=meta))
    assert len(items)==497 and len(features)==1144
    # Label quartiles/tertiles depend only on frozen pixels, never on predictions.
    cutoffs={}
    for field in ['box_mean','box_std','box_dark_fraction']:
        q=np.quantile([r[field] for r in features],[1/3,2/3]);cutoffs[field]=q.tolist()
        for r in features:r[field+'_bin']='low' if r[field]<q[0] else 'mid' if r[field]<q[1] else 'high'
    return items,features,images,cutoffs

def summarize_targets(rows,field):
    groups=defaultdict(list)
    for r in rows:groups[str(r.get(field))].append(r)
    result={}
    for k,v in sorted(groups.items()):
        counts=Counter(r['miss_reason'] for r in v)
        result[k]=dict(gt=len(v),tp=sum(r['matched'] for r in v),recall=ratio(sum(r['matched'] for r in v),len(v)),reasons=dict(counts),
            median_side_px=float(np.median([r['box_long_side_px'] for r in v])),
            clipped_fraction=ratio(sum(r['clipped'] is True for r in v),sum(r['clipped'] is not None for r in v)))
    return result

def standardize(rows,field,groups):
    """Descriptive common-stratum rates; these are not causal effects or tests."""
    cells=defaultdict(lambda:defaultdict(list))
    for r in rows:cells[(r['tile'],r['size_bin'],r['clipped'],r['kind'])][str(r[field])].append(r)
    totals={g:0. for g in groups};weight=0;appearances=0;strata=0
    for cell in cells.values():
        if any(len(cell[g])<5 for g in groups):continue
        n=min(len(cell[g]) for g in groups);weight+=n;strata+=1
        appearances+=sum(len(cell[g]) for g in groups)
        for g in groups:totals[g]+=n*np.mean([r['matched'] for r in cell[g]])
    return dict(rates={g:ratio(v,weight) for g,v in totals.items()},common_strata=strata,covered_gt_appearances=appearances,
        method='Equal common-stratum weights within tile, visible pixel size, clipping and crop kind; minimum 5 appearances per group. Descriptive only.')

def analyze_run(run,predictions,features,items,images,expected):
    by_stem=defaultdict(list)
    for r in features:by_stem[r['stem']].append(r)
    rows=[];imgrows=[];fps=[]
    for item in items:
        stem=item['stem'];gt=item['box_list'];pb,ps,pc=[np.asarray(v,np.float32) for v in predictions[stem]];pb=pb.reshape(-1,4)
        assert len(pb)==len(ps)==len(pc) and np.all(pc==0)
        keep=ps>=run['threshold'];kept=pb[keep];scores=ps[keep]
        hits=match_image([(0,*b) for b in gt],pb,ps,pc,.5,run['threshold'],class_aware=False)
        used={int(h[4]) for h in hits if h[1]};reason_counts=Counter()
        for feature,hit in zip(by_stem[stem],hits):
            row=dict(feature,run=run['key'],threshold=run['threshold'],matched=bool(hit[1]))
            ai,aj=max_iou(feature['box'],pb);ki,kj=max_iou(feature['box'],kept)
            row.update(best_iou_any_cached=ai,best_iou_locked=ki,
                best_conf_any_cached=float(ps[aj]) if aj>=0 else None,
                best_conf_locked=float(scores[kj]) if kj>=0 else None,
                best_pred_locked=kept[kj].tolist() if kj>=0 else None,
                miss_reason=classify_miss(hit[1],ki,ai))
            reason_counts[row['miss_reason']]+=1
            if kj>=0 and ki>=.1:
                row.update(geometry(feature['box'],kept[kj]))
                row['area_error']='too_small' if row['pred_area_ratio']<2/3 else 'too_large' if row['pred_area_ratio']>1.5 else 'similar_area'
                row['fix_center_alone']=row['centered_iou']>=.5
                row['fix_size_alone']=row['resized_iou']>=.5
            rows.append(row)
        for j,box in enumerate(kept):
            if j in used:continue
            bi,gj=max_iou(box,gt)
            fps.append(dict(run=run['key'],stem=stem,tile=item['tile'],box=box.tolist(),score=float(scores[j]),max_gt_iou=bi,
                kind='duplicate_or_competing_label' if bi>=.5 else 'near_labeled_target' if bi>=.1 else 'background_or_unlabeled_target',
                box_long_side_px=float(max(box[2]-box[0],box[3]-box[1])),image_gt_count=len(gt)))
        im=dict(images[stem],run=run['key'],threshold=run['threshold'],tp=len(used),fp=len(kept)-len(used),fn=len(gt)-len(used),
            candidates=len(kept),cache_candidates=len(pb),cache_at_300_cap=len(pb)==300,reasons=dict(reason_counts))
        imgrows.append(im)
    counts=dict(images=len(imgrows),gt=len(rows),tp=sum(r['matched'] for r in rows),fp=len(fps),
        fn=sum(not r['matched'] for r in rows),candidates=sum(r['candidates'] for r in imgrows))
    for field in ['images','gt','tp','fp','fn','candidates']:assert counts[field]==expected[field],(run['key'],field,counts[field],expected[field])
    tiles={}
    for tile in sorted({r['tile'] for r in imgrows}):
        imgs=[r for r in imgrows if r['tile']==tile];targets=[r for r in rows if r['tile']==tile]
        ng=sum(r['gt'] for r in imgs);nt=sum(r['tp'] for r in imgs);nf=sum(r['fp'] for r in imgs)
        tiles[tile]=dict(images=len(imgs),gt=ng,tp=nt,fp=nf,recall=ratio(nt,ng),precision=ratio(nt,nt+nf),fp_per_image=nf/len(imgs),
            empty_images=sum(r['gt']==0 for r in imgs),empty_images_with_fp=sum(r['gt']==0 and r['fp']>0 for r in imgs),
            median_long_side_px=float(np.median([r['box_long_side_px'] for r in targets])),
            clipped_fraction=float(np.mean([r['clipped'] for r in targets if r['clipped'] is not None])),
            reasons=dict(Counter(r['miss_reason'] for r in targets)))
    failed_geometry=[r for r in rows if r['miss_reason']=='overlap_but_iou_below_0.5']
    def med(field):return float(np.median([r[field] for r in failed_geometry])) if failed_geometry else None
    geometry_summary=dict(n=len(failed_geometry),area_errors=dict(Counter(r['area_error'] for r in failed_geometry)),
        median_pred_area_ratio=med('pred_area_ratio'),median_abs_dx= float(np.median([abs(r['center_dx_fraction']) for r in failed_geometry])),
        median_abs_dy=float(np.median([abs(r['center_dy_fraction']) for r in failed_geometry])),
        median_signed_dx=med('center_dx_fraction'),median_signed_dy=med('center_dy_fraction'),
        fix_center_only=sum(r['fix_center_alone'] and not r['fix_size_alone'] for r in failed_geometry),
        fix_size_only=sum(r['fix_size_alone'] and not r['fix_center_alone'] for r in failed_geometry),
        either_alone=sum(r['fix_size_alone'] and r['fix_center_alone'] for r in failed_geometry),
        neither_alone=sum(not r['fix_size_alone'] and not r['fix_center_alone'] for r in failed_geometry))
    result=dict(**run,counts=counts,recall=ratio(counts['tp'],counts['gt']),precision=ratio(counts['tp'],counts['candidates']),
        reasons=dict(Counter(r['miss_reason'] for r in rows)),fp_types=dict(Counter(r['kind'] for r in fps)),tiles=tiles,
        geometry=geometry_summary,cache_at_300_cap_images=sum(r['cache_at_300_cap'] for r in imgrows),
        group_stats={f:summarize_targets(rows,f) for f in ['size_bin','physical_bin','position','touching_border','clipped','kind','density_bin','box_mean_bin','box_std_bin','box_dark_fraction_bin']},
        standardized={f:standardize(rows,f,groups) for f,groups in [('position',['central','outer']),('density_bin',['1','2_4','ge5']),('box_mean_bin',['low','mid','high']),('box_std_bin',['low','mid','high'])]})
    return result,rows,imgrows,fps

def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})

def human_sensitivity(rows,items):
    # Preserve exactly the changing review sidecars read by this analysis.
    for name in ['manual_boxes.reviewed.csv','review_exclusions.json']:
        (OUT/name).write_bytes((REBUILD/'manual_review'/name).read_bytes())
    (OUT/'object_identity_adjudications.json').write_bytes((REBUILD/'object_identity_adjudications.json').read_bytes())
    manual=OUT/'manual_boxes.reviewed.csv';excluded=read(OUT/'review_exclusions.json')
    with manual.open(encoding='utf-8-sig',newline='') as handle:records={r['tile']+':'+r['source_id']:r for r in csv.DictReader(handle) if r['confirmed'].lower()=='true'}
    aliases=read(OUT/'object_identity_adjudications.json')['adjudications']
    meta={i['stem']:i['meta'] for i in items};result=[]
    for r in rows:
        if r['source_id'] is None:continue
        key=r['tile']+':'+str(r['source_id'])
        if key in aliases:key=aliases[key]['canonical_key']
        if key in excluded or key not in records:continue
        rec=records[key];native=[float(rec[k]) for k in ['x0','y0','x1','y1']];m=meta[r['stem']]
        scale=1024/m['size'] if m['kind']=='center' else 1
        box=[(native[i]-m['x' if i%2==0 else 'y'])*scale for i in range(4)]
        complete=min(box[0],box[1],1024-box[2],1024-box[3])>=-.01
        if not complete:continue
        pred=r['best_pred_locked'];old_iou=r['best_iou_locked']
        # Track SAME best-old proposal; main() separately computes best-human proposal.
        human_iou=max_iou(box,[pred])[0] if pred is not None else 0
        result.append(dict(run=r['run'],stem=r['stem'],source_id=r['source_id'],canonical_key=key,
            gt_index=r['gt_index'],old_box=r['box'],human_box=box,old_iou=old_iou,human_iou_same_proposal=human_iou,
            human_to_old_area_ratio=area(box)/area(r['box']),old_to_human_box_iou=max_iou(box,[r['box']])[0]))
    return result

def training_distribution():
    rows=[];anchors=[];rebuild_records={}
    plan=read(ROOT/'data/processed/manual_yolo_20261009_v8/export_plan.json')
    # All-label distribution includes neighbors and repeated appearances.
    for label in sorted((DATA/'labels/train').glob('*.txt')):
        boxes=original_boxes(label,1024,1024)
        for b in boxes:
            cx,cy=(b[0]+b[2])/2048,(b[1]+b[3])/2048
            rows.append(dict(position='central' if .25<=cx<=.75 and .25<=cy<=.75 else 'outer',size_bin=size_bin(max(b[2]-b[0],b[3]-b[1]))))
    return dict(label_appearances=len(rows),position=dict(Counter(r['position'] for r in rows)),size=dict(Counter(r['size_bin'] for r in rows)),
        note='All train boxes including neighbors and repeated crops; not unique target count. Crop anchor is more centered than all-label distribution.')

def main():
    if OUT.exists():raise FileExistsError('Preserve prior diagnosis; select a new output version')
    OUT.mkdir(parents=True)
    archive_path=FOLDER/'completed_eval_artifacts.zip';receipt=read(archive_path.with_suffix('.receipt.json'))
    assert sha(archive_path)==receipt['sha256']
    items,features,images,cutoffs=build_features()
    write_csv(OUT/'ground_truth_features.csv',features)
    print('FEATURES',len(features),'mapped',sum(r['source_id'] is not None for r in features),'clipped',sum(r['clipped'] is True for r in features),flush=True)
    allrows=[];allimages=[];allfps=[];results=[];fixed=[];pools={}
    with zipfile.ZipFile(archive_path) as archive:
        for run in receipt['runs']:
            blob=archive.read(run['key']+'/predictions.json');assert hashlib.sha256(blob).hexdigest()==run['predictions_sha256']
            predictions=json.loads(blob);assert set(predictions)==set(images);pools[run['key']]=predictions
            expected=json.loads(archive.read(run['key']+'/test_metrics.json'))['operating_point']
            result,rows,imgrows,fps=analyze_run(run,predictions,features,items,images,expected)
            results.append(result);allrows+=rows;allimages+=imgrows;allfps+=fps
            fixed_run=dict(run,key=run['key']+'_common_conf_0.1',threshold=.1)
            # Shared-confidence counts do not replace validation-locked metrics.
            counts=Counter()
            for item in items:
                pb,ps,pc=[np.asarray(v,np.float32) for v in predictions[item['stem']]]
                hits=match_image([(0,*b) for b in item['box_list']],pb.reshape(-1,4),ps,pc,.5,.1,False)
                counts['tp']+=sum(h[1] for h in hits);counts['candidates']+=int(np.sum(ps>=.1));counts['gt']+=len(hits);counts['images']+=1
            counts['fp']=counts['candidates']-counts['tp'];counts['fn']=counts['gt']-counts['tp']
            fr,_,_,_=analyze_run(fixed_run,predictions,features,items,images,counts);fixed.append(fr)
            print('RUN',run['key'],result['counts'],'misses',result['reasons'],flush=True)
    human=human_sensitivity(allrows,items)
    for r in human:
        pb,ps,_=[np.asarray(v,np.float32) for v in pools[r['run']][r['stem']]]
        threshold=next(x['threshold'] for x in receipt['runs'] if x['key']==r['run'])
        r['human_best_iou_locked']=max_iou(r['human_box'],pb.reshape(-1,4)[ps>=threshold])[0]
    hs={}
    for run in receipt['runs']:
        subset=[r for r in human if r['run']==run['key']]
        hs[run['key']]=dict(appearances=len(subset),unique_canonical_objects=len({r['canonical_key'] for r in subset}),
            old_localized=sum(r['old_iou']>=.5 for r in subset),human_localized=sum(r['human_best_iou_locked']>=.5 for r in subset),
            old_yes_human_no=sum(r['old_iou']>=.5 and r['human_best_iou_locked']<.5 for r in subset),
            old_no_human_yes=sum(r['old_iou']<.5 and r['human_best_iou_locked']>=.5 for r in subset),
            median_human_to_old_area_ratio=float(np.median([r['human_to_old_area_ratio'] for r in subset])) if subset else None)
    unique_sha={r['image_sha256'] for r in images.values()}
    # Stable image identity deduplication is reported as sensitivity, not used to
    # silently change the 497-image reproduction.
    canonical={};duplicate_stems=[]
    for stem,r in images.items():
        if r['image_sha256'] in canonical:duplicate_stems.append(stem)
        else:canonical[r['image_sha256']]=stem
    dedup={}
    for run in receipt['runs']:
        im=[r for r in allimages if r['run']==run['key'] and r['stem'] not in duplicate_stems]
        dedup[run['key']]=dict(images=len(im),gt=sum(r['gt'] for r in im),tp=sum(r['tp'] for r in im),fp=sum(r['fp'] for r in im),
            recall=ratio(sum(r['tp'] for r in im),sum(r['gt'] for r in im)))
    summary=dict(status='complete_read_only_cached_diagnosis',archive_sha256=receipt['sha256'],runs=results,common_confidence_0_1=fixed,
        shp_sha256=sha(SHP),analysis_code_sha256=sha(Path(__file__)),
        test=dict(images=len(items),unique_images=len(unique_sha),gt_appearances=len(features),
            mapped_gt_appearances=sum(r['source_id'] is not None for r in features),
            unique_mapped_source_ids=len({r['source_id'] for r in features if r['source_id'] is not None}),
            clipped=sum(r['clipped'] is True for r in features),
            complete=sum(r['clipped'] is False for r in features),image_duplicates=duplicate_stems),
        photometric_cutoffs=cutoffs,training_all_label_distribution=training_distribution(),
        human_box_sensitivity=hs,image_dedup_sensitivity=dedup,
        identity_adjudications_sha256=sha(OUT/'object_identity_adjudications.json'),
        manual_review_sha256=sha(OUT/'manual_boxes.reviewed.csv'),exclusions_sha256=sha(OUT/'review_exclusions.json'),
        caveats=['Old SHP labels and clipped appearances are provisional scoring truth; unmatched predictions may contain unlabeled genuine landslides.',
            'All geometric diagnostics operate on saved post-NMS conf>=0.001 pools; no pre-NMS logits or unseen detections are recoverable.',
            'Locked thresholds differ by run; shared confidence 0.1 is descriptive only, not retuned on test.',
            'Long-side pixel bins refer to resized 1024 images; physical bins use mapped source AABB extent, not a field landslide length.',
            'Brightness and texture are proxies from grayscale pixels inside old boxes, not physical illumination or geological visibility labels.',
            'Correlation is confounded by tile, scale, clipping, crop construction and label quality; common-stratum standardization is descriptive.',
            'Repeated crops of the same source and six models are not independent samples; no pooled significance tests are claimed.',
            'Human sensitivity includes only reviewed objects whose full new box fits the unchanged old image; this selected subset is not rebuilt evaluation performance.',
            'The label matcher is GT-order and confidence-order greedy; its conflict category includes duplicate labels and genuine crowded objects.'])
    atomic(OUT/'summary.json',summary)
    write_csv(OUT/'per_gt_all_six_runs.csv',allrows);write_csv(OUT/'per_image_all_six_runs.csv',allimages)
    write_csv(OUT/'unmatched_predictions.csv',allfps);write_csv(OUT/'human_box_sensitivity.csv',human)
    atomic(OUT/'input_image_receipt.json',list(images.values()))
    print('OUT',OUT,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    OUT=parser.parse_args().out.resolve()
    main()
