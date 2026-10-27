"""Diagnostic review of high-score old-test unmatched regions; never auto-negative.

Freeze the complete >=0.5 selection before review, group overlapping native
regions across six runs, and keep every contributing prediction in the manifest.
Human decisions are separate sidecars, not training or evaluation label edits.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import threading
from urllib.parse import parse_qs,urlparse

import numpy as np
from PIL import Image
import shapefile
import tifffile
from audit_yolo_splits import parse_patch_name
from diagnose_six_cached_runs import max_iou
from make_centered_review_pack import (SHP,discover_tifs,read_geotransform,
    map_source_records,_record_for_tile,_tile_bounds,_intersects)
from rebuild_manual_eval_review import PACK
from review_eval_labels import ROOT, read, sha, atomic, original_boxes

DIAG=ROOT/'runs/ablations/six_run_error_diagnosis_20261011_v2'
OUT=ROOT/'runs/ablations/high_score_unmatched_review_20261011_v1'
DATA=ROOT/'data/processed/manual_yolo_20261009_v8/manual'
STATUSES={'known_localization','known_duplicate','missing_landslide','true_background','uncertain'}


def source_geometry(rep,width,height):
    if 'window_px' in rep:
        a,b,c,d=rep['window_px']
        return [a,b],[width/(c-a),height/(d-b)]
    meta=parse_patch_name(rep['stem'])
    return [meta['x'],meta['y']],([width/meta['size'],height/meta['size']] if meta['kind']=='center' else [1.,1.])


def group_regions(rows,threshold=.5):
    """Deterministic representative-IoU groups; not physical-object adjudication."""
    groups=[]
    for r in sorted(rows,key=lambda r:(-r['score'],r['tile'],r['stem'],r['run'],r['native_box'])):
        choices=[g for g in groups if g['tile']==r['tile'] and max_iou(r['native_box'],[g['native_box']])[0]>=threshold]
        if choices:
            g=max(choices,key=lambda g:max_iou(r['native_box'],[g['native_box']])[0]);g['members'].append(r)
        else:groups.append(dict(tile=r['tile'],native_box=r['native_box'],score=r['score'],representative=r,members=[r]))
    return groups


def prepare(out):
    receipt=read(DIAG/'analysis_receipt.json');csv_path=DIAG/'unmatched_predictions.csv'
    if sha(csv_path)!=receipt['files'][csv_path.name]:raise ValueError('Diagnosis source CSV changed')
    if sha(DIAG/'input_image_receipt.json')!=receipt['files']['input_image_receipt.json']:raise ValueError('Image receipt changed')
    images=read(DIAG/'input_image_receipt.json')
    # The source receipt maps each old test stem to immutable image/label hashes.
    if isinstance(images,list):images={r['stem']:r for r in images}
    rows=[]
    with csv_path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if float(r['score'])<.5:continue
            stem=r['stem'];path=DATA/'images/test'/(stem+'.png')
            with Image.open(path) as im:w,h=im.size
            meta=parse_patch_name(stem);sx=w/meta['size'] if meta['kind']=='center' else 1.;sy=h/meta['size'] if meta['kind']=='center' else 1.
            box=json.loads(r['box']);native=[box[i]/(sx if i%2==0 else sy)+meta['x' if i%2==0 else 'y'] for i in range(4)]
            rows.append(dict(run=r['run'],stem=stem,tile=r['tile'],box=box,native_box=native,
                score=float(r['score']),old_max_gt_iou=float(r['max_gt_iou']),old_relation=r['kind'],image=str(path)))
    groups=group_regions(rows)
    priority={'E-088_N-08':0,'E-064_N-08':1}
    groups.sort(key=lambda g:(priority.get(g['tile'],2),-g['score'],g['tile'],g['native_box']))
    plan=read(PACK/'evaluation_plan.json');native=discover_tifs(Path(plan['ctx_cache']))
    tifs={}
    for tile in {g['tile'] for g in groups}:
        path=native[tile];gt,w,h=read_geotransform(path)
        tifs[tile]=dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,
                        width=w,height=h,geotransform=list(gt))
    for n,g in enumerate(groups):
        g['candidate_id']=f'U{n+1:04d}';g['batch']=n//50+1
        g['models']=sorted({r['run'] for r in g['members']})
        path=Path(g['representative']['image']);lp=DATA/'labels/test'/(path.stem+'.txt')
        hashes=images[path.stem]
        if sha(path)!=hashes['image_sha256'] or sha(lp)!=hashes['label_sha256']:raise ValueError('Frozen old image/label changed')
        g['original_image_sha256']=sha(path);g['original_label_sha256']=sha(lp)
    out.mkdir(parents=True,exist_ok=False);(out/'inputs').mkdir()
    for source,name in [(SHP,'reference.shp'),(SHP.with_suffix('.shx'),'reference.shx'),(SHP.with_suffix('.dbf'),'reference.dbf'),
        (PACK/'manual_review/manual_boxes.reviewed.csv','manual.csv')]:
        (out/'inputs'/name).write_bytes(source.read_bytes())
    manifest=dict(schema_version=1,role='diagnostic_only_never_training_or_auto_ground_truth',
        created_utc=datetime.now(timezone.utc).isoformat(),selection=dict(min_confidence=.5,
        pool='All locked-threshold unmatched old-test predictions in six runs',
        grouping='Representative native AABB IoU >=0.5; overlapping region groups, not confirmed unique landslides',
        ordering='E-088_N-08, E-064_N-08, other tiles; score descending within tile priority'),
        source_csv_sha256=sha(csv_path),raw_prediction_count=len(rows),region_count=len(groups),
        tifs=tifs,items=groups,input_hashes={p.name:sha(p) for p in (out/'inputs').iterdir()},
        limitations=['Model-selected diagnostic regions; cannot estimate unbiased precision from this review alone.',
                    'true_background is a local diagnostic decision, never a training negative from held-out tiles.',
                    'missing_landslide decisions require separate complete-boundary annotation before changing evaluation answers.'])
    atomic(out/'manifest.json',manifest);atomic(out/'decisions.json',{'records':{}})
    counts=dict(Counter(g['tile'] for g in groups))
    (out/'README.md').write_text(f'# 高分额外候选复核\n\n分数≥0.5的{len(rows)}次预测，按原图幅坐标重叠合并为{len(groups)}个候选区域，{math.ceil(len(groups)/50)}批。\n图幅：{counts}\n\n橙框：模型候选；蓝框：旧测试标签；粉线：SHP；绿框：本次人工完整框。\n先判断橙框属于已知滑坡的偏框／重复框、漏标滑坡、真实背景，还是看不清。\n这不是要求重新画一遍；保存分类判断即可。所有判断单独保存，不自动改标签、不进训练。\n\n本页是诊断清单，可以留到当前完整目标审核之后。\n',encoding='utf-8')
    return manifest


def prepare_rebuilt(out,frozen_folder,results_folder):
    from reevaluate_rebuilt_six import validate_frozen
    from eval_recall_report import match_image
    from rebuild_manual_eval_review import verify_plan
    frozen=validate_frozen(frozen_folder);verify_plan(frozen)
    results=read(results_folder/'results.json')
    if results['frozen_sha256']!=sha(frozen_folder/'frozen.json'):raise ValueError('Rebuilt results refer to different pixels')
    cohort={r['stem']:r for r in frozen['items'] if r['arm']=='offset' and r['split']=='test'}
    rows=[];prediction_hashes={}
    for r in results['results']:
        if (r['arm'],r['split'])!=('offset','test'):continue
        path=results_folder/r['run']/'offset/test/predictions.json'
        if sha(path)!=r['predictions_sha256']:raise ValueError('Rebuilt prediction cache changed')
        prediction_hashes[str(path)]=sha(path)
        for stem,values in read(path).items():
            item=cohort[stem];boxes=item['human']['boxes']
            pb,ps,pc=[np.asarray(a,np.float32) for a in values];pb=pb.reshape(-1,4)
            keep=ps>=r['threshold'];kept=pb[keep];scores=ps[keep]
            hits=match_image([(0,*b) for b in boxes],pb,ps,pc,.5,r['threshold'],class_aware=False)
            used={h[4] for h in hits if h[1]}
            origin,scales=source_geometry(item,item['width'],item['height'])
            for j,box in enumerate(kept):
                if j in used or scores[j]<.5:continue
                native=[float(box[i])/scales[i%2]+origin[i%2] for i in range(4)]
                rows.append(dict(run=r['run'],stem=stem,tile=item['tile'],box=box.tolist(),native_box=native,
                    score=float(scores[j]),reference_max_gt_iou=max_iou(box,boxes)[0],
                    image=str(Path(item['image']).resolve()),label=str(Path(item['label']).resolve()),window_px=item['window_px']))
    if len(prediction_hashes)!=6:raise ValueError('Six rebuilt primary test runs required')
    groups=group_regions(rows);priority={'E-088_N-08':0,'E-064_N-08':1}
    groups.sort(key=lambda g:(priority.get(g['tile'],2),-g['score'],g['tile'],g['native_box']))
    prep=read(Path(frozen['preparation'])/'plan.json');tifs={}
    for tile in {g['tile'] for g in groups}:
        s=prep['tile_sources'][tile]
        tifs[tile]=dict(path=s['path'],bytes=s['bytes'],mtime_ns=s['mtime_ns'],
                       width=s['tile_size_px'][0],height=s['tile_size_px'][1],geotransform=s['geotransform'])
    for n,g in enumerate(groups):
        g['candidate_id']=f'U{n+1:04d}';g['batch']=n//50+1;g['models']=sorted({r['run'] for r in g['members']})
        rep=g['representative'];item=cohort[rep['stem']]
        g['original_image_sha256']=item['image_sha256'];g['original_label_sha256']=item['label_sha256']
    out.mkdir(parents=True,exist_ok=False);(out/'inputs').mkdir()
    for k,name in [('shp','reference.shp'),('shx','reference.shx'),('dbf','reference.dbf'),('manual','manual.csv')]:
        src=prep['inputs'][k]
        if sha(src['snapshot'])!=src['sha256']:raise ValueError('Frozen reference snapshot changed')
        (out/'inputs'/name).write_bytes(Path(src['snapshot']).read_bytes())
    manifest=dict(schema_version=1,role='diagnostic_only_never_training_or_auto_ground_truth',
        created_utc=datetime.now(timezone.utc).isoformat(),source_label='本轮冻结的完整人工参考框',
        frozen_sha256=sha(frozen_folder/'frozen.json'),results_sha256=sha(results_folder/'results.json'),
        prediction_hashes=prediction_hashes,selection=dict(min_confidence=.5,
        pool='All six rebuilt primary-offset test runs, unmatched at original locked thresholds',
        grouping='Representative native AABB IoU >=0.5; overlapping regions, not confirmed unique landslides',
        ordering='E-088_N-08, E-064_N-08, other tiles; score descending within tile priority'),
        raw_prediction_count=len(rows),region_count=len(groups),tifs=tifs,items=groups,
        input_hashes={p.name:sha(p) for p in (out/'inputs').iterdir()},
        limitations=['Review applies to the representative orange box, not automatically every member.',
                    'Model-selected diagnostic subset, not unbiased precision estimation.',
                    'No review decision changes frozen evaluation answers or enters training automatically.'])
    atomic(out/'manifest.json',manifest);atomic(out/'decisions.json',{'records':{}})
    (out/'README.md').write_text(f'# 本轮人工评估：高分额外候选\n\n当前六组最佳模型在新冻结偏移测试图上的未匹配候选。分数≥0.5的{len(rows)}次预测合并为{len(groups)}个重叠区域，先看首批50个。\n\n判断代表橙框属于偏框、重复／碎框、漏标滑坡、背景，还是看不清。所有判断单独保存，不自动修改已冻结标签、不进入训练。\n\n蓝框为本轮完整人工评分标签，粉线为原SHP，绿框为人工审核快照。可切换原图上下文与模型实际测试裁片。\n',encoding='utf-8')
    return manifest


class Review:
    def __init__(self,folder):
        self.folder=folder;self.manifest=read(folder/'manifest.json');self.digest=sha(folder/'manifest.json')
        self.items={r['candidate_id']:r for r in self.manifest['items']};self.lock=threading.Lock()
        for n,h in self.manifest['input_hashes'].items():
            if sha(folder/'inputs'/n)!=h:raise ValueError('Frozen diagnostic reference changed')
        with shapefile.Reader(str(folder/'inputs/reference.shp')) as reader:sources=map_source_records(reader.shapeRecords())
        self.references={}
        for tile,t in self.manifest['tifs'].items():
            gt=t['geotransform'];bounds=_tile_bounds(gt,t['width'],t['height'])
            self.references[tile]=[r for sid,s in sources.items() if (r:=_record_for_tile(sid,s,gt,bounds)) is not None]
        with (folder/'inputs/manual.csv').open(encoding='utf-8-sig',newline='') as f:self.manual=list(csv.DictReader(f))

    @lru_cache(maxsize=4)
    def view(self,cid,mode='native'):
        item=self.items[cid];rep=item['representative'];tile=item['tile'];t=self.manifest['tifs'][tile]
        if mode=='original':
            path=Path(rep['image'])
            if sha(path)!=item['original_image_sha256']:raise ValueError('Original test image changed')
            lp=Path(rep.get('label',DATA/'labels/test'/(rep['stem']+'.txt')))
            if sha(lp)!=item['original_label_sha256']:raise ValueError('Original test labels changed')
            with Image.open(path) as im:array=np.asarray(im.convert('L'))
            h,w=array.shape;origin,scales=source_geometry(rep,w,h)
            old=original_boxes(lp,w,h)
        elif mode=='native':
            b=item['native_box'];side=math.ceil(max(2048,max(b[2]-b[0],b[3]-b[1])*3))
            w,h=min(side,t['width']),min(side,t['height'])
            x=max(0,min(round((b[0]+b[2]-w)/2),t['width']-w));y=max(0,min(round((b[1]+b[3]-h)/2),t['height']-h))
            path=Path(t['path'])
            if path.stat().st_size!=t['bytes'] or path.stat().st_mtime_ns!=t['mtime_ns']:raise ValueError('Native TIFF changed')
            stride=max(1,math.ceil(max(w,h)/4096));array=np.asarray(tifffile.memmap(path)[y:y+h:stride,x:x+w:stride]).copy()
            origin=[x,y];scales=[1/stride,1/stride];h,w=array.shape;old=[]
        else:raise ValueError('Unknown view mode')
        transform=lambda b:[(v-origin[i%2])*scales[i%2] for i,v in enumerate(b)]
        frame=[origin[0],origin[1],origin[0]+w/scales[0],origin[1]+h/scales[1]]
        polygons=[dict(source_id=r['source_id'],parts=[[[ (x-origin[0])*scales[0],(y-origin[1])*scales[1]] for x,y in part] for part in r['polygon_px']]) for r in self.references[tile] if _intersects(tuple(r['aabb_px']),tuple(frame))]
        manual=[dict(source_id=int(r['source_id']),box=transform([float(r[k]) for k in ('x0','y0','x1','y1')])) for r in self.manual if r['tile']==tile and r['confirmed']=='true']
        meta=dict(candidate_id=cid,mode=mode,width=w,height=h,prediction_box=transform(item['native_box']),
            polygons=polygons,manual=manual,old_boxes=old,tile=tile,score=item['score'],models=item['models'],member_count=len(item['members']))
        buffer=io.BytesIO();Image.fromarray(array).save(buffer,format='PNG')
        return meta,buffer.getvalue()

    def save(self,payload):
        cid=payload['candidate_id']
        if cid not in self.items or payload.get('status') not in STATUSES:raise ValueError('Unknown candidate/decision')
        if payload.get('confirmed') is not True:raise ValueError('请先确认已经查看橙框')
        if not str(payload.get('reviewer','')).strip():raise ValueError('请填写审核人')
        with self.lock:
            if sha(self.folder/'manifest.json')!=self.digest:raise ValueError('Diagnostic manifest changed')
            state=read(self.folder/'decisions.json');prior=state['records'].get(cid);revision=prior['revision'] if prior else 0
            if payload.get('revision')!=revision:raise ValueError('此候选已在另一个页面保存，请刷新')
            result=dict(candidate_id=cid,status=payload['status'],reviewer=str(payload['reviewer']).strip(),
                reason=str(payload.get('reason','')).strip(),confirmed=True,revision=revision+1,
                saved_utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=self.digest,
                decision_scope='representative_candidate_only',training_eligible=False,automatic_label_change=False)
            with (self.folder/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'previous':prior,'saved':result},ensure_ascii=False)+'\n')
            state['records'][cid]=result;atomic(self.folder/'decisions.json',state)
            return result


def serve(folder,port):
    review=Review(folder)
    class Handler(BaseHTTPRequestHandler):
        def response(self,value,kind='application/json; charset=utf-8',status=200):
            blob=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(blob)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(blob)
        def do_GET(self):
            try:
                url=urlparse(self.path);q=parse_qs(url.query)
                if url.path=='/':return self.response(Path(__file__).with_suffix('.html').read_bytes(),'text/html; charset=utf-8')
                if url.path=='/api/state':return self.response({'manifest':review.manifest,'records':read(folder/'decisions.json')['records']})
                if url.path in ('/api/view','/api/image'):
                    meta,png=review.view(q['id'][0],q.get('mode',['native'])[0]);return self.response(meta) if url.path=='/api/view' else self.response(png,'image/png')
                self.response({'error':'not found'},status=404)
            except (ValueError,KeyError) as e:self.response({'error':str(e)},status=400)
        def do_POST(self):
            try:
                if self.path!='/api/save' or self.headers.get('X-Manual-Review')!='1':raise ValueError('Invalid save request')
                if self.headers.get('Origin') not in (None,'http://'+self.headers.get('Host','')):raise ValueError('Origin mismatch')
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=16384:raise ValueError('Invalid payload size')
                self.response(review.save(json.loads(self.rfile.read(n))))
            except (ValueError,KeyError) as e:self.response({'error':str(e)},status=400)
    print(f'DIAGNOSTIC_REVIEW http://127.0.0.1:{port}/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','prepare-rebuilt','serve']);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--port',type=int,default=8775)
    p.add_argument('--frozen',type=Path,default=ROOT/'data/processed/manual_eval_frozen_20261011_v1')
    p.add_argument('--results',type=Path,default=ROOT/'runs/ablations/rebuilt_six_reevaluation_20261011_v1');a=p.parse_args()
    if a.mode=='serve':serve(a.out,a.port)
    else:
        m=prepare_rebuilt(a.out,a.frozen,a.results) if a.mode=='prepare-rebuilt' else prepare(a.out)
        print(json.dumps({'raw_predictions':m['raw_prediction_count'],'regions':m['region_count']}))
