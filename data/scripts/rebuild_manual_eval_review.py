"""Independent full-context val/test review; reuse original spatial assignments."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
import os
import math
from pathlib import Path
import shutil
import tempfile
import time
from http.server import ThreadingHTTPServer
import shapefile
from check_tile_connectivity import tiles_of_bbox
from make_centered_review_pack import (SHP, map_source_records, read_geotransform,
    _record_for_tile, _tile_bounds, make_review_row)
from fetch_v2_train_tiles import download_zip, extract_verified, existing_tif, MANIFEST
from manual_box_review import Review, ReviewBatch, handler_for, HTML, has_complete_geometry
from review_eval_labels import ROOT, read, atomic, sha

PACK=ROOT/'data/processed/manual_eval_rebuild_20261011_v1'
CACHE=Path(tempfile.gettempdir())/'Mars-KG-Vis-eval-rebuild-20261011'

def containing_window(box, side, width, height, offset=(.5,.5)):
    """Move native crop origins only within bounds that keep the full box visible."""
    a,b,c,d=map(float,box)
    if not all(math.isfinite(v) for v in [a,b,c,d,*offset]) or not 0<=a<c<=width or not 0<=b<d<=height:
        raise ValueError('Invalid complete target bounds')
    if type(side) is not int or side<1 or any(not 0<=v<=1 for v in offset):
        raise ValueError('Invalid crop window')
    w,h=min(side,width),min(side,height)
    xlo,xhi=max(0,math.ceil(c-w)),min(width-w,math.floor(a))
    ylo,yhi=max(0,math.ceil(d-h)),min(height-h,math.floor(b))
    if xlo>xhi or ylo>yhi:
        raise ValueError('Window would truncate the full target; increase its size')
    x=round(xlo+(xhi-xlo)*offset[0]);y=round(ylo+(yhi-ylo)*offset[1])
    return [x,y,x+w,y+h]

def prepare():
    path=PACK/'evaluation_plan.json'
    if path.exists(): return read(path)
    base=ROOT/'data/processed/manual_yolo_20261009_v8/manual/images'
    splits={s:{'_'.join(p.stem.split('_')[:2]) for p in (base/s).glob('*.png')} for s in ['train','val','test']}
    assert len(splits['val'])==6 and len(splits['test'])==4
    audit=ROOT/'runs/ablations/train_eval_leakage_20261009/audit.json'
    assert not read(audit)['leakage_detected']
    with shapefile.Reader(str(SHP)) as reader: sources=map_source_records(reader.shapeRecords())
    targets=[]
    for sid,source in sources.items():
        hits=tiles_of_bbox(*source.shape.bbox)
        assignment=[s for s in ['val','test'] if hits & splits[s]]
        if not assignment: continue
        if len(assignment)!=1 or hits & splits['train']:
            raise ValueError(f'Spatial split conflict: {sid}')
        split=assignment[0]
        targets.append(dict(source_id=sid,split=split,review_tile=sorted(hits & splits[split])[0],
            bbox_lonlat=list(source.shape.bbox),context_tiles=sorted(hits),cross_tile=len(hits)>1))
    protected=[audit,SHP,SHP.with_suffix('.dbf'),SHP.with_suffix('.shx'),
               ROOT/'data/processed/manual_yolo_20261009_v8/export_plan.json']
    result=dict(schema_version=1,role='evaluation_only_never_training',
        tile_splits={t:s for s in ['val','test'] for t in sorted(splits[s])},targets=targets,
        candidate_counts=dict(Counter(t['split'] for t in targets)),cross_tile_candidates=sum(t['cross_tile'] for t in targets),
        ctx_cache=str(CACHE),protected_inputs={str(p):sha(p) for p in protected},
        policies=dict(annotation='Full identifiable landslide in large native context; background crops separately confirmed.',
            geometry='Cross-tile targets remain pending until required neighboring pixels are mosaicked; never auto-negative.',
            primary_crop='After human boxes: deterministic shifted windows constrained to contain full targets; centers vary.',
            centered_crop='Separate diagnostic arm on the same independent targets, not the primary score.',
            adjacent='All recognizable neighbor targets must be labeled or the crop held.',
            negatives='Empty YOLO labels for exact human-confirmed rectangular background crops; no background class.',
            evaluation='Freeze pixels and manual labels before inference; reuse six best checkpoints; no retraining.',
            thresholds='Keep original validation locks for initial comparison; any new threshold uses val only and is separately reported.',
            history='Existing observed val/test regions; this is a rebuilt development evaluation, not fresh blind final test.'))
    PACK.mkdir(parents=True,exist_ok=True);atomic(path,result)
    return result

def verify_plan(plan):
    for path,digest in plan['protected_inputs'].items():
        if sha(path)!=digest: raise ValueError(f'Protected input changed: {path}')

def restore(tile):
    plan=prepare();verify_plan(plan)
    if tile not in plan['tile_splits']: raise ValueError('Not a fixed evaluation tile')
    root=Path(plan['ctx_cache']);root.mkdir(parents=True,exist_ok=True)
    catalog={r['tile']:r for r in read(MANIFEST)};row=catalog[tile]
    archive=f'MurrayLab_GlobalCTXMosaic_V01_{tile}.zip'
    if row['zip']!=archive or row['url']!=f'https://murray-lab.caltech.edu/CTX/V01/tiles/{archive}':
        raise ValueError('Unexpected source URL')
    progress=PACK/f'restore_{tile}.json'
    found=existing_tif(root,tile)
    if not found:
        if shutil.disk_usage(root).free<row['size_bytes']*3+3*(1<<30):
            raise OSError('Insufficient space for bounded ZIP restore')
        atomic(progress,dict(status='downloading',tile=tile,ctx=str(root),zip_bytes=row['size_bytes']))
        print(f'DOWNLOAD {tile} {row["size_bytes"]} bytes -> {root}',flush=True)
        zipped=download_zip(row,root,4)
        atomic(progress,dict(status='extracting',tile=tile))
        found=extract_verified(row,zipped,root)
    # Build metadata only. The editor reads native pixels and expands the view on demand.
    paths=list(root.glob(f'*/MurrayLab_CTX_V01_{tile}_Mosaic.tif'))
    assert len(paths)==1
    gt,width,height=read_geotransform(paths[0]);bounds=_tile_bounds(gt,width,height)
    with shapefile.Reader(str(SHP)) as reader:sources=map_source_records(reader.shapeRecords())
    records=[r for sid,s in sources.items() if (r:=_record_for_tile(sid,s,gt,bounds)) is not None]
    candidates=[t for t in plan['targets'] if t['review_tile']==tile]
    rows=[make_review_row(tile,t['source_id'],records,(width,height),margin_factor=2.,min_window_px=2048,max_window_px=72000) for t in candidates]
    out=PACK/'tiles'/tile;out.mkdir(parents=True,exist_ok=True)
    target=out/'centered_review_manifest.csv'
    if not target.exists():
        with target.open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader()
            writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
    batch=dict(batch=1,items=[dict(tile=tile,source_id=r['source_id']) for r in rows if has_complete_geometry({k:str(v) for k,v in r.items()})],
        pending_cross_tile=[t for t in candidates if t['cross_tile']])
    atomic(out/'batch.json',batch)
    verify_plan(plan)
    atomic(progress,dict(status='ready',tile=tile,candidates=len(rows),ready=len(batch['items']),
        pending_cross_tile=len(batch['pending_cross_tile']),ctx=str(root),verified=found))
    print(f'READY {tile}: {len(batch["items"])} review targets',flush=True)
    return batch

def ready_groups():
    """Append stable groups; never renumber an already presented review batch."""
    path=PACK/'review_batches.json'
    groups=read(path)['groups'] if path.exists() else []
    registered={i['tile'] for g in groups for i in g['items']}
    for batch_path in sorted((PACK/'tiles').glob('*/batch.json')):
        batch=read(batch_path)
        if not batch['items'] or batch['items'][0]['tile'] in registered:
            continue
        for start in range(0,len(batch['items']),100):
            groups.append(dict(batch=len(groups)+1,items=batch['items'][start:start+100]))
        registered.add(batch['items'][0]['tile'])
    identities=[(i['tile'],i['source_id']) for g in groups for i in g['items']]
    if len(set(identities))!=len(identities): raise ValueError('Duplicate review batch targets')
    if not groups: raise ValueError('No restored review targets')
    atomic(path,dict(groups=groups))
    return groups

def restore_all():
    """Resume the fixed evaluation queue with one tile and four ranges at a time."""
    plan=prepare();verify_plan(plan)
    wanted=sorted(plan['tile_splits'],key=lambda t:(plan['tile_splits'][t]!='test',t))
    status=dict(status='running',wanted=wanted,tiles={},failed={},
        purpose='evaluation review only; no automatic annotation or training',
        connections_per_tile=4,ctx=plan['ctx_cache'])
    report=PACK/'restore_queue.json'
    def save():
        status['updated_epoch']=time.time();atomic(report,status)
    save()
    for tile in wanted:
        status['current_tile']=tile;status['tiles'][tile]={'status':'checking'};save()
        try:
            restore(tile)
            groups=ready_groups()
            status['tiles'][tile]=read(PACK/f'restore_{tile}.json')
            status['review_batches']=len(groups)
            status['review_targets']=sum(len(g['items']) for g in groups)
        except OSError as exc:
            status['tiles'][tile]={'status':'failed','error':str(exc)}
            status['failed'][tile]=str(exc)
            print(f'FAILED {tile}: {exc}',flush=True)
            # Space failures cannot be fixed by moving to another tile.
            if 'space' in str(exc).lower():
                status['status']='stopped_for_space';save();return
        except Exception as exc:
            status['tiles'][tile]={'status':'failed','error':str(exc)}
            status['failed'][tile]=str(exc)
            print(f'FAILED {tile}: {exc}',flush=True)
        save()
    status['current_tile']=None
    status['status']='complete' if not status['failed'] else 'complete_with_failures'
    verify_plan(plan);save()
    print(f'QUEUE {status["status"]}: {status.get("review_targets",0)} review targets',flush=True)

class GrowingEvaluationReview(ReviewBatch):
    """Publish verified queue additions when the page reloads, keeping saved data."""
    def __init__(self,*args,ctx=None,**kwargs):
        self.ctx=ctx
        super().__init__(*args,ctx=ctx,**kwargs)

    def batch_info(self):
        with self.lock:
            path=self.pack/'review_batches.json'
            groups=read(path)['groups']
            if groups!=self.groups:
                items=[i for g in groups for i in g['items']]
                previous={(i['tile'],int(i['source_id'])) for i in self.items}
                incoming={(i['tile'],int(i['source_id'])) for i in items}
                if not previous<=incoming or len(incoming)!=len(items):
                    raise ValueError('审核队列不得移除或重复已发布目标')
                selections={}
                for i in items:selections.setdefault(i['tile'],[]).append(int(i['source_id']))
                for tile,ids in selections.items():
                    if self.evaluation_splits.get(tile) not in {'val','test'}:
                        raise ValueError('非评估图幅不得加入队列')
                    if tile not in self.reviews:
                        addition=Review(tile,ids,self.pack,self.ctx)
                        addition.lock=self.lock;self.reviews[tile]=addition
                    elif set(ids)!=set(self.reviews[tile].rows):
                        raise ValueError('已发布图幅的目标清单发生变化')
                self.groups,self.items=groups,items
            return super().batch_info()

def serve(tile,port=8774):
    plan=prepare();verify_plan(plan)
    groups=ready_groups()
    selected=next(g for g in groups if any(i['tile']==tile for i in g['items']))
    items=[i for g in groups for i in g['items']]
    review=GrowingEvaluationReview(items,pack=PACK,ctx=Path(plan['ctx_cache']),batch_number=selected['batch'],groups=groups,dataset_role='evaluation')
    html=HTML.read_text(encoding='utf-8').replace('滑坡人工修框','验证／测试集完整范围审核')
    html=html.replace('才可生成训练标签','才可生成评估标签').replace('绿色框为滑坡，','本页仅用于验证／测试，不进入训练。绿色框为滑坡，')
    html=html.replace('<option value="visible_core">仅清晰核心区（单独保留）</option>','')
    html=html.replace('特征不明显，不作为正样本','看不清，完整范围无法核准').replace('不作为正样本</button>','看不清，暂扣</button>')
    html=html.replace('历史批次','审核批次')
    html=html.replace('if(document.title.includes(',"if(batch.dataset_role==='evaluation'||document.title.includes(")
    html=html.replace('function populateTargets()',"function setEvaluationHeading(){document.title='验证／测试集完整范围审核';document.querySelector('h1').textContent='验证／测试集完整范围审核 · 第 '+batch.batch_number+' 批'}\nfunction populateTargets()")
    html=html.replace('populateTargets();load()', 'populateTargets();setEvaluationHeading();load()')
    html=html.replace('populateTargets();await load()', 'populateTargets();setEvaluationHeading();await load()')
    local_html=PACK/'review.html';local_html.write_text(html,encoding='utf-8')
    # Handler reads this module-level HTML path; keep the training file untouched.
    import manual_box_review as editor
    editor.HTML=local_html
    print(f'EVAL_REVIEW http://127.0.0.1:{port}/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),handler_for(review)).serve_forever()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['plan','restore','restore-all','serve','restore-and-serve'])
    parser.add_argument('--tile',default='E-056_N-16');parser.add_argument('--port',type=int,default=8774)
    args=parser.parse_args()
    if args.mode=='plan':
        p=prepare();print(json.dumps(dict(counts=p['candidate_counts'],cross_tile=p['cross_tile_candidates'],pack=str(PACK)),ensure_ascii=False))
    if args.mode in {'restore','restore-and-serve'}:restore(args.tile)
    if args.mode=='restore-all':restore_all()
    if args.mode in {'serve','restore-and-serve'}:serve(args.tile,args.port)
