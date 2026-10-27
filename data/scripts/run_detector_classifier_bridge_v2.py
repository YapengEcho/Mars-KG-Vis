"""Validation-only bridge v2: correct interpolation to native training BILINEAR.

Restore six catalogued CTX sources into a dedicated C: temporary cache. Rebuild
224 classifier inputs with the training crop/contrast policy. Keep all candidates,
misses and unmatched predictions in explicit localization and joint metrics.
"""
from __future__ import annotations
import argparse
from collections import Counter
import ctypes
from datetime import datetime,timezone
import hashlib
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid
import warnings
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.venv/Lib/site-packages'))
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_OFFLINE='true')
import numpy as np
from PIL import Image
import tifffile
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from sklearn.preprocessing import StandardScaler
import run_roi_context_pooling as P
import build_native_classification_dataset as N
from build_native_classification_crop_pilot import choose_window
from recover_reviewed_mosaic_samples import world_box
from make_centered_review_pack import read_geotransform
from fetch_v2_train_tiles import download_zip,extract_verified,existing_tif,MANIFEST
from eval_recall_report import match_image
from train_stage2_cls import classifier_class_weights
C=P.C
R=P.R
OUT=ROOT/'runs/experiments/detector_classifier_bridge_20261015_v2'
CACHE=Path(tempfile.gettempdir())/'Mars-KG-Vis-detector-classifier-20261015'
FROZEN=ROOT/'data/processed/manual_eval_frozen_20261011_v1/frozen.json'
DETECTOR=ROOT/'runs/ablations/full_pretrained_validation_20261014_v2'
CATALOG=ROOT/'data/raw/mars_landslides.json'
TRAIN_NATIVE=ROOT/'data/processed/manual_stage2_native_20261015_v1/manifest.json'
CLASS_NAMES=['SL','SP','FL','CL']


def utc():return datetime.now(timezone.utc).isoformat()
def write(path,value):C.write_json_atomic(path,value)


def planned_window(box,source):
    gt=source['geotransform'];w,h=source['size_px']
    try:
        win,geometry=choose_window(box,gt,[source]);geometry['context_exception']=False
    except ValueError:
        # Same boundary principle as the native training build: full target first,
        # finite available context. Never pad or silently clip the target itself.
        side=min(max(math.ceil(1.5*max(box[2]-box[0],box[3]-box[1])),1),w,h)
        lo=[max(0,math.ceil(box[2]-side)),max(0,math.ceil(box[3]-side))]
        hi=[min(w-side,math.floor(box[0])),min(h-side,math.floor(box[1]))]
        assert all(a<=b for a,b in zip(lo,hi))
        xy=[max(lo[j],min(hi[j],math.floor((box[j]+box[j+2]-side)/2))) for j in range(2)]
        win=[xy[0],xy[1],xy[0]+side,xy[1]+side]
        geometry=dict(context_exception=True,reason='available tile boundary cannot retain nominal25percent per-axis context')
    assert win[0]<=box[0]<box[2]<=win[2] and win[1]<=box[1]<box[3]<=win[3]
    return win,geometry


def native_box(box,item):
    x0,y0,x1,y1=item['window_px']
    return [x0+box[0]*(x1-x0)/item['width'],y0+box[1]*(y1-y0)/item['height'],
            x0+box[2]*(x1-x0)/item['width'],y0+box[3]*(y1-y0)/item['height']]


def prepare():
    P.check();C.new_directory(OUT)
    frozen=C.read_json(FROZEN);prep_path=ROOT/Path(frozen['preparation'])/'plan.json';prep=C.read_json(prep_path)
    assert C.sha256(prep_path)==frozen['preparation_sha256']
    detector_spec=C.read_json(DETECTOR/'protocol.json');old_results=C.read_json(DETECTOR/'comparison.json')
    assert C.sha256(FROZEN)==detector_spec['frozen_sha256']
    items=[r for r in frozen['items'] if (r['arm'],r['split'])==('offset','val')]
    samples={s['stem']:s for s in prep['samples'] if s['split']=='val'}
    train=C.verify_plan(P.SOURCE/'plan.json');train_native=C.read_json(TRAIN_NATIVE)
    train_world=np.array([s['native_window_world_m'] for s in train_native['samples']])
    train_tiles={t for r in train['rows'] for t in r['source_tiles']};val_tiles={r['tile'] for r in items}
    assert not train_tiles&val_tiles
    train_ids={int(r['source_id']) for r in train['rows']}
    catalog={int(r['source_id']):r for r in C.read_json(CATALOG)}
    source_catalog={r['tile']:r for r in C.read_json(MANIFEST)}
    sources={tile:dict(tile=tile,size_px=prep['tile_sources'][tile]['tile_size_px'],
        geotransform=prep['tile_sources'][tile]['geotransform'],expected_native_bytes=prep['tile_sources'][tile]['bytes'],
        download=source_catalog[tile]) for tile in sorted(val_tiles)}
    for tile,s in sources.items():
        row=s['download'];assert row['zip']==f'MurrayLab_GlobalCTXMosaic_V01_{tile}.zip'
        assert row['url']=='https://murray-lab.caltech.edu/CTX/V01/tiles/'+row['zip']
    cases=[];unique={};oracle={};eval_items=[];models={}
    def add_case(tile,box):
        key=(tile,tuple(float(v).hex() for v in box))
        if key in unique:return unique[key]
        source=sources[tile];win,geometry=planned_window(box,source);extent=world_box(source['geotransform'],win)
        overlap=(np.minimum(train_world[:,2],extent[2])-np.maximum(train_world[:,0],extent[0])>1e-5)&(np.minimum(train_world[:,3],extent[3])-np.maximum(train_world[:,1],extent[1])>1e-5)
        assert not overlap.any(),'Native classification normalization window overlaps training pixels'
        idx=len(cases);unique[key]=idx;side=win[2]-win[0]
        projected=[(box[0]-win[0])*224/side,(box[1]-win[1])*224/side,(box[2]-win[0])*224/side,(box[3]-win[1])*224/side]
        assert 0<=projected[0]<projected[2]<=224 and 0<=projected[1]<projected[3]<=224
        cases.append(dict(index=idx,tile=tile,box_native=list(box),window_px=win,target_box224=projected,
            geometry=geometry,world_window=extent,crop=str(OUT/'inputs'/f'roi_{idx:05d}.png')))
        return idx
    for item in items:
        sample=samples[item['stem']];assert item['window_px']==sample['windows']['offset']
        assert item['source_ids']==[l['source_id'] for l in sample['labels']]
        labels=[]
        for sid,label,fbox in zip(item['source_ids'],sample['labels'],item['human']['boxes']):
            assert int(sid) not in train_ids
            box=label['manual_box'];back=native_box(fbox,item);assert np.allclose(box,back,rtol=0,atol=1e-8)
            types={('FL' if catalog[int(cid)]['type']=='DF' else catalog[int(cid)]['type']) for cid in label['catalog_ids']}
            assert len(types)==1 and next(iter(types)) in CLASS_NAMES,'Ambiguous alias subtype needs review'
            typ=next(iter(types));entry=dict(source_id=int(sid),catalog_class=typ,class_id=CLASS_NAMES.index(typ),case_index=add_case(item['tile'],box))
            if sid in oracle:assert oracle[sid]==entry
            oracle[sid]=entry;labels.append(dict(entry,box=fbox))
        assert len(labels)==len(item['human']['boxes'])
        assert C.sha256(ROOT/item['image'])==item['image_sha256']
        eval_items.append(dict(stem=item['stem'],tile=item['tile'],window_px=item['window_px'],width=item['width'],height=item['height'],
            source_ids=item['source_ids'],labels=labels,native_crop_sha256=item['native_crop_sha256']))
    assert len(items)==173 and len(oracle)==183 and sum(len(r['labels']) for r in eval_items)==560
    for name,checkpoint in detector_spec['checkpoints'].items():
        cache=DETECTOR/name/'predictions.json';assert C.sha256(cache)==old_results['results'][name]['predictions_sha256']
        assert C.sha256(checkpoint['path'])==checkpoint['sha256']
        prediction=C.read_json(cache);threshold=old_results['results'][name]['budgets']['1071']['threshold'];selected={};tp=0;nc=0
        for item in items:
            boxes,confidence,classes=[np.asarray(v,np.float32) for v in prediction[item['stem']]];boxes=boxes.reshape(-1,4)
            assert np.isfinite(boxes).all() and len(boxes)==len(confidence)==len(classes) and np.all(classes==0)
            indices=np.flatnonzero(confidence>=threshold);gt=[(0,*b) for b in item['human']['boxes']]
            matches=match_image(gt,boxes,confidence,classes,.5,threshold,class_aware=False)
            selected[item['stem']]=dict(predictions=[],matches=[dict(gt_index=i,matched=bool(r[1]),kept_prediction_index=int(r[4]),IoU=float(r[3])) for i,r in enumerate(matches)])
            for original_index in indices:
                b=boxes[original_index].astype(np.float64);clipped=np.clip(b,[0,0,0,0],[item['width'],item['height'],item['width'],item['height']])
                assert clipped[0]<clipped[2] and clipped[1]<clipped[3]
                selected[item['stem']]['predictions'].append(dict(original_index=int(original_index),box=b.tolist(),score=float(confidence[original_index]),
                    case_index=add_case(item['tile'],native_box(clipped,item)),image_boundary_clip=bool(np.any(b!=clipped))))
            tp+=sum(int(r[1]) for r in matches);nc+=len(indices)
        old=old_results['results'][name]['budgets']['1071']['metrics'];assert nc==old['candidates']==1071 and tp==old['tp']
        models[name]=dict(weights=checkpoint,prediction_cache=str(cache),threshold=threshold,expected_detection_TP=tp,candidates=nc,items=selected)
    write(OUT/'input_plan.json',dict(status='metadata_frozen_no_scores',created_utc=utc(),cases=cases,items=eval_items,
        oracle=list(oracle.values()),models=models,sources=sources,subtype_labels_independently_reviewed=False))
    files=[FROZEN,prep_path,CATALOG,MANIFEST,TRAIN_NATIVE,P.OUT/'protocol.json',P.OUT/'pooled_features.npz',P.OUT/'feature_receipt.json',DETECTOR/'protocol.json',DETECTOR/'comparison.json']
    files += [Path(r['prediction_cache']) for r in models.values()]
    code=[Path(__file__),Path(P.__file__),Path(C.__file__),Path(R.__file__),Path(N.__file__),ROOT/'data/scripts/build_native_classification_crop_pilot.py',
          ROOT/'data/scripts/fetch_v2_train_tiles.py',ROOT/'data/scripts/eval_recall_report.py',ROOT/'data/scripts/start_detector_classifier_bridge_v2.ps1']
    checkpoint=C.read_json(P.OUT/'protocol.json')['checkpoint']
    spec=dict(status='locked_before_native_restore_or_classification',created_utc=utc(),scope='development_validation_only_not_blind_final_test',
        images=173,GT_appearances=560,unique_GT=183,native_crops=len(cases),models=list(models),source_cache=str(CACHE.resolve()),
        restoration_tiles=sorted(sources),download_bytes=sum(s['download']['size_bytes'] for s in sources.values()),
        files_sha256={str(p):C.sha256(p) for p in files},code_hashes={str(p):C.sha256(p) for p in code},
        input_plan_sha256=C.sha256(OUT/'input_plan.json'),package_versions=R.package_versions(),
        checkpoint=checkpoint,checkpoint_files_sha256=C.read_json(P.OUT/'feature_receipt.json')['checkpoint_files_sha256'],
        classifier=dict(head='logistic_regression',balance_mode='balanced',C=1.0,max_iter=10000,random_state=20261014,features=['full_cls','roi_only'],train_unique_IDs=1083),
        extraction=dict(device='cuda',precision='float32',batch_size=8,threads=4,AMP=False,TF32=False),
        match_rule='Original greedy GT-order, score-order class-agnostic IoU>=.5. Classification scored on those fixed pairs without rematching.',
        classification_input='Native square1.5x longest candidate side with25percent per-axis context, training-matched crop-local nonzero2/98 uint8 LUT, direct BILINEAR224; full candidate preserved at boundaries.',
        limitations=['All183 validation catalog subtypes unreviewed; only2 unique FL. Development data already viewed, not blind acceptance.',
            'No pure-background validation images; unmatched predictions may include omissions and are not confirmed geological false positives.',
            'Repeated target appearances are not independent slides. Counts and unique oracle classification reported separately.',
            'No background filter or classifier rejection threshold. Every retained detector candidate receives one subtype.',
            'Detector budgets/thresholds copied from previous validation comparison; no selection using new subtype scores.'])
    write(OUT/'protocol.json',spec);(OUT/'protocol.sha256').write_text(C.sha256(OUT/'protocol.json')+'\n',encoding='utf-8')
    write(OUT/'queue.json',dict(status='prepared_not_started',created_utc=utc()))
    print('PREPARED_NATIVE_DETECTOR_CLASSIFIER_BRIDGE',len(cases),'crops; downloadGiB',spec['download_bytes']/1024**3,flush=True)


def check():
    P.check();spec=C.read_json(OUT/'protocol.json')
    assert C.sha256(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for field in ['files_sha256','code_hashes']:
        for path,digest in spec[field].items():assert C.sha256(path)==digest,path
    assert C.sha256(OUT/'input_plan.json')==spec['input_plan_sha256'] and R.package_versions()==spec['package_versions']
    return spec


def restore(plan,queue):
    CACHE.mkdir(parents=True,exist_ok=True);verified={}
    for tile,source in plan['sources'].items():
        queue.update(phase='restoring_native_validation_tiles',current_tile=tile,restored_tiles=list(verified));write(OUT/'queue.json',queue)
        found=existing_tif(CACHE,tile)
        if found is None:
            assert shutil.disk_usage(CACHE).free>source['download']['size_bytes']*3+3*1024**3
            zipped=download_zip(source['download'],CACHE,4);found=extract_verified(source['download'],zipped,CACHE)
        path=Path(found['path']);gt,w,h=read_geotransform(path)
        assert path.stat().st_size==source['expected_native_bytes'] and [w,h]==source['size_px']
        assert np.allclose(gt,source['geotransform'],atol=1e-7,rtol=0)
        verified[tile]=dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,header_verified=True)
        write(OUT/'native_sources.json',verified)
    return verified


def build_crops(plan,sources,queue):
    (OUT/'inputs').mkdir(exist_ok=True);(OUT/'receipts').mkdir(exist_ok=True)
    scratch=(CACHE/'_classifier_scratch').resolve();scratch.mkdir(exist_ok=True)
    for tile,source in sources.items():
        raw=tifffile.memmap(source['path']);assert raw.dtype==np.uint8
        try:
            # Prove exact restoration against every frozen native detection window.
            for item in plan['items']:
                if item['tile']!=tile:continue
                x0,y0,x1,y1=item['window_px'];digest=hashlib.sha256()
                for y in range(y0,y1,128):digest.update(np.ascontiguousarray(raw[y:min(y+128,y1),x0:x1]).tobytes())
                assert digest.hexdigest()==item['native_crop_sha256'],'Restored native pixels differ from frozen source'
            for case in plan['cases']:
                if case['tile']!=tile:continue
                dest=Path(case['crop']);receipt_path=OUT/'receipts'/f"roi_{case['index']:05d}.json"
                if receipt_path.exists():
                    assert C.sha256(dest)==C.read_json(receipt_path)['crop_sha256'];continue
                assert not dest.exists(),'Unreceipted crop preserved'
                x0,y0,x1,y1=case['window_px'];w,h=x1-x0,y1-y0
                assert shutil.disk_usage(scratch).free>w*h+2*1024**3
                counts=np.zeros(256,dtype=np.int64)
                for y in range(y0,y1,128):counts+=np.bincount(np.ascontiguousarray(raw[y:min(y+128,y1),x0:x1]).ravel(),minlength=256)
                lut,contrast=N.histogram_lut(counts);temp=(scratch/('stretch_'+uuid.uuid4().hex+'.bin')).resolve()
                assert temp.parent==scratch
                stretched=None;image=None
                try:
                    with temp.open('xb'):pass
                    stretched=np.memmap(temp,mode='r+',dtype=np.uint8,shape=(h,w))
                    for y in range(0,h,128):stretched[y:y+128]=lut[raw[y0+y:min(y0+y+128,y1),x0:x1]]
                    stretched.flush();image=Image.fromarray(stretched);image.resize((224,224),Image.Resampling.BILINEAR).save(dest)
                    write(receipt_path,dict(case_index=case['index'],crop_sha256=C.sha256(dest),contrast=contrast,
                        source=source,box224=case['target_box224'],window_px=case['window_px'],context_exception=case['geometry']['context_exception']))
                finally:
                    if image is not None:image.close()
                    if stretched is not None:stretched._mmap.close()
                    temp.unlink(missing_ok=True)
                queue.update(phase='building_native_classifier_crops',current_tile=tile,completed_crops=len(list((OUT/'receipts').glob('roi_*.json'))));write(OUT/'queue.json',queue)
        finally:raw._mmap.close()
    assert len(list((OUT/'receipts').glob('roi_*.json')))==len(plan['cases'])


def features(plan,spec,queue):
    import torch
    from transformers import AutoModel
    assert torch.cuda.is_available();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModel.from_pretrained(spec['checkpoint'],local_files_only=True,use_safetensors=True).to('cuda').float().eval().requires_grad_(False)
    cls=[];roi=[]
    with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
        for start in range(0,len(plan['cases']),8):
            batch=plan['cases'][start:start+8];images=[]
            try:
                for case in batch:
                    assert C.sha256(case['crop'])==C.read_json(OUT/'receipts'/f"roi_{case['index']:05d}.json")['crop_sha256']
                    with Image.open(case['crop']) as im:assert im.size==(224,224);images.append(im.convert('RGB'))
                tokens=model(pixel_values=C.prepare_pixels(images).to('cuda',dtype=torch.float32)).last_hidden_state
            finally:
                for image in images:image.close()
            assert tokens.shape==(len(batch),257,768)
            weights=np.array([P.patch_areas(case['target_box224']) for case in batch]);weights/=weights.sum(1,keepdims=True)
            weights=torch.as_tensor(weights,device='cuda',dtype=torch.float32)
            cls.append(tokens[:,0].cpu().numpy());roi.append(torch.einsum('bp,bpd->bd',weights,tokens[:,1:]).cpu().numpy())
            queue.update(phase='extracting_native_classifier_features',completed_feature_rows=start+len(batch));write(OUT/'queue.json',queue)
    result=dict(full_cls=np.concatenate(cls),roi_only=np.concatenate(roi));assert all(np.isfinite(x).all() for x in result.values())
    np.savez_compressed(OUT/'features.npz',**result);return result


def score(plan,X,spec):
    with np.load(P.OUT/'pooled_features.npz',allow_pickle=False) as f:training={name:f[name] for name in ['full_cls','roi_only']};y=f['y']
    predicted={};fit={}
    for arm in ['full_cls','roi_only']:
        scaler=StandardScaler().fit(training[arm]);cw=classifier_class_weights(y,'balanced')
        head=LogisticRegression(C=1.0,class_weight=cw,max_iter=10000,random_state=20261014)
        with warnings.catch_warnings():warnings.simplefilter('error',ConvergenceWarning);head.fit(scaler.transform(training[arm]),y)
        predicted[arm]=head.predict(scaler.transform(X[arm]));fit[arm]=dict(train_unique_IDs=1083,class_weights=cw,fit_only_training_features=True)
    np.savez_compressed(OUT/'class_predictions.npz',**predicted)
    oracle={};results={};traces=[];item_map={r['stem']:r for r in plan['items']}
    for arm,classes in predicted.items():
        gy=[r['class_id'] for r in plan['oracle']];gp=[int(classes[r['case_index']]) for r in plan['oracle']]
        oracle[arm]=dict(unique_targets=len(gy),macro_f1=float(f1_score(gy,gp,labels=range(4),average='macro',zero_division=0)),
            per_class=classification_report(gy,gp,labels=range(4),target_names=CLASS_NAMES,output_dict=True,zero_division=0),
            confusion_matrix=confusion_matrix(gy,gp,labels=range(4)).tolist())
        results[arm]={}
        for name,model in plan['models'].items():
            gt_count=np.zeros(4,dtype=int);pred_count=np.zeros(4,dtype=int);correct=np.zeros(4,dtype=int);matched_y=[];matched_p=[];located=0
            for stem,entry in model['items'].items():
                labels=item_map[stem]['labels'];pc=[int(classes[r['case_index']]) for r in entry['predictions']]
                for c in pc:pred_count[c]+=1
                for match,label in zip(entry['matches'],labels):
                    gt_count[label['class_id']]+=1;typ=None
                    if match['matched']:
                        located+=1;typ=pc[match['kept_prediction_index']];matched_y.append(label['class_id']);matched_p.append(typ)
                        if typ==label['class_id']:correct[typ]+=1
                    traces.append(dict(classifier=arm,detector=name,stem=stem,source_id=label['source_id'],truth=label['catalog_class'],
                        located=match['matched'],IoU=match['IoU'],prediction=CLASS_NAMES[typ] if typ is not None else None,
                        jointly_correct=bool(typ==label['class_id']) if typ is not None else False))
            assert located==model['expected_detection_TP'] and int(gt_count.sum())==560 and int(pred_count.sum())==1071
            precision=np.divide(correct,pred_count,out=np.zeros(4),where=pred_count>0);recall=correct/gt_count
            f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(4),where=precision+recall>0)
            results[arm][name]=dict(candidates=1071,GT_appearances=560,localized=located,localization_recall=located/560,
                reference_unmatched_predictions=1071-located,matched_classification_accuracy=sum(a==b for a,b in zip(matched_y,matched_p))/located,
                matched_classification_report=classification_report(matched_y,matched_p,labels=range(4),target_names=CLASS_NAMES,output_dict=True,zero_division=0),
                jointly_correct=int(correct.sum()),joint_precision=float(correct.sum()/1071),joint_recall=float(correct.sum()/560),joint_macro_f1=float(f1.mean()),
                per_class={c:dict(GT=int(gt_count[i]),predictions=int(pred_count[i]),joint_TP=int(correct[i]),precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i])) for i,c in enumerate(CLASS_NAMES)})
    write(OUT/'per_target_trace.json',traces)
    report=dict(status='completed_development_native_detector_classifier_bridge',created_utc=utc(),validation_images=173,
        oracle_unique_targets=183,GT_appearances=560,oracle_classification=oracle,pipeline=results,fit_receipts=fit,
        protocol_sha256=C.sha256(OUT/'protocol.json'),features_sha256=C.sha256(OUT/'features.npz'),
        limitations=spec['limitations'],classification_threshold='argmax only; no rejection',test_images_read=False,labels_modified=False)
    write(OUT/'comparison.json',report)
    lines=['# 真实检测框→原生裁片→分类：开发验证','','固定173张偏移验证图、560次人工框、183个独立源编号；沿用原1071候选预算的既定阈值。','',
        '|分类特征|检测器|定位命中|命中框类型正确率|定位且类型正确|联合召回|参考联合精确率|','|---|---|---:|---:|---:|---:|---:|']
    for arm,models in results.items():
        for name,r in models.items():lines.append(f"|{arm}|{name}|{r['localized']}/560|{r['matched_classification_accuracy']:.1%}|{r['jointly_correct']}|{r['joint_recall']:.1%}|{r['joint_precision']:.1%}|")
    lines+=['','人工提供完整区域的独立目标分类参考：','', '|特征|183个独立目标macro-F1|正确率|','|---|---:|---:|']
    for arm,r in oracle.items():lines.append(f"|{arm}|{r['macro_f1']:.1%}|{r['per_class']['accuracy']:.1%}|")
    lines+=['','目录类型未经独立复核，独立FL仅2个。验证区域已用于开发且没有纯背景图，不能当新的盲测或整幅CTX误报验收。',
        '定位漏检保留在560分母，全部1071预测保留在精确率分母；定位固定配对后核对类型，不通过改类重匹配追分。']
    (OUT/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def run():
    spec=check();assert C.read_json(OUT/'queue.json')['status']=='prepared_not_started'
    independence=R.F.process_independence();assert independence['scheduler_owned'];R.ensure_gpu_idle()
    kernel=ctypes.windll.kernel32
    if not kernel.SetThreadExecutionState(0x80000001):raise ctypes.WinError()
    plan=C.read_json(OUT/'input_plan.json');queue=dict(status='running',started_utc=utc(),pid=os.getpid(),phase='native_restore',
        process_independence=independence,temporary_system_awake=True,network_phase='bounded_catalog_source_restoration_only')
    write(OUT/'queue.json',queue);attempts=[]
    try:
        sources=restore(plan,queue);attempts=R.F.prevent_network();queue.update(network_phase='offline_crop_inference_scoring');write(OUT/'queue.json',queue)
        build_crops(plan,sources,queue);X=features(plan,spec,queue);check();report=score(plan,X,spec)
        queue.update(status=report['status'],phase='complete')
    except BaseException as error:
        queue.update(status='failed',error=repr(error));raise
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        queue.update(finished_utc=utc(),temporary_system_awake=False,blocked_network_attempts_after_restore=attempts);write(OUT/'queue.json',queue)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','scheduled-run']);args=parser.parse_args()
    if args.mode=='prepare':prepare()
    else:
        sys.stdout=(OUT/'training.log').open('x',encoding='utf-8',buffering=1);sys.stderr=(OUT/'training.err.log').open('x',encoding='utf-8',buffering=1);run()
