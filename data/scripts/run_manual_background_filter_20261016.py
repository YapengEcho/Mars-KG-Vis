"""Finite offline appearance-filter trial on confirmed native train backgrounds."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.venv/Lib/site-packages'),str(ROOT/'data/scripts')]
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_OFFLINE='true')
import numpy as np
import tifffile
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
import extract_manual_classification_features as C
import export_manual_detection_dataset as E

OUT=ROOT/'runs/experiments/manual_background_filter_20261016_v1'
DRAFT=ROOT/'runs/research/manual_background_filter_20261016_v1/input_manifest.draft.json'
EXPORT=ROOT/'data/processed/manual_yolo_20261009_v8/export_plan.json'
POSITIVE=ROOT/'runs/experiments/native_crop_comparison_20261015_v2/native/features'
COMBINED=ROOT/'runs/experiments/combined_detection_classification_20261015_v1'
SEEDS=[20261015,20261016,20261017]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)
def now():return datetime.now(timezone.utc).isoformat()


def prepare():
    if OUT.exists():raise FileExistsError('Preserve existing protocol; use run for recovery')
    draft=read(DRAFT);positive=read(POSITIVE/'receipt.json');evaluation=read(COMBINED/'input_plan.json')
    assert draft['summary']['eligible_backgrounds']==1355 and draft['summary']['held_backgrounds']==0
    groups=draft['groups'];negative_counts=[0,0,0];assignment={}
    for group in groups:
        folds=group['historical_positive_folds']
        if folds:
            assert len(folds)==1
            assignment[group['group']]=folds[0];negative_counts[folds[0]]+=group['background']
    for group in groups:
        if group['group'] not in assignment:
            fold=min(range(3),key=lambda f:(negative_counts[f],f))
            assignment[group['group']]=fold;negative_counts[fold]+=group['background']
    rows=[dict(row,fold_id=assignment[row['spatial_group_id']]) for row in draft['rows']]
    pins={str(p):sha(p) for p in [DRAFT,EXPORT,POSITIVE/'features.npz',POSITIVE/'receipt.json',
        COMBINED/'input_plan.json',Path(__file__),Path(C.__file__),Path(E.__file__)]}
    for name,expected in positive['checkpoint_files_sha256'].items():
        p=Path(positive['checkpoint'])/name;assert sha(p)==expected;pins[str(p)]=expected
    for row in rows:
        if row['binary_label']==1:
            assert sha(row['crop'])==row['crop_sha256']
    for row in evaluation['cases']:pins[row['crop224']]=sha(row['crop224'])
    for seed in SEEDS:
        path=COMBINED/f'seed{seed}/original_joint224.npz';pins[str(path)]=sha(path)
    assert sha(POSITIVE/'features.npz')==positive['feature_file_sha256']
    spec=dict(status='locked_before_filter_features_or_fits',created_utc=now(),rows=rows,
        folds=3,positive_rows=1083,background_rows=1355,spatial_groups=41,
        fold_counts=[dict(fold=i,positive=sum(r['binary_label']==1 and r['fold_id']==i for r in rows),
            background=sum(r['binary_label']==0 and r['fold_id']==i for r in rows)) for i in range(3)],
        checkpoint=positive['checkpoint'],encoder='frozen pretrained DINOv2-base CLS, no LoRA',
        feature_rule='Reuse exact native224 positive vectors after smoke verification; encode negative and development cached native224 crops with same FP32 endpoint and normalization',
        head=dict(kind='StandardScaler plus balanced LogisticRegression',C=1.,max_iter=2000,seed=20261016),
        threshold_rule=dict(positive_OOF_retention_min=.97,grid=[0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99],
            selection='Largest grid threshold preserving >=97% training-region positive OOF retention; no development scores for threshold selection'),
        evaluation=dict(source=str(COMBINED/'input_plan.json'),images=173,GT_appearances=560,unique_targets=183,
            pools=['global_only','global_local'],candidates_each=1071,
            classification='Read-only original joint224 predictions, three existing optimization seeds',
            matching='Freeze existing localization assignments; removed matched predictions are lost, no score reranking or rematching',
            interpretation='Fixed-assignment retention ablation, not a rematched detector/AP evaluation or independent final test'),
        native_crop_rule='Inscribed square inside human-confirmed background; uint8 nonzero2/98 stretch; LANCZOS224; same encoder RGB normalization',
        source_sha256=pins,final_test_images_read=False,automatic_next_experiment=False)
    OUT.mkdir(parents=True);(OUT/'backgrounds').mkdir()
    write(OUT/'protocol.json',spec);(OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')
    write(OUT/'queue.json',dict(status='prepared_not_started',created_utc=now()))
    print(json.dumps(dict(status=spec['status'],fold_counts=spec['fold_counts']),ensure_ascii=True))


def read_region(path,box):
    """Read exact integer uint8 pixels; decode only intersecting TIFF segments."""
    x0,y0,x1,y1=map(int,box)
    with tifffile.TiffFile(str(path)) as tif:
        page=tif.pages[0];height,width=page.shape
        assert page.dtype==np.uint8 and 0<=x0<x1<=width and 0<=y0<y1<=height
        if page.is_memmappable:
            pixels=tifffile.memmap(str(path));result=np.array(pixels[y0:y1,x0:x1],copy=True);pixels._mmap.close()
            return result
        result=np.empty((y1-y0,x1-x0),dtype=np.uint8);copied=0
        if page.is_tiled:
            th,tw=page.tilelength,page.tilewidth;across=(width+tw-1)//tw
            indices=[ty*across+tx for ty in range(y0//th,(y1-1)//th+1) for tx in range(x0//tw,(x1-1)//tw+1)]
        else:
            th=page.rowsperstrip;indices=range(y0//th,(y1-1)//th+1)
        for index in indices:
            tif.filehandle.seek(page.dataoffsets[index]);raw=tif.filehandle.read(page.databytecounts[index])
            decoded,position,_=page.decode(raw,index)
            assert decoded is not None and decoded.ndim==4 and decoded.shape[0]==1 and decoded.shape[3]==1
            sy,sx=position[2:4];block=decoded[0,:,:,0]
            left,top=max(x0,sx),max(y0,sy);right,bottom=min(x1,sx+block.shape[1]),min(y1,sy+block.shape[0])
            if left<right and top<bottom:
                result[top-y0:bottom-y0,left-x0:right-x0]=block[top-sy:bottom-sy,left-sx:right-sx]
                copied+=(bottom-top)*(right-left)
        assert copied==result.size
        return result


def export_backgrounds(spec,queue):
    originals={s['background_id']:s for s in read(EXPORT)['samples'] if s['kind']=='negative'}
    receipts=[]
    for index,row in enumerate(r for r in spec['rows'] if r['binary_label']==0):
        path=OUT/'backgrounds'/f"{row['row_id']}.png";receipt_path=path.with_suffix('.json')
        if receipt_path.is_file():
            receipt=read(receipt_path);assert receipt['protocol_sha256']==sha(OUT/'protocol.json') and receipt['image_sha256']==sha(path)
            receipts.append(receipt);continue
        source=row['source'];stat=Path(source['path']).stat()
        assert (stat.st_size,stat.st_mtime_ns)==(source['bytes'],source['mtime_ns'])
        ox,oy=source['origin_px'];original=row['confirmed_background_box_px']
        raw=read_region(source['path'],[original[0]-ox,original[1]-oy,original[2]-ox,original[3]-oy])
        assert hashlib.sha256(raw.tobytes()).hexdigest()==originals[row['row_id']]['native_crop_sha256'],row['row_id']
        x0,y0,x1,y1=row['window_px'];region=raw[y0-original[1]:y1-original[1],x0-original[0]:x1-original[0]]
        image=Image.fromarray(E.stretch_uint8(region)).resize((224,224),Image.Resampling.LANCZOS)
        image.save(path)
        after=Path(source['path']).stat();assert (after.st_size,after.st_mtime_ns)==(stat.st_size,stat.st_mtime_ns)
        receipt=dict(row_id=row['row_id'],image_sha256=sha(path),source_native_rectangle_hash_verified=True,
            confirmed_rectangle=original,inscribed_window=row['window_px'],source=source['path'],
            protocol_sha256=sha(OUT/'protocol.json'))
        write(receipt_path,receipt);receipts.append(receipt)
        if index%25==0:
            queue.update(phase='verify_and_export_native_backgrounds',completed=index+1,total=1355);write(OUT/'queue.json',queue)
            print('BACKGROUND',index+1,flush=True)
    assert len(receipts)==1355
    write(OUT/'background_export_verification.json',dict(count=len(receipts),all_original_native_hashes_verified=True,
        all_ROIs_inside_human_confirmation=True,receipts=receipts))


def encode(endpoint,paths,queue,phase,torch):
    result=[]
    for start in range(0,len(paths),16):
        images=[]
        for path in paths[start:start+16]:
            with Image.open(path) as im:images.append(im.copy())
        with torch.inference_mode():values=endpoint(C.prepare_pixels(images).to('cuda')).float().cpu().numpy()
        assert values.shape==(len(images),768) and np.isfinite(values).all()
        result.append(values)
        queue.update(phase=phase,completed=min(start+16,len(paths)),total=len(paths));write(OUT/'queue.json',queue)
    return np.concatenate(result)


def fitted(x,y):
    model=Pipeline([('scale',StandardScaler()),('lr',LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,random_state=20261016))])
    model.fit(x,y);return model


def evaluate_pool(plan,pool,keep,classes):
    candidates=tp=0;localized=set();joint=[0]*3;unique=[set() for _ in SEEDS]
    per_class={str(i):dict(gt=0,located=0,joint=[0]*3) for i in range(4)}
    for item in plan['items']:
        value=plan['models'][pool]['items'][item['stem']]
        candidates+=sum(keep[p['case_index']] for p in value['predictions'])
        for match in value['matches']:
            label=item['labels'][match['gt_index']];per_class[str(label['class_id'])]['gt']+=1
            if not match['matched']:continue
            prediction=value['predictions'][match['kept_prediction_index']];case=prediction['case_index']
            if not keep[case]:continue
            tp+=1;localized.add(label['source_id']);per_class[str(label['class_id'])]['located']+=1
            for si,pred in enumerate(classes):
                if int(pred[case])==label['class_id']:
                    joint[si]+=1;unique[si].add(label['source_id']);per_class[str(label['class_id'])]['joint'][si]+=1
    return dict(candidates=int(candidates),localized_appearances=tp,recall=tp/560,
        reference_precision=tp/candidates if candidates else 0.,unique_localized=len(localized),
        unmatched_reference_predictions=int(candidates-tp),joint_appearances=joint,
        joint_recall=[v/560 for v in joint],joint_reference_precision=[v/candidates if candidates else 0. for v in joint],
        unique_joint=[len(s) for s in unique],per_class=per_class)


def run():
    import run_full_pretrained_detector as F
    attempts=F.prevent_network();import torch;import joblib
    torch.set_num_threads(4)
    spec=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for name,expected in spec['source_sha256'].items():assert sha(name)==expected,name
    queue=dict(status='running',pid=os.getpid(),started_utc=now(),scope='finite_confirmed_background_filter',
        automatic_following_experiments=False,offline=True,stage='native_export')
    write(OUT/'queue.json',queue);ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        export_backgrounds(spec,queue)
        rows=spec['rows'];positive_rows=[r for r in rows if r['binary_label']==1]
        negatives=[r for r in rows if r['binary_label']==0]
        with np.load(POSITIVE/'features.npz',allow_pickle=False) as cache:
            assert cache['source_id'].tolist()==[r['source_id'] for r in positive_rows]
            positive_x=cache['X'].copy()
        endpoint,dimension,receipt=C.load_frozen_encoder('dinov2',spec['checkpoint'],'cuda');assert dimension==768
        smoke=encode(endpoint,[r['crop'] for r in positive_rows[:8]],queue,'positive_cache_smoke',torch)
        difference=float(np.max(np.abs(smoke-positive_x[:8])))
        assert difference<1e-4,('Positive encoder cache differs',difference)
        write(OUT/'encoder_smoke.json',dict(positive_cache_max_abs_error=difference,encoder=receipt))
        negative_x=encode(endpoint,[OUT/'backgrounds'/f"{r['row_id']}.png" for r in negatives],queue,'encode_backgrounds',torch)
        x=np.concatenate([positive_x,negative_x]);y=np.array([r['binary_label'] for r in rows]);folds=np.array([r['fold_id'] for r in rows])
        assert x.shape==(2438,768)
        oof=np.full(len(rows),np.nan)
        for fold in range(3):
            fit=np.flatnonzero(folds!=fold);held=np.flatnonzero(folds==fold)
            assert not ({rows[i]['spatial_group_id'] for i in fit}&{rows[i]['spatial_group_id'] for i in held})
            model=fitted(x[fit],y[fit]);oof[held]=model.predict_proba(x[held])[:,1]
            joblib.dump(model,OUT/f'fold{fold}.joblib')
        assert np.isfinite(oof).all()
        sweep=[dict(threshold=t,positive_retention=float(np.mean(oof[y==1]>=t)),
            background_rejection=float(np.mean(oof[y==0]<t))) for t in spec['threshold_rule']['grid']]
        threshold=max(r['threshold'] for r in sweep if r['positive_retention']>=.97)
        selection=dict(threshold=threshold,selection_source='training-region three-fold OOF only',sweep=sweep,
            roc_auc=float(roc_auc_score(y,oof)),balanced_accuracy_at_half=float(balanced_accuracy_score(y,oof>=.5)),
            threshold_locked_before_development_feature_encoding=True)
        write(OUT/'threshold_selection.json',selection)
        np.savez_compressed(OUT/'training_oof.npz',features=x,labels=y,folds=folds,scores=oof,
            row_id=np.array([r['row_id'] for r in rows]))
        model=fitted(x,y);joblib.dump(model,OUT/'full_filter.joblib')
        plan=read(COMBINED/'input_plan.json')
        candidate_x=encode(endpoint,[r['crop224'] for r in plan['cases']],queue,'encode_fixed_development_ROIs',torch)
        score=model.predict_proba(candidate_x)[:,1];np.savez_compressed(OUT/'development_scores.npz',features=candidate_x,scores=score)
        classes=[]
        for seed in SEEDS:
            with np.load(COMBINED/f'seed{seed}/original_joint224.npz',allow_pickle=False) as cache:classes.append(cache['predictions'].copy())
        results={}
        for pool in spec['evaluation']['pools']:
            before=evaluate_pool(plan,pool,np.ones(len(score),dtype=bool),classes)
            after=evaluate_pool(plan,pool,score>=threshold,classes)
            assert before['candidates']==1071
            results[pool]=dict(off=before,on=after,removed_candidates=before['candidates']-after['candidates'],
                lost_localized_appearances=before['localized_appearances']-after['localized_appearances'],
                lost_unique_localized=before['unique_localized']-after['unique_localized'])
        result=dict(status='completed_fixed_assignment_development_ablation',created_utc=now(),
            threshold=threshold,training_region_selection=selection,results=results,classification_seeds=SEEDS,
            detector_or_classifier_retrained=False,background_filter_fitted=True,model_promoted=False,
            limitations=['Generic pretrained image features; confirmed-window negatives may differ from detector mistakes.',
                'Existing localization assignments retained; no rematching or AP recomputation.',
                'Development regions have been reused; unmatched predictions may include catalogue omissions.',
                'Three seeds are fixed type-classifier predictions, not three independent filter trainings.'],
            protocol_sha256=sha(OUT/'protocol.json'))
        write(OUT/'comparison.json',result)
        lines=['# 已确认背景的过滤开关对照','',
            '固定旧有完整预训练检测候选与原joint224四分类；只新拟合冻结DINOv2特征上的二分类LR。',
            '训练为1083个正目标＋1355个确认背景、41空间组；阈值仅由训练区域三折OOF选择。','',
            '|候选池|过滤|保留候选|定位召回|参考精确率|独立定位／183|联合召回均值|独立正确均数／183|',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for pool,pair in results.items():
            for mode in ['off','on']:
                r=pair[mode];lines.append(f"|{pool}|{mode}|{r['candidates']}|{r['recall']:.2%}|{r['reference_precision']:.2%}|{r['unique_localized']}|{np.mean(r['joint_recall']):.2%}|{np.mean(r['unique_joint']):.2f}|")
        lines+=['',f'阈值{threshold}；固定原定位配对，过滤后被移除的命中按漏检计，不重排或重新匹配。',
                '当前为反复使用开发区的诊断结果，不是独立验收；未自动替换模型。']
        (OUT/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        queue.update(status='completed',phase='finite_trial_finished',finished_utc=now(),threshold=threshold,
            blocked_network_attempts=attempts);write(OUT/'queue.json',queue)
        print(json.dumps(dict(status=queue['status'],threshold=threshold,results=results),ensure_ascii=True),flush=True)
    except BaseException as exc:
        queue.update(status='failed',error=repr(exc),finished_utc=now());write(OUT/'queue.json',queue);raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','scheduled-run']);args=parser.parse_args()
    if args.mode=='prepare':prepare()
    else:
        if args.mode=='scheduled-run':
            sys.stdout=(OUT/'training.log').open('a',encoding='utf-8',buffering=1)
            sys.stderr=(OUT/'training.err.log').open('a',encoding='utf-8',buffering=1)
        run()
