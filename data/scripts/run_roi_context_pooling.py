"""Frozen DINOv2 ROI/context pooling on unchanged full native224 inputs.

No image masking, cropping, subtype edits, or encoder fine-tuning. The duplicate
ROI control accounts for concatenation dimensions and effective regularization.
"""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import warnings

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.venv/Lib/site-packages'))
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_OFFLINE='true')
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,f1_score,classification_report
from sklearn.preprocessing import StandardScaler
import extract_manual_classification_features as C
import run_native_crop_comparison as R
from train_stage2_cls import classifier_class_weights

SOURCE=ROOT/'runs/experiments/context_margin_shape_20261015_v2/context_full'
OUT=ROOT/'runs/experiments/roi_context_pooling_20261015_v1'
ARMS=('full_cls','full_patch_mean','roi_only','roi_plus_ring','roi_plus_cls','roi_duplicate')
PRIMARY='roi_plus_ring'


def utc():return datetime.now(timezone.utc).isoformat()
def write(path,value):C.write_json_atomic(path,value)


def patch_areas(box):
    x0,y0,x1,y1=box
    assert 0<=x0<x1<=224 and 0<=y0<y1<=224
    edge=np.arange(17,dtype=np.float64)*14
    x=np.maximum(0,np.minimum(edge[1:],x1)-np.maximum(edge[:-1],x0))
    y=np.maximum(0,np.minimum(edge[1:],y1)-np.maximum(edge[:-1],y0))
    return (y[:,None]*x[None,:]).reshape(-1)


def weights_for(box):
    x0,y0,x1,y1=box;dx=.1*(x1-x0);dy=.1*(y1-y0)
    expanded=[max(0,x0-dx),max(0,y0-dy),min(224,x1+dx),min(224,y1+dy)]
    areas=patch_areas(box);ring=patch_areas(expanded)-areas
    assert np.all(ring>=-1e-10)
    ring=np.maximum(ring,0)
    assert abs(areas.sum()-(x1-x0)*(y1-y0))<1e-8
    assert abs(ring.sum()-((expanded[2]-expanded[0])*(expanded[3]-expanded[1])-(x1-x0)*(y1-y0)))<1e-8
    assert ring.sum()>0
    return areas/areas.sum(),ring/ring.sum(),expanded


def prepare():
    plan,X,y,fold,receipt=C.load_verified_features(SOURCE/'plan.json',SOURCE/'features')
    assert X.shape==(1083,768) and plan['folds']==3
    assert len(set(r['spatial_group_id'] for r in plan['rows']))==40
    C.new_directory(OUT)
    # Analytic layout checks independent of real target labels or scores.
    known=patch_areas([14,28,42,56]);expected=np.zeros(256);expected[[33,34,49,50]]=196
    assert np.array_equal(known,expected)
    corner=patch_areas([210,210,224,224]);assert corner[255]==196 and np.count_nonzero(corner)==1
    assert patch_areas([0,0,224,224]).sum()==224*224
    roi,ring,regions=[],[],[]
    for row in plan['rows']:
        if C.sha256(row['crop'])!=row['crop_sha256']:raise ValueError('Input image changed')
        with Image.open(row['crop']) as im:assert im.mode=='L' and im.size==(224,224)
        a,b,expanded=weights_for(row['target_box_image_px']);roi.append(a);ring.append(b)
        regions.append(dict(source_id=row['source_id'],box=row['target_box_image_px'],context_box=expanded,
                            roi_patches=int(np.count_nonzero(a)),ring_patches=int(np.count_nonzero(b)),
                            shared_patches=int(np.count_nonzero((a>0)&(b>0))),
                            normalized_weight_overlap=float(np.minimum(a,b).sum())))
    roi,ring=np.array(roi),np.array(ring)
    assert np.allclose(roi.sum(1),1,atol=1e-12) and np.allclose(ring.sum(1),1,atol=1e-12)
    np.savez_compressed(OUT/'pooling_weights.npz',roi=roi,ring=ring,source_id=np.array([r['source_id'] for r in plan['rows']]))
    write(OUT/'pooling_regions.json',dict(targets=1083,patch_order='row-major 16x16, each14x14 pixels',regions=regions))
    write(OUT/'pooling_verification.json',dict(status='passed_analytic_known_cells_and_all1083_area_sums',targets=1083,
        all_weights_finite=bool(np.isfinite(roi).all() and np.isfinite(ring).all()),all_weights_nonnegative=bool((roi>=0).all() and (ring>=0).all()),
        exact_source_hashes_and224_gray_verified=True,images_modified=False,
        note='Partial patches can contribute to both ROI and ring; tokens are contextualized by full-image self-attention.'))
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    canvas=Image.new('RGB',(732,1088),'#181818');draw=ImageDraw.Draw(canvas)
    for j,title in enumerate(['Unchanged input224','ROI pooling weights','10% ring weights']):draw.text((8+j*244,7),title,fill='white',font=font)
    for i,sid in enumerate(['490','491','2156','1348']):
        idx=next(k for k,r in enumerate(plan['rows']) if r['source_id']==sid)
        with Image.open(plan['rows'][idx]['crop']) as im:base=im.convert('RGB')
        canvas.paste(base,(8,35+i*262))
        for j,w in enumerate([roi[idx],ring[idx]],1):
            overlay=base.copy();grid=ImageDraw.Draw(overlay,'RGBA');color=(255,100,40) if j==1 else (20,210,255)
            for k,value in enumerate(w):
                if value>0:
                    x,y=(k%16)*14,(k//16)*14
                    grid.rectangle((x,y,x+13,y+13),fill=(*color,int(150*value/w.max())))
            canvas.paste(overlay,(8+j*244,35+i*262))
        draw.text((8,262+i*262),'ID '+sid,fill='white',font=font)
    canvas.save(OUT/'pooling_examples.png')
    source_files=[SOURCE/'plan.json',SOURCE/'features/features.npz',SOURCE/'features/receipt.json',SOURCE/'cv/cv_report.json']
    code=[Path(__file__),Path(C.__file__),Path(R.__file__),Path(R.F.__file__),ROOT/'data/scripts/train_stage2_cls.py',
          ROOT/'data/scripts/start_roi_context_pooling.ps1',ROOT/'.venv/Lib/site-packages/transformers/models/dinov2/modeling_dinov2.py']
    spec=dict(status='locked_before_encoder_execution',created_utc=utc(),scope=C.POLICY,
        targets=1083,spatial_groups=40,folds=3,source_plan=str(SOURCE/'plan.json'),source_feature_dir=str(SOURCE/'features'),
        source_files_sha256={str(p):C.sha256(p) for p in source_files},source_X_digest=R.matrix_digest(X),
        code_hashes={str(p):C.sha256(p) for p in code},package_versions=R.package_versions(),
        checkpoint=receipt['checkpoint'],checkpoint_files_sha256=receipt['checkpoint_files_sha256'],
        weights_sha256=C.sha256(OUT/'pooling_weights.npz'),regions_sha256=C.sha256(OUT/'pooling_regions.json'),
        arms=list(ARMS),primary=dict(arm=PRIMARY,reference='full_cls',head='logistic_regression',balance_mode='balanced',C=1.0),
        controls=['full_patch_mean separates token-pooling endpoint from spatial selection.',
                  'roi_duplicate repeats identical ROI features to match concatenation dimensions/regularization without new context.'],
        fixed_head=dict(head='logistic_regression',balance_modes=['none','sqrt','balanced'],C=1.0,max_iter=10000,random_state=20261014),
        extraction=dict(device='cuda',precision='float32',batch_size=8,threads=4,autocast=False,TF32=False,patch_size=14,patch_count=256),
        cls_reproduction_tolerance=dict(rtol=1e-5,atol=1e-5,baseline_uses_original_cached_vectors=True),
        limitations=['Internal GT-box classification; catalog types unreviewed, FL15; no independent acceptance.',
            'Patch tokens see the full image through self-attention, not pure object/background segmentation.',
            '14px patch overlap means ROI and ring can share tokens; geometry still affects pooling, not geometry-invariant features.',
            'Concatenation adds dimensions; duplicate-ROI control is diagnostic, not a full capacity equivalence proof.',
            'No input pixel editing, encoder fine-tuning, new targets, or split changes.'])
    write(OUT/'protocol.json',spec);(OUT/'protocol.sha256').write_text(C.sha256(OUT/'protocol.json')+'\n',encoding='utf-8')
    write(OUT/'queue.json',dict(status='prepared_not_started',created_utc=utc()))
    print('PREPARED_ROI_CONTEXT_POOLING',OUT,flush=True)


def check():
    spec=C.read_json(OUT/'protocol.json')
    assert C.sha256(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for field in ['source_files_sha256','code_hashes']:
        for path,digest in spec[field].items():assert C.sha256(path)==digest,path
    assert R.package_versions()==spec['package_versions']
    for name,digest in spec['checkpoint_files_sha256'].items():assert C.sha256(Path(spec['checkpoint'])/name)==digest
    assert C.sha256(OUT/'pooling_weights.npz')==spec['weights_sha256']
    assert C.sha256(OUT/'pooling_regions.json')==spec['regions_sha256']
    C.verify_plan(spec['source_plan'])
    return spec


def extract(spec,queue):
    import torch
    from transformers import AutoModel
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no implicit CPU fallback')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan,baseline,y,fold,old=C.load_verified_features(spec['source_plan'],spec['source_feature_dir'])
    for row in plan['rows']:
        if C.sha256(row['crop'])!=row['crop_sha256']:raise ValueError('Original PNG changed')
    config=C.read_json(Path(spec['checkpoint'])/'config.json')
    assert config['hidden_size']==768 and config['patch_size']==14 and config['model_type']=='dinov2'
    model=AutoModel.from_pretrained(spec['checkpoint'],local_files_only=True,use_safetensors=True).to('cuda').float().eval().requires_grad_(False)
    assert not model.training and all(not p.requires_grad and p.device.type=='cuda' and p.dtype==torch.float32 for p in model.parameters())
    with np.load(OUT/'pooling_weights.npz',allow_pickle=False) as data:
        roi=data['roi'];ring=data['ring'];assert np.array_equal(data['source_id'],[r['source_id'] for r in plan['rows']])
    chunks={name:[] for name in ['cls_recomputed','full_patch_mean','roi_only','ring']}
    with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
        for start in range(0,1083,8):
            rows=plan['rows'][start:start+8];images=[]
            try:
                for row in rows:
                    with Image.open(row['crop']) as im:images.append(im.convert('RGB'))
                tokens=model(pixel_values=C.prepare_pixels(images).to('cuda',dtype=torch.float32)).last_hidden_state
            finally:
                for image in images:image.close()
            assert tuple(tokens.shape)==(len(rows),257,768) and tokens.dtype==torch.float32
            patch=tokens[:,1:]
            wr=torch.as_tensor(roi[start:start+len(rows)],device='cuda',dtype=torch.float32)
            wc=torch.as_tensor(ring[start:start+len(rows)],device='cuda',dtype=torch.float32)
            vectors=dict(cls_recomputed=tokens[:,0],full_patch_mean=patch.mean(dim=1),
                         roi_only=torch.einsum('bp,bpd->bd',wr,patch),ring=torch.einsum('bp,bpd->bd',wc,patch))
            for name,tensor in vectors.items():chunks[name].append(tensor.cpu().numpy().astype(np.float32))
            queue.update(phase='extracting_full_image_patch_features',completed_targets=start+len(rows));write(OUT/'queue.json',queue)
    features={name:np.concatenate(parts) for name,parts in chunks.items()}
    new_cls=features.pop('cls_recomputed');assert np.allclose(new_cls,baseline,rtol=1e-5,atol=1e-5)
    features.update(full_cls=baseline,roi_plus_ring=np.concatenate([features['roi_only'],features['ring']],axis=1),
        roi_plus_cls=np.concatenate([features['roi_only'],baseline],axis=1),roi_duplicate=np.concatenate([features['roi_only'],features['roi_only']],axis=1))
    del features['ring']
    assert set(features)==set(ARMS) and all(np.isfinite(x).all() and x.shape[0]==1083 for x in features.values())
    np.savez_compressed(OUT/'pooled_features.npz',**features,y=y,fold=fold,source_id=np.array([r['source_id'] for r in plan['rows']]))
    write(OUT/'feature_receipt.json',dict(status='completed_frozen_full224_patch_pooling',created_utc=utc(),protocol_sha256=C.sha256(OUT/'protocol.json'),
        features_sha256=C.sha256(OUT/'pooled_features.npz'),shapes={name:list(x.shape) for name,x in features.items()},
        checkpoint_files_sha256=spec['checkpoint_files_sha256'],CUDA_FP32=True,batch_size=8,threads=4,
        cls_recomputation_bitwise_equal=bool(np.array_equal(new_cls,baseline)),cls_max_absolute_difference=float(np.max(np.abs(new_cls-baseline))),
        cls_recomputation_within_locked_tolerance=True,source_baseline_vectors_bitwise_reused=True,images_changed=False))
    return plan,features,y,fold


def fit(plan,features,y,fold,queue,spec):
    previous={(r['head'],r['balance_mode']):r for r in C.read_json(SOURCE/'cv/cv_report.json')['results']}
    for arm in ARMS:
        folder=OUT/arm;folder.mkdir()
        results=[];failed=[];X=features[arm]
        for mode in spec['fixed_head']['balance_modes']:
            queue.update(phase='fitting_LR',current_arm=arm,current_balance=mode);write(OUT/'queue.json',queue)
            try:
                predictions=np.full(len(y),-1,dtype=np.int64);fr=[]
                for f in range(3):
                    train,held=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
                    scaler=StandardScaler().fit(X[train]);weights=classifier_class_weights(y[train],mode)
                    head=LogisticRegression(C=1.0,class_weight=weights,max_iter=10000,random_state=20261014)
                    with warnings.catch_warnings():
                        warnings.simplefilter('error',ConvergenceWarning);head.fit(scaler.transform(X[train]),y[train])
                    pred=head.predict(scaler.transform(X[held]));predictions[held]=pred
                    fr.append(dict(fold=f,train_only_class_weights=weights,held_counts=C.class_counts(y[held]),
                        macro_f1=float(f1_score(y[held],pred,labels=range(4),average='macro',zero_division=0)),
                        confusion_matrix=confusion_matrix(y[held],pred,labels=range(4)).tolist()))
                assert np.all(predictions>=0)
                if arm=='full_cls':assert predictions.tolist()==previous[('logistic_regression',mode)]['source_order_predictions']
                result=dict(head='logistic_regression',balance_mode=mode,out_of_fold_macro_f1=float(f1_score(y,predictions,labels=range(4),average='macro',zero_division=0)),
                    per_class=classification_report(y,predictions,labels=range(4),target_names=C.CLASS_NAMES,output_dict=True,zero_division=0),
                    confusion_matrix=confusion_matrix(y,predictions,labels=range(4)).tolist(),folds=fr,source_order_predictions=predictions.tolist())
                results.append(result);write(folder/(mode+'.json'),result)
            except ConvergenceWarning as error:
                failed.append(dict(balance_mode=mode,status='unavailable_nonconvergence_not_scored',error=str(error)))
            write(folder/'cv_report.json',dict(status='internal_grouped_catalog_diagnostic',source_ids=[r['source_id'] for r in plan['rows']],
                results=results,failed=failed,features_shape=list(X.shape),final_classifier_exported=False))
        queue.setdefault('completed_arms',[]).append(arm);write(OUT/'queue.json',queue)


def summarize(spec):
    reports={arm:C.read_json(OUT/arm/'cv_report.json') for arm in ARMS}
    primary={arm:next((r for r in report['results'] if r['balance_mode']=='balanced'),None) for arm,report in reports.items()}
    assert all(r is not None for r in primary.values()),'Primary score unavailable; do not select a different configuration'
    result=dict(status='completed_internal_roi_context_pooling',created_utc=utc(),primary_configuration=spec['primary'],
        targets=1083,primary_results=primary,configurations_per_arm={arm:len(r['results']) for arm,r in reports.items()},
        failed_configurations={arm:r['failed'] for arm,r in reports.items()},limitations=spec['limitations'],
        features_sha256=C.sha256(OUT/'pooled_features.npz'),protocol_sha256=C.sha256(OUT/'protocol.json'),
        reference_predictions_exactly_reproduced=True,external_evaluation_read=False,labels_changed=False)
    write(OUT/'comparison.json',result)
    names=dict(full_cls='原整图CLS',full_patch_mean='整图局部特征平均',roi_only='只汇总目标区域',roi_plus_ring='目标＋10%周边',roi_plus_cls='目标＋整图CLS',roi_duplicate='重复目标特征对照')
    lines=['# 完整图上的目标与周边特征对照','','同一1083张原生224图、人工框、类型、40空间组及三折；没有遮黑或改动任何输入像素，DINOv2冻结。',
        '主配置预先固定LR逆频率，主要对比目标＋10%周边与原整图CLS。','',
        '|特征读取方式|四类平均F1|分类正确率|SL正确|SP正确|FL正确|CL正确|','|---|---:|---:|---:|---:|---:|---:|']
    for arm,r in primary.items():lines.append(f"|{names[arm]}|{r['out_of_fold_macro_f1']:.2%}|{r['per_class']['accuracy']:.2%}|"+'|'.join(str(r['confusion_matrix'][i][i]) for i in range(4))+'|')
    lines+=['','|LR加权|'+ '|'.join(names[a] for a in ARMS)+'|','|---|'+'---:|'*len(ARMS)]
    for mode in ['none','sqrt','balanced']:
        rr=[next((r for r in reports[a]['results'] if r['balance_mode']==mode),None) for a in ARMS]
        lines.append('|'+mode+'|'+'|'.join(f"{r['out_of_fold_macro_f1']:.2%}" if r else '未收敛，不计分' for r in rr)+'|')
    lines+=['','目标和周边都由完整图特征汇总，特征已通过自注意力看过整图，14像素块也可能跨边界；不是纯目标／纯背景分割。',
        '目录类型未独立复核，FL仅15个，使用人工框，尚无独立检测框与级联成绩。']
    (OUT/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def run():
    attempts=R.F.prevent_network();spec=check()
    assert C.read_json(OUT/'queue.json')['status']=='prepared_not_started'
    independence=R.F.process_independence();assert independence['scheduler_owned'];R.ensure_gpu_idle()
    kernel=ctypes.windll.kernel32
    if not kernel.SetThreadExecutionState(0x80000001):raise ctypes.WinError()
    queue=dict(status='running',started_utc=utc(),pid=os.getpid(),phase='checking_inputs',process_independence=independence,
               offline=True,temporary_system_awake=True,completed_targets=0)
    write(OUT/'queue.json',queue)
    try:
        plan,features,y,fold=extract(spec,queue);fit(plan,features,y,fold,queue,spec);check();result=summarize(spec)
        queue.update(status=result['status'],phase='complete')
    except BaseException as error:
        queue.update(status='failed',error=repr(error));raise
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        queue.update(finished_utc=utc(),temporary_system_awake=False,blocked_network_attempts=attempts);write(OUT/'queue.json',queue)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','scheduled-run']);args=parser.parse_args()
    if args.mode=='prepare':prepare()
    else:
        sys.stdout=(OUT/'training.log').open('x',encoding='utf-8',buffering=1)
        sys.stderr=(OUT/'training.err.log').open('x',encoding='utf-8',buffering=1)
        run()
