"""Multiseed nested-spatial four-class fine-tuning comparison (head-only vs LoRA).

Differs from classifier_lora_20261015_v1 in exactly two rigor-relevant ways:
  * the corrected pixel-donor spatial grouping (44 groups) replaces the
    superseded 40-group split that reused donor tiles across folds;
  * the whole nested-spatial pipeline is repeated over several seeds and is
    reported as mean/std with per-seed paired differences.

Everything else is held identical to v1: same fixed head, normalization, loss,
batch, optimizer, encoder mode, inner early-stopping selection and refit rules;
the encoder stays frozen; runs are deterministic, offline and never read test
images. Inner early stopping uses an inner spatial split of the outer-training
IDs only; the outer held-out IDs are scored exactly once per fold.
"""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.venv/Lib/site-packages'))
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_OFFLINE='true',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from sklearn.preprocessing import StandardScaler
import extract_manual_classification_features as C
import run_roi_context_pooling as P
import run_full_pretrained_detector as FENV
from landslide_model_candidates import attach_dinov2_lora
from train_stage2_cls import classifier_class_weights

OUT=ROOT/'runs/experiments/classifier_lora_multiseed_20261015_v1'
PLAN=ROOT/'runs/research/classification_pixel_provenance_20261015_v2/plan.json'
CONTEXT_PLAN=ROOT/'runs/experiments/context_margin_shape_20261015_v2/context_full/plan.json'
DESIGN_OUT=OUT/'design.json'
ARMS=['head_only','qv_last2_LoRA']
NAMES=['SL','SP','FL','CL']
SEEDS=[20261015,20261016,20261017]


def utc():return datetime.now(timezone.utc).isoformat()
def write(path,value):C.write_json_atomic(path,value)
def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
def device_setup():
    assert torch.cuda.is_available(),'No CPU fallback'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
def atomic_torch(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.writing')
    torch.save(value,temp);os.replace(temp,path)
def tensor_hash(values):
    digest=hashlib.sha256()
    for key,t in sorted(values.items()):
        t=t.detach().cpu().contiguous();digest.update(key.encode());digest.update(t.numpy().tobytes())
    return digest.hexdigest()


def build_design():
    corrected=C.read_json(PLAN);crows=corrected['rows']
    images=C.read_json(CONTEXT_PLAN);irows=images['rows']
    assert len(crows)==1083 and corrected['folds']==3 and len(irows)==1083
    assert corrected['class_mapping']=={'SL':0,'SP':1,'FL':2,'CL':3}
    img={r['source_id']:r for r in irows}
    assert set(img)=={r['source_id'] for r in crows},'Corrected and native image plans must cover the same targets'
    rows=[]
    for r in crows:
        m=img[r['source_id']]
        assert m['class_id']==r['class_id'] and m['catalog_class']==r['catalog_class'],'Class drift between grouping and image plans'
        assert C.sha256(m['crop'])==m['crop_sha256']
        rows.append(dict(source_id=r['source_id'],catalog_class=r['catalog_class'],class_id=r['class_id'],
            crop=m['crop'],crop_sha256=m['crop_sha256'],spatial_group_id=r['spatial_group_id'],fold_id=r['fold_id']))
    y=np.array([r['class_id'] for r in rows]);groups=np.array([r['spatial_group_id'] for r in rows])
    folds=np.array([r['fold_id'] for r in rows])
    assert len({r['source_id'] for r in rows})==1083
    assert len(set(groups))==44,'Corrected grouping must have 44 spatial groups'
    assert np.bincount(y,minlength=4).tolist()==[531,94,15,443]
    plans=[]
    for outer in range(3):
        train=np.flatnonzero(folds!=outer);held=np.flatnonzero(folds==outer)
        candidates=[]
        # Split using training class counts only, never features or held-out scores.
        splitter=GroupShuffleSplit(n_splits=200,test_size=.2,random_state=20261015+outer)
        for attempt,(a,b) in enumerate(splitter.split(np.zeros((len(train),1)),y[train],groups[train])):
            fit,stop=train[a],train[b]
            fc=np.bincount(y[fit],minlength=4);sc=np.bincount(y[stop],minlength=4)
            if min(fc)==0 or min(sc)==0 or fc[2]<5:continue
            deviation=float(np.abs(sc/np.bincount(y[train],minlength=4)-.2).mean())
            candidates.append((deviation,attempt,fit,stop))
        assert candidates,f'No valid inner split for corrected fold {outer}'
        deviation,attempt,fit,stop=min(candidates,key=lambda x:(x[0],x[1]))
        assert set(fit)|set(stop)==set(train) and not set(fit)&set(stop)
        assert not set(groups[fit])&set(groups[stop])
        assert not set(groups[train])&set(groups[held])
        plans.append(dict(outer_fold=outer,outer_train_indices=train.tolist(),outer_held_indices=held.tolist(),
            inner_fit_indices=fit.tolist(),inner_stop_indices=stop.tolist(),inner_count_only_attempt=attempt,
            inner_count_deviation=deviation,
            counts={k:np.bincount(y[v],minlength=4).tolist() for k,v in [('fit',fit),('stop',stop),('held',held)]}))
    return dict(status='prepared_not_started; multiseed runner and GPU smoke still required',
        grouping='corrected pixel-donor spatial grouping (44 groups) with native224 images',
        source_plan=str(PLAN),source_plan_sha256=C.sha256(PLAN),
        image_plan=str(CONTEXT_PLAN),image_plan_sha256=C.sha256(CONTEXT_PLAN),unique_targets=1083,spatial_groups=44,
        class_counts={NAMES[i]:int(np.bincount(y,minlength=4)[i]) for i in range(4)},
        all_types_independently_reviewed=False,seeds=SEEDS,rows=rows,outer_folds=plans)



def prepare():
    C.new_directory(OUT);design=build_design();write(DESIGN_OUT,design)
    checkpoint=C.read_json(P.OUT/'protocol.json')['checkpoint']
    files=[PLAN,CONTEXT_PLAN,DESIGN_OUT,P.OUT/'pooled_features.npz',P.OUT/'pooling_weights.npz',P.OUT/'protocol.json']
    code=[Path(__file__),ROOT/'data/scripts/landslide_model_candidates.py',Path(C.__file__),Path(FENV.__file__),
          ROOT/'data/scripts/start_classifier_lora_multiseed.ps1',ROOT/'data/scripts/train_stage2_cls.py']
    spec=dict(status='locked_before_GPU_smoke_and_training',created_utc=utc(),design=design,
        design_json=str(DESIGN_OUT),design_sha256=C.sha256(DESIGN_OUT),checkpoint=checkpoint,
        checkpoint_files_sha256={n:C.sha256(Path(checkpoint)/n) for n in ['config.json','model.safetensors']},
        files_sha256={str(p):C.sha256(p) for p in files},code_hashes={str(p):C.sha256(p) for p in code},
        seeds=SEEDS,arms=ARMS,precision='FP32; TF32/AMP disabled',batch=8,accumulate=2,max_epochs=40,patience=8,
        optimizer=dict(name='AdamW',head_lr=.001,adapter_lr=.0001,weight_decay=.0001,gradient_clip_norm=1,foreach=False),
        normalization='Fixed StandardScaler fit only initial frozen ROI features of the actual training IDs; identical per-stage in both arms.',
        loss='sqrt inverse-frequency mean-one weights from actual fitting IDs; weighted CE normalized across each effective batch of up to16 IDs',
        encoder_mode='eval in both arms; gradients enabled only for LoRA q/v in final2 blocks; dropout and stochastic depth disabled',
        checkpoint_policy='last each complete epoch includes optimizer/history; manual resume only; best selected solely on inner spatial macro-F1',
        baseline='Cached frozen ROI endpoint, same SGD-style minibatch optimization of a linear CE head; not the previous LR classifier',
        selection_ties='strict improvement1e-12, earlier epoch retained',test_images_read=False,automatic_thread_continuation=False,
        versions={k:v for k,v in P.R.package_versions().items()})
    write(OUT/'protocol.json',spec);(OUT/'protocol.sha256').write_text(C.sha256(OUT/'protocol.json')+'\n',encoding='utf-8')
    write(OUT/'queue.json',dict(status='prepared_not_started',created_utc=utc(),seeds=SEEDS))
    print('PREPARED_MULTISEED_CLASSIFIER_LORA',flush=True)


def check():
    spec=C.read_json(OUT/'protocol.json')
    assert C.sha256(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for field in ['files_sha256','code_hashes']:
        for p,d in spec[field].items():assert C.sha256(p)==d,p
    for name,d in spec['checkpoint_files_sha256'].items():assert C.sha256(Path(spec['checkpoint'])/name)==d
    assert spec['design_sha256']==C.sha256(DESIGN_OUT)
    assert spec['versions']==P.R.package_versions()
    return spec


def load_data(spec):
    rows=spec['design']['rows'];images=[]
    for row in rows:
        assert C.sha256(row['crop'])==row['crop_sha256']
        with Image.open(row['crop']) as im:
            assert im.mode=='L' and im.size==(224,224);images.append(np.array(im))
    gray=np.stack(images)
    with np.load(P.OUT/'pooled_features.npz',allow_pickle=False) as d:
        order=np.array([r['source_id'] for r in rows]);stored=d['source_id']
        assert set(order.tolist())==set(stored.tolist())
        index={s:i for i,s in enumerate(stored.tolist())}
        take=np.array([index[s] for s in order.tolist()])
        X=torch.from_numpy(d['roi_only'][take].copy());y=d['y'][take].copy()
    with np.load(P.OUT/'pooling_weights.npz',allow_pickle=False) as d:
        index={s:i for i,s in enumerate(d['source_id'].tolist())}
        roi=torch.from_numpy(d['roi'][np.array([index[s] for s in order.tolist()])].astype(np.float32))
    assert np.array_equal(y,[r['class_id'] for r in rows]),'Feature class order drift'
    assert X.shape==(1083,768) and roi.shape==(1083,256)
    return dict(gray=gray,X=X,y=torch.as_tensor(y,dtype=torch.long),roi=roi,rows=rows)


def pixels(data,indices):
    arr=np.repeat(data['gray'][indices][...,None],3,axis=-1)
    value=torch.from_numpy(arr).permute(0,3,1,2).float()/255
    return ((value-torch.tensor([.485,.456,.406]).view(1,3,1,1))/torch.tensor([.229,.224,.225]).view(1,3,1,1)).to('cuda')


class Classifier(nn.Module):
    def __init__(self,spec,arm,fit,data,fold,phase,run_seed):
        super().__init__();self.encoder=None;self.arm=arm
        s=run_seed+fold*101+(10000 if phase=='refit' else 0);seed(s)
        if arm!='head_only':
            from transformers import AutoModel
            self.encoder=AutoModel.from_pretrained(spec['checkpoint'],local_files_only=True,use_safetensors=True).float().eval()
            self.adapter_names=attach_dinov2_lora(self.encoder,last_blocks=2,rank=4)
        # Same head tensors, independent of RNG consumed by adapter creation.
        seed(s+11);self.head=nn.Linear(768,4)
        scaler=StandardScaler().fit(data['X'][fit].numpy())
        self.register_buffer('mean',torch.from_numpy(scaler.mean_.astype(np.float32)))
        self.register_buffer('scale',torch.from_numpy(scaler.scale_.astype(np.float32)))
        self.to('cuda');self.eval()
    def features(self,data,indices):
        if self.encoder is None:return data['X'][indices].to('cuda')
        tokens=self.encoder(pixel_values=pixels(data,indices)).last_hidden_state
        return torch.einsum('bp,bpd->bd',data['roi'][indices].to('cuda'),tokens[:,1:])
    def forward(self,data,indices):return self.head((self.features(data,indices)-self.mean)/self.scale)


def optimizer(model,spec):
    groups=[dict(params=list(model.head.parameters()),lr=spec['optimizer']['head_lr'])]
    if model.encoder is not None:groups.append(dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=spec['optimizer']['adapter_lr']))
    return torch.optim.AdamW(groups,weight_decay=spec['optimizer']['weight_decay'],foreach=False)
def weights(data,fit):
    w=classifier_class_weights(data['y'][fit].numpy(),'sqrt')
    assert set(w)==set(range(4));return torch.tensor([w[i] for i in range(4)],device='cuda',dtype=torch.float32)
def parameter_state(model):return {k:v.detach().cpu().clone() for k,v in model.named_parameters() if v.requires_grad}
def load_parameters(model,state):
    params={k:v for k,v in model.named_parameters() if v.requires_grad};assert set(state)==set(params)
    with torch.no_grad():
        for k,v in state.items():params[k].copy_(v.to(params[k].device))


def update_group(model,opt,data,indices,w):
    opt.zero_grad(set_to_none=True);target=data['y'][indices].to('cuda');denominator=w[target].sum();loss_sum=0.
    for start in range(0,len(indices),8):
        batch=indices[start:start+8];labels=data['y'][batch].to('cuda')
        logits=model(data,batch);loss=F.cross_entropy(logits,labels,weight=w,reduction='sum')/denominator
        assert torch.isfinite(loss);loss.backward();loss_sum+=float(loss.detach())
    params=[p for p in model.parameters() if p.requires_grad]
    for p in params:assert p.grad is not None and torch.isfinite(p.grad).all()
    norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
    opt.step();return loss_sum


def predict(model,data,indices):
    values=[]
    with torch.no_grad():
        for start in range(0,len(indices),8):values.append(model(data,indices[start:start+8]).cpu())
    logits=torch.cat(values).numpy();assert np.isfinite(logits).all()
    return logits.argmax(1),logits


def save_checkpoint(path,model,opt,history,epoch,best_epoch,best_score,arm,fold,phase,run_seed):
    atomic_torch(path,dict(protocol_sha256=C.sha256(OUT/'protocol.json'),parameters=parameter_state(model),optimizer=opt.state_dict(),
        history=history,completed_epoch=epoch,best_epoch=best_epoch,best_score=best_score,arm=arm,fold=fold,phase=phase,run_seed=run_seed))


def smoke():
    spec=check();device_setup();blocked=FENV.prevent_network();data=load_data(spec)
    split=spec['design']['outer_folds'][0];fit=split['inner_fit_indices'];run_seed=SEEDS[0]
    ids=[]
    for cls in range(4):ids += [i for i in fit if data['y'][i]==cls][:4]
    ids=np.array(ids,dtype=int);assert len(ids)==16
    torch.cuda.reset_peak_memory_stats();model=Classifier(spec,ARMS[1],fit,data,0,'select',run_seed);opt=optimizer(model,spec);w=weights(data,fit)
    assert sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)==24576
    frozen_before=tensor_hash({k:v for k,v in model.encoder.named_parameters() if not v.requires_grad})
    with torch.no_grad():
        actual=model.features(data,ids[:8]).cpu();expected=data['X'][ids[:8]]
        maxdiff=float((actual-expected).abs().max());assert torch.allclose(actual,expected,rtol=2e-4,atol=2e-4),maxdiff
    a=update_group(model,opt,data,ids,w)
    temporary=OUT/'smoke_checkpoint.pt';save_checkpoint(temporary,model,opt,[],1,1,0.,ARMS[1],0,'select',run_seed)
    before_prediction=predict(model,data,ids[:8])[1]
    b=update_group(model,opt,data,ids,w);expected_state=parameter_state(model)
    assert any(v.abs().sum()>0 for k,v in expected_state.items() if k.endswith('.A'))
    assert frozen_before==tensor_hash({k:v for k,v in model.encoder.named_parameters() if not v.requires_grad})
    peak=torch.cuda.max_memory_allocated()/2**20
    del model,opt;gc.collect();torch.cuda.empty_cache()
    model=Classifier(spec,ARMS[1],fit,data,0,'select',run_seed);opt=optimizer(model,spec);saved=torch.load(temporary,map_location='cpu',weights_only=False)
    load_parameters(model,saved['parameters']);opt.load_state_dict(saved['optimizer'])
    assert np.array_equal(before_prediction,predict(model,data,ids[:8])[1])
    resumed=update_group(model,opt,data,ids,w)
    assert tensor_hash(expected_state)==tensor_hash(parameter_state(model)) and resumed==b
    trained_adapters={k:v for k,v in parameter_state(model).items() if not k.startswith('head.')}
    assert any(v.abs().sum()>0 for k,v in trained_adapters.items() if k.endswith('.B'))
    del model,opt;gc.collect();torch.cuda.empty_cache()
    head=Classifier(spec,ARMS[0],fit,data,0,'select',run_seed);head_opt=optimizer(head,spec);c=update_group(head,head_opt,data,ids,w)
    result=dict(status='passed_real_GPU_smoke',created_utc=utc(),images=16,classes=4,adapter_parameters=24576,
        gradient_steps=3,losses=[a,b,c],frozen_weights_unchanged=True,cached_ROI_initial_max_abs_diff=maxdiff,
        checkpoint_prediction_bitwise_equal=True,resumed_next_update_bitwise_equal=True,peak_allocated_mib=peak,
        CUDA_device=torch.cuda.get_device_name(),input_precision='FP32',blocked_network_attempts=blocked,
        protocol_sha256=C.sha256(OUT/'protocol.json'))
    write(OUT/'readiness.json',result);print(json.dumps(result),flush=True)


def stage(spec,data,split,arm,phase,queue,run_seed):
    fold=split['outer_fold'];folder=OUT/arm/f'seed{run_seed}'/f'fold{fold}'/phase;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completed.json').exists():return C.read_json(folder/'completed.json')
    fit=split['inner_fit_indices'] if phase=='select' else split['outer_train_indices']
    held=split['inner_stop_indices'] if phase=='select' else split['outer_held_indices']
    limit=spec['max_epochs'] if phase=='select' else C.read_json(folder.parent/'select/completed.json')['best_epoch']
    model=Classifier(spec,arm,fit,data,fold,phase,run_seed);opt=optimizer(model,spec);w=weights(data,fit)
    start=1;history=[];best=-1.;best_epoch=0
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        assert saved['protocol_sha256']==C.sha256(OUT/'protocol.json') and (saved['arm'],saved['fold'],saved['phase'],saved['run_seed'])==(arm,fold,phase,run_seed)
        load_parameters(model,saved['parameters']);opt.load_state_dict(saved['optimizer'])
        history=saved['history'];start=saved['completed_epoch']+1;best=saved['best_score'];best_epoch=saved['best_epoch']
    torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
    for epoch in range(start,limit+1):
        if phase=='select' and epoch-1-best_epoch>=spec['patience']:break
        seed_value=run_seed+fold*101+(10000 if phase=='refit' else 0)+epoch
        order=np.random.default_rng(seed_value).permutation(np.array(fit,dtype=int));losses=[]
        for offset in range(0,len(order),16):
            losses.append(update_group(model,opt,data,order[offset:offset+16],w))
            queue.update(run_seed=run_seed,arm=arm,outer_fold=fold,phase=phase,epoch=epoch,batches_completed=(offset//16)+1,batches_per_epoch=(len(order)+15)//16)
            if offset%128==0:write(OUT/'queue.json',queue)
        row=dict(epoch=epoch,mean_effective_batch_loss=float(np.mean(losses)))
        if phase=='select':
            predictions,_=predict(model,data,held);score=float(f1_score(data['y'][held].numpy(),predictions,labels=range(4),average='macro',zero_division=0))
            row['inner_macro_f1']=score
            if score>best+1e-12:
                best=score;best_epoch=epoch
                save_checkpoint(folder/'best.pt',model,opt,history+[row],epoch,best_epoch,best,arm,fold,phase,run_seed)
        history.append(row);save_checkpoint(folder/'last.pt',model,opt,history,epoch,best_epoch,best,arm,fold,phase,run_seed)
        write(folder/'metrics.json',dict(history=history,train_ID_count=len(fit),inner_stop_ID_count=len(held) if phase=='select' else 0,train_only_class_weights=w.cpu().tolist()))
        queue.update(completed_epoch=epoch,inner_best_epoch=best_epoch,inner_best_macro_f1=best,epoch_seconds=(time.perf_counter()-began)/(epoch-start+1));write(OUT/'queue.json',queue)
        print(f'seed{run_seed} {arm} fold{fold} {phase} epoch{epoch} loss={row["mean_effective_batch_loss"]:.5f} inner={row.get("inner_macro_f1","not_read")}',flush=True)
    result=dict(status='completed',arm=arm,fold=fold,phase=phase,run_seed=run_seed,completed_epochs=history[-1]['epoch'],best_epoch=best_epoch,
                best_inner_macro_f1=best,fit_unique_IDs=len(fit),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    if phase=='refit':
        predictions,logits=predict(model,data,held);np.savez_compressed(folder/'outer_predictions.npz',indices=np.array(held),predictions=predictions,logits=logits)
        result.update(selected_epochs=limit,outer_targets=len(held),outer_read_only_after_refit=True)
    write(folder/'completed.json',result);del model,opt;gc.collect();torch.cuda.empty_cache();return result


def _arm_seed_metrics(data,arm,run_seed,folds):
    oof=np.full(1083,-1,dtype=int);per_fold=[]
    for split in folds:
        folder=OUT/arm/f'seed{run_seed}'/f"fold{split['outer_fold']}"/'refit'
        with np.load(folder/'outer_predictions.npz',allow_pickle=False) as d:
            ids=d['indices'];assert set(ids)==set(split['outer_held_indices']) and np.all(oof[ids]==-1);oof[ids]=d['predictions']
        per_fold.append(dict(fold=split['outer_fold'],macro_f1=float(f1_score(data['y'][ids],oof[ids],labels=range(4),average='macro',zero_division=0))))
    assert np.all(oof>=0)
    return dict(seed=run_seed,macro_f1=float(f1_score(data['y'],oof,labels=range(4),average='macro',zero_division=0)),
        accuracy=float(np.mean(oof==data['y'].numpy())),
        per_class=classification_report(data['y'],oof,labels=range(4),target_names=NAMES,output_dict=True,zero_division=0),
        confusion_matrix=confusion_matrix(data['y'],oof,labels=range(4)).tolist(),folds=per_fold,predictions=oof.tolist())


def summarize(spec,data):
    folds=spec['design']['outer_folds'];per_seed={arm:{s:_arm_seed_metrics(data,arm,s,folds) for s in SEEDS} for arm in ARMS}
    aggregate={}
    for arm in ARMS:
        m=[per_seed[arm][s]['macro_f1'] for s in SEEDS];a=[per_seed[arm][s]['accuracy'] for s in SEEDS]
        recalls={n:[per_seed[arm][s]['per_class'][n]['recall'] for s in SEEDS] for n in NAMES}
        aggregate[arm]=dict(macro_f1_mean=float(np.mean(m)),macro_f1_std=float(np.std(m)),
            accuracy_mean=float(np.mean(a)),accuracy_std=float(np.std(a)),
            per_class_recall_mean={n:float(np.mean(recalls[n])) for n in NAMES},
            per_class_recall_std={n:float(np.std(recalls[n])) for n in NAMES},
            per_class_recall_correct_mean={n:float(np.mean([per_seed[arm][s]['per_class'][n]['recall']*per_seed[arm][s]['per_class'][n]['support'] for s in SEEDS])) for n in NAMES})
    paired=[per_seed['qv_last2_LoRA'][s]['macro_f1']-per_seed['head_only'][s]['macro_f1'] for s in SEEDS]
    comparison=dict(status='completed_multiseed_nested_spatial_LoRA_comparison',created_utc=utc(),unique_targets=1083,spatial_groups=44,
        seeds=SEEDS,grouping='corrected pixel-donor grouping (44 groups)',test_images_read=False,labels_modified=False,
        all_types_independently_reviewed=False,arms=ARMS,per_seed=per_seed,aggregate=aggregate,
        paired_lora_minus_head=dict(per_seed=paired,mean=float(np.mean(paired)),std=float(np.std(paired)),
            wins=int(sum(1 for d in paired if d>0)),seeds=len(SEEDS)),protocol_sha256=C.sha256(OUT/'protocol.json'))
    write(OUT/'comparison.json',comparison)
    lines=['# 四分类微调对照：多种子重复（修正分组）','',
        '同一1083个编号、44空间组、修正后的3折；每折内层空间早停、外层只评一次；'+str(len(SEEDS))+'个种子。仅内部探索，类型标签未独立复核，不读测试集。','',
        '|方案|OOF macro-F1 均值±标准差|正确率 均值±标准差|SL召回|SP召回|FL召回|CL召回|','|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        g=aggregate[arm];r=g['per_class_recall_mean']
        lines.append(f'|{arm}|{g["macro_f1_mean"]:.2%}±{g["macro_f1_std"]:.2%}|{g["accuracy_mean"]:.2%}±{g["accuracy_std"]:.2%}|{r["SL"]:.1%}|{r["SP"]:.1%}|{r["FL"]:.1%}|{r["CL"]:.1%}|')
    pd=comparison['paired_lora_minus_head']
    lines+=['',f'配对差（+LoRA − head_only，逐种子）：均值 {pd["mean"]:+.2%}，标准差 {pd["std"]:.2%}，{pd["wins"]}/{pd["seeds"]} 个种子为正。',
        '','逐种子 OOF macro-F1：']
    for arm in ARMS:lines.append(f'- {arm}: '+' / '.join(f'{per_seed[arm][s]["macro_f1"]:.2%}' for s in SEEDS))
    (OUT/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(resume=False):
    spec=check();ready=C.read_json(OUT/'readiness.json');assert ready['status']=='passed_real_GPU_smoke' and ready['protocol_sha256']==C.sha256(OUT/'protocol.json')
    state=C.read_json(OUT/'queue.json');assert state['status'] in (['failed','running'] if resume else ['prepared_not_started'])
    independence=FENV.process_independence();assert independence['scheduler_owned'];P.R.ensure_gpu_idle()
    blocked=FENV.prevent_network();device_setup();kernel=ctypes.windll.kernel32
    assert kernel.SetThreadExecutionState(0x80000001)
    queue=dict(status='running',started_utc=state.get('started_utc',utc()),pid=os.getpid(),phase='loading_local_training_inputs',
               seeds=SEEDS,process_independence=independence,temporary_system_awake=True,offline=True,manual_resume=resume)
    write(OUT/'queue.json',queue)
    try:
        data=load_data(spec)
        for run_seed in SEEDS:
            for split in spec['design']['outer_folds']:
                for arm in ARMS:
                    for phase in ['select','refit']:stage(spec,data,split,arm,phase,queue,run_seed)
        check();summarize(spec,data);queue.update(status='completed_multiseed_nested_spatial_LoRA_comparison',phase='complete')
    except BaseException as e:queue.update(status='failed',error=repr(e));raise
    finally:
        kernel.SetThreadExecutionState(0x80000000);queue.update(finished_utc=utc(),temporary_system_awake=False,blocked_network_attempts=blocked);write(OUT/'queue.json',queue)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','smoke','scheduled-run','scheduled-resume']);args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='smoke':smoke()
    else:
        suffix='' if args.mode=='scheduled-run' else '_resume_'+str(int(time.time()))
        sys.stdout=(OUT/f'training{suffix}.log').open('x',encoding='utf-8',buffering=1);sys.stderr=(OUT/f'training{suffix}.err.log').open('x',encoding='utf-8',buffering=1)
        run(resume=args.mode=='scheduled-resume')
