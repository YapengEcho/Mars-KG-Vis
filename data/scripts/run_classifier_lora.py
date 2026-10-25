"""Fixed nested-spatial head-only/LoRA comparison, local and resumable.

No outer-fold early stopping, augmentation, new labels or external evaluation.
Each outer fit selects epochs on an inner spatial split, then refits from the
same initialization on all outer-training IDs. Frozen baseline features are
cached; both arms use the same head, normalization, batches and weighted loss.
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

OUT=ROOT/'runs/experiments/classifier_lora_20261015_v1'
DESIGN=ROOT/'runs/research/next_model_experiments_20261015_v1/classifier_adaptation_plan.json'
ARMS=['head_only','qv_last2_LoRA']
NAMES=['SL','SP','FL','CL']

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

def prepare():
    C.new_directory(OUT);design=C.read_json(DESIGN);source=C.read_json(design['source_plan'])
    assert C.sha256(design['source_plan'])==design['source_plan_sha256'] and len(source['rows'])==1083
    assert design['ordered_source_ids']==[r['source_id'] for r in source['rows']]
    for row in source['rows']:assert C.sha256(row['crop'])==row['crop_sha256']
    for split in design['outer_folds']:
        a,b,t,h=[set(split[k]) for k in ['inner_fit_indices','inner_stop_indices','outer_train_indices','outer_held_indices']]
        assert not a&b and a|b==t and not t&h and t|h==set(range(1083))
        group=lambda ids:{source['rows'][i]['spatial_group_id'] for i in ids}
        assert not group(a)&group(b) and not group(t)&group(h)
    checkpoint=C.read_json(P.OUT/'protocol.json')['checkpoint']
    files=[DESIGN,Path(design['source_plan']),P.OUT/'pooled_features.npz',P.OUT/'pooling_weights.npz',P.OUT/'protocol.json']
    code=[Path(__file__),ROOT/'data/scripts/landslide_model_candidates.py',Path(C.__file__),Path(FENV.__file__),
          ROOT/'data/scripts/start_classifier_lora.ps1',ROOT/'data/scripts/train_stage2_cls.py']
    spec=dict(status='locked_before_GPU_smoke_and_training',created_utc=utc(),design=design,
        source_plan=design['source_plan'],checkpoint=checkpoint,checkpoint_files_sha256={n:C.sha256(Path(checkpoint)/n) for n in ['config.json','model.safetensors']},
        files_sha256={str(p):C.sha256(p) for p in files},code_hashes={str(p):C.sha256(p) for p in code},
        precision='FP32; TF32/AMP disabled',batch=8,accumulate=2,seed=20261015,max_epochs=40,patience=8,
        optimizer=dict(name='AdamW',head_lr=.001,adapter_lr=.0001,weight_decay=.0001,gradient_clip_norm=1,foreach=False),
        normalization='Fixed StandardScaler fit only initial frozen ROI features of the actual training IDs; identical per-stage in both arms.',
        loss='sqrt inverse-frequency mean-one weights from actual fitting IDs; weighted CE normalized across each effective batch of up to16 IDs',
        encoder_mode='eval in both arms; gradients enabled only for LoRA q/v in final2 blocks; dropout and stochastic depth disabled',
        checkpoint_policy='last each complete epoch includes optimizer/history; manual resume only; best selected solely on inner spatial macro-F1',
        baseline='Cached frozen ROI endpoint, same SGD-style minibatch optimization of a linear CE head; not the previous LR classifier',
        selection_ties='strict improvement1e-12, earlier epoch retained',test_images_read=False,automatic_thread_continuation=False,
        versions={k:v for k,v in P.R.package_versions().items()})
    write(OUT/'protocol.json',spec);(OUT/'protocol.sha256').write_text(C.sha256(OUT/'protocol.json')+'\n',encoding='utf-8')
    write(OUT/'queue.json',dict(status='prepared_not_started',created_utc=utc()))
    print('PREPARED_NESTED_CLASSIFIER_LORA',flush=True)

def check():
    spec=C.read_json(OUT/'protocol.json')
    assert C.sha256(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for field in ['files_sha256','code_hashes']:
        for p,d in spec[field].items():assert C.sha256(p)==d,p
    for name,d in spec['checkpoint_files_sha256'].items():assert C.sha256(Path(spec['checkpoint'])/name)==d
    assert spec['versions']==P.R.package_versions()
    return spec

def load_data(spec):
    source=C.read_json(spec['source_plan']);rows=source['rows'];images=[]
    for row in rows:
        assert C.sha256(row['crop'])==row['crop_sha256']
        with Image.open(row['crop']) as im:
            assert im.mode=='L' and im.size==(224,224);images.append(np.array(im))
    # 54MB of gray pixels, batch normalization follows the historical CPU path.
    gray=np.stack(images)
    with np.load(P.OUT/'pooled_features.npz',allow_pickle=False) as d:
        X=torch.from_numpy(d['roi_only'].copy());y=d['y'].copy()
        assert np.array_equal(d['source_id'],[r['source_id'] for r in rows]) and np.array_equal(y,[r['class_id'] for r in rows])
    with np.load(P.OUT/'pooling_weights.npz',allow_pickle=False) as d:roi=torch.from_numpy(d['roi'].astype(np.float32))
    assert X.shape==(1083,768) and roi.shape==(1083,256)
    return dict(gray=gray,X=X,y=torch.as_tensor(y,dtype=torch.long),roi=roi,rows=rows)

def pixels(data,indices):
    arr=np.repeat(data['gray'][indices][...,None],3,axis=-1)
    value=torch.from_numpy(arr).permute(0,3,1,2).float()/255
    return ((value-torch.tensor([.485,.456,.406]).view(1,3,1,1))/torch.tensor([.229,.224,.225]).view(1,3,1,1)).to('cuda')

class Classifier(nn.Module):
    def __init__(self,spec,arm,fit,data,fold,phase):
        super().__init__();self.encoder=None;self.arm=arm
        s=spec['seed']+fold*101+(10000 if phase=='refit' else 0);seed(s)
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

def save_checkpoint(path,model,opt,history,epoch,best_epoch,best_score,arm,fold,phase):
    atomic_torch(path,dict(protocol_sha256=C.sha256(OUT/'protocol.json'),parameters=parameter_state(model),optimizer=opt.state_dict(),
        history=history,completed_epoch=epoch,best_epoch=best_epoch,best_score=best_score,arm=arm,fold=fold,phase=phase))

def smoke():
    spec=check();device_setup();blocked=FENV.prevent_network();data=load_data(spec);split=spec['design']['outer_folds'][0];fit=split['inner_fit_indices']
    ids=[]
    for cls in range(4):ids += [i for i in fit if data['y'][i]==cls][:4]
    ids=np.array(ids,dtype=int);assert len(ids)==16
    torch.cuda.reset_peak_memory_stats();model=Classifier(spec,ARMS[1],fit,data,0,'select');opt=optimizer(model,spec);w=weights(data,fit)
    assert sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)==24576
    frozen_before=tensor_hash({k:v for k,v in model.encoder.named_parameters() if not v.requires_grad})
    with torch.no_grad():
        actual=model.features(data,ids[:8]).cpu();expected=data['X'][ids[:8]]
        maxdiff=float((actual-expected).abs().max());assert torch.allclose(actual,expected,rtol=2e-4,atol=2e-4),maxdiff
    a=update_group(model,opt,data,ids,w)
    temporary=OUT/'smoke_checkpoint.pt';save_checkpoint(temporary,model,opt,[],1,1,0.,ARMS[1],0,'select')
    before_prediction=predict(model,data,ids[:8])[1]
    b=update_group(model,opt,data,ids,w);expected_state=parameter_state(model)
    assert any(v.abs().sum()>0 for k,v in expected_state.items() if k.endswith('.A'))
    assert frozen_before==tensor_hash({k:v for k,v in model.encoder.named_parameters() if not v.requires_grad})
    peak=torch.cuda.max_memory_allocated()/2**20
    del model,opt;gc.collect();torch.cuda.empty_cache()
    model=Classifier(spec,ARMS[1],fit,data,0,'select');opt=optimizer(model,spec);saved=torch.load(temporary,map_location='cpu',weights_only=False)
    load_parameters(model,saved['parameters']);opt.load_state_dict(saved['optimizer'])
    assert np.array_equal(before_prediction,predict(model,data,ids[:8])[1])
    resumed=update_group(model,opt,data,ids,w)
    assert tensor_hash(expected_state)==tensor_hash(parameter_state(model)) and resumed==b
    trained_adapters={k:v for k,v in parameter_state(model).items() if not k.startswith('head.')}
    assert any(v.abs().sum()>0 for k,v in trained_adapters.items() if k.endswith('.B'))
    del model,opt;gc.collect();torch.cuda.empty_cache()
    head=Classifier(spec,ARMS[0],fit,data,0,'select');head_opt=optimizer(head,spec);c=update_group(head,head_opt,data,ids,w)
    result=dict(status='passed_real_GPU_smoke',created_utc=utc(),images=16,classes=4,adapter_parameters=24576,
        gradient_steps=3,losses=[a,b,c],frozen_weights_unchanged=True,cached_ROI_initial_max_abs_diff=maxdiff,
        checkpoint_prediction_bitwise_equal=True,resumed_next_update_bitwise_equal=True,peak_allocated_mib=peak,
        CUDA_device=torch.cuda.get_device_name(),input_precision='FP32',blocked_network_attempts=blocked,
        protocol_sha256=C.sha256(OUT/'protocol.json'))
    write(OUT/'readiness.json',result);print(json.dumps(result),flush=True)

def stage(spec,data,split,arm,phase,queue):
    fold=split['outer_fold'];folder=OUT/arm/f'fold{fold}'/phase;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completed.json').exists():return C.read_json(folder/'completed.json')
    fit=split['inner_fit_indices'] if phase=='select' else split['outer_train_indices']
    held=split['inner_stop_indices'] if phase=='select' else split['outer_held_indices']
    limit=spec['max_epochs'] if phase=='select' else C.read_json(folder.parent/'select/completed.json')['best_epoch']
    model=Classifier(spec,arm,fit,data,fold,phase);opt=optimizer(model,spec);w=weights(data,fit)
    start=1;history=[];best=-1.;best_epoch=0
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        assert saved['protocol_sha256']==C.sha256(OUT/'protocol.json') and (saved['arm'],saved['fold'],saved['phase'])==(arm,fold,phase)
        load_parameters(model,saved['parameters']);opt.load_state_dict(saved['optimizer'])
        history=saved['history'];start=saved['completed_epoch']+1;best=saved['best_score'];best_epoch=saved['best_epoch']
    torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
    for epoch in range(start,limit+1):
        if phase=='select' and epoch-1-best_epoch>=spec['patience']:break
        seed_value=spec['seed']+fold*101+(10000 if phase=='refit' else 0)+epoch
        order=np.random.default_rng(seed_value).permutation(np.array(fit,dtype=int));losses=[]
        for offset in range(0,len(order),16):
            losses.append(update_group(model,opt,data,order[offset:offset+16],w))
            queue.update(arm=arm,outer_fold=fold,phase=phase,epoch=epoch,batches_completed=(offset//16)+1,batches_per_epoch=(len(order)+15)//16)
            if offset%128==0:write(OUT/'queue.json',queue)
        row=dict(epoch=epoch,mean_effective_batch_loss=float(np.mean(losses)))
        if phase=='select':
            predictions,_=predict(model,data,held);score=float(f1_score(data['y'][held].numpy(),predictions,labels=range(4),average='macro',zero_division=0))
            row['inner_macro_f1']=score
            if score>best+1e-12:
                best=score;best_epoch=epoch
                save_checkpoint(folder/'best.pt',model,opt,history+[row],epoch,best_epoch,best,arm,fold,phase)
        history.append(row);save_checkpoint(folder/'last.pt',model,opt,history,epoch,best_epoch,best,arm,fold,phase)
        write(folder/'metrics.json',dict(history=history,train_ID_count=len(fit),inner_stop_ID_count=len(held) if phase=='select' else 0,train_only_class_weights=w.cpu().tolist()))
        queue.update(completed_epoch=epoch,inner_best_epoch=best_epoch,inner_best_macro_f1=best,epoch_seconds=(time.perf_counter()-began)/(epoch-start+1));write(OUT/'queue.json',queue)
        print(f'{arm} fold{fold} {phase} epoch{epoch} loss={row["mean_effective_batch_loss"]:.5f} inner={row.get("inner_macro_f1","not_read")}',flush=True)
    result=dict(status='completed',arm=arm,fold=fold,phase=phase,completed_epochs=history[-1]['epoch'],best_epoch=best_epoch,
                best_inner_macro_f1=best,fit_unique_IDs=len(fit),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    if phase=='refit':
        predictions,logits=predict(model,data,held);np.savez_compressed(folder/'outer_predictions.npz',indices=np.array(held),predictions=predictions,logits=logits)
        result.update(selected_epochs=limit,outer_targets=len(held),outer_read_only_after_refit=True)
    write(folder/'completed.json',result);del model,opt;gc.collect();torch.cuda.empty_cache();return result

def summarize(spec,data):
    results={}
    for arm in ARMS:
        oof=np.full(1083,-1,dtype=int);folds=[]
        for split in spec['design']['outer_folds']:
            folder=OUT/arm/f"fold{split['outer_fold']}"/'refit'
            with np.load(folder/'outer_predictions.npz',allow_pickle=False) as d:
                ids=d['indices'];assert set(ids)==set(split['outer_held_indices']) and np.all(oof[ids]==-1);oof[ids]=d['predictions']
            folds.append(dict(fold=split['outer_fold'],macro_f1=float(f1_score(data['y'][ids],oof[ids],labels=range(4),average='macro',zero_division=0))))
        assert np.all(oof>=0)
        results[arm]=dict(macro_f1=float(f1_score(data['y'],oof,labels=range(4),average='macro',zero_division=0)),
            accuracy=float(np.mean(oof==data['y'].numpy())),per_class=classification_report(data['y'],oof,labels=range(4),target_names=NAMES,output_dict=True,zero_division=0),
            confusion_matrix=confusion_matrix(data['y'],oof,labels=range(4)).tolist(),folds=folds,predictions=oof.tolist())
    write(OUT/'comparison.json',dict(status='completed_internal_nested_spatial_LoRA_comparison',created_utc=utc(),unique_targets=1083,spatial_groups=40,
        test_images_read=False,labels_modified=False,all_types_independently_reviewed=False,results=results,protocol_sha256=C.sha256(OUT/'protocol.json')))
    lines=['# 四分类少量微调：空间嵌套对照','','|方案|OOF macro-F1|正确率|','|---|---:|---:|']
    for arm,r in results.items():lines.append(f'|{arm}|{r["macro_f1"]:.2%}|{r["accuracy"]:.2%}|')
    lines+=['','仅内部空间交叉验证，目录类型未独立核准；不是新验证区或完整检测流程成绩。','每折只用内层空间集合早停，按所选轮数重新训练全部外层训练编号，外层只在最后评分。']
    (OUT/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def run(resume=False):
    spec=check();ready=C.read_json(OUT/'readiness.json');assert ready['status']=='passed_real_GPU_smoke' and ready['protocol_sha256']==C.sha256(OUT/'protocol.json')
    state=C.read_json(OUT/'queue.json');assert state['status'] in (['failed','running'] if resume else ['prepared_not_started'])
    independence=FENV.process_independence();assert independence['scheduler_owned'];P.R.ensure_gpu_idle()
    blocked=FENV.prevent_network();device_setup();kernel=ctypes.windll.kernel32
    assert kernel.SetThreadExecutionState(0x80000001)
    queue=dict(status='running',started_utc=state.get('started_utc',utc()),pid=os.getpid(),phase='loading_local_training_inputs',
               process_independence=independence,temporary_system_awake=True,offline=True,manual_resume=resume)
    write(OUT/'queue.json',queue)
    try:
        data=load_data(spec)
        for split in spec['design']['outer_folds']:
            for arm in ARMS:
                for phase in ['select','refit']:stage(spec,data,split,arm,phase,queue)
        check();summarize(spec,data);queue.update(status='completed_internal_nested_spatial_LoRA_comparison',phase='complete')
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
