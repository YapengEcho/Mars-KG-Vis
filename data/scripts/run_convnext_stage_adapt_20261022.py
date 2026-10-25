"""Paired native40 ConvNeXt head-only versus last-stage adaptation, one seed."""
from copy import deepcopy
import ctypes
import gc
import os
from pathlib import Path
import sys
import time

import run_convnext_frozen_comparison_20261022 as F

N,M,C,np,torch=F.N,F.M,F.C,F.np,F.torch
ROOT=F.ROOT
OUT=ROOT/'runs/experiments/convnext_stage_adapt_20261022_v1'
TASK='MarsKG-convnext-stage-adapt-20261022-v1'
ARMS=['head_only','last_stage']
SEED=20261015

def write(path,value):N.write(path,value)


def prepare():
    if OUT.exists():raise FileExistsError('Preserve existing experiment')
    parent=F.check()
    assert C.read_json(F.OUT/'queue.json')['status']=='completed'
    assert C.read_json(F.OUT/'independent_verification.json')['protocol_sha256']==C.sha256(F.OUT/'protocol.json')
    spec=deepcopy(parent)
    for key in ['encoder_updates','frozen_features_label_independent']:
        spec.pop(key,None)
    spec.update(created_utc=M.utc(),arms=ARMS,seed=SEED,max_epochs=40,patience=8,batch=8,accumulate=2,
        optimizer=dict(name='AdamW',head_lr=.001,adapter_lr=.00001,weight_decay=.0001,foreach=False,gradient_clip_norm=1.),
        primary='last_stage minus matched AdamW head_only, same initialization and mini-batch order',
        representation='ConvNeXt ROI, pretrained LayerNorm after area pooling; identical initial frozen features',
        trainable_encoder='features.7 (last three ConvNeXt blocks) and classifier.0 (final LayerNorm) only',
        encoder_mode='eval in both arms, dropout/stochastic depth disabled; gradients only for named last-stage parameters',
        head_normalization='Same fixed StandardScaler fitted only initial frozen ROI features of actual fit IDs; never refit using held data',
        selection='Inner spatial macro-F1 selects epochs max40/patience8; fresh outer-train refit for fixed selected epochs',
        previous_LR_is_not_matched_control=True,encoder_forward_precision='FP32, TF32disabled',
        classifier='Matched AdamW-trained linear heads; inherited LR classification procedure is replaced',
        cached_initial_features_label_independent=True,encoder_adaptation_uses_actual_fit_IDs_only=True,
        encoder_updates_by_arm={'head_only':False,'last_stage':True},
        exploratory_seed_count=1,automatic_following_experiments=False,
        limitations=['Single seed nested40-group development comparison; not fresh blind test',
            'Same ConvNeXt pretraining and inputs in both arms; comparison is partial adaptation plus its LR setting',
            'Manual boxes provide ROI; real detector-box evaluation remains separate',
            'One predeclared encoder LR; no general optimality claim',
            'Last-stage partial fine-tuning is not a small-rank adapter; trainable parameter count measured in smoke',
            'Checkpoint saves trainable tensors/optimizer and deterministic per-epoch order; no partial-epoch gradient replay'])
    for p in [F.OUT/'features.npz',F.OUT/'feature_receipt.json',F.OUT/'protocol.json',F.OUT/'comparison.json',F.OUT/'independent_verification.json']:
        spec['files_sha256'][str(p)]=C.sha256(p)
    for p in [Path(__file__),ROOT/'data/scripts/start_convnext_stage_adapt_20261022.ps1']:
        spec['code_hashes'][str(p)]=C.sha256(p)
    spec['code_hashes'][str(ROOT/'data/scripts/run_classifier_lora_multiseed.py')]=C.sha256(ROOT/'data/scripts/run_classifier_lora_multiseed.py')
    OUT.mkdir(parents=True);write(OUT/'protocol.json',spec)
    (OUT/'protocol.sha256').write_text(C.sha256(OUT/'protocol.json')+'\n')
    write(OUT/'queue.json',dict(status='prepared_not_started',completed_units=0,total_units=6))
    print('PREPARED_MATCHED_CONVNEXT_PARTIAL_ADAPTATION',flush=True)


def check():
    spec=C.read_json(OUT/'protocol.json');assert C.sha256(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for field in ['files_sha256','code_hashes']:
        for path,digest in spec[field].items():assert C.sha256(path)==digest,path
    assert spec['versions']==M.P.R.package_versions() and spec['arms']==ARMS
    assert N.H.G.spatial_guard(spec['design']['rows'],spec['design']['outer_folds'],C.read_json(N.H.G.MANIFEST))['passed']
    return spec


def load(spec):
    data=N.data_for(spec);images=[]
    for row in data['rows']:
        with F.Image.open(row['crop']) as image:
            assert image.mode=='L' and image.size==(224,224);images.append(np.array(image))
    data['gray']=np.stack(images)
    with np.load(F.OUT/'features.npz') as saved:
        assert saved['source_ids'].tolist()==[str(r['source_id']) for r in data['rows']]
        data['X']=torch.from_numpy(saved['convnext_roi'].copy())
    data['roi7']=torch.from_numpy(np.stack([F.area_weights(r['target_box_image_px'],7) for r in data['rows']]))
    return data


class Classifier(M.nn.Module):
    def __init__(self,spec,arm,fit,data,fold,phase,seed):
        super().__init__();self.encoder=None
        if arm=='last_stage':
            self.encoder=F.model()
            for name,p in self.encoder.named_parameters():
                p.requires_grad_(name.startswith('features.7.') or name.startswith('classifier.0.'))
            assert any(p.requires_grad for p in self.encoder.parameters())
        else:assert arm=='head_only'
        M.seed(seed+fold*101+(10000 if phase=='refit' else 0)+11)
        self.head=M.nn.Linear(768,4)
        scaler=M.StandardScaler().fit(data['X'][fit].numpy())
        self.register_buffer('mean',torch.from_numpy(scaler.mean_.astype(np.float32)))
        self.register_buffer('scale',torch.from_numpy(scaler.scale_.astype(np.float32)))
        self.cuda();self.eval()
    def features(self,data,indices):
        if self.encoder is None:return data['X'][indices].cuda()
        x=torch.from_numpy(data['gray'][indices]).float().unsqueeze(1).repeat(1,3,1,1).cuda()/255.
        x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
        maps=self.encoder.features(x)
        pooled=torch.einsum('bchw,bhw->bc',maps,data['roi7'][indices].cuda())
        return self.encoder.classifier[0](pooled[:,:,None,None]).flatten(1)
    def forward(self,data,indices):return self.head((self.features(data,indices)-self.mean)/self.scale)


def smoke(spec,data):
    split=spec['design']['outer_folds'][0];fit=split['inner_fit_indices']
    ids=np.array(sum(([i for i in fit if int(data['y'][i])==c][:4] for c in range(4)),[]));assert len(ids)==16
    initial_heads=[];receipts={}
    for arm in ARMS:
        torch.cuda.reset_peak_memory_stats()
        model=Classifier(spec,arm,fit,data,0,'select',SEED)
        initial_heads.append(M.tensor_hash(model.head.state_dict()))
        opt=M.optimizer(model,spec);w=M.weights(data,fit)
        frozen_before=None;train_before=None;maxdiff=0.
        if model.encoder is not None:
            frozen_before=M.tensor_hash({k:v for k,v in model.encoder.named_parameters() if not v.requires_grad})
            train_before=M.tensor_hash({k:v for k,v in model.encoder.named_parameters() if v.requires_grad})
            with torch.no_grad():
                actual=model.features(data,ids[:8]).cpu();maxdiff=float((actual-data['X'][ids[:8]]).abs().max())
                assert torch.allclose(actual,data['X'][ids[:8]],atol=2e-4,rtol=2e-4)
            assert all((name.startswith('features.7.') or name.startswith('classifier.0.')) for name,p in model.encoder.named_parameters() if p.requires_grad)
        trained=sum(p.numel() for p in model.parameters() if p.requires_grad)
        start=time.perf_counter();loss1=M.update_group(model,opt,data,ids,w);torch.cuda.synchronize()
        first_seconds=time.perf_counter()-start
        if model.encoder is not None:
            assert frozen_before==M.tensor_hash({k:v for k,v in model.encoder.named_parameters() if not v.requires_grad})
            assert train_before!=M.tensor_hash({k:v for k,v in model.encoder.named_parameters() if v.requires_grad})
        path=OUT/f'smoke_{arm}.pt';M.save_checkpoint(path,model,opt,[],1,1,0.,arm,0,'select',SEED)
        prediction=M.predict(model,data,ids)[1]
        start=time.perf_counter();loss2=M.update_group(model,opt,data,ids,w);torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        expected=M.tensor_hash(M.parameter_state(model));peak=torch.cuda.max_memory_allocated()/2**20
        del model,opt;gc.collect();torch.cuda.empty_cache()
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=Classifier(spec,arm,fit,data,0,'select',SEED);opt=M.optimizer(model,spec)
        M.load_parameters(model,saved['parameters']);opt.load_state_dict(saved['optimizer'])
        assert np.array_equal(prediction,M.predict(model,data,ids)[1])
        resumed_loss=M.update_group(model,opt,data,ids,w)
        assert loss2==resumed_loss and expected==M.tensor_hash(M.parameter_state(model))
        receipts[arm]=dict(trainable_parameters=trained,peak_allocated_mib=peak,first_update_seconds=first_seconds,
            warmed_effective16_update_seconds=seconds,initial_cache_max_abs_diff=maxdiff,losses=[loss1,loss2],
            checkpoint_predictions_and_next_update_bitwise_equal=True,frozen_encoder_parameters_unchanged=True)
        del model,opt,saved;gc.collect();torch.cuda.empty_cache()
    assert len(set(initial_heads))==1
    write(OUT/'readiness.json',dict(status='passed_matched_heads_GPU_updates_frozen_weights_and_resume',arms=receipts,
        identical_initial_heads=True,encoder_eval_in_both=True,protocol_sha256=C.sha256(OUT/'protocol.json')))
    print('PASSED_CONVNEXT_PARTIAL_ADAPTATION_SMOKE',receipts,flush=True)


def scheduled(resume=False):
    spec=check();state=C.read_json(OUT/'queue.json')
    assert state['status'] in (['failed','paused_by_user'] if resume else ['prepared_not_started'])
    independent=M.FENV.process_independence();assert independent['scheduler_owned']
    M.P.R.ensure_gpu_idle();blocked=M.FENV.prevent_network();M.device_setup();torch.set_num_threads(2)
    M.OUT=OUT;M.Classifier=Classifier;M.write=write;M.weights=N.H.weights
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    queue=dict(status='running',pid=os.getpid(),started_utc=M.utc(),phase='smoke',completed_units=0,total_units=6,
        process_independence=independent,offline=True,temporary_system_awake=True,seed=SEED)
    write(OUT/'queue.json',queue)
    try:
        data=load(spec)
        if not resume:smoke(spec,data)
        assert C.read_json(OUT/'readiness.json')['protocol_sha256']==C.sha256(OUT/'protocol.json')
        for split in spec['design']['outer_folds']:
            for phase in ['select','refit']:
                for arm in ARMS:M.stage(spec,data,split,arm,phase,queue,SEED)
            queue['completed_units']+=2;write(OUT/'queue.json',queue)
        results={arm:M._arm_seed_metrics(data,arm,SEED,spec['design']['outer_folds']) for arm in ARMS}
        check();write(OUT/'comparison.json',dict(status='completed_matched_ConvNeXt_partial_adaptation',results=results,
            macro_f1_delta=results['last_stage']['macro_f1']-results['head_only']['macro_f1'],seed=SEED,
            targets=1083,spatial_groups=40,protocol_sha256=C.sha256(OUT/'protocol.json'),final_test_read=False,
            old_frozen_LR_reference_is_different_training=True))
        queue.update(status='completed',phase='complete')
    except BaseException as error:queue.update(status='failed',error=repr(error));raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        queue.update(finished_utc=M.utc(),temporary_system_awake=False,blocked_network_attempts=blocked);write(OUT/'queue.json',queue)


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    elif mode in ['scheduled-run','scheduled-resume']:
        stamp=str(int(time.time()))
        sys.stdout=(OUT/f'training_{stamp}.log').open('a',encoding='utf-8',buffering=1)
        sys.stderr=(OUT/f'training_{stamp}.err.log').open('a',encoding='utf-8',buffering=1)
        scheduled(mode=='scheduled-resume')
    else:raise ValueError(mode)
