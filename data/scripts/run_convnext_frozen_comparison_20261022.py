"""Fixed native224 ConvNeXt/DINOv2 frozen representations and nested LR."""
from copy import deepcopy
import ctypes
import gc
import os
from pathlib import Path
import sys
import time

import run_classifier_geometry_terrain_20261022 as L
from PIL import Image
from torchvision.models import convnext_tiny
import torchvision
from threadpoolctl import threadpool_limits
import joblib

N,M,C,np,torch=L.N,L.N.M,L.C,L.np,L.N.torch
ROOT=L.ROOT
OUT=ROOT/'runs/experiments/convnext_frozen_comparison_20261022_v1'
WEIGHT=ROOT/'weights/convnext_tiny_imagenet1k_v1/convnext_tiny-983f1562.pth'
DINO=ROOT/'runs/experiments/roi_context_pooling_20261015_v1'
ARMS=['dino_roi','dino_global','convnext_roi','convnext_global']
SEED=20261015

def write(path,value):N.write(path,value)


def area_weights(box,side):
    x0,y0,x1,y1=box;assert 0<=x0<x1<=224 and 0<=y0<y1<=224
    edge=np.linspace(0,224,side+1)
    xx=np.maximum(0,np.minimum(edge[1:],x1)-np.maximum(edge[:-1],x0))
    yy=np.maximum(0,np.minimum(edge[1:],y1)-np.maximum(edge[:-1],y0))
    areas=yy[:,None]*xx[None,:]
    assert abs(areas.sum()-(x1-x0)*(y1-y0))<1e-8
    return (areas/areas.sum()).astype(np.float32)


def prepare():
    if OUT.exists():raise FileExistsError('Preserve existing experiment')
    parent=N.check();receipt=C.read_json(WEIGHT.parent/'source_receipt.json')
    assert C.sha256(WEIGHT)==receipt['sha256'] and receipt['sha256'].startswith('983f1562')
    features=DINO/'pooled_features.npz';weights=DINO/'pooling_weights.npz'
    assert C.sha256(features)==parent['files_sha256'][str(features)]
    with np.load(weights) as saved:
        lookup={str(s):i for i,s in enumerate(saved['source_id'])}
        for row in parent['design']['rows']:
            assert C.sha256(row['crop'])==row['crop_sha256']
            assert np.allclose(area_weights(row['target_box_image_px'],16).ravel(),saved['roi'][lookup[str(row['source_id'])]],atol=1e-7)
    files=[features,weights,DINO/'protocol.json',WEIGHT,WEIGHT.parent/'source_receipt.json',N.OUT/'protocol.json']
    pins={str(p):C.sha256(p) for p in files}
    pins.update({r['crop']:r['crop_sha256'] for r in parent['design']['rows']})
    dino_spec=C.read_json(DINO/'protocol.json')
    for name,digest in dino_spec['checkpoint_files_sha256'].items():
        p=Path(dino_spec['checkpoint'])/name;assert C.sha256(p)==digest;pins[str(p)]=digest
    code=dict(parent['code_hashes'])
    for p in [Path(__file__),ROOT/'data/scripts/start_convnext_frozen_comparison_20261022.ps1',Path(L.__file__),Path(torchvision.models.convnext.__file__)]:code[str(p)]=C.sha256(p)
    spec=dict(created_utc=M.utc(),design=deepcopy(parent['design']),arms=ARMS,seed=SEED,targets=1083,spatial_groups=40,
        C_grid=L.C_GRID,max_iter=3000,tolerance=1e-6,versions=M.P.R.package_versions(),files_sha256=pins,code_hashes=code,
        preprocessing='Original full224 grayscale replicatedRGB /255, ImageNet mean/std; no resizing, cropping or masking',
        preprocessing_difference='Official ConvNeXt resize236+centre224 is deliberately not used to preserve same full target extent',
        representation='ConvNeXt-Tiny ImageNet1K_V1; DINOv2-base cached frozen features; both output768 dimensions',
        ROI='Normalized nominal grid-cell overlap with identical manual AABB; ConvNeXt7x7 stride32, DINO16x16 stride14',
        normalization='ConvNeXt pretrained classifier LayerNorm after pooled vector; DINO token final norm precedes pooling',
        primary='convnext_roi versus dino_roi; global pooling arms predeclared diagnostics',
        classifier='Same weighted multinomial LR, train-only StandardScaler, inner5C grid, outer refit and one score',
        class_weights='Actual fit square-root inverse-frequency mean-one sample weights',
        inference_batch=8,precision='FP32, TF32disabled, encoder eval/frozen',encoder_updates=0,
        frozen_features_label_independent=True,repeat_seed_runs=False,
        limitations=['Different supervised ImageNet versus self-supervised DINO pretraining; not a pure architecture ablation',
            'Both full-image receptive fields contextualize ROI features; ROI pooling does not erase background',
            'Frozen deterministic models: one nested3fold evaluation, no fabricated3-seed repetition',
            'Repeated internal spatial development cohort, not fresh blind-test or real detector-box evaluation',
            'Results do not rule out fine-tuning or other CNNs'],
        labels_modified=False,new_CTX_downloads=0,final_test_read=False,automatic_following_experiments=False)
    OUT.mkdir(parents=True);write(OUT/'protocol.json',spec)
    (OUT/'protocol.sha256').write_text(C.sha256(OUT/'protocol.json')+'\n')
    write(OUT/'queue.json',dict(status='prepared_not_started',completed_units=0,total_units=12))
    print('PREPARED_4_FROZEN_REPRESENTATION_ARMS',flush=True)


def check():
    spec=C.read_json(OUT/'protocol.json');assert C.sha256(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for field in ['files_sha256','code_hashes']:
        for path,digest in spec[field].items():assert C.sha256(path)==digest,path
    assert spec['versions']==M.P.R.package_versions() and spec['arms']==ARMS
    assert N.H.G.spatial_guard(spec['design']['rows'],spec['design']['outer_folds'],C.read_json(N.H.G.MANIFEST))['passed']
    return spec


def model():
    net=convnext_tiny(weights=None)
    state=torch.load(WEIGHT,map_location='cpu',weights_only=True)
    net.load_state_dict(state,strict=True)
    net.requires_grad_(False);net.eval();return net.cuda()


def pixels(rows):
    gray=[]
    for row in rows:
        with Image.open(row['crop']) as im:
            assert im.mode=='L' and im.size==(224,224);gray.append(np.array(im))
    x=torch.from_numpy(np.stack(gray)).float().unsqueeze(1).repeat(1,3,1,1).cuda()/255.
    return (x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]


def extract(spec,queue):
    rows=spec['design']['rows'];net=model();before=M.tensor_hash(net.state_dict())
    outputs={a:[] for a in ['convnext_roi','convnext_global']};chunks=OUT/'feature_chunks';chunks.mkdir(exist_ok=True)
    def forward(net,batch):
        with torch.inference_mode():
            maps=net.features(pixels(batch));assert maps.shape==(len(batch),768,7,7)
            w=torch.from_numpy(np.stack([area_weights(r['target_box_image_px'],7) for r in batch])).cuda()
            roi=torch.einsum('bchw,bhw->bc',maps,w)
            glob=maps.mean(dim=(2,3))
            return {key:net.classifier[0](value[:,:,None,None]).flatten(1).cpu().numpy().copy() for key,value in [('convnext_roi',roi),('convnext_global',glob)]}
    first=forward(net,rows[:8]);second=forward(net,rows[:8])
    restored=model();third=forward(restored,rows[:8]);del restored;torch.cuda.empty_cache()
    assert all(np.array_equal(first[k],second[k]) and np.array_equal(first[k],third[k]) for k in first)
    assert not any(p.requires_grad for p in net.parameters()) and not net.training
    write(OUT/'readiness.json',dict(status='passed_frozen_GPU_feature_and_reload_checks',
        encoder_updates=0,same_batch_predictions_bitwise_equal=True,reloaded_weight_predictions_bitwise_equal=True,
        official_weight_strict_load=True,actual_shape=[8,768,7,7],protocol_sha256=C.sha256(OUT/'protocol.json')))
    for start in range(0,len(rows),8):
        stop=min(start+8,len(rows));path=chunks/f'{start:04d}.npz';receipt=path.with_suffix('.json')
        if receipt.exists():
            saved=C.read_json(receipt);assert saved['protocol_sha256']==C.sha256(OUT/'protocol.json') and saved['sha256']==C.sha256(path)
            with np.load(path) as data:
                assert data['source_ids'].tolist()==[str(r['source_id']) for r in rows[start:stop]]
                features={k:data[k].copy() for k in outputs}
        else:
            features=forward(net,rows[start:stop])
            np.savez_compressed(path,source_ids=np.array([str(r['source_id']) for r in rows[start:stop]]),**features)
            write(receipt,dict(sha256=C.sha256(path),protocol_sha256=C.sha256(OUT/'protocol.json')))
        for k,v in features.items():assert v.shape==(stop-start,768) and np.isfinite(v).all();outputs[k].append(v)
        queue.update(phase='extract_frozen_convnext',images_completed=stop,images_total=len(rows));write(OUT/'queue.json',queue)
    assert M.tensor_hash(net.state_dict())==before
    del net;gc.collect();torch.cuda.empty_cache()
    outputs={k:np.concatenate(v) for k,v in outputs.items()}
    with np.load(DINO/'pooled_features.npz') as data:
        lookup={str(s):i for i,s in enumerate(data['source_id'])};ix=[lookup[str(r['source_id'])] for r in rows]
        assert np.array_equal(data['y'][ix],[r['class_id'] for r in rows])
        outputs['dino_roi']=data['roi_only'][ix].copy();outputs['dino_global']=data['full_patch_mean'][ix].copy()
    np.savez_compressed(OUT/'features.npz',source_ids=np.array([str(r['source_id']) for r in rows]),**outputs)
    write(OUT/'feature_receipt.json',dict(sha256=C.sha256(OUT/'features.npz'),encoder_state_unchanged=True,
        full_model_state_sha256=before,protocol_sha256=C.sha256(OUT/'protocol.json')))
    return outputs


def summary(spec,data):
    results={};y=data['y'].numpy()
    for arm in ARMS:
        oof=np.full(len(y),-1,dtype=int);folds=[]
        for split in spec['design']['outer_folds']:
            folder=OUT/arm/f'seed{SEED}'/f"fold{split['outer_fold']}"/'refit'
            with np.load(folder/'outer_predictions.npz') as saved:
                ix=saved['indices'];assert np.array_equal(ix,split['outer_held_indices']) and np.all(oof[ix]==-1)
                oof[ix]=saved['predictions'];folds.append(dict(fold=split['outer_fold'],macro_f1=float(L.f1_score(y[ix],oof[ix],labels=range(4),average='macro',zero_division=0))))
        assert np.all(oof>=0)
        results[arm]=dict(macro_f1=float(L.f1_score(y,oof,labels=range(4),average='macro',zero_division=0)),accuracy=float(np.mean(y==oof)),
            per_class=L.classification_report(y,oof,labels=range(4),target_names=M.NAMES,output_dict=True,zero_division=0),
            confusion_matrix=L.confusion_matrix(y,oof,labels=range(4)).tolist(),predictions=oof.tolist(),folds=folds)
    write(OUT/'comparison.json',dict(status='completed_frozen_CNN_Transformer_comparison',results=results,
        primary_macro_delta=results['convnext_roi']['macro_f1']-results['dino_roi']['macro_f1'],targets=1083,spatial_groups=40,
        protocol_sha256=C.sha256(OUT/'protocol.json'),encoder_updates=0,final_test_read=False,fine_tuning_compared=False))


def scheduled(resume=False):
    spec=check();state=C.read_json(OUT/'queue.json')
    assert state['status'] in (['failed','paused_by_user'] if resume else ['prepared_not_started'])
    independent=M.FENV.process_independence();assert independent['scheduler_owned']
    M.P.R.ensure_gpu_idle();blocked=M.FENV.prevent_network();M.device_setup();torch.set_num_threads(2)
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    queue=dict(status='running',started_utc=M.utc(),pid=os.getpid(),process_independence=independent,
        offline=True,temporary_system_awake=True,completed_units=0,total_units=12)
    write(OUT/'queue.json',queue)
    try:
        data=N.data_for(spec);data.update(extract(spec,queue))
        L.OUT=OUT;L.ARMS={a:[a] for a in ARMS}
        with threadpool_limits(limits=2):
            for split in spec['design']['outer_folds']:
                for phase in ['select','refit']:
                    for arm in ARMS:L.unit(spec,data,split,SEED,arm,phase,queue)
                queue['completed_units']+=4;write(OUT/'queue.json',queue)
        check();summary(spec,data);queue.update(status='completed',phase='complete')
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
