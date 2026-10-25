"""Finite, phase-aligned fixed-LoRA head comparison; no encoder fitting."""
from copy import deepcopy
import ctypes
import gc
import math
import os
from pathlib import Path
import shutil
import sys
import time

import run_classifier_head_rebalance_native40 as H
from sklearn.ensemble import RandomForestClassifier
import joblib

M,C,np,torch=H.M,H.C,H.np,H.torch
ROOT=H.ROOT
SOURCE=H.OUT
OUT=ROOT/'runs/experiments/classifier_cached_heads_20261022_v1'
TASK='MarsKG-classifier-cached-heads-20261022-v1'
SEEDS=H.SEEDS
ARMS=['linear_bias','linear_nobias','cosine16','random_forest']
NEW_NEURAL=['linear_nobias','cosine16']
RF_GRID=[dict(max_features=features,min_samples_leaf=leaf) for features in ['sqrt',.25] for leaf in [1,3]]


def write(path,value):H.G.write(path,value)


def prepare():
    if OUT.exists():raise FileExistsError('Preserve existing experiment')
    old=H.check()
    assert C.read_json(SOURCE/'completion_verification.json')['status']=='passed_independent_prediction_and_phase_provenance_verification'
    spec=deepcopy(old)
    for key in ['balance_modes','primary_contrast','previous_joint_LoRA_reference']:
        spec.pop(key,None)
    spec.update(created_utc=M.utc(),arms=ARMS,seeds=SEEDS,spatial_groups=40,
        source_experiment=str(SOURCE),source_protocol_sha256=C.sha256(SOURCE/'protocol.json'),
        primary_contrast='cosine16 versus linear_nobias; RF versus same-representation linear_bias is exploratory',
        cached_baseline='Read-only reuse of verified head_sqrt predictions; not new training',
        normalization='Same train-only StandardScaler for all arms; cosine then L2-normalizes features and weights',
        class_weighting='All arms use square-root inverse frequency from actual fitting IDs; RF sample_weight',
        neural_arms=NEW_NEURAL,cosine_scale=16.,cosine_bias=False,encoder_parameters_trained=0,
        neural_training='Same fixed head LR0.001, AdamW, batch8/accumulate2, max40/patience8, inner selects epochs',
        rf_grid=RF_GRID,rf_trees=300,rf_jobs=2,rf_selection='4 predeclared configurations; inner macro-F1, grid order breaks ties',
        outer_policy='Choose epochs/config only on inner stopping IDs, refit on outer train, outer held scored once',
        labels_modified=False,final_test_read=False,external_validation_images_read=False,
        periodic_trigger=False,automatic_following_experiments=False,
        limitations=['Internal spatial CV already repeatedly used for development, not a fresh blind test',
            'Frozen seed/fold/phase-specific LoRA features; this is not a comparison of joint encoder fine-tuning',
            'RF gets four inner configurations, cosine one fixed scale; report unequal search budgets',
            'FL has15 independent targets; seed spread is not a confidence interval over geography'])
    files=dict(old['files_sha256'])
    files.update({str(SOURCE/name):C.sha256(SOURCE/name) for name in ['protocol.json','comparison.json','completion_verification.json']})
    OUT.mkdir(parents=True)
    for seed in SEEDS:
        for split in spec['design']['outer_folds']:
            fold=split['outer_fold']
            for phase in ['select','refit']:
                for suffix in ['npz','json']:
                    path=SOURCE/'features'/f'seed{seed}'/f'fold{fold}'/f'{phase}.{suffix}'
                    files[str(path)]=C.sha256(path)
            base=SOURCE/'head_sqrt'/f'seed{seed}'/f'fold{fold}'
            for path in [base/'select/completed.json',base/'refit/completed.json',base/'refit/last.pt',base/'refit/outer_predictions.npz']:
                files[str(path)]=C.sha256(path)
            dest=OUT/'linear_bias'/f'seed{seed}'/f'fold{fold}'/'refit'
            dest.mkdir(parents=True)
            shutil.copy2(base/'refit/outer_predictions.npz',dest/'outer_predictions.npz')
            files[str(dest/'outer_predictions.npz')]=C.sha256(dest/'outer_predictions.npz')
            write(dest/'reused.json',dict(source=str(base),source_predictions_sha256=files[str(base/'refit/outer_predictions.npz')],new_training=False))
    spec['files_sha256']=files
    spec['code_hashes']=dict(old['code_hashes'])
    for path in [Path(__file__),ROOT/'data/scripts/start_classifier_cached_heads_20261022.ps1']:
        spec['code_hashes'][str(path)]=C.sha256(path)
    write(OUT/'protocol.json',spec)
    (OUT/'protocol.sha256').write_text(C.sha256(OUT/'protocol.json')+'\n')
    write(OUT/'queue.json',dict(status='prepared_not_started',arms=ARMS,seeds=SEEDS,spatial_groups=40))
    print('PREPARED_PHASE_ALIGNED_CLASSIFIER_HEADS',len(files),flush=True)


def check():
    spec=C.read_json(OUT/'protocol.json')
    assert C.sha256(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for field in ['files_sha256','code_hashes']:
        for path,digest in spec[field].items():assert C.sha256(path)==digest,path
    assert spec['versions']==M.P.R.package_versions()
    assert H.G.spatial_guard(spec['design']['rows'],spec['design']['outer_folds'],C.read_json(H.G.MANIFEST))['passed']
    assert spec['arms']==ARMS and spec['rf_grid']==RF_GRID
    return spec


def data_for(spec):
    rows=spec['design']['rows']
    return dict(rows=rows,y=torch.tensor([r['class_id'] for r in rows]),balance_mode='sqrt')


def features(spec,data,split,seed,phase):
    folder=SOURCE/'features'/f'seed{seed}'/f"fold{split['outer_fold']}"
    payload=folder/f'{phase}.npz';receipt=C.read_json(folder/f'{phase}.json')
    assert receipt['protocol_sha256']==spec['source_protocol_sha256']
    assert receipt['cache_sha256']==C.sha256(payload)==spec['files_sha256'][str(payload)]
    source=spec['source_encoders'][str(seed)][str(split['outer_fold'])][phase]
    assert receipt['source_checkpoint']==source
    assert receipt['source_checkpoint_sha256']==spec['files_sha256'][source]
    fit=split['inner_fit_indices'] if phase=='select' else split['outer_train_indices']
    held=split['inner_stop_indices'] if phase=='select' else split['outer_held_indices']
    expected=np.array(sorted(fit+held),dtype=np.int64)
    with np.load(payload,allow_pickle=False) as cached:
        assert np.array_equal(cached['indices'],expected)
        values=cached['features'].copy()
    assert receipt['ordered_source_ids']==[data['rows'][i]['source_id'] for i in expected]
    assert receipt['fit_indices']==fit and receipt['held_indices']==held
    assert values.shape==(len(expected),768) and np.isfinite(values).all()
    dense=torch.full((len(data['rows']),768),float('nan'))
    dense[expected]=torch.from_numpy(values)
    if phase=='select':assert torch.isnan(dense[split['outer_held_indices']]).all()
    return dense


class CosineHead(M.nn.Module):
    def __init__(self,weight):
        super().__init__();self.weight=M.nn.Parameter(weight.detach().clone())
    def forward(self,value):
        return 16.*M.F.linear(M.F.normalize(value,dim=1),M.F.normalize(self.weight,dim=1))


class CachedClassifier(M.nn.Module):
    def __init__(self,spec,arm,fit,data,fold,phase,run_seed):
        super().__init__();self.encoder=None
        M.seed(run_seed+fold*101+(10000 if phase=='refit' else 0)+11)
        linear=M.nn.Linear(768,4)
        if arm=='linear_bias':self.head=linear
        elif arm=='linear_nobias':
            self.head=M.nn.Linear(768,4,bias=False)
            with torch.no_grad():self.head.weight.copy_(linear.weight)
        elif arm=='cosine16':self.head=CosineHead(linear.weight)
        else:raise ValueError(arm)
        scaler=M.StandardScaler().fit(data['X'][fit].numpy())
        self.register_buffer('mean',torch.from_numpy(scaler.mean_.astype(np.float32)))
        self.register_buffer('scale',torch.from_numpy(scaler.scale_.astype(np.float32)))
        self.to('cuda');self.eval()
    def forward(self,data,indices):
        value=data['X'][indices].to('cuda');assert torch.isfinite(value).all()
        return self.head((value-self.mean)/self.scale)


def bind():
    M.OUT=OUT;M.Classifier=CachedClassifier;M.weights=H.weights;M.write=write


def smoke(spec):
    M.device_setup();data=data_for(spec);parity=[]
    for seed in SEEDS:
        for split in spec['design']['outer_folds']:
            data['X']=features(spec,data,split,seed,'refit')
            fold=split['outer_fold'];held=split['outer_held_indices']
            model=CachedClassifier(spec,'linear_bias',split['outer_train_indices'],data,fold,'refit',seed)
            ck=torch.load(SOURCE/'head_sqrt'/f'seed{seed}'/f'fold{fold}'/'refit/last.pt',map_location='cpu',weights_only=False)
            assert ck['protocol_sha256']==spec['source_protocol_sha256']
            M.load_parameters(model,ck['parameters'])
            pred,logits=M.predict(model,data,held)
            with np.load(OUT/'linear_bias'/f'seed{seed}'/f'fold{fold}'/'refit/outer_predictions.npz') as saved:
                assert np.array_equal(saved['indices'],held)
                assert np.array_equal(saved['predictions'],pred) and np.array_equal(saved['logits'],logits)
            parity.append(dict(seed=seed,fold=fold,predictions_and_logits_bitwise_equal=True))
            del model,ck
    split=spec['design']['outer_folds'][0];fit=split['inner_fit_indices'];seed=SEEDS[0]
    data['X']=features(spec,data,split,seed,'select')
    ids=np.array(sum(([i for i in fit if int(data['y'][i])==c][:4] for c in range(4)),[]))
    assert len(ids)==16
    initial_weights=[]
    for arm in NEW_NEURAL:
        model=CachedClassifier(spec,arm,fit,data,0,'select',seed)
        initial_weights.append(M.tensor_hash({'weight':model.head.weight}))
        assert sum(p.numel() for p in model.parameters())==3072
        if arm=='cosine16':
            with torch.no_grad():
                z=(data['X'][ids].cuda()-model.mean)/model.scale
                reference=16.*(z/z.norm(dim=1,keepdim=True))@(model.head.weight/model.head.weight.norm(dim=1,keepdim=True)).T
                assert torch.allclose(model(data,ids),reference,atol=2e-6,rtol=2e-6)
        opt=M.optimizer(model,spec);w=M.weights(data,fit)
        M.update_group(model,opt,data,ids,w)
        path=OUT/f'smoke_{arm}.pt'
        M.save_checkpoint(path,model,opt,[],1,1,0.,arm,0,'select',seed)
        prediction=M.predict(model,data,ids)[1]
        loss=M.update_group(model,opt,data,ids,w);expected=M.tensor_hash(M.parameter_state(model))
        del model,opt
        ck=torch.load(path,map_location='cpu',weights_only=False)
        model=CachedClassifier(spec,arm,fit,data,0,'select',seed);opt=M.optimizer(model,spec)
        M.load_parameters(model,ck['parameters']);opt.load_state_dict(ck['optimizer'])
        assert np.array_equal(prediction,M.predict(model,data,ids)[1])
        assert M.update_group(model,opt,data,ids,w)==loss
        assert M.tensor_hash(M.parameter_state(model))==expected
        del model,opt,ck
    assert len(set(initial_weights))==1
    forest=RandomForestClassifier(n_estimators=16,max_features='sqrt',min_samples_leaf=1,random_state=seed,n_jobs=2)
    x=data['X'][ids].numpy();y=data['y'][ids].numpy();forest.fit(x,y)
    joblib.dump(forest,OUT/'smoke_forest.joblib')
    assert np.array_equal(forest.predict_proba(x),joblib.load(OUT/'smoke_forest.joblib').predict_proba(x))
    write(OUT/'readiness.json',dict(status='passed',protocol_sha256=C.sha256(OUT/'protocol.json'),
        reused_baseline_parity=parity,neural_heads_same_initial_weights=True,encoder_updates=0,
        actual_GPU_updates_and_bitwise_resume=True,cosine_formula_checked=True,RF_reload_checked=True,
        outer_held_excluded_from_selection_features=True))
    gc.collect();torch.cuda.empty_cache()
    print('PASSED_CACHED_HEAD_SMOKE_AND_9_BASELINE_PARITIES',flush=True)


def rf_stage(spec,data,split,seed,phase,queue):
    fold=split['outer_fold'];folder=OUT/'random_forest'/f'seed{seed}'/f'fold{fold}'/phase
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completed.json').exists():return
    fit=split['inner_fit_indices'] if phase=='select' else split['outer_train_indices']
    held=split['inner_stop_indices'] if phase=='select' else split['outer_held_indices']
    scaler=M.StandardScaler().fit(data['X'][fit].numpy())
    x=scaler.transform(data['X'][fit].numpy());y=data['y'][fit].numpy()
    target=scaler.transform(data['X'][held].numpy())
    cls=M.classifier_class_weights(y,'sqrt');sw=np.array([cls[int(c)] for c in y])
    choices=list(enumerate(spec['rf_grid'])) if phase=='select' else [(C.read_json(folder.parent/'select/completed.json')['selected_index'],None)]
    best=-1.;selected=None;scores=[];chosen_model=None
    for index,cfg in choices:
        if cfg is None:cfg=spec['rf_grid'][index]
        queue.update(arm='random_forest',run_seed=seed,outer_fold=fold,phase=phase,config_index=index)
        write(OUT/'queue.json',queue)
        model=RandomForestClassifier(n_estimators=spec['rf_trees'],n_jobs=2,random_state=seed+fold*101+(10000 if phase=='refit' else 0)+11,**cfg)
        model.fit(x,y,sample_weight=sw)
        assert np.array_equal(model.classes_,np.arange(4))
        if phase=='select':
            score=float(M.f1_score(data['y'][held],model.predict(target),labels=range(4),average='macro',zero_division=0))
            scores.append(dict(index=index,config=cfg,inner_macro_f1=score))
            if score>best+1e-12:best=score;selected=index;chosen_model=model
        else:chosen_model=model;selected=index
        print('RF',seed,fold,phase,index,'done',flush=True)
    path=folder/'model.joblib';joblib.dump(dict(model=chosen_model,scaler=scaler),path,compress=3)
    restored=joblib.load(path)
    assert np.array_equal(chosen_model.predict_proba(target),restored['model'].predict_proba(target))
    if phase=='refit':
        probabilities=chosen_model.predict_proba(target)
        np.savez_compressed(folder/'outer_predictions.npz',indices=np.array(held),predictions=chosen_model.classes_[probabilities.argmax(1)],probabilities=probabilities)
    write(folder/'completed.json',dict(status='completed',selected_index=selected,scores=scores,
        fit_unique_IDs=len(fit),held_IDs=len(held),model_sha256=C.sha256(path),train_only_class_weights=cls,
        outer_read_only_after_refit=phase=='refit',source_protocol_sha256=spec['source_protocol_sha256'],protocol_sha256=C.sha256(OUT/'protocol.json')))
    del chosen_model,model;gc.collect()


def summarize(spec,data):
    per_seed={a:{str(s):M._arm_seed_metrics(data,a,s,spec['design']['outer_folds']) for s in SEEDS} for a in ARMS}
    old=C.read_json(SOURCE/'comparison.json')['per_seed']['head_sqrt']
    for seed in SEEDS:assert per_seed['linear_bias'][str(seed)]['predictions']==old[str(seed)]['predictions']
    aggregate={}
    for arm,values in per_seed.items():
        rows=list(values.values())
        aggregate[arm]=dict(macro_f1_mean=float(np.mean([v['macro_f1'] for v in rows])),macro_f1_std=float(np.std([v['macro_f1'] for v in rows])),
            accuracy_mean=float(np.mean([v['accuracy'] for v in rows])),
            per_class={n:{k:float(np.mean([v['per_class'][n][k] for v in rows])) for k in ['precision','recall','f1-score','support']} for n in M.NAMES})
    contrasts={}
    for a,b in [('linear_nobias','linear_bias'),('cosine16','linear_nobias'),('cosine16','linear_bias'),('random_forest','linear_bias')]:
        delta=[per_seed[a][str(s)]['macro_f1']-per_seed[b][str(s)]['macro_f1'] for s in SEEDS]
        contrasts[a+'_minus_'+b]=dict(per_seed=delta,mean=float(np.mean(delta)),std=float(np.std(delta)),wins=sum(v>0 for v in delta))
    write(OUT/'comparison.json',dict(status='completed_cached_head_comparison',targets=1083,spatial_groups=40,seeds=SEEDS,
        aggregate=aggregate,per_seed=per_seed,paired=contrasts,protocol_sha256=C.sha256(OUT/'protocol.json'),
        final_test_read=False,encoder_updates=0,not_comparable_as_one_factor_to_joint_LoRA=True))
    write(OUT/'completion_verification.json',dict(status='passed_saved_OOF_recomputation',all_4_arms_all_3_seeds_complete=True,
        linear_reference_exactly_reproduced=True,protocol_sha256=C.sha256(OUT/'protocol.json')))


def scheduled(resume=False):
    spec=check();bind()
    state=C.read_json(OUT/'queue.json')
    assert state['status'] in (['failed','paused_by_user'] if resume else ['prepared_not_started'])
    independent=M.FENV.process_independence();assert independent['scheduler_owned']
    M.P.R.ensure_gpu_idle();blocked=M.FENV.prevent_network();M.device_setup()
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    queue=dict(status='running',pid=os.getpid(),started_utc=M.utc(),process_independence=independent,
        offline=True,temporary_system_awake=True,arms=ARMS,seeds=SEEDS,completed_units=0,total_new_units=27)
    write(OUT/'queue.json',queue)
    try:
        if not resume:smoke(spec)
        assert C.read_json(OUT/'readiness.json')['protocol_sha256']==C.sha256(OUT/'protocol.json')
        data=data_for(spec)
        for seed in SEEDS:
            for split in spec['design']['outer_folds']:
                for phase in ['select','refit']:
                    data['X']=features(spec,data,split,seed,phase)
                    for arm in NEW_NEURAL:M.stage(spec,data,split,arm,phase,queue,seed)
                    rf_stage(spec,data,split,seed,phase,queue)
                queue['completed_units']+=3;write(OUT/'queue.json',queue)
        check();summarize(spec,data);queue.update(status='completed',phase='complete')
    except BaseException as error:
        queue.update(status='failed',error=repr(error));raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        queue.update(finished_utc=M.utc(),temporary_system_awake=False,blocked_network_attempts=blocked)
        write(OUT/'queue.json',queue)


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    elif mode in ['scheduled-run','scheduled-resume']:
        stamp=str(int(time.time()))
        sys.stdout=(OUT/f'training_{stamp}.log').open('a',encoding='utf-8',buffering=1)
        sys.stderr=(OUT/f'training_{stamp}.err.log').open('a',encoding='utf-8',buffering=1)
        scheduled(resume=mode=='scheduled-resume')
    else:raise ValueError(mode)
