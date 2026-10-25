"""Nested spatial fixed-LoRA/geometry/native-terrain comparison, CPU only."""
from copy import deepcopy
import ctypes
import gc
import os
from pathlib import Path
import sys
import time
import warnings

import run_classifier_cached_heads_20261022 as N
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score,classification_report,confusion_matrix
from threadpoolctl import threadpool_limits
import joblib

C,np=N.C,N.np
ROOT=N.ROOT
OUT=ROOT/'runs/experiments/classifier_geometry_terrain_20261022_v1'
AUX=ROOT/'runs/research/geometry_terrain_features_20261022_v1'
TASK='MarsKG-classifier-geometry-terrain-20261022-v1'
ARMS={'image':['image'],'geometry':['geometry'],'geometry_terrain':['geometry','terrain'],
      'image_geometry':['image','geometry'],'image_geometry_terrain':['image','geometry','terrain']}
C_GRID=[.001,.01,.1,1.,10.]
SEEDS=N.SEEDS


def write(path,value):N.write(path,value)


def prepare():
    if OUT.exists():raise FileExistsError('Preserve existing experiment')
    parent=N.check();aux=C.read_json(AUX/'features.json')
    assert C.read_json(N.OUT/'queue.json')['status']=='completed'
    assert C.sha256(aux['feature_file'])==aux['feature_sha256']
    for path,digest in aux['input_sha256'].items():assert C.sha256(path)==digest,path
    assert C.sha256(aux['DEM']['path'])==aux['DEM']['sha256']
    spec={key:deepcopy(parent[key]) for key in ['design','files_sha256','code_hashes','source_encoders','source_protocol_sha256','versions']}
    for path in [AUX/'features.json',AUX/'features.npz',N.OUT/'protocol.json',N.OUT/'comparison.json',ROOT/'runs/research/classifier_cached_heads_verification_20261022_v1/verification.json']:
        spec['files_sha256'][str(path)]=C.sha256(path)
    spec['files_sha256'][aux['DEM']['path']]=aux['DEM']['sha256']
    for path in [Path(__file__),ROOT/'data/scripts/start_classifier_geometry_terrain_20261022.ps1']:
        spec['code_hashes'][str(path)]=C.sha256(path)
    spec.update(created_utc=N.M.utc(),arms=ARMS,seeds=SEEDS,spatial_groups=40,targets=1083,
        C_grid=C_GRID,solver='lbfgs',max_iter=3000,tolerance=1e-6,class_weight='train-only sqrt inverse frequency sample weights',
        normalization='StandardScaler fit only on actual fitting IDs, separately for selection and refitting',
        feature_names=aux['feature_names'],representation='Fixed phase-specific LoRA from existing cached features',
        primary_contrasts=['image_geometry minus image','image_geometry_terrain minus image_geometry'],
        classifier_scope='All five arms use the same fully fitted multinomial logistic regression and C grid',
        selection='Inner spatial macro-F1; grid order breaks ties; outer score only after refit',
        image_baseline_changed_from_previous='New converged/tuned LR reference, not the old AdamW-trained linear head',
        geometry_seed_policy='Geometry-only LR is deterministic: its 3 repeated seed results are not independent evidence',
        final_test_read=False,external_validation_images_read=False,labels_modified=False,new_downloads=0,
        GPU_used=False,periodic_trigger=False,automatic_following_experiments=False,
        limitations=aux['limitations']+['Repeated internal40-group CV is development evidence, not a fresh blind test',
            'Shared MOLA pixels or overlapping AABBs across groups are checked before training',
            'All inputs are manual AABBs; real detector-box evaluation remains separate'])
    OUT.mkdir(parents=True)
    # Native DEM pixels must not bridge the spatial train/held boundary.
    aux_rows={r['source_id']:r for r in aux['rows']}
    support=[aux_rows[str(row['source_id'])]['native_DEM_window'] for row in spec['design']['rows']]
    def intersect(a,b):return min(a[2]+1,b[2]+1)>max(a[0]-1,b[0]-1) and min(a[3]+1,b[3]+1)>max(a[1]-1,b[1]-1)
    conflicts=[]
    for split in spec['design']['outer_folds']:
        for phase,fit,held in [('select',split['inner_fit_indices'],split['inner_stop_indices']),('refit',split['outer_train_indices'],split['outer_held_indices'])]:
            for i in fit:
                for j in held:
                    if intersect(support[i],support[j]):conflicts.append(dict(fold=split['outer_fold'],phase=phase,fit_id=spec['design']['rows'][i]['source_id'],held_id=spec['design']['rows'][j]['source_id']))
    write(OUT/'DEM_spatial_guard.json',dict(passed=not conflicts,conflicts=conflicts,includes_one_native_pixel_halo=True))
    assert not conflicts,'Shared native DEM support crosses existing spatial folds; redesign grouping before training'
    spec['files_sha256'][str(OUT/'DEM_spatial_guard.json')]=C.sha256(OUT/'DEM_spatial_guard.json')
    write(OUT/'protocol.json',spec)
    (OUT/'protocol.sha256').write_text(C.sha256(OUT/'protocol.json')+'\n')
    write(OUT/'queue.json',dict(status='prepared_not_started',total_units=45,completed_units=0,arms=list(ARMS),seeds=SEEDS))
    print('PREPARED_5_MATCHED_LR_ARMS_NATIVE_DEM_SUPPORT_ISOLATED',flush=True)


def check():
    spec=C.read_json(OUT/'protocol.json')
    assert C.sha256(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for field in ['files_sha256','code_hashes']:
        for path,digest in spec[field].items():assert C.sha256(path)==digest,path
    assert spec['versions']==N.M.P.R.package_versions()
    assert N.H.G.spatial_guard(spec['design']['rows'],spec['design']['outer_folds'],C.read_json(N.H.G.MANIFEST))['passed']
    assert spec['arms']==ARMS and spec['C_grid']==C_GRID
    return spec


def load(spec):
    data=N.data_for(spec)
    with np.load(AUX/'features.npz',allow_pickle=False) as saved:
        assert saved['source_ids'].tolist()==[str(r['source_id']) for r in data['rows']]
        data['geometry']=saved['geometry'].copy();data['terrain']=saved['terrain'].copy()
    return data


def matrix(data,arm,indices):
    blocks=[data[name][indices] for name in ARMS[arm]]
    x=np.ascontiguousarray(np.concatenate(blocks,axis=1),dtype=np.float64)
    assert np.isfinite(x).all()
    return x


def fit_model(spec,x,y,c_value,seed):
    weights=N.M.classifier_class_weights(y,'sqrt')
    sw=np.array([weights[int(v)] for v in y])
    scaler=StandardScaler().fit(x)
    model=LogisticRegression(C=c_value,solver='lbfgs',max_iter=spec['max_iter'],tol=spec['tolerance'],random_state=seed)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always',ConvergenceWarning)
        model.fit(scaler.transform(x),y,sample_weight=sw)
    assert not any(issubclass(w.category,ConvergenceWarning) for w in records),'LR did not converge; preserve failure, do not score'
    assert np.array_equal(model.classes_,np.arange(4))
    return dict(model=model,scaler=scaler),weights


def probabilities(model,x):return model['model'].predict_proba(model['scaler'].transform(x))


def smoke(spec):
    data=load(spec);split=spec['design']['outer_folds'][0]
    data['image']=N.features(spec,data,split,SEEDS[0],'select').numpy()
    fit=split['inner_fit_indices']
    ix=np.array(sum(([i for i in fit if int(data['y'][i])==c][:4] for c in range(4)),[]))
    assert len(ix)==16 and np.isnan(data['image'][split['outer_held_indices']]).all()
    for arm in ARMS:
        x=matrix(data,arm,ix);fitted,_=fit_model(spec,x,data['y'][ix].numpy(),.1,SEEDS[0])
        path=OUT/f'smoke_{arm}.joblib';joblib.dump(fitted,path)
        assert np.array_equal(probabilities(fitted,x),probabilities(joblib.load(path),x))
        assert np.allclose(fitted['scaler'].mean_,x.mean(axis=0),atol=1e-12)
    write(OUT/'readiness.json',dict(status='passed_CPU_smoke_all5arms',actual_fits_and_reloads=True,
        inner_image_cache_excludes_outer_held=True,train_only_standardization=True,converged=True,
        native_DEM_support_guard_passed=True,protocol_sha256=C.sha256(OUT/'protocol.json')))


def unit(spec,data,split,seed,arm,phase,queue):
    folder=OUT/arm/f'seed{seed}'/f"fold{split['outer_fold']}"/phase;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completed.json').exists():return
    fit=split['inner_fit_indices'] if phase=='select' else split['outer_train_indices']
    held=split['inner_stop_indices'] if phase=='select' else split['outer_held_indices']
    x=matrix(data,arm,fit);target=matrix(data,arm,held);y=data['y'][fit].numpy()
    configs=list(enumerate(C_GRID)) if phase=='select' else [(C.read_json(folder.parent/'select/completed.json')['selected_index'],None)]
    records=[];best=-1.;selected=None;chosen=None
    for index,c_value in configs:
        if c_value is None:c_value=C_GRID[index]
        queue.update(arm=arm,run_seed=seed,outer_fold=split['outer_fold'],phase=phase,C=c_value)
        write(OUT/'queue.json',queue)
        model,weights=fit_model(spec,x,y,c_value,seed)
        if phase=='select':
            probs=probabilities(model,target);pred=probs.argmax(1)
            score=float(f1_score(data['y'][held],pred,labels=range(4),average='macro',zero_division=0))
            np.savez_compressed(folder/f'candidate_{index}.npz',indices=np.array(held),probabilities=probs,predictions=pred)
            records.append(dict(index=index,C=c_value,inner_macro_f1=score,iterations=model['model'].n_iter_.tolist()))
            if score>best+1e-12:best=score;selected=index;chosen=model
        else:selected=index;chosen=model
        print(seed,arm,split['outer_fold'],phase,'C',c_value,'done',flush=True)
    path=folder/'model.joblib';joblib.dump(chosen,path,compress=3)
    assert np.array_equal(probabilities(chosen,target),probabilities(joblib.load(path),target))
    if phase=='refit':
        probs=probabilities(chosen,target)
        np.savez_compressed(folder/'outer_predictions.npz',indices=np.array(held),probabilities=probs,predictions=probs.argmax(1))
    write(folder/'completed.json',dict(status='completed',selected_index=selected,selected_C=C_GRID[selected],candidates=records,
        fit_indices=fit,held_indices=held,dimensions=x.shape[1],model_sha256=C.sha256(path),train_only_class_weights=weights,
        converged=True,outer_read_only_after_refit=phase=='refit',protocol_sha256=C.sha256(OUT/'protocol.json')))
    del chosen,model;gc.collect()


def summarize(spec,data):
    y=data['y'].numpy();per_seed={};aggregate={}
    for arm in ARMS:
        per_seed[arm]={}
        for seed in SEEDS:
            oof=np.full(len(y),-1,dtype=int);folds=[]
            for split in spec['design']['outer_folds']:
                folder=OUT/arm/f'seed{seed}'/f"fold{split['outer_fold']}"/'refit'
                with np.load(folder/'outer_predictions.npz',allow_pickle=False) as saved:
                    ix=saved['indices'];assert np.array_equal(ix,split['outer_held_indices']) and np.all(oof[ix]==-1)
                    oof[ix]=saved['predictions']
                    folds.append(dict(fold=split['outer_fold'],macro_f1=float(f1_score(y[ix],oof[ix],labels=range(4),average='macro',zero_division=0))))
            assert np.all(oof>=0)
            per_seed[arm][str(seed)]=dict(macro_f1=float(f1_score(y,oof,labels=range(4),average='macro',zero_division=0)),
                accuracy=float(np.mean(y==oof)),per_class=classification_report(y,oof,labels=range(4),target_names=N.M.NAMES,output_dict=True,zero_division=0),
                confusion_matrix=confusion_matrix(y,oof,labels=range(4)).tolist(),predictions=oof.tolist(),folds=folds)
        results=list(per_seed[arm].values())
        aggregate[arm]=dict(macro_f1_mean=float(np.mean([r['macro_f1'] for r in results])),macro_f1_std=float(np.std([r['macro_f1'] for r in results])),
            accuracy_mean=float(np.mean([r['accuracy'] for r in results])),per_class={name:{key:float(np.mean([r['per_class'][name][key] for r in results])) for key in ['precision','recall','f1-score','support']} for name in N.M.NAMES})
    for arm in ['geometry','geometry_terrain']:
        assert all(per_seed[arm][str(s)]['predictions']==per_seed[arm][str(SEEDS[0])]['predictions'] for s in SEEDS)
    paired={}
    for a,b in [('image_geometry','image'),('image_geometry_terrain','image_geometry'),('geometry_terrain','geometry')]:
        delta=[per_seed[a][str(s)]['macro_f1']-per_seed[b][str(s)]['macro_f1'] for s in SEEDS]
        paired[a+'_minus_'+b]=dict(per_seed=delta,mean=float(np.mean(delta)),std=float(np.std(delta)),wins=sum(d>0 for d in delta))
    write(OUT/'comparison.json',dict(status='completed_5arm_geometry_terrain_nested_comparison',targets=1083,spatial_groups=40,
        seeds=SEEDS,aggregate=aggregate,per_seed=per_seed,paired=paired,protocol_sha256=C.sha256(OUT/'protocol.json'),
        final_test_read=False,GPU_used=False,geometry_only_seed_repeats_not_independent=True,manual_box_evaluation_only=True))


def scheduled(resume=False):
    spec=check();state=C.read_json(OUT/'queue.json')
    assert state['status'] in (['failed','paused_by_user'] if resume else ['prepared_not_started'])
    independent=N.M.FENV.process_independence();assert independent['scheduler_owned']
    blocked=N.M.FENV.prevent_network()
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    queue=dict(status='running',pid=os.getpid(),started_utc=N.M.utc(),process_independence=independent,
        offline=True,temporary_system_awake=True,GPU_used=False,completed_units=0,total_units=45)
    write(OUT/'queue.json',queue)
    try:
        with threadpool_limits(limits=2):
            if not resume:smoke(spec)
            assert C.read_json(OUT/'readiness.json')['protocol_sha256']==C.sha256(OUT/'protocol.json')
            data=load(spec)
            for seed in SEEDS:
                for split in spec['design']['outer_folds']:
                    for phase in ['select','refit']:
                        data['image']=N.features(spec,data,split,seed,phase).numpy()
                        for arm in ARMS:unit(spec,data,split,seed,arm,phase,queue)
                    queue['completed_units']+=5;write(OUT/'queue.json',queue)
            check();summarize(spec,data)
        queue.update(status='completed',phase='complete')
    except BaseException as error:queue.update(status='failed',error=repr(error));raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        queue.update(finished_utc=N.M.utc(),temporary_system_awake=False,blocked_network_attempts=blocked)
        write(OUT/'queue.json',queue)


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    elif mode in ['scheduled-run','scheduled-resume']:
        stamp=str(int(time.time()))
        sys.stdout=(OUT/f'training_{stamp}.log').open('a',encoding='utf-8',buffering=1)
        sys.stderr=(OUT/f'training_{stamp}.err.log').open('a',encoding='utf-8',buffering=1)
        scheduled(mode=='scheduled-resume')
    else:raise ValueError(mode)
