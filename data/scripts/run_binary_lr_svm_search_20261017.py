"""Bounded, nested-spatial extension of the completed binary LR/SVM trial.

Only the C/gamma search space changes. Standardization, folds, calibration,
selection metric, retention rules, candidates and subtype predictions stay fixed.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import warnings

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.venv/Lib/site-packages'),str(ROOT/'data/scripts')]
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import run_binary_lr_svm_20261017 as R

OUT=ROOT/'runs/experiments/binary_lr_svm_search_20261017_v2'
PARENT=R.OUT
CS=[.001,.01,.1,1.,10.,100.]
GAMMA_FACTORS=[.1,1.,10.]
read,write,sha,now=R.read,R.write,R.sha,R.now


def grid(arm):
    return [dict(C=c,gamma_factor=g if arm=='RBF_SVM' else None,
        gamma='scale' if g==1. else g/768.) if arm=='RBF_SVM' else
        dict(C=c,gamma_factor=None,gamma=None)
        for c in CS for g in (GAMMA_FACTORS if arm=='RBF_SVM' else [1.])]


def make_estimator(arm,config):
    model=R.estimator(arm,config['C'])
    if arm=='RBF_SVM':model.set_params(head__gamma=config['gamma'])
    return model


def calibrate(arm,config,splits):
    return CalibratedClassifierCV(make_estimator(arm,config),method='sigmoid',cv=splits,ensemble=False,n_jobs=1)


def prepare():
    if OUT.exists():raise FileExistsError('Preserve prior protocol; use scheduled-run to resume')
    old=read(PARENT/'protocol.json');verified=read(PARENT/'completion_verification.json')
    assert read(PARENT/'queue.json')['status']=='completed' and verified['status'].startswith('passed_')
    assert sha(PARENT/'protocol.json')==verified['protocol_sha256']==(PARENT/'protocol.sha256').read_text().strip()
    rows,x,y,folds,groups=R.load_training()
    assert sha(R.BASE/'training_oof.npz')==old['source_sha256'][str(R.BASE/'training_oof.npz')]
    for entry in old['outer']:
        fit=np.array(entry['fit']);held=np.array(entry['held']);R.split_receipt(rows,fit,held)
        for s in entry['inner']:R.split_receipt(rows,fit[np.array(s['fit'])],fit[np.array(s['held'])])
    paths=[Path(__file__),Path(R.__file__),Path(R.B.__file__),PARENT/'protocol.json',
        PARENT/'completion_verification.json',PARENT/'comparison.json',R.BASE/'protocol.json',
        R.BASE/'training_oof.npz',R.BASE/'development_scores.npz',R.B.COMBINED/'input_plan.json']
    paths += [R.B.COMBINED/f'seed{s}/original_joint224.npz' for s in R.B.SEEDS]
    # Pin every reused tuning score and selected calibrated parent model.
    for arm in R.ARMS:
        paths += [PARENT/arm/'selection.json',PARENT/arm/'training_predictions.npz',PARENT/arm/'development_predictions.npz']
        for stage in ['outer0','outer1','outer2','full']:
            paths += [PARENT/arm/stage/name for name in ['tuning.json','tuning_scores.npz','model.joblib','model_receipt.json']]
    source_pins={str(p):sha(p) for p in sorted(set(paths))}
    original_row_keys=[r['row_id'] for r in rows]
    spec=dict(status='locked_before_extended_search_fits',created_utc=now(),parent=str(PARENT),
        change='Only expand C grid and search RBF gamma; no representation, sampling, loss, folding or threshold objective changes',
        rows=2438,positive=1083,background=1355,features=768,spatial_groups=41,
        row_order_sha256=hashlib.sha256('\n'.join(original_row_keys).encode()).hexdigest(),
        arms=R.ARMS,configurations={arm:grid(arm) for arm in R.ARMS},
        C_grid=CS,gamma_factors=GAMMA_FACTORS,
        gamma_interpretation='For factor1 use original scale exactly; others use factor/768 on fold-standardized features. Approximately 0.1x/1x/10x the original gamma.',
        outer=old['outer'],full_fit_cv=old['full_fit_cv'],tuning=old['tuning'],
        calibration=old['calibration'],threshold_policies=old['threshold_policies'],threshold_rule=old['threshold_rule'],
        development=old['development'],environment=old['environment'],
        reuse='Reuse old held-fold margins only for exact same C and original scale gamma, pinned features/splits/code. Reuse calibrated models only for identical selected configuration and folds.',
        bounded_execution=dict(max_C=100.,min_C=.001,max_gamma_factor=10.,min_gamma_factor=.1,
            automatic_boundary_extension=False,calibrated_models_expected=12,threads=4),
        CPU_only=True,encoder_or_detector_retrained=False,old_test_backgrounds_added=False,
        new_features_encoded=0,final_test_pixels_read=False,automatic_model_promotion=False,
        posthoc_followup_to_completed_v1=True,previously_reused_development_regions=True,
        source_sha256=source_pins)
    OUT.mkdir(parents=True);write(OUT/'protocol.json',spec)
    (OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n',encoding='ascii')
    write(OUT/'queue.json',dict(status='prepared_not_started',created_utc=now(),CPU_only=True))
    print(json.dumps(dict(status=spec['status'],configurations={a:len(grid(a)) for a in R.ARMS},
        total_candidate_fold_evaluations=sum(len(grid(a))*12 for a in R.ARMS),GPU_required=False),ensure_ascii=True))


def key(config):return 'C'+format(config['C'],'g')+'_G'+str(config['gamma_factor'])


def old_tuning_scores(arm,stage,config):
    if config['C'] not in R.C_GRID or (arm=='RBF_SVM' and config['gamma']!='scale'):return None
    path=PARENT/arm/stage/'tuning_scores.npz'
    with np.load(path,allow_pickle=False) as data:
        index=data['C_grid'].tolist().index(config['C'])
        return data['decisions'][index].copy(),data['labels'].copy()


def tune(arm,stage,x,y,splits,configs,queue):
    folder=OUT/arm/stage;folder.mkdir(parents=True,exist_ok=True);trials=[];all_scores=[]
    for ci,config in enumerate(configs):
        name=key(config);receipt_path=folder/(name+'.json');score_path=folder/(name+'.npz')
        queue.update(arm=arm,stage=stage,phase='spatial_C_gamma_search',configuration=ci+1,
            configurations_total=len(configs),parameters=config);write(OUT/'queue.json',queue)
        if receipt_path.exists():
            trial=read(receipt_path);assert trial['protocol_sha256']==sha(OUT/'protocol.json') and trial['scores_sha256']==sha(score_path)
            with np.load(score_path,allow_pickle=False) as data:
                scores=data['decisions'].copy();assert np.array_equal(data['labels'],y)
        else:
            started=time.monotonic();reused=old_tuning_scores(arm,stage,config)
            if reused:
                scores,labels=reused;assert np.array_equal(labels,y)
                reuse_source=str(PARENT/arm/stage/'tuning_scores.npz')
            else:
                scores=np.full(len(y),np.nan)
                for fit,held in splits:
                    model=make_estimator(arm,config);model.fit(x[fit],y[fit])
                    assert np.allclose(model['scale'].mean_,x[fit].mean(0,dtype=np.float64),rtol=1e-7,atol=1e-8)
                    scores[held]=model.decision_function(x[held])
                reuse_source=None
            assert np.isfinite(scores).all()
            aucs=[float(roc_auc_score(y[held],scores[held])) for fit,held in splits]
            np.savez_compressed(score_path,decisions=scores,labels=y)
            trial=dict(parameters=config,fold_ROC_AUC=aucs,mean_ROC_AUC=float(np.mean(aucs)),
                seconds=time.monotonic()-started,reused_source=reuse_source,
                protocol_sha256=sha(OUT/'protocol.json'),scores_sha256=sha(score_path))
            write(receipt_path,trial)
        trials.append(trial);all_scores.append(scores)
        print('GRID',arm,stage,config,trial['mean_ROC_AUC'],'reused',bool(trial['reused_source']),flush=True)
    # Predeclared deterministic tie order: smaller C, then gamma closest to scale,
    # then smaller gamma factor. The selection metric stays the same as v1.
    best=min(trials,key=lambda t:(-t['mean_ROC_AUC'],t['parameters']['C'],
        abs(np.log10(t['parameters']['gamma_factor'] or 1.)),t['parameters']['gamma_factor'] or 1.))
    selection=dict(arm=arm,stage=stage,selected_parameters=best['parameters'],
        trials=trials,protocol_sha256=sha(OUT/'protocol.json'))
    write(folder/'tuning.json',selection);return best['parameters']


def selected_model(arm,stage,config,splits,x,y,queue):
    folder=OUT/arm/stage;path=folder/'model.joblib';receipt=folder/'model_receipt.json'
    queue.update(arm=arm,stage=stage,phase='spatial_probability_calibration',parameters=config);write(OUT/'queue.json',queue)
    if receipt.exists():
        previous=read(receipt);assert previous['parameters']==config and previous['model_sha256']==sha(path)
        assert previous['protocol_sha256']==sha(OUT/'protocol.json');return joblib.load(path)
    old_receipt=read(PARENT/arm/stage/'model_receipt.json')
    same=config['C']==old_receipt['C'] and (arm!='RBF_SVM' or config['gamma']=='scale')
    if same:
        shutil.copy2(PARENT/arm/stage/'model.joblib',path);model=joblib.load(path)
        reused=str(PARENT/arm/stage/'model.joblib')
    else:
        model=calibrate(arm,config,splits);model.fit(x,y);joblib.dump(model,path);reused=None
    assert model.calibrated_classifiers_[0].estimator['scale'].n_samples_seen_==len(y)
    write(receipt,dict(parameters=config,reused_model=reused,model_sha256=sha(path),protocol_sha256=sha(OUT/'protocol.json')))
    return model


def fit_arm(arm,spec,rows,x,y,folds,queue):
    folder=OUT/arm;folder.mkdir(exist_ok=True);oof=np.full(len(y),np.nan);choices=[]
    for entry in spec['outer']:
        fit=np.array(entry['fit']);held=np.array(entry['held']);stage=f"outer{entry['fold']}"
        splits=[(np.array(s['fit']),np.array(s['held'])) for s in entry['inner']]
        config=tune(arm,stage,x[fit],y[fit],splits,spec['configurations'][arm],queue)
        model=selected_model(arm,stage,config,splits,x[fit],y[fit],queue)
        oof[held]=model.predict_proba(x[held])[:,1];choices.append(dict(fold=entry['fold'],parameters=config))
    assert np.isfinite(oof).all()
    classes=np.array([r['catalog_class'] or 'BACKGROUND' for r in rows])
    selection=dict(arm=arm,metrics=R.summary(y,oof,folds),
        policies=[R.policy(oof,y,classes,p) for p in spec['threshold_policies']],
        outer_models=choices,locked_before_development_scoring=True,
        threshold_source='Nested training OOF only',protocol_sha256=sha(OUT/'protocol.json'))
    write(folder/'training_selection.json',selection)
    np.savez_compressed(folder/'training_predictions.npz',scores=oof,labels=y,folds=folds,
        row_id=np.array([r['row_id'] for r in rows]),catalog_class=classes)
    splits=[(np.array(s['fit']),np.array(s['held'])) for s in spec['full_fit_cv']]
    config=tune(arm,'full',x,y,splits,spec['configurations'][arm],queue)
    model=selected_model(arm,'full',config,splits,x,y,queue)
    with np.load(R.BASE/'development_scores.npz',allow_pickle=False) as data:dx=data['features'].copy()
    assert dx.shape==(2009,768)
    scores=model.predict_proba(dx)[:,1]
    np.savez_compressed(folder/'development_predictions.npz',scores=scores,case_index=np.arange(2009))
    selection.update(full_model_parameters=config,full_model_sha256=sha(folder/'full/model.joblib'),
        training_selection_sha256=sha(folder/'training_selection.json'))
    write(folder/'selection.json',selection);return selection,scores


def render(result):
    lines=['# 二分类扩大C／gamma搜索：有限追加对照','',
        '同1083个正目标、1355个背景、41空间组及3折；只扩展C和RBF gamma，保留原标准化、空间校准、AUC选参和训练OOF阈值规则。',
        'C=0.001／0.01／0.1／1／10／100；RBF gamma约为原规则的0.1／1／10倍。旧范围配置的空间留出分数在核验后复用。','',
        '|分类器|轮次|外折平均ROC-AUC|0.5处平衡正确率|训练逐类97%规则的背景排除率|',
        '|---|---|---:|---:|---:|']
    for arm,selection in result['selections'].items():
        for label,s in [('v1有限范围',result['parent_selections'][arm]),('v2扩大搜索',selection)]:
            lines.append(f"|{arm}|{label}|{s['metrics']['mean_fold_ROC_AUC']:.4f}|{s['metrics']['balanced_accuracy_at_half']:.2%}|{s['policies'][0]['training_OOF_background_rejection']:.2%}|")
    for policy in ['each_type_positive97','overall_positive97']:
        lines += ['',f'## 实际候选：{policy}','',
            '|候选池|方案|候选数|定位召回|参考精确率|独立定位／183|联合召回均值|独立正确均数／183|',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for pool,variants in result['development'].items():
            for label,r in [('off',variants['off'])]+[(a,variants[a][policy]) for a in R.ARMS]:
                lines.append(f"|{pool}|{label}|{r['candidates']}|{r['recall']:.2%}|{r['reference_precision']:.2%}|{r['unique_localized']}|{np.mean(r['joint_recall']):.2%}|{np.mean(r['unique_joint']):.2f}|")
    lines += ['', '97%是训练区域OOF选阈值要求，不能保证新区域留存；各模型实际留存不同，需同时查看正例损失和背景排除。',
        '外折只评估已在内层选好的模型；最终模型的C和gamma只由全部训练区域的空间CV决定，开发区不用来选择参数。',
        '既有开发区域多次使用；冻结原定位配对，不重匹配、不重算AP，未匹配不一定为真背景。',
        '这是已看过首轮结果后的有限追加研究，不是新盲测；不自动替换模型，不自动扩展下一轮。',
        f'CPU耗时{result["CPU_seconds"]:.1f}秒；没有重训DINOv2或读取最终测试像素。']
    (OUT/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run():
    spec=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for p,digest in spec['source_sha256'].items():assert sha(p)==digest,p
    attempts=[]
    def blocked(*a,**k):attempts.append('blocked_connection');raise OSError('Offline experiment')
    socket.socket.connect=blocked;socket.socket.connect_ex=blocked;socket.create_connection=blocked
    queue=dict(status='running',started_utc=now(),CPU_only=True,offline=True,threads=4,
        no_automatic_following_experiment=True,protocol_sha256=sha(OUT/'protocol.json'),process=R.process_receipt())
    write(OUT/'queue.json',queue);started=time.monotonic();ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        warnings.simplefilter('error',ConvergenceWarning)
        rows,x,y,folds,groups=R.load_training();selections={};scores={}
        with threadpool_limits(limits=4):
            for arm in spec['arms']:selections[arm],scores[arm]=fit_arm(arm,spec,rows,x,y,folds,queue)
        plan=read(R.B.COMBINED/'input_plan.json');classes=[]
        for seed in R.B.SEEDS:
            with np.load(R.B.COMBINED/f'seed{seed}/original_joint224.npz',allow_pickle=False) as data:classes.append(data['predictions'].copy())
        result=dict(status='completed_extended_C_gamma_spatial_search',created_utc=now(),
            selections=selections,parent_selections=read(PARENT/'comparison.json')['selections'],
            CPU_seconds=time.monotonic()-started,protocol_sha256=sha(OUT/'protocol.json'),
            automatic_model_promotion=False,development={},boundary_flags={})
        for arm,selection in selections.items():
            configs=[r['parameters'] for r in selection['outer_models']]+[selection['full_model_parameters']]
            result['boundary_flags'][arm]=[dict(parameters=c,C_at_boundary=c['C'] in (min(CS),max(CS)),
                gamma_at_boundary=arm=='RBF_SVM' and c['gamma_factor'] in (min(GAMMA_FACTORS),max(GAMMA_FACTORS))) for c in configs]
        for pool in spec['development']['pools']:
            variants={'off':R.B.evaluate_pool(plan,pool,np.ones(2009,dtype=bool),classes)}
            for arm in R.ARMS:
                variants[arm]={p['name']:R.B.evaluate_pool(plan,pool,scores[arm]>=p['threshold'],classes) for p in selections[arm]['policies']}
            result['development'][pool]=variants
        write(OUT/'comparison.json',result);render(result)
        for p,digest in spec['source_sha256'].items():assert sha(p)==digest,p
        queue.update(status='completed',phase='finite_extended_search_finished',finished_utc=now(),
            elapsed_seconds=time.monotonic()-started,blocked_network_attempts=attempts)
        write(OUT/'queue.json',queue)
        print(json.dumps(dict(status=queue['status'],seconds=queue['elapsed_seconds'],selected={a:s['full_model_parameters'] for a,s in selections.items()}),ensure_ascii=True),flush=True)
    except BaseException as exc:
        queue.update(status='failed',error=repr(exc),finished_utc=now());write(OUT/'queue.json',queue);raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def smoke():
    rng=np.random.default_rng(R.SEED);x=rng.normal(size=(90,8)).astype('float32');y=(x[:,0]>0).astype(int)
    folds=np.arange(90)%3;splits=[(np.flatnonzero(folds!=f),np.flatnonzero(folds==f)) for f in range(3)]
    with threadpool_limits(limits=4):
        for arm in R.ARMS:
            assert len(grid(arm))==(18 if arm=='RBF_SVM' else 6)
            config=next(c for c in grid(arm) if c['C']==1. and (arm!='RBF_SVM' or c['gamma']=='scale'))
            a=make_estimator(arm,config).fit(x,y);b=R.estimator(arm,1.).fit(x,y)
            assert np.array_equal(a.decision_function(x),b.decision_function(x))
            model=calibrate(arm,config,splits).fit(x,y);assert np.isfinite(model.predict_proba(x)).all()
            print('SMOKE',arm,'baseline_equal',True,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','prepare','run','scheduled-run']);args=parser.parse_args()
    if args.mode=='smoke':smoke()
    elif args.mode=='prepare':prepare()
    else:
        if args.mode=='scheduled-run':
            sys.stdout=(OUT/'training.log').open('a',encoding='utf-8',buffering=1)
            sys.stderr=(OUT/'training.err.log').open('a',encoding='utf-8',buffering=1)
        run()
