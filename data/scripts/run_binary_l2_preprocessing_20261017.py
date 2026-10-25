"""Finite L2-before-standardization comparison on frozen binary features.

The completed extended-grid experiment is the pinned reference. This experiment
adds row-wise L2 normalization before the same train-fitted StandardScaler and
uses the same C/gamma search, nested spatial splits, calibration and thresholds.
It does not test L2-only/cosine classifiers or ingest new unreviewed negatives.
"""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import socket
import sys
import time
import warnings

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.venv/Lib/site-packages'),str(ROOT/'data/scripts')]
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from threadpoolctl import threadpool_limits
import run_binary_lr_svm_search_20261017 as S

R=S.R
PARENT=S.OUT
OUT=ROOT/'runs/experiments/binary_l2_preprocessing_20261017_v1'
read,write,sha,now=R.read,R.write,R.sha,R.now


def estimator(arm,config):
    base=S.make_estimator(arm,config)
    return Pipeline([('normalize',Normalizer(norm='l2'))]+base.steps)


def check_scaler(pipeline,x):
    normalized=pipeline['normalize'].transform(x)
    lengths=np.linalg.norm(normalized,axis=1)
    assert np.all((np.abs(lengths-1)<1e-5)|(lengths==0))
    assert pipeline['scale'].n_samples_seen_==len(x)
    assert np.allclose(pipeline['scale'].mean_,normalized.mean(0,dtype=np.float64),rtol=1e-7,atol=1e-8)


def prepare():
    if OUT.exists():raise FileExistsError('Preserve existing experiment; resume with scheduled-run')
    parent=read(PARENT/'protocol.json');verified=read(PARENT/'completion_verification.json')
    assert read(PARENT/'queue.json')['status']=='completed'
    assert verified['status'].startswith('passed_') and verified['protocol_sha256']==sha(PARENT/'protocol.json')
    rows,x,y,folds,groups=R.load_training()
    assert np.all(np.linalg.norm(x,axis=1)>0)
    pins=dict(parent['source_sha256'])
    paths=[Path(__file__),Path(S.__file__),PARENT/'protocol.json',PARENT/'comparison.json',
        PARENT/'completion_verification.json']
    for arm in R.ARMS:
        paths += [PARENT/arm/'selection.json',PARENT/arm/'training_predictions.npz',
            PARENT/arm/'development_predictions.npz']
        for stage in ['outer0','outer1','outer2','full']:
            paths += [PARENT/arm/stage/'model.joblib',PARENT/arm/stage/'model_receipt.json']
    pins.update({str(p):sha(p) for p in paths})
    for p,digest in pins.items():assert sha(p)==digest,p
    spec=dict(status='locked_before_L2_fits',created_utc=now(),reference=str(PARENT),
        change='Add sample-wise L2 Normalizer before the unchanged train-fitted StandardScaler; no new data',
        not_tested=['L2-only without StandardScaler','cosine kernel','hard-negative additions','different threshold objective'],
        rows=2438,positive=1083,background=1355,features=768,spatial_groups=41,
        arms=R.ARMS,configurations=parent['configurations'],C_grid=parent['C_grid'],
        gamma_factors=parent['gamma_factors'],gamma_interpretation=parent['gamma_interpretation'],
        outer=parent['outer'],full_fit_cv=parent['full_fit_cv'],tuning=parent['tuning'],
        calibration=parent['calibration'],threshold_policies=parent['threshold_policies'],
        threshold_rule=parent['threshold_rule'],development=parent['development'],
        environment=parent['environment'],source_sha256=pins,
        reuse='Only the completed no-L2 comparison is reused; all L2 configurations are newly fit',
        bounded_execution=dict(threads=4,maximum_minutes=30,automatic_followup=False),
        CPU_only=True,offline=True,new_images_encoded=0,final_test_pixels_read=0,
        old_test_backgrounds_added=0,unreviewed_negatives_added=0,automatic_model_promotion=False,
        exploratory_followup=True,development_regions_previously_used=True)
    OUT.mkdir(parents=True);write(OUT/'protocol.json',spec)
    (OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n',encoding='ascii')
    write(OUT/'queue.json',dict(status='prepared_not_started',created_utc=now()))
    print(json.dumps(dict(status=spec['status'],candidate_fold_evaluations=360,CPU_only=True)))


def tune(arm,stage,x,y,splits,spec,queue,deadline):
    folder=OUT/arm/stage;folder.mkdir(parents=True,exist_ok=True);trials=[]
    for i,config in enumerate(spec['configurations'][arm]):
        if time.monotonic()>deadline:raise TimeoutError('Finite 30 minute experiment budget exhausted')
        queue.update(arm=arm,stage=stage,phase='L2_spatial_search',configuration=i+1,
            configurations_total=len(spec['configurations'][arm]),parameters=config)
        write(OUT/'queue.json',queue)
        path=folder/(S.key(config)+'.npz');receipt=path.with_suffix('.json')
        if receipt.exists():
            trial=read(receipt)
            assert trial['parameters']==config and trial['scores_sha256']==sha(path)
            assert trial['protocol_sha256']==sha(OUT/'protocol.json')
            with np.load(path,allow_pickle=False) as d:
                margins=d['decisions'].copy();assert np.array_equal(d['labels'],y)
        else:
            started=time.monotonic();margins=np.full(len(y),np.nan)
            for fit,held in splits:
                model=estimator(arm,config).fit(x[fit],y[fit]);check_scaler(model,x[fit])
                margins[held]=model.decision_function(x[held])
            assert np.isfinite(margins).all()
            aucs=[float(roc_auc_score(y[b],margins[b])) for a,b in splits]
            np.savez_compressed(path,decisions=margins,labels=y)
            trial=dict(parameters=config,fold_ROC_AUC=aucs,mean_ROC_AUC=float(np.mean(aucs)),
                seconds=time.monotonic()-started,scores_sha256=sha(path),protocol_sha256=sha(OUT/'protocol.json'))
            write(receipt,trial)
        trials.append(trial)
        print('GRID',arm,stage,config,trial['mean_ROC_AUC'],flush=True)
    best=min(trials,key=lambda t:(-t['mean_ROC_AUC'],t['parameters']['C'],
        abs(np.log10(t['parameters']['gamma_factor'] or 1.)),t['parameters']['gamma_factor'] or 1.))
    write(folder/'tuning.json',dict(arm=arm,stage=stage,selected_parameters=best['parameters'],
        trials=trials,protocol_sha256=sha(OUT/'protocol.json')))
    return best['parameters']


def fit_selected(arm,stage,config,splits,x,y,queue):
    folder=OUT/arm/stage;path=folder/'model.joblib';receipt=folder/'model_receipt.json'
    queue.update(arm=arm,stage=stage,phase='L2_spatial_calibration',parameters=config);write(OUT/'queue.json',queue)
    if receipt.exists():
        saved=read(receipt)
        assert saved['parameters']==config and saved['model_sha256']==sha(path)
        assert saved['protocol_sha256']==sha(OUT/'protocol.json');model=joblib.load(path)
    else:
        model=CalibratedClassifierCV(estimator(arm,config),method='sigmoid',cv=splits,ensemble=False,n_jobs=1).fit(x,y)
        joblib.dump(model,path)
        write(receipt,dict(parameters=config,model_sha256=sha(path),protocol_sha256=sha(OUT/'protocol.json')))
    check_scaler(model.calibrated_classifiers_[0].estimator,x)
    return model


def fit_arm(arm,spec,rows,x,y,folds,queue,deadline):
    folder=OUT/arm;folder.mkdir(exist_ok=True);oof=np.full(len(y),np.nan);choices=[]
    for entry in spec['outer']:
        fit=np.array(entry['fit']);held=np.array(entry['held']);stage=f"outer{entry['fold']}"
        splits=[(np.array(s['fit']),np.array(s['held'])) for s in entry['inner']]
        config=tune(arm,stage,x[fit],y[fit],splits,spec,queue,deadline)
        model=fit_selected(arm,stage,config,splits,x[fit],y[fit],queue)
        oof[held]=model.predict_proba(x[held])[:,1]
        choices.append(dict(fold=entry['fold'],parameters=config))
    assert np.isfinite(oof).all()
    classes=np.array([r['catalog_class'] or 'BACKGROUND' for r in rows])
    selection=dict(arm=arm,metrics=R.summary(y,oof,folds),
        policies=[R.policy(oof,y,classes,p) for p in spec['threshold_policies']],outer_models=choices,
        locked_before_development_scoring=True,threshold_source='Nested spatial training OOF only',
        protocol_sha256=sha(OUT/'protocol.json'))
    write(folder/'training_selection.json',selection)
    np.savez_compressed(folder/'training_predictions.npz',scores=oof,labels=y,folds=folds,
        row_id=np.array([r['row_id'] for r in rows]),catalog_class=classes)
    splits=[(np.array(s['fit']),np.array(s['held'])) for s in spec['full_fit_cv']]
    config=tune(arm,'full',x,y,splits,spec,queue,deadline)
    model=fit_selected(arm,'full',config,splits,x,y,queue)
    with np.load(R.BASE/'development_scores.npz',allow_pickle=False) as d:dx=d['features'].copy()
    scores=model.predict_proba(dx)[:,1]
    np.savez_compressed(folder/'development_predictions.npz',scores=scores,case_index=np.arange(2009))
    selection.update(full_model_parameters=config,full_model_sha256=sha(folder/'full/model.joblib'),
        training_selection_sha256=sha(folder/'training_selection.json'))
    write(folder/'selection.json',selection)
    return selection,scores


def render(result):
    lines=['# L2归一化后再标准化：有限二分类对照','',
        '同一冻结DINOv2特征、41空间组、参数网格、校准和阈值规则；唯一新增处理是先将每个向量归一化到单位长度。',
        '该对照保留StandardScaler，不等同于只做L2或直接采用余弦核。','',
        '|分类器|预处理|外折平均ROC-AUC|0.5处平衡正确率|训练正例留存|训练背景排除|',
        '|---|---|---:|---:|---:|---:|']
    for arm in R.ARMS:
        for name,selection in [('原标准化',result['reference_selections'][arm]),('L2后再标准化',result['selections'][arm])]:
            m=selection['metrics'];p=selection['policies'][0]
            lines.append(f"|{arm}|{name}|{m['mean_fold_ROC_AUC']:.4f}|{m['balanced_accuracy_at_half']:.2%}|{p['training_OOF_positive_retention']:.2%}|{p['training_OOF_background_rejection']:.2%}|")
    for policy in ['each_type_positive97','overall_positive97']:
        lines += ['',f'## 实际候选：{policy}','','|候选池|方法|预处理|候选数|定位召回|参考精确率|独立定位|独立联合正确均数|',
            '|---|---|---|---:|---:|---:|---:|---:|']
        for pool,variants in result['development'].items():
            off=variants['off'];lines.append(f"|{pool}|off|不加过滤|{off['candidates']}|{off['recall']:.2%}|{off['reference_precision']:.2%}|{off['unique_localized']}|{np.mean(off['unique_joint']):.2f}|")
            for arm in R.ARMS:
                for name,group in [('原标准化',result['reference_development'][pool]),('L2后再标准化',variants)]:
                    r=group[arm][policy]
                    lines.append(f"|{pool}|{arm}|{name}|{r['candidates']}|{r['recall']:.2%}|{r['reference_precision']:.2%}|{r['unique_localized']}|{np.mean(r['unique_joint']):.2f}|")
    lines += ['', '97%为训练OOF选阈值条件，不是新区域保留率保证。开发区域已反复使用，不能视为新盲测。',
        '原定位配对固定，不重匹配或重算AP；未匹配不一定是真背景。新困难背景未入训练。',
        f'计算耗时{result["elapsed_seconds"]:.1f}秒，不自动替换模型或追加下一轮。']
    (OUT/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run():
    spec=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for p,digest in spec['source_sha256'].items():assert sha(p)==digest,p
    attempts=[]
    def blocked(*a,**k):attempts.append('connection_blocked');raise OSError('Offline experiment')
    socket.socket.connect=blocked;socket.socket.connect_ex=blocked;socket.create_connection=blocked
    queue=dict(status='running',started_utc=now(),CPU_only=True,offline=True,threads=4,
        process=R.process_receipt(),no_automatic_following_experiment=True,protocol_sha256=sha(OUT/'protocol.json'))
    write(OUT/'queue.json',queue);started=time.monotonic();deadline=started+1800
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        warnings.simplefilter('error',ConvergenceWarning)
        rows,x,y,folds,groups=R.load_training();selections={};scores={}
        with threadpool_limits(limits=4):
            for arm in R.ARMS:selections[arm],scores[arm]=fit_arm(arm,spec,rows,x,y,folds,queue,deadline)
        parent=read(PARENT/'comparison.json');plan=read(R.B.COMBINED/'input_plan.json');classes=[]
        for seed in R.B.SEEDS:
            with np.load(R.B.COMBINED/f'seed{seed}/original_joint224.npz',allow_pickle=False) as d:classes.append(d['predictions'].copy())
        result=dict(status='completed_L2_before_standardization',created_utc=now(),
            selections=selections,reference_selections=parent['selections'],reference_development=parent['development'],
            development={},elapsed_seconds=time.monotonic()-started,protocol_sha256=sha(OUT/'protocol.json'),
            automatic_model_promotion=False)
        for pool in spec['development']['pools']:
            variants={'off':R.B.evaluate_pool(plan,pool,np.ones(2009,dtype=bool),classes)}
            for arm in R.ARMS:
                variants[arm]={p['name']:R.B.evaluate_pool(plan,pool,scores[arm]>=p['threshold'],classes) for p in selections[arm]['policies']}
            result['development'][pool]=variants
        write(OUT/'comparison.json',result);render(result)
        for p,digest in spec['source_sha256'].items():assert sha(p)==digest,p
        queue.update(status='completed',finished_utc=now(),elapsed_seconds=time.monotonic()-started,
            phase='finite_L2_comparison_finished',blocked_network_attempts=attempts)
        write(OUT/'queue.json',queue);print(json.dumps(dict(status='completed',seconds=queue['elapsed_seconds'])),flush=True)
    except BaseException as exc:
        queue.update(status='failed',error=repr(exc),finished_utc=now());write(OUT/'queue.json',queue);raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def smoke():
    rng=np.random.default_rng(R.SEED);x=rng.normal(size=(90,768)).astype('float32');x*=rng.uniform(.3,3,(90,1))
    y=(x[:,0]>0).astype(int);f=np.arange(90)%3
    splits=[(np.flatnonzero(f!=i),np.flatnonzero(f==i)) for i in range(3)]
    with threadpool_limits(limits=4):
        for arm in R.ARMS:
            config=next(c for c in S.grid(arm) if c['C']==1 and (arm!='RBF_SVM' or c['gamma']=='scale'))
            model=estimator(arm,config).fit(x,y);check_scaler(model,x)
            z=Normalizer().fit_transform(x)
            reference=S.make_estimator(arm,config).fit(z,y)
            assert np.allclose(model.decision_function(x),reference.decision_function(z),rtol=0,atol=1e-12)
            calibrated=CalibratedClassifierCV(estimator(arm,config),cv=splits,method='sigmoid',ensemble=False).fit(x,y)
            assert np.isfinite(calibrated.predict_proba(x)).all()
            print('SMOKE',arm,'manual_normalization_equivalent',True,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','prepare','run','scheduled-run']);args=parser.parse_args()
    if args.mode=='smoke':smoke()
    elif args.mode=='prepare':prepare()
    else:
        if args.mode=='scheduled-run':
            sys.stdout=(OUT/'training.log').open('a',encoding='utf-8',buffering=1)
            sys.stderr=(OUT/'training.err.log').open('a',encoding='utf-8',buffering=1)
        run()
