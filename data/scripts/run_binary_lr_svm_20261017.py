"""Finite CPU comparison of binary LR/linear-SVM/RBF-SVM on frozen DINOv2 features.

Nested spatial tuning, the same spatial sigmoid calibration for every arm,
training-OOF-only retention thresholds, and read-only existing development pools.
No image re-encoding, detector training, expert label changes, or test pixels.
"""
from __future__ import annotations
import argparse
from collections import Counter
import ctypes
from datetime import datetime,timezone
import hashlib
import inspect
import json
import math
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
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,balanced_accuracy_score,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC,SVC
from threadpoolctl import threadpool_limits
import run_manual_background_filter_20261016 as B

BASE=B.OUT
OUT=ROOT/'runs/experiments/binary_lr_svm_20261017_v1'
ARMS=['LR','linear_SVM','RBF_SVM']
C_GRID=[.1,1.,10.]
TYPES=['SL','SP','FL','CL']
SEED=20261017


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def estimator(arm,c):
    if arm=='LR':head=LogisticRegression(C=c,class_weight='balanced',solver='lbfgs',max_iter=3000,random_state=SEED)
    elif arm=='linear_SVM':head=LinearSVC(C=c,class_weight='balanced',dual='auto',max_iter=20000,tol=1e-4,random_state=SEED)
    elif arm=='RBF_SVM':head=SVC(C=c,kernel='rbf',gamma='scale',class_weight='balanced',
        cache_size=512,tol=1e-4,random_state=SEED)
    else:raise ValueError(arm)
    return Pipeline([('scale',StandardScaler()),('head',head)])


def calibrated(arm,c,splits):
    # All arms use explicit group-disjoint folds. libsvm's internal random
    # probability fitting is disabled, including for RBF SVM.
    return CalibratedClassifierCV(estimator(arm,c),method='sigmoid',cv=splits,ensemble=False,n_jobs=1)


def load_training():
    parent=read(BASE/'protocol.json');rows=parent['rows']
    with np.load(BASE/'training_oof.npz',allow_pickle=False) as data:
        x,y,folds=[data[k].copy() for k in ['features','labels','folds']]
        assert data['row_id'].tolist()==[r['row_id'] for r in rows]
    assert x.shape==(2438,768) and np.isfinite(x).all()
    assert y.tolist()==[r['binary_label'] for r in rows] and folds.tolist()==[r['fold_id'] for r in rows]
    assert np.bincount(y).tolist()==[1355,1083]
    groups=np.array([r['spatial_group_id'] for r in rows])
    assert len(set(groups))==41
    for group in set(groups):assert len(set(folds[groups==group]))==1
    return rows,x,y,folds,groups


def split_receipt(rows,global_fit,global_held):
    a={rows[int(i)]['spatial_group_id'] for i in global_fit};b={rows[int(i)]['spatial_group_id'] for i in global_held}
    ta={t for i in global_fit for t in rows[int(i)]['source_tiles']}
    tb={t for i in global_held for t in rows[int(i)]['source_tiles']}
    assert not a&b and not ta&tb
    return dict(fit=len(global_fit),held=len(global_held),fit_groups=len(a),held_groups=len(b),
        fit_positive=sum(rows[int(i)]['binary_label']==1 for i in global_fit),
        held_positive=sum(rows[int(i)]['binary_label']==1 for i in global_held),
        group_and_donor_tile_disjoint=True)


def prepare():
    if OUT.exists():raise FileExistsError('Preserve the existing experiment; use run to resume')
    assert read(BASE/'queue.json')['status']=='completed'
    assert read(BASE/'completion_verification.json')['status'].startswith('passed_')
    assert sha(BASE/'protocol.json')==(BASE/'protocol.sha256').read_text().strip()
    rows,x,y,folds,groups=load_training();outer=[]
    for fold in range(3):
        fit=np.flatnonzero(folds!=fold);held=np.flatnonzero(folds==fold)
        receipt=split_receipt(rows,fit,held);inner=[]
        splitter=StratifiedGroupKFold(3,shuffle=True,random_state=SEED+fold)
        for a,b in splitter.split(np.zeros(len(fit)),y[fit],groups[fit]):
            assert set(y[fit[a]])==set(y[fit[b]])=={0,1}
            inner.append(dict(fit=a.tolist(),held=b.tolist(),receipt=split_receipt(rows,fit[a],fit[b])))
        outer.append(dict(fold=fold,fit=fit.tolist(),held=held.tolist(),receipt=receipt,inner=inner))
    full_cv=[dict(fit=np.flatnonzero(folds!=fold).tolist(),held=np.flatnonzero(folds==fold).tolist()) for fold in range(3)]
    diagnostic=read(ROOT/'data/processed/reviewed_background_20261012_v1/manifest.json')
    assert not {r['tile'] for r in diagnostic['samples']}&{t for r in rows for t in r['source_tiles']}
    paths=[Path(__file__),Path(B.__file__),BASE/'protocol.json',BASE/'training_oof.npz',
        BASE/'development_scores.npz',BASE/'completion_verification.json',
        B.COMBINED/'input_plan.json',ROOT/'data/processed/reviewed_background_20261012_v1/manifest.json']
    paths += [BASE/f'fold{f}.joblib' for f in range(3)]+[BASE/'full_filter.joblib']
    paths += [B.COMBINED/f'seed{s}/original_joint224.npz' for s in B.SEEDS]
    paths += [Path(inspect.getfile(cls)) for cls in [CalibratedClassifierCV,LogisticRegression,LinearSVC,SVC,StandardScaler,StratifiedGroupKFold]]
    spec=dict(status='locked_before_binary_classifier_fits',created_utc=now(),
        parent=str(BASE),rows=len(rows),positive=1083,background=1355,features=768,spatial_groups=41,
        arms=ARMS,C_grid=C_GRID,RBF_gamma='scale',class_weight='balanced',seed=SEED,
        outer=outer,full_fit_cv=full_cv,
        tuning='Inner three spatial folds; maximize equally weighted mean ROC-AUC, ties choose smaller C. Outer labels never choose C.',
        calibration=dict(method='sigmoid',same_for_all_arms=True,ensemble=False,
            cv='Explicit inner spatial folds for outer models; original outer spatial folds for final all-training model',
            libsvm_random_probability_cv=False),
        threshold_policies=['each_type_positive97','overall_positive97'],
        threshold_rule='Using only nested training OOF positive scores, largest observed cutoff retaining >=97% of each expert subtype (primary) or all positives (secondary). Threshold fitting on OOF is a training selection, not independent retention evidence.',
        development=dict(parent=str(B.COMBINED),images=173,GT_occurrences=560,unique_GT=183,
            cases=2009,candidates_each_pool=1071,pools=['global_only','global_local'],
            fixed_localization_assignments=True,rematching=False,AP_recomputed=False,
            type_classifier_seeds=B.SEEDS,previously_reused_development_regions=True),
        environment=dict(python=sys.version.split()[0],sklearn=sklearn.__version__,numpy=np.__version__,threads=4),
        no_encoder_or_detector_training=True,new_images_encoded=0,test_pixels_read=False,
        old_test_backgrounds_in_training=0,expert_types_changed=False,automatic_model_promotion=False,
        automatic_next_experiment=False,source_sha256={str(p):sha(p) for p in sorted(set(paths))})
    OUT.mkdir(parents=True);write(OUT/'protocol.json',spec)
    (OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n',encoding='ascii')
    write(OUT/'queue.json',dict(status='prepared_not_started',created_utc=now(),CPU_only=True))
    print(json.dumps(dict(status=spec['status'],arms=ARMS,rows=len(rows),GPU_required=False),ensure_ascii=True),flush=True)


def tune(arm,x,y,splits,folder):
    folder.mkdir(exist_ok=True)
    cached=folder/'tuning.json'
    if cached.is_file():
        result=read(cached);assert result['protocol_sha256']==sha(OUT/'protocol.json')
        assert result['scores_sha256']==sha(folder/'tuning_scores.npz')
        return result
    decisions=np.full((len(C_GRID),len(y)),np.nan);trials=[]
    for ci,c in enumerate(C_GRID):
        metrics=[]
        for fi,(fit,held) in enumerate(splits):
            started=time.monotonic();model=estimator(arm,c);model.fit(x[fit],y[fit])
            assert np.allclose(model['scale'].mean_,x[fit].mean(0,dtype=np.float64),rtol=1e-7,atol=1e-8)
            scores=np.asarray(model.decision_function(x[held]),np.float64);decisions[ci,held]=scores
            metrics.append(dict(fold=fi,ROC_AUC=float(roc_auc_score(y[held],scores)),
                seconds=time.monotonic()-started,scaler_fit_rows=int(model['scale'].n_samples_seen_)))
        trials.append(dict(C=c,folds=metrics,mean_ROC_AUC=float(np.mean([m['ROC_AUC'] for m in metrics]))))
        print('TUNE',arm,folder.name,'C',c,'mean_AUC',trials[-1]['mean_ROC_AUC'],flush=True)
    assert np.isfinite(decisions).all()
    best=min(trials,key=lambda r:(-r['mean_ROC_AUC'],r['C']))
    np.savez_compressed(folder/'tuning_scores.npz',decisions=decisions,labels=y,C_grid=np.array(C_GRID))
    result=dict(arm=arm,selected_C=best['C'],trials=trials,protocol_sha256=sha(OUT/'protocol.json'),
        scores_sha256=sha(folder/'tuning_scores.npz'))
    write(cached,result);return result


def cutoff(scores):
    assert len(scores)>0 and np.isfinite(scores).all()
    return float(np.sort(scores)[math.floor(.03*len(scores))])


def policy(scores,y,classes,name):
    threshold=min(cutoff(scores[classes==c]) for c in TYPES) if name=='each_type_positive97' else cutoff(scores[y==1])
    keep=scores>=threshold
    per_class={c:dict(count=int(sum(classes==c)),retained=int(sum(keep[classes==c])),
        retention=float(np.mean(keep[classes==c]))) for c in TYPES}
    assert np.mean(keep[y==1])>=.97
    if name=='each_type_positive97':assert all(r['retention']>=.97 for r in per_class.values())
    return dict(name=name,threshold=threshold,training_OOF_positive_retention=float(np.mean(keep[y==1])),
        training_OOF_background_rejection=float(np.mean(~keep[y==0])),per_class=per_class)


def summary(y,scores,folds):
    by_fold=[dict(fold=f,count=int(sum(folds==f)),ROC_AUC=float(roc_auc_score(y[folds==f],scores[folds==f])),
        positive_average_precision=float(average_precision_score(y[folds==f],scores[folds==f]))) for f in range(3)]
    return dict(pooled_ROC_AUC=float(roc_auc_score(y,scores)),positive_average_precision=float(average_precision_score(y,scores)),
        mean_fold_ROC_AUC=float(np.mean([r['ROC_AUC'] for r in by_fold])),
        balanced_accuracy_at_half=float(balanced_accuracy_score(y,scores>=.5)),folds=by_fold)


def fit_arm(arm,spec,rows,x,y,folds,queue):
    folder=OUT/arm;folder.mkdir(exist_ok=True);oof=np.full(len(y),np.nan);models=[]
    for entry in spec['outer']:
        fold=entry['fold'];fit=np.array(entry['fit']);held=np.array(entry['held'])
        splits=[(np.array(s['fit']),np.array(s['held'])) for s in entry['inner']]
        stage=folder/f'outer{fold}';stage.mkdir(exist_ok=True)
        queue.update(arm=arm,phase='nested_outer_fit',outer_fold=fold);write(OUT/'queue.json',queue)
        selected=tune(arm,x[fit],y[fit],splits,stage)
        path=stage/'model.joblib';receipt=stage/'model_receipt.json'
        if path.is_file() and receipt.is_file():
            saved=read(receipt);assert saved['protocol_sha256']==sha(OUT/'protocol.json') and saved['model_sha256']==sha(path)
            model=joblib.load(path)
        else:
            model=calibrated(arm,selected['selected_C'],splits);model.fit(x[fit],y[fit]);joblib.dump(model,path)
            write(receipt,dict(C=selected['selected_C'],model_sha256=sha(path),protocol_sha256=sha(OUT/'protocol.json')))
        oof[held]=model.predict_proba(x[held])[:,1]
        models.append(dict(fold=fold,C=selected['selected_C'],model_sha256=sha(path)))
        print('OUTER_COMPLETE',arm,fold,flush=True)
    assert np.isfinite(oof).all() and np.all((oof>=0)&(oof<=1))
    classes=np.array([r['catalog_class'] or 'BACKGROUND' for r in rows])
    policies=[policy(oof,y,classes,name) for name in spec['threshold_policies']]
    selection=dict(arm=arm,metrics=summary(y,oof,folds),policies=policies,
        threshold_source='Nested spatial OOF training scores only',locked_before_development_scoring=True,
        outer_models=models,protocol_sha256=sha(OUT/'protocol.json'))
    write(folder/'selection.json',selection)
    write(folder/'training_selection.json',selection)
    np.savez_compressed(folder/'training_predictions.npz',scores=oof,labels=y,folds=folds,
        row_id=np.array([r['row_id'] for r in rows]),catalog_class=classes)
    splits=[(np.array(s['fit']),np.array(s['held'])) for s in spec['full_fit_cv']]
    stage=folder/'full';stage.mkdir(exist_ok=True)
    queue.update(arm=arm,phase='all_training_tuning_and_calibration');write(OUT/'queue.json',queue)
    selected=tune(arm,x,y,splits,stage)
    path=stage/'model.joblib';receipt=stage/'model_receipt.json'
    if path.is_file() and receipt.is_file():
        saved=read(receipt);assert saved['protocol_sha256']==sha(OUT/'protocol.json') and saved['model_sha256']==sha(path)
        model=joblib.load(path)
    else:
        model=calibrated(arm,selected['selected_C'],splits);model.fit(x,y);joblib.dump(model,path)
        write(receipt,dict(C=selected['selected_C'],model_sha256=sha(path),protocol_sha256=sha(OUT/'protocol.json')))
    # Development features/labels play no role in tuning or operating thresholds.
    with np.load(BASE/'development_scores.npz',allow_pickle=False) as data:development=data['features'].copy()
    assert development.shape==(2009,768) and np.isfinite(development).all()
    scores=model.predict_proba(development)[:,1]
    np.savez_compressed(folder/'development_predictions.npz',scores=scores,case_index=np.arange(2009))
    selection.update(full_model_C=selected['selected_C'],full_model_sha256=sha(path),
        training_selection_sha256=sha(folder/'training_selection.json'))
    write(folder/'selection.json',selection);return selection,scores


def process_receipt():
    import psutil
    parent=psutil.Process().parent();executable=Path(parent.exe()) if parent else None
    return dict(pid=os.getpid(),parent_pid=os.getppid(),parent_executable=str(executable),
        scheduler_owned=bool(executable and executable.name.lower() in ('svchost.exe','taskeng.exe','taskhostw.exe')))


def run():
    spec=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for p,digest in spec['source_sha256'].items():assert sha(p)==digest,p
    attempts=[]
    def blocked(*args,**kwargs):
        attempts.append('blocked_socket_connection');raise OSError('Offline experiment')
    socket.socket.connect=blocked;socket.socket.connect_ex=blocked;socket.create_connection=blocked
    queue=dict(status='running',started_utc=now(),CPU_only=True,threads=4,offline=True,
        no_automatic_following_experiment=True,protocol_sha256=sha(OUT/'protocol.json'),process=process_receipt())
    write(OUT/'queue.json',queue);started=time.monotonic()
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        warnings.simplefilter('error',ConvergenceWarning)
        rows,x,y,folds,groups=load_training();selections={};scores={}
        with threadpool_limits(limits=4):
            for arm in spec['arms']:
                selections[arm],scores[arm]=fit_arm(arm,spec,rows,x,y,folds,queue)
        plan=read(B.COMBINED/'input_plan.json');classes=[]
        assert len(plan['cases'])==2009 and len(plan['items'])==173
        for seed in B.SEEDS:
            with np.load(B.COMBINED/f'seed{seed}/original_joint224.npz',allow_pickle=False) as data:classes.append(data['predictions'].copy())
        results={}
        for pool in spec['development']['pools']:
            results[pool]={'off':B.evaluate_pool(plan,pool,np.ones(2009,dtype=bool),classes)}
            assert results[pool]['off']['candidates']==1071
            for arm in spec['arms']:
                results[pool][arm]={p['name']:B.evaluate_pool(plan,pool,scores[arm]>=p['threshold'],classes)
                    for p in selections[arm]['policies']}
        result=dict(status='completed_nested_spatial_binary_LR_SVM_comparison',created_utc=now(),
            selections=selections,development=results,CPU_seconds=time.monotonic()-started,
            protocol_sha256=sha(OUT/'protocol.json'),automatic_model_promotion=False,
            limitations=['Same pre-existing DINOv2 native224 features and expert types; no new image information.',
                'All three arms receive the same spatial sigmoid calibration and C search. The historical C1 uncalibrated LR is not the primary controlled comparator.',
                'Thresholds are fitted on training OOF scores and transferred to full-fit scores; retention is not guaranteed in development or new regions.',
                'One large spatial group can dominate an inner held fold; three folds are not three independent studies.',
                'Development regions have been used repeatedly; model-selected unmatched candidates may include missing labels.',
                'Existing localization matches retained. No rematching or AP recomputation.',
                'Three fixed subtype seeds are downstream readouts, not three binary-filter training seeds.',
                'CTX-only classifier comparison is not a full reproduction of the multimodal Mars-DiSVM paper.'])
        write(OUT/'comparison.json',result)
        lines=['# DINOv2二分类：LR／线性SVM／RBF-SVM','',
            '复用1083个正目标、1355个确认背景的768维原生224特征，沿用41空间组和3外折。',
            '三臂均在内层空间折选择C，使用相同空间分组的sigmoid分数校准；主要阈值要求训练OOF每类正例至少保留97%。','',
            '|分类器|外折平均ROC-AUC|合并OOF ROC-AUC|0.5处平衡正确率|逐类97%规则的训练背景排除率|FL保留|',
            '|---|---:|---:|---:|---:|---:|']
        for arm,s in selections.items():
            p=s['policies'][0];m=s['metrics'];fl=p['per_class']['FL']
            lines.append(f"|{arm}|{m['mean_fold_ROC_AUC']:.3f}|{m['pooled_ROC_AUC']:.3f}|{m['balanced_accuracy_at_half']:.2%}|{p['training_OOF_background_rejection']:.2%}|{fl['retained']}/{fl['count']}|")
        for policy_name in spec['threshold_policies']:
            lines += ['',f'## 实际候选：{policy_name}','',
                '|候选池|分类器|候选数|定位召回|参考精确率|独立定位／183|联合召回均值|独立正确均数／183|',
                '|---|---|---:|---:|---:|---:|---:|---:|']
            for pool,variants in results.items():
                for arm in ['off']+spec['arms']:
                    r=variants[arm] if arm=='off' else variants[arm][policy_name]
                    lines.append(f"|{pool}|{arm}|{r['candidates']}|{r['recall']:.2%}|{r['reference_precision']:.2%}|{r['unique_localized']}|{np.mean(r['joint_recall']):.2%}|{np.mean(r['unique_joint']):.2f}|")
        lines += ['', '训练OOF阈值留存是选阈值结果，不能解释为新区域的97%保证；实际候选表才反映本次迁移代价。',
            '冻结原定位配对，不重匹配、不重算AP；已反复使用的开发区域不是最终盲测。没有自动替换模型。',
            '三臂的本轮同协议结果相互比较；历史未校准C1逻辑回归有不同训练配方，不能将所有差异归因于分类器。',
            f'CPU总耗时：{result["CPU_seconds"]:.1f}秒。没有GPU训练、重新编码图像或读取最终测试像素。']
        (OUT/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        for p,digest in spec['source_sha256'].items():assert sha(p)==digest,p
        queue.update(status='completed',phase='finite_CPU_comparison_finished',finished_utc=now(),
            elapsed_seconds=time.monotonic()-started,blocked_network_attempts=attempts)
        write(OUT/'queue.json',queue)
        print(json.dumps(dict(status=queue['status'],seconds=queue['elapsed_seconds'],selected_C={a:s['full_model_C'] for a,s in selections.items()}),ensure_ascii=True),flush=True)
    except BaseException as exc:
        queue.update(status='failed',error=repr(exc),finished_utc=now());write(OUT/'queue.json',queue);raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def smoke():
    rng=np.random.default_rng(SEED);x=rng.normal(size=(120,8)).astype('float32')
    y=(x[:,0]+.5*x[:,1]>0).astype(int);groups=np.repeat(np.arange(12),10)
    cv=list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED).split(x,y,groups))
    with threadpool_limits(limits=4):
        for arm in ARMS:
            model=calibrated(arm,1.,cv);model.fit(x,y);p=model.predict_proba(x)[:,1]
            assert np.isfinite(p).all() and model.calibrated_classifiers_[0].estimator['scale'].n_samples_seen_==120
            print('SYNTHETIC_SMOKE',arm,float(roc_auc_score(y,p)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','prepare','run','scheduled-run']);args=parser.parse_args()
    if args.mode=='smoke':smoke()
    elif args.mode=='prepare':prepare()
    else:
        if args.mode=='scheduled-run':
            sys.stdout=(OUT/'training.log').open('a',encoding='utf-8',buffering=1)
            sys.stderr=(OUT/'training.err.log').open('a',encoding='utf-8',buffering=1)
        run()
