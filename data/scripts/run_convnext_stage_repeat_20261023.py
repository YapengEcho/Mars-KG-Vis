"""Two additional paired ConvNeXt seeds; frozen first seed is reused read-only."""
from copy import deepcopy
import ctypes
import os
from pathlib import Path
import sys
import time

import run_convnext_stage_adapt_20261022 as S

N,M,C,np,torch=S.N,S.M,S.C,S.np,S.torch
ROOT=S.ROOT
SOURCE=S.OUT
OUT=ROOT/'runs/experiments/convnext_stage_repeat_20261023_v1'
TASK='MarsKG-convnext-stage-repeat-20261023-v1'
NEW_SEEDS=[20261016,20261017]
ALL_SEEDS=[20261015]+NEW_SEEDS


def bind():
    S.OUT=OUT;S.SEED=NEW_SEEDS[0]
    M.OUT=OUT;M.Classifier=S.Classifier;M.write=S.write;M.weights=N.H.weights


def prepare():
    assert not OUT.exists(), 'Preserve existing repeat experiment'
    parent=S.check()
    proof=C.read_json(SOURCE/'independent_verification.json')
    assert proof['status']=='passed_saved_selection_and_actual_refit_weights_GPU_replay'
    assert proof['protocol_sha256']==C.sha256(SOURCE/'protocol.json')
    assert proof['refit_weight_predictions_bitwise_equal']==6
    assert C.read_json(SOURCE/'queue.json')['status']=='completed'
    spec=deepcopy(parent)
    spec.update(created_utc=M.utc(),seed=NEW_SEEDS[0],seeds=ALL_SEEDS,new_seeds=NEW_SEEDS,
        exploratory_seed_count=3,source_seed_reused_read_only=20261015,
        repeat_policy='Same data, splits, initialization, hyperparameters and stopping; only optimization seed varies',
        primary='Paired last_stage minus head_only OOF macro-F1 for three seeds; report class tradeoffs',
        automatic_following_experiments=False)
    spec['limitations']=[x for x in parent['limitations'] if not x.startswith('Single seed')]+[
        'Two repeats scheduled after observing first seed; development replication, not an independent blind test',
        'Three seeds share the same spatial folds; seed variation does not estimate geographic uncertainty',
        'FL remains 15 targets from few geographic groups; report integer correct counts and precision too']
    for p in [SOURCE/'protocol.json',SOURCE/'comparison.json',SOURCE/'independent_verification.json']:
        spec['files_sha256'][str(p)]=C.sha256(p)
    spec['files_sha256'].update(proof['checkpoint_hashes'])
    for p in [Path(__file__),ROOT/'data/scripts/start_convnext_stage_repeat_20261023.ps1']:
        spec['code_hashes'][str(p)]=C.sha256(p)
    OUT.mkdir(parents=True)
    S.write(OUT/'protocol.json',spec)
    (OUT/'protocol.sha256').write_text(C.sha256(OUT/'protocol.json')+'\n')
    S.write(OUT/'queue.json',dict(status='prepared_not_started',completed_units=0,total_units=12,new_seeds=NEW_SEEDS))
    bind();S.check()
    print('PREPARED_TWO_PAIRED_REPEATS',flush=True)


def summarize(spec,data):
    old=C.read_json(SOURCE/'comparison.json')
    per_seed={arm:{str(20261015):old['results'][arm]} for arm in S.ARMS}
    # Recompute the first seed's OOF instead of trusting only its summary.
    M.OUT=SOURCE
    try:
        for arm in S.ARMS:
            assert M._arm_seed_metrics(data,arm,20261015,spec['design']['outer_folds'])==old['results'][arm]
    finally:M.OUT=OUT
    for arm in S.ARMS:
        for seed in NEW_SEEDS:
            per_seed[arm][str(seed)]=M._arm_seed_metrics(data,arm,seed,spec['design']['outer_folds'])
    aggregate={}
    for arm in S.ARMS:
        rows=list(per_seed[arm].values())
        aggregate[arm]=dict(macro_f1_mean=float(np.mean([r['macro_f1'] for r in rows])),
            macro_f1_std=float(np.std([r['macro_f1'] for r in rows])),
            accuracy_mean=float(np.mean([r['accuracy'] for r in rows])),
            per_class={name:{metric:dict(mean=float(np.mean([r['per_class'][name][metric] for r in rows])),
                std=float(np.std([r['per_class'][name][metric] for r in rows])))
                for metric in ['precision','recall','f1-score']} for name in M.NAMES})
    delta=[per_seed['last_stage'][str(seed)]['macro_f1']-per_seed['head_only'][str(seed)]['macro_f1'] for seed in ALL_SEEDS]
    result=dict(status='completed_three_seed_ConvNeXt_paired_replication',seeds=ALL_SEEDS,
        per_seed=per_seed,aggregate=aggregate,paired_delta=dict(values=delta,mean=float(np.mean(delta)),
        std=float(np.std(delta)),positive_seeds=sum(d>0 for d in delta)),std_ddof=0,
        targets=1083,spatial_groups=40,final_test_read=False,source_seed_reused_read_only=True,
        protocol_sha256=C.sha256(OUT/'protocol.json'))
    S.write(OUT/'comparison.json',result)
    lines=['# ConvNeXt三种子配对重复','',
        '同一1083目标、40供图空间组、3折嵌套。首种子复用已核验结果，另补两个种子；不读取最终测试集。','',
        '|种子|只训头macro-F1|微调末级macro-F1|配对差|','|---|---:|---:|---:|']
    for seed,d in zip(ALL_SEEDS,delta):
        lines.append(f'|{seed}|{per_seed["head_only"][str(seed)]["macro_f1"]:.2%}|{per_seed["last_stage"][str(seed)]["macro_f1"]:.2%}|{d:+.2%}|')
    lines+=['','种子共享空间划分，标准差仅表示优化种子波动；重复是在看到首种子后追加，不能视为独立盲测。',
        '完整逐类precision/recall/F1、逐折与混淆矩阵见comparison.json。FL仅15个目标，需同时查看召回与误判。']
    (OUT/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def scheduled(resume=False):
    bind();spec=S.check();state=C.read_json(OUT/'queue.json')
    assert state['status'] in (['failed','paused_by_user'] if resume else ['prepared_not_started'])
    assert spec['new_seeds']==NEW_SEEDS and spec['seeds']==ALL_SEEDS
    independent=M.FENV.process_independence();assert independent['scheduler_owned']
    M.P.R.ensure_gpu_idle();blocked=M.FENV.prevent_network();M.device_setup();torch.set_num_threads(2)
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    queue=dict(status='running',pid=os.getpid(),started_utc=state.get('started_utc',M.utc()),phase='smoke',
        completed_units=0,total_units=12,new_seeds=NEW_SEEDS,process_independence=independent,
        offline=True,temporary_system_awake=True)
    S.write(OUT/'queue.json',queue)
    try:
        data=S.load(spec)
        if not resume:
            S.smoke(spec,data)
            split=spec['design']['outer_folds'][0];fit=split['inner_fit_indices'];head_hashes=[]
            for seed in NEW_SEEDS:
                model=S.Classifier(spec,'head_only',fit,data,0,'select',seed)
                head_hashes.append(M.tensor_hash(model.head.state_dict()));del model
            assert len(set(head_hashes))==2,'Seeds must change actual initialization'
            S.write(OUT/'seed_initialization_check.json',dict(seeds=NEW_SEEDS,head_hashes=head_hashes,passed=True))
        assert C.read_json(OUT/'readiness.json')['protocol_sha256']==C.sha256(OUT/'protocol.json')
        assert C.read_json(OUT/'seed_initialization_check.json')['passed']
        for seed in NEW_SEEDS:
            for split in spec['design']['outer_folds']:
                for phase in ['select','refit']:
                    for arm in S.ARMS:M.stage(spec,data,split,arm,phase,queue,seed)
                queue['completed_units']+=2;S.write(OUT/'queue.json',queue)
        S.check();summarize(spec,data);queue.update(status='completed',phase='complete')
    except BaseException as error:queue.update(status='failed',error=repr(error));raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        queue.update(finished_utc=M.utc(),temporary_system_awake=False,blocked_network_attempts=blocked)
        S.write(OUT/'queue.json',queue)


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    elif mode in ['scheduled-run','scheduled-resume']:
        stamp=str(time.time_ns())
        sys.stdout=(OUT/f'training_{stamp}.log').open('x',encoding='utf-8',buffering=1)
        sys.stderr=(OUT/f'training_{stamp}.err.log').open('x',encoding='utf-8',buffering=1)
        scheduled(mode=='scheduled-resume')
    else:raise ValueError(mode)
