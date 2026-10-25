"""Finite seed0 optimizer-recipe comparison with matched warmup and data."""
from pathlib import Path
from copy import deepcopy
import argparse
import ctypes
import gc
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.venv/Lib/site-packages'))
import run_detector_recipe_factorial_20261020 as R

P,F,torch=R.P,R.F,R.torch
PARENT=R.OUT
REPEAT=ROOT/'runs/experiments/detector_recipe_repeat_20261021_v1'
OUT=ROOT/'runs/experiments/detector_optimizer_20261022_v1'
TASK='MarsKG-detector-optimizer-20261022-v1'
ARMS=['adamw_lr3e4','sgd_lr1e2','adamw_lr1e3']


def prepare():
    if OUT.exists():raise FileExistsError('Preserve existing run')
    parent=R.check()
    assert P.read(PARENT/'queue.json')['status']=='completed'
    assert P.read(REPEAT/'queue.json')['status']=='completed'
    repeated=P.read(REPEAT/'comparison.json')
    P.receipt_check(REPEAT/'protocol.json',repeated['protocol_sha256'])
    # Check both seed result/weight bindings before retaining the reference recipe.
    for seed,results in repeated['results_by_seed'].items():
        base=PARENT if seed=='0' else REPEAT
        for arm,row in results.items():
            P.receipt_check(base/'training'/arm/'train/weights/best.pt',row['checkpoint_sha256'])
    cfg={a:deepcopy(parent['configs']['aug_on_nbs64']) for a in ARMS}
    for a in ARMS:cfg[a]['warmup_bias_lr']=0.0
    cfg[ARMS[0]].update(optimizer='AdamW',lr0=.0003)
    cfg[ARMS[1]].update(optimizer='SGD',lr0=.01)
    cfg[ARMS[2]].update(optimizer='AdamW',lr0=.001)
    for a in [ARMS[0],ARMS[2]]:
        assert {k for k in cfg[a] if cfg[a][k]!=cfg[ARMS[1]][k]}=={'optimizer','lr0'}
    assert {k for k in cfg[ARMS[0]] if cfg[ARMS[0]][k]!=cfg[ARMS[2]][k]}=={'lr0'}
    assert {k for k in cfg[ARMS[1]] if cfg[ARMS[1]][k]!=parent['configs']['aug_on_nbs64'][k]}=={'warmup_bias_lr'}
    OUT.mkdir(parents=True)
    for name in ['train.txt','val.txt','smoke_train.txt','smoke_val.txt']:(OUT/name).write_bytes((PARENT/name).read_bytes())
    for name in ['data.yaml','smoke.yaml']:
        (OUT/name).write_text((PARENT/name).read_text().replace(PARENT.as_posix(),OUT.as_posix()),encoding='utf-8')
    pins=dict(parent['files_sha256'])
    for path in [Path(__file__),ROOT/'data/scripts/start_detector_optimizer_20261022.ps1',
        PARENT/'protocol.json',PARENT/'comparison.json',REPEAT/'protocol.json',REPEAT/'comparison.json']+list(OUT.glob('*.txt'))+list(OUT.glob('*.yaml')):
        pins[str(path)]=P.sha(path)
    spec=dict(status='locked_before_gpu_smoke',created_utc=R.utc(),seed=0,arms=ARMS,configs=cfg,
        initialization=parent['initialization'],environment=P.environment(),files_sha256=pins,
        physical_batch=8,nbs=64,train_exposures=4860,train_unique=2371,val_images=173,val_boxes=560,val_unique=183,
        primary='Two predeclared AdamW learning rates versus simultaneous matched-warmup SGD; exploratory configuration comparison',
        differences=['optimizer and optimizer-specific lr0; remaining contemporaneous configs identical',
            'All new arms start bias warmup at0, unlike the archived SGD reference0.1',
            'The momentum argument0.937 means Nesterov SGD momentum or AdamW beta1; beta2 is SDK0.999',
            'SGD uses coupled regularization, AdamW uses decoupled decay with the same configured weight_decay0.0005'],
        optimizer_auto_used=False,numerical_runner_modified=False,
        budgets=[300,600,1071],selection='best EMA development AP50-95; max60/patience15',
        primary_metrics=['AP50-95','unique_slides_at_at_most1071_candidates'],
        imagery_downloads=0,labels_modified=False,final_test_read=False,
        automatic_following_experiments=False,periodic_trigger=False,
        limitations=['Single seed with two AdamW LR candidates; do not claim a pure optimizer algorithm advantage or blind-test gain.',
            'If promising, repeat the selected configuration against paired SGD at a new seed before replacement.',
            'Epoch checkpoint restores SDK quantized EMA/optimizer, not raw student/RNG/partial gradients.'])
    R.write(OUT/'protocol.json',spec);(OUT/'protocol.sha256').write_text(P.sha(OUT/'protocol.json')+'\n')
    R.write(OUT/'queue.json',dict(status='prepared_not_started',task_name=TASK,total_arms=3,completed_arms=0))
    print('PREPARED_OPTIMIZER_THREE_ARMS',len(pins),flush=True)


def configure():
    spec=P.read(OUT/'protocol.json')
    P.receipt_check(OUT/'protocol.json',(OUT/'protocol.sha256').read_text().strip())
    for path,digest in spec['files_sha256'].items():P.receipt_check(path,digest)
    assert P.environment()==spec['environment']
    R.OUT=OUT;R.TASK=TASK;R.ARMS=ARMS
    return spec


def optimizer_receipt(spec,arm,smoke=False):
    path=OUT/('smoke' if smoke else 'training')/arm
    ck=torch.load(path/'recovery_last.pt',map_location='cpu',weights_only=False)
    groups=ck['optimizer']['param_groups'];cfg=spec['configs'][arm]
    assert ck['train_args']['optimizer']==cfg['optimizer']
    assert ck['train_args']['lr0']==cfg['lr0'] and ck['train_args']['warmup_bias_lr']==0
    assert ck['train_args']['nbs']==64
    if cfg['optimizer']=='AdamW':assert all(tuple(g['betas'])==(.937,.999) for g in groups)
    else:assert all('betas' not in g and abs(g['momentum']-.937)<1e-12 for g in groups)
    actual=[{k:g[k] for k in ['lr','initial_lr','momentum','betas','weight_decay','nesterov','param_group'] if k in g} for g in groups]
    R.write(path/'optimizer_receipt.json',dict(optimizer=cfg['optimizer'],lr0=cfg['lr0'],groups=actual,
        checkpoint_sha256=P.sha(path/'recovery_last.pt'),actual_explicit_configuration_verified=True))
    del ck;gc.collect()


def smoke(spec):
    records={}
    for arm in ARMS:
        q=P.read(OUT/'queue.json');q.update(phase='smoke',current_arm=arm);R.write(OUT/'queue.json',q)
        records[arm]=R.execute(spec,arm,smoke=True);optimizer_receipt(spec,arm,smoke=True)
    assert len({r['initial_model_sha256'] for r in records.values()})==1
    assert all(r['initial_batch_hashes']==records[ARMS[0]]['initial_batch_hashes'] for r in records.values())
    assert all(r['actual_optimizer_steps']==2 for r in records.values())
    resumed=R.execute(spec,ARMS[0],smoke=True,resume=True)
    assert resumed['resume_history'][-1]['verified_exact_checkpoint_restoration']
    optimizer_receipt(spec,ARMS[0],smoke=True)
    R.write(OUT/'readiness.json',dict(status='passed_three_actual_optimizers_LR_BF16_paired_input_and_AdamW_resume',
        within_seed_initialization_and_input_equal=True,actual_initial_optimizer_steps={a:r['actual_optimizer_steps'] for a,r in records.items()},
        AdamW_saved_optimizer_state_restore_verified=True,protocol_sha256=P.sha(OUT/'protocol.json'),quality_result=False))


def summarize(spec):
    current={a:P.read(OUT/'evaluation'/a/'metrics.json') for a in ARMS}
    legacy=P.read(PARENT/'comparison.json')['results']['aug_on_nbs64']
    contrasts={}
    for a in [ARMS[0],ARMS[2]]:
        x=current[a]['budgets']['1071'];y=current[ARMS[1]]['budgets']['1071']
        contrasts[a]=dict(AP50_95_delta=x['metrics']['cached_map50_95']-y['metrics']['cached_map50_95'],
            AP50_delta=x['metrics']['cached_map50']-y['metrics']['cached_map50'],
            unique_slide_delta=x['unique_recalled']-y['unique_recalled'])
    R.write(OUT/'comparison.json',dict(status='completed_single_seed_optimizer_recipe_comparison',
        results=current,AdamW_minus_matched_SGD=contrasts,historical_SGD_bias_warmup_0p1=legacy,
        protocol_sha256=P.sha(OUT/'protocol.json'),final_test_read=False,pure_optimizer_algorithm_claim=False))


def scheduled(resume=False):
    spec=configure();owned=F.process_independence();assert owned['scheduler_owned']
    F.ensure_idle(OUT);blocked=F.prevent_network();torch.set_num_threads(2)
    assert torch.cuda.mem_get_info()[0]>4*2**30
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    queue=dict(status='running',started_utc=R.utc(),pid=__import__('os').getpid(),task_name=TASK,seed=0,
        process_independence=owned,offline=True,temporary_system_awake=True,completed_arms=0,total_arms=3,phase='smoke')
    R.write(OUT/'queue.json',queue)
    try:
        if not resume:smoke(spec)
        assert P.read(OUT/'readiness.json')['protocol_sha256']==P.sha(OUT/'protocol.json')
        for arm in ARMS:
            path=OUT/'training'/arm/'audit.json'
            queue.update(current_arm=arm,phase='training');R.write(OUT/'queue.json',queue)
            if not path.exists() or P.read(path)['status']!='complete':R.execute(spec,arm,resume=path.exists())
            optimizer_receipt(spec,arm)
            if not (OUT/'evaluation'/arm/'metrics.json').exists():R.evaluate(arm)
            queue.update(P.read(OUT/'queue.json'));queue['completed_arms']+=1;R.write(OUT/'queue.json',queue)
        summarize(spec);queue.update(status='completed',phase='complete')
    except BaseException as e:queue.update(status='failed',error=repr(e));raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        queue.update(finished_utc=R.utc(),temporary_system_awake=False,blocked_network_attempts=blocked)
        R.write(OUT/'queue.json',queue)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','scheduled-run','scheduled-resume'])
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    else:
        stamp=str(int(time.time()));sys.stdout=(OUT/f'training_{stamp}.log').open('a',encoding='utf-8',buffering=1)
        sys.stderr=(OUT/f'training_{stamp}.err.log').open('a',encoding='utf-8',buffering=1)
        scheduled(args.mode=='scheduled-resume')
