"""Second paired seed: retained augmentation, nbs64 versus nbs8; finite local job."""
from pathlib import Path
from copy import deepcopy
import argparse
import ast
import ctypes
import gc
import inspect
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.venv/Lib/site-packages'))
import run_detector_recipe_factorial_20261020 as R

P,F,torch=R.P,R.F,R.torch
PARENT=R.OUT
OUT=ROOT/'runs/experiments/detector_recipe_repeat_20261021_v1'
INITIALIZATION_SOURCE=ROOT/'runs/experiments/confirmation_20261016_v1/detector_seed1'
ARMS=['aug_on_nbs64','aug_on_nbs8']
TASK='MarsKG-detector-recipe-repeat-20261021-v1'


def effective_source():
    source=inspect.getsource(R.execute)
    before='init_seeds(0,deterministic=True)'
    after="init_seeds(spec['seed'],deterministic=True)"
    assert source.count(before)==1
    updated=source.replace(before,after)
    original_tree=ast.parse(source);updated_tree=ast.parse(updated)
    normal=deepcopy(updated_tree);changes=0
    for node in ast.walk(normal):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='init_seeds':
            assert ast.dump(node.args[0])==ast.dump(ast.parse("spec['seed']",mode='eval').body)
            node.args[0]=ast.Constant(value=0);changes+=1
    assert changes==1 and ast.dump(normal)==ast.dump(original_tree)
    return updated


def prepare():
    if OUT.exists():raise FileExistsError('Preserve existing run')
    parent=R.check()
    assert P.read(PARENT/'queue.json')['status']=='completed'
    result=P.read(PARENT/'comparison.json')
    assert result['protocol_sha256']==P.sha(PARENT/'protocol.json')
    for arm,row in result['results'].items():
        assert row['checkpoint_sha256']==P.sha(PARENT/'training'/arm/'train/weights/best.pt')
    initial=P.read(INITIALIZATION_SOURCE/'protocol.json')['models']['control']
    P.receipt_check(initial['checkpoint'],initial['checkpoint_sha256'])
    a=torch.load(initial['checkpoint'],map_location='cpu',weights_only=False)['model'].float()
    b=torch.load(parent['initialization']['checkpoint'],map_location='cpu',weights_only=False)['model'].float()
    assert P.state_hash(a.state_dict())==initial['full_state_sha256']
    assert P.state_hash(a.state_dict())!=P.state_hash(b.state_dict())
    assert P.state_hash(a.state_dict(),True)==P.state_hash(b.state_dict(),True)
    del a,b;gc.collect()
    OUT.mkdir(parents=True)
    for name in ['train.txt','val.txt','smoke_train.txt','smoke_val.txt']:
        (OUT/name).write_bytes((PARENT/name).read_bytes())
    for name in ['data.yaml','smoke.yaml']:
        text=(PARENT/name).read_text().replace(PARENT.as_posix(),OUT.as_posix())
        (OUT/name).write_text(text,encoding='utf-8')
    source=effective_source();(OUT/'effective_execute.py').write_text(source,encoding='utf-8')
    cfg={arm:deepcopy(parent['configs'][arm]) for arm in ARMS}
    for row in cfg.values():row['seed']=1
    assert {k for k in cfg[ARMS[0]] if cfg[ARMS[0]][k]!=cfg[ARMS[1]][k]}=={'nbs'}
    pins=dict(parent['files_sha256'])
    for path in [Path(__file__),ROOT/'data/scripts/start_detector_recipe_repeat_20261021.ps1',
        PARENT/'protocol.json',PARENT/'comparison.json',INITIALIZATION_SOURCE/'protocol.json',Path(initial['checkpoint'])]+list(OUT.glob('*.txt'))+list(OUT.glob('*.yaml'))+[OUT/'effective_execute.py']:
        pins[str(path)]=P.sha(path)
    spec=dict(status='locked_before_GPU_smoke',created_utc=R.utc(),seed=1,arms=ARMS,configs=cfg,
        initialization=initial,environment=P.environment(),files_sha256=pins,
        parent_experiment=str(PARENT),parent_protocol_sha256=P.sha(PARENT/'protocol.json'),
        parent_results_sha256=P.sha(PARENT/'comparison.json'),physical_batch=8,
        train_exposures=4860,train_unique=2371,val_images=173,val_boxes=560,val_unique=183,
        primary='Paired nbs64 minus nbs8 with augmentation retained, repeated at seed1; report both seeds separately',
        primary_metrics=['cached_AP50_95','unique_slides_at_at_most1071_candidates'],diagnostic_budgets=[300,600],
        execution_change='Only one AST-verified seed0 reset becomes spec[seed]; original runner and SDK stay read-only',
        new_training_count=2,max_epochs=60,patience=15,optimizer='SGD; unchanged per-arm learning-rate recipe',
        labels_modified=False,new_imagery_downloads=False,final_test_read=False,
        periodic_trigger=False,automatic_following_experiments=False,
        limitations=['Two seeds are still limited development evidence, not final blind-test accuracy.',
            'Seed0 nbs64 includes manual epoch-boundary recovery; SDK resumes EMA, not raw student/RNG replay.',
            'This repeats the retained-augmentation nbs contrast, not the entire augmentation-by-nbs interaction.',
            'Do not select a winner solely on AP if independent-slide recall moves the other way.'])
    R.write(OUT/'protocol.json',spec);(OUT/'protocol.sha256').write_text(P.sha(OUT/'protocol.json')+'\n')
    R.write(OUT/'queue.json',dict(status='prepared_not_started',task_name=TASK,total_arms=2,completed_arms=0))
    print('PREPARED_TWO_PAIRED_SEED1_ARMS',len(pins),flush=True)


def configure():
    spec=P.read(OUT/'protocol.json')
    P.receipt_check(OUT/'protocol.json',(OUT/'protocol.sha256').read_text().strip())
    for path,digest in spec['files_sha256'].items():P.receipt_check(path,digest)
    assert P.environment()==spec['environment']
    generated=(OUT/'effective_execute.py').read_text()
    assert generated==effective_source()
    # Process-local binding of the verified function; no archived source or SDK file is edited.
    R.OUT=OUT;R.TASK=TASK;R.ARMS=ARMS
    exec(compile(generated,str(OUT/'effective_execute.py'),'exec'),R.__dict__)
    return spec


def smoke(spec):
    records={}
    for arm in ARMS:
        q=P.read(OUT/'queue.json');q.update(phase='smoke',current_arm=arm);R.write(OUT/'queue.json',q)
        records[arm]=R.execute(spec,arm,smoke=True)
    first,second=[records[a] for a in ARMS]
    assert first['initial_model_sha256']==second['initial_model_sha256']==spec['initialization']['full_state_sha256']
    assert first['initial_batch_hashes']==second['initial_batch_hashes']
    seed0=P.read(PARENT/'smoke/aug_on_nbs64/audit.json')
    assert first['initial_batch_hashes']!=seed0['initial_batch_hashes'],'Seed was not passed to the data stream'
    assert first['actual_optimizer_steps']==2 and second['actual_optimizer_steps']==16
    resumed=R.execute(spec,ARMS[0],smoke=True,resume=True)
    assert resumed['resume_history'][-1]['verified_exact_checkpoint_restoration']
    R.write(OUT/'readiness.json',dict(status='passed_paired_seed1_input_update_BF16_and_resume_checks',
        seed1_data_stream_differs_from_seed0=True,within_seed_inputs_and_initialization_match=True,
        actual_first_epoch_optimizer_steps={a:r['actual_optimizer_steps'] for a,r in records.items()},
        resumed_second_epoch_verified=True,protocol_sha256=P.sha(OUT/'protocol.json'),quality_result=False))


def summarize(spec):
    prior=P.read(PARENT/'comparison.json')['results']
    current={a:P.read(OUT/'evaluation'/a/'metrics.json') for a in ARMS}
    paired={}
    for seed,results in [(0,prior),(1,current)]:
        left=results[ARMS[0]]['budgets']['1071'];right=results[ARMS[1]]['budgets']['1071']
        paired[str(seed)]=dict(AP50_95_delta=left['metrics']['cached_map50_95']-right['metrics']['cached_map50_95'],
            AP50_delta=left['metrics']['cached_map50']-right['metrics']['cached_map50'],
            unique_slide_delta=left['unique_recalled']-right['unique_recalled'],
            candidate_counts=[left['metrics']['candidates'],right['metrics']['candidates']])
    R.write(OUT/'comparison.json',dict(status='completed_two_seed_augmented_nbs_comparison',
        results_by_seed={'0':{a:prior[a] for a in ARMS},'1':current},paired_nbs64_minus8=paired,
        protocol_sha256=P.sha(OUT/'protocol.json'),final_test_read=False))


def scheduled(resume=False):
    spec=configure();owned=F.process_independence();assert owned['scheduler_owned']
    F.ensure_idle(OUT);blocked=F.prevent_network();torch.set_num_threads(2)
    assert torch.cuda.mem_get_info()[0]>4*2**30
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    queue=dict(status='running',started_utc=R.utc(),pid=os.getpid(),task_name=TASK,seed=1,
        process_independence=owned,offline=True,temporary_system_awake=True,
        completed_arms=0,total_arms=2,phase='smoke')
    R.write(OUT/'queue.json',queue)
    try:
        if not resume:smoke(spec)
        assert P.read(OUT/'readiness.json')['protocol_sha256']==P.sha(OUT/'protocol.json')
        for arm in ARMS:
            path=OUT/'training'/arm/'audit.json'
            queue.update(current_arm=arm,phase='training');R.write(OUT/'queue.json',queue)
            if not path.exists() or P.read(path)['status']!='complete':R.execute(spec,arm,resume=path.exists())
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
