"""Second detector seed: matched-warmup AdamW3e-4 versus SGD1e-2, finite pair."""
from copy import deepcopy
import ctypes
import gc
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.venv/Lib/site-packages'))
import run_detector_optimizer_20261022 as O
import run_detector_recipe_repeat_20261021 as S
import continue_detector_optimizer_receipt_20261022 as FIX

R,P,F,torch=O.R,O.P,O.F,O.torch
SOURCE=ROOT/'runs/experiments/detector_optimizer_20261022_v1'
OUT=ROOT/'runs/experiments/detector_optimizer_repeat_20261024_v1'
TASK='MarsKG-detector-optimizer-repeat-20261024-v1'
ARMS=['adamw_lr3e4','sgd_lr1e2']
BRIDGE=ROOT/'runs/experiments/latest_detector_classifier_bridge_20261024_v1'


def prepare():
    assert not OUT.exists(),'Preserve existing repeat'
    parent=P.read(SOURCE/'protocol.json')
    P.receipt_check(SOURCE/'protocol.json',(SOURCE/'protocol.sha256').read_text().strip())
    for path,digest in parent['files_sha256'].items():P.receipt_check(path,digest)
    assert P.read(SOURCE/'queue.json')['status']=='completed'
    proof=P.read(BRIDGE/'independent_completion_verification.json')
    assert proof['score_cells_verified']==270 and proof['comparison_sha256']==P.sha(BRIDGE/'comparison.json')
    initial=P.read(S.INITIALIZATION_SOURCE/'protocol.json')['models']['control']
    P.receipt_check(initial['checkpoint'],initial['checkpoint_sha256'])
    original=torch.load(parent['initialization']['checkpoint'],map_location='cpu',weights_only=False)['model'].float()
    repeated=torch.load(initial['checkpoint'],map_location='cpu',weights_only=False)['model'].float()
    assert P.state_hash(repeated.state_dict())==initial['full_state_sha256']
    assert P.state_hash(repeated.state_dict())!=P.state_hash(original.state_dict())
    assert P.state_hash(repeated.state_dict(),True)==P.state_hash(original.state_dict(),True)
    del original,repeated;gc.collect()
    OUT.mkdir(parents=True)
    for name in ['train.txt','val.txt','smoke_train.txt','smoke_val.txt']:(OUT/name).write_bytes((SOURCE/name).read_bytes())
    for name in ['data.yaml','smoke.yaml']:
        (OUT/name).write_text((SOURCE/name).read_text().replace(SOURCE.as_posix(),OUT.as_posix()),encoding='utf-8')
    generated=S.effective_source();(OUT/'effective_execute.py').write_text(generated,encoding='utf-8')
    configs={arm:deepcopy(parent['configs'][arm]) for arm in ARMS}
    for cfg in configs.values():cfg['seed']=1
    assert {k for k in configs[ARMS[0]] if configs[ARMS[0]][k]!=configs[ARMS[1]][k]}=={'optimizer','lr0'}
    pins=dict(parent['files_sha256'])
    for p in [Path(__file__),ROOT/'data/scripts/start_detector_optimizer_repeat_20261024.ps1',
              Path(FIX.__file__),Path(S.__file__),SOURCE/'protocol.json',SOURCE/'comparison.json',
              BRIDGE/'protocol.json',BRIDGE/'comparison.json',BRIDGE/'independent_completion_verification.json',
              S.INITIALIZATION_SOURCE/'protocol.json',Path(initial['checkpoint'])]+list(OUT.glob('*.txt'))+list(OUT.glob('*.yaml'))+[OUT/'effective_execute.py']:
        pins[str(p)]=P.sha(p)
    spec=dict(status='locked_before_two_arm_GPU_smoke',created_utc=R.utc(),seed=1,arms=ARMS,configs=configs,
              initialization=initial,environment=P.environment(),files_sha256=pins,physical_batch=8,nbs=64,
              train_exposures=4860,train_unique=2371,val_images=173,val_boxes=560,val_unique=183,
              selected_after_reading_development_bridge=True,
              selection_reason='Both AdamW rates had67.67 mean jointly correct IDs versus SGD64.33; choose lower LR3e-4 with higher300-candidate coverage; exploratory choice, not known optimal',
              purpose='Repeat chosen AdamW recipe against simultaneous matched-warmup SGD at a different detector initialization and data seed',
              classifier_evaluation_later='Same existing DINO three seeds and fixed300/600/1071 budgets; no ratio or seed selection',
              common_recipe='YOLOv8n full pretrained state, augmentation on, batch8 nbs64 BF16 workers2; max60 patience15; warmup_bias_lr0',
              primary='Report both detector seeds separately; detection AP plus eventual localization-and-type correct unique IDs',
              original_seed0_AdamW_interrupted_after_epoch25=True,
              new_training_arms=2,labels_modified=False,new_catalog_dataset_imported=False,
              final_test_read=False,automatic_following_experiments=False,periodic_trigger=False,
              execution_change='Same previously AST-verified one-line spec seed injection; numerical runner and SDK are read-only',
              limitations=['Selection follows repeatedly used development results; this is not a blind confirmation',
                           'One chosen AdamW learning rate, not a general optimizer theorem',
                           'Seed0 AdamW includes an EMA checkpoint resume, unlike new planned uninterrupted pair',
                           'Joint classification scores are not calculated from manual ROIs',
                           'New global human/SHP export runs independently and cannot alter frozen2371 training images'])
    R.write(OUT/'protocol.json',spec);(OUT/'protocol.sha256').write_text(P.sha(OUT/'protocol.json')+'\n')
    R.write(OUT/'queue.json',dict(status='prepared_not_started',seed=1,total_arms=2,completed_arms=0))
    print('PREPARED_MATCHED_OPTIMIZER_SEED1_REPEAT',len(pins),'pinned inputs',flush=True)


def configure():
    spec=P.read(OUT/'protocol.json');P.receipt_check(OUT/'protocol.json',(OUT/'protocol.sha256').read_text().strip())
    for path,digest in spec['files_sha256'].items():P.receipt_check(path,digest)
    assert P.environment()==spec['environment']
    generated=(OUT/'effective_execute.py').read_text();assert generated==S.effective_source()
    R.OUT=OUT;R.TASK=TASK;R.ARMS=ARMS
    O.OUT=OUT;O.ARMS=ARMS
    exec(compile(generated,str(OUT/'effective_execute.py'),'exec'),R.__dict__)
    return spec


def smoke(spec):
    records={}
    for arm in ARMS:
        state=P.read(OUT/'queue.json');state.update(phase='smoke',current_arm=arm);R.write(OUT/'queue.json',state)
        records[arm]=R.execute(spec,arm,smoke=True)
        FIX.checked_receipt(spec,arm,smoke=True)
    assert all(r['initial_model_sha256']==spec['initialization']['full_state_sha256'] for r in records.values())
    assert records[ARMS[0]]['initial_batch_hashes']==records[ARMS[1]]['initial_batch_hashes']
    assert all(r['actual_optimizer_steps']==2 for r in records.values())
    seed0=P.read(SOURCE/'smoke/adamw_lr3e4/audit.json')
    assert records[ARMS[0]]['initial_batch_hashes']!=seed0['initial_batch_hashes']
    resumed=R.execute(spec,ARMS[0],smoke=True,resume=True)
    assert resumed['resume_history'][-1]['verified_exact_checkpoint_restoration']
    FIX.checked_receipt(spec,ARMS[0],smoke=True)
    R.write(OUT/'readiness.json',dict(status='passed_two_seed1_paired_optimizer_input_BF16_and_AdamW_restore',
                                     within_seed_initialization_and_inputs_match=True,new_seed_inputs_differ_from_source=True,
                                     actual_optimizer_steps={arm:r['actual_optimizer_steps'] for arm,r in records.items()},
                                     resume_checkpoint_verified=True,protocol_sha256=P.sha(OUT/'protocol.json'),quality_result=False))


def summarize(spec):
    previous=P.read(SOURCE/'comparison.json')['results'];current={arm:P.read(OUT/'evaluation'/arm/'metrics.json') for arm in ARMS}
    diffs={}
    for seed,rows in [('0',previous),('1',current)]:
        a,b=[rows[arm]['budgets']['1071'] for arm in ARMS]
        diffs[seed]=dict(AP50_95=a['metrics']['cached_map50_95']-b['metrics']['cached_map50_95'],
                         AP50=a['metrics']['cached_map50']-b['metrics']['cached_map50'],
                         unique_detected=a['unique_recalled']-b['unique_recalled'])
    R.write(OUT/'comparison.json',dict(status='completed_two_seed_selected_optimizer_recipe_repeat',
                                      results_by_seed={'0':{a:previous[a] for a in ARMS},'1':current},
                                      AdamW_minus_SGD=diffs,protocol_sha256=P.sha(OUT/'protocol.json'),
                                      final_test_read=False,joint_classifier_evaluation_pending=True))


def scheduled(resume=False):
    spec=configure();owned=F.process_independence();assert owned['scheduler_owned']
    F.ensure_idle(OUT);blocked=F.prevent_network();torch.set_num_threads(2)
    assert torch.cuda.mem_get_info()[0]>4*2**30,'Insufficient free GPU memory'
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    queue=dict(status='running',started_utc=R.utc(),pid=os.getpid(),task_name=TASK,seed=1,
               process_independence=owned,offline=True,temporary_system_awake=True,completed_arms=0,total_arms=2,phase='smoke')
    R.write(OUT/'queue.json',queue)
    try:
        if not resume:smoke(spec)
        assert P.read(OUT/'readiness.json')['protocol_sha256']==P.sha(OUT/'protocol.json')
        for arm in ARMS:
            path=OUT/'training'/arm/'audit.json';queue.update(current_arm=arm,phase='training');R.write(OUT/'queue.json',queue)
            if not path.exists() or P.read(path)['status']!='complete':R.execute(spec,arm,resume=path.exists())
            FIX.checked_receipt(spec,arm)
            if not (OUT/'evaluation'/arm/'metrics.json').exists():R.evaluate(arm)
            queue.update(P.read(OUT/'queue.json'));queue['completed_arms']+=1;R.write(OUT/'queue.json',queue)
        summarize(spec);queue.update(status='completed',phase='complete')
    except BaseException as error:
        queue.update(status='failed_resumable',error=repr(error));raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        queue.update(finished_utc=R.utc(),temporary_system_awake=False,blocked_network_attempts=blocked);R.write(OUT/'queue.json',queue)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        stamp=str(int(time.time()))
        with (OUT/f'training_{stamp}.log').open('a',encoding='utf-8',buffering=1) as log,(OUT/f'training_{stamp}.err.log').open('a',encoding='utf-8',buffering=1) as err:
            sys.stdout=log;sys.stderr=err
            try:scheduled(sys.argv[1]=='scheduled-resume')
            except BaseException:traceback.print_exc(file=err);raise
