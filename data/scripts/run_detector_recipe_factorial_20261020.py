"""Finite 2x2 YOLO recipe study: augmentation package x nominal update batch."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import ctypes
import gc
import json
import os
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.venv/Lib/site-packages'))
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
    CUBLAS_WORKSPACE_CONFIG=':4096:8',YOLO_OFFLINE='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import init_seeds
import probe_p2_detector as P
import run_full_pretrained_detector as F

OUT=ROOT/'runs/experiments/detector_recipe_factorial_20261020_v1'
SOURCE=ROOT/'runs/experiments/detector_context_20261015_v1'
TASK='MarsKG-detector-recipe-factorial-20261020-v1'
ARMS=['aug_on_nbs64','aug_off_nbs64','aug_on_nbs8','aug_off_nbs8']


def utc():return datetime.now(timezone.utc).isoformat()


def write(path,value):
    path=Path(path)
    blob=json.dumps(value,indent=2,allow_nan=False)+'\n'
    for attempt in range(8):
        temporary=path.with_suffix('.writing.json')
        try:
            temporary.write_text(blob,encoding='utf-8');temporary.replace(path);return
        except PermissionError:
            if attempt==7:raise
            time.sleep(.1*(attempt+1))


def prepare():
    if OUT.exists():raise FileExistsError('Preserve existing experiment')
    source=P.read(SOURCE/'protocol.json');receipt=P.read(SOURCE/'source_receipt.json')
    P.receipt_check(SOURCE/'protocol.json',(SOURCE/'protocol.sha256').read_text().strip())
    P.receipt_check(SOURCE/'source_receipt.json',source['source_receipt_sha256'])
    entry=source['models']['control'];P.receipt_check(entry['checkpoint'],entry['checkpoint_sha256'])
    OUT.mkdir(parents=True)
    train_source=ROOT/'runs/experiments/p2_detector_probe_20261013_v2/dataset/train.txt'
    train=train_source.read_text().splitlines();assert len(train)==4860 and len(set(train))==2371
    # Original aspect-ratio images in every arm: this restores the old input pipeline.
    # No prepared-square training images or additional positive exposures are used.
    (OUT/'train.txt').write_text('\n'.join(train)+'\n',encoding='utf-8')
    import evaluate_p2_validation_comparison as V
    items=V.validate_validation_inputs()
    (OUT/'val.txt').write_text('\n'.join(r['image'] for r in items)+'\n',encoding='utf-8')
    smoke_train=[r['image'] for r in receipt['smoke_inputs'] if r['split']=='train']
    smoke_val=[r['image'] for r in receipt['smoke_inputs'] if r['split']=='val']
    assert len(smoke_train)==16 and len(smoke_val)==8
    (OUT/'smoke_train.txt').write_text('\n'.join(smoke_train*8)+'\n',encoding='utf-8')
    (OUT/'smoke_val.txt').write_text('\n'.join(smoke_val)+'\n',encoding='utf-8')
    for mode in ['data','smoke']:
        prefix='' if mode=='data' else 'smoke_'
        (OUT/f'{mode}.yaml').write_text(f"path: {OUT.as_posix()}\ntrain: {(OUT/(prefix+'train.txt')).as_posix()}\nval: {(OUT/(prefix+'val.txt')).as_posix()}\nnames:\n  0: LANDSLIDE\n",encoding='utf-8')
    base=deepcopy(source['training'])
    for key in ['project','exist_ok']:base.pop(key,None)
    base.update(epochs=60,patience=15,workers=2,seed=0,plots=False,amp='bf16',deterministic=True)
    configs={}
    for arm in ARMS:
        cfg=deepcopy(base);cfg['nbs']=64 if arm.endswith('64') else 8
        if arm.startswith('aug_off'):
            cfg.update(hsv_h=0.,hsv_s=0.,hsv_v=0.,translate=0.,scale=0.,fliplr=0.,mosaic=0.,close_mosaic=0)
        configs[arm]=cfg
    for left,right in [('aug_on_nbs64','aug_on_nbs8'),('aug_off_nbs64','aug_off_nbs8')]:
        assert {k for k in configs[left] if configs[left][k]!=configs[right][k]}=={'nbs'}
    pins={str(Path(p)):P.sha(p) for p in [Path(__file__),ROOT/'data/scripts/start_detector_recipe_factorial_20261020.ps1',
        Path(P.__file__),Path(F.__file__),SOURCE/'protocol.json',SOURCE/'source_receipt.json',Path(entry['checkpoint']),
        V.FROZEN,train_source]+list(OUT.glob('*.txt'))+list(OUT.glob('*.yaml'))}
    for row in receipt['copied_inputs']:
        P.receipt_check(row['path'],row['sha256']);pins[row['path']]=row['sha256']
    for row in receipt['smoke_inputs']:
        for key in ['image','label']:
            P.receipt_check(row[key],row[key+'_sha256']);pins[row[key]]=row[key+'_sha256']
    code=[ROOT/'data/scripts/evaluate_p2_validation_comparison.py',ROOT/'data/scripts/evaluate_reviewed_eval_labels.py',
        ROOT/'data/scripts/evaluate_nested_box_filter.py',ROOT/'data/scripts/eval_recall_report.py',ROOT/'data/scripts/compare_native_resolution_budget.py']
    pins.update({str(p):P.sha(p) for p in code})
    spec=dict(status='locked_before_gpu_smoke',created_utc=utc(),arms=ARMS,seed=0,configs=configs,
        initialization=entry,environment=P.environment(),files_sha256=pins,
        train_exposures=4860,train_unique=2371,val_images=173,val_boxes=560,val_unique=183,
        physical_batch=8,independent_variables=['augmentation_package','nbs_64_or_8'],
        common_recipe='SGD lr0.01,linear final0.01x,warmup3,1024,BF16,batch8,workers2,max60/patience15',
        augmentation='Historical brightness/scale/translation/flip plus mosaic until displayed epoch11; off zeros these probabilities',
        primary='Each augmentation and nbs contrast, then interaction; single-seed development evidence',
        primary_candidate_budget=1071,diagnostic_budgets=[300,600],selection='best EMA validation AP50-95',
        fairness='Same exact full initialization, ordered original4860 exposure list, labels, spatial validation, remaining recipe and scorer',
        limitation=['Different update count/noise is part of nbs treatment; warmup ramps accumulation in nbs64 arms.',
            'Original aspect-ratio sources differ from the prior no-augmentation square-input study, so this is not its exact rerun.',
            'One seed on reused development area; no blind test. Unmatched predictions are not automatically background.',
            'SDK epoch resume starts student from quantized EMA; no raw student/RNG/partial gradient replay.'],
        new_images_downloaded=0,new_labels_imported=0,expert_labels_modified=False,test_images_read=False,
        periodic_trigger=False,automatic_following_experiments=False)
    write(OUT/'protocol.json',spec);(OUT/'protocol.sha256').write_text(P.sha(OUT/'protocol.json')+'\n')
    write(OUT/'queue.json',dict(status='prepared_not_started',task_name=TASK,completed_arms=0,total_arms=4))
    print('PREPARED',len(pins),'pinned files',flush=True)


def check():
    P.receipt_check(OUT/'protocol.json',(OUT/'protocol.sha256').read_text().strip())
    spec=P.read(OUT/'protocol.json')
    for path,digest in spec['files_sha256'].items():P.receipt_check(path,digest)
    assert P.environment()==spec['environment']
    return spec


def execute(spec,arm,smoke=False,resume=False):
    folder=OUT/('smoke' if smoke else 'training')/arm;folder.mkdir(parents=True,exist_ok=True)
    run=folder/'train';audit_path=folder/'audit.json'
    cfg=deepcopy(spec['configs'][arm]);cfg.update(data=str(OUT/('smoke.yaml' if smoke else 'data.yaml')),
        project=str(folder),name='train',exist_ok=True)
    if smoke:cfg.update(epochs=2,warmup_epochs=0.,close_mosaic=0)
    context=None
    if resume:
        # Preserve SDK's unstripped epoch checkpoint before final_eval strips last/best.
        history=folder/('resume_history_'+str(int(time.time())));history.mkdir()
        for name in ['last','best']:
            if (run/f'weights/{name}.pt').exists():shutil.copy2(run/f'weights/{name}.pt',history/f'{name}.pt')
            shutil.copy2(folder/f'recovery_{name}.pt',run/f'weights/{name}.pt')
        context=P.checkpoint_resume_context(run,cfg)
        audit=P.read(audit_path);audit.setdefault('resume_history',[]).append(context)
        audit['actual_optimizer_steps']=context['ema_updates']
        audit['epochs']=[r for r in audit.get('epochs',[]) if r['epoch']<=context['completed_epochs']]
    else:
        if audit_path.exists():raise FileExistsError('Use resume for existing arm')
        audit=dict(arm=arm,smoke=smoke,status='starting',actual_optimizer_steps=0,epochs=[],
            initial_model_sha256=None,initial_batch_hashes=[],finite_gradient_checks=0)
    write(audit_path,audit)
    outer_spec=spec;handles=[];began=time.perf_counter();last_batch_time=[began]

    class RecipeTrainer(DetectionTrainer):
        def get_model(self,cfg=None,weights=None,verbose=True):
            assert isinstance(weights,torch.nn.Module)
            model=deepcopy(weights).float()
            assert model.yaml['nc']==self.data['nc']==1
            model=self.set_model_names_for_load(model)
            assert P.state_hash(model.state_dict())==P.state_hash(weights.float().state_dict())
            init_seeds(0,deterministic=True)
            return model

        def _build_train_pipeline(self):
            if self.batch_size!=8 or getattr(self,'_oom_retries',0):raise RuntimeError('Fixed batch OOM; do not reduce silently')
            return super()._build_train_pipeline()

        def preprocess_batch(self,batch):
            batch=super().preprocess_batch(batch)
            if not resume and len(audit['initial_batch_hashes'])<(16 if smoke else 1):
                audit['initial_batch_hashes'].append(P.state_hash({'image':batch['img']}))
            return batch

        def optimizer_step(self):
            if smoke or audit['actual_optimizer_steps']%100==0:
                norms=torch.stack([p.grad.detach().float().norm() for p in self.model.parameters() if p.grad is not None])
                assert torch.isfinite(norms).all().item() and norms.max().item()>0
                audit['finite_gradient_checks']+=1
            return super().optimizer_step()

    def initialized(t):
        actual=P.state_hash(t.model.state_dict())
        expected=context['expected_student_sha256'] if context else outer_spec['initialization']['full_state_sha256']
        assert actual==expected,'Initialization changed'
        assert int(t.args.nbs)==int(cfg['nbs']) and t.batch_size==8
        assert (len(t.train_loader.dataset),len(t.test_loader.dataset))==((128,8) if smoke else (4860,173))
        if context:
            P.restore_stopper(t,context)
            assert P.state_hash(t.ema.ema.state_dict())==context['expected_ema_sha256']
            assert P.nested_state_hash(t.optimizer.state_dict())==context['expected_optimizer_sha256']
            assert t.ema.updates==context['ema_updates']
            audit['resume_history'][-1]['verified_exact_checkpoint_restoration']=True
        else:audit['initial_model_sha256']=actual
        audit.update(status='running',actual_args=vars(t.args),actual_accumulate_at_start=t.accumulate)
        t.optimizer.register_step_post_hook(lambda *_:audit.update(actual_optimizer_steps=audit['actual_optimizer_steps']+1))
        def seen(module,inputs,output):
            if isinstance(module,torch.nn.Conv2d) and module.training and torch.is_grad_enabled() and inputs[0].is_cuda:
                assert output.dtype==torch.bfloat16
                audit['actual_training_dtype']=str(output.dtype);handles[-1].remove()
        handles.append(torch.nn.modules.module.register_module_forward_hook(seen))
        write(audit_path,audit)

    def saved(t):
        for name,path in [('last',t.last),('best',t.best)]:
            if Path(path).exists():
                temp=folder/f'recovery_{name}.writing.pt';shutil.copy2(path,temp);temp.replace(folder/f'recovery_{name}.pt')

    def epoch_end(t):
        record=dict(epoch=t.epoch+1,accumulate=t.accumulate,optimizer_steps=audit['actual_optimizer_steps'],
            AP50=float(t.metrics['metrics/mAP50(B)']),AP50_95=float(t.metrics['metrics/mAP50-95(B)']),
            best_epoch=int(t.stopper.best_epoch),epoch_seconds=float(t.epoch_time))
        audit['epochs'].append(record);write(audit_path,audit)
        if smoke and t.epoch==0:t.stop=True  # deliberate one-epoch stop; enables a genuine resume check
        if not smoke:
            q=P.read(OUT/'queue.json');q.update(current_arm=arm,epoch=t.epoch+1,completed_epoch=t.epoch+1,
                metrics=record,phase='training');write(OUT/'queue.json',q)

    def batch_end(t):
        audit['batches_completed_this_epoch']=audit.get('batches_completed_this_epoch',0)+1
        assert torch.isfinite(t.loss.detach()).all().item(),'Nonfinite loss'
        now=time.perf_counter()
        if not smoke and now-last_batch_time[0]>15:
            q=P.read(OUT/'queue.json');q.update(current_arm=arm,epoch=t.epoch+1,
                batches_completed=audit['batches_completed_this_epoch'],batches_per_epoch=len(t.train_loader),
                optimizer_steps=audit['actual_optimizer_steps'],phase='training')
            write(OUT/'queue.json',q);last_batch_time[0]=now

    def epoch_start(t):
        audit['batches_completed_this_epoch']=0
        if not smoke:
            q=P.read(OUT/'queue.json');q.update(current_arm=arm,epoch=t.epoch+1,batches_completed=0,
                batches_per_epoch=len(t.train_loader),phase='training');write(OUT/'queue.json',q)

    try:
        weight=folder/'recovery_last.pt' if context else Path(spec['initialization']['checkpoint'])
        model=YOLO(str(weight))
        for event,callback in [('on_pretrain_routine_end',initialized),('on_model_save',saved),
            ('on_fit_epoch_end',epoch_end),('on_train_batch_end',batch_end),
            ('on_train_epoch_start',epoch_start)]:model.add_callback(event,callback)
        if context:cfg['resume']=str(weight)
        model.train(trainer=RecipeTrainer,**cfg)
        assert audit['actual_optimizer_steps']>0 and audit['actual_training_dtype']=='torch.bfloat16'
        checkpoint=torch.load(folder/'recovery_last.pt',map_location='cpu',weights_only=False)
        assert checkpoint['optimizer'] is not None and int(checkpoint['updates'])==audit['actual_optimizer_steps']
        reloaded=YOLO(str(folder/'recovery_last.pt')).model.cpu().float()
        assert P.state_hash(reloaded.state_dict())==P.state_hash(checkpoint['ema'].float().state_dict())
        audit.update(status='passed_smoke' if smoke else 'complete',best_epoch=int(model.trainer.stopper.best_epoch),
            checkpoint_roundtrip=True,seconds=time.perf_counter()-began,
            best_checkpoint_sha256=P.sha(run/'weights/best.pt'),
            recovery_last_sha256=P.sha(folder/'recovery_last.pt'))
        write(audit_path,audit)
        print('ARM_COMPLETED',arm,'smoke',smoke,'epoch',audit['epochs'][-1]['epoch'],flush=True)
        del model,reloaded,checkpoint;gc.collect();torch.cuda.empty_cache()
        return audit
    except BaseException as e:
        audit.update(status='failed',error=repr(e));write(audit_path,audit);raise
    finally:
        for handle in handles:handle.remove()


def smoke(spec):
    records={}
    for arm in ARMS:
        q=P.read(OUT/'queue.json');q.update(phase='smoke',current_arm=arm);write(OUT/'queue.json',q)
        records[arm]=execute(spec,arm,smoke=True)
    assert len({r['initial_model_sha256'] for r in records.values()})==1
    for left,right in [(ARMS[0],ARMS[2]),(ARMS[1],ARMS[3])]:
        assert records[left]['initial_batch_hashes']==records[right]['initial_batch_hashes'],'Paired inputs differ'
        assert records[left]['actual_optimizer_steps']==2 and records[right]['actual_optimizer_steps']==16
    assert records[ARMS[0]]['initial_batch_hashes']!=records[ARMS[1]]['initial_batch_hashes']
    resumed=execute(spec,ARMS[0],smoke=True,resume=True)
    assert resumed['resume_history'][-1]['verified_exact_checkpoint_restoration']
    write(OUT/'readiness.json',dict(status='passed_four_actual_nbs_BF16_updates_input_pairing_and_resume',
        initial_model_sha256=records[ARMS[0]]['initial_model_sha256'],
        pairwise_augmented_input_hashes_equal=True,actual_first_epoch_steps={a:r['actual_optimizer_steps'] for a,r in records.items()},
        genuine_second_epoch_resume_verified=True,protocol_sha256=P.sha(OUT/'protocol.json'),quality_result=False))


def evaluate(arm):
    import evaluate_p2_validation_comparison as V
    from compare_native_resolution_budget import equal_budget_threshold
    from evaluate_reviewed_eval_labels import metrics
    from evaluate_nested_box_filter import score
    from eval_recall_report import match_image
    items=V.validate_validation_inputs();folder=OUT/'evaluation'/arm;folder.mkdir(parents=True)
    weight=OUT/'training'/arm/'train/weights/best.pt'
    pred,timing=V.infer(weight,items,folder);write(folder/'predictions.json',{k:[a.tolist() for a in v] for k,v in pred.items()})
    results={}
    for budget in [300,600,1071]:
        threshold=equal_budget_threshold(pred,budget);m=metrics(items,pred,'human',threshold)
        other=score(items,pred,threshold)[0];assert all(m[k]==other[k] for k in ['gt','tp','fp','candidates'])
        ids=set()
        for item in items:
            b,c,classes=pred[item['stem']]
            hits=match_image([(0,*box) for box in item['human']['boxes']],b,c,classes,.5,threshold,class_aware=False)
            ids.update(int(s) for s,h in zip(item['source_ids'],hits) if h[1])
        results[str(budget)]=dict(metrics=m,unique_recalled=len(ids),unique_IDs=sorted(ids))
    write(folder/'metrics.json',dict(budgets=results,timing=timing,checkpoint_sha256=P.sha(weight)))


def scheduled(resume=False):
    spec=check();owned=F.process_independence();assert owned['scheduler_owned']
    F.ensure_idle(OUT);blocked=F.prevent_network();torch.set_num_threads(2)
    assert torch.cuda.mem_get_info()[0]>4*2**30,'Insufficient free GPU memory'
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    queue=dict(status='running',started_utc=utc(),pid=os.getpid(),task_name=TASK,process_independence=owned,
        offline=True,temporary_system_awake=True,completed_arms=0,total_arms=4,phase='smoke')
    write(OUT/'queue.json',queue)
    try:
        if not resume:smoke(spec)
        ready=P.read(OUT/'readiness.json');assert ready['protocol_sha256']==P.sha(OUT/'protocol.json')
        for arm in ARMS:
            path=OUT/'training'/arm/'audit.json'
            queue.update(current_arm=arm,phase='training');write(OUT/'queue.json',queue)
            if not path.exists() or P.read(path)['status']!='complete':execute(spec,arm,resume=path.exists())
            if not (OUT/'evaluation'/arm/'metrics.json').exists():evaluate(arm)
            queue.update(P.read(OUT/'queue.json'));queue['completed_arms']+=1;write(OUT/'queue.json',queue)
        result={a:P.read(OUT/'evaluation'/a/'metrics.json') for a in ARMS}
        write(OUT/'comparison.json',dict(status='completed_single_seed_2x2_development',results=result,
            protocol_sha256=P.sha(OUT/'protocol.json'),final_test_read=False))
        queue.update(status='completed',phase='complete')
    except BaseException as e:queue.update(status='failed',error=repr(e));raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        queue.update(finished_utc=utc(),temporary_system_awake=False,blocked_network_attempts=blocked)
        write(OUT/'queue.json',queue)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','scheduled-run','scheduled-resume'])
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    else:
        tag=str(int(time.time()));sys.stdout=(OUT/f'training_{tag}.log').open('a',encoding='utf-8',buffering=1)
        sys.stderr=(OUT/f'training_{tag}.err.log').open('a',encoding='utf-8',buffering=1)
        scheduled(args.mode=='scheduled-resume')
