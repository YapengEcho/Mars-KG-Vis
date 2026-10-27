"""Run explicit smoke checks or one immutable formal comparison arm."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
import yaml

from prepare_manual_box_experiment import EXPERIMENT, DATASET, OLD, environment
from export_manual_detection_dataset import read_json, write_json, link_or_copy
from build_center_candidate_manifest import sha256_file


def load_spec(folder):
    path=folder/'experiment.json'
    if sha256_file(path)!=(folder/'experiment.sha256').read_text().strip():
        raise ValueError('Frozen experiment specification changed')
    spec=read_json(path)
    if sha256_file(Path(spec['weights']))!=spec['weights_sha256']:
        raise ValueError('Initial checkpoint changed')
    if environment()!=spec['environment']:raise ValueError('Frozen runtime environment changed')
    for path,digest in read_json(folder/'code_fingerprints.json').items():
        if sha256_file(Path(path))!=digest:raise ValueError('Experiment generator code changed: '+path)
    return spec


def preflight(folder,spec,arm):
    if sha256_file(folder/'input_checksums.json')!=spec['input_checksums_sha256']:
        raise ValueError('Input checksum manifest changed')
    for record in read_json(folder/'input_checksums.json')[arm]:
        if sha256_file(Path(record['path']))!=record['sha256']:
            raise ValueError('Frozen input changed: '+record['path'])
    for path_key,hash_key in [('data','data_sha256'),('train_list','train_list_sha256')]:
        if sha256_file(Path(spec['arms'][arm][path_key]))!=spec['arms'][arm][hash_key]:
            raise ValueError('Data configuration or exposure list changed')


def tensor_hash(tensor):
    t=tensor.detach().cpu().contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def model_hash(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode());digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_pair_audit(folder,mode,arm,seed,field,value):
    comparisons=['manual','shp_control','historical'] if field=='initial_model_sha256' else ['manual','shp_control']
    if arm not in comparisons:return
    for other in comparisons:
        if other==arm:continue
        peer=folder/mode/f'{other}_seed{seed}'/'training_audit.json'
        if peer.exists():
            reference=read_json(peer).get(field)
            if reference and reference!=value:raise ValueError(f'Paired startup audit differs: {field} {arm} vs {other}')


def smoke_dataset(folder,arm,spec):
    root=folder/'smoke_datasets'/arm
    if root.exists():raise FileExistsError('Smoke dataset already exists; preserve it and use a new experiment version')
    original=Path(spec['dataset_root'])/arm if arm!='historical' else OLD
    train=sorted((original/'images/train').glob('*.png'))
    groups=[[p for p in train if bool((original/'labels/train'/(p.stem+'.txt')).read_text().strip())==positive]
            for positive in [True,False]]
    selected=[g[round(i*(len(g)-1)/3)] for g in groups for i in range(4)]
    val=sorted((OLD/'images/val').glob('*.png'))
    eval_images=[p for positive in [True,False] for p in
                 [x for x in val if bool((OLD/'labels/val'/(x.stem+'.txt')).read_text().strip())==positive][:4]]
    for split,images,source in [('train',selected,original),('val',eval_images,OLD)]:
        (root/'labels'/split).mkdir(parents=True)
        for image in images:
            link_or_copy(image,root/'images'/split/image.name)
            shutil.copy2(source/'labels'/split/(image.stem+'.txt'),root/'labels'/split/(image.stem+'.txt'))
    data=root/'data.yaml';data.write_text(yaml.safe_dump(dict(path=root.as_posix(),train='images/train',val='images/val',names={0:'LANDSLIDE'})),encoding='utf-8')
    return data


def run(folder,spec,arm,seed,smoke=False):
    mode='smoke' if smoke else 'formal';runroot=folder/mode/f'{arm}_seed{seed}'
    if runroot.exists():raise FileExistsError('Existing run is never overwritten: '+str(runroot))
    preflight(folder,spec,arm)
    data=smoke_dataset(folder,arm,spec) if smoke else Path(spec['arms'][arm]['data'])
    kwargs=copy.deepcopy(spec['training'])
    if smoke:kwargs.update(epochs=1,nbs=4,warmup_epochs=0.0,close_mosaic=0)
    kwargs.update(data=str(data),seed=seed,project=str(folder/mode),name=f'{arm}_seed{seed}')
    audit=dict(status='starting',arm=arm,seed=seed,mode=mode,formal_result=not smoke,
               experiment_sha256=sha256_file(folder/'experiment.json'),
               initial_checkpoint_sha256=spec['weights_sha256'],actual_optimizer_steps=0,
               optimizer_step_attempts=0,expected_epochs=kwargs['epochs'],started_epoch=time.time())
    runroot.mkdir(parents=True)
    # Ultralytics must write into this known empty output, never increment the name.
    kwargs['exist_ok']=True
    write_json(runroot/'training_audit.json',audit)

    class AuditedTrainer(DetectionTrainer):
        def optimizer_step(self):
            audit['optimizer_step_attempts']+=1
            return super().optimizer_step()

        def preprocess_batch(self,batch):
            batch=super().preprocess_batch(batch)
            if 'first_batch_image_sha256' not in audit:
                value=tensor_hash(batch['img']);assert_pair_audit(folder,mode,arm,seed,'first_batch_image_sha256',value)
                audit['first_batch_image_sha256']=value
                write_json(runroot/'training_audit.json',audit)
            return batch

    def initialized(trainer):
        value=model_hash(trainer.model);assert_pair_audit(folder,mode,arm,seed,'initial_model_sha256',value)
        count=len(trainer.train_loader.dataset)
        if count!=(8 if smoke else 4868):raise ValueError('Unexpected training exposure')
        if len(trainer.train_loader)!=(2 if smoke else spec['expected_batches_per_epoch']):
            raise ValueError('Unequal dataloader batches')
        audit.update(status='training',initial_model_sha256=value,train_entries=count,
                     batches_per_epoch=len(trainer.train_loader),optimizer=type(trainer.optimizer).__name__,
                     actual_args=vars(trainer.args))
        def stepped(optimizer,args,kwargs):audit['actual_optimizer_steps']+=1
        trainer.optimizer.register_step_post_hook(stepped)
        write_json(runroot/'training_audit.json',audit)

    def epoch_finished(trainer):
        losses=trainer.tloss.values() if isinstance(trainer.tloss,dict) else [trainer.tloss]
        if any(v is not None and not torch.isfinite(torch.as_tensor(v)).all() for v in losses):
            raise ValueError('Nonfinite training loss; later comparison arms must not silently continue')
        audit['completed_epochs']=trainer.epoch+1
        audit['peak_gpu_allocated_mb']=torch.cuda.max_memory_allocated()/2**20
        write_json(runroot/'training_audit.json',audit)

    try:
        torch.cuda.reset_peak_memory_stats()
        model=YOLO(spec['weights'])
        model.add_callback('on_pretrain_routine_end',initialized)
        model.add_callback('on_train_epoch_end',epoch_finished)
        model.train(trainer=AuditedTrainer,**kwargs)
        assert audit['completed_epochs']==kwargs['epochs']
        assert audit['actual_optimizer_steps']>0
        best=runroot/'weights/best.pt';last=runroot/'weights/last.pt'
        assert best.exists() and last.exists()
        audit.update(status='completed_smoke_not_formal_result' if smoke else 'complete',
                     elapsed_seconds=time.time()-audit['started_epoch'],
                     best_sha256=sha256_file(best),last_sha256=sha256_file(last),
                     skipped_optimizer_steps=audit['optimizer_step_attempts']-audit['actual_optimizer_steps'])
        write_json(runroot/'training_audit.json',audit)
        del model;torch.cuda.empty_cache()
        print('ARM_COMPLETED',mode,arm,seed,audit['actual_optimizer_steps'],flush=True)
        return audit
    except Exception as exc:
        audit.update(status='failed',error=repr(exc));write_json(runroot/'training_audit.json',audit)
        raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['smoke','train','preflight'])
    ap.add_argument('--experiment',type=Path,default=EXPERIMENT)
    ap.add_argument('--arm',choices=['manual','shp_control','historical'])
    ap.add_argument('--seed',type=int,choices=[0,1,2],default=0)
    args=ap.parse_args();folder=args.experiment.resolve();spec=load_spec(folder)
    if args.mode=='smoke':
        rows=[run(folder,spec,arm,args.seed,True) for arm in ([args.arm] if args.arm else spec['arms'])]
        write_json(folder/'smoke_summary.json',dict(status='smoke_checks_passed',formal_training_started=False,runs=rows))
    elif args.mode=='preflight':
        for arm in ([args.arm] if args.arm else spec['arms']):preflight(folder,spec,arm)
        print('PREFLIGHT_PASSED')
    else:
        if not args.arm:ap.error('train requires --arm')
        run(folder,spec,args.arm,args.seed)


if __name__=='__main__':main()
