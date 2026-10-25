"""Reinfer six archived best checkpoints on immutable rebuilt evaluation pixels.

Use original validation-locked thresholds, never select new thresholds on test.
The historical prediction cache is NOT reused for the new crop images.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import zipfile

from review_eval_labels import ROOT, read, sha, atomic

ARCHIVE=ROOT/'runs/ablations/eval_label_review_20261009/completed_eval_artifacts.zip'
EXPECTED_ARCHIVE='567db16ded57741a3825842bade584823368d49636908497eaffea2e2e3134bb'
EXPERIMENT=ROOT/'runs/experiments/manual_box_v8_20261009/experiment.json'
OUT=ROOT/'runs/ablations/rebuilt_six_reevaluation_20261011_v1'


def prepare_weights(archive,out):
    if sha(archive)!=EXPECTED_ARCHIVE:raise ValueError('Six-run archive differs from verified backup')
    dst=out/'weights';dst.mkdir(parents=True,exist_ok=True)
    runs=[]
    with zipfile.ZipFile(archive) as z:
        receipt=json.loads(z.read('receipt.json'))
        if receipt['status']!='complete' or len(receipt['runs'])!=6:raise ValueError('Six completed runs required')
        for run in receipt['runs']:
            k=run['key']
            if not k or any(c not in 'abcdefghijklmnopqrstuvwxyz_0123456789' for c in k):raise ValueError('Invalid run identity')
            lock=json.loads(z.read(k+'/validation_lock.json'));audit=json.loads(z.read(k+'/training_audit.json'))
            if not run['checkpoint_sha256']==lock['checkpoint_sha256']==audit['best_sha256']:
                raise ValueError('Best checkpoint receipt mismatch')
            if run['threshold']!=lock['threshold']:raise ValueError('Validation threshold lock mismatch')
            blob=z.read(k+'/best.pt')
            if hashlib.sha256(blob).hexdigest()!=run['checkpoint_sha256']:raise ValueError('Checkpoint bytes mismatch')
            path=dst/(k+'.pt')
            if path.exists():
                if sha(path)!=run['checkpoint_sha256']:raise ValueError('Extracted checkpoint changed')
            else:path.write_bytes(blob)
            runs.append(dict(**run,weights=str(path),original_validation_lock=lock))
    settings=read(EXPERIMENT)['eval']
    result=dict(status='weights_verified_ready_for_frozen_data',archive_sha256=sha(archive),
        experiment_sha256=sha(EXPERIMENT),settings=settings,runs=runs,
        threshold_policy='Original locked per-run thresholds; no new selection.',
        inference_policy='Fresh predictions on each new arm and split, 1024 input, batch1, no augmentation, rect=False.')
    path=out/'protocol.json'
    if path.exists():
        if read(path)!=result:raise ValueError('Existing reevaluation protocol differs')
    else:atomic(path,result)
    return result


def validate_frozen(folder):
    frozen=read(folder/'frozen.json')
    if frozen['status']!='frozen' or frozen['role']!='evaluation_only_never_training':
        raise ValueError('Require frozen evaluation, not a preparation plan')
    items=frozen['items']
    identities={(r['arm'],r['split'],r['stem']) for r in items}
    if len(identities)!=len(items):raise ValueError('Duplicate frozen image identity')
    for arm in ('offset','centered'):
        for split in ('val','test'):
            if not any(r['arm']==arm and r['split']==split for r in items):raise ValueError('Incomplete paired val/test images')
    for r in items:
        if sha(r['image'])!=r['image_sha256'] or sha(r['label'])!=r['label_sha256']:
            raise ValueError('Frozen image/label changed')
    for split in ('val','test'):
        cohorts=[{(r['stem'],tuple(r['source_ids'])) for r in items if r['split']==split and r['arm']==arm} for arm in ('offset','centered')]
        if cohorts[0]!=cohorts[1]:raise ValueError('Paired target identities differ')
    sources={s:{sid for r in items if r['split']==s for sid in r['source_ids']} for s in ('val','test')}
    if sources['val'] & sources['test']:raise ValueError('A source landslide is shared by val and test')
    hashes={s:{r['image_sha256'] for r in items if r['split']==s} for s in ('val','test')}
    if hashes['val'] & hashes['test']:raise ValueError('Frozen pixels shared by val and test')
    return frozen


def evaluate(folder,archive,out):
    # This check occurs BEFORE importing inference/GPU libraries.
    frozen=validate_frozen(folder);protocol=prepare_weights(archive,out)
    from evaluate_manual_box_experiment import inference
    from evaluate_reviewed_eval_labels import metrics
    import torch, ultralytics
    if not torch.cuda.is_available():raise RuntimeError('Configured inference device0 requires a CUDA GPU')
    input_hash=sha(folder/'frozen.json')
    results=[]
    for run in protocol['runs']:
        for arm in ('offset','centered'):
            for split in ('val','test'):
                selected=[r for r in frozen['items'] if r['arm']==arm and r['split']==split]
                sub=out/run['key']/arm/split;sub.mkdir(parents=True,exist_ok=True)
                report=sub/'metrics.json';predfile=sub/'predictions.json'
                if report.exists():
                    prior=read(report)
                    if prior['frozen_sha256']!=input_hash or prior['checkpoint_sha256']!=run['checkpoint_sha256'] or prior['threshold']!=run['threshold'] or sha(predfile)!=prior['predictions_sha256']:
                        raise ValueError('Existing evaluation results differ; use a new output directory')
                    results.append(prior);continue
                print(f"Inference {run['key']} {arm} {split}: {len(selected)} images",flush=True)
                start=time.time();model,predictions=inference(Path(run['weights']),[Path(r['image']) for r in selected],sub,protocol['settings'])
                if predfile.exists():raise FileExistsError('Unscored existing predictions: inspect before overwriting')
                atomic(predfile,{k:[a.tolist() for a in v] for k,v in predictions.items()})
                scored=metrics(selected,predictions,'human',run['threshold'])
                scored['background_image_count']=sum(r['kind']=='background' for r in selected)
                scored['unique_source_objects']=len({sid for r in selected for sid in r['source_ids']})
                scored['duplicate_image_hashes']=len(selected)-len({r['image_sha256'] for r in selected})
                scored['max_det_cap_images']=sum(len(v[0])>=protocol['settings']['max_det'] for v in predictions.values())
                result=dict(run=run['key'],arm=arm,split=split,threshold=run['threshold'],
                    checkpoint_sha256=run['checkpoint_sha256'],frozen_sha256=input_hash,predictions_sha256=sha(predfile),
                    elapsed_seconds=time.time()-start,metrics=scored)
                atomic(report,result);results.append(result);del model;torch.cuda.empty_cache()
    validate_frozen(folder)
    payload=dict(status='complete_rebuilt_development_evaluation',fresh_blind_test=False,no_training=True,
        created_utc=datetime.now(timezone.utc).isoformat(),frozen_sha256=input_hash,protocol_sha256=sha(out/'protocol.json'),
        environment=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,gpu=torch.cuda.get_device_name(0)),
        results=results,notes=['Offset is primary; centered is a separate position diagnostic.',
            'AP uses saved predict pools and Ultralytics matching. Compare all six within this method, not directly to historical model.val AP.',
            'Reported GT counts are appearances; repeated objects/crops are not independent new landslides.',
            'Cross-tile and neighbor holds are reported in frozen.json. Background FP rate has limited support when very few background crops are saved.'])
    atomic(out/'results.json',payload)
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['weights','run'])
    p.add_argument('--archive',type=Path,default=ARCHIVE);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--frozen',type=Path);a=p.parse_args()
    if a.mode=='run' and a.frozen is None:p.error('--frozen is required for run')
    r=prepare_weights(a.archive,a.out) if a.mode=='weights' else evaluate(a.frozen,a.archive,a.out)
    print(json.dumps({'status':r['status'],'runs':len(r.get('runs',r.get('results',[])))},ensure_ascii=False))
