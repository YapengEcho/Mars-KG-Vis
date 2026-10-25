"""Score frozen human vs original labels on identical cached predictions, on CPU."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import numpy as np
import torch
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class
from eval_recall_report import iou_matrix, match_image
from review_eval_labels import sha, read

def metrics(items, predictions, label_kind, threshold):
    counts=Counter(); all_correct=[]; all_scores=[]; all_targets=[]
    validator=SimpleNamespace(iouv=torch.linspace(.5,.95,10))
    by_size={}; by_position={}
    for item in items:
        boxes=item['original_boxes'] if label_kind=='original' else item['human']['boxes']
        gt=[(0,*box) for box in boxes]
        pboxes, scores, classes=[np.asarray(v,dtype=np.float32) for v in predictions[item['stem']]]
        pboxes=pboxes.reshape(-1,4)
        if len(pboxes)!=len(scores) or len(scores)!=len(classes) or np.any(classes!=0):
            raise ValueError('Invalid single-class prediction cache')
        counts['images']+=1; counts['gt']+=len(gt); counts['candidates']+=int(np.count_nonzero(scores>=threshold))
        hits=match_image(gt,pboxes,scores,classes,.5,threshold,class_aware=False)
        counts['tp']+=sum(int(h[1]) for h in hits)
        if not gt:
            counts['empty_images']+=1; counts['empty_images_with_fp']+=int(np.any(scores>=threshold))
        for row, hit in zip(gt,hits):
            _,a,b,c,d=row
            central=.25<=(a+c)/(2*item['width'])<=.75 and .25<=(b+d)/(2*item['height'])<=.75
            position='central_half_side' if central else 'outside_central_half_side'
            for group,key in [(by_size,hit[2]),(by_position,position)]:
                values=group.setdefault(key,dict(gt=0,tp=0))
                values['gt']+=1; values['tp']+=int(hit[1])
        overlaps=iou_matrix(np.asarray(boxes,dtype=np.float32).reshape(-1,4),pboxes)
        correct=BaseValidator.match_predictions(validator,torch.from_numpy(classes),torch.zeros(len(gt)),torch.from_numpy(overlaps)).numpy()
        all_correct.append(correct); all_scores.append(scores); all_targets.extend([0]*len(gt))
    correct=np.concatenate(all_correct,axis=0); scores=np.concatenate(all_scores)
    ap=ap_per_class(correct,scores,np.zeros(len(scores)),np.asarray(all_targets),plot=False)[5] if all_targets else np.empty((0,10))
    result=dict(counts)
    result.update(threshold=threshold,fp=counts['candidates']-counts['tp'],fn=counts['gt']-counts['tp'],
        precision=counts['tp']/counts['candidates'] if counts['candidates'] else 0.,
        recall=counts['tp']/counts['gt'] if counts['gt'] else None,
        fp_per_image=(counts['candidates']-counts['tp'])/counts['images'],
        empty_image_fp_rate=counts['empty_images_with_fp']/counts['empty_images'] if counts['empty_images'] else None,
        cached_map50=float(ap[:,0].mean()) if len(ap) else None,
        cached_map50_95=float(ap.mean()) if len(ap) else None,
        by_size=by_size,by_position=by_position)
    return result

def evaluate(frozen_folder, archive_path, out):
    frozen_folder,archive_path,out=map(Path,(frozen_folder,archive_path,out))
    if out.exists():
        raise FileExistsError('Keep existing evaluation receipts; use a new output version')
    frozen=read(frozen_folder/'frozen.json'); items=frozen['accepted']
    if not items or len({r['stem'] for r in items})!=len(items):
        raise ValueError('Frozen image list is empty or has duplicates')
    for item in items:
        assert item['split']=='test' and item['human']['status']=='accepted'
        assert sha(item['image'])==item['image_sha256'] and sha(item['label'])==item['label_sha256']
        assert item['human']['image_sha256']==item['image_sha256']
    results=[]
    with zipfile.ZipFile(archive_path) as archive:
        receipt=json.loads(archive.read('receipt.json'))
        assert receipt['status']=='complete' and len(receipt['runs'])==6
        for run in receipt['runs']:
            key=run['key']; blob=archive.read(key+'/predictions.json')
            assert hashlib.sha256(blob).hexdigest()==run['predictions_sha256']
            report=json.loads(archive.read(key+'/test_metrics.json')); lock=json.loads(archive.read(key+'/validation_lock.json'))
            audit=json.loads(archive.read(key+'/training_audit.json'))
            assert run['checkpoint_sha256']==lock['checkpoint_sha256']==report['checkpoint_sha256']==audit['best_sha256']
            assert lock['threshold']==run['threshold']==report['threshold']
            if key+'/best.pt' in archive.namelist():
                assert hashlib.sha256(archive.read(key+'/best.pt')).hexdigest()==run['checkpoint_sha256']
            predictions=json.loads(blob)
            if any(r['stem'] not in predictions for r in items):
                raise ValueError('Prediction cache does not cover all frozen images')
            scores={kind:metrics(items,predictions,kind,lock['threshold']) for kind in ['original','human']}
            fields=['cached_map50','cached_map50_95','recall','precision','fp_per_image']
            delta={name:scores['human'][name]-scores['original'][name] if scores['human'][name] is not None and scores['original'][name] is not None else None for name in fields}
            results.append(dict(**run,actual_epochs=audit['completed_epochs'],early_stopped=audit.get('early_stopped',False),
                original=scores['original'],human=scores['human'],human_minus_original=delta))
    result=dict(status='complete_cached_label_diagnostic',frozen_sha256=sha(frozen_folder/'frozen.json'),
        archive_sha256=sha(archive_path),batch=frozen['batch'],scored_images=len(items),held_images=len(frozen['held']),
        created_utc=datetime.now(timezone.utc).isoformat(),results=results,no_new_inference=True,no_training=True,no_threshold_tuning=True,
        notes=['Existing held-out test, previously observed; not a new blind test.',
            'All six models scored on the exact same accepted image subset with original validation-locked thresholds.',
            'Cached AP uses the saved model.predict pool and Ultralytics matching/AP. It can differ slightly from a fresh model.val run; compare old/human AP within this cache.',
            'Report each run separately: stopping rules and training image counts differ. Do not interpret the subset as the full test-set metric.',
            'Whole-image unscorable decisions are excluded identically from both label versions and reported explicitly.'])
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2)
    return dict(images=len(items),runs=len(results),out=str(out))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frozen',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); print(json.dumps(evaluate(args.frozen,args.cache,args.out)))

if __name__=='__main__':
    main()
