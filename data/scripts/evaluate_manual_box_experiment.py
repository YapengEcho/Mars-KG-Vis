"""Select an operating threshold on validation, lock it, then evaluate test once."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image
from ultralytics import YOLO

from prepare_manual_box_experiment import EXPERIMENT, DATASET
from run_manual_box_experiment import load_spec, preflight
from export_manual_detection_dataset import read_json, write_json
from build_center_candidate_manifest import sha256_file
from eval_recall_report import match_image, SIZE_BINS


def summarize_predictions(gt,preds,tau,match_iou=.5):
    totals=Counter();hits=Counter();n_gt=n_pred=n_tp=empty=empty_fp=0
    for stem,rows in gt.items():
        boxes,scores,classes=preds[stem]
        count=int(np.count_nonzero(scores>=tau));n_pred+=count;n_gt+=len(rows)
        if not rows:empty+=1;empty_fp+=int(count>0)
        records=match_image(rows,boxes,scores,classes,match_iou,tau,class_aware=False)
        for _,matched,size,_,_ in records:
            totals[size]+=1;hits[size]+=int(matched);n_tp+=int(matched)
    fp=n_pred-n_tp
    return dict(threshold=float(tau),images=len(gt),gt=n_gt,tp=n_tp,fn=n_gt-n_tp,
        candidates=n_pred,fp=fp,precision=n_tp/n_pred if n_pred else 0.,
        recall=n_tp/n_gt if n_gt else 0.,fp_per_image=fp/len(gt) if gt else 0.,
        empty_images=empty,empty_images_with_fp=empty_fp,
        empty_image_fp_rate=empty_fp/empty if empty else 0.,
        by_size={s:dict(gt=totals[s],tp=hits[s],recall=hits[s]/totals[s] if totals[s] else None) for s in SIZE_BINS})


def choose_threshold(rows,budget):
    eligible=[r for r in rows if r['candidates']<=budget]
    if not eligible:raise ValueError('No validation operating point meets the candidate budget')
    return max(eligible,key=lambda r:(r['recall'],r['precision'],-r['candidates'],r['threshold']))


def ground_truth(root,split):
    images=sorted((root/'images'/split).glob('*.png'));gt={}
    for image in images:
        with Image.open(image) as im:w,h=im.size
        rows=[]
        for line in (root/'labels'/split/(image.stem+'.txt')).read_text().splitlines():
            c,x,y,bw,bh=map(float,line.split());assert c==0
            rows.append((0,(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h))
        gt[image.stem]=rows
    return images,gt


def inference(weights,images,folder,settings):
    source=folder/'inference_source.txt';source.write_text('\n'.join(p.as_posix() for p in images)+'\n',encoding='utf-8')
    model=YOLO(str(weights));assert len(model.names)==1
    preds={}
    for result in model.predict(source=str(source),imgsz=1024,conf=settings['conf_pool'],
            iou=settings['nms_iou'],max_det=settings['max_det'],agnostic_nms=True,
            device=0,batch=1,stream=True,verbose=False,augment=False,rect=False):
        boxes=result.boxes
        if boxes is None:
            values=(np.empty((0,4),np.float32),np.empty(0,np.float32),np.empty(0,np.float32))
        else:values=(boxes.xyxy.cpu().numpy(),boxes.conf.cpu().numpy(),boxes.cls.cpu().numpy())
        assert not len(values[2]) or np.all(values[2]==0)
        preds[Path(result.path).stem]=values
    assert set(preds)=={p.stem for p in images}
    return model,preds


def evaluate(folder,spec,arm,seed,split):
    run=folder/'formal'/f'{arm}_seed{seed}';audit=read_json(run/'training_audit.json')
    if audit['status']!='complete' or not audit['formal_result']:raise ValueError('A completed formal training run is required')
    if audit['experiment_sha256']!=sha256_file(folder/'experiment.json'):raise ValueError('Training used another protocol')
    preflight(folder,spec,arm)
    weight=run/'weights/best.pt'
    if sha256_file(weight)!=audit['best_sha256']:raise ValueError('Selected checkpoint changed')
    out=run/'evaluation';out.mkdir(exist_ok=True);lock=out/'validation_lock.json'
    if split=='val':
        if lock.exists():raise FileExistsError('Validation selection is already locked; do not retune after test')
    else:
        if not lock.exists():raise ValueError('Select and lock a validation threshold before reading test predictions')
        if (out/'test_metrics.json').exists():raise FileExistsError('Test was already evaluated; use the saved report')
        locked=read_json(lock)
        if locked['checkpoint_sha256']!=audit['best_sha256'] or locked['experiment_sha256']!=audit['experiment_sha256']:
            raise ValueError('Validation lock or selected checkpoint differs')
    image_root=DATASET/'manual'
    images,gt=ground_truth(image_root,split)
    assert len(images)==(615 if split=='val' else 497)
    assert sum(map(len,gt.values()))==(812 if split=='val' else 1144)
    splitout=out/split;splitout.mkdir(exist_ok=True)
    start=time.time();model,preds=inference(weight,images,splitout,spec['eval'])
    prediction_file=splitout/'predictions.json'
    write_json(prediction_file,{k:[x.tolist() for x in v] for k,v in preds.items()})
    settings=spec['eval']
    if split=='val':
        rows=[summarize_predictions(gt,preds,t,settings['match_iou']) for t in settings['threshold_grid']]
        chosen=choose_threshold(rows,settings['candidate_budget']);tau=chosen['threshold']
        write_json(out/'validation_threshold_sweep.json',rows)
    else:tau=locked['threshold'];chosen=summarize_predictions(gt,preds,tau,settings['match_iou'])
    standard=model.val(data=spec['arms']['manual']['data'],split=split,imgsz=1024,conf=settings['conf_pool'],
            iou=settings['nms_iou'],max_det=settings['max_det'],device=0,batch=4,workers=0,
            plots=False,augment=False,rect=False,project=str(splitout),name='map',exist_ok=False)
    report=dict(arm=arm,seed=seed,split=split,checkpoint_sha256=audit['best_sha256'],
                experiment_sha256=audit['experiment_sha256'],threshold=tau,operating_point=chosen,
                map50=float(standard.box.map50),map50_95=float(standard.box.map),
                predictions_sha256=sha256_file(prediction_file),elapsed_seconds=time.time()-start)
    write_json(out/('validation_metrics.json' if split=='val' else 'test_metrics.json'),report)
    if split=='val':
        write_json(lock,dict(threshold=tau,selection_split='val',checkpoint_sha256=audit['best_sha256'],
                   experiment_sha256=audit['experiment_sha256'],validation_report_sha256=sha256_file(out/'validation_metrics.json'),
                   rule=settings['threshold_selection'],candidate_budget=settings['candidate_budget']))
    print(json.dumps({k:report[k] for k in ['arm','seed','split','threshold','map50','map50_95']}))
    return report


def aggregate(folder,spec):
    rows=[]
    for seed in spec['seeds']:
        for arm in spec['arms']:
            root=folder/'formal'/f'{arm}_seed{seed}';path=root/'evaluation/test_metrics.json'
            if not path.exists():continue
            r=read_json(path);a=read_json(root/'training_audit.json');o=r['operating_point']
            rows.append(dict(arm=arm,seed=seed,map50=r['map50'],map50_95=r['map50_95'],
                             precision=o['precision'],recall=o['recall'],fp_per_image=o['fp_per_image'],
                             empty_image_fp_rate=o['empty_image_fp_rate'],threshold=r['threshold'],
                             actual_optimizer_steps=a['actual_optimizer_steps']))
    metrics=['map50','map50_95','precision','recall','fp_per_image','empty_image_fp_rate']
    group={}
    for arm in spec['arms']:
        selected=[r for r in rows if r['arm']==arm]
        if not selected:continue
        group[arm]={m:dict(mean=statistics.mean(r[m] for r in selected),
                          sample_std=statistics.stdev(r[m] for r in selected) if len(selected)>1 else None,
                          seeds=[r['seed'] for r in selected]) for m in metrics}
    paired={}
    for name,arms in spec['comparisons'].items():
        deltas=[]
        for seed in spec['seeds']:
            selected={r['arm']:r for r in rows if r['seed']==seed}
            if all(a in selected for a in arms):
                deltas.append(dict(seed=seed,**{m:selected[arms[0]][m]-selected[arms[1]][m] for m in metrics}))
        paired[name]=dict(direction=arms[0]+' minus '+arms[1],per_seed=deltas,
                         mean_delta={m:statistics.mean(r[m] for r in deltas) for m in metrics} if deltas else {})
    step_audits={s:{r['arm']:r['actual_optimizer_steps'] for r in rows if r['seed']==s} for s in spec['seeds']}
    report=dict(status='complete' if len(rows)==9 else 'partial',completed_runs=len(rows),
                expected_runs=9,raw=rows,means_and_sample_std=group,paired=paired,
                optimizer_steps_by_seed=step_audits,
                optimizer_steps_equal_per_completed_seed={s:len(set(v.values()))<=1 for s,v in step_audits.items()},
                note='Three seeds and few geographic tiles do not establish statistical significance.')
    write_json(folder/'comparison_results.json',report);print(json.dumps(report,indent=2))
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['select','test','summarize'])
    ap.add_argument('--experiment',type=Path,default=EXPERIMENT)
    ap.add_argument('--arm',choices=['manual','shp_control','historical']);ap.add_argument('--seed',type=int,choices=[0,1,2],default=0)
    args=ap.parse_args();folder=args.experiment.resolve();spec=load_spec(folder)
    if args.mode=='summarize':aggregate(folder,spec)
    else:
        if not args.arm:ap.error('An explicit --arm is required')
        evaluate(folder,spec,args.arm,args.seed,'val' if args.mode=='select' else 'test')


if __name__=='__main__':main()
