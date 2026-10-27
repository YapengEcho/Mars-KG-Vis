"""Train full-cohort LoRA using predeclared epochs; evaluate frozen ROI pools."""
from __future__ import annotations
import argparse
from copy import deepcopy
import ctypes
import gc
import json
import os
from pathlib import Path
import sys
import time

import run_classifier_lora_native40 as G
import run_detector_classifier_bridge_v2 as B
M, C, np, torch = G.M, G.C, G.np, G.torch
ROOT = G.ROOT
OUT = ROOT / 'runs/experiments/full_lora_bridge_20261015_v1'
PLAN = ROOT / 'runs/research/next_after_head_rebalance_20261015_v1/plan.json'


def score_predictions(plan, classes):
    oracle = plan['oracle']
    truth = [r['class_id'] for r in oracle]
    pred = [int(classes[r['case_index']]) for r in oracle]
    result = dict(oracle=dict(unique_targets=len(truth), macro_f1=float(M.f1_score(truth, pred, labels=range(4), average='macro', zero_division=0)),
                             accuracy=float(np.mean(np.array(truth) == pred)),
                             per_class=M.classification_report(truth, pred, labels=range(4), target_names=M.NAMES, output_dict=True, zero_division=0),
                             confusion_matrix=M.confusion_matrix(truth, pred, labels=range(4)).tolist()), pipeline={})
    items = {x['stem']: x for x in plan['items']}
    for name, model in plan['models'].items():
        gt, count, correct = np.zeros(4, int), np.zeros(4, int), np.zeros(4, int)
        y, detector, manual = [], [], []
        for stem, entry in model['items'].items():
            predicted = [int(classes[r['case_index']]) for r in entry['predictions']]
            count += np.bincount(predicted, minlength=4)
            for match, label in zip(entry['matches'], items[stem]['labels']):
                typ = label['class_id']; gt[typ] += 1
                if match['matched']:
                    value = predicted[match['kept_prediction_index']]
                    correct[typ] += value == typ
                    y.append(typ); detector.append(value); manual.append(int(classes[label['case_index']]))
        located = len(y)
        assert located == model['expected_detection_TP'] and gt.sum() == 560 and count.sum() == 1071
        precision = np.divide(correct, count, out=np.zeros(4), where=count > 0)
        recall = correct / gt
        y, detector, manual = map(np.array, [y, detector, manual])
        result['pipeline'][name] = dict(localized=located, candidates=1071, GT_appearances=560,
            matched_classification_accuracy=float(np.mean(y == detector)), jointly_correct=int(correct.sum()),
            joint_recall=float(correct.sum()/560), reference_joint_precision=float(correct.sum()/1071),
            per_class={n:dict(GT=int(gt[i]), predicted=int(count[i]), joint_TP=int(correct[i]), precision=float(precision[i]), recall=float(recall[i])) for i,n in enumerate(M.NAMES)},
            same_matched_targets=dict(occurrences=located, manual_correct=int(sum(y==manual)), detector_correct=int(sum(y==detector)),
                manual_macro_f1=float(M.f1_score(y,manual,labels=range(4),average='macro',zero_division=0)),
                detector_macro_f1=float(M.f1_score(y,detector,labels=range(4),average='macro',zero_division=0))))
    return result


def prepare():
    source=G.check(); B.check(); proposal=C.read_json(PLAN)
    C.new_directory(OUT)
    plan=C.read_json(B.OUT/'input_plan.json')
    assert len(plan['cases'])==3396 and len(plan['oracle'])==183
    files=list(map(Path,source['files_sha256']))+[PLAN,B.OUT/'protocol.json',B.OUT/'input_plan.json',B.OUT/'comparison.json',B.OUT/'class_predictions.npz']
    for case in plan['cases']:
        receipt=B.OUT/'receipts'/f"roi_{case['index']:05d}.json"
        assert C.sha256(case['crop'])==C.read_json(receipt)['crop_sha256']
        files += [Path(case['crop']),receipt]
    code=list(map(Path,source['code_hashes']))+[Path(__file__),Path(B.__file__),ROOT/'data/scripts/start_full_lora_bridge.ps1']
    spec=deepcopy(source)
    spec.update(created_utc=M.utc(),source_experiment=str(G.OUT),bridge_plan=str(B.OUT/'input_plan.json'),
                epoch_rules=proposal['priority1']['full_training_epoch_rules'],seeds=G.SEEDS,
                expert_labels_retained=True,labels_modified=False,external_evaluation='Existing173-image development region, not blind test',
                final_test_images_read=False,validation_selection=False,automatic_following_experiments=False)
    spec['files_sha256']={str(p):C.sha256(p) for p in sorted(set(files))}
    spec['code_hashes']={str(p):C.sha256(p) for p in sorted(set(code))}
    G.write(OUT/'protocol.json',spec)
    (OUT/'protocol.sha256').write_text(C.sha256(OUT/'protocol.json')+'\n',encoding='utf-8')
    G.write(OUT/'queue.json',dict(status='prepared_not_started',created_utc=M.utc()))
    print('PREPARED_FULL_LORA_BRIDGE',flush=True)


def check():
    spec=C.read_json(OUT/'protocol.json')
    assert C.sha256(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for field in ['files_sha256','code_hashes']:
        for p,d in spec[field].items():assert C.sha256(p)==d,p
    for name,d in spec['checkpoint_files_sha256'].items():assert C.sha256(Path(spec['checkpoint'])/name)==d
    assert spec['versions']==M.P.R.package_versions()
    return spec


def external_data(plan):
    gray=[];weights=[]
    for case in plan['cases']:
        with M.Image.open(case['crop']) as im:
            assert im.mode=='L' and im.size==(224,224);gray.append(np.array(im))
        a=M.P.patch_areas(case['target_box224']);weights.append(a/a.sum())
    return dict(gray=np.stack(gray),roi=torch.tensor(np.array(weights),dtype=torch.float32))


def full_train(spec,data,seed,queue):
    folder=OUT/f'seed{seed}';folder.mkdir(exist_ok=True)
    fit=list(range(1083));limit=spec['epoch_rules'][str(seed)]['full_training_epochs']
    model=M.Classifier(spec,'qv_last2_LoRA',fit,data,0,'refit',seed)
    opt=M.optimizer(model,spec);w=M.weights(data,fit);history=[];start=1
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        assert saved['protocol_sha256']==C.sha256(OUT/'protocol.json')
        M.load_parameters(model,saved['parameters']);opt.load_state_dict(saved['optimizer'])
        start=saved['completed_epoch']+1;history=saved['history']
    for epoch in range(start,limit+1):
        order=np.random.default_rng(seed+10000+epoch).permutation(1083);loss=[]
        for offset in range(0,1083,16):
            loss.append(M.update_group(model,opt,data,order[offset:offset+16],w))
            if offset%128==0:
                queue.update(phase='full_training',seed=seed,epoch=epoch,epochs=limit,completed_images=offset+min(16,1083-offset));G.write(OUT/'queue.json',queue)
        history.append(dict(epoch=epoch,loss=float(np.mean(loss))))
        M.save_checkpoint(folder/'last.pt',model,opt,history,epoch,0,-1.,'qv_last2_LoRA',0,'refit',seed)
        G.write(folder/'training.json',dict(history=history,fit_IDs=1083,train_only_weights=w.cpu().tolist(),fixed_epochs=limit,external_scores_used=False))
        print(f'seed{seed} full epoch{epoch}/{limit} loss={np.mean(loss):.5f}',flush=True)
    G.write(folder/'completed.json',dict(status='completed',seed=seed,epochs=limit,fit_IDs=1083,checkpoint_sha256=C.sha256(folder/'last.pt')))
    return model


def smoke():
    spec=check();M.device_setup();attempts=M.FENV.prevent_network();data=M.load_data(spec)
    model=M.Classifier(spec,'qv_last2_LoRA',list(range(1083)),data,0,'refit',G.SEEDS[0]);opt=M.optimizer(model,spec)
    ids=np.array(sum(([i for i,r in enumerate(data['rows']) if r['class_id']==c][:4] for c in range(4)),[]))
    M.update_group(model,opt,data,ids,M.weights(data,list(range(1083))))
    temp=OUT/'smoke.pt';M.save_checkpoint(temp,model,opt,[],1,0,-1.,'qv_last2_LoRA',0,'refit',G.SEEDS[0])
    before=M.predict(model,data,ids)[1]
    fresh=M.Classifier(spec,'qv_last2_LoRA',list(range(1083)),data,0,'refit',G.SEEDS[0])
    M.load_parameters(fresh,torch.load(temp,map_location='cpu',weights_only=False)['parameters'])
    assert np.array_equal(before,M.predict(fresh,data,ids)[1])
    G.write(OUT/'readiness.json',dict(status='passed_real_GPU_smoke',images=16,restored_prediction_bitwise_equal=True,
                                    no_external_scores_for_selection=True,blocked_network_attempts=attempts,protocol_sha256=C.sha256(OUT/'protocol.json')))
    print('PASSED_FULL_LORA_BRIDGE_SMOKE',flush=True)


def run():
    spec=check();assert C.read_json(OUT/'readiness.json')['protocol_sha256']==C.sha256(OUT/'protocol.json')
    independence=M.FENV.process_independence();assert independence['scheduler_owned']
    blocked=M.FENV.prevent_network();M.device_setup();assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    queue=dict(status='running',started_utc=M.utc(),pid=os.getpid(),process_independence=independence,offline=True,temporary_system_awake=True)
    G.write(OUT/'queue.json',queue)
    try:
        data=M.load_data(spec);plan=C.read_json(spec['bridge_plan']);external=external_data(plan)
        with np.load(B.OUT/'class_predictions.npz',allow_pickle=False) as d:baseline=d['roi_only'].copy()
        scores={'old_frozen_ROI':score_predictions(plan,baseline)}
        assert abs(scores['old_frozen_ROI']['oracle']['macro_f1']-C.read_json(B.OUT/'comparison.json')['oracle_classification']['roi_only']['macro_f1'])<1e-12
        for seed in spec['seeds']:
            model=full_train(spec,data,seed,queue);pred=[];logits=[]
            for start in range(0,3396,8):
                ix=np.arange(start,min(start+8,3396));p,z=M.predict(model,external,ix);pred.extend(p.tolist());logits.append(z)
                if start%128==0:
                    queue.update(phase='adapted_external_ROI_inference',seed=seed,completed_ROIs=start+len(ix),total_ROIs=3396);G.write(OUT/'queue.json',queue)
            pred=np.array(pred);np.savez_compressed(OUT/f'seed{seed}'/'external_predictions.npz',predictions=pred,logits=np.concatenate(logits))
            scores[str(seed)]=score_predictions(plan,pred)
            G.write(OUT/f'seed{seed}'/'evaluation.json',scores[str(seed)])
            del model;gc.collect();torch.cuda.empty_cache()
        aggregate=dict(oracle_macro_f1_mean=float(np.mean([scores[str(s)]['oracle']['macro_f1'] for s in G.SEEDS])),
                       oracle_macro_f1_std=float(np.std([scores[str(s)]['oracle']['macro_f1'] for s in G.SEEDS])),
                       oracle_accuracy_mean=float(np.mean([scores[str(s)]['oracle']['accuracy'] for s in G.SEEDS])),pipeline={})
        for name in plan['models']:
            aggregate['pipeline'][name]={k:float(np.mean([scores[str(s)]['pipeline'][name][k] for s in G.SEEDS])) for k in ['joint_recall','reference_joint_precision','matched_classification_accuracy','jointly_correct']}
        check();G.write(OUT/'comparison.json',dict(status='completed_full_training_LoRA_bridge',scores=scores,aggregate=aggregate,
                                                seeds=G.SEEDS,validation_images=173,unique_oracle=183,GT_appearances=560,
                                                labels_modified=False,final_test_images_read=False,external_selection=False,protocol_sha256=C.sha256(OUT/'protocol.json')))
        queue.update(status='completed_full_training_LoRA_bridge',phase='complete')
    except BaseException as error:queue.update(status='failed',error=repr(error));raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        queue.update(finished_utc=M.utc(),temporary_system_awake=False,blocked_network_attempts=blocked);G.write(OUT/'queue.json',queue)


M.OUT=OUT;M.write=G.write
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','smoke','scheduled-run']);args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='smoke':smoke()
    else:
        sys.stdout=(OUT/'training.log').open('a',encoding='utf-8',buffering=1);sys.stderr=(OUT/'training.err.log').open('a',encoding='utf-8',buffering=1);run()
