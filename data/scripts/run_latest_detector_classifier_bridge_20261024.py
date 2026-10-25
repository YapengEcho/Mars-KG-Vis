"""Reuse archived classifiers on native ROIs from all seven recent detector arms."""
from collections import Counter
import ctypes
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.venv/Lib/site-packages'),str(ROOT/'data/scripts')]
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import run_convnext_dino_bridge_20261024 as E
import run_full_lora_bridge as D
import run_detector_classifier_bridge_v2 as B
from scipy.special import softmax

M,C,np,torch=D.M,D.C,D.np,D.torch
DINO_CLASS=M.Classifier
CONV_CLASS=E.S.Classifier
OUT=ROOT/'runs/experiments/latest_detector_classifier_bridge_20261024_v1'
PREP=ROOT/'runs/research/latest_detector_roi_preparation_20261024_v1'
MAINT=ROOT/'runs/maintenance/catalog_expansion_20261024_v1'
SEEDS=[20261015,20261016,20261017]


def utc():return datetime.now(timezone.utc).isoformat()


def write(p,value):D.G.write(p,value)


def prepare():
    assert not OUT.exists(),'Preserve previous inference'
    parent=C.read_json(PREP/'protocol.json');plan=C.read_json(PREP/'input_plan.json')
    assert C.sha256(PREP/'input_plan.json')==parent['input_plan_sha256']
    assert len(plan['cases'])==10893 and len(plan['models'])==10
    pins={str(p):C.sha256(p) for p in [PREP/'protocol.json',PREP/'input_plan.json',Path(__file__),
                                      ROOT/'data/scripts/start_latest_detector_classifier_bridge_20261024.ps1',
                                      Path(D.__file__),Path(E.__file__),Path(B.__file__),
                                      D.OUT/'protocol.json',E.OUT/'protocol.json',E.OUT/'independent_completion_verification.json']}
    for seed in SEEDS:
        for folder in [D.OUT,E.OUT]:
            for filename in ['last.pt','completed.json','external_predictions.npz']:
                p=folder/f'seed{seed}'/filename;pins[str(p)]=C.sha256(p)
            assert C.sha256(folder/f'seed{seed}/last.pt')==C.read_json(folder/f'seed{seed}/completed.json')['checkpoint_sha256']
    old=C.read_json(B.DETECTOR/'comparison.json')
    thresholds={name:{b:old['results'][name]['budgets'][b]['threshold'] for b in ['300','600','1071']} for name in ['full_pretrained','partial_pretrained','p2']}
    expected={name:{b:old['results'][name]['budgets'][b]['metrics'] for b in ['300','600','1071']} for name in thresholds}
    audit=C.read_json(ROOT/'runs/research/latest_detector_coverage_20261024_v1/protocol.json')
    for arm in audit['detectors']:
        metrics=C.read_json(arm['metrics']);pins[arm['metrics']]=C.sha256(arm['metrics'])
        thresholds[arm['name']]={b:metrics['budgets'][b]['metrics']['threshold'] for b in ['300','600','1071']}
        expected[arm['name']]={b:metrics['budgets'][b]['metrics'] for b in ['300','600','1071']}
    spec=dict(status='locked_before_inference',created_utc=utc(),files_sha256=pins,seeds=SEEDS,
              thresholds=thresholds,expected_detection_metrics=expected,original_ROIs=3396,new_ROIs=7497,
              detector_pools=10,plan_sha256=parent['input_plan_sha256'],
              classifier_arms=['dino','convnext','equal_ensemble'],combination='T1 softmax probabilities, fixed half each',
              archived_ROI_batch_boundary=3396,batch=8,precision='FP32 TF32disabled',
              encoder_updates=0,training=False,expert_labels_changed=False,final_test_images_read=False,
              rule='Use all seven predeclared recent arms and three historical controls; no checkpoint, seed, threshold, ratio or epoch selection',
              scope='173 repeatedly used development images,183 catalog IDs,560 appearances; FL2 CL15',
              native_source_policy='Wait for verified finite6 restoration, recheck historical pixel hashes, use native224 BILINEAR crops only')
    OUT.mkdir(parents=True);write(OUT/'protocol.json',spec);(OUT/'protocol.sha256').write_text(C.sha256(OUT/'protocol.json')+'\n')
    write(OUT/'queue.json',dict(status='prepared_not_started',completed_classifier_seeds=0,total_classifier_seeds=3,
                               training=False,encoder_updates=0))
    print('PREPARED_LATEST_NATIVE_ROI_CLASSIFIER_BRIDGE',flush=True)


def scores(plan,classes,spec):
    truth={str(r['source_id']):r['class_id'] for r in plan['oracle']};result={};names=M.NAMES
    oracle_y=[r['class_id'] for r in plan['oracle']];oracle_p=[int(classes[r['case_index']]) for r in plan['oracle']]
    result['manual_oracle']=dict(macro_f1=float(M.f1_score(oracle_y,oracle_p,labels=range(4),average='macro',zero_division=0)))
    by_stem={r['stem']:r for r in plan['items']}
    for name,model in plan['models'].items():
        result[name]={}
        for budget in ['300','600','1071']:
            threshold=spec['thresholds'][name][budget];located=set();correct=set();candidate_count=0;tp=0;joint=0
            class_gt=Counter();class_tp=Counter()
            for stem,entry in model['items'].items():
                labels=by_stem[stem]['labels'];rows=[r for r in entry['predictions'] if r['score']>=threshold]
                boxes=np.asarray([r['box'] for r in rows],np.float32).reshape(-1,4)
                confidence=np.asarray([r['score'] for r in rows],np.float32)
                matches=B.match_image([(0,*l['box']) for l in labels],boxes,confidence,np.zeros(len(rows),np.float32),.5,threshold,class_aware=False)
                candidate_count+=len(rows)
                for label,match in zip(labels,matches):
                    typ=int(label['class_id']);class_gt[typ]+=1
                    if not match[1]:continue
                    sid=str(label['source_id']);located.add(sid);tp+=1
                    value=int(classes[rows[match[4]]['case_index']])
                    if value==truth[sid]:correct.add(sid);joint+=1;class_tp[typ]+=1
            expected=spec['expected_detection_metrics'][name][budget]
            assert candidate_count==expected['candidates']==int(budget) and tp==expected['tp']
            result[name][budget]=dict(candidates=candidate_count,located_IDs=len(located),jointly_correct_IDs=len(correct),
                                      joint_unique_recall=len(correct)/183,located_appearances=tp,joint_TP_appearances=joint,
                                      joint_occurrence_recall=joint/560,reference_joint_precision=joint/candidate_count,
                                      correct_source_ids=sorted(correct,key=int),
                                      per_class={n:dict(unique_GT=sum(t==i for t in truth.values()),
                                                        located_IDs=sum(truth[s]==i for s in located),
                                                        joint_correct_IDs=sum(truth[s]==i for s in correct),
                                                        GT_appearances=class_gt[i],joint_TP_appearances=class_tp[i]) for i,n in enumerate(names)})
    return result


def infer(model,data,start,stop,queue,arm,seed):
    values=[]
    for offset in range(start,stop,8):
        ids=np.arange(offset,min(offset+8,stop));values.append(M.predict(model,data,ids)[1])
        if (offset-start)%128==0:
            queue.update(phase='classifier_inference',classifier=arm,seed=seed,completed_ROIs=offset+len(ids),total_ROIs=10893);write(OUT/'queue.json',queue)
    return np.concatenate(values)


def run():
    spec=C.read_json(OUT/'protocol.json');assert C.sha256(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for p,d in spec['files_sha256'].items():assert C.sha256(p)==d,p
    state=C.read_json(OUT/'queue.json');assert state['status'] in ['prepared_not_started','failed_resumable','paused_by_user']
    queue=dict(status='running',started_utc=state.get('started_utc',utc()),pid=os.getpid(),phase='waiting_verified_native_sources',
               training=False,encoder_updates=0,completed_classifier_seeds=0,total_classifier_seeds=3)
    write(OUT/'queue.json',queue);ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        deadline=time.time()+6*3600
        while True:
            restored=C.read_json(MAINT/'eval_restore_status.json')
            if restored['status']=='completed':break
            assert time.time()<deadline,'Native restoration not completed within finite6-hour wait'
            time.sleep(10)
        plan=C.read_json(PREP/'input_plan.json');sources=restored['tiles']
        for tile,src in sources.items():assert Path(src['path']).is_file(),tile
        # Cropping is finite and CPU-only. Preserve archived3396 crops; build new7497.
        B.OUT=OUT;B.CACHE=Path(restored['cache']);(PREP/'pending_inputs').mkdir(exist_ok=True)
        fresh=dict(plan,cases=plan['cases'][3396:]);B.build_crops(fresh,sources,queue)
        pixels={str(r['crop']):C.sha256(r['crop']) for r in plan['cases']}
        write(OUT/'crop_files_sha256.json',pixels)
        M.P.R.ensure_gpu_idle();M.FENV.prevent_network();M.device_setup();torch.set_num_threads(2)
        dspec=C.read_json(D.OUT/'protocol.json');cspec=C.read_json(E.OUT/'protocol.json')
        dtrain=M.load_data(dspec);ctrain=E.S.load(cspec);external=D.external_data(plan)
        conv_external=dict(gray=external['gray'],roi7=torch.from_numpy(np.stack([E.S.F.area_weights(r['target_box224'],7) for r in plan['cases']])))
        results={}
        for seed in SEEDS:
            matrices={};folder=OUT/f'seed{seed}';folder.mkdir(exist_ok=True)
            for arm,klass,parent,source,train,inputs in [('dino',DINO_CLASS,D.OUT,dspec,dtrain,external),
                                                        ('convnext',CONV_CLASS,E.OUT,cspec,ctrain,conv_external)]:
                mode='qv_last2_LoRA' if arm=='dino' else 'last_stage'
                phase='refit' if arm=='dino' else 'full_refit'
                model=klass(source,mode,list(range(1083)),train,0,phase,seed)
                saved=torch.load(parent/f'seed{seed}/last.pt',map_location='cpu',weights_only=False)
                M.load_parameters(model,saved['parameters'])
                # Preserve original tail batch4: joining pools into one batch
                # would otherwise change numerical kernels for the last4 ROIs.
                old=infer(model,inputs,0,3396,queue,arm,seed)
                with np.load(parent/f'seed{seed}/external_predictions.npz') as archived:
                    assert np.array_equal(old,archived['logits']),f'Archived {arm} logits not reproduced'
                new=infer(model,inputs,3396,10893,queue,arm,seed);matrix=np.concatenate([old,new])
                np.savez_compressed(folder/f'{arm}_predictions.npz',logits=matrix,predictions=matrix.argmax(1))
                matrices[arm]=matrix
                del model,saved;gc.collect();torch.cuda.empty_cache()
            joint=((softmax(matrices['dino'].astype(np.float64),axis=1)+softmax(matrices['convnext'].astype(np.float64),axis=1))/2).argmax(1)
            predictions=dict(dino=matrices['dino'].argmax(1),convnext=matrices['convnext'].argmax(1),equal_ensemble=joint)
            results[str(seed)]={arm:scores(plan,value,spec) for arm,value in predictions.items()}
            write(folder/'scores.json',results[str(seed)])
            write(folder/'replay_receipt.json',dict(status='both_archived3396_ROI_logits_bitwise_reproduced',seed=seed,encoder_updates=0,
                                                   dino_checkpoint_sha256=C.sha256(D.OUT/f'seed{seed}/last.pt'),
                                                   convnext_checkpoint_sha256=C.sha256(E.OUT/f'seed{seed}/last.pt')))
            queue.update(completed_classifier_seeds=SEEDS.index(seed)+1);write(OUT/'queue.json',queue)
            print('SCORED_LATEST_DETECTORS',seed,flush=True)
        for p,d in pixels.items():assert C.sha256(p)==d,p
        write(OUT/'comparison.json',dict(status='completed_native_ROI_inference_on_all10_detector_pools',per_seed=results,
                                        protocol_sha256=C.sha256(OUT/'protocol.json'),training=False,encoder_updates=0,
                                        final_test_read=False,development_images=173,unique_target_IDs=183,
                                        candidate_budgets=[300,600,1071],new_native_ROIs=7497,labels_changed=False))
        queue.update(status='completed',phase='complete_no_training',finished_utc=utc());write(OUT/'queue.json',queue)
    except BaseException as error:
        queue.update(status='failed_resumable',error=repr(error),updated_utc=utc());write(OUT/'queue.json',queue);raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        with (OUT/'run.log').open('a',encoding='utf-8',buffering=1) as stream:sys.stdout=stream;sys.stderr=stream;run()
