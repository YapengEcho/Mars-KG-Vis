"""Fixed full-cohort ConvNeXt fits and equal ensemble on existing detector ROIs."""
from copy import deepcopy
import ctypes
import gc
import os
from pathlib import Path
import sys
import time

import run_convnext_stage_repeat_20261023 as R
import run_full_lora_bridge as B

S,M,C,np,torch=R.S,R.M,R.C,R.np,R.torch
ROOT=R.ROOT
OUT=ROOT/'runs/experiments/convnext_dino_bridge_20261024_v1'
TASK='MarsKG-convnext-dino-bridge-20261024-v1'
AUDIT=ROOT/'runs/research/convnext_dino_disagreement_20261024_v1'


def bind():
    S.OUT=OUT;S.SEED=20261015
    M.OUT=OUT;M.Classifier=S.Classifier;M.write=S.write;M.weights=R.N.H.weights


def prepare():
    assert not OUT.exists(),'Preserve existing bridge'
    R.bind();parent=S.check();old=B.check()
    assert C.read_json(R.OUT/'independent_verification_20261024.json')['actual_refit_replays']==18
    assert C.read_json(AUDIT/'independent_recomputation.json')['reconstructed_reports']==9
    assert C.read_json(B.OUT/'queue.json')['status']=='completed_full_training_LoRA_bridge'
    assert C.read_json(B.OUT/'completion_verification.json')['status'].startswith('passed')
    plan=C.read_json(old['bridge_plan'])
    assert len(plan['cases'])==3396 and len(plan['items'])==173 and len(plan['oracle'])==183
    assert not {str(r['source_id']) for r in parent['design']['rows']}&{str(r['source_id']) for r in plan['oracle']}
    spec=deepcopy(parent);rules={};pins={}
    for key in ['new_seeds','source_seed_reused_read_only','repeat_policy','exploratory_seed_count']:
        spec.pop(key,None)
    for seed in R.ALL_SEEDS:
        base=R.SOURCE if seed==20261015 else R.OUT
        selected=[]
        for fold in range(3):
            path=base/'last_stage'/f'seed{seed}'/f'fold{fold}/select/completed.json'
            selected.append(C.read_json(path)['best_epoch']);pins[str(path)]=C.sha256(path)
        rules[str(seed)]=dict(inner_selected_epochs=selected,full_training_epochs=int(np.median(selected)),
            rule='Median of three preexisting inner-stop selections; no external scores or outer OOF used to choose epochs')
        for p in [B.OUT/f'seed{seed}/external_predictions.npz',B.OUT/f'seed{seed}/completed.json']:
            pins[str(p)]=C.sha256(p)
    for p in [Path(old['bridge_plan']),B.OUT/'protocol.json',B.OUT/'comparison.json',B.OUT/'completion_verification.json',
              R.OUT/'protocol.json',R.OUT/'comparison.json',R.OUT/'independent_verification_20261024.json',
              AUDIT/'protocol.json',AUDIT/'comparison.json',AUDIT/'independent_recomputation.json']:
        pins[str(p)]=C.sha256(p)
    # Preserve all source bridge input, crop, coordinate and training provenance guards.
    pins.update(old['files_sha256'])
    spec['files_sha256'].update(pins)
    spec.update(created_utc=M.utc(),seed=20261015,seeds=R.ALL_SEEDS,epoch_rules=rules,
        bridge_plan=old['bridge_plan'],training_arms=['last_stage'],evaluation_arms=['dino','convnext','equal_ensemble'],
        combination='Softmax each four-class logits at T1; arithmetic mean weight0.5 each, no tuning',
        training='Fresh ConvNeXt original pretraining and all1083 train IDs; last_stage only; no CV held model deployment',
        external_evaluation='Existing173-image development cohort,183 target IDs,560 box appearances; not final blind test',
        selection='Fixed epochs from source inner medians only, no external early stopping or seed selection',
        primary='Same frozen1071 candidates per detector, paired DINO versus equal ensemble joint unique-target recall',
        automatic_following_experiments=False,final_test_read=False,
        limitations=parent['limitations']+[
            'Existing detector pools are historical full/partial/P2; do not imply results for latest recipe detector',
            'Full-cohort training has no validation-based early stopping; fixed epochs do not ensure optimal fit',
            'External cohort is repeatedly used development data; FL only2 targets, CL15',
            'Joint precision uses reference annotations and may count unlabelled landslides as unmatched'])
    for p in [Path(__file__),ROOT/'data/scripts/start_convnext_dino_bridge_20261024.ps1',Path(B.__file__)]:
        spec['code_hashes'][str(p)]=C.sha256(p)
    OUT.mkdir(parents=True);S.write(OUT/'protocol.json',spec)
    (OUT/'protocol.sha256').write_text(C.sha256(OUT/'protocol.json')+'\n')
    S.write(OUT/'queue.json',dict(status='prepared_not_started',completed_units=0,total_units=3))
    bind();S.check();print('PREPARED_FIXED_BRIDGE',rules,flush=True)


def external_data(plan):
    images=[];weights=[]
    for case in plan['cases']:
        with S.F.Image.open(case['crop']) as im:
            assert im.mode=='L' and im.size==(224,224);images.append(np.array(im))
        weights.append(S.F.area_weights(case['target_box224'],7))
    return dict(gray=np.stack(images),roi7=torch.from_numpy(np.stack(weights)))


def train(spec,data,seed,queue):
    fit=list(range(1083));limit=spec['epoch_rules'][str(seed)]['full_training_epochs']
    folder=OUT/f'seed{seed}';folder.mkdir(exist_ok=True)
    model=S.Classifier(spec,'last_stage',fit,data,0,'refit',seed)
    opt=M.optimizer(model,spec);w=M.weights(data,fit);history=[];start=1
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        assert saved['protocol_sha256']==C.sha256(OUT/'protocol.json')
        assert (saved['arm'],saved['phase'],saved['run_seed'])==('last_stage','full_refit',seed)
        M.load_parameters(model,saved['parameters']);opt.load_state_dict(saved['optimizer'])
        history=saved['history'];start=saved['completed_epoch']+1;del saved
    for epoch in range(start,limit+1):
        order=np.random.default_rng(seed+10000+epoch).permutation(1083);losses=[]
        for offset in range(0,len(order),16):
            losses.append(M.update_group(model,opt,data,order[offset:offset+16],w))
            if offset%128==0:
                queue.update(phase='full_training',seed=seed,epoch=epoch,epochs=limit,completed_images=offset)
                S.write(OUT/'queue.json',queue)
        history.append(dict(epoch=epoch,loss=float(np.mean(losses))))
        M.save_checkpoint(folder/'last.pt',model,opt,history,epoch,0,-1.,'last_stage',0,'full_refit',seed)
        S.write(folder/'training.json',dict(history=history,fit_IDs=1083,fixed_epochs=limit,external_scores_used=False))
        print('FULL_CONVNEXT',seed,epoch,limit,history[-1],flush=True)
    S.write(folder/'completed.json',dict(status='completed',seed=seed,epochs=limit,fit_IDs=1083,
        checkpoint_sha256=C.sha256(folder/'last.pt')))
    del opt;return model


def softmax(x):
    x=x.astype(np.float64);x=np.exp(x-x.max(axis=1,keepdims=True));return x/x.sum(axis=1,keepdims=True)


def scores(plan,classes):
    result=B.score_predictions(plan,classes)
    truth={str(r['source_id']):int(r['class_id']) for r in plan['oracle']}
    items={x['stem']:x for x in plan['items']}
    for name,model in plan['models'].items():
        located=set();correct=set()
        for stem,entry in model['items'].items():
            for match,label in zip(entry['matches'],items[stem]['labels']):
                if not match['matched']:continue
                sid=str(label['source_id']);located.add(sid)
                case=entry['predictions'][match['kept_prediction_index']]['case_index']
                if int(classes[case])==truth[sid]:correct.add(sid)
        result['pipeline'][name]['unique_targets']=dict(total=len(truth),located=len(located),
            jointly_correct=len(correct),joint_recall=len(correct)/len(truth),
            correctly_typed_IDs=sorted(correct,key=int),per_class={n:dict(total=sum(v==i for v in truth.values()),
                jointly_correct=sum(truth[s]==i for s in correct)) for i,n in enumerate(M.NAMES)})
    return result


def scheduled(resume=False):
    bind();spec=S.check();state=C.read_json(OUT/'queue.json')
    assert state['status'] in (['failed','paused_by_user'] if resume else ['prepared_not_started'])
    independent=M.FENV.process_independence();assert independent['scheduler_owned']
    M.P.R.ensure_gpu_idle();blocked=M.FENV.prevent_network();M.device_setup();torch.set_num_threads(2)
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    queue=dict(status='running',pid=os.getpid(),started_utc=state.get('started_utc',M.utc()),phase='smoke',
        completed_units=0,total_units=3,offline=True,temporary_system_awake=True,process_independence=independent)
    S.write(OUT/'queue.json',queue)
    try:
        data=S.load(spec)
        if not resume:S.smoke(spec,data)
        assert C.read_json(OUT/'readiness.json')['protocol_sha256']==C.sha256(OUT/'protocol.json')
        plan=C.read_json(spec['bridge_plan']);external=external_data(plan);per_seed={}
        for seed in R.ALL_SEEDS:
            model=train(spec,data,seed,queue);pieces=[]
            for offset in range(0,3396,8):
                ix=np.arange(offset,min(offset+8,3396));pieces.append(M.predict(model,external,ix)[1])
                if offset%128==0:
                    queue.update(phase='external_inference',seed=seed,completed_ROIs=offset+len(ix),total_ROIs=3396)
                    S.write(OUT/'queue.json',queue)
            logits=np.concatenate(pieces);classes=logits.argmax(1)
            np.savez_compressed(OUT/f'seed{seed}/external_predictions.npz',logits=logits,predictions=classes)
            del model;gc.collect();torch.cuda.empty_cache()
            # Full reload and complete-pool replay, not merely a sampled checkpoint test.
            saved=torch.load(OUT/f'seed{seed}/last.pt',map_location='cpu',weights_only=False)
            model=S.Classifier(spec,'last_stage',list(range(1083)),data,0,'refit',seed)
            M.load_parameters(model,saved['parameters']);p,z=M.predict(model,external,np.arange(3396))
            assert np.array_equal(p,classes) and np.array_equal(z,logits)
            del saved,model;gc.collect();torch.cuda.empty_cache()
            with np.load(B.OUT/f'seed{seed}/external_predictions.npz') as old:
                dl=old['logits'].copy();assert np.array_equal(old['predictions'],dl.argmax(1))
            assert dl.shape==logits.shape==(3396,4)
            combined=(.5*softmax(dl)+.5*softmax(logits)).argmax(1)
            computed={name:scores(plan,pred) for name,pred in [('dino',dl.argmax(1)),('convnext',classes),('equal_ensemble',combined)]}
            previous=C.read_json(B.OUT/'comparison.json')['scores'][str(seed)]
            assert computed['dino']['oracle']==previous['oracle']
            for name,pipeline in previous['pipeline'].items():
                assert {k:v for k,v in computed['dino']['pipeline'][name].items() if k!='unique_targets'}==pipeline
            per_seed[str(seed)]=computed
            S.write(OUT/f'seed{seed}/evaluation.json',computed)
            S.write(OUT/f'seed{seed}/prediction_replay.json',dict(status='passed_full3396_ROI_checkpoint_replay',
                checkpoint_sha256=C.sha256(OUT/f'seed{seed}/last.pt'),prediction_sha256=C.sha256(OUT/f'seed{seed}/external_predictions.npz')))
            queue['completed_units']+=1;S.write(OUT/'queue.json',queue)
        S.check();S.write(OUT/'comparison.json',dict(status='completed_fixed_equal_ensemble_detector_bridge',
            per_seed=per_seed,seeds=R.ALL_SEEDS,validation_images=173,unique_targets=183,GT_appearances=560,
            final_test_read=False,external_selection=False,protocol_sha256=C.sha256(OUT/'protocol.json')))
        queue.update(status='completed',phase='complete')
    except BaseException as error:queue.update(status='failed',error=repr(error));raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        queue.update(finished_utc=M.utc(),temporary_system_awake=False,blocked_network_attempts=blocked)
        S.write(OUT/'queue.json',queue)


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    elif mode in ['scheduled-run','scheduled-resume']:
        stamp=str(time.time_ns())
        sys.stdout=(OUT/f'training_{stamp}.log').open('x',encoding='utf-8',buffering=1)
        sys.stderr=(OUT/f'training_{stamp}.err.log').open('x',encoding='utf-8',buffering=1)
        scheduled(mode=='scheduled-resume')
    else:raise ValueError(mode)
