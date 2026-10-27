"""Infer the single newly matched ROI, then score fused detections using fixed archived classifiers."""
from pathlib import Path
import sys
import gc

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.venv/Lib/site-packages'), str(ROOT / 'data/scripts'), str(ROOT / 'tools/maintenance')]
import run_optimizer_joint_bridge_20261025 as R
import verify_optimizer_pair_20261025 as V
import numpy as np

OUT = ROOT / 'runs/research/optimizer_rank_ensemble_20261025_v1'
PARENT = R.OUT
SUPP = OUT / 'type_supplement'


def main():
    assert not (OUT / 'joint_results.json').exists()
    assert V.read(OUT / 'independent_verification.json')['status'] == 'passed_official_NMS_replay_and_vector_matching'
    plan = V.read(PARENT / 'input_plan.json'); spec = V.read(PARENT / 'protocol.json')
    missing = V.read(OUT / 'pending_subtype_inputs.json')['missing']
    unique = {(r['seed'],r['stem'],r['source'],r['source_index']):r for r in missing}
    assert len(unique) == 1, 'This supplement is restricted to the one audited ROI'
    key, row = next(iter(unique.items())); item = next(i for i in plan['items'] if i['stem'] == row['stem'])
    box = np.clip(np.asarray(row['box'],np.float64),[0]*4,[item['width'],item['height']]*2)
    native = R.B.native_box(box,item); win,geom = R.B.planned_window(native,plan['sources'][item['tile']])
    entries = V.read(R.NATIVE / item['tile'] / 'manifest.json')['entries']
    assert R.K.covered(win,[e['window_px'] for e in entries]), 'Missing pixels; do not download automatically'
    side = win[2]-win[0]
    case = dict(tile=item['tile'],box_native=native,window_px=win,geometry=geom,
                target_box224=[(native[0]-win[0])*224/side,(native[1]-win[1])*224/side,
                               (native[2]-win[0])*224/side,(native[3]-win[1])*224/side],
                crop=str(SUPP/'single_native_ROI.png'))
    SUPP.mkdir(exist_ok=False)
    R.OUT = SUPP
    pins = {str(p):V.sha(p) for p in [Path(__file__),PARENT/'protocol.json',PARENT/'comparison.json',
            OUT/'protocol.json',OUT/'results.json',OUT/'independent_verification.json',
            R.NATIVE/item['tile']/'manifest.json',Path(R.__file__)]}
    for s in R.SEEDS:
        p = R.D.OUT/f'seed{s}/last.pt';pins[str(p)] = V.sha(p)
    V.output(SUPP/'protocol.json',dict(files_sha256=pins,missing_key=list(key),case=case,
        classifier_seeds=R.SEEDS,replay_first_8_archived_logits=True,padded_batch_size=8,
        training=False,downloads=False,final_test_read=False))
    reader=R.NativeReader(budget=512<<20)
    try:
        receipt=R.render(case,plan,R.sources(spec),reader,Path(case['crop']))
        V.output(SUPP/'native_receipt.json',receipt)
    finally:
        while reader.cache:reader.evict()
    external=R.D.external_data(dict(cases=plan['cases'][:8]+[case]+plan['cases'][:7]))
    R.M.P.R.ensure_gpu_idle();R.M.FENV.prevent_network();R.M.device_setup();R.torch.set_num_threads(2)
    ds=V.read(R.D.OUT/'protocol.json');train=R.M.load_data(ds);supplement={}
    for seed in R.SEEDS:
        model=R.H.SOURCE_CTOR(ds,'qv_last2_LoRA',list(range(1083)),train,0,'refit',seed)
        saved=R.torch.load(R.D.OUT/f'seed{seed}/last.pt',map_location='cpu',weights_only=False)
        R.M.load_parameters(model,saved['parameters'])
        replay=R.M.predict(model,external,np.arange(8))[1]
        with np.load(PARENT/f'seed{seed}/dino_predictions.npz') as old:
            assert np.array_equal(replay,old['logits'][:8]),'Supplement weights/input replay failed'
        logits=R.M.predict(model,external,np.arange(8,16))[1][0]
        supplement[str(seed)]=dict(logits=logits.tolist(),prediction=int(logits.argmax()),replay_bitwise=True)
        del model,saved;gc.collect();R.torch.cuda.empty_cache()
    V.output(SUPP/'predictions.json',dict(status='completed_single_native_ROI_fixed_three_seed_inference',
        values=supplement,missing_key=list(key),native_crop_sha256=V.sha(case['crop']),training=False,downloads=False))
    # One subtype is now available for every fixed matched pair, including the audited supplement.
    items={i['stem']:i for i in plan['items']};truth={str(r['source_id']):r['class_id'] for r in plan['oracle']}
    result={}
    for detseed in [0,1]:
        rows=V.read(OUT/f'seed{detseed}_fused_predictions.json');result[str(detseed)]={}
        for budget in [300,600,1071]:
            selected={}
            for r in rows[:budget]:selected.setdefault(r['stem'],[]).append(r)
            hits=[]
            for stem,item in items.items():hits.extend((stem,l,r) for l,r in V.match(item['labels'],selected.get(stem,[])))
            expected=V.read(OUT/'results.json')['results'][str(detseed)][str(budget)]['fused']
            assert len(hits)==expected['tp']
            scores={}
            for seed in R.SEEDS:
                with np.load(PARENT/f'seed{seed}/dino_predictions.npz') as z:predictions=z['predictions']
                correct=[];classes={n:0 for n in V.NAMES};located={str(l['source_id']) for _,l,_ in hits}
                for stem,label,r in hits:
                    name=('optimizer_' if detseed==0 else 'optimizer_seed1_')+r['source']
                    entry=next((p for p in plan['models'][name]['items'][stem]['predictions'] if p['original_index']==r['source_index']),None)
                    index=entry['case_index'] if entry else None
                    if index is None:
                        assert (detseed,stem,r['source'],r['source_index'])==key
                        value=supplement[str(seed)]['prediction']
                    else:value=int(predictions[index])
                    sid=str(label['source_id'])
                    if value==truth[sid]:correct.append(sid)
                ids=set(correct)
                for sid in ids:classes[V.NAMES[truth[sid]]]+=1
                scores[str(seed)]=dict(candidates=budget,located_IDs=len(located),jointly_correct_IDs=len(ids),
                    joint_TP_appearances=len(correct),reference_joint_precision=len(correct)/budget,
                    joint_occurrence_recall=len(correct)/560,joint_unique_recall=len(ids)/183,
                    correct_source_ids=sorted(ids,key=int),per_class_joint_correct_IDs=classes)
            result[str(detseed)][str(budget)]=scores
    V.output(OUT/'joint_results.json',dict(status='completed_fixed_three_seed_rank_fusion_joint_evaluation',
        results=result,supplement_sha256=V.sha(SUPP/'predictions.json'),input_scores_sha256=V.sha(PARENT/'comparison.json'),
        fusion_results_sha256=V.sha(OUT/'results.json'),implementation_sha256=V.sha(__file__),training=False,downloads=False,final_test_read=False))
    for s,rows in result.items():
        print('FUSION_JOINT',s,{b:float(np.mean([r['jointly_correct_IDs'] for r in values.values()])) for b,values in rows.items()},flush=True)


if __name__=='__main__':main()
