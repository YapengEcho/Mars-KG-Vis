"""Independently replay vector matches and subtype lookups for the fixed rank fusion."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.venv/Lib/site-packages'),str(ROOT/'data/scripts'),str(ROOT/'tools/maintenance')]
import numpy as np
from eval_recall_report import match_image
import verify_optimizer_pair_20261025 as V

OUT=ROOT/'runs/research/optimizer_rank_ensemble_20261025_v1'
PARENT=ROOT/'runs/experiments/optimizer_joint_bridge_20261025_v1'


def main():
    report=V.read(OUT/'joint_results.json');plan=V.read(PARENT/'input_plan.json')
    assert V.sha(OUT/'type_supplement/predictions.json')==report['supplement_sha256']
    assert V.sha(OUT/'results.json')==report['fusion_results_sha256']
    spec=V.read(OUT/'type_supplement/protocol.json');supp=V.read(OUT/'type_supplement/predictions.json')
    for path,digest in spec['files_sha256'].items():assert V.sha(path)==digest,path
    assert V.sha(spec['case']['crop'])==supp['native_crop_sha256']
    for r in supp['values'].values():
        values=np.asarray(r['logits']);assert values.shape==(4,) and np.isfinite(values).all()
        assert int(values.argmax())==r['prediction'] and r['replay_bitwise']
    items={i['stem']:i for i in plan['items']};truth={str(r['source_id']):r['class_id'] for r in plan['oracle']}
    verified=0
    for detseed in [0,1]:
        pred=V.read(OUT/f'seed{detseed}_fused_predictions.json')
        for budget in [300,600,1071]:
            by_stem={}
            for r in pred[:budget]:by_stem.setdefault(r['stem'],[]).append(r)
            for seed in [20261015,20261016,20261017]:
                with np.load(PARENT/f'seed{seed}/dino_predictions.npz') as old:classes=old['predictions']
                correct=[];located=set()
                for stem,item in items.items():
                    rows=by_stem.get(stem,[])
                    matches=match_image([(0,*l['box']) for l in item['labels']],
                        np.asarray([r['box'] for r in rows],np.float32).reshape(-1,4),
                        np.asarray([r['score'] for r in rows],np.float32),np.zeros(len(rows),np.float32),.5,0.,class_aware=False)
                    for label,hit in zip(item['labels'],matches):
                        if not hit[1]:continue
                        sid=str(label['source_id']);located.add(sid);row=rows[hit[4]]
                        name=('optimizer_' if detseed==0 else 'optimizer_seed1_')+row['source']
                        candidates=plan['models'][name]['items'][stem]['predictions']
                        old_case=next((r for r in candidates if r['original_index']==row['source_index']),None)
                        index=old_case['case_index'] if old_case else None
                        if index is None:
                            assert [detseed,stem,row['source'],row['source_index']]==supp['missing_key']
                            value=supp['values'][str(seed)]['prediction']
                        else:value=int(classes[index])
                        if value==truth[sid]:correct.append(sid)
                expected=report['results'][str(detseed)][str(budget)][str(seed)];ids=set(correct)
                assert expected['jointly_correct_IDs']==len(ids) and expected['located_IDs']==len(located)
                assert expected['correct_source_ids']==sorted(ids,key=int)
                assert expected['joint_TP_appearances']==len(correct)
                assert expected['reference_joint_precision']==len(correct)/budget
                assert expected['joint_occurrence_recall']==len(correct)/560
                assert expected['joint_unique_recall']==len(ids)/183
                assert expected['per_class_joint_correct_IDs']=={n:sum(truth[s]==i for s in ids) for i,n in enumerate(V.NAMES)}
                verified+=1
    V.output(OUT/'joint_independent_verification.json',dict(status='passed_vector_matching_and_subtype_reconstruction',
        verified_score_cells=verified,joint_results_sha256=V.sha(OUT/'joint_results.json'),
        verifier_sha256=V.sha(__file__),training=False,downloads=False,final_test_read=False))
    print('VERIFIED_FUSION_JOINT',verified,'cells',flush=True)


if __name__=='__main__':main()
