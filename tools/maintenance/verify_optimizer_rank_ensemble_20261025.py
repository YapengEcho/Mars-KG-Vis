"""Independent Torchvision NMS replay and fixed-budget fusion metric verification."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.venv/Lib/site-packages'), str(ROOT / 'tools/maintenance'), str(ROOT / 'data/scripts')]
import numpy as np
import torch
from torchvision.ops import nms
from eval_recall_report import match_image
import verify_optimizer_pair_20261025 as V

OUT = ROOT / 'runs/research/optimizer_rank_ensemble_20261025_v1'
JOINT = ROOT / 'runs/experiments/optimizer_joint_bridge_20261025_v1'


def main():
    spec = V.read(OUT / 'protocol.json'); report = V.read(OUT / 'results.json')
    assert report['protocol_sha256'] == V.sha(OUT / 'protocol.json')
    assert report['implementation_sha256'] == V.sha(ROOT / 'tools/maintenance/run_optimizer_rank_ensemble_20261025.py')
    for path, digest in spec['files_sha256'].items():
        assert V.sha(path) == digest
    plan = V.read(ROOT / 'runs/research/latest_detector_roi_preparation_20261024_v1/input_plan.json')
    joint_plan = V.read(JOINT / 'input_plan.json')
    items = {i['stem']:i for i in plan['items']}; matched_cases = {}; unknown = []
    cells = []; total_NMS = 0
    for seed in [0, 1]:
        by_stem = {}
        for source in [s for s in spec['sources'] if s['detector_seed'] == seed]:
            raw = V.read(source['predictions']); ranked = []
            for stem, arrays in raw.items():
                for i, score in enumerate(arrays[1]):
                    ranked.append((float(np.float32(score)), stem, i, arrays[0][i]))
            ranked.sort(key=lambda x:(-x[0], x[1], x[2]))
            for rank, (_, stem, i, box) in enumerate(ranked):
                key = (-1/(rank+1), source['arm'] != 'sgd_lr1e2', stem, i)
                by_stem.setdefault(stem, []).append((key, source['arm'], i, box))
        expected_order = []
        for stem, entries in by_stem.items():
            entries.sort(key=lambda x:x[0])
            boxes = torch.tensor(np.asarray([e[3] for e in entries], np.float32))
            # Strictly ordered integer scores independently implement the declared tie resolution.
            order = nms(boxes, torch.arange(len(entries), 0, -1, dtype=torch.float32), .7).tolist()
            expected_order.extend(entries[j] for j in order); total_NMS += 1
        expected_order.sort(key=lambda x:x[0])
        path = OUT / f'seed{seed}_fused_predictions.json'; actual = V.read(path)
        assert V.sha(path) == report['prediction_files_sha256'][str(path)]
        assert len(actual) == len(expected_order)
        for rank, (row, expected) in enumerate(zip(actual, expected_order)):
            assert (row['stem'], row['source'], row['source_index']) == (expected[0][2], expected[1], expected[2])
            assert np.array_equal(np.asarray(row['box'], np.float32), np.asarray(expected[3], np.float32))
            assert row['rank_after_nms'] == rank and row['score'] == 1/(rank+1)
        for budget in spec['budgets']:
            selected = {}; tp = ngt = 0; ids = set(); size = {}; needed = []
            for row in actual[:budget]:
                selected.setdefault(row['stem'], []).append(row)
            for stem, item in items.items():
                rows = selected.get(stem, [])
                hit = match_image([(0,*l['box']) for l in item['labels']],
                    np.asarray([r['box'] for r in rows], np.float32).reshape(-1,4),
                    np.asarray([r['score'] for r in rows], np.float32), np.zeros(len(rows), np.float32), .5, 0., class_aware=False)
                ngt += len(item['labels'])
                for label, match in zip(item['labels'], hit):
                    group = size.setdefault(match[2], dict(gt=0, tp=0)); group['gt'] += 1; group['tp'] += int(match[1])
                    if not match[1]:
                        continue
                    tp += 1; ids.add(int(label['source_id']))
                    row = rows[match[4]]
                    name = ('optimizer_' if seed == 0 else 'optimizer_seed1_')+row['source']
                    candidates = joint_plan['models'][name]['items'][stem]['predictions']
                    candidate = next((r for r in candidates if r['original_index'] == row['source_index']), None)
                    index = candidate['case_index'] if candidate else None
                    if index is None:
                        needed.append(dict(seed=seed, budget=budget, stem=stem, source=row['source'],
                                           source_index=row['source_index'], box=row['box'], source_id=label['source_id']))
            expected = report['results'][str(seed)][str(budget)]['fused']
            assert ngt == 560 and len(actual[:budget]) == budget
            assert tp == expected['tp'] and expected['unique_IDs'] == sorted(ids)
            assert expected['precision'] == tp/budget and expected['recall'] == tp/ngt
            assert expected['by_size'] == size
            cells.append(dict(seed=seed,budget=budget,unique_recalled=len(ids),tp=tp))
            matched_cases[f'{seed}:{budget}'] = dict(matched_appearances=tp, missing_subtype_cases=len(needed))
            unknown.extend(needed)
    V.output(OUT / 'independent_verification.json', dict(status='passed_official_NMS_replay_and_vector_matching',
        verified_image_NMS=total_NMS, verified_budget_cells=cells, verifier_sha256=V.sha(__file__),
        results_sha256=V.sha(OUT / 'results.json'), predictions_preserve_original_coordinates=True,
        final_test_read=False, training=False, downloads=False))
    V.output(OUT / 'pending_subtype_inputs.json', dict(status='geometry_audit_only_not_type_scores',
        coverage=matched_cases, missing=unknown, classification_evaluated=False,
        joint_scores_sha256=V.sha(JOINT / 'comparison.json'), fusion_results_sha256=V.sha(OUT / 'results.json')))
    print('VERIFIED_FUSION',total_NMS,'image NMS;',len(cells),'budgets; type coverage',matched_cases,flush=True)


if __name__ == '__main__':
    main()
