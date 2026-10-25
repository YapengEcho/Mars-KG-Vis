"""Execute one frozen, prediction-only rank/NMS fusion on existing caches; no model fitting."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.venv/Lib/site-packages'), str(ROOT / 'tools/maintenance')]
import numpy as np
import verify_optimizer_pair_20261025 as V

OUT = ROOT / 'runs/research/optimizer_rank_ensemble_20261025_v1'


def generate(sources):
    """No labels accepted here. Ranking is global within each component detector."""
    by_stem = {}
    for source in sources:
        pred = V.read(source['predictions']); flat = []
        for stem, arrays in pred.items():
            boxes, scores, classes = [np.asarray(a, np.float32) for a in arrays]
            assert np.isfinite(boxes).all() and np.isfinite(scores).all() and np.all(classes == 0)
            for i, (box, score) in enumerate(zip(boxes, scores)):
                flat.append(dict(stem=stem, box=box.tolist(), raw_score=float(score),
                                 source=source['arm'], source_index=i))
        flat.sort(key=lambda r:(-r['raw_score'], r['stem'], r['source_index']))
        for rank, row in enumerate(flat):
            row['source_rank_score'] = 1/(rank+1)
            by_stem.setdefault(row['stem'], []).append(row)
    def order(row):
        return (-row['source_rank_score'], row['source'] != 'sgd_lr1e2', row['stem'], row['source_index'])
    kept = []
    for stem, rows in sorted(by_stem.items()):
        rows.sort(key=order); boxes = np.asarray([r['box'] for r in rows], np.float32)
        active = np.arange(len(rows)); selected = []
        while len(active):
            j = int(active[0]); selected.append(j); remaining = active[1:]
            if not len(remaining):
                break
            lo = np.maximum(boxes[j, :2], boxes[remaining, :2])
            hi = np.minimum(boxes[j, 2:], boxes[remaining, 2:])
            wh = np.maximum(hi-lo, 0); inter = wh[:, 0]*wh[:, 1]
            a = np.prod(boxes[j, 2:]-boxes[j, :2])
            b = np.prod(boxes[remaining, 2:]-boxes[remaining, :2], axis=1)
            overlap = inter/(a+b-inter)
            active = remaining[overlap <= .7]
        # Independently verify every retained pair meets the fixed NMS criterion.
        for i, j in enumerate(selected):
            for k in selected[:i]:
                assert V.iou(rows[j]['box'], rows[k]['box']) <= float(np.float32(.7))+1e-7
        kept.extend(rows[j] for j in selected)
    kept.sort(key=order)
    for rank, row in enumerate(kept):
        row.update(rank_after_nms=rank, score=1/(rank+1))
    # Unique emitted ranks encode the declared tie break; they are not probabilities.
    return kept


def evaluate(items, rows, budget):
    chosen = rows[:budget]; assert len(chosen) == budget
    by_stem = {}
    for row in chosen:
        by_stem.setdefault(row['stem'], []).append(row)
    tp = gt = 0; ids = set(); size = {}
    for item in items:
        pairs = V.match(item['labels'], by_stem.get(item['stem'], []))
        hit = {int(label['source_id']) for label, _ in pairs}
        ids |= hit; tp += len(pairs); gt += len(item['labels'])
        for label in item['labels']:
            x0, y0, x1, y1 = label['box']; side = max(x1-x0, y1-y0)
            name = 'tiny_lt64' if side < 64 else 'small_64_256' if side < 256 else 'medium_256_819' if side < 819 else 'large_ge819'
            group = size.setdefault(name, dict(gt=0, tp=0)); group['gt'] += 1; group['tp'] += int(label['source_id'] in hit)
    assert gt == 560
    return dict(candidates=budget, tp=tp, fp=budget-tp, fn=gt-tp, precision=tp/budget, recall=tp/gt,
                unique_recalled=len(ids), unique_IDs=sorted(ids), by_size=size,
                source_candidates={name:sum(r['source'] == name for r in chosen) for name in ['adamw_lr3e4', 'sgd_lr1e2']})


def main():
    assert not (OUT / 'results.json').exists(), 'Preserve prior experiment'
    spec = V.read(OUT / 'protocol.json')
    assert V.sha(OUT / 'protocol.json') == (OUT / 'protocol.sha256').read_text().strip()
    for path, digest in spec['files_sha256'].items():
        assert V.sha(path) == digest, path
    # Prediction generation for both seeds is committed before loading any reference labels.
    generated = {}
    for seed in [0, 1]:
        sources = [s for s in spec['sources'] if s['detector_seed'] == seed]
        generated[seed] = generate(sources)
        p = OUT / f'seed{seed}_fused_predictions.json'
        V.output(p, generated[seed])
    plan = V.read(ROOT / 'runs/research/latest_detector_roi_preparation_20261024_v1/input_plan.json')
    results = {}
    for seed in [0, 1]:
        results[str(seed)] = {}
        for budget in spec['budgets']:
            fused = evaluate(plan['items'], generated[seed], budget)
            controls = {}
            for source in [s for s in spec['sources'] if s['detector_seed'] == seed]:
                row = V.read(source['metrics'])['budgets'][str(budget)]
                controls[source['arm']] = dict(unique_recalled=row['unique_recalled'],
                    tp=row['metrics']['tp'], precision=row['metrics']['precision'], recall=row['metrics']['recall'])
            results[str(seed)][str(budget)] = dict(fused=fused, controls=controls)
            print('FUSION', seed, budget, fused['unique_recalled'], 'IDs; TP', fused['tp'],
                  'controls', {a:r['unique_recalled'] for a,r in controls.items()}, flush=True)
    V.output(OUT / 'results.json', dict(status='completed_one_fixed_prediction_only_rank_NMS_fusion',
        results=results, protocol_sha256=V.sha(OUT / 'protocol.json'), implementation_sha256=V.sha(__file__),
        prediction_files_sha256={str(OUT / f'seed{s}_fused_predictions.json'):V.sha(OUT / f'seed{s}_fused_predictions.json') for s in [0,1]},
        classification_not_evaluated=True, training=False, downloads=False, final_test_read=False,
        AP_not_reported=True, scores_are_ranks_not_probabilities=True))


if __name__ == '__main__':
    main()
