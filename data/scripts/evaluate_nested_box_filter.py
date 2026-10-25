"""Validation-selected containment filter on existing frozen prediction caches.

The rule uses predictions only. Human labels are used for scoring, never for
deciding which box to remove. This is retrospective development evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np

from eval_recall_report import iou_matrix, match_image
from reevaluate_rebuilt_six import validate_frozen
from review_eval_labels import ROOT, atomic, read, sha

FROZEN = ROOT / 'data/processed/manual_eval_frozen_20261011_v1'
CACHE = ROOT / 'runs/ablations/rebuilt_six_reevaluation_20261011_v1'
OUT = ROOT / 'runs/ablations/nested_box_filter_20261013_v1'


def filter_boxes(boxes, scores, image_area, policy):
    """Remove outer boxes supported by two distinct, retained smaller boxes."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32)
    keep = np.ones(len(boxes), dtype=bool)
    evidence = {}
    if policy is None or len(boxes) < 3:
        return keep, evidence
    area = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    lower = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    upper = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    intersection = np.maximum(upper - lower, 0).prod(axis=2)
    covered = np.divide(intersection, area[None, :], out=np.zeros_like(intersection), where=area[None, :] > 0)
    ious = iou_matrix(boxes, boxes)
    # Smallest first: already suppressed children cannot suppress their parent.
    for outer in np.argsort(area, kind='stable'):
        if area[outer] < policy['min_outer_area_fraction'] * image_area:
            continue
        children = np.flatnonzero(keep & (area > 0) & (area <= .5 * area[outer])
            & (covered[outer] >= .9) & (scores >= policy['child_score_ratio'] * scores[outer]))
        pair = next(((int(a), int(b)) for n, a in enumerate(children) for b in children[n + 1:]
                     if ious[a, b] <= .3), None)
        if pair is not None:
            keep[outer] = False
            evidence[int(outer)] = list(pair)
    return keep, evidence


def score(items, predictions, threshold, policy=None):
    counts = Counter()
    size_groups = {}
    hits = set()
    removed = []
    for item in items:
        boxes, scores, classes = [np.asarray(a, np.float32) for a in predictions[item['stem']]]
        boxes = boxes.reshape(-1, 4)
        if not len(boxes) == len(scores) == len(classes) or np.any(classes != 0):
            raise ValueError('Invalid single-class cache')
        original_indices = np.flatnonzero(scores >= threshold)
        pb, ps, pc = boxes[original_indices], scores[original_indices], classes[original_indices]
        keep, evidence = filter_boxes(pb, ps, item['width'] * item['height'], policy)
        for index, children in evidence.items():
            removed.append(dict(stem=item['stem'], tile=item['tile'],
                prediction_index=int(original_indices[index]), box=pb[index].tolist(), score=float(ps[index]),
                supporting_prediction_indices=[int(original_indices[j]) for j in children]))
        gt = [(0, *b) for b in item['human']['boxes']]
        records = match_image(gt, pb[keep], ps[keep], pc[keep], .5, threshold, class_aware=False)
        counts.update(images=1, gt=len(gt), candidates=int(keep.sum()), tp=sum(int(r[1]) for r in records))
        for index, record in enumerate(records):
            group = size_groups.setdefault(record[2], dict(gt=0, tp=0))
            group['gt'] += 1
            group['tp'] += int(record[1])
            if record[1]:
                hits.add((item['stem'], index))
    summary = dict(counts)
    summary.update(fp=counts['candidates'] - counts['tp'], fn=counts['gt'] - counts['tp'],
        precision=counts['tp'] / counts['candidates'] if counts['candidates'] else 0.,
        recall=counts['tp'] / counts['gt'] if counts['gt'] else 0.,
        fp_per_image=(counts['candidates'] - counts['tp']) / counts['images'], by_size=size_groups)
    return summary, hits, removed


def compare(baseline, changed):
    before, old_hits, _ = baseline
    after, new_hits, removed = changed
    return dict(before=before, after=after, removed_predictions=len(removed),
        lost_target_appearances=len(old_hits - new_hits), gained_target_appearances=len(new_hits - old_hits),
        unmatched_reduction=before['fp'] - after['fp'])


def load_cache(run, split, result_index, items):
    result = result_index[(run, split)]
    path = CACHE / run / 'offset' / split / 'predictions.json'
    if sha(path) != result['predictions_sha256']:
        raise ValueError(f'Changed cache: {path}')
    predictions = read(path)
    if set(predictions) != {item['stem'] for item in items}:
        raise ValueError('Prediction cohort differs from frozen evaluation')
    baseline = score(items, predictions, result['threshold'])
    for key in ('images', 'gt', 'candidates', 'tp', 'fp'):
        if baseline[0][key] != result['metrics'][key]:
            raise ValueError(f'Baseline mismatch: {run}/{split}/{key}')
    return predictions, result['threshold'], baseline, sha(path)


def run(out):
    if out.exists():
        raise FileExistsError('Keep prior audit; choose a new output directory')
    frozen = validate_frozen(FROZEN)
    result_path = CACHE / 'results.json'
    result = read(result_path)
    frozen_hash = sha(FROZEN / 'frozen.json')
    if result['frozen_sha256'] != frozen_hash:
        raise ValueError('Frozen label receipt differs')
    index = {(r['run'], r['split']): r for r in result['results'] if r['arm'] == 'offset'}
    val_items = [r for r in frozen['items'] if (r['arm'], r['split']) == ('offset', 'val')]
    val = {run: load_cache(run, 'val', index, val_items) for run in ('manual_seed0', 'manual_seed1')}
    policies = [None] + [dict(child_score_ratio=ratio, min_outer_area_fraction=area)
                         for ratio, area in product((.5, 1., 1.5), (.05, .1, .25))]
    sweep = []
    for policy in policies:
        comparisons = {}
        for name, (predictions, threshold, baseline, _) in val.items():
            comparisons[name] = compare(baseline, score(val_items, predictions, threshold, policy))
        sweep.append(dict(policy=policy, runs=comparisons,
            feasible=all(r['lost_target_appearances'] == 0 for r in comparisons.values()),
            unmatched_reduction=sum(r['unmatched_reduction'] for r in comparisons.values())))
    feasible = [r for r in sweep if r['feasible']]
    # Tie-break towards the more conservative child score and outer area gates.
    chosen = max(feasible, key=lambda r: (r['unmatched_reduction'],
        r['policy']['child_score_ratio'] if r['policy'] else float('inf'),
        r['policy']['min_outer_area_fraction'] if r['policy'] else float('inf')))
    out.mkdir(parents=True)
    lock = dict(selected_on='offset validation, both manual seeds', policy=chosen['policy'],
        selection='maximize unmatched reduction with zero lost validation target appearances in either manual seed',
        fixed_geometry=dict(inner_area_ratio_max=.5, containment_min=.9, sibling_iou_max=.3),
        frozen_sha256=frozen_hash, code_sha256=sha(Path(__file__)),
        validation_cache_sha256={name: v[3] for name, v in val.items()},
        original_confidence_thresholds_unchanged=True, validation_sweep=sweep)
    atomic(out / 'validation_lock.json', lock)
    # Test predictions are opened only after the validation policy is saved.
    test_items = [r for r in frozen['items'] if (r['arm'], r['split']) == ('offset', 'test')]
    test_results = {}
    removed_records = []
    for name in sorted({k[0] for k in index}):
        predictions, threshold, baseline, cache_hash = load_cache(name, 'test', index, test_items)
        changed = score(test_items, predictions, threshold, chosen['policy'])
        test_results[name] = dict(threshold=threshold, cache_sha256=cache_hash, **compare(baseline, changed))
        removed_records.extend(dict(run=name, **r) for r in changed[2])
    report = dict(status='complete_retrospective_development_ablation', created_utc=datetime.now(timezone.utc).isoformat(),
        fresh_blind_test=False, training_started=False, source_results_sha256=sha(result_path),
        validation_lock_sha256=sha(out / 'validation_lock.json'), validation=chosen, test=test_results,
        limitations=['Previously inspected evaluation regions; not a new blind test.',
            'Unmatched predictions are relative to frozen labels, not confirmed geological backgrounds.',
            'Recall counts are target appearances, not independent landslides.',
            'The rule cannot recover missing small predictions or repair box coordinates.',
            'Experimental filter is not enabled in the application.'])
    atomic(out / 'comparison.json', report)
    atomic(out / 'removed_predictions.json', removed_records)
    print('selected', chosen['policy'], 'validation_unmatched_reduction', chosen['unmatched_reduction'], flush=True)
    for name, r in test_results.items():
        print(name, 'removed', r['removed_predictions'], 'lost', r['lost_target_appearances'],
              'precision', round(r['before']['precision'], 4), '->', round(r['after']['precision'], 4), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    run(parser.parse_args().out)
