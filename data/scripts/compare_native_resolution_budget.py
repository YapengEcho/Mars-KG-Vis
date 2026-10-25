"""Validation-only equal-prediction-budget comparison for the 1536 inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evaluate_nested_box_filter import FROZEN, CACHE, load_cache, score
from reevaluate_rebuilt_six import validate_frozen
from review_eval_labels import ROOT, atomic, read, sha

SOURCE = ROOT / 'runs/ablations/native_resolution_20261013_v1'


def equal_budget_threshold(predictions, budget):
    scores = np.concatenate([np.asarray(values[1], np.float32) for values in predictions.values()])
    scores = np.sort(scores)[::-1]
    if budget <= 0:
        return float(np.nextafter(scores[0], np.float32(np.inf))) if len(scores) else 1.0
    if len(scores) <= budget:
        return 0.001
    # Threshold above the first excluded score. Ties can make the retained
    # count slightly less than the target budget; never exceed it.
    if scores[budget - 1] == scores[budget]:
        return float(np.nextafter(scores[budget], np.float32(np.inf)))
    return float((float(scores[budget - 1]) + float(scores[budget])) / 2)


def run(source):
    target = source / 'fixed_budget_validation.json'
    if target.exists():
        raise FileExistsError('Keep the existing fixed-budget comparison')
    raw = read(source / 'comparison.json')
    if raw['evaluation_split'] != 'val' or raw['test_evaluated']:
        raise ValueError('Only validation scores can set a new threshold')
    frozen = validate_frozen(FROZEN)
    items = [r for r in frozen['items'] if (r['arm'], r['split']) == ('offset', 'val')]
    results = read(CACHE / 'results.json')
    index = {(r['run'], r['split']): r for r in results['results'] if r['arm'] == 'offset'}
    comparisons = {}
    for name in ('manual_seed0', 'manual_seed1'):
        _, old_threshold, baseline, _ = load_cache(name, 'val', index, items)
        budget = baseline[0]['candidates']
        comparisons[name] = dict(baseline_threshold=old_threshold, baseline=baseline[0], budget=budget, arms={})
        for arm in ('enlarged1024_at1536', 'native1536'):
            path = source / name / arm / 'predictions.json'
            timing = read(source / name / arm / 'timing.json')
            if sha(path) != timing['predictions_sha256']:
                raise ValueError('Resolution prediction cache changed')
            predictions = read(path)
            if set(predictions) != {item['stem'] for item in items}:
                raise ValueError('Resolution prediction cohort differs')
            threshold = equal_budget_threshold(predictions, budget)
            summary = score(items, predictions, threshold)[0]
            if summary['candidates'] > budget:
                raise AssertionError('Candidate budget exceeded')
            comparisons[name]['arms'][arm] = dict(validation_selected_threshold=threshold,
                metrics=summary, prediction_cache_sha256=sha(path),
                target_appearance_gain=summary['tp'] - baseline[0]['tp'])
    report = dict(status='validation_only_equal_candidate_budget', created_utc=datetime.now(timezone.utc).isoformat(),
        split='val', test_evaluated=False, source_comparison_sha256=sha(source / 'comparison.json'),
        selection='Score cutoff selected solely to retain at most as many validation predictions as archived 1024 baseline',
        comparisons=comparisons,
        limitations=['This is validation development evidence, not a fresh independent test.',
            'Old model and images are unchanged; native1536 changes the exported pixel detail, while both arms use 1536 inference input.',
            'Any proposed operational threshold must be locked here before one test run.'])
    atomic(target, report)
    for name, value in comparisons.items():
        print(name, 'baseline', value['baseline']['tp'], '/', value['budget'])
        for arm, a in value['arms'].items():
            print(' ', arm, 'TP', a['metrics']['tp'], 'predictions', a['metrics']['candidates'],
                  'small TP', a['metrics']['by_size'].get('small_64_256', {}).get('tp'))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    run(parser.parse_args().source)
