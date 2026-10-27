"""Recompute frozen detector and joint metrics without retraining, downloads or test-set access."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.venv/Lib/site-packages'), str(ROOT / 'data/scripts')]
import numpy as np

REPEAT = ROOT / 'runs/experiments/detector_optimizer_repeat_20261024_v1'
JOINT = ROOT / 'runs/experiments/optimizer_joint_bridge_20261025_v1'
OLD_JOINT = ROOT / 'runs/experiments/latest_detector_classifier_bridge_20261024_v1'
SEEDS = [20261015, 20261016, 20261017]
NAMES = ['SL', 'SP', 'FL', 'CL']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda: stream.read(1 << 20), b''):
            digest.update(value)
    return digest.hexdigest()


def output(path, value):
    assert not path.exists(), 'Preserve earlier verification'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def iou(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    w = max(np.float32(0), min(a[2], b[2])-max(a[0], b[0]))
    h = max(np.float32(0), min(a[3], b[3])-max(a[1], b[1]))
    intersection = w*h
    denominator = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return float(intersection/denominator) if denominator > 0 else 0.


def match(labels, rows):
    order = np.argsort(-np.asarray([r['score'] for r in rows], np.float32))
    used = set(); pairs = []
    for label in labels:
        for j in order:
            j = int(j)
            if j not in used and iou(label['box'], rows[j]['box']) >= .5:
                used.add(j); pairs.append((label, rows[j])); break
    return pairs


def detection():
    assert read(REPEAT / 'queue.json')['status'] == 'completed'
    report = read(REPEAT / 'comparison.json')
    assert report['protocol_sha256'] == sha(REPEAT / 'protocol.json')
    plan = read(ROOT / 'runs/research/latest_detector_roi_preparation_20261024_v1/input_plan.json')
    items = plan['items']; checks = []; pins = {}
    from evaluate_reviewed_eval_labels import metrics as official_metrics
    official_items = [dict(stem=i['stem'], width=i['width'], height=i['height'],
                           human=dict(boxes=[l['box'] for l in i['labels']])) for i in items]
    for seed, source in [('0', ROOT / 'runs/experiments/detector_optimizer_20261022_v1'), ('1', REPEAT)]:
        for arm in ['adamw_lr3e4', 'sgd_lr1e2']:
            path = source / 'evaluation' / arm / 'predictions.json'
            pred = read(path); pins[str(path)] = sha(path)
            metric_path = source / 'evaluation' / arm / 'metrics.json'
            archived = read(metric_path); pins[str(metric_path)] = sha(metric_path)
            weight = source / 'training' / arm / 'train/weights/best.pt'
            assert sha(weight) == archived['checkpoint_sha256']
            official = official_metrics(official_items, pred, 'human', archived['budgets']['1071']['metrics']['threshold'])
            for budget in ['300', '600', '1071']:
                row = report['results_by_seed'][seed][arm]['budgets'][budget]
                expected = row['metrics']; threshold = expected['threshold']
                ids = set(); tp = candidates = ngt = 0
                for item in items:
                    boxes, scores, classes = [np.asarray(v, np.float32) for v in pred[item['stem']]]
                    assert np.all(classes == 0)
                    rows = [dict(box=boxes[j], score=float(scores[j])) for j in np.flatnonzero(scores >= threshold)]
                    pairs = match(item['labels'], rows)
                    tp += len(pairs); candidates += len(rows); ngt += len(item['labels'])
                    ids.update(int(l['source_id']) for l, _ in pairs)
                assert ngt == 560 and candidates == int(budget)
                assert (tp, candidates-tp, ngt-tp) == (expected['tp'], expected['fp'], expected['fn'])
                assert abs(expected['precision']-tp/candidates) < 1e-12 and abs(expected['recall']-tp/ngt) < 1e-12
                assert sorted(ids) == row['unique_IDs'] and len(ids) == row['unique_recalled']
                for key in ['cached_map50', 'cached_map50_95']:
                    assert abs(expected[key]-official[key]) < 1e-12
                checks.append(dict(seed=int(seed), arm=arm, budget=int(budget), unique_detected=len(ids),
                                   scalar_matching_verified=True, AP_replayed_from_full_saved_cache=True))
    result = dict(status='passed_two_seed_detector_cache_verification', cells=checks,
                  prediction_and_metric_files_sha256=pins, comparison_sha256=sha(REPEAT / 'comparison.json'),
                  verifier_sha256=sha(__file__), verified_utc=datetime.now(timezone.utc).isoformat(),
                  AP_method='Replay pinned official matching/AP implementation; scalar budget matching independently reconstructed',
                  final_test_read=False, training=False, downloads=False)
    output(REPEAT / 'independent_completion_verification.json', result)
    print('VERIFIED_DETECTOR_PAIR', len(checks), 'budget cells; four full-cache AP replays', flush=True)


def joint():
    assert read(JOINT / 'queue.json')['status'] == 'completed'
    spec = read(JOINT / 'protocol.json'); plan = read(JOINT / 'input_plan.json'); report = read(JOINT / 'comparison.json')
    assert sha(JOINT / 'protocol.json') == report['protocol_sha256'] == (JOINT / 'protocol.sha256').read_text().strip()
    assert sha(JOINT / 'input_plan.json') == spec['plan_sha256']
    for path, digest in spec['files_sha256'].items():
        assert sha(path) == digest, path
    truth = {str(r['source_id']):r['class_id'] for r in plan['oracle']}
    items = {r['stem']:r for r in plan['items']}; matches = {}; cells = 0
    for name, model in plan['models'].items():
        matches[name] = {}
        for budget in ['300', '600', '1071']:
            pairs = []; nc = 0
            for stem, entry in model['items'].items():
                rows = [r for r in entry['predictions'] if r['score'] >= spec['thresholds'][name][budget]]
                nc += len(rows)
                pairs.extend(match(items[stem]['labels'], rows))
            assert nc == int(budget) and len(pairs) == spec['expected_detection_metrics'][name][budget]['tp']
            assert all(row['case_index'] is not None for _, row in pairs)
            matches[name][budget] = pairs
    for seed in SEEDS:
        path = JOINT / f'seed{seed}/dino_predictions.npz'
        with np.load(path) as matrix, np.load(OLD_JOINT / f'seed{seed}/dino_predictions.npz') as old:
            logits = matrix['logits']; values = matrix['predictions']
            assert logits.shape == (len(plan['cases']), 4) and np.isfinite(logits).all()
            assert np.array_equal(logits.argmax(1), values)
            assert np.array_equal(logits[:10893], old['logits'])
        seed_result = report['per_seed'][str(seed)]
        for name in plan['models']:
            for budget in ['300', '600', '1071']:
                pairs = matches[name][budget]
                located = {str(l['source_id']) for l, _ in pairs}
                correct = [str(l['source_id']) for l, row in pairs if int(values[row['case_index']]) == truth[str(l['source_id'])]]
                ids = set(correct); expected = seed_result[name][budget]; nc = int(budget)
                assert expected['correct_source_ids'] == sorted(ids, key=int)
                assert expected['jointly_correct_IDs'] == len(ids) and expected['located_IDs'] == len(located)
                assert expected['joint_TP_appearances'] == len(correct) and expected['located_appearances'] == len(pairs)
                assert expected['joint_unique_recall'] == len(ids)/183
                assert expected['reference_joint_precision'] == len(correct)/nc
                assert expected['joint_occurrence_recall'] == len(correct)/560
                for i, label in enumerate(NAMES):
                    row = expected['per_class'][label]
                    assert row['joint_correct_IDs'] == sum(truth[s] == i for s in ids)
                    assert row['located_IDs'] == sum(truth[s] == i for s in located)
                    assert row['joint_TP_appearances'] == sum(truth[s] == i for s in correct)
                cells += 1
    for path, digest in read(JOINT / 'crop_files_sha256.json').items():
        assert sha(path) == digest
    result = dict(status='passed_fixed_classifier_optimizer_pair_joint_verification', verified_score_cells=cells,
                  verified_matched_budget_pairs=12, comparison_sha256=sha(JOINT / 'comparison.json'),
                  protocol_sha256=sha(JOINT / 'protocol.json'), verifier_sha256=sha(__file__),
                  all_candidates_in_denominators=True, matched_only_inference_does_not_change_metrics=True,
                  final_test_read=False, training=False, downloads=False)
    output(JOINT / 'independent_completion_verification.json', result)
    lines = ['# 优化器配对检测与分类联评', '',
             '同一173张开发图、183个滑坡编号；固定DINO三个分类种子，表中为定位且类型正确的不同编号数均值。', '',
             '|检测种子|优化器|300候选|600候选|1071候选|1071定位编号|', '|---|---|---:|---:|---:|---:|']
    means = {}
    for detector_seed in [0, 1]:
        for arm in ['adamw_lr3e4', 'sgd_lr1e2']:
            name = ('optimizer_' if detector_seed == 0 else 'optimizer_seed1_')+arm
            means[name] = {b:float(np.mean([report['per_seed'][str(s)][name][b]['jointly_correct_IDs'] for s in SEEDS]))
                           for b in ['300', '600', '1071']}
            located = report['per_seed'][str(SEEDS[0])][name]['1071']['located_IDs']
            lines.append(f'|{detector_seed}|{arm}|{means[name]["300"]:.2f}|{means[name]["600"]:.2f}|{means[name]["1071"]:.2f}|{located}|')
    lines += ['', '不是盲测验收。分类种子不是独立地理样本；FL仅2个、CL15个开发编号。',
              '新检测池仅对固定匹配候选计算类型，全部候选仍进入联合精确率分母；未输出未匹配框的类型，不报告其逐类型精确率。',
              '种子0 AdamW与种子1 SGD含EMA检查点恢复；不得按最好种子挑选优化器。没有重新训练分类器、修改阈值、读测试集或新增下载。', '']
    (JOINT / 'verified_results.md').write_text('\n'.join(lines), encoding='utf-8')
    print('VERIFIED_JOINT_PAIR', cells, 'score cells', json.dumps(means), flush=True)


if __name__ == '__main__':
    {'detection':detection, 'joint':joint}[sys.argv[1]]()
