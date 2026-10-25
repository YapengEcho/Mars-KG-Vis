"""Compare best control/P2 checkpoints and archived manual predictions on validation only."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from review_eval_labels import ROOT, read, atomic, sha
from evaluate_nested_box_filter import score
from compare_native_resolution_budget import equal_budget_threshold
from evaluate_reviewed_eval_labels import metrics


OUT = ROOT / 'runs/ablations/p2_validation_comparison_20261014_v1'
SOURCE = ROOT / 'runs/experiments/p2_detector_probe_20261013_v2'
EXTENSION = ROOT / 'runs/experiments/p2_until_earlystop_20261014_v1'
FROZEN = ROOT / 'data/processed/manual_eval_frozen_20261011_v1/frozen.json'
ARCHIVED = ROOT / 'runs/ablations/rebuilt_six_reevaluation_20261011_v1'
BUDGETS = (300, 600, 1071)
SETTINGS = dict(imgsz=1024, conf=0.001, iou=0.7, max_det=300,
                agnostic_nms=True, device=0, batch=1, verbose=False,
                augment=False, rect=False, half=False)


def validate_validation_inputs():
    receipt = read(SOURCE / 'source_receipt.json')
    if sha(FROZEN) != receipt['frozen_eval_sha256']:
        raise ValueError('Frozen evaluation manifest changed')
    # Read manifest metadata, but never open test images, labels or predictions.
    items = [r for r in read(FROZEN)['items'] if (r['arm'], r['split']) == ('offset', 'val')]
    if len(items) != 173 or len({r['stem'] for r in items}) != 173:
        raise ValueError('Unexpected validation cohort')
    if sum(len(r['human']['boxes']) for r in items) != 560:
        raise ValueError('Unexpected validation labels')
    copied = {Path(r['source']).resolve(): r['sha256'] for r in receipt['copied_inputs']}
    for item in items:
        for kind in ('image', 'label'):
            path = Path(item[kind]).resolve()
            if sha(path) != item[kind + '_sha256'] or copied.get(path) != item[kind + '_sha256']:
                raise ValueError('Validation pixels/labels differ from training validation receipt')
    return items


def infer(weights, items, folder):
    import torch
    from ultralytics import YOLO
    start = time.perf_counter()
    model = YOLO(str(weights))
    if len(model.names) != 1:
        raise ValueError('Expected single-class detector')
    source = folder / 'inference_source.txt'
    source.write_text('\n'.join(Path(r['image']).as_posix() for r in items) + '\n', encoding='utf-8')
    model.predict(source=items[0]['image'], **SETTINGS)
    torch.cuda.synchronize()
    load_warmup_seconds = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    predictions, speeds = {}, []
    for result in model.predict(source=str(source), stream=True, **SETTINGS):
        box = result.boxes
        predictions[Path(result.path).stem] = [box.xyxy.cpu().numpy(), box.conf.cpu().numpy(), box.cls.cpu().numpy()]
        speeds.append(dict(result.speed))
        if len(predictions) % 25 == 0:
            print(folder.name, 'images', len(predictions), '/', len(items), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if set(predictions) != {r['stem'] for r in items}:
        raise ValueError('Inference cohort mismatch')
    timing = dict(images=len(items), load_and_warmup_seconds=load_warmup_seconds,
                  warmed_stream_seconds=elapsed, wall_ms_per_image=1000 * elapsed / len(items),
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  mean_sdk_speed_ms={k: float(np.mean([s[k] for s in speeds])) for k in speeds[0]})
    del model
    torch.cuda.empty_cache()
    return predictions, timing


def main():
    if OUT.exists():
        raise FileExistsError('Preserve previous comparison; use a new output version')
    if read(EXTENSION / 'queue.json')['status'] != 'stopped_by_user':
        raise ValueError('Stop training before using GPU for comparison')
    items = validate_validation_inputs()
    weights = dict(control=SOURCE / 'pilot/r2_b8/control_seed0/weights/best.pt',
                   p2=EXTENSION / 'pilot/r2_b8/p2_seed0/weights/best.pt')
    archived_results = read(ARCHIVED / 'results.json')
    historical = next(r for r in archived_results['results']
                      if (r['run'], r['arm'], r['split']) == ('manual_seed0', 'offset', 'val'))
    old_path = ARCHIVED / 'manual_seed0/offset/val/predictions.json'
    if historical['frozen_sha256'] != sha(FROZEN) or sha(old_path) != historical['predictions_sha256']:
        raise ValueError('Historical cache receipt mismatch')
    import torch, ultralytics
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    checkpoints = {}
    for name, path in weights.items():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        audit = read(path.parents[1] / 'training_audit.json')
        # Completed SDK runs strip optimizer and replace epoch with -1.
        # Bind their best epoch to the saved audit and its checkpoint digest.
        if name == 'control':
            if sha(path) != audit['checkpoint_best_sha256']:
                raise ValueError('Completed control best checkpoint digest changed')
            epoch = int(audit['best_epoch'])
        else:
            epoch = int(saved['epoch']) + 1
            if epoch != int(audit['best_epoch']):
                raise ValueError('P2 best epoch differs from stopped audit')
        if epoch != (36 if name == 'control' else 49):
            raise ValueError('Selected best checkpoint epoch changed')
        checkpoints[name] = dict(path=str(path), sha256=sha(path), best_epoch=epoch)
        del saved
    OUT.mkdir(parents=True)
    protocol = dict(status='validation_comparison_locked_before_inference', split='offset val',
                    test_images_read=0, test_predictions_read=0, images=173, gt_appearances=560,
                    unique_source_objects=len({s for r in items for s in r['source_ids']}),
                    budgets=list(BUDGETS), primary_budget=1071,
                    budget_origin='1071 is the archived manual_seed0 validation candidate count; 300/600 are lower burden diagnostics',
                    cutoff_rule='Each model cutoff retains at most budget predictions; tied boundary scores excluded together',
                    settings=SETTINGS, frozen_sha256=sha(FROZEN), checkpoints=checkpoints,
                    historical_cache_sha256=sha(old_path), code_sha256=sha(Path(__file__)),
                    limitations=['Validation development evidence; not independent final test.',
                                 'P2 extension is exploratory beyond original 60-epoch protocol.',
                                 'Historical model differs in training recipe/initialization; not a pure architecture control.',
                                 'Validation has no pure background images; unmatched predictions may include annotation omissions.',
                                 'Target appearances are correlated repeated views, not independent slides.'])
    atomic(OUT / 'protocol.json', protocol)
    atomic(OUT / 'state.json', dict(status='running', created_utc=datetime.now(timezone.utc).isoformat()))
    results = {}
    for name in ('control', 'p2', 'historical_manual_seed0'):
        folder = OUT / name
        folder.mkdir()
        if name in weights:
            predictions, timing = infer(weights[name], items, folder)
            atomic(folder / 'predictions.json', {k: [a.tolist() for a in v] for k, v in predictions.items()})
        else:
            predictions = read(old_path)
            timing = dict(status='archived_cache_no_current_timing_measurement')
        if set(predictions) != {r['stem'] for r in items}:
            raise ValueError('Prediction cohort differs')
        comparisons = {}
        for budget in BUDGETS:
            threshold = equal_budget_threshold(predictions, budget)
            summary, _, _ = score(items, predictions, threshold)
            if summary['candidates'] > budget:
                raise AssertionError('Candidate budget exceeded')
            scored = metrics(items, predictions, 'human', threshold)
            if any(summary[k] != scored[k] for k in ('gt', 'tp', 'fp', 'candidates')):
                raise AssertionError('Scoring implementations disagree')
            tiles = {tile: score([r for r in items if r['tile'] == tile], predictions, threshold)[0]
                     for tile in sorted({r['tile'] for r in items})}
            comparisons[str(budget)] = dict(threshold=threshold, metrics=scored, by_tile=tiles)
        results[name] = dict(timing=timing, max_det_cap_images=sum(len(v[0]) >= 300 for v in predictions.values()),
                             budgets=comparisons)
        atomic(folder / 'metrics.json', results[name])
        print('SCORED', name, comparisons['1071']['metrics'], flush=True)
    report = dict(status='complete_validation_only', test_evaluated=False, protocol_sha256=sha(OUT / 'protocol.json'),
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  environment=dict(torch=torch.__version__, ultralytics=ultralytics.__version__,
                                   gpu=torch.cuda.get_device_name(0)), results=results)
    atomic(OUT / 'comparison.json', report)
    lines = ['# 普通YOLO与P2统一验证比较', '', '训练已按用户要求停止；普通版取36轮最佳，P2取49轮最佳。', '',
             '173张相同偏移验证图、560次框出现、183个独立目标；本轮没有打开测试图。', '',
             '| 模型 | 候选框 | 找到真值 | 精确率 | 召回率 | 未匹配框 | mAP50 | mAP50–95 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, result in results.items():
        m = result['budgets']['1071']['metrics']
        lines.append(f"| {name} | {m['candidates']} | {m['tp']} | {m['precision']:.1%} | {m['recall']:.1%} | {m['fp']} | {m['cached_map50']:.2%} | {m['cached_map50_95']:.2%} |")
    lines += ['', '主表将候选总数限制为1071；300和600候选的对照、尺寸/位置/图幅分组见comparison.json。', '',
              '缓存AP使用完整低阈值候选池，和训练日志的验证器AP口径可能不同；此表内可以比较。', '',
              '本轮为开发验证，不是最终测试。P2续训另记为探索；历史模型训练配方不同。',
              '没有纯背景图，未匹配框还可能含漏标；不据此自动把候选当负样本。',
              '计时为预热后batch1完整预测流（含读取及后处理），历史缓存没有同场实测速度。']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    atomic(OUT / 'state.json', dict(status='complete', finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
