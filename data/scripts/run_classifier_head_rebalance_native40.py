"""Decoupled head-weight comparison on fixed, phase-specific native40 LoRA.

The inner-fit encoder is used for inner selection; the outer-refit encoder is
used only for fixed-epoch outer refitting. Never use an outer-trained encoder
to choose head epochs on the inner stopping set. Expert labels stay unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
import gc
import json
import os
from pathlib import Path
import sys
import time
import uuid

import run_classifier_lora_native40 as G

M, C, np, torch = G.M, G.C, G.np, G.torch
ROOT = G.ROOT
OUT = ROOT / 'runs/experiments/classifier_head_rebalance_native40_20261015_v1'
SOURCE = G.OUT
SOURCE_CTOR = M.Classifier
ARMS = {'head_none': 'none', 'head_sqrt': 'sqrt', 'head_balanced': 'balanced'}
SEEDS = G.SEEDS


def source_folder(seed, fold):
    base = G.REUSE / 'qv_last2_LoRA' if seed == SEEDS[0] else SOURCE / 'qv_last2_LoRA' / f'seed{seed}'
    return base / f'fold{fold}'


def prepare():
    source = G.check()
    assert C.read_json(SOURCE / 'queue.json')['status'] == 'completed_native40_multiseed_nested_comparison'
    assert C.read_json(SOURCE / 'completion_verification.json')['status'] == 'passed_independent_saved_prediction_recomputation'
    first = C.read_json(G.REUSE / 'protocol.json')
    assert first['design']['outer_folds'] == source['design']['outer_folds']
    assert first['design']['ordered_source_ids'] == [r['source_id'] for r in source['design']['rows']]
    C.new_directory(OUT)
    files = list(map(Path, source['files_sha256']))
    files += [SOURCE / name for name in ['protocol.json', 'comparison.json', 'completion_verification.json']]
    files += [G.REUSE / 'protocol.json', G.REUSE / 'comparison.json']
    sources = {}
    for seed in SEEDS:
        sources[str(seed)] = {}
        for split in source['design']['outer_folds']:
            fold = split['outer_fold']
            base = source_folder(seed, fold)
            selected, fitted = [C.read_json(base / phase / 'completed.json') for phase in ['select', 'refit']]
            assert selected['best_epoch'] == fitted['selected_epochs'] == fitted['completed_epochs']
            assert selected['fit_unique_IDs'] == len(split['inner_fit_indices'])
            assert fitted['fit_unique_IDs'] == len(split['outer_train_indices'])
            sources[str(seed)][str(fold)] = {phase: str(base / phase / name) for phase, name in [('select', 'best.pt'), ('refit', 'last.pt')]}
            for phase, name in [('select', 'best.pt'), ('refit', 'last.pt')]:
                files += [base / phase / n for n in [name, 'completed.json', 'metrics.json']]
    spec = deepcopy(source)
    spec.update(created_utc=M.utc(), arms=list(ARMS), balance_modes=ARMS, seeds=SEEDS, reused_seed=None,
                source_experiment=str(SOURCE), source_encoders=sources, design=deepcopy(source['design']),
                primary_contrast='head_balanced minus head_sqrt; head_none is an unweighted diagnostic control',
                representation='Fixed final2-q/v rank4 LoRA from the corresponding seed/fold and select/refit phase; no encoder updates',
                expert_type_policy='Retain original expert catalog labels; no nonexpert subtype review prerequisite',
                normalization='StandardScaler fit on adapted ROI features of the actual fitting IDs; common to all three heads',
                loss='none / sqrt inverse frequency / inverse frequency mean-one class weights from actual fitting IDs; same effective-batch weighted CE',
                head_initialization='Fresh 768x4 linear head for each arm; identical tensors and minibatch order within each seed/fold/phase',
                cache_policy='Inner cache covers outer-training IDs only; outer cache covers outer train/held; source checkpoint and ID alignment pinned',
                inner_encoder_policy='Best source select checkpoint trained only on inner_fit; inner_stop previously used only for early stopping',
                refit_encoder_policy='Source refit encoder fixed before new heads; outer_train only; head epochs selected using separate inner encoder',
                previous_joint_LoRA_reference='Historical reference only; new heads are decoupled and standardized on adapted features, so not a one-factor comparison with joint LoRA',
                test_images_read=False, external_validation_images_read=False, labels_modified=False,
                evaluation_scope='Repeated internal40-group development CV; not fresh blind evaluation',
                automatic_thread_continuation=False, automatic_following_experiments=False)
    spec.pop('newly_trained_seeds', None)
    spec.pop('reuse_source', None)
    spec.pop('reuse_seed', None)
    spec['files_sha256'] = {str(p): C.sha256(p) for p in sorted(set(files))}
    code = list(map(Path, source['code_hashes'])) + [Path(__file__), ROOT / 'data/scripts/start_classifier_head_rebalance_native40.ps1']
    spec['code_hashes'] = {str(p): C.sha256(p) for p in sorted(set(code))}
    G.write(OUT / 'protocol.json', spec)
    (OUT / 'protocol.sha256').write_text(C.sha256(OUT / 'protocol.json') + '\n', encoding='utf-8')
    G.write(OUT / 'queue.json', dict(status='prepared_not_started', created_utc=M.utc(), seeds=SEEDS,
                                  arms=list(ARMS), spatial_groups=40, labels_modified=False))
    print('PREPARED_FIXED_LORA_HEAD_REBALANCE_3ARMS_3SEEDS', flush=True)


def check():
    spec = C.read_json(OUT / 'protocol.json')
    assert C.sha256(OUT / 'protocol.json') == (OUT / 'protocol.sha256').read_text().strip()
    for field in ['files_sha256', 'code_hashes']:
        for p, digest in spec[field].items():
            assert C.sha256(p) == digest, p
    for name, digest in spec['checkpoint_files_sha256'].items():
        assert C.sha256(Path(spec['checkpoint']) / name) == digest
    assert spec['versions'] == M.P.R.package_versions()
    assert G.spatial_guard(spec['design']['rows'], spec['design']['outer_folds'], C.read_json(G.MANIFEST))['passed']
    assert spec['balance_modes'] == ARMS
    return spec


def cached_features(spec, data, split, seed, phase, queue=None):
    fold = split['outer_fold']
    source = Path(spec['source_encoders'][str(seed)][str(fold)][phase])
    fit = split['inner_fit_indices'] if phase == 'select' else split['outer_train_indices']
    held = split['inner_stop_indices'] if phase == 'select' else split['outer_held_indices']
    indices = np.array(sorted(fit + held), dtype=np.int64)
    if phase == 'select':
        assert set(indices) == set(split['outer_train_indices'])
        assert not set(indices) & set(split['outer_held_indices'])
    folder = OUT / 'features' / f'seed{seed}' / f'fold{fold}'
    folder.mkdir(parents=True, exist_ok=True)
    payload, receipt_path = folder / f'{phase}.npz', folder / f'{phase}.json'
    if receipt_path.exists():
        receipt = C.read_json(receipt_path)
        assert receipt['protocol_sha256'] == C.sha256(OUT / 'protocol.json')
        assert receipt['source_checkpoint_sha256'] == C.sha256(source)
        assert receipt['cache_sha256'] == C.sha256(payload)
        with np.load(payload, allow_pickle=False) as saved:
            assert np.array_equal(saved['indices'], indices)
            features = saved['features'].copy()
        assert receipt['ordered_source_ids'] == [data['rows'][i]['source_id'] for i in indices]
    else:
        assert C.sha256(source) == spec['files_sha256'][str(source)]
        source_spec = C.read_json((G.REUSE if seed == SEEDS[0] else SOURCE) / 'protocol.json')
        saved = torch.load(source, map_location='cpu', weights_only=False)
        assert saved['protocol_sha256'] == C.sha256((G.REUSE if seed == SEEDS[0] else SOURCE) / 'protocol.json')
        assert (saved['arm'], saved['fold'], saved['phase']) == ('qv_last2_LoRA', fold, phase)
        assert saved.get('run_seed', source_spec.get('seed')) == seed
        assert saved['completed_epoch'] == (saved['best_epoch'] if phase == 'select' else C.read_json(source.parent / 'completed.json')['selected_epochs'])
        model = SOURCE_CTOR(source_spec, 'qv_last2_LoRA', fit, data, fold, phase, seed)
        M.load_parameters(model, saved['parameters'])
        adapter_hash = M.tensor_hash({k: v for k, v in saved['parameters'].items() if not k.startswith('head.')})
        assert any(v.abs().sum() > 0 for k, v in saved['parameters'].items() if k.endswith('.B'))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        assert not model.encoder.training and not any(p.requires_grad for p in model.parameters())
        values = []
        with torch.no_grad():
            for start in range(0, len(indices), 8):
                ix = indices[start:start + 8]
                values.append(model.features(data, ix).cpu().numpy())
                if queue is not None and start % 128 == 0:
                    queue.update(phase='extract_fixed_features_' + phase, run_seed=seed, outer_fold=fold,
                                 feature_images_completed=start + len(ix), feature_images_total=len(indices))
                    G.write(OUT / 'queue.json', queue)
        features = np.concatenate(values).astype(np.float32)
        assert features.shape == (len(indices), 768) and np.isfinite(features).all()
        temp = payload.with_name(payload.name + '.' + uuid.uuid4().hex + '.tmp')
        with temp.open('xb') as stream:
            np.savez_compressed(stream, indices=indices, features=features)
        os.replace(temp, payload)
        receipt = dict(seed=seed, fold=fold, phase=phase, images=len(indices),
                       ordered_source_ids=[data['rows'][i]['source_id'] for i in indices],
                       fit_indices=fit, held_indices=held, cache_sha256=C.sha256(payload),
                       source_checkpoint=str(source), source_checkpoint_sha256=C.sha256(source),
                       adapter_parameter_hash=adapter_hash, all_encoder_parameters_frozen=True,
                       outer_held_features_absent=phase == 'select', protocol_sha256=C.sha256(OUT / 'protocol.json'))
        G.write(receipt_path, receipt)
        del model, saved, values
        gc.collect()
        torch.cuda.empty_cache()
    assert features.shape == (len(indices), 768) and np.isfinite(features).all()
    # NaNs make accidental use of an unavailable outer-held vector fail loudly.
    dense = torch.full((len(data['rows']), 768), float('nan'), dtype=torch.float32)
    dense[indices] = torch.from_numpy(features)
    return dense


class CachedHead(M.nn.Module):
    def __init__(self, spec, arm, fit, data, fold, phase, run_seed):
        super().__init__()
        assert arm in ARMS and data['balance_mode'] == ARMS[arm]
        self.encoder = None
        M.seed(run_seed + fold * 101 + (10000 if phase == 'refit' else 0) + 11)
        self.head = M.nn.Linear(768, 4)
        assert torch.isfinite(data['X'][fit]).all()
        scaler = M.StandardScaler().fit(data['X'][fit].numpy())
        self.register_buffer('mean', torch.from_numpy(scaler.mean_.astype(np.float32)))
        self.register_buffer('scale', torch.from_numpy(scaler.scale_.astype(np.float32)))
        self.to('cuda')
        self.eval()

    def forward(self, data, indices):
        values = data['X'][indices].to('cuda')
        assert torch.isfinite(values).all(), 'Attempted to use features not allowed in this phase'
        return self.head((values - self.mean) / self.scale)


def weights(data, fit):
    values = M.classifier_class_weights(data['y'][fit].numpy(), data['balance_mode'])
    if values is None:
        return torch.ones(4, device='cuda', dtype=torch.float32)
    assert set(values) == set(range(4))
    return torch.tensor([values[i] for i in range(4)], device='cuda', dtype=torch.float32)


def smoke():
    spec = check()
    M.device_setup()
    blocked = M.FENV.prevent_network()
    data = M.load_data(spec)
    split = spec['design']['outer_folds'][0]
    frozen = data['X']
    data['X'] = cached_features(spec, data, split, SEEDS[0], 'select')
    assert torch.isnan(data['X'][split['outer_held_indices']]).all()
    assert not torch.allclose(data['X'][split['inner_fit_indices']], frozen[split['inner_fit_indices']])
    fit = split['inner_fit_indices']
    ids = np.array(sum(([i for i in fit if data['y'][i] == c][:4] for c in range(4)), []), dtype=int)
    assert len(ids) == 16
    initial_hashes, arm_weights = {}, {}
    for arm, mode in ARMS.items():
        data['balance_mode'] = mode
        model = CachedHead(spec, arm, fit, data, 0, 'select', SEEDS[0])
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 3076
        initial_hashes[arm] = M.tensor_hash(M.parameter_state(model))
        arm_weights[arm] = weights(data, fit).cpu().tolist()
        opt = M.optimizer(model, spec)
        M.update_group(model, opt, data, ids, weights(data, fit))
        del model, opt
    assert len(set(initial_hashes.values())) == 1
    assert arm_weights['head_none'] == [1.] * 4
    assert arm_weights['head_balanced'][2] > arm_weights['head_sqrt'][2] > 1.
    data['balance_mode'] = 'sqrt'
    model = CachedHead(spec, 'head_sqrt', fit, data, 0, 'select', SEEDS[0])
    opt = M.optimizer(model, spec)
    w = weights(data, fit)
    M.update_group(model, opt, data, ids, w)
    temporary = OUT / 'smoke_checkpoint.pt'
    M.save_checkpoint(temporary, model, opt, [], 1, 1, 0., 'head_sqrt', 0, 'select', SEEDS[0])
    prediction = M.predict(model, data, ids)[1]
    loss = M.update_group(model, opt, data, ids, w)
    expected = M.tensor_hash(M.parameter_state(model))
    del model, opt
    saved = torch.load(temporary, map_location='cpu', weights_only=False)
    model = CachedHead(spec, 'head_sqrt', fit, data, 0, 'select', SEEDS[0])
    opt = M.optimizer(model, spec)
    M.load_parameters(model, saved['parameters'])
    opt.load_state_dict(saved['optimizer'])
    assert np.array_equal(prediction, M.predict(model, data, ids)[1])
    assert M.update_group(model, opt, data, ids, w) == loss
    assert expected == M.tensor_hash(M.parameter_state(model))
    G.write(OUT / 'readiness.json', dict(status='passed_real_GPU_smoke', created_utc=M.utc(), images=16, classes=4,
                                       trained_encoder_parameters=0, trained_head_parameters=3076,
                                       same_initial_head_across_arms=True, class_weights=arm_weights,
                                       inner_cache_excludes_outer_held=True, adapted_features_differ_from_original_frozen_cache=True,
                                       checkpoint_prediction_bitwise_equal=True, resumed_next_update_bitwise_equal=True,
                                       blocked_network_attempts=blocked, protocol_sha256=C.sha256(OUT / 'protocol.json')))
    print('PASSED_FIXED_LORA_HEAD_GPU_SMOKE', flush=True)


def summarize(spec, data):
    per_seed = {a: {str(s): M._arm_seed_metrics(data, a, s, spec['design']['outer_folds']) for s in SEEDS} for a in ARMS}
    aggregate = {}
    for arm in ARMS:
        values = list(per_seed[arm].values())
        aggregate[arm] = dict(macro_f1_mean=float(np.mean([v['macro_f1'] for v in values])),
                              macro_f1_std=float(np.std([v['macro_f1'] for v in values])),
                              accuracy_mean=float(np.mean([v['accuracy'] for v in values])),
                              per_class={n: {k: float(np.mean([v['per_class'][n][k] for v in values])) for k in ['precision', 'recall', 'f1-score', 'support']} for n in M.NAMES})
    paired = {}
    for arm in ['head_none', 'head_balanced']:
        delta = [per_seed[arm][str(s)]['macro_f1'] - per_seed['head_sqrt'][str(s)]['macro_f1'] for s in SEEDS]
        paired[arm + '_minus_sqrt'] = dict(per_seed=delta, mean=float(np.mean(delta)), std=float(np.std(delta)), wins=sum(x > 0 for x in delta))
    result = dict(status='completed_fixed_LoRA_head_rebalancing', created_utc=M.utc(), targets=1083, spatial_groups=40,
                  seeds=SEEDS, arms=ARMS, aggregate=aggregate, per_seed=per_seed, paired_differences=paired,
                  original_joint_LoRA_reference=C.read_json(SOURCE / 'comparison.json')['aggregate']['qv_last2_LoRA'],
                  original_reference_is_not_same_training_procedure=True, std_definition='population ddof0 across optimization seeds',
                  labels_modified=False, test_images_read=False, external_validation_images_read=False,
                  protocol_sha256=C.sha256(OUT / 'protocol.json'))
    G.write(OUT / 'comparison.json', result)
    lines = ['# 固定LoRA表示：分类头类别权重对照', '', '保留专家类型，1083个编号、40组、3折、3种子。表示按种子／折／阶段固定；三臂只改变分类头的类别权重。', '',
             '|分类头权重|macro-F1均值±标准差|正确率|SL召回|SP召回|FL召回|CL召回|', '|---|---:|---:|---:|---:|---:|---:|']
    for arm, mode in ARMS.items():
        g = aggregate[arm]
        lines.append(f'|{mode}|{g["macro_f1_mean"]:.2%}±{g["macro_f1_std"]:.2%}|{g["accuracy_mean"]:.2%}|'+
                     '|'.join(f'{g["per_class"][n]["recall"]:.1%}' for n in M.NAMES)+'|')
    lines += ['', '主对照为balanced−sqrt；none为无权重参照。旧联合LoRA成绩仅作历史参照，不能把重训分类头及标准化变化归因于类别权重。', '',
              '这是已多次用于开发的内部空间交叉验证。保留原专家类别；没有要求用户重新鉴定类型；未读取最终测试或173张外部开发验证图。']
    (OUT / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(resume=False):
    spec = check()
    ready = C.read_json(OUT / 'readiness.json')
    assert ready['status'] == 'passed_real_GPU_smoke' and ready['protocol_sha256'] == C.sha256(OUT / 'protocol.json')
    state = C.read_json(OUT / 'queue.json')
    assert state['status'] in (['running', 'failed'] if resume else ['prepared_not_started'])
    independence = M.FENV.process_independence()
    assert independence['scheduler_owned']
    M.P.R.ensure_gpu_idle()
    blocked = M.FENV.prevent_network()
    M.device_setup()
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    queue = dict(status='running', started_utc=state.get('started_utc', M.utc()), pid=os.getpid(),
                 seeds=SEEDS, arms=list(ARMS), spatial_groups=40, labels_modified=False,
                 process_independence=independence, offline=True, temporary_system_awake=True,
                 automatic_thread_continuation=False, manual_resume=resume)
    G.write(OUT / 'queue.json', queue)
    try:
        data = M.load_data(spec)
        original = data['X']
        for seed in SEEDS:
            for split in spec['design']['outer_folds']:
                # Every mode completes inner selection before any outer head is fitted.
                for phase in ['select', 'refit']:
                    data['X'] = original
                    features = cached_features(spec, data, split, seed, phase, queue)
                    for arm, mode in ARMS.items():
                        data['X'], data['balance_mode'] = features, mode
                        M.stage(spec, data, split, arm, phase, queue, seed)
        check()
        summarize(spec, data)
        queue.update(status='completed_fixed_LoRA_head_rebalancing', phase='complete')
    except BaseException as error:
        queue.update(status='failed', error=repr(error))
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        queue.update(finished_utc=M.utc(), temporary_system_awake=False, blocked_network_attempts=blocked)
        G.write(OUT / 'queue.json', queue)


M.OUT = OUT
M.Classifier = CachedHead
M.weights = weights
M.write = G.write

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'check', 'smoke', 'scheduled-run', 'scheduled-resume'])
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'check':
        check()
        print('PROTOCOL_SOURCE_ENCODERS_AND_NATIVE40_GUARD_PASSED')
    elif args.mode == 'smoke':
        smoke()
    else:
        suffix = '' if args.mode == 'scheduled-run' else '_resume_' + str(int(time.time()))
        sys.stdout = (OUT / f'training{suffix}.log').open('x', encoding='utf-8', buffering=1)
        sys.stderr = (OUT / f'training{suffix}.err.log').open('x', encoding='utf-8', buffering=1)
        run(resume=args.mode == 'scheduled-resume')
