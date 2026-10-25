"""Native224 / native40 nested comparison; reuse audited seed, fit two new seeds.

Only orchestration, provenance guards and status I/O change. Numerical training
functions are imported read-only from the archived multiseed implementation.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import ctypes
import gc
import json
import os
from pathlib import Path
import sys
import time
import uuid

import run_classifier_lora_multiseed as M
import run_classifier_lora as ORIGINAL

C, np, torch = M.C, M.np, M.torch
ROOT = M.ROOT
OUT = ROOT / 'runs/experiments/classifier_lora_native40_multiseed_20261015_v1'
AUDIT = ROOT / 'runs/research/classifier_experiment_verification_20261015_v1'
NEXT = AUDIT / 'native40_next_plan.json'
PLAN = ROOT / 'runs/experiments/context_margin_shape_20261015_v2/context_full/plan.json'
MANIFEST = ROOT / 'data/processed/manual_stage2_native_20261015_v1/manifest.json'
REUSE = ROOT / 'runs/experiments/classifier_lora_20261015_v1'
SEEDS = [20261015, 20261016, 20261017]
NEW_SEEDS = SEEDS[1:]
ARMS = ['head_only', 'qv_last2_LoRA']


def write(path, value):
    """Atomic snapshot with bounded retries for transient Windows reader locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(12):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(.1 + .05 * attempt)


def spatial_guard(rows, splits, manifest):
    native = {r['source_id']: r for r in manifest['samples']}
    boxes = np.array([native[r['source_id']]['native_window_world_m'] for r in rows])
    donors, declared_donors = [], []
    for row in rows:
        n = native[row['source_id']]
        assert row['crop'] == n['crop'] and row['crop_sha256'] == n['crop_sha256']
        assert n['complete_native_coverage'] and n['full_human_box_preserved']
        assert sum(n['actual_donor_pixel_counts'].values()) == int(np.prod(n['size_px']))
        donors.append(set(n['actual_donor_tiles']))
        declared_donors.append({s['tile'] for s in n['sources']})
    width = np.minimum(boxes[:, None, 2], boxes[None, :, 2]) - np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    height = np.minimum(boxes[:, None, 3], boxes[None, :, 3]) - np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    overlaps = np.maximum(width, 0) * np.maximum(height, 0) > 1e-5
    checks = []
    for split in splits:
        for name, a, b in [('outer', split['outer_train_indices'], split['outer_held_indices']),
                           ('inner', split['inner_fit_indices'], split['inner_stop_indices'])]:
            assert not set(a) & set(b)
            groups = lambda ix: {rows[i]['spatial_group_id'] for i in ix}
            assert not groups(a) & groups(b)
            pairs = [[rows[a[i]]['source_id'], rows[b[j]]['source_id']]
                     for i, j in np.argwhere(overlaps[np.ix_(a, b)])]
            used = lambda source, ix: set().union(*(source[i] for i in ix))
            shared = sorted(used(donors, a) & used(donors, b))
            declared_shared = sorted(used(declared_donors, a) & used(declared_donors, b))
            checks.append(dict(fold=split['outer_fold'], split=name, overlap_pairs=pairs,
                               shared_actual_donor_tiles=shared, shared_declared_donor_tiles=declared_shared))
    passed = all(not c['overlap_pairs'] and not c['shared_actual_donor_tiles'] and
                 not c['shared_declared_donor_tiles'] for c in checks)
    return dict(passed=passed, checks=checks, targets=len(rows),
                groups=len({r['spatial_group_id'] for r in rows}),
                coverage='Pinned native-build actual donor coverage and full normalization windows; no protected evaluation pixels read')


def numeric_code_equivalence():
    """Training math unchanged; only seed argument and bookkeeping differ."""
    trees = [ast.parse(Path(module.__file__).read_text(encoding='utf-8')) for module in [ORIGINAL, M]]
    find = lambda tree, name: next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    names = ['pixels', 'optimizer', 'weights', 'update_group', 'predict', 'parameter_state', 'load_parameters', 'device_setup']
    for name in names:
        assert ast.dump(find(trees[0], name), include_attributes=False) == ast.dump(find(trees[1], name), include_attributes=False), name
    old = ast.unparse(find(trees[0], 'Classifier'))
    new = ast.unparse(find(trees[1], 'Classifier')).replace(', run_seed)', ')').replace("run_seed + fold", "spec['seed'] + fold")
    assert old == new, 'Numerical classifier implementation drift'
    return dict(exact_AST_functions=names, classifier_AST_equal_after_explicit_seed_argument=True)


def verify_reuse(design):
    old = C.read_json(REUSE / 'protocol.json')
    assert C.sha256(REUSE / 'protocol.json') == (REUSE / 'protocol.sha256').read_text().strip()
    for field in ['files_sha256', 'code_hashes']:
        for path, digest in old[field].items():
            assert C.sha256(path) == digest, path
    assert old['seed'] == SEEDS[0]
    assert old['design']['outer_folds'] == design['outer_folds']
    assert old['design']['ordered_source_ids'] == [r['source_id'] for r in design['rows']]
    receipts = {str(REUSE / 'protocol.json'): C.sha256(REUSE / 'protocol.json')}
    receipts[str(REUSE / 'comparison.json')] = C.sha256(REUSE / 'comparison.json')
    for arm in ARMS:
        for split in design['outer_folds']:
            fold = split['outer_fold']
            folder = REUSE / arm / f'fold{fold}'
            selected = C.read_json(folder / 'select/completed.json')
            fitted = C.read_json(folder / 'refit/completed.json')
            assert selected['status'] == fitted['status'] == 'completed'
            assert selected['best_epoch'] == fitted['selected_epochs'] == fitted['completed_epochs']
            for phase in ['select', 'refit']:
                for name in ['completed.json', 'metrics.json', 'last.pt']:
                    p = folder / phase / name
                    receipts[str(p)] = C.sha256(p)
            p = folder / 'refit/outer_predictions.npz'
            with np.load(p, allow_pickle=False) as d:
                assert d['indices'].tolist() == split['outer_held_indices']
                assert np.array_equal(d['predictions'], d['logits'].argmax(1)) and np.isfinite(d['logits']).all()
            receipts[str(p)] = C.sha256(p)
    return old, dict(status='verified_read_only_reuse', seed=SEEDS[0], source=str(REUSE),
                     files_sha256=receipts, code_equivalence=numeric_code_equivalence(),
                     new_outer_model_evaluations=0, copied_weights=False)


def prepare():
    C.new_directory(OUT)
    plan, next_plan, manifest = [C.read_json(p) for p in [PLAN, NEXT, MANIFEST]]
    design = dict(rows=deepcopy(plan['rows']), outer_folds=next_plan['outer_folds'],
                  spatial_groups=40, unique_targets=1083, grouping='Native224 actual donor components, 40 groups',
                  all_types_independently_reviewed=False)
    assert [r['source_id'] for r in design['rows']] == next_plan['ordered_source_ids']
    assert len(design['rows']) == 1083 and len({r['spatial_group_id'] for r in design['rows']}) == 40
    for row in design['rows']:
        assert C.sha256(row['crop']) == row['crop_sha256']
    guard = spatial_guard(design['rows'], design['outer_folds'], manifest)
    assert guard['passed']
    rejected = C.read_json(ROOT / 'runs/experiments/classifier_lora_multiseed_20261015_v1/design.json')
    negative = spatial_guard(rejected['rows'], rejected['outer_folds'], manifest)
    assert not negative['passed'], 'Guard must reject known overlapping 44-group native scheme'
    write(OUT / 'spatial_guard_verification.json', dict(native40=guard, known_invalid44_negative_control=negative))
    original, receipt = verify_reuse(design)
    write(OUT / 'reuse_receipt.json', receipt)
    write(OUT / 'design.json', design)
    spec = deepcopy(original)
    spec.pop('seed')
    spec.update(created_utc=M.utc(), design=design, seeds=SEEDS, newly_trained_seeds=NEW_SEEDS, arms=ARMS,
                reuse_seed=SEEDS[0], reuse_source=str(REUSE), labels_modified=False,
                source_plan=str(PLAN), source_native_manifest=str(MANIFEST),
                authorization='User explicitly requested start; earlier prepared-plan no_auto_launch was not launch authorization',
                spatial_guard='Actual native donor tiles AND full source windows, outer and inner, checked before training',
                external_validation_images_read=False, automatic_thread_continuation=False,
                snapshot_io='Unique temporary JSON files; bounded Windows replace-lock retries; numerical training unchanged')
    files = [PLAN, NEXT, MANIFEST, OUT / 'design.json', OUT / 'reuse_receipt.json',
             OUT / 'spatial_guard_verification.json', M.P.OUT / 'pooled_features.npz',
             M.P.OUT / 'pooling_weights.npz', M.P.OUT / 'protocol.json']
    code = [Path(__file__), Path(M.__file__), Path(ORIGINAL.__file__), Path(C.__file__),
            Path(M.P.__file__), Path(M.P.R.__file__), Path(M.FENV.__file__),
            ROOT / 'data/scripts/landslide_model_candidates.py', ROOT / 'data/scripts/train_stage2_cls.py',
            ROOT / 'data/scripts/start_classifier_lora_native40.ps1']
    spec['files_sha256'] = {str(p): C.sha256(p) for p in files}
    spec['code_hashes'] = {str(p): C.sha256(p) for p in code}
    assert spec['versions'] == M.P.R.package_versions()
    write(OUT / 'protocol.json', spec)
    (OUT / 'protocol.sha256').write_text(C.sha256(OUT / 'protocol.json') + '\n', encoding='utf-8')
    write(OUT / 'queue.json', dict(status='prepared_not_started', created_utc=M.utc(), seeds=SEEDS,
                                  reused_seed=SEEDS[0], newly_trained_seeds=NEW_SEEDS, spatial_groups=40))
    print('PREPARED_NATIVE40_MULTISEED; correct guard passed, known44 guard rejected', flush=True)


def check():
    spec = C.read_json(OUT / 'protocol.json')
    assert C.sha256(OUT / 'protocol.json') == (OUT / 'protocol.sha256').read_text().strip()
    for field in ['files_sha256', 'code_hashes']:
        for path, digest in spec[field].items():
            assert C.sha256(path) == digest, path
    for name, digest in spec['checkpoint_files_sha256'].items():
        assert C.sha256(Path(spec['checkpoint']) / name) == digest
    for path, digest in C.read_json(OUT / 'reuse_receipt.json')['files_sha256'].items():
        assert C.sha256(path) == digest, path
    assert spec['versions'] == M.P.R.package_versions()
    assert spatial_guard(spec['design']['rows'], spec['design']['outer_folds'], C.read_json(MANIFEST))['passed']
    return spec


def metrics(arm, run_seed, spec, data):
    oof = np.full(len(data['rows']), -1, dtype=int)
    for split in spec['design']['outer_folds']:
        base = REUSE / arm if run_seed == SEEDS[0] else OUT / arm / f'seed{run_seed}'
        with np.load(base / f"fold{split['outer_fold']}" / 'refit/outer_predictions.npz', allow_pickle=False) as d:
            ids = d['indices']
            assert ids.tolist() == split['outer_held_indices'] and np.all(oof[ids] == -1)
            assert np.array_equal(d['predictions'], d['logits'].argmax(1))
            oof[ids] = d['predictions']
    assert np.all(oof >= 0)
    y = data['y'].numpy()
    return dict(seed=run_seed, reused=run_seed == SEEDS[0],
                macro_f1=float(M.f1_score(y, oof, labels=range(4), average='macro', zero_division=0)),
                accuracy=float(np.mean(oof == y)), predictions=oof.tolist(),
                per_class=M.classification_report(y, oof, labels=range(4), target_names=M.NAMES, output_dict=True, zero_division=0),
                confusion_matrix=M.confusion_matrix(y, oof, labels=range(4)).tolist())


def summarize(spec, data):
    per_seed = {arm: {str(s): metrics(arm, s, spec, data) for s in SEEDS} for arm in ARMS}
    aggregate = {}
    for arm in ARMS:
        values = list(per_seed[arm].values())
        aggregate[arm] = dict(macro_f1_mean=float(np.mean([v['macro_f1'] for v in values])),
                              macro_f1_std=float(np.std([v['macro_f1'] for v in values])),
                              accuracy_mean=float(np.mean([v['accuracy'] for v in values])),
                              per_class_recall_mean={n: float(np.mean([v['per_class'][n]['recall'] for v in values])) for n in M.NAMES},
                              per_class_f1_mean={n: float(np.mean([v['per_class'][n]['f1-score'] for v in values])) for n in M.NAMES})
    paired = [per_seed[ARMS[1]][str(s)]['macro_f1'] - per_seed[ARMS[0]][str(s)]['macro_f1'] for s in SEEDS]
    result = dict(status='completed_native40_multiseed_nested_comparison', created_utc=M.utc(), unique_targets=1083,
                  spatial_groups=40, seeds=SEEDS, reused_seed=SEEDS[0], newly_trained_seeds=NEW_SEEDS,
                  per_seed=per_seed, aggregate=aggregate,
                  paired_lora_minus_head=dict(per_seed=paired, mean=float(np.mean(paired)), std=float(np.std(paired)),
                                              wins=sum(d > 0 for d in paired), seeds=3),
                  std_definition='population ddof0 over optimization seeds; not geographic uncertainty',
                  all_types_independently_reviewed=False, labels_modified=False, test_images_read=False,
                  external_validation_images_read=False, protocol_sha256=C.sha256(OUT / 'protocol.json'))
    write(OUT / 'comparison.json', result)
    lines = ['# Native40：三种子四分类对照', '', '1083个目标、40空间组、3折；复用已核验首种子，仅新训另两个。类别标签尚未独立核准；属于内部空间交叉验证。', '',
             '|方案|macro-F1 均值±标准差|正确率|SL召回|SP召回|FL召回|CL召回|', '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        g = aggregate[arm]
        lines.append(f'|{arm}|{g["macro_f1_mean"]:.2%}±{g["macro_f1_std"]:.2%}|{g["accuracy_mean"]:.2%}|'+
                     '|'.join(f'{g["per_class_recall_mean"][n]:.1%}' for n in M.NAMES)+'|')
    lines += ['', '逐种子LoRA−头：' + '、'.join(f'{v:+.2%}' for v in paired), '',
              '3种子同向只能作为初步证据。仍须看逐类表现、人工类型审核与独立地区评估，不能单凭内部macro-F1发布可用结论。']
    (OUT / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(resume=False):
    spec = check()
    ready = C.read_json(OUT / 'readiness.json')
    assert ready['status'] == 'passed_real_GPU_smoke' and ready['protocol_sha256'] == C.sha256(OUT / 'protocol.json')
    state = C.read_json(OUT / 'queue.json')
    assert state['status'] in (['failed', 'running'] if resume else ['prepared_not_started'])
    independence = M.FENV.process_independence()
    assert independence['scheduler_owned']
    M.P.R.ensure_gpu_idle()
    blocked = M.FENV.prevent_network()
    M.device_setup()
    kernel = ctypes.windll.kernel32
    assert kernel.SetThreadExecutionState(0x80000001)
    queue = dict(status='running', started_utc=state.get('started_utc', M.utc()), pid=os.getpid(),
                 phase='loading_verified_local_inputs', spatial_groups=40, seeds=SEEDS,
                 reused_seed=SEEDS[0], newly_trained_seeds=NEW_SEEDS, process_independence=independence,
                 offline=True, temporary_system_awake=True, manual_resume=resume,
                 automatic_thread_continuation=False)
    write(OUT / 'queue.json', queue)
    try:
        data = M.load_data(spec)
        for run_seed in NEW_SEEDS:
            for split in spec['design']['outer_folds']:
                for arm in ARMS:
                    for phase in ['select', 'refit']:
                        M.stage(spec, data, split, arm, phase, queue, run_seed)
        check()
        summarize(spec, data)
        queue.update(status='completed_native40_multiseed_nested_comparison', phase='complete')
    except BaseException as error:
        queue.update(status='failed', error=repr(error))
        raise
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        queue.update(finished_utc=M.utc(), temporary_system_awake=False, blocked_network_attempts=blocked)
        write(OUT / 'queue.json', queue)


# Reuse training math only; substitute orchestration dependencies explicitly.
M.OUT = OUT
M.check = check
M.write = write
M.SEEDS = SEEDS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'check', 'smoke', 'scheduled-run', 'scheduled-resume'])
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'check':
        check()
        print('NATIVE40_PROTOCOL_AND_SOURCE_GUARDS_PASSED')
    elif args.mode == 'smoke':
        M.smoke()
    else:
        suffix = '' if args.mode == 'scheduled-run' else '_resume_' + str(int(time.time()))
        sys.stdout = (OUT / f'training{suffix}.log').open('x', encoding='utf-8', buffering=1)
        sys.stderr = (OUT / f'training{suffix}.err.log').open('x', encoding='utf-8', buffering=1)
        run(resume=args.mode == 'scheduled-resume')
