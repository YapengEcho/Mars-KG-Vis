"""Fixed-classifier development evaluation of the completed two-seed optimizer pair; no downloads/training."""
from collections import Counter
from copy import deepcopy
import ctypes
import gc
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.venv/Lib/site-packages'), str(ROOT / 'data/scripts')]
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
import run_latest_detector_classifier_bridge_20261024 as L
import run_classifier_head_rebalance_native40 as H
import reuse_catalog_native_caches_20261024 as K
from build_native_classification_dataset import NativeReader, histogram_lut
from PIL import Image

C, M, D, B, np, torch = L.C, L.M, L.D, L.B, L.np, L.torch
OUT = ROOT / 'runs/experiments/optimizer_joint_bridge_20261025_v1'
REPEAT = ROOT / 'runs/experiments/detector_optimizer_repeat_20261024_v1'
NATIVE = ROOT / 'data/processed/catalog_native_regions_20261024_v1'
TASK = 'MarsKG-optimizer-joint-bridge-20261025-v1'
SEEDS = [20261015, 20261016, 20261017]
BASE = 10893
REPLAY = 3396
ARMS = ['adamw_lr3e4', 'sgd_lr1e2']


def write(path, value):
    L.write(path, value)


def prepare():
    assert not OUT.exists(), 'Preserve previous joint evaluation'
    assert C.read_json(REPEAT / 'queue.json')['status'] == 'completed'
    plan = deepcopy(C.read_json(L.PREP / 'input_plan.json'))
    assert len(plan['cases']) == BASE
    old_scores = C.read_json(L.OUT / 'comparison.json')
    assert C.read_json(L.OUT / 'independent_completion_verification.json')['score_cells_verified'] == 270
    inherited = C.read_json(L.OUT / 'protocol.json')
    old_names = ['optimizer_' + a for a in ARMS]
    plan['models'] = {name: plan['models'][name] for name in old_names}
    thresholds = {n: inherited['thresholds'][n] for n in old_names}
    expected = {n: inherited['expected_detection_metrics'][n] for n in old_names}
    pins = {str(p): C.sha256(p) for p in [Path(__file__), Path(L.__file__), Path(B.__file__),
            Path(sys.modules[H.SOURCE_CTOR.__module__].__file__),
            L.PREP / 'input_plan.json', L.OUT / 'protocol.json', L.OUT / 'comparison.json',
            L.OUT / 'independent_completion_verification.json', REPEAT / 'protocol.json', REPEAT / 'comparison.json',
            ROOT / 'data/scripts/start_optimizer_joint_bridge_20261025.ps1',
            D.OUT / 'protocol.json', ROOT / 'data/processed/manual_stage2_native_20261015_v1/manifest.json']}
    assert H.SOURCE_CTOR.__module__ == 'run_classifier_lora_multiseed'
    native = {}
    for tile in plan['sources']:
        p = NATIVE / tile / 'manifest.json'; d = C.read_json(p)
        assert d['status'] == 'verified_complete_native_regions'
        pins[str(p)] = C.sha256(p); native[tile] = d
    training = C.read_json(ROOT / 'data/processed/manual_stage2_native_20261015_v1/manifest.json')
    train_world = np.array([r['native_window_world_m'] for r in training['samples']], np.float64)
    items = {r['stem']: r for r in plan['items']}
    unique = {(r['tile'], tuple(float(v).hex() for v in r['box_native'])): r['index'] for r in plan['cases']}
    omitted = []; required_new_predictions = 0
    for arm in ARMS:
        folder = REPEAT / 'evaluation' / arm
        metrics = C.read_json(folder / 'metrics.json'); predictions = C.read_json(folder / 'predictions.json')
        name = 'optimizer_seed1_' + arm
        thresholds[name] = {b: metrics['budgets'][b]['metrics']['threshold'] for b in ['300', '600', '1071']}
        expected[name] = {b: metrics['budgets'][b]['metrics'] for b in thresholds[name]}
        for p in [folder / 'metrics.json', folder / 'predictions.json', REPEAT / 'training' / arm / 'audit.json',
                  REPEAT / 'training' / arm / 'train/weights/best.pt']:
            pins[str(p)] = C.sha256(p)
        assert metrics['checkpoint_sha256'] == C.sha256(REPEAT / 'training' / arm / 'train/weights/best.pt')
        selected = {}
        for stem, item in items.items():
            boxes, confidence, classes = [np.asarray(v, np.float32) for v in predictions[stem]]
            boxes = boxes.reshape(-1, 4); needed = set()
            gt = [(0, *l['box']) for l in item['labels']]
            for budget in ['300', '600', '1071']:
                threshold = thresholds[name][budget]; keep = np.flatnonzero(confidence >= threshold)
                matches = B.match_image(gt, boxes, confidence, classes, .5, threshold, class_aware=False)
                needed.update(int(keep[m[4]]) for m in matches if m[1])
            rows = []
            for j in np.flatnonzero(confidence >= thresholds[name]['1071']):
                index = None
                if int(j) in needed:
                    required_new_predictions += 1
                    clipped = np.clip(boxes[j].astype(np.float64), [0]*4, [item['width'], item['height']]*2)
                    box = B.native_box(clipped, item); key = item['tile'], tuple(float(v).hex() for v in box)
                    if key in unique:
                        index = unique[key]
                    else:
                        win, geometry = B.planned_window(box, plan['sources'][item['tile']])
                        assert K.covered(win, [e['window_px'] for e in native[item['tile']]['entries']]), 'Required matched ROI lacks native pixels'
                        world = B.world_box(plan['sources'][item['tile']]['geotransform'], win)
                        overlap = ((np.minimum(train_world[:, 2], world[2])-np.maximum(train_world[:, 0], world[0]) > 1e-5) &
                                   (np.minimum(train_world[:, 3], world[3])-np.maximum(train_world[:, 1], world[1]) > 1e-5))
                        assert not overlap.any(), 'Classifier normalization window touches training pixels'
                        side = win[2]-win[0]; index = len(plan['cases']); unique[key] = index
                        projected = [(box[0]-win[0])*224/side, (box[1]-win[1])*224/side,
                                     (box[2]-win[0])*224/side, (box[3]-win[1])*224/side]
                        assert 0 <= projected[0] < projected[2] <= 224 and 0 <= projected[1] < projected[3] <= 224
                        plan['cases'].append(dict(index=index, tile=item['tile'], box_native=box, window_px=win,
                            target_box224=projected, geometry=geometry, world_window=world,
                            crop=str(OUT / 'inputs' / f'roi_{index:05d}.png')))
                else:
                    omitted.append(dict(model=name, stem=stem, original_index=int(j),
                                        reason='Unmatched at every fixed budget; subtype cannot affect any reported joint metric'))
                rows.append(dict(original_index=int(j), box=boxes[j].tolist(), score=float(confidence[j]), case_index=index))
            selected[stem] = dict(predictions=rows)
        plan['models'][name] = dict(prediction_cache=str(folder / 'predictions.json'), items=selected)
    for seed in SEEDS:
        for p in [D.OUT / f'seed{seed}/last.pt', D.OUT / f'seed{seed}/completed.json',
                  L.OUT / f'seed{seed}/dino_predictions.npz']:
            pins[str(p)] = C.sha256(p)
    # Confirm every fixed matched pair has a classifier case before seeing any new subtype result.
    for name, model in plan['models'].items():
        for budget in ['300', '600', '1071']:
            count = tp = 0
            for stem, entry in model['items'].items():
                threshold = thresholds[name][budget]
                rows = [r for r in entry['predictions'] if r['score'] >= threshold]
                matches = B.match_image([(0, *l['box']) for l in items[stem]['labels']],
                    np.array([r['box'] for r in rows], np.float32).reshape(-1, 4),
                    np.array([r['score'] for r in rows], np.float32), np.zeros(len(rows), np.float32), .5, threshold, class_aware=False)
                count += len(rows); tp += sum(m[1] for m in matches)
                assert all(rows[m[4]]['case_index'] is not None for m in matches if m[1])
            assert count == int(budget) and tp == expected[name][budget]['tp']
    OUT.mkdir(parents=True); (OUT / 'inputs').mkdir(); (OUT / 'receipts').mkdir()
    write(OUT / 'input_plan.json', plan)
    write(OUT / 'omitted_unmatched_subtype_inference.json', omitted)
    spec = dict(status='locked_before_new_classifier_inference', created_utc=L.utc(), files_sha256=pins,
        plan_sha256=C.sha256(OUT / 'input_plan.json'), seeds=SEEDS, thresholds=thresholds,
        expected_detection_metrics=expected, archived_cases=BASE, replay_cases=REPLAY, new_cases=len(plan['cases'])-BASE,
        required_new_predictions=required_new_predictions, classifier='Archived DINOv2 plus last2 q/v rank4 LoRA, unchanged three seeds',
        native_manifests={t:str(NATIVE / t / 'manifest.json') for t in native}, batch=8,
        training=False, encoder_updates=0, downloads=False, final_test_read=False,
        reject_or_filter_candidates=False, matched_only_type_inference_for_new_pools=True,
        rule='All fixed candidates remain in precision denominator; only fixed greedy matched pairs require subtype inference. No rejection or rematching.',
        limits=['Repeatedly used development set, not blind acceptance', 'FL2 CL15 independent development IDs',
                'Both detector seeds reported; never select the highest seed',
                'Seed0 AdamW and seed1 SGD include epoch-boundary EMA restores',
                'Unmatched new candidates have no subtype output; this is a metric evaluation, not full-application prediction export'])
    write(OUT / 'protocol.json', spec); (OUT / 'protocol.sha256').write_text(C.sha256(OUT / 'protocol.json')+'\n')
    write(OUT / 'queue.json', dict(status='prepared_not_started', completed_seeds=0, total_seeds=3, training=False, downloads=False))
    print('PREPARED_FIXED_OPTIMIZER_JOINT_EVAL', spec['new_cases'], 'new native ROIs; all matches covered; no downloads', flush=True)


def sources(spec):
    result = {}
    for tile, path in spec['native_manifests'].items():
        d = C.read_json(path)
        result[tile] = [dict(tile=tile, path=e['path'], bytes=e['bytes'], mtime_ns=e['mtime_ns'],
            geotransform=e['geotransform'], size_px=e['size_px'], cache_sha256=e['sha256']) for e in d['entries']]
    return result


def render(case, plan, pool, reader, dest):
    x0, y0, x1, y1 = case['window_px']; w, h = x1-x0, y1-y0
    t = plan['sources'][case['tile']]['geotransform']
    gt = [t[0]+x0*t[1], t[1], 0., t[3]+y0*t[5], 0., t[5]]
    scratch = OUT / 'scratch'; scratch.mkdir(exist_ok=True)
    import shutil
    assert shutil.disk_usage(scratch).free > 2*w*h+(4 << 30)
    token = uuid.uuid4().hex
    paths = [scratch / (token+'_raw.bin'), scratch / (token+'_stretch.bin')]
    arrays = []; image = None; histogram = np.zeros(256, np.int64); digest = hashlib.sha256()
    try:
        raw = np.memmap(paths[0], mode='w+', shape=(h, w), dtype=np.uint8); arrays.append(raw)
        entry = dict(geotransform=gt, size_px=[w, h], sources=pool[case['tile']])
        for row, pixels, _ in reader.slabs(entry):
            raw[row:row+len(pixels)] = pixels; histogram += np.bincount(pixels.ravel(), minlength=256); digest.update(pixels.tobytes())
        raw.flush(); lut, contrast = histogram_lut(histogram)
        stretched = np.memmap(paths[1], mode='w+', shape=(h, w), dtype=np.uint8); arrays.append(stretched)
        for row in range(0, h, 128):
            stretched[row:row+128] = lut[raw[row:row+128]]
        stretched.flush(); image = Image.frombuffer('L', (w, h), stretched, 'raw', 'L', 0, 1)
        with image.resize((224, 224), Image.Resampling.BILINEAR) as resized:
            resized.save(dest)
        return dict(native_pixels_sha256=digest.hexdigest(), contrast=contrast, crop_sha256=C.sha256(dest))
    finally:
        if image is not None:
            image.close()
        for array in arrays:
            array._mmap.close()
        for path in paths:
            assert path.resolve().parent == scratch.resolve()
            path.unlink(missing_ok=True)


def run():
    spec = C.read_json(OUT / 'protocol.json')
    assert C.sha256(OUT / 'protocol.json') == (OUT / 'protocol.sha256').read_text().strip()
    for path, digest in spec['files_sha256'].items():
        assert C.sha256(path) == digest, path
    assert C.sha256(OUT / 'input_plan.json') == spec['plan_sha256']
    state = C.read_json(OUT / 'queue.json'); assert state['status'] in ['prepared_not_started', 'failed_resumable', 'paused_by_user']
    state.update(status='running', pid=os.getpid(), phase='native_crop_replay', started_utc=L.utc())
    write(OUT / 'queue.json', state)
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    reader = NativeReader(budget=512 << 20)
    try:
        plan = C.read_json(OUT / 'input_plan.json'); pool = sources(spec)
        # Replay one historical, covered small window per donor tile through the new reader.
        replayed = []
        for tile in plan['sources']:
            manifests = C.read_json(spec['native_manifests'][tile])['entries']
            case = next(r for r in plan['cases'][:BASE] if r['tile'] == tile and
                (r['window_px'][2]-r['window_px'][0])**2 <= 20_000_000 and K.covered(r['window_px'], [e['window_px'] for e in manifests]))
            dest = OUT / 'receipts' / (tile+'_native_replay.png')
            render(case, plan, pool, reader, dest)
            with Image.open(case['crop']) as expected, Image.open(dest) as actual:
                assert np.array_equal(np.array(actual), np.array(expected)), 'Native cache crop differs from archived pixels'
            replayed.append(dict(tile=tile, case_index=case['index'], exact_resized_pixels=True))
        write(OUT / 'native_replay.json', dict(status='passed_all_six_donor_native_crop_replays', cases=replayed))
        for i, case in enumerate(plan['cases'][BASE:], 1):
            receipt = OUT / 'receipts' / f"roi_{case['index']:05d}.json"
            if receipt.exists():
                assert C.sha256(case['crop']) == C.read_json(receipt)['crop_sha256']
            else:
                assert not Path(case['crop']).exists(), 'Preserve unreceipted crop'
                r = render(case, plan, pool, reader, Path(case['crop']))
                write(receipt, dict(r, case_index=case['index'], window_px=case['window_px']))
            if i % 20 == 0 or i == spec['new_cases']:
                state.update(phase='building_native_crops', completed_crops=i, total_crops=spec['new_cases']); write(OUT / 'queue.json', state)
        while reader.cache:
            reader.evict()
        needed = plan['cases'][:REPLAY]+plan['cases'][BASE:]
        external = D.external_data(dict(cases=needed))
        crop_pins = {r['crop']: C.sha256(r['crop']) for r in needed}
        write(OUT / 'crop_files_sha256.json', crop_pins)
        M.P.R.ensure_gpu_idle(); M.FENV.prevent_network(); M.device_setup(); torch.set_num_threads(2)
        dspec = C.read_json(D.OUT / 'protocol.json'); train = M.load_data(dspec)
        results = {}
        for seed in SEEDS:
            folder = OUT / f'seed{seed}'; folder.mkdir(exist_ok=True)
            model = H.SOURCE_CTOR(dspec, 'qv_last2_LoRA', list(range(1083)), train, 0, 'refit', seed)
            saved = torch.load(D.OUT / f'seed{seed}/last.pt', map_location='cpu', weights_only=False)
            M.load_parameters(model, saved['parameters'])
            def infer(start, stop):
                logits = []
                for offset in range(start, stop, 8):
                    ids = np.arange(offset, min(offset+8, stop)); logits.append(M.predict(model, external, ids)[1])
                    if (offset-start) % 128 == 0:
                        state.update(phase='inference', seed=seed, completed_ROIs=offset+len(ids), total_ROIs=len(needed)); write(OUT / 'queue.json', state)
                return np.concatenate(logits)
            with np.load(L.OUT / f'seed{seed}/dino_predictions.npz') as cached:
                old = cached['logits'].copy(); assert old.shape == (BASE, 4)
            replay = infer(0, REPLAY)
            assert np.array_equal(replay, old[:REPLAY]), 'Archived classifier logits did not replay bitwise'
            fresh = infer(REPLAY, len(needed)); matrix = np.concatenate([old, fresh])
            assert matrix.shape == (len(plan['cases']), 4) and np.isfinite(matrix).all()
            np.savez_compressed(folder / 'dino_predictions.npz', logits=matrix, predictions=matrix.argmax(1))
            result = L.scores(plan, matrix.argmax(1), spec)
            old_result = C.read_json(L.OUT / f'seed{seed}/scores.json')['dino']
            for name in ['optimizer_'+a for a in ARMS]:
                assert result[name] == old_result[name], 'Historical seed0 joint metrics changed'
            write(folder / 'scores.json', result)
            write(folder / 'replay_receipt.json', dict(archived_logits_bitwise=True, replayed_ROIs=REPLAY,
                checkpoint_sha256=C.sha256(D.OUT / f'seed{seed}/last.pt'), seed=seed))
            results[str(seed)] = result
            del model, saved; gc.collect(); torch.cuda.empty_cache()
            state.update(completed_seeds=SEEDS.index(seed)+1); write(OUT / 'queue.json', state)
        for path, digest in crop_pins.items():
            assert C.sha256(path) == digest
        write(OUT / 'comparison.json', dict(status='completed_fixed_DINO_joint_optimizer_repeat', per_seed=results,
            protocol_sha256=C.sha256(OUT / 'protocol.json'), training=False, downloads=False, final_test_read=False,
            all_fixed_candidates_counted=True, unmatched_new_subtypes_not_inferred=True))
        state.update(status='completed', phase='complete_no_training_or_downloads', finished_utc=L.utc()); write(OUT / 'queue.json', state)
    except BaseException as error:
        state.update(status='failed_resumable', error=repr(error), updated_utc=L.utc()); write(OUT / 'queue.json', state)
        raise
    finally:
        while reader.cache:
            reader.evict()
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    if sys.argv[1:] == ['prepare']:
        prepare()
    elif sys.argv[1:] == ['run']:
        with (OUT / 'worker.log').open('a', encoding='utf-8', buffering=1) as stream:
            sys.stdout = stream; sys.stderr = stream
            try:
                run()
            except BaseException:
                traceback.print_exc(file=stream); raise
    else:
        raise ValueError('Use prepare or run')
