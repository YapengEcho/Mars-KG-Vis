"""Immutable frozen feature extraction and internal grouped four-class CV.

No downloads, detector-label reads, image augmentation, or external val/test access.
The prepare/preflight commands never load an encoder or fit a classifier. Extract
freezes an already cached encoder. CV is a separate, explicit future CPU action.
Extraction defaults to CPU; --device cuda explicitly selects offline FP32 GPU
inference and records that execution mode separately from the frozen data plan.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / 'runs/research/manual_classification_protocol_20261014_v1/protocol.json'
DEFAULT_OUT = ROOT / 'runs/research/classification_next_20261014_v1'
CLASS_NAMES = ('SL', 'SP', 'FL', 'CL')
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
POLICY = 'training_internal_catalog_label_exploration_not_acceptance_or_independent_test'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_json_atomic(path, value):
    """Readers see a whole status/receipt even while the extractor is running."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    write_json(temporary, value)
    temporary.replace(path)


def new_directory(path):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f'Prior results preserved; choose a new directory: {path}')
    path.mkdir(parents=True)
    return path


def eligible_rows(protocol, allow_catalog_labels=False):
    """One canonical catalog ID per row; no detector class 0 conversion."""
    if protocol['class_mapping']['classifier_class_to_id'] != CLASS_TO_ID:
        raise ValueError('Classifier mapping must be explicitly SL/SP/FL/CL, not detector labels')
    if sha256(protocol['input_manifest']) != protocol['input_manifest_sha256']:
        raise ValueError('Source manifest changed since the grouping protocol was prepared')
    seen, tile_groups, digest_groups = set(), {}, {}
    result = []
    for row in protocol['targets']:
        identifier = str(row['source_id']).strip()
        if not identifier.isdecimal() or int(identifier) <= 0:
            raise ValueError('Invalid catalog source ID')
        identifier = str(int(identifier))
        if identifier in seen:
            raise ValueError(f'Duplicate source_id: {identifier}')
        seen.add(identifier)
        group = row['spatial_group_id']
        for tile in row['source_tiles']:
            previous = tile_groups.setdefault(tile, group)
            if previous != group:
                raise ValueError('The same tile crosses spatial groups')
        if not row['exploratory_eligible']:
            continue
        if not allow_catalog_labels and not row['subtype_training_approved']:
            continue
        name = row['catalog_class']
        if name not in CLASS_TO_ID or row['class_id'] != CLASS_TO_ID[name]:
            raise ValueError(f'Conflicting catalog subtype mapping for {identifier}')
        if not allow_catalog_labels and row['class_reviewed'] is not True:
            raise ValueError('Approved subtype row is not independently reviewed')
        if not row.get('crop_sha256') or not row.get('crop'):
            raise ValueError(f'Missing crop provenance for {identifier}')
        previous = digest_groups.setdefault(row['crop_sha256'], group)
        if previous != group:
            raise ValueError('Identical crop hashes cross spatial groups; resolve identity before CV')
        copied = dict(row)
        copied['source_id'] = identifier
        result.append(copied)
    if not result:
        raise ValueError('No approved subtype rows. Catalog exploration requires --allow-catalog-labels')
    return sorted(result, key=lambda row: int(row['source_id']))


def class_counts(labels):
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=4)
    return {name: int(counts[i]) for i, name in enumerate(CLASS_NAMES)}


def assert_fold_isolation(rows, fold_ids, folds, min_held_fl=2, min_train_fl=5):
    fold_ids = np.asarray(fold_ids)
    if fold_ids.shape != (len(rows),) or set(fold_ids.tolist()) != set(range(folds)):
        raise ValueError('Incomplete fold assignment')
    y = np.array([r['class_id'] for r in rows])
    groups = np.array([r['spatial_group_id'] for r in rows])
    ids = np.array([r['source_id'] for r in rows])
    if len(set(ids)) != len(ids):
        raise ValueError('Repeated source_id cannot be treated as an independent target')
    summaries = []
    for fold in range(folds):
        held, train = np.flatnonzero(fold_ids == fold), np.flatnonzero(fold_ids != fold)
        if set(groups[held]) & set(groups[train]) or set(ids[held]) & set(ids[train]):
            raise ValueError('A source ID or tile-connected group crosses a CV fold')
        train_count, held_count = class_counts(y[train]), class_counts(y[held])
        if min(train_count.values()) == 0 or min(held_count.values()) == 0:
            raise ValueError('Every train and held partition must contain all four catalog classes')
        if held_count['FL'] < min_held_fl or train_count['FL'] < min_train_fl:
            raise ValueError('FL counts are too small for the declared development-fold minimum')
        summaries.append({'fold': fold, 'train_targets': len(train), 'held_targets': len(held),
                          'train_counts': train_count, 'held_counts': held_count,
                          'train_groups': len(set(groups[train])), 'held_groups': len(set(groups[held])),
                          'train_source_ids': ids[train].tolist(), 'held_source_ids': ids[held].tolist()})
    return summaries


def make_folds(rows, folds=3, seed=20261014, attempts=200):
    """Choose folds using catalog counts only, before any features or scores exist."""
    from sklearn.model_selection import StratifiedGroupKFold

    if folds not in (2, 3):
        raise ValueError('FL has only 15 IDs; this initial development protocol permits 2 or 3 folds')
    y = np.array([r['class_id'] for r in rows])
    groups = np.array([r['spatial_group_id'] for r in rows])
    fl_groups = len(set(groups[y == CLASS_TO_ID['FL']]))
    if fl_groups < folds:
        raise ValueError('Too few independent FL groups')
    candidates = []
    global_prop = np.bincount(y, minlength=4) / len(y)
    for offset in range(attempts):
        candidate_seed = seed + offset
        assigned = np.full(len(rows), -1, dtype=np.int64)
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=candidate_seed)
        for fold, (_, held) in enumerate(splitter.split(np.zeros((len(y), 1)), y, groups)):
            assigned[held] = fold
        try:
            summary = assert_fold_isolation(rows, assigned, folds)
        except ValueError:
            continue
        # Count-only balancing; never inspect model outcomes to choose a split.
        size = np.array([s['held_targets'] for s in summary]) / len(rows)
        proportions = np.array([[s['held_counts'][c] / s['held_targets'] for c in CLASS_NAMES] for s in summary])
        objective = float(np.sum((size - 1 / folds) ** 2) + np.sum((proportions - global_prop) ** 2))
        candidates.append((objective, candidate_seed, assigned, summary))
    if not candidates:
        raise ValueError('No group-isolated split satisfies all four classes and FL minimum counts')
    score, selected_seed, assigned, summary = min(candidates, key=lambda x: (x[0], x[1]))
    return assigned, summary, {'base_seed': seed, 'selected_count_only_seed': selected_seed,
                              'candidate_seeds_checked': attempts, 'count_balance_objective': score}


def cached_dino_snapshot():
    cache = Path(os.environ.get('HF_HUB_CACHE', str(Path.home() / '.cache/huggingface/hub')))
    candidates = []
    for path in (cache / 'models--facebook--dinov2-base/snapshots').glob('*'):
        if (path / 'config.json').is_file() and (path / 'model.safetensors').is_file():
            candidates.append(path.resolve())
    return str(sorted(candidates)[-1]) if candidates else None


def preflight(protocol_path=DEFAULT_PROTOCOL):
    import importlib.util

    protocol = read_json(protocol_path)
    rows = eligible_rows(protocol, allow_catalog_labels=True)
    local_dino = cached_dino_snapshot()
    return {'status': 'preflight_only_no_encoder_loaded_or_classifier_fitted',
            'scope': POLICY, 'source_protocol': str(Path(protocol_path).resolve()),
            'source_protocol_sha256': sha256(protocol_path),
            'eligible_unique_targets': len(rows), 'catalog_counts': class_counts([r['class_id'] for r in rows]),
            'approved_subtype_targets': sum(r['subtype_training_approved'] for r in rows),
            'modules': {name: importlib.util.find_spec(name) is not None for name in ('torch', 'torchvision', 'transformers', 'sklearn', 'timm')},
            'local_dinov2_snapshot': local_dino,
            'local_dinov2_weights_bytes': (Path(local_dino) / 'model.safetensors').stat().st_size if local_dino else None,
            'resource_receipt': {'cpu_only': True, 'gpu_used': False, 'downloads': False, 'training_started': False},
            'remaining_conditions': ['Catalog subtype review, especially 15 FL and 94 SP',
                                     'Source identity holds remain excluded',
                                     'Tile components do not prove buffer/adjacent-context separation',
                                     'Independent final type regions are not present in this training-only manifest']}


def prepare(protocol_path=DEFAULT_PROTOCOL, out=DEFAULT_OUT, allow_catalog_labels=False, folds=3, seed=20261014):
    protocol_path = Path(protocol_path).resolve()
    protocol = read_json(protocol_path)
    rows = eligible_rows(protocol, allow_catalog_labels)
    fold_ids, summaries, split_receipt = make_folds(rows, folds, seed)
    missing = [r['source_id'] for r in rows if not Path(r['crop']).is_file()]
    if missing:
        raise ValueError(f'Missing crop files: {missing[:10]}')
    plan = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'prepared_no_encoder_loaded_no_features_extracted_no_classifiers_fitted',
            'scope': POLICY, 'allow_catalog_labels': allow_catalog_labels,
            'source_protocol': str(protocol_path), 'source_protocol_sha256': sha256(protocol_path),
            'source_manifest': protocol['input_manifest'], 'source_manifest_sha256': protocol['input_manifest_sha256'],
            'class_names': list(CLASS_NAMES), 'class_mapping': CLASS_TO_ID,
            'unique_source_count': len(rows), 'catalog_counts': class_counts([r['class_id'] for r in rows]),
            'approved_subtype_count': sum(r['subtype_training_approved'] for r in rows),
            'folds': folds, 'fold_assignment_receipt': split_receipt, 'fold_summaries': summaries,
            'group_policy': protocol['group_policy'],
            'preprocessing': {'encoder_mode': 'eval, requires_grad=False, CPU, float32, inference_mode',
                              'input': 'full square crop RGB, direct bilinear antialias resize to 224; no center crop or augmentation',
                              'normalization': {'mean': [.485, .456, .406], 'std': [.229, .224, .225]},
                              'dimension': 'DINOv2-base CLS=768; local ResNet18 average-pooled=512',
                              'fold_transform': 'StandardScaler fit only to each fold train rows; no PCA or tuning'},
            'cv_comparison': {'heads': ['logistic_regression', 'linear_svm'], 'C': 1.0,
                              'balance_modes': ['none', 'sqrt', 'balanced'],
                              'weight_counts_scope': 'Each fold training rows only',
                              'hyperparameter_search': False,
                              'final_deployment_classifier': False,
                              'warning': 'Fixed C is a screening protocol; FL scores from 15 IDs are unstable. No final test claims.'},
            'held_source_ids': [r['source_id'] for r in protocol['targets'] if not r['exploratory_eligible']],
            'rows': [dict(r, fold_id=int(fold)) for r, fold in zip(rows, fold_ids)],
            'resource_receipt': {'gpu_used': False, 'encoder_loaded': False, 'features_extracted': False,
                                 'training_started': False, 'detector_runner_modified': False,
                                 'source_manifest_modified': False, 'downloads': False}}
    out = new_directory(out)
    write_json(out / 'plan.json', plan)
    write_json(out / 'environment_preflight.json', preflight(protocol_path))
    return {'status': plan['status'], 'plan': str(out / 'plan.json'), 'targets': len(rows),
            'fold_counts': [s['held_counts'] for s in summaries]}


def verify_plan(plan_path):
    plan = read_json(plan_path)
    for source, expected in (('source_protocol', 'source_protocol_sha256'), ('source_manifest', 'source_manifest_sha256')):
        if sha256(plan[source]) != plan[expected]:
            raise ValueError(f'{source} changed after the plan was frozen')
    if plan['scope'] != POLICY or plan['class_mapping'] != CLASS_TO_ID:
        raise ValueError('Not the declared internal four-class development protocol')
    protocol = read_json(plan['source_protocol'])
    expected = eligible_rows(protocol, plan['allow_catalog_labels'])
    fields = ('source_id', 'class_id', 'catalog_class', 'spatial_group_id', 'crop', 'crop_sha256', 'class_reviewed')
    if len(expected) != len(plan['rows']) or any(any(a[k] != b[k] for k in fields) for a, b in zip(expected, plan['rows'])):
        raise ValueError('Plan rows disagree with source protocol')
    assert_fold_isolation(plan['rows'], [r['fold_id'] for r in plan['rows']], plan['folds'])
    return plan


def prepare_pixels(images):
    """Preserve every part of each square crop instead of a central crop."""
    import torch
    from PIL import Image

    rgb = [np.asarray(im.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8) for im in images]
    pixels = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2).float() / 255
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    return (pixels - mean) / std


def load_frozen_encoder(kind, checkpoint, device='cpu'):
    """All weights must already exist locally; no AutoModel online fallback."""
    import torch
    import importlib.metadata

    if device not in ('cpu', 'cuda'):
        raise ValueError('Frozen extraction permits only explicit cpu or cuda execution')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable; CPU fallback would change the declared run')
    actual_device = torch.device('cuda', torch.cuda.current_device()) if device == 'cuda' else torch.device('cpu')
    if device == 'cuda':
        # FP32 is the declared comparison mode: no autocast or TF32 shortcuts.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    checkpoint = Path(checkpoint).resolve()
    if kind == 'dinov2':
        from transformers import AutoModel

        if not checkpoint.is_dir() or not (checkpoint / 'model.safetensors').is_file():
            raise ValueError('DINOv2 needs an existing local snapshot directory with safetensors')
        config = read_json(checkpoint / 'config.json')
        if config.get('model_type') != 'dinov2' or config.get('hidden_size') != 768 or config.get('patch_size') != 14:
            raise ValueError('Expected the declared DINOv2-base/14 encoder')
        files = {name: sha256(checkpoint / name) for name in ('config.json', 'model.safetensors')}
        model = AutoModel.from_pretrained(str(checkpoint), local_files_only=True, use_safetensors=True)
        def endpoint(pixels):
            return model(pixel_values=pixels).last_hidden_state[:, 0]
        dimension, endpoint_name = 768, 'last_hidden_state[:, 0] CLS token'
        provider = 'facebook/dinov2-base local HuggingFace snapshot; exact bytes hashed'
    elif kind == 'resnet18':
        from torchvision.models import resnet18

        if not checkpoint.is_file():
            raise ValueError('ResNet18 needs an existing local torchvision state dict')
        files = {checkpoint.name: sha256(checkpoint)}
        model = resnet18(weights=None)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        model.fc = torch.nn.Identity()
        endpoint = model
        dimension, endpoint_name = 512, 'avgpool flattened, fc=Identity'
        provider = 'User-supplied local state dict; ImageNet origin must be separately verified'
    else:
        raise ValueError('Unsupported encoder')
    model.to(actual_device).float().eval().requires_grad_(False)
    if model.training or any(p.requires_grad or p.device != actual_device for p in model.parameters()):
        raise RuntimeError('Frozen encoder device/gradient invariant failed')
    if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
        raise RuntimeError('Frozen encoder parameters must remain float32')
    if any(b.device != actual_device or (b.is_floating_point() and b.dtype != torch.float32) for b in model.buffers()):
        raise RuntimeError('Frozen encoder buffers disagree with the declared device/precision')
    receipt = {'encoder': kind, 'checkpoint': str(checkpoint), 'checkpoint_files_sha256': files,
               'provider_weight_origin': provider,
               'package_versions': {name: importlib.metadata.version(name) for name in
                                    ('torch', 'torchvision', 'transformers', 'pillow', 'numpy')},
               'feature_endpoint': endpoint_name, 'features': dimension,
               'requested_device': device, 'device': str(actual_device),
               'parameter_dtype': 'float32', 'output_dtype': 'float32', 'autocast_enabled': False,
               'tf32_enabled': False if device == 'cuda' else None,
               'cuda_device_name': torch.cuda.get_device_name(actual_device) if device == 'cuda' else None,
               'cuda_runtime': torch.version.cuda if device == 'cuda' else None,
               'requires_grad': False, 'mode': 'eval', 'downloads': False,
               'pretraining_region_overlap_unknown': True,
               'warning': 'Frozen generic pretraining provenance is separate from supervised spatial folds.'}
    return endpoint, dimension, receipt


def extract(plan_path, out, encoder='dinov2', checkpoint=None, batch_size=4, threads=4, device='cpu'):
    import torch
    from PIL import Image

    if batch_size < 1 or threads < 1:
        raise ValueError('Batch size and CPU thread count must be positive')
    if device not in ('cpu', 'cuda'):
        raise ValueError('Device must be cpu or cuda')
    plan = verify_plan(plan_path)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f'Prior feature results preserved: {out}')
    out = new_directory(out)
    started = time.monotonic()
    progress = {'status': 'verifying_crops', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'plan': str(Path(plan_path).resolve()), 'plan_sha256': sha256(plan_path),
                'extractor_script_sha256': sha256(__file__),
                'encoder': encoder, 'requested_device': device,
                'completed_feature_rows': 0, 'verified_crop_rows': 0, 'total': len(plan['rows']),
                'batch_size': batch_size, 'threads': threads, 'cpu_only': device == 'cpu',
                'downloads': False, 'classifier_training_started': False}
    def publish(status):
        progress.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=round(time.monotonic() - started, 3))
        write_json_atomic(out / 'status.json', progress)
    try:
        publish('verifying_crops')
        checkpoint = checkpoint or (cached_dino_snapshot() if encoder == 'dinov2' else None)
        if not checkpoint:
            raise ValueError('No cached checkpoint; this entry point does not download weights')
        # Verify every crop hash before loading an encoder, on either device.
        for index, row in enumerate(plan['rows']):
            if sha256(row['crop']) != row['crop_sha256']:
                raise ValueError(f"Changed crop: {row['source_id']}")
            progress['verified_crop_rows'] = index + 1
            if (index + 1) % 100 == 0:
                publish('verifying_crops')
        torch.set_num_threads(threads)
        publish('loading_local_frozen_encoder')
        endpoint, dimension, receipt = load_frozen_encoder(encoder, checkpoint, device)
        actual_device = torch.device(receipt['device'])
        progress['device'] = receipt['device']
        chunks = []
        publish('extracting')
        with torch.inference_mode(), torch.autocast(device_type=actual_device.type, enabled=False):
            for start in range(0, len(plan['rows']), batch_size):
                rows = plan['rows'][start:start + batch_size]
                images = []
                try:
                    for row in rows:
                        with Image.open(row['crop']) as im:
                            if im.width != im.height:
                                raise ValueError('Declared square subtype crop was not square')
                            images.append(im.convert('RGB'))
                    features = endpoint(prepare_pixels(images).to(actual_device, dtype=torch.float32))
                    if features.dtype != torch.float32 or features.device != actual_device:
                        raise RuntimeError('Frozen features disagree with the declared device/precision')
                    vector = features.detach().cpu().numpy().astype(np.float32)
                finally:
                    for im in images:
                        im.close()
                if vector.shape != (len(rows), dimension) or not np.isfinite(vector).all():
                    raise RuntimeError('Unexpected or non-finite frozen features')
                chunks.append(vector)
                progress['completed_feature_rows'] = start + len(rows)
                publish('extracting')
                print(json.dumps(progress), flush=True)
        X = np.concatenate(chunks)
        publish('saving_verified_features')
        temporary_features = out / 'features.incomplete.npz'
        np.savez_compressed(temporary_features, X=X,
                            y=np.array([r['class_id'] for r in plan['rows']], dtype=np.int64),
                            source_id=np.array([r['source_id'] for r in plan['rows']]),
                            group=np.array([r['spatial_group_id'] for r in plan['rows']]),
                            fold_id=np.array([r['fold_id'] for r in plan['rows']], dtype=np.int64))
        temporary_features.replace(out / 'features.npz')
        receipt.update({'status': 'frozen_features_extracted_no_classifier_fitted', 'scope': POLICY,
                        'plan': str(Path(plan_path).resolve()), 'plan_sha256': sha256(plan_path),
                        'extractor_script_sha256': progress['extractor_script_sha256'],
                        'plan_declared_encoder_mode': plan['preprocessing']['encoder_mode'],
                        'execution_device_is_explicit_cli_choice': True,
                        'cross_device_bitwise_equivalence_claimed': False,
                        'feature_file_sha256': sha256(out / 'features.npz'), 'shape': list(X.shape),
                        'torch_version': str(torch.__version__), 'threads': threads, 'batch_size': batch_size,
                        'elapsed_seconds': round(time.monotonic() - started, 3),
                        'class_mapping': CLASS_TO_ID, 'source_ids': [r['source_id'] for r in plan['rows']],
                        'subtype_labels_independently_reviewed': not plan['allow_catalog_labels'],
                        'gpu_used': actual_device.type == 'cuda', 'classifier_training_started': False})
        write_json_atomic(out / 'receipt.json', receipt)
        publish('frozen_features_extracted_no_classifier_fitted')
        return receipt
    except BaseException as error:
        progress.update(error_type=type(error).__name__, error=str(error),
                        output_is_complete=False)
        publish('interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed')
        raise


def load_verified_features(plan_path, feature_dir):
    plan = verify_plan(plan_path)
    feature_dir = Path(feature_dir)
    receipt = read_json(feature_dir / 'receipt.json')
    if receipt['status'] != 'frozen_features_extracted_no_classifier_fitted':
        raise ValueError('Feature extraction did not complete successfully')
    status_path = feature_dir / 'status.json'
    if status_path.exists() and read_json(status_path)['status'] != 'frozen_features_extracted_no_classifier_fitted':
        raise ValueError('Feature progress is incomplete or failed; do not fit a classifier')
    if receipt['plan_sha256'] != sha256(plan_path) or receipt['feature_file_sha256'] != sha256(feature_dir / 'features.npz'):
        raise ValueError('Feature/plan provenance mismatch')
    with np.load(feature_dir / 'features.npz', allow_pickle=False) as cache:
        X, y = cache['X'], cache['y']
        for name, expected in (('source_id', [r['source_id'] for r in plan['rows']]),
                               ('group', [r['spatial_group_id'] for r in plan['rows']]),
                               ('fold_id', [r['fold_id'] for r in plan['rows']]),
                               ('y', [r['class_id'] for r in plan['rows']])):
            if not np.array_equal(cache[name], expected):
                raise ValueError(f'Feature cache row alignment failure: {name}')
        if X.ndim != 2 or X.shape != (len(plan['rows']), receipt['features']) or not np.isfinite(X).all():
            raise ValueError('Invalid feature matrix')
        return plan, X, y, cache['fold_id'], receipt


def grouped_cv(plan_path, feature_dir, out):
    """Explicit future action: CPU head fits; does not export a final classifier."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from train_stage2_cls import classifier_class_weights

    plan, X, y, fold_ids, receipt = load_verified_features(plan_path, feature_dir)
    if Path(out).exists():
        raise FileExistsError('Prior CV results preserved; choose a new output directory')
    results = []
    for head in plan['cv_comparison']['heads']:
        for balance in plan['cv_comparison']['balance_modes']:
            predictions = np.full(len(y), -1, dtype=np.int64)
            fold_reports = []
            for fold in range(plan['folds']):
                train, held = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
                scaler = StandardScaler().fit(X[train])
                weights = classifier_class_weights(y[train], balance)
                common = dict(C=plan['cv_comparison']['C'], class_weight=weights, max_iter=10000, random_state=20261014)
                classifier = LogisticRegression(**common) if head == 'logistic_regression' else LinearSVC(dual='auto', **common)
                # Convergence failure is a failed configuration, never a publishable score.
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    classifier.fit(scaler.transform(X[train]), y[train])
                pred = classifier.predict(scaler.transform(X[held]))
                predictions[held] = pred
                fold_reports.append({'fold': fold, 'train_only_class_weights': weights,
                                     'held_counts': class_counts(y[held]),
                                     'macro_f1': float(f1_score(y[held], pred, labels=range(4), average='macro', zero_division=0)),
                                     'confusion_matrix': confusion_matrix(y[held], pred, labels=range(4)).tolist()})
            if np.any(predictions < 0):
                raise RuntimeError('Not every independent target received exactly one held-fold prediction')
            group_reports = []
            spatial_groups = np.array([r['spatial_group_id'] for r in plan['rows']])
            for group in sorted(set(spatial_groups)):
                mask = spatial_groups == group
                present = np.unique(y[mask])
                group_reports.append({'spatial_group_id': str(group),
                                      'unique_targets': int(mask.sum()), 'catalog_counts': class_counts(y[mask]),
                                      'macro_f1_observed_classes': float(f1_score(y[mask], predictions[mask], labels=present, average='macro', zero_division=0)),
                                      'confusion_matrix': confusion_matrix(y[mask], predictions[mask], labels=range(4)).tolist()})
            results.append({'head': head, 'balance_mode': balance, 'fixed_C': plan['cv_comparison']['C'],
                            'out_of_fold_macro_f1': float(f1_score(y, predictions, labels=range(4), average='macro', zero_division=0)),
                            'per_class': classification_report(y, predictions, labels=range(4), target_names=CLASS_NAMES, output_dict=True, zero_division=0),
                            'confusion_matrix': confusion_matrix(y, predictions, labels=range(4)).tolist(),
                            'folds': fold_reports, 'spatial_groups': group_reports,
                            'source_order_predictions': predictions.tolist()})
    out = new_directory(out)
    report = {'status': 'internal_grouped_catalog_diagnostic_not_acceptance', 'scope': POLICY,
              'plan_sha256': sha256(plan_path), 'feature_receipt_sha256': sha256(Path(feature_dir) / 'receipt.json'),
              'unique_targets': len(y), 'catalog_counts': class_counts(y), 'encoder': receipt['encoder'],
              'class_names': list(CLASS_NAMES), 'source_ids': [r['source_id'] for r in plan['rows']],
              'results': results, 'final_classifier_exported': False, 'external_val_test_read': False,
              'limitations': ['15 unreviewed FL IDs cannot establish reliable FL acceptance',
                              'Catalog labels have not been separately adjudicated',
                              'Tile-connected internal groups do not certify adjacent-context buffer isolation',
                              'Detection-trained regions cannot provide independent pipeline test scores']}
    write_json(out / 'cv_report.json', report)
    return {'status': report['status'], 'report': str(out / 'cv_report.json'), 'configurations': len(results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('preflight')
    check.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    prep = commands.add_parser('prepare')
    prep.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    prep.add_argument('--out', type=Path, default=DEFAULT_OUT)
    prep.add_argument('--allow-catalog-labels', action='store_true')
    prep.add_argument('--folds', type=int, default=3)
    prep.add_argument('--seed', type=int, default=20261014)
    ext = commands.add_parser('extract')
    ext.add_argument('--plan', type=Path, required=True)
    ext.add_argument('--out', type=Path, required=True)
    ext.add_argument('--encoder', choices=('dinov2', 'resnet18'), default='dinov2')
    ext.add_argument('--checkpoint', type=Path)
    ext.add_argument('--batch-size', type=int, default=4)
    ext.add_argument('--threads', type=int, default=4)
    ext.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    cv = commands.add_parser('cv')
    cv.add_argument('--plan', type=Path, required=True)
    cv.add_argument('--feature-dir', type=Path, required=True)
    cv.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'preflight':
        result = preflight(args.protocol)
    elif args.command == 'prepare':
        result = prepare(args.protocol, args.out, args.allow_catalog_labels, args.folds, args.seed)
    elif args.command == 'extract':
        result = extract(args.plan, args.out, args.encoder, args.checkpoint, args.batch_size, args.threads, args.device)
    else:
        result = grouped_cv(args.plan, args.feature_dir, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
