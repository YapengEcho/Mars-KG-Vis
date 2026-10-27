"""Prepare a controlled P3/P2 comparison and run explicitly bounded smoke training.

The smoke results are engineering checks, never model-quality evidence. Original
training labels and frozen evaluation files are only read. All cache writes,
snapshots and checkpoints live under the new experiment directory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'runs/experiments/p2_detector_probe_20261013_v1'
ORIGINAL = ROOT / 'runs/experiments/manual_box_v8_20261009'
EVAL = ROOT / 'data/processed/manual_eval_frozen_20261011_v1/frozen.json'
EXPECTED_EVAL_SHA = 'd6db367fd3317623b27c96def521c9ed0210152cbbe2bfd52b16e53ae175793c'
ARMS = ('control', 'p2')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def common_key(key, end=15):
    fields = key.split('.')
    return len(fields) > 2 and fields[0] == 'model' and fields[1].isdigit() and int(fields[1]) <= end


def copy_shared_state(source, target, end=15):
    """Copy only tensors belonging to explicitly equivalent common layers."""
    selected = {key: value.detach().clone() for key, value in source.items() if common_key(key, end)}
    if not selected:
        raise ValueError('Empty shared-layer initialization')
    for key, value in selected.items():
        if key not in target or target[key].shape != value.shape:
            raise ValueError('Shared-layer shape mismatch: ' + key)
    with __import__('torch').no_grad():
        for key, value in selected.items():
            target[key].copy_(value.to(target[key].device, target[key].dtype))
    return selected


def state_hash(state, common_only=False):
    import torch
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        if common_only and not common_key(key):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def precision_evidence(inputs, output, device='cuda'):
    import torch
    return dict(autocast_enabled=torch.is_autocast_enabled(device),
                autocast_dtype=str(torch.get_autocast_dtype(device)),
                input_dtype=str(inputs[0].dtype), output_dtype=str(output.dtype),
                device=str(output.device))


def mosaic_transform_state(transforms):
    """Inspect actual Mosaic transform probabilities without walking datasets."""
    probabilities = []
    visited = {}
    def visit(value, path, parent_probability=1.0):
        if value is None or parent_probability <= visited.get(id(value), -1.0):
            return
        # Shared pre_transform objects may first be reached through a disabled
        # MixUp and later directly; keep the most reachable path, not first path.
        visited[id(value)] = parent_probability
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f'{path}[{index}]', parent_probability)
            return
        probability = float(getattr(value, 'p', 1.0))
        effective = parent_probability * probability
        if any(base.__name__ == 'Mosaic' for base in type(value).__mro__):
            probabilities.append(dict(path=path, p=probability, reachable_probability=effective))
        for attribute in ('transforms', 'pre_transform'):
            if hasattr(value, attribute):
                visit(getattr(value, attribute), path + '.' + attribute, effective)
    visit(transforms, 'transforms')
    return dict(probabilities=probabilities,
                active=any(row['reachable_probability'] > 0 for row in probabilities))


def spread_select(items, count):
    items = sorted(items, key=lambda p: str(p))
    if len(items) < count:
        raise ValueError('Not enough source images for the specified smoke composition')
    return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)] if count > 1 else items[:1]


def filter_held_images(train, samples, hold_ids):
    """Hold complete images with questioned labels, never convert them to negatives."""
    hold_ids = set(hold_ids)
    cases = []
    held_stems = set()
    for sample in samples:
        label_ids = {int(row['source_id']) for row in sample.get('labels', [])}
        matching = sorted(label_ids & hold_ids)
        if matching:
            held_stems.add(sample['stem'])
            cases.append(dict(stem=sample['stem'], tile=sample['tile'], label_source_ids=sorted(label_ids),
                held_source_ids=matching, reason='neighbor_label_identity_or_geographic_association_requires_review'))
    observed = {sid for case in cases for sid in case['held_source_ids']}
    if observed != hold_ids:
        raise ValueError('Requested hold IDs absent from the frozen label manifest: ' + str(sorted(hold_ids - observed)))
    filtered = [path for path in train if path.stem not in held_stems]
    for case in cases:
        case['removed_training_exposures'] = sum(path.stem == case['stem'] for path in train)
    return filtered, cases


def link_image(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return 'hardlink_read_only_training_use'
    except OSError:
        shutil.copy2(source, target)
        return 'copy'


def receipt_check(path, expected):
    if sha(path) != expected:
        raise ValueError('Immutable input changed: ' + str(path))


def environment():
    import torch
    import ultralytics
    package = Path(ultralytics.__file__).parent
    code = ['engine/trainer.py', 'models/yolo/detect/train.py', 'data/dataset.py',
            'nn/tasks.py', 'nn/modules/head.py', 'utils/loss.py', 'cfg/default.yaml']
    return dict(python=platform.python_version(), torch=torch.__version__,
                ultralytics=ultralytics.__version__, cuda=torch.version.cuda,
                cuda_available=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                gpu_total_mib=torch.cuda.get_device_properties(0).total_memory / 2**20 if torch.cuda.is_available() else None,
                bf16_supported=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
                package_code={p: sha(package / p) for p in code})


def prepare(out, hold_ids=(), planned_batch=2):
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    import yaml

    out = out.resolve()
    if out.exists():
        raise FileExistsError('Preserve the existing experiment; use a new directory')
    receipt_check(ORIGINAL / 'experiment.json', (ORIGINAL / 'experiment.sha256').read_text().strip())
    original = read(ORIGINAL / 'experiment.json')
    receipt_check(ORIGINAL / 'input_checksums.json', original['input_checksums_sha256'])
    receipt_check(Path(original['weights']), original['weights_sha256'])
    receipt_check(Path(original['dataset_root']) / 'export_plan.json', original['dataset_plan_sha256'])
    receipt_check(EVAL, EXPECTED_EVAL_SHA)
    receipt_check(Path(original['arms']['manual']['train_list']), original['arms']['manual']['train_list_sha256'])
    source_rows = [row for row in read(ORIGINAL / 'input_checksums.json')['manual']
                   if Path(row['path']).parent.name == 'train']
    train = [Path(line) for line in Path(original['arms']['manual']['train_list']).read_text(encoding='utf-8').splitlines()]
    unique = sorted(set(train))
    if len(train) != 4868 or len(unique) != 2375 or len(source_rows) != 2 * len(unique):
        raise ValueError('Unexpected frozen training composition')
    for index, row in enumerate(source_rows):
        receipt_check(row['path'], row['sha256'])
        if index % 1000 == 0:
            print('TRAIN_SOURCE_HASH_CHECK', index, len(source_rows), flush=True)
    export_plan = read(Path(original['dataset_root']) / 'export_plan.json')
    train, held_cases = filter_held_images(train, export_plan['samples'], hold_ids)
    unique = sorted(set(train))
    frozen = read(EVAL)
    val = [item for item in frozen['items'] if (item['arm'], item['split']) == ('offset', 'val')]
    if len(val) != 173:
        raise ValueError('Expected exactly 173 rebuilt offset validation images')
    for item in val:
        for kind in ['image', 'label']:
            receipt_check(resolve_path(item[kind]), item[kind + '_sha256'])

    out.mkdir(parents=True)
    write(out / 'state.json', dict(status='preparing', formal_training_started=False))
    write(out / 'held_cases.json', dict(source_plan_sha256=original['dataset_plan_sha256'],
        hold_ids=sorted(hold_ids), held_images=len(held_cases),
        removed_training_exposures=sum(case['removed_training_exposures'] for case in held_cases), cases=held_cases))
    package = Path(ultralytics.__file__).parent
    image_map = {}
    copied = []
    for image in unique:
        label = image.parents[1].parent / 'labels/train' / (image.stem + '.txt')
        for source, dest, kind in [(image, out / 'dataset/images/train' / image.name, 'image'),
                                   (label, out / 'dataset/labels/train' / label.name, 'label')]:
            dest.parent.mkdir(parents=True, exist_ok=True)
            method = link_image(source, dest) if kind == 'image' else 'copy'
            if kind == 'label':
                shutil.copy2(source, dest)
            copied.append(dict(source=str(source), path=str(dest), sha256=sha(dest), method=method))
        image_map[image] = (out / 'dataset/images/train' / image.name).resolve()
    val_map = {}
    for item in val:
        image = resolve_path(item['image']); label = resolve_path(item['label'])
        dest_image = out / 'dataset/images/val' / image.name
        dest_label = out / 'dataset/labels/val' / label.name
        link_image(image, dest_image); dest_label.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(label, dest_label)
        val_map[image] = dest_image.resolve()
        copied.extend([dict(source=str(image), path=str(dest_image), sha256=item['image_sha256'], method='hardlink_or_copy'),
                       dict(source=str(label), path=str(dest_label), sha256=item['label_sha256'], method='copy')])
    exposures = [image_map[p].as_posix() for p in train]
    (out / 'dataset/train.txt').write_text('\n'.join(exposures) + '\n', encoding='utf-8')
    full_data = dict(path=(out / 'dataset').as_posix(), train=(out / 'dataset/train.txt').as_posix(),
                     val='images/val', names={0: 'LANDSLIDE'})
    (out / 'dataset/data.yaml').write_text(yaml.safe_dump(full_data, sort_keys=False), encoding='utf-8')
    positive = [p for p in unique if (p.parents[1].parent / 'labels/train' / (p.stem + '.txt')).read_text(encoding='utf-8').strip()]
    positive_set = set(positive)
    negative = [p for p in unique if p not in positive_set]
    smoke_train = spread_select(positive, 8) + spread_select(negative, 8)
    smoke_val = spread_select(list(val_map), 8)
    smoke_records = []
    for split, images, mapping in [('train', smoke_train, image_map), ('val', smoke_val, val_map)]:
        for source_image in images:
            image = mapping[source_image]
            label = image.parents[1].parent / 'labels' / split / (image.stem + '.txt')
            dest_image = out / 'smoke_dataset/images' / split / image.name
            dest_label = out / 'smoke_dataset/labels' / split / label.name
            link_image(image, dest_image); dest_label.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(label, dest_label)
            smoke_records.append(dict(split=split, image=str(dest_image), label=str(dest_label),
                                      image_sha256=sha(dest_image), label_sha256=sha(dest_label),
                                      positive=bool(label.read_text(encoding='utf-8').strip())))
    (out / 'smoke_dataset/data.yaml').write_text(yaml.safe_dump(dict(path=(out / 'smoke_dataset').as_posix(),
        train='images/train', val='images/val', names={0: 'LANDSLIDE'}), sort_keys=False), encoding='utf-8')
    write(out / 'source_receipt.json', dict(training_original=original,
        original_experiment_sha256=sha(ORIGINAL / 'experiment.json'), training_checked_files=len(source_rows),
        frozen_eval_sha256=sha(EVAL), validation_checked_images=len(val), test_images_read=0,
        train_entries=len(train), train_unique_images=len(unique), train_positive_images=len(positive),
        train_negative_images=len(negative), held_cases_sha256=sha(out / 'held_cases.json'),
        copied_inputs=copied, smoke_inputs=smoke_records))
    models = {}
    source_checkpoint = torch.load(original['weights'], map_location='cpu', weights_only=False)
    source_model = (source_checkpoint.get('ema') or source_checkpoint['model']).float()
    source_state = source_model.state_dict()
    for arm, official in [('control', 'yolov8.yaml'), ('p2', 'yolov8-p2.yaml')]:
        cfg = yaml.safe_load((package / 'cfg/models/v8' / official).read_text(encoding='utf-8'))
        cfg['nc'] = 1; cfg['scale'] = 'n'
        cfg_path = out / 'configs' / ('yolov8n.yaml' if arm == 'control' else 'yolov8n-p2.yaml')
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        init_seeds(0, deterministic=True)
        model = YOLO(str(cfg_path)).model.cpu().float()
        model.names = {0: 'LANDSLIDE'}
        selected = copy_shared_state(source_state, model.state_dict())
        checkpoint = out / 'initial_weights' / (arm + '_seed0.pt')
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=deepcopy(model).float(), ema=None, train_args={'task': 'detect', 'model': str(cfg_path)},
                        date=datetime.now(timezone.utc).isoformat(), version=ultralytics.__version__), checkpoint)
        models[arm] = dict(config=str(cfg_path), config_sha256=sha(cfg_path), checkpoint=str(checkpoint),
            checkpoint_sha256=sha(checkpoint), full_state_sha256=state_hash(model.state_dict()),
            common_state_sha256=state_hash(model.state_dict(), common_only=True),
            parameters=sum(p.numel() for p in model.parameters()), strides=model.stride.tolist(),
            copied_tensors=[dict(name=k, shape=list(v.shape), elements=v.numel()) for k, v in sorted(selected.items())],
            fresh_tensor_names=[key for key in model.state_dict() if not common_key(key)])
        print('INITIALIZED', arm, models[arm]['parameters'], models[arm]['strides'], flush=True)
    if models['control']['common_state_sha256'] != models['p2']['common_state_sha256']:
        raise ValueError('Shared initialization differs between paired architectures')
    settings = deepcopy(original['training'])
    settings.update(epochs=60, close_mosaic=50, patience=20, batch=planned_batch, nbs=64,
                    project=str(out / 'pilot'), exist_ok=False, pretrained=True)
    settings.pop('resume', None)
    protocol = dict(status='prepared_pilot_not_started', formal_training_started=False,
        created_utc=datetime.now(timezone.utc).isoformat(), preparation_code_sha256=sha(__file__),
        source_checkpoint=str(original['weights']), source_checkpoint_sha256=original['weights_sha256'],
        environment=environment(), dataset=str(out / 'dataset/data.yaml'), dataset_sha256=sha(out / 'dataset/data.yaml'),
        smoke_dataset=str(out / 'smoke_dataset/data.yaml'), smoke_dataset_sha256=sha(out / 'smoke_dataset/data.yaml'),
        source_receipt_sha256=sha(out / 'source_receipt.json'), models=models, seed=0, training=settings,
        held_cases_sha256=sha(out / 'held_cases.json'),
        batch_requires_smoke_confirmation=True,
        policies={'test': 'not read, not used for parameter choice',
                  'initialization': 'same seed, COCO shared layers 0-15 only; all subsequent layers initialized anew',
                  'mosaic': 'closed at zero-based epoch 10 / displayed epoch 11; actual event must be recorded',
                  'stopping': 'max60 / patience20; best.pt chosen on current validation; early completion accepted',
                  'fairness': 'concurrent control is necessary: old six runs used different initialization and recipe',
                  'quality_claim': 'smoke cannot be used as model-quality evidence'})
    write(out / 'protocol.json', protocol)
    (out / 'protocol.sha256').write_text(sha(out / 'protocol.json') + '\n', encoding='utf-8')
    write(out / 'state.json', dict(status='prepared', formal_training_started=False))
    return protocol


def load_protocol(out):
    receipt_check(out / 'protocol.json', (out / 'protocol.sha256').read_text().strip())
    spec = read(out / 'protocol.json')
    receipt_check(out / 'source_receipt.json', spec['source_receipt_sha256'])
    if spec.get('held_cases_sha256'):
        receipt_check(out / 'held_cases.json', spec['held_cases_sha256'])
    for model in spec['models'].values():
        receipt_check(model['config'], model['config_sha256']); receipt_check(model['checkpoint'], model['checkpoint_sha256'])
    receipt_check(spec['dataset'], spec['dataset_sha256'])
    receipt_check(spec['smoke_dataset'], spec['smoke_dataset_sha256'])
    current = environment()
    if current != spec['environment']:
        raise ValueError('Prepared runtime environment has changed')
    return spec


def nested_state_hash(value):
    """Fingerprint optimizer state after its SDK FP16 -> parameter-FP32 cast."""
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu()
            if tensor.is_floating_point():
                tensor = tensor.float()
            digest.update(state_hash({'tensor': tensor}).encode())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda k: str(k)):
                digest.update(('key:' + str(key)).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b'list')
            for child in item: visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True).encode())
    visit(value)
    return digest.hexdigest()


def checkpoint_resume_context(run, expected_args):
    """Read CPU checkpoint metadata; reject finished/mismatched continuation."""
    import torch
    checkpoint_path = run / 'weights/last.pt'; best_path = run / 'weights/best.pt'
    checkpoint_sha = sha(checkpoint_path); best_sha = sha(best_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    best = torch.load(best_path, map_location='cpu', weights_only=False)
    completed = int(checkpoint.get('epoch', -1)) + 1
    if not 0 < completed < int(expected_args['epochs']) or checkpoint.get('optimizer') is None:
        raise ValueError('Checkpoint has no unfinished resumable optimizer state')
    saved_args = checkpoint['train_args']
    for key, expected in expected_args.items():
        if key == 'resume':
            continue
        # SDK persists CUDA device 0 as the canonical string "0".
        same = str(saved_args.get(key)) == str(expected) if key == 'device' else saved_args.get(key) == expected
        if key not in saved_args or not same:
            raise ValueError('Frozen resume argument differs: ' + key)
    ema = checkpoint.get('ema')
    if ema is None or checkpoint.get('updates') is None:
        raise ValueError('Checkpoint has no EMA/update state')
    fitness = float(checkpoint['best_fitness']); best_epoch = int(best['epoch']) + 1
    if not math.isfinite(fitness) or float(best['best_fitness']) != fitness or not 0 < best_epoch <= completed:
        raise ValueError('Best checkpoint disagrees with last checkpoint selection state')
    patience = int(expected_args['patience'])
    if patience > 0 and completed - best_epoch >= patience:
        raise ValueError('Checkpoint already reached early stopping; do not restart its patience clock')
    training_times = checkpoint.get('train_results', {}).get('time', [])
    elapsed = float(training_times[-1]) if training_times else 0.0
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Invalid accumulated checkpoint training time')
    return dict(checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
        best_checkpoint_sha256=best_sha, completed_epochs=completed, start_epoch=completed,
        displayed_resume_epoch=completed + 1, best_epoch=best_epoch, best_fitness=fitness,
        ema_updates=int(checkpoint['updates']), expected_student_sha256=state_hash(ema.float().state_dict()),
        expected_ema_sha256=state_hash(ema.state_dict()),
        expected_optimizer_sha256=nested_state_hash(checkpoint['optimizer']),
        saved_optimizer_lrs=[group['lr'] for group in checkpoint['optimizer']['param_groups']],
        prior_training_seconds=elapsed, scaler_state=checkpoint.get('scaler', {}),
        rng_state_saved=False, student_restoration='SDK resumes student from saved EMA; raw student is absent',
        limitation='Epoch-boundary continuation, not bitwise replay; RNG/sampler/current partial epoch and gradients were not saved')


def resumed_audit(previous, context):
    """Retain committed counters; detach events belonging to interrupted epochs."""
    audit = deepcopy(previous)
    completed = context['completed_epochs']; updates = context['ema_updates']
    epochs = [row for row in audit['epochs'] if int(row['displayed_epoch']) <= completed]
    if [row['displayed_epoch'] for row in epochs] != list(range(1, completed + 1)):
        raise ValueError('No complete contiguous audit history at the checkpoint boundary')
    if epochs[-1]['optimizer_steps'] != updates:
        raise ValueError('Checkpoint EMA updates and committed optimizer audit differ')
    batches = int(audit['batches_per_epoch']) * completed
    if len(audit['batch_seconds']) < batches or len(audit['validation_passes']) < completed:
        raise ValueError('Committed checkpoint is ahead of available audited batches/validation')
    interrupted_events = [event for event in audit['mosaic_close_events'] if event['displayed_epoch'] > completed]
    audit['mosaic_close_events'] = [event for event in audit['mosaic_close_events'] if event['displayed_epoch'] <= completed]
    audit.setdefault('interrupted_mosaic_close_events', []).extend(deepcopy(interrupted_events))
    audit['epochs'] = epochs; audit['batch_seconds'] = audit['batch_seconds'][:batches]
    audit['validation_passes'] = audit['validation_passes'][:completed]
    audit.update(completed_epochs=completed, actual_optimizer_steps=updates,
        finite_loss_checks=batches, finite_gradient_checks=updates,
        nonzero_gradient_steps=min(audit['nonzero_gradient_steps'], updates))
    audit['observed_oom_retries'] = 0
    return audit


def restore_stopper(trainer, context):
    """Called after SDK resume_training, before the first resumed epoch."""
    if trainer.start_epoch != context['start_epoch'] or trainer.epochs <= trainer.start_epoch:
        raise ValueError('SDK did not resume at the expected checkpoint epoch')
    if float(trainer.best_fitness) != context['best_fitness']:
        raise ValueError('SDK did not restore maximum best fitness')
    trainer.stopper.best_epoch = context['best_epoch']
    trainer.stopper.best_fitness = context['best_fitness']
    delta = context['completed_epochs'] - context['best_epoch']
    trainer.stopper.possible_stop = delta >= trainer.stopper.patience - 1


def execute(out, arm, *, smoke, batch, attempt, resume=False):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import init_seeds

    spec = load_protocol(out)
    if not spec['environment']['cuda_available'] or not spec['environment']['bf16_supported']:
        raise ValueError('CUDA BF16 hardware required')
    mode = 'smoke' if smoke else 'pilot'
    run = out / mode / f'{attempt}_b{batch}' / f'{arm}_seed0'
    if resume and (smoke or not run.exists()):
        raise ValueError('Resume requires an existing pilot run')
    if run.exists() and not resume:
        raise FileExistsError('Existing runs must be retained: ' + str(run))
    receipt = read(out / 'source_receipt.json')
    records = receipt['smoke_inputs'] if smoke else receipt['copied_inputs']
    for row in records:
        if smoke:
            receipt_check(row['image'], row['image_sha256']); receipt_check(row['label'], row['label_sha256'])
        else:
            receipt_check(row['path'], row['sha256'])
    kwargs = deepcopy(spec['training'])
    kwargs.update(batch=batch, seed=0, data=spec['smoke_dataset'] if smoke else spec['dataset'],
                  project=str(run.parent), name=run.name, exist_ok=True)
    if smoke:
        kwargs.update(epochs=1, nbs=batch, warmup_epochs=0.0, close_mosaic=0, patience=20)
    audit = dict(status='starting', formal_result=False if smoke else 'pilot_development_result',
        arm=arm, mode=mode, seed=0, requested_batch=batch, requested_epochs=kwargs['epochs'],
        effective_code_sha256=sha(__file__), protocol_sha256=sha(out / 'protocol.json'),
        completed_epochs=0, actual_optimizer_steps=0, finite_gradient_checks=0,
        nonzero_gradient_steps=0, finite_loss_checks=0, batch_seconds=[],
        validation_passes=[], epochs=[], mosaic_close_events=[], observed_oom_retries=0,
        mosaic_audit_version=2,
        initialization_checkpoint_sha256=spec['models'][arm]['checkpoint_sha256'])
    resume_context = None
    if resume:
        resume_context = checkpoint_resume_context(run, kwargs)
        previous = read(run / 'training_audit.json')
        if previous.get('protocol_sha256') != sha(out / 'protocol.json') or previous.get('arm') != arm:
            raise ValueError('Resume audit belongs to a different protocol/arm')
        audit = resumed_audit(previous, resume_context)
        history_number = len(audit.get('resume_history', [])) + 1
        history = run / 'resume_history' / f'{history_number:03d}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
        history.mkdir(parents=True, exist_ok=False)
        backup = []
        for source in [run / 'training_audit.json', run / 'results.csv', run / 'args.yaml',
                       run / 'weights/last.pt', run / 'weights/best.pt']:
            target = history / source.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
            digest = sha(source); receipt_check(target, digest)
            backup.append(dict(source=str(source), backup=str(target), sha256=digest))
        receipt_check(run / 'weights/last.pt', resume_context['checkpoint_sha256'])
        receipt_check(run / 'weights/best.pt', resume_context['best_checkpoint_sha256'])
        record = dict(resume_context, history_number=history_number,
            started_utc=datetime.now(timezone.utc).isoformat(), effective_code_sha256=sha(__file__), backups=backup,
            prior_audit_status=previous['status'], interrupted_mosaic_events=deepcopy(previous.get('mosaic_close_events', [])),
            prior_training_precision=deepcopy(previous.get('training_precision')))
        audit.setdefault('resume_history', []).append(record)
        audit.pop('training_precision', None)
        audit.update(status='running', effective_code_sha256=sha(__file__))
        write(history / 'recovery_plan.json', record)
        kwargs['resume'] = resume_context['checkpoint']
    else:
        run.mkdir(parents=True)
        audit['status'] = 'running'
    write(run / 'training_audit.json', audit)
    if not smoke:
        write(out / 'state.json', dict(status='pilot_running', formal_training_started=True,
            active_arm=arm, active_run=str(run)))
    started = time.perf_counter()
    trainer_reference = [None]

    class AuditedTrainer(DetectionTrainer):
        def get_model(self, *args, **kw):
            model = super().get_model(*args, **kw)
            # Architecture consumes different initialization RNG; reset explicitly
            # so model size does not silently change the shuffled data stream.
            init_seeds(0, deterministic=True)
            return model

        def _build_train_pipeline(self):
            audit['observed_oom_retries'] = max(audit['observed_oom_retries'], getattr(self, '_oom_retries', 0))
            if self.batch_size != batch:
                raise RuntimeError('OOM changed frozen batch; rerun BOTH arms at the same lower batch in a new attempt')
            return super()._build_train_pipeline()

        def preprocess_batch(self, value):
            value = super().preprocess_batch(value)
            if resume_context and 'first_resume_batch_image_sha256' not in audit['resume_history'][-1]:
                audit['resume_history'][-1]['first_resume_batch_image_sha256'] = state_hash({'image': value['img']})
                audit['resume_history'][-1]['first_resume_batch_image_files'] = value['im_file']
            # Verify identical shuffled/augmented input in the paired startup.
            # Keep every tiny-smoke batch, but avoid full-run CPU image copies.
            if smoke or 'first_batch_image_sha256' not in audit:
                image_hash = state_hash({'image': value['img']})
                audit.setdefault('checked_batch_image_sha256', []).append(image_hash)
                if 'first_batch_image_sha256' not in audit:
                    audit['first_batch_image_sha256'] = image_hash
                    audit['first_batch_image_files'] = value['im_file']
                    other = 'p2' if arm == 'control' else 'control'
                    peer = run.parent / f'{other}_seed0/training_audit.json'
                    if peer.exists() and read(peer).get('first_batch_image_sha256') != image_hash:
                        raise ValueError('Paired augmented first image batch differs')
            return value

        def save_metrics(self, metrics):
            # SDK resets train_time_start each process. Keep CSV elapsed time
            # monotonic without counting laptop-off downtime as training.
            original_start = self.train_time_start
            if resume_context:
                self.train_time_start -= resume_context['prior_training_seconds']
            try:
                return super().save_metrics(metrics)
            finally:
                self.train_time_start = original_start

        def _close_dataloader_mosaic(self):
            audit['mosaic_close_events'].append(dict(zero_based_epoch=self.epoch, displayed_epoch=self.epoch + 1,
                before=mosaic_transform_state(self.train_loader.dataset.transforms)))
            value = super()._close_dataloader_mosaic()
            audit['mosaic_close_events'][-1]['after'] = mosaic_transform_state(self.train_loader.dataset.transforms)
            if audit['mosaic_close_events'][-1]['after']['active']:
                raise RuntimeError('Actual Mosaic transform remained active after the close event')
            write(run / 'training_audit.json', audit)
            return value

        def optimizer_step(self):
            nonzero = False
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all().item():
                        raise RuntimeError('Non-finite gradient')
                    nonzero |= bool(parameter.grad.any().item())
            audit['finite_gradient_checks'] += 1
            audit['nonzero_gradient_steps'] += int(nonzero)
            return super().optimizer_step()

    def initialized(trainer):
        trainer_reference[0] = trainer
        expected = resume_context['expected_student_sha256'] if resume_context else spec['models'][arm]['full_state_sha256']
        actual = state_hash(trainer.model.state_dict())
        if actual != expected:
            raise ValueError('Trainer did not preserve checkpoint/initialization state')
        if resume_context:
            restore_stopper(trainer, resume_context)
            ema_hash = state_hash(trainer.ema.ema.state_dict())
            optimizer_hash = nested_state_hash(trainer.optimizer.state_dict())
            if ema_hash != resume_context['expected_ema_sha256'] or trainer.ema.updates != resume_context['ema_updates']:
                raise ValueError('SDK EMA recovery differs from the saved checkpoint')
            if optimizer_hash != resume_context['expected_optimizer_sha256']:
                raise ValueError('SDK optimizer recovery differs from the saved checkpoint')
            if trainer.scheduler.last_epoch != resume_context['start_epoch'] - 1:
                raise ValueError('SDK resumed scheduler at the wrong epoch')
            if trainer.scaler.state_dict() != resume_context['scaler_state']:
                raise ValueError('SDK scaler recovery differs from the saved checkpoint')
            audit['resume_history'][-1].update(status='verified_restored', restored_student_sha256=actual,
                restored_ema_sha256=ema_hash, restored_optimizer_sha256=optimizer_hash,
                restored_start_epoch=trainer.start_epoch, restored_best_epoch=trainer.stopper.best_epoch,
                restored_best_fitness=trainer.best_fitness, scheduler_last_epoch=trainer.scheduler.last_epoch,
                optimizer_lrs=[group['lr'] for group in trainer.optimizer.param_groups])
        else:
            audit.update(initial_model_sha256=actual, common_state_sha256=state_hash(trainer.model.state_dict(), True))
        audit.update(actual_batch=trainer.batch_size, actual_args=vars(trainer.args),
                     train_images=len(trainer.train_loader.dataset), val_images=len(trainer.test_loader.dataset),
                     val_batch=trainer.test_loader.batch_size, batches_per_epoch=len(trainer.train_loader),
                     gpu_free_at_start_mib=torch.cuda.mem_get_info()[0] / 2**20)
        if (audit['train_images'], audit['val_images']) != ((16, 8) if smoke else (receipt['train_entries'], 173)):
            raise ValueError('Unexpected actual loader input counts')
        def stepped(_optimizer, _args, _kwargs):
            audit['actual_optimizer_steps'] += 1
        trainer.optimizer.register_step_post_hook(stepped)
        torch.cuda.reset_peak_memory_stats()
        write(run / 'training_audit.json', audit)

    batch_started = [None]
    val_started = [None]
    precision_handles = []
    validation_precision = [None]

    def precision_hook(record, require_bf16=False):
        handle = [None]
        def observed(module, inputs, output):
            if isinstance(module, torch.nn.Conv2d) and inputs[0].is_cuda:
                evidence = precision_evidence(inputs, output)
                record.update(evidence)
                handle[0].remove()
                if require_bf16 and (not evidence['autocast_enabled']
                        or evidence['autocast_dtype'] != 'torch.bfloat16'
                        or evidence['output_dtype'] != 'torch.bfloat16'):
                    raise RuntimeError('Requested BF16 training did not execute in BF16')
        # This hook is global and transient: it is never serialized into a model.
        handle[0] = torch.nn.modules.module.register_module_forward_hook(observed)
        precision_handles.append(handle[0])

    def on_epoch_start(trainer):
        torch.cuda.synchronize()
        # Each training epoch starts a fresh peak window after the prior val.
        torch.cuda.reset_peak_memory_stats()
        if not audit.get('training_precision'):
            audit['training_precision'] = {}
            precision_hook(audit['training_precision'], require_bf16=True)

    def on_batch_start(trainer):
        torch.cuda.synchronize(); batch_started[0] = time.perf_counter()

    def on_batch_end(trainer):
        torch.cuda.synchronize()
        audit['batch_seconds'].append(time.perf_counter() - batch_started[0])
        values = trainer.loss_items.values() if isinstance(trainer.loss_items, dict) else trainer.loss_items
        if not all(torch.isfinite(value).all().item() for value in values):
            raise RuntimeError('Non-finite training loss')
        audit['finite_loss_checks'] += 1
        audit['train_peak_allocated_mib'] = max(audit.get('train_peak_allocated_mib', 0), torch.cuda.max_memory_allocated() / 2**20)
        audit['train_peak_reserved_mib'] = max(audit.get('train_peak_reserved_mib', 0), torch.cuda.max_memory_reserved() / 2**20)

    def on_epoch_end(trainer):
        audit['completed_epochs'] = trainer.epoch + 1
        audit['epochs'].append(dict(displayed_epoch=trainer.epoch + 1,
            mosaic_state=mosaic_transform_state(trainer.train_loader.dataset.transforms),
            optimizer_steps=audit['actual_optimizer_steps']))
        write(run / 'training_audit.json', audit)

    def on_val_start(validator):
        torch.cuda.synchronize(); val_started[0] = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        validation_precision[0] = {}
        precision_hook(validation_precision[0])

    def on_val_end(validator):
        torch.cuda.synchronize()
        audit['validation_passes'].append(dict(seconds=time.perf_counter() - val_started[0],
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            dataset_images=len(validator.dataloader.dataset), batches=len(validator.dataloader),
            observed_precision=deepcopy(validation_precision[0]),
            displayed_epoch=trainer_reference[0].epoch + 1,
            phase='epoch_validation' if validator.training else 'final_best_validation'))
        write(run / 'training_audit.json', audit)

    def on_fit_epoch_end(trainer):
        record = audit['epochs'][-1]
        record.update(fitness=float(trainer.fitness), best_epoch=int(trainer.stopper.best_epoch),
                      early_stop_possible=bool(trainer.stopper.possible_stop), stop=bool(trainer.stop))
        losses = {k: float(v) for k, v in trainer.metrics.items() if k.startswith('val/')}
        if not all(math.isfinite(value) for value in losses.values()):
            raise RuntimeError('Non-finite validation loss')
        record['validation_losses_finite'] = True
        write(run / 'training_audit.json', audit)

    try:
        model = YOLO(resume_context['checkpoint'] if resume_context else spec['models'][arm]['checkpoint'])
        for event, callback in [('on_pretrain_routine_end', initialized), ('on_train_batch_start', on_batch_start),
                ('on_train_epoch_start', on_epoch_start),
                ('on_train_batch_end', on_batch_end), ('on_train_epoch_end', on_epoch_end),
                ('on_val_start', on_val_start), ('on_val_end', on_val_end), ('on_fit_epoch_end', on_fit_epoch_end)]:
            model.add_callback(event, callback)
        model.train(trainer=AuditedTrainer, **kwargs)
        trainer = model.trainer
        best = run / 'weights/best.pt'; last = run / 'weights/last.pt'
        if audit['actual_optimizer_steps'] == 0 or not best.exists() or not last.exists() or not audit['validation_passes']:
            raise RuntimeError('Missing completed optimization, validation or checkpoint')
        if audit['observed_oom_retries']:
            raise RuntimeError('OOM retry invalidated the fixed-batch smoke')
        audit.update(status='passed_smoke_not_quality_result' if smoke else 'completed_pilot',
            elapsed_seconds=(resume_context['prior_training_seconds'] if resume_context else 0) + time.perf_counter() - started,
            current_session_elapsed_seconds=time.perf_counter() - started,
            mean_batch_seconds=sum(audit['batch_seconds']) / len(audit['batch_seconds']),
            steady_mean_batch_seconds=(sum(audit['batch_seconds'][1:]) / len(audit['batch_seconds'][1:]))
                if len(audit['batch_seconds']) > 1 else None,
            best_epoch=int(trainer.stopper.best_epoch), early_stopped=audit['completed_epochs'] < kwargs['epochs'],
            stop_reason='early_stopping' if audit['completed_epochs'] < kwargs['epochs'] else 'max_epochs',
            timing_includes='CUDA synchronization, finite loss/gradient auditing; smoke additionally copies each image batch to CPU for paired hash checks',
            checkpoint_best_sha256=sha(best), checkpoint_last_sha256=sha(last))
        write(run / 'training_audit.json', audit)
        if not smoke:
            write(out / 'state.json', dict(status='pilot_arm_completed', formal_training_started=True,
                last_completed_arm=arm, last_completed_run=str(run)))
        print('SMOKE_PASSED' if smoke else 'PILOT_COMPLETED', arm, audit['actual_optimizer_steps'], audit['elapsed_seconds'], flush=True)
        del model; torch.cuda.empty_cache()
        return audit
    except Exception as error:
        audit.update(status='failed', error=repr(error),
            elapsed_seconds=(resume_context['prior_training_seconds'] if resume_context else 0) + time.perf_counter() - started,
            current_session_elapsed_seconds=time.perf_counter() - started)
        write(run / 'training_audit.json', audit)
        raise
    finally:
        for handle in precision_handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'smoke', 'train', 'resume'])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--batch', type=int, choices=[1, 2, 4, 8], default=2)
    parser.add_argument('--hold-ids', default='', help='Comma-separated source IDs; hold entire containing training images')
    parser.add_argument('--attempt', default='r1')
    args = parser.parse_args(); out = args.out.resolve()
    if args.mode == 'prepare':
        hold_ids = [int(value.strip()) for value in args.hold_ids.split(',') if value.strip()]
        prepare(out, hold_ids=hold_ids, planned_batch=args.batch)
    elif args.mode == 'smoke':
        arms = [args.arm] if args.arm else list(ARMS)
        rows = [execute(out, arm, smoke=True, batch=args.batch, attempt=args.attempt) for arm in arms]
        if len(rows) == 2 and rows[0]['checked_batch_image_sha256'] != rows[1]['checked_batch_image_sha256']:
            raise ValueError('Paired augmented smoke image streams differ')
        summary = dict(status='passed_smoke_not_quality_result', formal_training_started=False,
                       effective_code_sha256=sha(__file__), batch=args.batch, runs=rows,
                       paired_arms_complete=arms == list(ARMS))
        write(out / f'smoke_{args.attempt}_b{args.batch}.json', summary)
    else:
        if not args.arm:
            parser.error('train requires --arm; formal runs are explicitly selected')
        summary = read(out / f'smoke_{args.attempt}_b{args.batch}.json')
        if not summary.get('paired_arms_complete') or summary['status'] != 'passed_smoke_not_quality_result':
            raise ValueError('Both architectures must pass the same-batch smoke first')
        if args.batch != load_protocol(out)['training']['batch']:
            raise ValueError('Pilot batch must match the prepared protocol; use a new experiment version to change it')
        execute(out, args.arm, smoke=False, batch=args.batch, attempt=args.attempt, resume=args.mode == 'resume')


if __name__ == '__main__':
    main()
