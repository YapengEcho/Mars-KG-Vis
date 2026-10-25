"""Isolated, offline, full-epoch workers 0/2 engineering comparison.

The 60-epoch optimization schedule is preserved. Each arm is intentionally
stopped after two complete epochs; these checkpoints are not quality results.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
if 'scheduled-run' in sys.argv or 'arm' in sys.argv:
    sys.path.insert(0, str(ROOT / '.venv/Lib/site-packages'))

import probe_p2_detector as P
import run_full_pretrained_detector as F

SOURCE = ROOT / 'runs/experiments/full_pretrained_detector_20261014_v1'
OUT = ROOT / 'runs/experiments/loader_workers_profile_20261014_v1'
ARMS = (0, 2)
CELLS = ((0, 'on'), (2, 'on'), (0, 'off'), (2, 'off'))
STOP_AFTER_EPOCHS = 2

# Windows spawn re-imports this guarded module for loader workers. Explicitly
# reinstall the socket guard; monkeypatched Python sockets do not survive spawn.
if __name__ == '__mp_main__':
    _worker_blocked_network_attempts = F.prevent_network()


def utc():
    return datetime.now(timezone.utc).isoformat()


def prepare(out):
    """Copy only small labels/weights; hardlink images for private SDK caches."""
    import yaml
    if out.exists():
        raise FileExistsError('Preserve previous profile; choose a new output directory')
    P.receipt_check(SOURCE / 'protocol.json', (SOURCE / 'protocol.sha256').read_text().strip())
    source = P.read(SOURCE / 'protocol.json')
    P.receipt_check(SOURCE / 'source_receipt.json', source['source_receipt_sha256'])
    P.receipt_check(source['dataset'], source['dataset_sha256'])
    original = P.read(SOURCE / 'source_receipt.json')
    out.mkdir(parents=True)
    P.write(out / 'state.json', dict(status='preparing', gpu_work_started=False))
    rows = []
    source_root = Path(yaml.safe_load(Path(source['dataset']).read_text(encoding='utf-8'))['path'])
    image_map = {}
    for index, row in enumerate(original['copied_inputs']):
        before = Path(row['path'])
        P.receipt_check(before, row['sha256'])
        after = out / 'dataset' / before.relative_to(source_root)
        after.parent.mkdir(parents=True, exist_ok=True)
        if before.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            method = P.link_image(before, after)
            image_map[before.resolve()] = after.resolve()
        else:
            shutil.copy2(before, after)
            method = 'copy'
        rows.append(dict(source=str(before), path=str(after), sha256=row['sha256'], method=method))
        if index % 1000 == 0:
            print('INPUT_PREPARE', index, len(original['copied_inputs']), flush=True)
    original_data = yaml.safe_load(Path(source['dataset']).read_text(encoding='utf-8'))
    train_source = Path(original_data['train'])
    source_exposures = [Path(line).resolve() for line in train_source.read_text(encoding='utf-8').splitlines() if line]
    exposures = [image_map[p].as_posix() for p in source_exposures]
    if len(exposures) != original['train_entries'] or len(set(exposures)) != original['train_unique_images']:
        raise ValueError('Frozen exposure composition changed')
    train_list = out / 'dataset/train.txt'
    train_list.write_text('\n'.join(exposures) + '\n', encoding='utf-8')
    dataset = out / 'dataset/data.yaml'
    dataset.write_text(yaml.safe_dump(dict(path=(out / 'dataset').as_posix(),
        train=train_list.as_posix(), val='images/val', names={0: 'LANDSLIDE'}), sort_keys=False), encoding='utf-8')
    init = source['models']['control']
    checkpoint = out / 'inputs/full_pretrained_seed0.pt'
    checkpoint.parent.mkdir()
    P.receipt_check(init['checkpoint'], init['checkpoint_sha256'])
    shutil.copy2(init['checkpoint'], checkpoint)
    P.receipt_check(checkpoint, init['checkpoint_sha256'])
    config = out / 'inputs/yolov8n.yaml'
    shutil.copy2(init['config'], config)
    P.receipt_check(config, init['config_sha256'])
    receipt = dict(source_receipt_sha256=source['source_receipt_sha256'],
        train_entries=original['train_entries'], train_unique_images=original['train_unique_images'],
        val_images=original['validation_checked_images'], test_images_read=0,
        original_train_list=str(train_source), original_train_list_sha256=P.sha(train_source),
        train_list=str(train_list), train_list_sha256=P.sha(train_list), copied_inputs=rows)
    P.write(out / 'input_receipt.json', receipt)
    protocol = dict(created_utc=utc(), status='prepared_not_started', engineering_only=True,
        source=str(SOURCE), source_protocol_sha256=P.sha(SOURCE / 'protocol.json'),
        code_sha256=P.sha(__file__), production_audit_code_sha256=P.sha(P.__file__),
        offline_helper_code_sha256=P.sha(F.__file__), environment=source['environment'],
        workers=list(ARMS), cells=[dict(workers=w, mosaic=m) for w, m in CELLS],
        source_training=source['training'],
        dataset=str(dataset), dataset_sha256=P.sha(dataset), receipt_sha256=P.sha(out / 'input_receipt.json'),
        checkpoint=str(checkpoint), checkpoint_sha256=P.sha(checkpoint),
        initial_state_sha256=init['full_state_sha256'], config=str(config), config_sha256=P.sha(config),
        stop_after_epochs=STOP_AFTER_EPOCHS, startup_warmup_epochs=[1], measured_epochs=[2],
        private_dataset_cache=True, cache=False,
        production_recipe_changes=['output paths', 'workers argument per cell',
            'mosaic=0 only in explicit Mosaic-off cells', 'intentional stop after 2 full epochs'],
        augmentation_equivalence=False,
        measurement_limits=[
            'Epoch 2 is steady startup but still within the unchanged 3-epoch SGD optimization warmup.',
            'Mosaic on/off is an explicit engineering phase factor; compare workers within the same phase.',
            'Off cells start from the same initial weights, rather than resume epoch 11; optimization state is not a late-stage replay.',
            'The worker argument also changes SDK validation workers: requested train workers 2 means requested val workers 4.',
            'Timing includes production CUDA synchronizations and finite-gradient/loss checks, plus equal loader timers/resource sampling.',
            'Direct loader-next time measures consumer blocking after prefetch, not total CPU augmentation effort.',
            'One measured epoch per cell, fixed 0 then 2 order within each phase; OS cache/thermal noise remains. Reverse-order repeats needed for a close result.',
            'Same input hashes and training recipe do not imply identical per-batch augmentation RNG or Mosaic buffer state.',
            'Short-run metrics and resulting weights must not be used to rank model quality.'])
    P.write(out / 'protocol.json', protocol)
    (out / 'protocol.sha256').write_text(P.sha(out / 'protocol.json') + '\n', encoding='utf-8')
    (out / 'README.md').write_text('''# Offline loader workers engineering profile

Prepared only. The coordinator runs four fresh Python processes in this order:
workers=0 / Mosaic on, workers=2 / Mosaic on, workers=0 / Mosaic off,
workers=2 / Mosaic off. Each cell uses the same complete frozen training exposure list
(4860 entries, 2371 unique images), 173 validation images and identical full
pretrained YOLOv8n initialization. Images are read-only hardlinks and labels are
copies under this directory, so SDK caches never touch the source dataset.

Both arms retain batch=8, imgsz=1024, BF16, SGD, cache=false, seed=0,
epochs=60, patience=20, warmup_epochs=3 and close_mosaic=50. A callback stops each
after epoch 2 without compressing the scheduler. Epoch 1 is startup warmup;
epoch 2 is the measured complete epoch, still in the unchanged optimizer warmup.
Mosaic is explicitly enabled or disabled as the engineering phase factor;
only compare workers within the same Mosaic phase. The off cells measure the
majority-of-production condition but start from initial weights and optimizer,
so do not replay the actual epoch-11+ training state. SDK validation workers are 2 times the training worker
argument (0 vs 4); actual values are recorded. This is a speed/stability profile,
not a new model-quality experiment.

The production finite loss/gradient checks and CUDA synchronization callbacks
are retained. Equal additional instrumentation times DataLoader next() calls,
preprocessing, validation, checkpoint saving and epoch totals; samples parent /
child RAM, process/system CPU, GPU utilization/VRAM/power/temperature/clocks.
The timed batch callback interval includes preprocessing/transfer and audits;
it is not pure GPU computation. Loader wait is consumer blocking, not total
worker augmentation effort. Startup and final validation are reported separately.
Best/last checkpoint SHA256 receipts are retained (no extra epoch snapshots).

Run prepare with the project venv. For an independent run use Windows Task
Scheduler with the base Python interpreter and mode scheduled-run. It explicitly
checks the scheduler process parent, does not need Codex/browser/network, blocks
network socket connections in the coordinator, each arm and spawned loader
workers, and temporarily prevents automatic sleep. Closing Codex, screen-off,
lock-screen and loss of Internet do not stop it. Shutdown/restart still stops it.

No automatic resume: a failed/partial attempt is retained, never silently reused.
A failure, changed batch, OOM retry, NaN recovery, incorrect worker count or
incomplete epoch invalidates that arm. Review comparison.json before choosing
workers for future experiments. One measured epoch and fixed 0 then 2 order are
only initial engineering evidence; repeat in reverse order if the result is close.
''', encoding='utf-8')
    P.write(out / 'state.json', dict(status='prepared_not_started', gpu_work_started=False))
    print('PREPARED', out, flush=True)


def check(out, check_inputs=True):
    P.receipt_check(out / 'protocol.json', (out / 'protocol.sha256').read_text().strip())
    spec = P.read(out / 'protocol.json')
    for path, expected in [(Path(__file__), spec['code_sha256']), (P.__file__, spec['production_audit_code_sha256']),
            (F.__file__, spec['offline_helper_code_sha256']), (spec['checkpoint'], spec['checkpoint_sha256']),
            (spec['config'], spec['config_sha256']), (spec['dataset'], spec['dataset_sha256']),
            (out / 'input_receipt.json', spec['receipt_sha256'])]:
        P.receipt_check(path, expected)
    receipt = P.read(out / 'input_receipt.json')
    P.receipt_check(receipt['train_list'], receipt['train_list_sha256'])
    if check_inputs:
        for row in receipt['copied_inputs']:
            P.receipt_check(row['path'], row['sha256'])
    return spec, receipt


class TimedLoader:
    """Transparent wrapper; no sample/RNG/buffer changes, only wall-clock reads."""
    def __init__(self, loader, waits):
        self.loader, self.waits = loader, waits

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, key):
        return getattr(self.loader, key)

    def __iter__(self):
        iterator = iter(self.loader)
        for _ in range(len(self.loader)):
            started = time.perf_counter()
            batch = next(iterator)
            self.waits.append(time.perf_counter() - started)
            yield batch


class ResourceSampler:
    def __init__(self):
        import psutil
        self.psutil = psutil
        self.process = psutil.Process()
        self.samples = []
        self.stop = threading.Event()
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        known = {}
        while not self.stop.is_set():
            sample = dict(seconds=time.perf_counter() - self.started,
                          system_cpu_percent=self.psutil.cpu_percent(),
                          system_available_ram_mib=self.psutil.virtual_memory().available / 2**20)
            parent_rss, child_rss, cpu_seconds = 0, 0, 0
            for proc in [self.process] + self.process.children(recursive=True):
                try:
                    memory = proc.memory_info().rss / 2**20
                    parent_rss += memory if proc.pid == self.process.pid else 0
                    child_rss += memory if proc.pid != self.process.pid else 0
                    times = proc.cpu_times()
                    cpu_seconds += times.user + times.system
                    known[proc.pid] = dict(pid=proc.pid, rss_mib=memory,
                        cpu_seconds=times.user + times.system, executable=proc.name())
                except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                    pass
            sample.update(parent_rss_mib=parent_rss, children_rss_mib=child_rss,
                          live_process_cpu_seconds=cpu_seconds, last_seen_processes=list(known.values()))
            try:
                value = subprocess.run(['nvidia-smi',
                    '--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if value.returncode == 0:
                    fields = value.stdout.splitlines()[0].split(',')
                    sample['gpu'] = dict(zip(['utilization_percent', 'memory_used_mib', 'power_watts',
                                              'temperature_celsius', 'sm_clock_mhz'],
                        [float(x.strip()) if x.strip().replace('.', '', 1).isdigit() else x.strip() for x in fields]))
                else:
                    sample['gpu_query_error'] = value.stderr.strip()[:300]
            except (OSError, subprocess.TimeoutExpired) as error:
                sample['gpu_query_error'] = repr(error)
            self.samples.append(sample)
            self.stop.wait(2)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        return self.samples


def run_arm(out, workers, mosaic):
    attempts = F.prevent_network()
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import init_seeds
    spec, receipt = check(out)
    if P.environment() != spec['environment']:
        raise ValueError('Prepared production runtime environment changed')
    run = out / 'arms' / f'mosaic_{mosaic}_workers{workers}'
    run.mkdir(parents=True, exist_ok=False)
    kwargs = deepcopy(spec['source_training'])
    kwargs.update(workers=workers, seed=0, data=spec['dataset'], project=str(run.parent),
                  name=run.name, exist_ok=True)
    if mosaic == 'off':
        kwargs['mosaic'] = 0.0
    audit = dict(status='running', engineering_only=True, quality_result=False, requested_workers=workers,
        mosaic_phase=mosaic, requested_stop_epochs=STOP_AFTER_EPOCHS, requested_args=kwargs, started_utc=utc(),
        pid=os.getpid(), initial_checkpoint_sha256=spec['checkpoint_sha256'], initial_state_sha256=spec['initial_state_sha256'],
        completed_epochs=0, actual_optimizer_steps=0, finite_gradient_checks=0,
        finite_loss_checks=0, observed_oom_retries=0, epochs=[], batch_seconds=[],
        fetch_seconds=[], preprocess_seconds=[], save_seconds=[], validations=[])
    P.write(run / 'profile.json', audit)
    state = dict(epoch_started=None, batch_started=None, val_started=None,
                 current=None, trainer=None, validation_precision=None)
    precision_handles = []

    class ProfileTrainer(DetectionTrainer):
        def get_model(self, *args, **kw):
            model = super().get_model(*args, **kw)
            init_seeds(0, deterministic=True)
            return model

        def _build_train_pipeline(self):
            if self.batch_size != 8 or getattr(self, '_oom_retries', 0):
                audit['observed_oom_retries'] = getattr(self, '_oom_retries', 0)
                raise RuntimeError('OOM/batch retry invalidates the fixed-batch engineering profile')
            return super()._build_train_pipeline()

        def preprocess_batch(self, value):
            before = time.perf_counter()
            value = super().preprocess_batch(value)
            audit['preprocess_seconds'].append(time.perf_counter() - before)
            if 'first_batch_image_sha256' not in audit:
                audit['first_batch_image_sha256'] = P.state_hash({'image': value['img']})
                audit['first_batch_image_files'] = value['im_file']
            return value

        def optimizer_step(self):
            nonzero = False
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all().item():
                        raise RuntimeError('Non-finite gradient')
                    nonzero |= bool(parameter.grad.any().item())
            audit['finite_gradient_checks'] += 1
            if not nonzero:
                audit['zero_gradient_steps'] = audit.get('zero_gradient_steps', 0) + 1
            return super().optimizer_step()

        def save_model(self):
            before = time.perf_counter()
            value = super().save_model()
            audit['save_seconds'].append(time.perf_counter() - before)
            return value

        def _handle_nan_recovery(self, epoch):
            if (self.loss is not None and not self.loss.isfinite()) or \
                    (self.fitness is not None and not __import__('math').isfinite(float(self.fitness))):
                audit['observed_nan_recovery_requests'] = audit.get('observed_nan_recovery_requests', 0) + 1
                raise RuntimeError('Non-finite optimization/validation invalidates engineering profile')
            return super()._handle_nan_recovery(epoch)

    def precision_hook(record, bf16=False):
        handle = [None]
        def observed(module, inputs, output):
            if isinstance(module, torch.nn.Conv2d) and inputs[0].is_cuda:
                record.update(P.precision_evidence(inputs, output))
                handle[0].remove()
                if bf16 and (not record['autocast_enabled'] or record['output_dtype'] != 'torch.bfloat16'):
                    raise RuntimeError('Training did not execute BF16')
        handle[0] = torch.nn.modules.module.register_module_forward_hook(observed)
        precision_handles.append(handle[0])

    def initialized(trainer):
        state['trainer'] = trainer
        initial = P.state_hash(trainer.model.state_dict())
        if initial != spec['initial_state_sha256']:
            raise ValueError('Initialization tensors changed')
        audit.update(actual_initial_state_sha256=initial, actual_args=vars(trainer.args),
            actual_batch=trainer.batch_size, train_entries=len(trainer.train_loader.dataset),
            val_images=len(trainer.test_loader.dataset), batches_per_epoch=len(trainer.train_loader),
            actual_train_workers=trainer.train_loader.num_workers,
            actual_val_workers=trainer.test_loader.num_workers,
            train_pin_memory=trainer.train_loader.pin_memory, val_pin_memory=trainer.test_loader.pin_memory,
            train_prefetch_factor=trainer.train_loader.prefetch_factor,
            val_prefetch_factor=trainer.test_loader.prefetch_factor)
        if (audit['train_entries'], audit['val_images']) != (receipt['train_entries'], receipt['val_images']):
            raise ValueError('Loader dataset composition changed')
        if audit['actual_train_workers'] != workers or audit['actual_batch'] != 8:
            raise ValueError('SDK did not use the requested workers/batch')
        def stepped(_optimizer, _args, _kwargs):
            audit['actual_optimizer_steps'] += 1
        trainer.optimizer.register_step_post_hook(stepped)
        trainer.train_loader = TimedLoader(trainer.train_loader, audit['fetch_seconds'])
        audit['setup_seconds'] = time.perf_counter() - started
        P.write(run / 'profile.json', audit)

    def epoch_start(trainer):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        state['epoch_started'] = time.perf_counter()
        state['current'] = dict(displayed_epoch=trainer.epoch + 1,
            timing_role='startup_warmup' if trainer.epoch == 0 else 'measured_full_epoch',
            slice_start=len(audit['batch_seconds']), fetch_slice_start=len(audit['fetch_seconds']),
            preprocess_slice_start=len(audit['preprocess_seconds']), save_slice_start=len(audit['save_seconds']),
            mosaic=P.mosaic_transform_state(trainer.train_loader.dataset.transforms))
        if 'training_precision' not in audit:
            audit['training_precision'] = {}
            precision_hook(audit['training_precision'], bf16=True)

    def batch_start(trainer):
        torch.cuda.synchronize()
        state['batch_started'] = time.perf_counter()

    def batch_end(trainer):
        torch.cuda.synchronize()
        audit['batch_seconds'].append(time.perf_counter() - state['batch_started'])
        values = trainer.loss_items.values() if isinstance(trainer.loss_items, dict) else trainer.loss_items
        if not all(torch.isfinite(value).all().item() for value in values):
            raise RuntimeError('Non-finite training loss')
        audit['finite_loss_checks'] += 1

    def epoch_end(trainer):
        torch.cuda.synchronize()
        state['current'].update(train_seconds=time.perf_counter() - state['epoch_started'],
            train_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            train_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)

    def val_start(validator):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        state['val_started'] = time.perf_counter()
        state['validation_precision'] = {}
        precision_hook(state['validation_precision'])

    def val_end(validator):
        torch.cuda.synchronize()
        record = dict(seconds=time.perf_counter() - state['val_started'],
            phase='epoch_validation' if validator.training else 'final_best_validation',
            displayed_epoch=state['trainer'].epoch + 1, actual_workers=validator.dataloader.num_workers,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            precision=state['validation_precision'])
        audit['validations'].append(record)

    def fit_end(trainer):
        current = state['current']
        # SDK invokes this callback again after final best validation.
        if current is None or current['displayed_epoch'] != trainer.epoch + 1:
            return
        torch.cuda.synchronize()
        current.update(epoch_total_seconds=time.perf_counter() - state['epoch_started'],
            production_sdk_epoch_seconds=float(trainer.epoch_time),
            batch_callback_seconds=sum(audit['batch_seconds'][current['slice_start']:]),
            fetch_wait_seconds=sum(audit['fetch_seconds'][current['fetch_slice_start']:]),
            preprocess_transfer_seconds=sum(audit['preprocess_seconds'][current['preprocess_slice_start']:]),
            checkpoint_save_seconds=sum(audit['save_seconds'][current['save_slice_start']:]),
            batch_count=len(audit['batch_seconds']) - current['slice_start'],
            fetched_batches=len(audit['fetch_seconds']) - current['fetch_slice_start'],
            optimizer_steps=audit['actual_optimizer_steps'])
        current['validation_seconds'] = sum(v['seconds'] for v in audit['validations']
            if v['phase'] == 'epoch_validation' and v['displayed_epoch'] == trainer.epoch + 1)
        current['other_seconds'] = current['epoch_total_seconds'] - current['batch_callback_seconds'] \
            - current['fetch_wait_seconds'] - current['validation_seconds'] - current['checkpoint_save_seconds']
        if current['batch_count'] != audit['batches_per_epoch'] or current['fetched_batches'] != audit['batches_per_epoch']:
            raise RuntimeError('Partial epoch invalidates profile')
        if current['mosaic']['active'] != (mosaic == 'on'):
            raise RuntimeError('Profile unexpectedly changed Mosaic phase')
        if getattr(trainer, 'nan_recovery_attempts', 0) or getattr(trainer, '_oom_retries', 0):
            raise RuntimeError('SDK recovery invalidates profile')
        audit['epochs'].append(current)
        audit['completed_epochs'] = trainer.epoch + 1
        if audit['completed_epochs'] >= STOP_AFTER_EPOCHS:
            trainer.stop = True
        P.write(run / 'profile.json', audit)
        state['current'] = None

    sampler = ResourceSampler()
    started = time.perf_counter()
    sampler.thread.start()
    model = None
    try:
        model = YOLO(spec['checkpoint'])
        for event, callback in [('on_pretrain_routine_end', initialized), ('on_train_epoch_start', epoch_start),
                ('on_train_batch_start', batch_start), ('on_train_batch_end', batch_end),
                ('on_train_epoch_end', epoch_end), ('on_val_start', val_start),
                ('on_val_end', val_end), ('on_fit_epoch_end', fit_end)]:
            model.add_callback(event, callback)
        model.train(trainer=ProfileTrainer, **kwargs)
        if audit['completed_epochs'] != STOP_AFTER_EPOCHS or audit['actual_optimizer_steps'] <= 0:
            raise RuntimeError('Engineering profile did not complete expected epochs')
        audit.update(status='completed_engineering_only', stop_reason='explicit_profile_bound_after_2_complete_epochs',
            elapsed_seconds=time.perf_counter() - started,
            checkpoint_best_sha256=P.sha(run / 'weights/best.pt'), checkpoint_last_sha256=P.sha(run / 'weights/last.pt'))
    except BaseException as error:
        audit.update(status='failed_invalid_profile', error=repr(error), elapsed_seconds=time.perf_counter() - started)
        raise
    finally:
        audit.update(finished_utc=utc(), blocked_network_attempts=attempts)
        samples = sampler.close()
        P.write(run / 'resource_samples.json', samples)
        if samples:
            audit['resources'] = dict(max_parent_rss_mib=max(x['parent_rss_mib'] for x in samples),
                max_children_rss_mib=max(x['children_rss_mib'] for x in samples),
                min_available_system_ram_mib=min(x['system_available_ram_mib'] for x in samples),
                mean_system_cpu_percent=statistics.mean(x['system_cpu_percent'] for x in samples),
                resource_samples=len(samples))
            numeric_gpu_samples = [x['gpu'] for x in samples if 'gpu' in x]
            for key in ['utilization_percent', 'memory_used_mib', 'power_watts', 'temperature_celsius', 'sm_clock_mhz']:
                values = [x[key] for x in numeric_gpu_samples if isinstance(x.get(key), (int, float))]
                if values:
                    audit['resources'][f'mean_gpu_{key}'] = statistics.mean(values)
                    audit['resources'][f'max_gpu_{key}'] = max(values)
        P.write(run / 'profile.json', audit)
        for handle in precision_handles:
            handle.remove()
        if model is not None:
            for loader in [getattr(model.trainer, 'train_loader', None), getattr(model.trainer, 'test_loader', None)]:
                if loader is not None and hasattr(loader, 'close'):
                    loader.close()
    return audit


def coordinator(out):
    attempts = F.prevent_network()
    spec, _ = check(out, check_inputs=False)
    F.ensure_idle(out)
    independence = F.process_independence()
    if not independence['scheduler_owned']:
        raise RuntimeError('Coordinator requires a Windows Task Scheduler-owned process')
    if (out / 'queue.json').exists():
        raise FileExistsError('Preserve prior launch; use a new output directory')
    kernel = ctypes.windll.kernel32
    if not kernel.SetThreadExecutionState(0x80000001):
        raise ctypes.WinError()
    queue = dict(status='running', started_utc=utc(), pid=os.getpid(), current_workers=None,
        process_independence=independence, offline=True, temporary_system_awake=True,
        engineering_only=True, completed_workers=[])
    P.write(out / 'queue.json', queue)
    try:
        for cell in spec['cells']:
            workers, mosaic = cell['workers'], cell['mosaic']
            queue['current_workers'] = workers
            queue['current_mosaic'] = mosaic
            name = f'mosaic_{mosaic}_workers{workers}'
            log = out / f'{name}.log'
            err = out / f'{name}.err.log'
            package_env = os.environ.copy()
            package_env.update(PYTHONPATH=os.pathsep.join([str(ROOT / '.venv/Lib/site-packages'),
                                                         str(ROOT / 'data/scripts')]), PYTHONUTF8='1')
            with log.open('xb') as stdout, err.open('xb') as stderr:
                proc = subprocess.Popen([getattr(sys, '_base_executable', sys.executable), '-u',
                    str(Path(__file__).resolve()), 'arm', '--out', str(out), '--workers', str(workers), '--mosaic', mosaic],
                    cwd=ROOT, env=package_env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), close_fds=True)
                queue['child_pid'] = proc.pid
                P.write(out / 'queue.json', queue)
                code = proc.wait()
            profile = out / 'arms' / name / 'profile.json'
            if code or not profile.exists() or P.read(profile)['status'] != 'completed_engineering_only':
                raise RuntimeError(f'{name} profile failed, process exit={code}; see retained logs')
            queue['completed_workers'].append(cell)
            P.write(out / 'queue.json', queue)
        arms = [P.read(out / 'arms' / f"mosaic_{c['mosaic']}_workers{c['workers']}" / 'profile.json') for c in spec['cells']]
        measured = [a['epochs'][1] for a in arms]
        comparison = dict(status='completed_engineering_only', quality_result=False,
            protocol_sha256=P.sha(out / 'protocol.json'), measured_epoch=2, cells=spec['cells'],
            measured_epochs=measured,
            phase_comparisons=[dict(mosaic_phase=mosaic,
                epoch_speed_ratio_workers0_over_workers2=measured[i]['epoch_total_seconds'] / measured[i+1]['epoch_total_seconds'],
                train_speed_ratio_workers0_over_workers2=measured[i]['train_seconds'] / measured[i+1]['train_seconds'])
                for i, mosaic in [(0, 'on'), (2, 'off')]],
            profiles=[str(out / 'arms' / f"mosaic_{c['mosaic']}_workers{c['workers']}" / 'profile.json') for c in spec['cells']],
            initialization_equal=len({a['actual_initial_state_sha256'] for a in arms}) == 1,
            resources=[a['resources'] for a in arms], limitations=spec['measurement_limits'])
        P.write(out / 'comparison.json', comparison)
        queue.update(status='completed_engineering_only', current_workers=None, current_mosaic=None)
    except BaseException as error:
        queue.update(status='failed', error=repr(error))
        raise
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        queue.update(finished_utc=utc(), temporary_system_awake=False, blocked_network_attempts=attempts)
        P.write(out / 'queue.json', queue)
        P.write(out / 'state.json', dict(status=queue['status'], gpu_work_started=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'scheduled-run', 'arm'])
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, choices=ARMS)
    parser.add_argument('--mosaic', choices=['on', 'off'], default='on')
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / 'runs/experiments'):
        raise ValueError('New profile directory must be inside runs/experiments')
    if args.mode == 'prepare':
        prepare(out)
    elif args.mode == 'scheduled-run':
        sys.stdout = (out / 'coordinator.log').open('x', encoding='utf-8', buffering=1)
        sys.stderr = (out / 'coordinator.err.log').open('x', encoding='utf-8', buffering=1)
        coordinator(out)
    else:
        if args.workers is None:
            parser.error('arm requires --workers')
        run_arm(out, args.workers, args.mosaic)


if __name__ == '__main__':
    main()
