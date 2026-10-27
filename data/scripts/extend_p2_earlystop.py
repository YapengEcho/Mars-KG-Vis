"""Continue an audited P2 checkpoint until patience expires, without an epoch cap.

The original 60-epoch LR schedule is preserved, then stays at its low LR floor.
Only a separately prepared experiment directory may use this entry point.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path

import probe_p2_detector as P


def lr_factor(epoch, horizon=60, floor=0.01):
    return max(1.0 - epoch / horizon, 0.0) * (1.0 - floor) + floor


def verify_output_directory(saved_args, run):
    if Path(saved_args.get('save_dir', '')).resolve() != Path(run).resolve():
        raise ValueError('Checkpoint save_dir must be the independent extension run')


def extension_trainer(base, expected_run=None):
    class UntilPatienceTrainer(base):
        def _setup_scheduler(self):
            from torch.optim.lr_scheduler import LambdaLR
            # Capture the original horizon; do not divide by the infinite cap.
            horizon, floor = int(self.args.epochs), float(self.args.lrf)
            if self.args.cos_lr:
                raise ValueError('This extension only preserves the original linear LR schedule')
            self.lf = lambda epoch: lr_factor(epoch, horizon, floor)
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lf)

        def resume_training(self, checkpoint):
            if not checkpoint or not self.resume:
                raise ValueError('Extension must restore a complete optimizer checkpoint')
            if expected_run is not None and Path(self.save_dir).resolve() != Path(expected_run).resolve():
                raise ValueError('SDK selected a different output directory')
            self.epochs = math.inf
            super().resume_training(checkpoint)
            # Infinite epochs would prevent the usual last-N mosaic close gate.
            self.epoch = self.start_epoch
            self._close_dataloader_mosaic()

    return UntilPatienceTrainer


def resume_context(run, expected_args):
    """The regular strict checks, except completed epochs may exceed the LR horizon."""
    import torch
    checkpoint_path, best_path = run / 'weights/last.pt', run / 'weights/best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    verify_output_directory(checkpoint['train_args'], run)
    best = torch.load(best_path, map_location='cpu', weights_only=False)
    completed = int(checkpoint.get('epoch', -1)) + 1
    if completed <= 0 or checkpoint.get('optimizer') is None:
        raise ValueError('Missing resumable optimizer state')
    for key, expected in expected_args.items():
        if key == 'resume':
            continue
        saved = checkpoint['train_args'].get(key)
        same = str(saved) == str(expected) if key == 'device' else saved == expected
        if key not in checkpoint['train_args'] or not same:
            raise ValueError('Extension argument differs: ' + key)
    ema = checkpoint.get('ema')
    if ema is None or checkpoint.get('updates') is None:
        raise ValueError('Missing EMA/update state')
    fitness, best_epoch = float(checkpoint['best_fitness']), int(best['epoch']) + 1
    if not math.isfinite(fitness) or float(best['best_fitness']) != fitness or not 0 < best_epoch <= completed:
        raise ValueError('Best checkpoint does not match saved early-stop state')
    patience = int(expected_args['patience'])
    if patience <= 0 or completed - best_epoch >= patience:
        raise ValueError('Preserve finite patience and do not restart a completed early stop')
    times = checkpoint.get('train_results', {}).get('time', [])
    elapsed = float(times[-1]) if times else 0.0
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Invalid checkpoint time')
    return dict(checkpoint=str(checkpoint_path), checkpoint_sha256=P.sha(checkpoint_path),
        best_checkpoint_sha256=P.sha(best_path), completed_epochs=completed, start_epoch=completed,
        displayed_resume_epoch=completed + 1, best_epoch=best_epoch, best_fitness=fitness,
        ema_updates=int(checkpoint['updates']), expected_student_sha256=P.state_hash(ema.float().state_dict()),
        expected_ema_sha256=P.state_hash(ema.state_dict()),
        expected_optimizer_sha256=P.nested_state_hash(checkpoint['optimizer']),
        saved_optimizer_lrs=[g['lr'] for g in checkpoint['optimizer']['param_groups']],
        prior_training_seconds=elapsed, scaler_state=checkpoint.get('scaler', {}),
        rng_state_saved=False, student_restoration='SDK restores EMA; raw student absent',
        limitation='Epoch-boundary exploratory continuation; not bitwise uninterrupted training')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(P.ROOT) or out == P.DEFAULT_OUT:
        raise ValueError('Use an independent extension directory in the project')
    settings = P.read(out / 'extension_settings.json')
    if settings['base_runner_sha256'] != P.sha(P.__file__):
        raise ValueError('Base audited runner changed')
    spec = P.load_protocol(out)
    if spec['training']['epochs'] != 60 or spec['training']['patience'] != 20:
        raise ValueError('Unexpected original horizon or patience')
    from recover_p2_pilot_pair import ensure_no_active_queue
    ensure_no_active_queue(out)
    import psutil
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            command = process.info['cmdline'] or []
            own = {os.getpid()} | {p.pid for p in psutil.Process().parents()}
            if process.pid not in own and any(Path(x).name == Path(__file__).name for x in command):
                if '--out' in command and Path(command[command.index('--out') + 1]).resolve() == out:
                    raise RuntimeError('Another extension process is active')
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    import ultralytics.models.yolo.detect.train as D
    D.DetectionTrainer = extension_trainer(D.DetectionTrainer, out / 'pilot/r2_b8/p2_seed0')
    P.checkpoint_resume_context = resume_context
    original_write = P.write

    def extension_write(path, value):
        if Path(path).name == 'training_audit.json':
            value.update(epoch_cap=None, requested_epochs=None, lr_schedule_horizon=60,
                         extension_code_sha256=P.sha(__file__),
                         formal_result='exploratory_extension_not_original_paired_protocol')
            if value.get('status') == 'completed_pilot':
                value.update(early_stopped=True, stop_reason='early_stopping')
        original_write(path, value)

    P.write = extension_write
    queue = dict(status='running', current_arm='p2', current_action='extend_until_early_stop',
                 parent_pid=os.getpid(), child_pid=os.getpid(), epoch_cap=None, patience=20,
                 experiment=str(out), source_experiment=settings['source_experiment'],
                 current_log=settings['current_log'], created_utc=datetime.now(timezone.utc).isoformat())
    original_write(out / 'queue.json', queue)
    try:
        audit = P.execute(out, 'p2', smoke=False, batch=8, attempt='r2', resume=True)
        queue.update(status='completed_training_validation_only', completed_epochs=audit['completed_epochs'],
                     best_epoch=audit['best_epoch'], stop_reason='early_stopping',
                     finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        queue.update(status='failed', error=repr(error), finished_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        original_write(out / 'queue.json', queue)


if __name__ == '__main__':
    main()
