"""Recover the paired detector queue after interruption, preserving completed arms."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'data/scripts/probe_p2_detector.py'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def arm_action(directory):
    audit = directory / 'training_audit.json'
    if audit.exists() and read(audit).get('status') == 'completed_pilot':
        if not (directory / 'weights/best.pt').exists():
            raise ValueError('Completed arm has no best checkpoint')
        return 'skip'
    if (directory / 'weights/last.pt').exists():
        return 'resume'
    if directory.exists():
        raise ValueError('Incomplete existing run has no usable checkpoint; preserve it for diagnosis')
    return 'train'


def ensure_no_active_queue(out):
    import psutil
    # Windows venv launchers are separate python.exe parents of the interpreter.
    own_launch = {os.getpid()} | {parent.pid for parent in psutil.Process().parents()}
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if process.pid in own_launch or not (process.info.get('name') or '').lower().startswith('python'):
                continue
            command = process.info.get('cmdline') or []
            if not any(Path(arg).name in {'probe_p2_detector.py', 'run_p2_pilot_pair.py',
                                         'recover_p2_pilot_pair.py'} for arg in command):
                continue
            if '--out' in command and Path(command[command.index('--out') + 1]).resolve() == out:
                raise RuntimeError(f'Another training process is active for this experiment: {process.pid}')
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--attempt', default='r2')
    parser.add_argument('--backup', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    backup = args.backup.resolve()
    if not out.is_relative_to(ROOT) or not backup.is_relative_to(out / 'recovery'):
        raise ValueError('Experiment and recovery backup must be inside the project')
    ensure_no_active_queue(out)
    from probe_p2_detector import load_protocol
    protocol = load_protocol(out)
    if protocol['training']['batch'] != args.batch:
        raise ValueError('Keep the frozen batch size on recovery')
    old_queue = read(backup / 'pilot' / f'{args.attempt}_b{args.batch}' / 'queue.json')
    old_source = backup / 'code/probe_p2_detector.py'
    if digest(old_source) != old_queue['runner_sha256']:
        raise ValueError('Archived original runner does not match the original queue')
    for record in read(backup / 'receipt.json')['files']:
        if digest(record['backup']) != record['sha256']:
            raise ValueError('Recovery backup has changed: ' + record['backup'])
    destination = out / 'pilot' / f'{args.attempt}_b{args.batch}'
    recovery = backup.parent
    receipt_path = recovery / 'queue.json'
    if receipt_path.exists():
        raise FileExistsError('Use a new recovery backup for another interruption')
    shutil.copy2(RUNNER, recovery / 'runner_snapshot.py')
    shutil.copy2(__file__, recovery / 'queue_runner_snapshot.py')
    code_sha = digest(RUNNER)
    state = dict(status='recovering', parent_pid=os.getpid(), created_utc=datetime.now(timezone.utc).isoformat(),
                 batch=args.batch, attempt=args.attempt, runner_sha256=code_sha,
                 original_runner_sha256=old_queue['runner_sha256'], backup=str(backup),
                 protocol_sha256=digest(out / 'protocol.json'), arms=['control', 'p2'],
                 completed=[], test_evaluation_started=False, recovery_receipt=str(receipt_path))

    def update():
        save(receipt_path, state)
        save(destination / 'queue.json', state)

    update()
    try:
        for arm in state['arms']:
            if digest(RUNNER) != code_sha:
                raise ValueError('Runner changed while recovery queue was active')
            directory = destination / f'{arm}_seed0'
            action = arm_action(directory)
            if action != 'skip':
                command = [sys.executable, '-u', str(RUNNER), action, '--out', str(out),
                           '--batch', str(args.batch), '--attempt', args.attempt, '--arm', arm]
                log_file = recovery / f'{arm}_{action}.log'
                with log_file.open('w', encoding='utf-8') as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                    state.update(status='running', current_arm=arm, current_action=action,
                                 child_pid=process.pid, current_log=str(log_file), command=command)
                    update()
                    exit_code = process.wait()
                if exit_code:
                    raise RuntimeError(f'{arm} {action} exited with code {exit_code}; inspect {log_file}')
            audit = read(directory / 'training_audit.json')
            if audit['status'] != 'completed_pilot':
                raise RuntimeError(f'{arm} did not complete training')
            state['completed'].append(dict(arm=arm, completed_epochs=audit['completed_epochs'],
                                           best_epoch=audit['best_epoch'], stop_reason=audit['stop_reason']))
            update()
        state.update(status='completed_training_validation_only', finished_utc=datetime.now(timezone.utc).isoformat())
        update()
    except Exception as error:
        state.update(status='failed', error=repr(error), finished_utc=datetime.now(timezone.utc).isoformat())
        update()
        raise


if __name__ == '__main__':
    main()
