"""Run the prepared control/P2 development comparison sequentially.

Each arm is a fresh process so GPU allocations cannot leak into the second arm.
The queue stops on any failure and never changes the fixed protocol or batch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'data/scripts/probe_p2_detector.py'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--attempt', default='r2')
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError('Output must be inside the project')
    smoke = json.loads((out / f'smoke_{args.attempt}_b{args.batch}.json').read_text(encoding='utf-8'))
    code_sha = digest(RUNNER)
    if smoke.get('status') != 'passed_smoke_not_quality_result' or not smoke.get('paired_arms_complete'):
        raise ValueError('Paired same-batch engineering check is required')
    if smoke['effective_code_sha256'] != code_sha:
        raise ValueError('Training runner changed since the engineering check')
    destination = out / 'pilot' / f'{args.attempt}_b{args.batch}'
    destination.mkdir(parents=True, exist_ok=True)
    queue_file = destination / 'queue.json'
    if queue_file.exists() or any((destination / f'{arm}_seed0').exists() for arm in ('control', 'p2')):
        raise FileExistsError('Preserve prior runs; this queue cannot be started twice')
    state = dict(status='starting', parent_pid=os.getpid(), created_utc=datetime.now(timezone.utc).isoformat(),
                 batch=args.batch, attempt=args.attempt, runner_sha256=code_sha,
                 protocol_sha256=digest(out / 'protocol.json'), smoke_sha256=digest(out / f'smoke_{args.attempt}_b{args.batch}.json'),
                 arms=['control', 'p2'], completed=[], test_evaluation_started=False)
    save(queue_file, state)
    try:
        for arm in state['arms']:
            if digest(RUNNER) != code_sha:
                raise ValueError('Runner changed while the queue was active')
            command = [sys.executable, '-u', str(RUNNER), 'train', '--out', str(out),
                       '--batch', str(args.batch), '--attempt', args.attempt, '--arm', arm]
            log_file = destination / f'{arm}_train.log'
            with log_file.open('w', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                state.update(status='running', current_arm=arm, child_pid=process.pid,
                             current_log=str(log_file), command=command)
                save(queue_file, state)
                exit_code = process.wait()
            if exit_code:
                raise RuntimeError(f'{arm} exited with code {exit_code}; inspect {log_file}')
            audit = json.loads((destination / f'{arm}_seed0/training_audit.json').read_text(encoding='utf-8'))
            if audit['status'] != 'completed_pilot':
                raise RuntimeError(f'{arm} did not produce a completed training audit')
            state['completed'].append(dict(arm=arm, completed_epochs=audit['completed_epochs'],
                                           best_epoch=audit['best_epoch'], stop_reason=audit['stop_reason']))
            save(queue_file, state)
        state.update(status='completed_training_validation_only', finished_utc=datetime.now(timezone.utc).isoformat())
        save(queue_file, state)
    except Exception as error:
        state.update(status='failed', error=repr(error), finished_utc=datetime.now(timezone.utc).isoformat())
        save(queue_file, state)
        raise


if __name__ == '__main__':
    main()
