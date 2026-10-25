"""Local, detached and network-disabled full-pretraining YOLO control experiment."""
from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

if len(sys.argv) > 1 and sys.argv[1] in ('scheduled-run', 'scheduled-resume'):
    # Task Scheduler launches the base interpreter directly, without the venv
    # redirector or a Codex process ancestor. Use the existing local packages.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '.venv/Lib/site-packages'))

import probe_p2_detector as P

SOURCE = P.ROOT / 'runs/experiments/p2_detector_probe_20261013_v2'
OUT = P.ROOT / 'runs/experiments/full_pretrained_detector_20261014_v1'


def transfer_compatible(source, target):
    """Transfer every shape-compatible COCO tensor, including neck and box head."""
    selected = {k: v.detach().clone() for k, v in source.items()
                if k in target and target[k].shape == v.shape}
    if not selected or not any(int(k.split('.')[1]) >= 16 for k in selected if k.startswith('model.')):
        raise ValueError('Full pretraining must include compatible neck/head tensors')
    with __import__('torch').no_grad():
        for key, value in selected.items():
            target[key].copy_(value.to(target[key].device, target[key].dtype))
    return selected


def prevent_network():
    os.environ.update(YOLO_OFFLINE='true', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      WANDB_MODE='disabled', COMET_MODE='DISABLED')
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append(repr(args[1:])[:200])
        raise OSError('Network disabled for local training')

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    # YOLO_OFFLINE avoids startup DNS probes. Disable optional integrations
    # in this process without writing the user's persistent global settings.
    from ultralytics.utils import SETTINGS
    dict.update(SETTINGS, {key: False for key in
                ('sync', 'wandb', 'comet', 'clearml', 'mlflow', 'neptune', 'raytune', 'tensorboard')})
    return attempts


def process_independence():
    import psutil
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    in_job = ctypes.c_int()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(in_job)):
        raise ctypes.WinError()
    parent = psutil.Process().parent()
    system_dir = Path(os.environ.get('SystemRoot', r'C:\Windows')).resolve()
    parent_exe = Path(parent.exe()).resolve() if parent else None
    scheduler_owned = bool(parent_exe and parent_exe.is_relative_to(system_dir) and
                           parent_exe.name.lower() in ('svchost.exe', 'taskeng.exe', 'taskhostw.exe'))
    return dict(pid=os.getpid(), parent_pid=os.getppid(), in_windows_job=bool(in_job.value),
                has_console=bool(kernel.GetConsoleWindow()), parent_executable=str(parent_exe),
                scheduler_owned=scheduler_owned)


def prepare(out):
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    if out.exists():
        raise FileExistsError('Keep old experiment intact')
    original = P.load_protocol(SOURCE)
    P.receipt_check(original['source_checkpoint'], original['source_checkpoint_sha256'])
    out.mkdir(parents=True)
    for name in ('source_receipt.json', 'held_cases.json'):
        shutil.copy2(SOURCE / name, out / name)
    config = out / 'configs/yolov8n.yaml'
    config.parent.mkdir()
    shutil.copy2(original['models']['control']['config'], config)
    init_seeds(0, deterministic=True)
    model = YOLO(str(config)).model.cpu().float()
    model.names = {0: 'LANDSLIDE'}
    checkpoint = torch.load(original['source_checkpoint'], map_location='cpu', weights_only=False)
    source = (checkpoint.get('ema') or checkpoint['model']).float().state_dict()
    copied = transfer_compatible(source, model.state_dict())
    if P.state_hash(model.state_dict(), True) != original['models']['control']['common_state_sha256']:
        raise ValueError('Common-layer initialization changed from paired baseline')
    weights = out / 'initial_weights/control_seed0.pt'
    weights.parent.mkdir()
    torch.save(dict(model=deepcopy(model).float(), ema=None,
                    train_args={'task': 'detect', 'model': str(config)},
                    date=datetime.now(timezone.utc).isoformat()), weights)
    entry = dict(config=str(config), config_sha256=P.sha(config), checkpoint=str(weights),
                 checkpoint_sha256=P.sha(weights), full_state_sha256=P.state_hash(model.state_dict()),
                 common_state_sha256=P.state_hash(model.state_dict(), True),
                 parameters=sum(x.numel() for x in model.parameters()), strides=model.stride.tolist(),
                 copied_tensors=[dict(name=k, shape=list(v.shape), elements=v.numel()) for k, v in sorted(copied.items())],
                 fresh_tensor_names=[k for k in model.state_dict() if k not in copied])
    spec = deepcopy(original)
    spec.update(created_utc=datetime.now(timezone.utc).isoformat(), source_paired_protocol_sha256=P.sha(SOURCE / 'protocol.json'),
                preparation_code_sha256=P.sha(__file__), models={'control': entry},
                source_receipt_sha256=P.sha(out / 'source_receipt.json'), held_cases_sha256=P.sha(out / 'held_cases.json'))
    spec['training']['project'] = str(out / 'pilot')
    spec['policies'].update(initialization='All compatible COCO tensors, including neck/regression head; incompatible class tensors fresh',
                           fairness='Same frozen data, batch8, workers0, BF16, optimizer, seed0, 60/patience20, mosaic close11; initialization alone differs',
                           offline='Every socket connection blocked; all inputs local; optional online integrations disabled',
                           independence='Detached process outside any Windows job; no Codex/browser/network runtime dependency')
    P.write(out / 'protocol.json', spec)
    (out / 'protocol.sha256').write_text(P.sha(out / 'protocol.json') + '\n', encoding='utf-8')
    P.write(out / 'state.json', dict(status='prepared', formal_training_started=False))
    (out / 'README.md').write_text('''# 完整预训练普通YOLO对照\n\n与本轮普通版保持同一图像、标签、暂扣、划分、增强、batch8、workers0、1024输入、SGD、BF16和seed0；只改初始化，加载全部尺寸相容的COCO权重（含neck和回归分支），单类不相容分类张量重新初始化。\n\n上限60轮、20轮无改善早停，最佳轮按验证mAP50–95选择。与P2探索续训分开保存。冻结数据通过原收据校验，共2371张独立训练图、4860次出现，173张验证图；不读取测试图。\n\n训练进程独立于客户端，使用DETACHED_PROCESS及BREAKAWAY启动，启动时确认不在Windows进程作业内；禁止网络socket连接，全部权重及数据本地读取。运行期间临时防止系统自动睡眠，完成/退出后解除，不改永久电源设置。允许屏幕熄灭；关机、重启或强制休眠仍会中断。\n\n日志为training.log，状态为queue.json，结果位于pilot/full_b8/control_seed0。每轮保存last.pt和best.pt。关闭客户端不会自动恢复关机后的训练，恢复须执行此脚本resume。\n''', encoding='utf-8')
    return spec


def ensure_idle(out):
    import psutil
    own = {os.getpid()} | {x.pid for x in psutil.Process().parents()}
    scripts = {'extend_p2_earlystop.py', 'probe_p2_detector.py', 'recover_p2_pilot_pair.py',
               'run_p2_pilot_pair.py', Path(__file__).name}
    for proc in psutil.process_iter(['cmdline']):
        try:
            cmd = proc.info['cmdline'] or []
            if proc.pid not in own and any(Path(x).name in scripts for x in cmd):
                raise RuntimeError(f'Another training process is active: {proc.pid}')
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def run(out, smoke=False, resume=False, scheduled=False):
    ensure_idle(out)
    attempts = prevent_network()
    independence = process_independence()
    detached = not independence['in_windows_job'] and not independence['has_console']
    if not smoke and not (detached or scheduled and independence['scheduler_owned']):
        raise RuntimeError('Formal training requires an independent detached process')
    if not smoke:
        receipt = P.read(out / 'smoke/offline_b8/control_seed0/training_audit.json')
        if receipt['status'] != 'passed_smoke_not_quality_result' or receipt['observed_oom_retries']:
            raise ValueError('Require successful offline GPU smoke first')
        P.receipt_check(__file__, P.read(out / 'protocol.json')['preparation_code_sha256'])
    kernel = ctypes.windll.kernel32
    previous = kernel.SetThreadExecutionState(0x80000001)
    if not previous:
        raise ctypes.WinError()
    queue = dict(status='running', current_arm='control', experiment=str(out),
                 child_pid=os.getpid(), parent_pid=os.getpid(), runner='run_full_pretrained_detector.py',
                 started_utc=datetime.now(timezone.utc).isoformat(), offline_socket_guard=True,
                 process_independence=independence, temporary_system_awake=True, mode='smoke' if smoke else 'pilot')
    P.write(out / ('offline_smoke_queue.json' if smoke else 'queue.json'), queue)
    try:
        audit = P.execute(out, 'control', smoke=smoke, batch=8, attempt='offline' if smoke else 'full', resume=resume)
        queue.update(status='passed_offline_smoke' if smoke else 'completed', completed_epochs=audit['completed_epochs'],
                     best_epoch=audit['best_epoch'], stop_reason=audit['stop_reason'])
        return audit
    except BaseException as error:
        queue.update(status='failed', error=repr(error))
        raise
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        queue.update(finished_utc=datetime.now(timezone.utc).isoformat(), blocked_network_attempts=attempts,
                     temporary_system_awake=False)
        P.write(out / ('offline_smoke_queue.json' if smoke else 'queue.json'), queue)


def launch(out, resume=False):
    ensure_idle(out)
    P.load_protocol(out)
    mode = 'resume' if resume else 'run'
    stdout_path = out / ('resume_training.log' if resume else 'training.log')
    stderr_path = out / ('resume_training.err.log' if resume else 'training.err.log')
    if stdout_path.exists():
        raise FileExistsError('Keep existing training log')
    # The Windows venv redirector owns a kill-on-close job for its child.
    # Launch the same base interpreter directly with venv packages first,
    # avoiding that redirector and any client-owned process job entirely.
    base_executable = getattr(sys, '_base_executable', sys.executable)
    launch_env = os.environ.copy()
    launch_env['PYTHONPATH'] = os.pathsep.join([str(P.ROOT / '.venv/Lib/site-packages'),
                                              str(P.ROOT / 'data/scripts')])
    with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
        proc = subprocess.Popen([base_executable, '-u', str(Path(__file__).resolve()), mode, '--out', str(out)],
            cwd=P.ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            env=launch_env, close_fds=True, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP |
                                         subprocess.CREATE_BREAKAWAY_FROM_JOB)
    P.write(out / 'launch_receipt.json', dict(wrapper_pid=proc.pid, launcher_pid=os.getpid(),
        executable=base_executable, local_package_path=launch_env['PYTHONPATH'],
        mode=mode, launched_utc=datetime.now(timezone.utc).isoformat(), stdout=str(stdout_path), stderr=str(stderr_path),
        flags=['DETACHED_PROCESS', 'CREATE_NEW_PROCESS_GROUP', 'CREATE_BREAKAWAY_FROM_JOB']))
    print('DETACHED_LAUNCHED', proc.pid, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'smoke', 'launch', 'run', 'launch-resume', 'resume',
                                       'scheduled-run', 'scheduled-resume'])
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(P.ROOT):
        raise ValueError('Experiment must remain inside project workspace')
    if args.mode == 'prepare':
        prepare(out)
    elif args.mode in ('launch', 'launch-resume'):
        launch(out, resume=args.mode == 'launch-resume')
    else:
        scheduled = args.mode.startswith('scheduled-')
        if scheduled:
            prefix = 'resume_training' if args.mode == 'scheduled-resume' else 'training'
            sys.stdout = (out / (prefix + '.log')).open('x', encoding='utf-8', buffering=1)
            sys.stderr = (out / (prefix + '.err.log')).open('x', encoding='utf-8', buffering=1)
        run(out, smoke=args.mode == 'smoke', resume=args.mode in ('resume', 'scheduled-resume'),
            scheduled=scheduled)


if __name__ == '__main__':
    main()
