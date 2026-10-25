"""Continue the finite optimizer experiment after correcting receipt-only validation."""
import gc
import math
import os
from pathlib import Path
import shutil
import sys
import time

import run_detector_optimizer_20261022 as O

ORIGINAL_RECEIPT=O.optimizer_receipt


def checked_receipt(spec,arm,smoke=False):
    if spec['configs'][arm]['optimizer']!='SGD':
        return ORIGINAL_RECEIPT(spec,arm,smoke=smoke)
    folder=O.OUT/('smoke' if smoke else 'training')/arm
    ck=O.torch.load(folder/'recovery_last.pt',map_location='cpu',weights_only=False)
    cfg=spec['configs'][arm];args=ck['train_args'];groups=ck['optimizer']['param_groups']
    for key in ['optimizer','lr0','warmup_bias_lr','nbs','momentum','warmup_momentum']:
        assert args[key]==cfg[key],key
    batch_count=math.ceil((128 if smoke else spec['train_exposures'])/spec['physical_batch'])
    warmup_epochs=min(args['warmup_epochs'],max(args['epochs']-1,0))
    warmup_steps=round(warmup_epochs*batch_count) if warmup_epochs>0 else 0
    # Pinned SDK uses ni < nw, so its last interpolation is nw-1, not nw.
    expected=float(cfg['momentum'])
    if warmup_steps:
        assert (ck['epoch']+1)*batch_count>=warmup_steps
        expected=float(O.R.np.interp(warmup_steps-1,[0,warmup_steps],[cfg['warmup_momentum'],cfg['momentum']]))
    assert all('betas' not in g and abs(g['momentum']-expected)<1e-12 and g['nesterov'] for g in groups)
    assert all(g['initial_lr']==cfg['lr0'] for g in groups)
    actual=[{k:g[k] for k in ['lr','initial_lr','momentum','betas','weight_decay','nesterov','param_group'] if k in g} for g in groups]
    O.R.write(folder/'optimizer_receipt.json',dict(optimizer=cfg['optimizer'],lr0=cfg['lr0'],groups=actual,
        checkpoint_sha256=O.P.sha(folder/'recovery_last.pt'),actual_explicit_configuration_verified=True,
        configured_momentum=cfg['momentum'],expected_saved_momentum=expected,warmup_steps=warmup_steps,
        reason='Pinned SDK interpolates momentum only for ni < nw; validate the actual nw-1 endpoint',
        training_math_or_checkpoint_modified=False,receipt_validator_sha256=O.P.sha(Path(__file__))))
    del ck;gc.collect()


if __name__=='__main__':
    stamp=str(int(time.time()))
    sys.stdout=(O.OUT/f'training_receipt_fix_{stamp}.log').open('a',encoding='utf-8',buffering=1)
    sys.stderr=(O.OUT/f'training_receipt_fix_{stamp}.err.log').open('a',encoding='utf-8',buffering=1)
    previous=O.P.read(O.OUT/'queue.json')
    assert previous['status']=='failed' and previous['error']=='AssertionError()'
    assert previous['current_arm']=='sgd_lr1e2'
    assert O.P.read(O.OUT/'training/sgd_lr1e2/audit.json')['status']=='complete'
    spec=O.configure()  # Verify all original pinned scripts, inputs and environment.
    history=O.OUT/'resume_history'/f'receipt_fix_{stamp}';history.mkdir()
    shutil.copy2(O.OUT/'queue.json',history/'queue.failed.json')
    shutil.copy2(O.OUT/'training/sgd_lr1e2/audit.json',history/'sgd.audit.before.json')
    checkpoints={}
    for arm in O.ARMS:
        for name in ['recovery_last.pt','recovery_best.pt','train/weights/best.pt']:
            path=O.OUT/'training'/arm/name
            if path.exists():checkpoints[str(path)]=O.P.sha(path)
    checked_receipt(spec,'sgd_lr1e2')
    O.R.write(history/'receipt.json',dict(created_utc=O.R.utc(),pid=os.getpid(),
        original_protocol_sha256=O.P.sha(O.OUT/'protocol.json'),pinned_files_verified=len(spec['files_sha256']),
        validator_sha256=O.P.sha(Path(__file__)),checkpoint_hashes=checkpoints,
        original_training_code_and_configuration_modified=False,
        corrected_only='SGD saved momentum assertion: pinned SDK stops interpolation at nw-1',
        no_completed_arm_retrained=True,final_test_read=False))
    O.optimizer_receipt=checked_receipt
    O.scheduled(resume=True)
