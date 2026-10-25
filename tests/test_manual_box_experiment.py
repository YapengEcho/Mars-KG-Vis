import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data/scripts'))
import evaluate_manual_box_experiment as evaluation
from run_manual_box_experiment import model_hash


def test_duplicate_predictions_and_empty_image_count_as_false_positives():
    gt={'slide':[(0,0,0,10,10)],'empty':[]}
    preds={'slide':(np.array([[0,0,10,10],[0,0,10,10]]),np.array([.9,.8]),np.zeros(2)),
           'empty':(np.array([[0,0,10,10]]),np.array([.7]),np.zeros(1))}
    r=evaluation.summarize_predictions(gt,preds,.5)
    assert r['tp']==1 and r['fp']==2 and r['candidates']==3
    assert r['empty_image_fp_rate']==1 and r['precision']==pytest.approx(1/3)


def test_one_prediction_cannot_match_two_ground_truth_objects():
    gt={'overlap':[(0,0,0,10,10),(0,0,0,10,10)]}
    preds={'overlap':(np.array([[0,0,10,10]]),np.array([.9]),np.zeros(1))}
    r=evaluation.summarize_predictions(gt,preds,.5)
    assert r['tp']==1 and r['fn']==1 and r['recall']==.5


def test_validation_budget_excludes_a_higher_recall_point():
    rows=[dict(threshold=.1,candidates=100,recall=1.,precision=.5),
          dict(threshold=.5,candidates=10,recall=.8,precision=.9)]
    assert evaluation.choose_threshold(rows,10)['threshold']==.5
    with pytest.raises(ValueError):evaluation.choose_threshold(rows,0)


def test_threshold_ties_use_fewer_candidates_then_higher_threshold():
    rows=[dict(threshold=.1,candidates=10,recall=.8,precision=.8),
          dict(threshold=.2,candidates=8,recall=.8,precision=.8),
          dict(threshold=.3,candidates=8,recall=.8,precision=.8)]
    assert evaluation.choose_threshold(rows,10)['threshold']==.3


def test_test_split_cannot_infer_without_a_validation_lock(tmp_path,monkeypatch):
    root=tmp_path/'formal/manual_seed0';(root/'weights').mkdir(parents=True)
    weight=root/'weights/best.pt';weight.write_bytes(b'test checkpoint')
    protocol=tmp_path/'experiment.json';protocol.write_text('{}')
    (root/'training_audit.json').write_text(json.dumps(dict(status='complete',formal_result=True,
        experiment_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        best_sha256=hashlib.sha256(weight.read_bytes()).hexdigest())))
    monkeypatch.setattr(evaluation,'preflight',lambda *a:None)
    def forbidden(*a,**k):raise AssertionError('Test prediction was accessed before locking validation')
    monkeypatch.setattr(evaluation,'inference',forbidden)
    with pytest.raises(ValueError,match='lock a validation threshold'):
        evaluation.evaluate(tmp_path,{},'manual',0,'test')


def test_initial_model_hash_includes_scalar_tracking_buffers():
    model=torch.nn.BatchNorm2d(3);prior=model_hash(model)
    assert prior==model_hash(model)
    model.num_batches_tracked.add_(1)
    assert model_hash(model)!=prior
