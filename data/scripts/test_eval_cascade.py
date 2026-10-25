# -*- coding: utf-8 -*-
"""eval_cascade 的门控/指标纯函数单测。"""
import numpy as np
import pytest

import eval_cascade as E


def test_gate_accepts_confident_non_border():
    pred, review = E.gate_decision(np.array([0.1, 0.75, 0.1, 0.05]), 0.5, 0.15)
    assert pred == 1
    assert review is False


def test_gate_reviews_when_below_tau_cls():
    pred, review = E.gate_decision(np.array([0.3, 0.3, 0.3, 0.1]), 0.5, 0.15)
    assert pred == 0
    assert review is True


def test_gate_reviews_sl_cl_tie_even_when_confident():
    # top1=CL(0.5), top2=SL(0.5)，落在 {SL,CL} 且差 < δ -> 必复核
    pred, review = E.gate_decision(np.array([0.5, 0.0, 0.0, 0.5]), 0.5, 0.15)
    assert pred in (0, 3)
    assert review is True


def test_gate_accepts_clear_sl_cl_margin():
    pred, review = E.gate_decision(np.array([0.15, 0.05, 0.05, 0.75]), 0.5, 0.15)
    assert pred == 3
    assert review is False


def test_prf1_perfect_matrix():
    cm = np.eye(4, dtype=int) * 3
    per_class, macro_f1, acc, out_cm = E.prf1_from_confusion(cm)
    assert macro_f1 == pytest.approx(1.0)
    assert acc == pytest.approx(1.0)
    assert [p["f1"] for p in per_class] == pytest.approx([1.0] * 4)
    assert out_cm.tolist() == cm.tolist()


def test_prf1_handles_empty_class():
    cm = np.zeros((4, 4), dtype=int)
    cm[0, 0] = 5
    per_class, macro_f1, acc, _ = E.prf1_from_confusion(cm)
    assert acc == pytest.approx(1.0)
    assert per_class[1]["f1"] == 0.0
    assert 0.0 < macro_f1 < 1.0


def test_prf1_counts_miss_column_as_fn():
    # 4x5 混淆矩阵：第 5 列 = 漏检。漏检必须计入 FN 并拉低 recall，不能被静默吞掉。
    cm = np.zeros((4, 5), dtype=int)
    cm[0, 0] = 3
    cm[0, 4] = 1        # 1 个 SL 完全没被框到
    per_class, _macro_f1, acc, _ = E.prf1_from_confusion(cm)
    sl = per_class[0]
    assert sl["tp"] == 3 and sl["fn"] == 1 and sl["fp"] == 0
    assert sl["recall"] == pytest.approx(0.75)
    assert acc == pytest.approx(0.75)   # 分母含漏检


def test_review_curve_all_confident_correct():
    probas = np.array([[0.9, 0.02, 0.02, 0.06], [0.05, 0.05, 0.05, 0.85]], dtype=float)
    gt = np.array([0, 3])
    rows = E.review_curve(probas, gt, taus=(0.5,), deltas=(0.05,))
    assert len(rows) == 1
    assert rows[0]["auto_ratio"] == pytest.approx(1.0)
    assert rows[0]["auto_accuracy"] == pytest.approx(1.0)


def test_review_curve_strict_tau_lowers_auto_ratio():
    probas = np.array([[0.4, 0.2, 0.2, 0.2], [0.05, 0.05, 0.05, 0.85]], dtype=float)
    gt = np.array([0, 3])
    loose = E.review_curve(probas, gt, taus=(0.3,), deltas=(0.05,))[0]
    strict = E.review_curve(probas, gt, taus=(0.6,), deltas=(0.05,))[0]
    assert strict["auto_ratio"] < loose["auto_ratio"]


def test_stage1_miss_bound_counts_unmatched_gt():
    gt = {"a": [(0, 0.0, 0.0, 100.0, 100.0)]}
    preds = {"a": (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32))}
    bound = E.stage1_miss_bound(gt, preds, match_iou=0.5, tau_det=0.05)
    assert bound["gt_total"] == 1
    assert bound["matched"] == 0
    assert bound["unrecoverable_ratio"] == pytest.approx(1.0)


def test_stage1_miss_bound_counts_match():
    gt = {"a": [(0, 0.0, 0.0, 100.0, 100.0)]}
    preds = {"a": (np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float32),
                   np.array([0.9], dtype=np.float32),
                   np.array([0.0], dtype=np.float32))}
    bound = E.stage1_miss_bound(gt, preds, match_iou=0.5, tau_det=0.05)
    assert bound["matched"] == 1
    assert bound["recall"] == pytest.approx(1.0)
    assert bound["unrecoverable_ratio"] == pytest.approx(0.0)


# ---- 真 FP 口径补充诊断（detection_report）----
# 主口径（aggregate）的 precision 恒为 1，因为只有匹配上的同类框才进混淆矩阵。
# detection_report 追加检测口径统计，让误检真正进入 FP。

def _one_box(box, cls, score=0.9):
    return (np.array([box], dtype=np.float32),
            np.array([score], dtype=np.float32),
            np.array([cls], dtype=int))


def test_match_detections_class_aware_rejects_wrong_class():
    gt = [(0, 0.0, 0.0, 100.0, 100.0)]
    boxes, scores, cls = _one_box((0.0, 0.0, 100.0, 100.0), cls=1)
    recs_ca, used_ca = E.match_detections(gt, boxes, cls, scores, 0.5, class_aware=True)
    assert recs_ca[0][1] is False          # 位置完全重合，但类别是 SP -> class-aware 不认
    assert used_ca.tolist() == [False]     # 该框未认领 -> 算 FP
    recs_ga, used_ga = E.match_detections(gt, boxes, cls, scores, 0.5, class_aware=False)
    assert recs_ga[0][1] is True           # class-agnostic 只看位置 -> 认领
    assert recs_ga[0][2] == 1              # 预测类是 SP，用于混淆矩阵列
    assert used_ga.tolist() == [True]


def test_detection_report_perfect_match_gives_unit_prf1():
    gt = {"a": [(0, 0.0, 0.0, 100.0, 100.0)]}
    boxes, scores, cls = _one_box((0.0, 0.0, 100.0, 100.0), cls=0)
    rep = E.detection_report(gt, {"a": (boxes, scores, cls)}, match_iou=0.5)
    sl = rep["per_class"][0]
    assert (sl["tp"], sl["fp"], sl["fn"]) == (1, 0, 0)
    assert sl["precision"] == pytest.approx(1.0)
    assert sl["recall"] == pytest.approx(1.0)
    assert sl["f1"] == pytest.approx(1.0)
    assert rep["n_tp"] == 1 and rep["n_fp"] == 0 and rep["n_fn"] == 0
    assert rep["n_gt"] == 1 and rep["n_pred"] == 1


def test_detection_report_wrong_class_counts_as_fp_and_fn():
    # SL 真值上压了一个 SP 框：class-aware 下既是 SL 的漏检，也是 SP 的误检。
    gt = {"a": [(0, 0.0, 0.0, 100.0, 100.0)]}
    boxes, scores, cls = _one_box((0.0, 0.0, 100.0, 100.0), cls=1)
    rep = E.detection_report(gt, {"a": (boxes, scores, cls)}, match_iou=0.5)
    sl, sp = rep["per_class"][0], rep["per_class"][1]
    assert (sl["tp"], sl["fp"], sl["fn"]) == (0, 0, 1)
    assert sl["precision"] == 0.0 and sl["recall"] == 0.0
    assert (sp["tp"], sp["fp"], sp["fn"]) == (0, 1, 0)
    assert sp["precision"] == 0.0            # 有 FP 无 TP -> precision 归零，不再恒 1
    assert rep["n_fp"] == 1
    # class-agnostic 混淆矩阵：非对角 SL->SP，第 5 列漏检为 0
    cm = rep["confusion_matrix_class_agnostic"]
    assert cm[0] == [0, 1, 0, 0, 0]
    assert rep["n_unmatched_pred"] == 0


def test_detection_report_missed_gt_enters_miss_column():
    gt = {"a": [(2, 0.0, 0.0, 100.0, 100.0)]}     # FL，Stage1 完全没出框
    rep = E.detection_report(
        gt, {"a": (np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                   np.zeros(0, dtype=int))}, match_iou=0.5)
    fl = rep["per_class"][2]
    assert (fl["tp"], fl["fp"], fl["fn"]) == (0, 0, 1)
    assert rep["confusion_matrix_class_agnostic"][2] == [0, 0, 0, 0, 1]
    assert rep["n_pred"] == 0 and rep["n_unmatched_pred"] == 0


def test_detection_report_counts_unmatched_prediction():
    gt = {"a": [(0, 0.0, 0.0, 100.0, 100.0)]}
    boxes = np.array([[0.0, 0.0, 100.0, 100.0], [500.0, 500.0, 600.0, 600.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    cls = np.array([0, 0], dtype=int)
    rep = E.detection_report(gt, {"a": (boxes, scores, cls)}, match_iou=0.5)
    sl = rep["per_class"][0]
    assert (sl["tp"], sl["fp"], sl["fn"]) == (1, 1, 0)
    assert sl["precision"] == pytest.approx(0.5)
    assert sl["recall"] == pytest.approx(1.0)
    assert rep["n_unmatched_pred"] == 1     # 多出来的同类框：位置没匹配上 GT
    assert rep["n_pred"] == 2 and rep["n_gt"] == 1
