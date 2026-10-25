# -*- coding: utf-8 -*-
"""Stage2 数据集构造的纯函数单测：裁切 / 几何特征 / 抖动 / 置信度匹配。"""
import numpy as np
import pytest

import build_stage2_dataset as B


def test_geom_features_shape_and_square():
    # 框统一为 (cls, x0, y0, x1, y1)，与 YOLO label 行顺序一致，全计划不得混用
    feats = B.geom_features((0, 0.0, 0.0, 512.0, 512.0), patch_px=1024)
    assert len(feats) == 5
    assert feats[0] == pytest.approx(0.5)   # w/patch
    assert feats[1] == pytest.approx(0.5)   # h/patch
    assert feats[2] == pytest.approx(1.0)   # 长宽比
    assert feats[4] == pytest.approx(0.25)  # 相对面积


def test_geom_features_elongated_is_larger_than_square():
    thin = B.geom_features((0, 0.0, 0.0, 400.0, 40.0), patch_px=1024)
    square = B.geom_features((0, 0.0, 0.0, 126.0, 127.0), patch_px=1024)
    assert thin[2] > square[2]
    assert thin[0] > thin[1]


def test_geom_features_clips_extreme_aspect():
    feats = B.geom_features((0, 0.0, 0.0, 1024.0, 1.0), patch_px=1024)
    assert feats[2] <= B.MAX_ASPECT


def test_crop_with_margin_inside_returns_filled_square():
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[30:70, 30:70] = 255
    out = B.crop_with_margin(arr, (0, 30.0, 30.0, 70.0, 70.0), margin=0.0, out_size=256)
    assert out.shape == (256, 256, 3)
    assert out.mean() > 200


def test_crop_with_margin_pads_out_of_bounds_with_zeros():
    arr = np.zeros((100, 100), dtype=np.uint8)
    out = B.crop_with_margin(arr, (0, -20.0, -20.0, 20.0, 20.0), margin=0.0, out_size=64)
    assert out.shape == (64, 64, 3)
    assert out.mean() == pytest.approx(0.0)


def test_crop_with_margin_margin_expands_window():
    arr = np.zeros((200, 200), dtype=np.uint8)
    arr[80:120, 80:120] = 255
    tight = B.crop_with_margin(arr, (0, 80.0, 80.0, 120.0, 120.0), margin=0.0, out_size=64)
    loose = B.crop_with_margin(arr, (0, 80.0, 80.0, 120.0, 120.0), margin=1.0, out_size=64)
    # 边距越大，目标在 crop 中占比越小 -> 平均亮度越低
    assert loose.mean() < tight.mean()


def test_jitter_box_deterministic_and_bounded():
    box = (0, 100.0, 100.0, 200.0, 160.0)
    a = B.jitter_box(box, np.random.RandomState(0))
    b = B.jitter_box(box, np.random.RandomState(0))
    assert a == pytest.approx(b)
    assert a[0] == box[0]   # 类别不变
    cx0, cy0 = (box[1] + box[3]) / 2, (box[2] + box[4]) / 2
    cx1, cy1 = (a[1] + a[3]) / 2, (a[2] + a[4]) / 2
    side = max(box[3] - box[1], box[4] - box[2])
    assert abs(cx1 - cx0) <= B.JITTER_SHIFT * side + 1e-6
    assert abs(cy1 - cy0) <= B.JITTER_SHIFT * side + 1e-6
    assert a[3] > a[1] and a[4] > a[2]


def test_match_conf_uses_best_matching_prediction():
    gt = (0, 0.0, 0.0, 100.0, 100.0)
    preds = np.array([[0.0, 0.0, 100.0, 100.0], [500.0, 500.0, 600.0, 600.0]], dtype=np.float32)
    scores = np.array([0.42, 0.99], dtype=np.float32)
    classes = np.array([3.0, 0.0], dtype=np.float32)
    assert B.match_conf(gt, preds, scores, classes, 0.5) == pytest.approx(0.42)


def test_match_conf_zero_when_no_overlap():
    gt = (0, 0.0, 0.0, 100.0, 100.0)
    preds = np.array([[500.0, 500.0, 600.0, 600.0]], dtype=np.float32)
    scores = np.array([0.99], dtype=np.float32)
    classes = np.array([0.0], dtype=np.float32)
    assert B.match_conf(gt, preds, scores, classes, 0.5) == 0.0


def test_match_conf_zero_when_no_predictions():
    gt = (0, 0.0, 0.0, 100.0, 100.0)
    empty = np.zeros((0, 4), dtype=np.float32)
    assert B.match_conf(gt, empty, np.zeros(0, np.float32), np.zeros(0, np.float32), 0.5) == 0.0


def test_feature_dim_constant():
    assert B.N_FEATURES == B.BACKBONE_DIM + B.GEOM_DIM + 1
    assert B.BACKBONE_DIM == 448
    assert B.GEOM_DIM == 5
