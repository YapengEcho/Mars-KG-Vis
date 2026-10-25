# -*- coding: utf-8 -*-
"""统一协议下端到端级联评估（spec §5/§7-3）。

协议（与 compare_runs.py / eval_recall_report.py 一致，不得改动）：
    原始 val split、无 TTA、imgsz=1024、conf=0.001、nms_iou=0.7、match_iou=0.5、
    class-aware 贪心匹配。

Stage1 候选去重（--stage1-nms）默认为 per_class，即不做跨类 NMS，只保留同类别内的
NMS 结果。原因是跨类 NMS 会把大滑坡居中裁剪（*_big_*）里 IoU 0.72–0.94 的跨类重复框
按分数压掉一条，19 条 GT 因此失去唯一候选，全部落在 *_big_* 切片上。

产出（spec §5 的五项）：
    1. per-class P/R/F1 + 混淆矩阵
    2. macro-F1 + one-vs-rest AUC（SL 与 CL 单列）
    3. Stage1 漏检上界（级联召回上限 = Stage1 在 τ_det 下的召回）
    4. 单阶段（Stage1 直接出类）vs 级联（Stage2 重判类）对照
    5. 复核率 vs 自动判定精度曲线

用法：
    python eval_cascade.py --split val
    python eval_cascade.py --split test --tau-det 0.05 --out runs/stage2/eval_cascade_test.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from stage2_features import FEATURE_MODES, TerrainExtractor, compose_features
import terrain_features

BASE = Path(__file__).resolve().parents[2]
DATA_YAML = BASE / "data" / "processed" / "yolo" / "data.yaml"
WEIGHTS = BASE / "runs" / "detect" / "yolov8n_mars_rebal_w2" / "weights" / "best.pt"
STAGE2_LR = BASE / "runs" / "stage2" / "stage2_lr.joblib"
OUT = BASE / "runs" / "stage2" / "eval_cascade.json"

CLASS_NAMES = ("SL", "SP", "FL", "CL")
SL_ID, CL_ID = 0, 3
PATCH_PX = 1024
TERRAIN_DEM = BASE / "data" / "raw" / "terrain" / "Mars_MGS_MOLA_DEM_mosaic_global_463m_parallel.tif"


def resolve_feature_mode(requested: str, bundle: dict) -> str:
    """Resolve CLI mode against classifier provenance and reject mismatches."""
    bundle_mode = bundle.get("feature_mode", "ctx")
    if bundle_mode not in FEATURE_MODES:
        raise ValueError(f"classifier bundle has unsupported feature mode: {bundle_mode}")
    if requested == "auto":
        return bundle_mode
    if requested != bundle_mode:
        raise ValueError(f"feature mode mismatch: classifier={bundle_mode}, cli={requested}")
    return requested


def gate_decision(proba: np.ndarray, tau_cls: float, delta: float):
    """级联门控（spec §4）：返回 (预测类, 是否需人工复核)。

    复核条件（任一命中）：
      1. 最高类概率 < tau_cls；
      2. top-1/top-2 落在 {SL, CL} 且概率差 < delta（定义重叠导致的不可分对）。
    """
    order = np.argsort(-np.asarray(proba, dtype=float))
    top1, top2 = int(order[0]), int(order[1])
    p1 = float(proba[top1])
    p2 = float(proba[top2])
    if p1 < tau_cls:
        return top1, True
    if {top1, top2} == {SL_ID, CL_ID} and (p1 - p2) < delta:
        return top1, True
    return top1, False


def prf1_from_confusion(cm: np.ndarray, n_classes: int = 4):
    """由混淆矩阵（行=真值，列=预测）算 per-class P/R/F1、macro-F1、accuracy。

    允许 cm 的列数 > n_classes：多出来的列表示"漏检"（该 GT 没被任何框匹配上）。
    此时 FN 计入漏检列，FP 只看真实类别列，accuracy 分母含漏检——这正是检测场景
    应有的口径（漏检必须拉低召回，不能被静默忽略）。
    """
    cm = np.asarray(cm, dtype=int)
    per_class, f1s = [], []
    for c in range(n_classes):
        tp = int(cm[c, c])
        fp = int(cm[:n_classes, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)   # 含漏检列
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class.append({"class": CLASS_NAMES[c], "gt": int(cm[c, :].sum()), "tp": tp,
                          "fp": fp, "fn": fn, "precision": round(prec, 4),
                          "recall": round(rec, 4), "f1": round(f1, 4)})
        f1s.append(f1)
    total = int(cm.sum())
    acc = float(sum(cm[i, i] for i in range(n_classes)) / total) if total else 0.0
    return per_class, float(np.mean(f1s)), round(acc, 4), cm


def review_curve(probas: np.ndarray, gt_cls: np.ndarray, taus, deltas):
    """扫描 tau_cls/delta，给出"复核多少比例能换来多少精度"（spec §5-⑤）。

    只统计"被 Stage2 判定为自动通过"的框：auto_ratio = 自动通过数 / 总框数，
    auto_accuracy = 自动通过中判对的比例（GT 类由 IoU 匹配给出），
    auto_macro_f1 在自动通过子集上按 4 类计算（空类记 0）。
    """
    probas = np.asarray(probas, dtype=float)
    gt_cls = np.asarray(gt_cls, dtype=int)
    rows = []
    for tau in taus:
        for delta in deltas:
            auto_pred, auto_gt = [], []
            for p, g in zip(probas, gt_cls):
                pred, review = gate_decision(p, tau, delta)
                if not review:
                    auto_pred.append(pred)
                    auto_gt.append(int(g))
            n_total = len(gt_cls)
            n_auto = len(auto_pred)
            if n_auto == 0:
                rows.append({"tau_cls": tau, "delta": delta, "auto_ratio": 0.0,
                             "auto_accuracy": 0.0, "auto_macro_f1": 0.0, "n_auto": 0})
                continue
            cm = np.zeros((4, 4), dtype=int)
            for t, p in zip(auto_gt, auto_pred):
                cm[t, p] += 1
            _per, macro_f1, acc, _cm = prf1_from_confusion(cm)
            rows.append({"tau_cls": tau, "delta": delta,
                         "auto_ratio": round(n_auto / max(n_total, 1), 4),
                         "auto_accuracy": acc, "auto_macro_f1": round(macro_f1, 4),
                         "n_auto": n_auto})
    return rows


def stage1_miss_bound(
    gt: dict,
    preds: dict,
    match_iou: float,
    tau_det: float,
    class_aware: bool = True,
):
    """Stage1 漏检上界（spec §5-③）：级联整体召回 <= Stage1 在 tau_det 下的召回。

    用 class-aware 贪心匹配统计"GT 是否被 Stage1 框到"——未被框到的 GT，
    Stage2 无论多强都无法补救，因此 unrecoverable_ratio 就是级联天花板。
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_recall_report import match_image

    total = matched = 0
    per_class = {n: {"gt": 0, "matched": 0} for n in CLASS_NAMES}
    for stem, rows in gt.items():
        boxes, scores, cls = preds.get(
            stem,
            (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)),
        )
        for cls_id, hit, _sbin, _biou, _j in match_image(
            rows, boxes, scores, cls, match_iou, tau_det, class_aware=class_aware
        ):
            name = CLASS_NAMES[cls_id]
            total += 1
            per_class[name]["gt"] += 1
            if hit:
                matched += 1
                per_class[name]["matched"] += 1
    for name in CLASS_NAMES:
        n = per_class[name]["gt"]
        per_class[name]["recall"] = round(per_class[name]["matched"] / n, 4) if n else 0.0
    return {
        "tau_det": tau_det,
        "gt_total": total,
        "matched": matched,
        "recall": round(matched / total, 4) if total else 0.0,
        "unrecoverable_gt": total - matched,
        "unrecoverable_ratio": round((total - matched) / total, 4) if total else 0.0,
        "per_class": per_class,
    }


def class_agnostic_nms(boxes, scores, iou: float):
    """压平类别做 NMS，返回**保留索引**。

    返回索引而非折叠后的数组，是为了保住每个框的原始类别标签——单阶段对照
    （Stage1 直接出类）必须复用**同一批框**，只换类别来源，否则对比不成立。
    """
    if boxes is None or len(boxes) == 0:
        return np.zeros(0, dtype=int)
    import torch
    from torchvision.ops import nms
    keep = nms(torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32)),
               torch.from_numpy(np.ascontiguousarray(scores, dtype=np.float32)), iou)
    return keep.cpu().numpy().astype(int)


def stage1_candidates(boxes, scores, iou: float, mode: str):
    """Stage1 出框后，哪些框晋升为 Stage2 候选（返回保留索引）。

    mode="agnostic"：压平类别做 NMS（冻结协议口径）。同一处滑坡被模型同时打上
    多个类时，几个框互相竞争，低分的那个（可能是对的类）被丢掉。
    mode="per_class"：只保留各类自己的 NMS 结果（ultralytics 推理时已做），
    同一处滑坡的多个类标框全部交给 Stage2，类别在重判之后才决定。
    """
    if mode == "agnostic":
        return class_agnostic_nms(boxes, scores, iou)
    if mode == "per_class":
        return np.arange(len(boxes), dtype=int)
    raise ValueError(f"unknown stage1-nms mode {mode}")


def _iou_matrix(gt_rows, boxes):
    """GT 行 (cls, x0, y0, x1, y1) 与预测框的逐对 IoU，形状 (n_gt, n_pred)。"""
    gt_arr = np.array([[g[1], g[2], g[3], g[4]] for g in gt_rows], dtype=np.float32)
    lt = np.maximum(gt_arr[:, None, :2], boxes[None, :, :2])
    rb = np.minimum(gt_arr[:, None, 2:], boxes[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    ag = (gt_arr[:, 2] - gt_arr[:, 0]) * (gt_arr[:, 3] - gt_arr[:, 1])
    ap = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = ag[:, None] + ap[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_detections(gt_rows, boxes, cls_arr, scores, match_iou: float, class_aware: bool):
    """贪心 IoU 匹配，按得分从高到低认领。

    返回 (records, matched_pred)：
        records     —— 逐 GT 的 (gt_cls, hit, pred_cls, best_iou, box_index)；
                       hit=False 时 pred_cls=-1、box_index=-1（漏检不被吞掉）。
        matched_pred—— 长度等于框数的 bool 数组，False 表示该框未被任何 GT 认领（即 FP）。

    class_aware=True 只允许同类匹配（检测口径的 TP/FP 判定）；
    class_aware=False 只看位置，类别仅用于填写混淆矩阵的列（非退化混淆矩阵）。
    """
    n_pd = 0 if boxes is None else len(boxes)
    if not gt_rows:
        return [], np.zeros(n_pd, dtype=bool)
    if n_pd == 0:
        return [(int(g[0]), False, -1, 0.0, -1) for g in gt_rows], np.zeros(0, dtype=bool)
    ious = _iou_matrix(gt_rows, boxes)
    used = np.zeros(n_pd, dtype=bool)
    order = np.argsort(-np.asarray(scores, dtype=float))
    records = []
    for g_i, g in enumerate(gt_rows):
        gt_cls = int(g[0])
        best_iou, best_j = 0.0, -1
        for j in order:
            if used[j] or (class_aware and int(cls_arr[j]) != gt_cls):
                continue
            if ious[g_i, j] > best_iou:
                best_iou, best_j = float(ious[g_i, j]), int(j)
                if best_iou >= match_iou:
                    break
        hit = best_iou >= match_iou and best_j >= 0
        if hit:
            used[best_j] = True
        records.append((gt_cls, hit, int(cls_arr[best_j]) if hit else -1,
                        best_iou, best_j if hit else -1))
    return records, used


def match_stage2_preds(gt_rows, boxes, cls_arr, scores, match_iou: float):
    """把 (框, 给定类别) 与 GT 做 class-aware 贪心匹配。

    返回逐 GT 记录 [(gt_cls, matched, pred_cls, best_iou, best_box_index)]；
    未匹配的 GT 也会返回（matched=False，best_box_index=-1），漏检因此不会被吞掉。
    """
    records, _used = match_detections(gt_rows, boxes, cls_arr, scores, match_iou,
                                      class_aware=True)
    return records


def detection_report(gt: dict, preds_per_image: dict, match_iou: float):
    """真 FP 口径补充诊断（追加口径，不替代主口径）。

    主口径 aggregate() 只把"与同类 GT 匹配上的框"放进混淆矩阵，因此 precision
    恒为 1.0、约两千个误检被静默丢弃。本函数按检测文献口径重算：

    1. class-aware 匹配下的 per-class TP/FP/FN —— 未被任何同类 GT 认领的预测框
       一律算 FP（框得再准，类别错/位置偏都是误检），漏检计入 FN；
    2. class-agnostic IoU 匹配（只看位置）产出的 4x5 混淆矩阵，行=GT 类、
       列=预测类、第 5 列=漏检——非对角项才可能出现，同时报出未匹配预测数。
    """
    n_cls = len(CLASS_NAMES)
    cm_agn = np.zeros((n_cls, n_cls + 1), dtype=int)
    tp = np.zeros(n_cls, dtype=int)
    fp = np.zeros(n_cls, dtype=int)
    fn = np.zeros(n_cls, dtype=int)
    n_pred = n_unmatched_pred = 0
    empty = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, dtype=int))
    for stem, rows_gt in gt.items():
        boxes, scores, cls_arr = preds_per_image.get(stem, empty)
        n_pred += len(boxes)
        recs_ca, used_ca = match_detections(rows_gt, boxes, cls_arr, scores,
                                            match_iou, class_aware=True)
        for gt_cls, hit, _pc, _iou, _j in recs_ca:
            if hit:
                tp[gt_cls] += 1
            else:
                fn[gt_cls] += 1
        for j in range(len(boxes)):
            if not used_ca[j]:
                fp[int(cls_arr[j])] += 1
        recs_agn, used_agn = match_detections(rows_gt, boxes, cls_arr, scores,
                                              match_iou, class_aware=False)
        for gt_cls, hit, pred_cls, _iou, _j in recs_agn:
            cm_agn[gt_cls, pred_cls if hit else n_cls] += 1
        n_unmatched_pred += int((~used_agn).sum())

    per_class, f1s = [], []
    for c in range(n_cls):
        t, f_p, f_n = int(tp[c]), int(fp[c]), int(fn[c])
        prec = t / (t + f_p) if t + f_p else 0.0
        rec = t / (t + f_n) if t + f_n else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class.append({"class": CLASS_NAMES[c], "gt": t + f_n, "tp": t, "fp": f_p,
                          "fn": f_n, "precision": round(prec, 4),
                          "recall": round(rec, 4), "f1": round(f1, 4)})
        f1s.append(f1)
    return {
        "per_class": per_class,
        "macro_f1": round(float(np.mean(f1s)), 4),
        "n_tp": int(tp.sum()), "n_fp": int(fp.sum()), "n_fn": int(fn.sum()),
        "n_pred": int(n_pred), "n_gt": int(cm_agn.sum()),
        "n_unmatched_pred": int(n_unmatched_pred),
        "confusion_matrix_class_agnostic": cm_agn.tolist(),
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument(
        "--feature-weights",
        default="",
        help="optional separate backbone weights for Stage2 features; defaults to --weights",
    )
    ap.add_argument("--stage2", default=str(STAGE2_LR))
    ap.add_argument("--data", default=str(DATA_YAML))
    ap.add_argument("--feature-mode", choices=("auto",) + FEATURE_MODES, default="auto",
                    help="auto 从分类器包读取；也可显式指定并校验")
    ap.add_argument("--terrain-dem", default=str(TERRAIN_DEM))
    ap.add_argument("--terrain-padding", type=int, default=2)
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.001, help="Stage1 推理 conf（统一协议固定值）")
    ap.add_argument("--tau-det", type=float, default=0.05, help="级联 Stage1 出框阈值（召回优先）")
    ap.add_argument("--nms-iou", type=float, default=0.7, help="统一协议值")
    ap.add_argument("--stage1-nms", choices=("agnostic", "per_class"), default="per_class",
                    help="Stage1 候选去重方式：per_class 为当前冻结口径（保留跨类重复框，"
                         "避免大滑坡居中裁剪中的跨类框互压），agnostic 为旧的跨类 NMS 口径")
    ap.add_argument(
        "--stage1-class-agnostic",
        action="store_true",
        help="ignore classes when computing the Stage1 miss bound (for 1-class candidate models)",
    )
    ap.add_argument("--match-iou", type=float, default=0.5, help="统一协议值")
    ap.add_argument("--tau-cls", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=0.15)
    ap.add_argument(
        "--accept-thresholds",
        default="",
        help="optional comma-separated Stage2 max-probability thresholds for candidate filtering",
    )
    ap.add_argument("--device", default="0")
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    return ap.parse_args()


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import joblib
    from PIL import Image
    from build_stage2_dataset import (FrozenBackboneFeaturizer, crop_with_margin,
                                      geom_features)
    from eval_recall_report import (infer, load_ground_truth, load_images,
                                    resolve_dataset_root)
    from train_stage2_cls import ovr_auc, sl_cl_auc

    args = parse_args()
    root = resolve_dataset_root(Path(args.data))
    images = load_images(root, args.split)
    if args.limit:
        images = images[:args.limit]
    gt = load_ground_truth(root, args.split, images, patch_px=PATCH_PX)

    preds_raw = infer(Path(args.weights), images, args.imgsz, args.conf, args.device, False)
    bound = stage1_miss_bound(
        gt,
        preds_raw,
        args.match_iou,
        args.tau_det,
        class_aware=not args.stage1_class_agnostic,
    )

    bundle = joblib.load(args.stage2)
    feature_mode = resolve_feature_mode(args.feature_mode, bundle)
    clf, scaler = bundle["model"], bundle["scaler"]
    need_ctx = feature_mode in ("ctx", "ctx_terrain")
    terrain_extractor = None
    if feature_mode in ("terrain", "ctx_terrain"):
        terrain_extractor = TerrainExtractor(Path(args.terrain_dem), padding_px=args.terrain_padding)

    # 第一遍：逐图 class-agnostic NMS + tau_det 出框，收集 crop 与框记录
    boxes_per_image = {}
    flat_crops = []
    flat_terrain = []
    for image in images:
        boxes, scores, classes = preds_raw.get(
            image.stem,
            (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)),
        )
        if len(boxes) == 0:
            boxes_per_image[image.stem] = (
                np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, dtype=int))
            continue
        keep = stage1_candidates(boxes, scores, args.nms_iou, args.stage1_nms)
        keep = keep[scores[keep] >= args.tau_det]
        kept_boxes = boxes[keep].astype(np.float32)
        kept_scores = scores[keep].astype(np.float32)
        kept_cls = classes[keep].astype(int)
        boxes_per_image[image.stem] = (kept_boxes, kept_scores, kept_cls)
        if len(kept_boxes) == 0:
            continue
        arr = np.asarray(Image.open(image).convert("L"))
        for b in kept_boxes:
            pixel_box = (0, float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            if need_ctx:
                flat_crops.append(crop_with_margin(arr, pixel_box, args.margin))
            if terrain_extractor is not None:
                terrain_box = (
                    0,
                    ((float(b[0]) + float(b[2])) / 2.0) / PATCH_PX,
                    ((float(b[1]) + float(b[3])) / 2.0) / PATCH_PX,
                    (float(b[2]) - float(b[0])) / PATCH_PX,
                    (float(b[3]) - float(b[1])) / PATCH_PX,
                )
                flat_terrain.append(terrain_extractor.for_patch(image.stem, terrain_box, patch_px=PATCH_PX))
            else:
                flat_terrain.append(np.zeros(len(terrain_features.TERRAIN_FEATURE_NAMES), dtype=np.float32))

    # 冻结骨干批量提特征（推理期与训练期共用同一个 featurizer 与同一裁切规则）
    if flat_crops:
        feature_weights = Path(args.feature_weights) if args.feature_weights else Path(args.weights)
        featurizer = FrozenBackboneFeaturizer(feature_weights, device=args.device)
        chunks = [featurizer.features(np.stack(flat_crops[i:i + args.batch]))
                  for i in range(0, len(flat_crops), args.batch)]
        backbone = np.concatenate(chunks, axis=0)
        del featurizer
    else:
        backbone = np.zeros((0, 0), dtype=np.float32)
    terrain_values = (
        np.stack(flat_terrain).astype(np.float32)
        if flat_terrain
        else np.zeros((0, len(terrain_features.TERRAIN_FEATURE_NAMES)), dtype=np.float32)
    )

    # 第二遍：逐框拼特征（backbone 448 + geom 5 + conf 1）-> Stage2 概率 -> 门控
    stage2_cls_per_image, proba_per_image, cursor = {}, {}, 0
    for image in images:
        kept_boxes, kept_scores, _kept_cls = boxes_per_image[image.stem]
        n = len(kept_boxes)
        if n == 0:
            stage2_cls_per_image[image.stem] = np.zeros(0, dtype=int)
            proba_per_image[image.stem] = np.zeros((0, 4), dtype=float)
            continue
        block = backbone[cursor:cursor + n]
        terrain_block = terrain_values[cursor:cursor + n]
        cursor += n
        geom = np.asarray([
            geom_features((0, float(b[0]), float(b[1]), float(b[2]), float(b[3])), patch_px=PATCH_PX)
            for b in kept_boxes], dtype=np.float32)
        X = compose_features(block, geom, kept_scores.reshape(-1, 1), terrain_block, feature_mode)
        probas = clf.predict_proba(scaler.transform(X))
        preds_cls = np.zeros(n, dtype=int)
        for i in range(n):
            preds_cls[i], _review = gate_decision(probas[i], args.tau_cls, args.delta)
        stage2_cls_per_image[image.stem] = preds_cls
        proba_per_image[image.stem] = probas

    # ---- 指标 ----
    def aggregate(mode: str, accept_threshold: float | None = None):
        """mode="stage2"：级联重判类别；mode="stage1"：Stage1 原始类别。

        两者共用同一批框（class-agnostic NMS + tau_det 之后），只换类别来源，
        否则"级联是否净赚"的对照不成立。混淆矩阵为 4x5：第 5 列是漏检
        （该 GT 在 tau_det 下没有任何框匹配上），漏检必须计入 FN，否则召回虚高。
        """
        cm = np.zeros((4, 5), dtype=int)
        for image in images:
            rows_gt = gt.get(image.stem, [])
            if not rows_gt:
                continue
            kept_boxes, kept_scores, stage1_cls = boxes_per_image[image.stem]
            cls_arr = stage1_cls if mode == "stage1" else stage2_cls_per_image[image.stem]
            if accept_threshold is not None and mode == "stage2":
                proba = proba_per_image[image.stem]
                keep_accept = (
                    np.max(proba, axis=1) >= accept_threshold
                    if len(proba)
                    else np.zeros(0, dtype=bool)
                )
                kept_boxes = kept_boxes[keep_accept]
                kept_scores = kept_scores[keep_accept]
                cls_arr = cls_arr[keep_accept]
            for gt_cls, hit, pred_cls, _iou, _j in match_stage2_preds(
                    rows_gt, kept_boxes, cls_arr, kept_scores, args.match_iou):
                if hit and 0 <= pred_cls < 4:
                    cm[gt_cls, pred_cls] += 1
                else:
                    cm[gt_cls, 4] += 1   # 漏检列
        per_class, macro_f1, acc, _ = prf1_from_confusion(cm)
        return {"per_class": per_class, "macro_f1": round(macro_f1, 4),
                "accuracy": acc, "confusion_matrix": cm.tolist()}

    n_boxes = int(sum(len(v[0]) for v in boxes_per_image.values()))
    report = {
        "protocol": {"split": args.split, "imgsz": args.imgsz, "conf": args.conf,
                     "nms_iou": args.nms_iou, "stage1_nms": args.stage1_nms,
                     "match_iou": args.match_iou,
                     "tau_det": args.tau_det, "tau_cls": args.tau_cls, "delta": args.delta,
                     "weights": str(args.weights),
                     "feature_weights": str(Path(args.feature_weights)) if args.feature_weights else str(args.weights),
                     "stage1_class_agnostic": bool(args.stage1_class_agnostic),
                     "stage2": str(args.stage2),
                     "feature_mode": feature_mode,
                     "terrain_dem": str(Path(args.terrain_dem)) if terrain_extractor else ""},
        "stage1_miss_bound": bound,
        "n_images": len(images),
        "n_boxes": n_boxes,
        # 补充诊断：真 FP 口径（Stage1 同一批框，与 Stage1/2 类别无关）
        "detection_fp": detection_report(gt, boxes_per_image, args.match_iou),
    }

    # 逐 GT 收集级联概率：只取 Stage1 能框到的 GT（匹配框的索引由
    # match_stage2_preds 直接给出，不再重算 IoU 反查），供 AUC 与复核曲线用。
    all_probas, all_gt = [], []
    for image in images:
        rows_gt = gt.get(image.stem, [])
        if not rows_gt:
            continue
        kept_boxes, kept_scores, stage1_cls = boxes_per_image[image.stem]
        probas = proba_per_image[image.stem]
        if len(kept_boxes) == 0 or len(probas) == 0:
            continue
        for gt_cls, hit, _pc, _iou, j in match_stage2_preds(
                rows_gt, kept_boxes, stage1_cls, kept_scores, args.match_iou):
            if not hit or j < 0:
                continue
            all_probas.append(probas[j])
            all_gt.append(int(gt_cls))

    all_probas = np.asarray(all_probas, dtype=float).reshape(-1, 4)
    all_gt = np.asarray(all_gt, dtype=int)
    if all_probas.size:
        report["cascade_matched"] = {
            "n": int(len(all_gt)),
            "ovr_auc": ovr_auc(all_probas, all_gt),
            "sl_cl_auc": sl_cl_auc(all_probas, all_gt),
        }
    report["cascade"] = aggregate("stage2")
    report["single_stage"] = aggregate("stage1")
    accept_thresholds = [
        float(x) for x in args.accept_thresholds.split(",") if x.strip()
    ]
    if accept_thresholds:
        report["cascade_acceptance_sweep"] = []
        for accept_threshold in accept_thresholds:
            filtered = aggregate("stage2", accept_threshold=accept_threshold)
            filtered_fp = {}
            for image in images:
                boxes, scores, _stage1_cls = boxes_per_image[image.stem]
                stage2_cls = stage2_cls_per_image[image.stem]
                proba = proba_per_image[image.stem]
                keep_accept = (
                    np.max(proba, axis=1) >= accept_threshold
                    if len(proba)
                    else np.zeros(0, dtype=bool)
                )
                filtered_fp[image.stem] = (
                    boxes[keep_accept], scores[keep_accept], stage2_cls[keep_accept]
                )
            fp = detection_report(gt, filtered_fp, args.match_iou)
            n_filtered = int(sum(len(v[0]) for v in filtered_fp.values()))
            report["cascade_acceptance_sweep"].append({
                "accept_threshold": accept_threshold,
                "n_boxes": n_filtered,
                "cascade": filtered,
                "detection_fp": fp,
            })
    report["review_curve"] = review_curve(
        all_probas, all_gt,
        taus=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), deltas=(0.0, 0.05, 0.1, 0.2))
    review_flags = [gate_decision(p, args.tau_cls, args.delta)[1] for p in all_probas]
    report["review_at_default"] = {
        "tau_cls": args.tau_cls, "delta": args.delta, "n": int(len(review_flags)),
        "review_ratio": round(float(np.mean(review_flags)) if review_flags else 0.0, 4),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Stage1(tau_det={args.tau_det}, nms={args.stage1_nms}) 召回上界 "
          f"{bound['recall']:.3f}，"
          f"不可补救 GT {bound['unrecoverable_gt']}（{bound['unrecoverable_ratio']:.3f}）")
    print(f"候选框 {n_boxes}")
    for name in ("single_stage", "cascade"):
        m = report[name]
        print(f"[{name}] macroF1={m['macro_f1']} acc={m['accuracy']} "
              + " ".join(f"{p['class']}:F1={p['f1']}" for p in m["per_class"]))
    fp_rep = report["detection_fp"]
    print(f"[detection_fp] 真FP口径 TP={fp_rep['n_tp']} FP={fp_rep['n_fp']} "
          f"FN={fp_rep['n_fn']} 未匹配框={fp_rep['n_unmatched_pred']} "
          f"macroF1={fp_rep['macro_f1']}")
    print(f"复核率(默认阈值) {report['review_at_default']['review_ratio']}")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
