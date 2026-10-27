# -*- coding: utf-8 -*-
"""训练 Stage2 级联分类器（spec §7-2）。

主路线：LR / GBDT（class_weight 平衡 + 分层**分组** CV 选超参）。
对照组：L2 归一化后的类原型（余弦最近原型）与 kNN(cosine)——针对 FL/SP 少样本的兜底。

关键点：
- 所有特征标准化统计量、超参搜索**只在 train 内**做；val 仅用于最终报告，绝不参与调参。
- 抖动增强样本与原样本共享 group id，CV 用 StratifiedGroupKFold 保证同组同折，
  否则同一滑坡的多个抖动副本跨折会造成 CV 虚高。
- 输出 one-vs-rest AUC 时把 SL vs CL 单列（spec §5-②：这是本项目的工程天花板）。
"""
import argparse
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "runs" / "stage2" / "stage2_dataset.npz"
OUT_DIR = BASE / "runs" / "stage2"
CLASS_NAMES = ("SL", "SP", "FL", "CL")
SL_ID, CL_ID = 0, 3


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--c-grid", default="0.01,0.03,0.1,0.3,1,3,10")
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--gbdt-iter", type=int, default=200)
    ap.add_argument("--skip-gbdt", action="store_true",
                    help="跳过耗时的 GBDT，只训练和评估 LR/原型/kNN")
    ap.add_argument("--dedupe-train-groups", action="store_true",
                    help="训练集每个原始框组只保留一个样本，验证集不变")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balance-mode", choices=("none", "sqrt", "balanced"), default="balanced",
                    help="类别权重对照：none不加权、sqrt温和加权、balanced逆频率；每折只用训练行计算")
    return ap.parse_args()


def load_dataset(path: Path):
    d = np.load(path, allow_pickle=False)
    X, y = d["X"], d["y"]
    split, group = d["split"], d["group"]
    return X, y, split, group


def load_feature_metadata(path: Path) -> dict:
    """Read feature provenance without loading model arrays into memory twice."""
    with np.load(path, allow_pickle=False) as d:
        mode_value = d["feature_mode"] if "feature_mode" in d else np.asarray("ctx")
        mode = str(mode_value.item() if mode_value.ndim == 0 else mode_value[0])
        names = d["feature_names"].astype(str).tolist() if "feature_names" in d else []
        terrain_dem = str(d["terrain_dem"].item()) if "terrain_dem" in d else ""
    return {"feature_mode": mode, "feature_names": names, "terrain_dem": terrain_dem}


def base_sample_mask(split: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Keep one original sample per train group; leave validation untouched."""
    keep = np.ones(len(split), dtype=bool)
    seen = set()
    for i, (part, group_id) in enumerate(zip(split.tolist(), group.tolist())):
        if part != "train":
            continue
        group_id = int(group_id)
        if group_id in seen:
            keep[i] = False
        else:
            seen.add(group_id)
    return keep


def scaled_fold_features(X_raw: np.ndarray, train_indices, held_indices):
    """Fit preprocessing on this CV fold's training rows, then transform both parts."""
    from sklearn.preprocessing import StandardScaler

    fold_scaler = StandardScaler().fit(X_raw[train_indices])
    return (fold_scaler.transform(X_raw[train_indices]),
            fold_scaler.transform(X_raw[held_indices]), fold_scaler)


def classifier_class_weights(y_train, mode="balanced"):
    """Compute mean-one weights using only the supplied training partition."""
    if mode not in {"none", "sqrt", "balanced"}:
        raise ValueError(f"Unknown balance mode: {mode}")
    labels = np.asarray(y_train)
    if labels.ndim != 1 or labels.size == 0 or labels.dtype.kind not in "iu":
        raise ValueError("Training labels must be a nonempty integer vector")
    if np.any(labels < 0) or np.any(labels >= len(CLASS_NAMES)):
        raise ValueError("Training subtype ID outside SL/SP/FL/CL mapping")
    if mode == "none":
        return None
    classes, counts = np.unique(labels, return_counts=True)
    raw = labels.size / (len(classes) * counts.astype(float))
    if mode == "sqrt":
        raw = np.sqrt(raw)
    weights = raw / (np.dot(counts, raw) / labels.size)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def classification_sample_weights(y_train, mode="balanced"):
    """Match LR class weights for estimators which accept per-row weights."""
    labels = np.asarray(y_train)
    weights = classifier_class_weights(labels, mode)
    if weights is None:
        return np.ones(labels.size, dtype=float)
    return np.asarray([weights[int(c)] for c in labels], dtype=float)


def prf1(y_true, y_pred, n_classes: int = 4):
    """返回 (per_class 列表, macro_f1, accuracy, 混淆矩阵)。混淆矩阵行=真值，列=预测。"""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    per_class, f1s = [], []
    for c in range(n_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class.append({"class": CLASS_NAMES[c], "tp": tp, "fp": fp, "fn": fn,
                          "gt": int(cm[c, :].sum()),
                          "precision": round(prec, 4), "recall": round(rec, 4),
                          "f1": round(f1, 4)})
        f1s.append(f1)
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    return per_class, float(np.mean(f1s)), round(acc, 4), cm


def ovr_auc(scores: np.ndarray, y_true: np.ndarray, n_classes: int = 4):
    """One-vs-rest AUC（Mann-Whitney U 口径，无 sklearn 依赖，便于单测与复核）。"""
    out = {}
    for c in range(n_classes):
        pos = scores[y_true == c, c]
        neg = scores[y_true != c, c]
        if pos.size == 0 or neg.size == 0:
            out[CLASS_NAMES[c]] = None
            continue
        allv = np.concatenate([pos, neg])
        ranks = np.argsort(np.argsort(allv)) + 1
        rp = ranks[:pos.size].sum()
        out[CLASS_NAMES[c]] = round(
            float((rp - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)), 4)
    return out


def sl_cl_auc(scores: np.ndarray, y_true: np.ndarray):
    """只取 GT∈{SL,CL} 的样本，用 (P(CL)-P(SL)) 做二分类 AUC。"""
    mask = (y_true == SL_ID) | (y_true == CL_ID)
    if mask.sum() == 0:
        return None
    s = scores[mask, CL_ID] - scores[mask, SL_ID]
    t = (y_true[mask] == CL_ID).astype(int)
    pos, neg = s[t == 1], s[t == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv)) + 1
    rp = ranks[:pos.size].sum()
    return round(float((rp - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)), 4)


def prototype_scores(X_train, y_train, X_eval, n_classes: int = 4):
    """类原型余弦相似度打分（L2 归一化后的类均值 = 原型）。"""
    norms = np.linalg.norm(X_train, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Ztr = X_train / norms
    proto = np.zeros((n_classes, X_train.shape[1]), dtype=np.float64)
    for c in range(n_classes):
        m = y_train == c
        if m.sum() == 0:
            continue
        proto[c] = Ztr[m].mean(axis=0)
    pn = np.linalg.norm(proto, axis=1, keepdims=True)
    pn[pn == 0] = 1.0
    proto = proto / pn
    ne = np.linalg.norm(X_eval, axis=1, keepdims=True)
    ne[ne == 0] = 1.0
    return (X_eval / ne) @ proto.T


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib

    args = parse_args()
    data_path = Path(args.data)
    X, y, split, group = load_dataset(data_path)
    metadata = load_feature_metadata(data_path)
    if args.dedupe_train_groups:
        keep = base_sample_mask(split, group)
        X, y, split, group = X[keep], y[keep], split[keep], group[keep]
    tr = split == "train"
    va = split == "val"
    if tr.sum() == 0:
        raise SystemExit("train split 为空，先跑 build_stage2_dataset.py")
    Xtr_raw, ytr, gtr = X[tr], y[tr], group[tr]
    has_val = bool(va.sum())
    Xva_raw, yva = X[va], y[va]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler().fit(Xtr_raw)
    Xtr, Xva = scaler.transform(Xtr_raw), (scaler.transform(Xva_raw) if has_val else None)
    report = {"n_train": int(tr.sum()), "n_val": int(va.sum()),
              "n_features": int(X.shape[1]), "folds": args.folds,
              "feature_mode": metadata["feature_mode"],
              "feature_names": metadata["feature_names"],
              "terrain_dem": metadata["terrain_dem"],
              "dedupe_train_groups": bool(args.dedupe_train_groups),
              "cv_preprocessing": "StandardScaler fit on each fold's training rows only",
              "balance_mode": args.balance_mode,
              "class_weight_policy": "Fit independently from each fold's training rows; final fit uses train only",
              "final_train_class_weights": classifier_class_weights(ytr, args.balance_mode),
              "models": {}}

    n_splits = min(args.folds, max(2, int(np.bincount(ytr, minlength=4)[np.bincount(ytr, minlength=4) > 0].min())))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    # ---- LR：对 C 做分层分组 CV ----
    c_grid = [float(c) for c in args.c_grid.split(",") if c.strip()]
    best_c, best_score = c_grid[0], -1.0
    cv_rows = []
    for c in c_grid:
        scores = []
        for tr_i, te_i in splitter.split(Xtr_raw, ytr, groups=gtr):
            fold_train, fold_held, _fold_scaler = scaled_fold_features(Xtr_raw, tr_i, te_i)
            # 注意：scikit-learn>=1.7 已移除 LogisticRegression 的 multi_class 参数，
            # 多分类默认即 multinomial（softmax），不要传该参数，否则 TypeError。
            clf = LogisticRegression(C=c, class_weight=classifier_class_weights(ytr[tr_i], args.balance_mode),
                                     max_iter=args.max_iter)
            clf.fit(fold_train, ytr[tr_i])
            scores.append(f1_score(ytr[te_i], clf.predict(fold_held), average="macro"))
        mean_f1 = float(np.mean(scores))
        cv_rows.append({"C": c, "cv_macro_f1": round(mean_f1, 4),
                        "folds": [round(float(s), 4) for s in scores]})
        if mean_f1 > best_score:
            best_c, best_score = c, mean_f1
    report["models"]["lr"] = {"best_C": best_c, "cv_macro_f1": round(best_score, 4),
                              "cv_grid": cv_rows}
    lr = LogisticRegression(C=best_c, class_weight=classifier_class_weights(ytr, args.balance_mode),
                            max_iter=args.max_iter).fit(Xtr, ytr)
    joblib.dump({"model": lr, "scaler": scaler, "kind": "lr", "classes": CLASS_NAMES,
                 "feature_mode": metadata["feature_mode"],
                 "feature_names": metadata["feature_names"]},
                out_dir / "stage2_lr.joblib")

    gbdt = None
    if not args.skip_gbdt:
        # ---- GBDT：sample_weight 做类别平衡 ----
        sw = classification_sample_weights(ytr, args.balance_mode)
        gbdt = HistGradientBoostingClassifier(max_iter=args.gbdt_iter, learning_rate=0.1,
                                              random_state=args.seed).fit(Xtr, ytr, sample_weight=sw)
        gb_scores = []
        for tr_i, te_i in splitter.split(Xtr_raw, ytr, groups=gtr):
            fold_train, fold_held, _fold_scaler = scaled_fold_features(Xtr_raw, tr_i, te_i)
            g = HistGradientBoostingClassifier(max_iter=args.gbdt_iter, learning_rate=0.1,
                                               random_state=args.seed)
            g.fit(fold_train, ytr[tr_i],
                  sample_weight=classification_sample_weights(ytr[tr_i], args.balance_mode))
            gb_scores.append(f1_score(ytr[te_i], g.predict(fold_held), average="macro"))
        report["models"]["gbdt"] = {"cv_macro_f1": round(float(np.mean(gb_scores)), 4),
                                    "folds": [round(float(s), 4) for s in gb_scores]}
        joblib.dump({"model": gbdt, "scaler": scaler, "kind": "gbdt", "classes": CLASS_NAMES,
                     "feature_mode": metadata["feature_mode"],
                     "feature_names": metadata["feature_names"]},
                    out_dir / "stage2_gbdt.joblib")

    # ---- 对照：原型 / kNN ----
    proto_va = prototype_scores(Xtr, ytr, Xva if has_val else Xtr)
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine").fit(Xtr, ytr)
    np.savez_compressed(out_dir / "stage2_proto.npz", scaler_mean=scaler.mean_,
                        scaler_scale=scaler.scale_)

    # ---- 评估（val 优先，无 val 则退回 train 作为自检并显式标注）----
    eval_X, eval_y, eval_tag = ((Xva, yva, "val") if has_val else (Xtr, ytr, "train_selfcheck"))
    eval_sets = {
        "lr": lr.predict(eval_X),
        "prototype": np.argmax(proto_va, axis=1) if has_val else
        np.argmax(prototype_scores(Xtr, ytr, Xtr), axis=1),
        "knn": knn.predict(eval_X),
    }
    proba_sets = {
        "lr": lr.predict_proba(eval_X),
        "prototype": prototype_scores(Xtr, ytr, eval_X),
    }
    if gbdt is not None:
        eval_sets["gbdt"] = gbdt.predict(eval_X)
        proba_sets["gbdt"] = gbdt.predict_proba(eval_X)
    for name, pred in eval_sets.items():
        per_class, macro_f1, acc, cm = prf1(eval_y, pred)
        entry = {"eval_on": eval_tag, "per_class": per_class, "macro_f1": round(macro_f1, 4),
                 "accuracy": acc, "confusion_matrix": cm.tolist()}
        if name in proba_sets:
            entry["ovr_auc"] = ovr_auc(proba_sets[name], eval_y)
            entry["sl_cl_auc"] = sl_cl_auc(proba_sets[name], eval_y)
        report["models"].setdefault(name, {})
        report["models"][name].update(entry)

    report_path = out_dir / "stage2_cv_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("lr", "gbdt", "prototype", "knn"):
        if name not in report["models"]:
            continue
        m = report["models"][name]
        print(f"[{name}] eval_on={m.get('eval_on','cv')} macroF1={m.get('macro_f1')} "
              f"acc={m.get('accuracy')} SL_CL_AUC={m.get('sl_cl_auc')}")
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
