# -*- coding: utf-8 -*-
"""Zero-cost evaluation of an existing checkpoint: recall by class and by scale.

Runs inference (optionally with TTA and multi-scale merging) on a YOLO split,
matches predictions to ground truth with a class-aware greedy IoU matcher, and
reports recall stratified by class, by ground-truth box size and by source tile.

No training is involved; this only reads a trained best.pt and the dataset.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

CLASS_NAMES = ("SL", "SP", "FL", "CL")
SIZE_BINS = ("tiny_lt64", "small_64_256", "medium_256_819", "large_ge819")
CONF_GRID = (0.001, 0.01, 0.05, 0.1, 0.25, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/detect/yolov8n_mars_full23/weights/best.pt"),
    )
    parser.add_argument("--data", type=Path, default=Path("data/processed/yolo/data.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument(
        "--class-agnostic",
        action="store_true",
        help="ignore predicted/ground-truth class during matching; measure candidate recall only",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--modes",
        default="base,tta,multiscale",
        help="comma separated subset of base,tta,multiscale",
    )
    parser.add_argument(
        "--scales",
        default="1024,1280",
        help="image sizes merged in multiscale mode",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/eval_recall_report.json"))
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only evaluate the first N images (0 = all), useful for a smoke test",
    )
    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.05,
        help="operating detection threshold used to define a 'tile-candidate' miss",
    )
    parser.add_argument(
        "--dump-unmatched",
        type=Path,
        default=None,
        help="write a per-GT diagnosis of the boxes missing at --det-conf (base mode)",
    )
    parser.add_argument(
        "--tau-sweep",
        action="store_true",
        help="sweep the Stage1 operating threshold and report recall/precision trade-off",
    )
    parser.add_argument(
        "--dump-suppressed",
        type=Path,
        default=None,
        help="write the GT boxes lost solely to the class-agnostic NMS (base mode)",
    )
    parser.add_argument(
        "--taus",
        default="0.001,0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.25",
        help="comma separated thresholds used by --tau-sweep",
    )
    return parser.parse_args()


def resolve_dataset_root(data_yaml: Path) -> Path:
    """Resolve the dataset root the same way ultralytics does for a local yaml."""
    root = data_yaml.parent
    if not (root / "images").exists():
        root = data_yaml.parent.parent
    return root


def load_images(root: Path, split: str) -> list[Path]:
    image_dir = root / "images" / split
    images = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"no images under {image_dir}")
    return images


def load_ground_truth(
    root: Path, split: str, images: list[Path], patch_px: int = 1024
) -> dict[str, list[tuple[int, float, float, float, float]]]:
    """Return {stem: [(cls, x0, y0, x1, y1) in pixel coords]}."""
    label_dir = root / "labels" / split
    gt: dict[str, list[tuple[int, float, float, float, float]]] = {}
    for image in images:
        rows: list[tuple[int, float, float, float, float]] = []
        path = label_dir / f"{image.stem}.txt"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                fields = line.split()
                cls = int(fields[0])
                cx, cy, w, h = (float(x) for x in fields[1:5])
                rows.append(
                    (
                        cls,
                        (cx - w / 2) * patch_px,
                        (cy - h / 2) * patch_px,
                        (cx + w / 2) * patch_px,
                        (cy + h / 2) * patch_px,
                    )
                )
        gt[image.stem] = rows
    return gt


@torch.no_grad()
def infer(
    weights: Path,
    images: list[Path],
    imgsz: int,
    conf: float,
    device: str,
    augment: bool,
    nms_iou: float = 0.7,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Run prediction and return {stem: (boxes_xyxy, scores, classes)}.

    The images are handed to ultralytics through a .txt manifest on purpose: a
    python list source would be routed to LoadPilAndNumpy, which loads every
    image into a single batch and blows up the GPU memory. A .txt manifest uses
    LoadImagesAndVideos, which honours the `batch` argument.
    """
    import os  # pylint: disable=import-outside-toplevel
    import tempfile  # pylint: disable=import-outside-toplevel

    from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

    fd, tmp_name = tempfile.mkstemp(prefix="eval_recall_", suffix=".txt")
    os.close(fd)
    manifest = Path(tmp_name)
    manifest.write_text("\n".join(str(p) for p in images), encoding="utf-8")
    try:
        model = YOLO(str(weights))
        results = model.predict(
            source=str(manifest),
            imgsz=imgsz,
            conf=conf,
            iou=nms_iou,
            augment=augment,
            device=device,
            verbose=False,
            stream=True,
            batch=1,
        )
        out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for res in results:
            path = Path(res.path)
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                out[path.stem] = (
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                )
                continue
            out[path.stem] = (
                boxes.xyxy.cpu().numpy().astype(np.float32),
                boxes.conf.cpu().numpy().astype(np.float32),
                boxes.cls.cpu().numpy().astype(np.float32),
            )
        del model
        torch.cuda.empty_cache()
        return out
    finally:
        manifest.unlink(missing_ok=True)


def merge_nms(
    per_image: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    iou: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate predictions from several runs and apply class-aware NMS."""
    from torchvision.ops import batched_nms  # pylint: disable=import-outside-toplevel

    boxes = np.concatenate([p[0] for p in per_image], axis=0)
    scores = np.concatenate([p[1] for p in per_image], axis=0)
    classes = np.concatenate([p[2] for p in per_image], axis=0)
    if boxes.shape[0] == 0:
        return boxes, scores, classes
    keep = batched_nms(
        torch.from_numpy(boxes),
        torch.from_numpy(scores),
        torch.from_numpy(classes).long(),
        iou,
    ).numpy()
    return boxes[keep], scores[keep], classes[keep]


def iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if gt.shape[0] == 0 or pred.shape[0] == 0:
        return np.zeros((gt.shape[0], pred.shape[0]), dtype=np.float32)
    lt = np.maximum(gt[:, None, :2], pred[None, :, :2])
    rb = np.minimum(gt[:, None, 2:], pred[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_gt = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    area_pr = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    union = area_gt[:, None] + area_pr[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def match_image(
    gt: list[tuple[int, float, float, float, float]],
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_cls: np.ndarray,
    match_iou: float,
    conf: float,
    class_aware: bool = True,
) -> list[tuple[int, bool, str, float, int]]:
    """Greedy matching.

    Returns one record per ground-truth box: (cls, matched, size_bin, best_iou,
    matched_box_index). The index refers to the arrays after the conf filter and
    is -1 when the box is unmatched. With class_aware=False the class is ignored,
    which measures the candidate pool a downstream re-classifier could draw on.
    """
    records: list[tuple[int, bool, str, float, int]] = []
    if not gt:
        return records
    gt_arr = np.array([[g[1], g[2], g[3], g[4]] for g in gt], dtype=np.float32)
    sizes = np.maximum(gt_arr[:, 2] - gt_arr[:, 0], gt_arr[:, 3] - gt_arr[:, 1])

    def size_bin(px: float) -> str:
        if px < 64:
            return "tiny_lt64"
        if px < 256:
            return "small_64_256"
        if px < 819:
            return "medium_256_819"
        return "large_ge819"

    keep = pred_scores >= conf
    pb, ps, pc = pred_boxes[keep], pred_scores[keep], pred_cls[keep]
    used = np.zeros(pb.shape[0], dtype=bool)
    ious = iou_matrix(gt_arr, pb)

    order = np.argsort(-ps)
    for g_idx, g in enumerate(gt):
        cls = g[0]
        best_iou = 0.0
        best_j = -1
        for j in order:
            if used[j] or (class_aware and int(pc[j]) != cls):
                continue
            if ious[g_idx, j] > best_iou:
                best_iou = ious[g_idx, j]
                best_j = j
                if best_iou >= match_iou:
                    break
        matched = best_iou >= match_iou
        if matched and best_j >= 0:
            used[best_j] = True
        records.append((cls, matched, size_bin(sizes[g_idx]), float(best_iou),
                        best_j if matched else -1))
    return records


def evaluate(
    gt: dict[str, list[tuple[int, float, float, float, float]]],
    preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    match_iou: float,
    class_aware: bool = True,
) -> dict:
    """Aggregate recall by class, size bin, tile and confidence threshold."""
    totals = Counter()
    hits = Counter()
    size_total = Counter()
    size_hits = Counter()
    class_size_total: dict[str, Counter] = defaultdict(Counter)
    class_size_hits: dict[str, Counter] = defaultdict(Counter)
    tile_total: Counter = Counter()
    tile_hits: Counter = Counter()
    class_tile_total: dict[str, Counter] = defaultdict(Counter)
    class_tile_hits: dict[str, Counter] = defaultdict(Counter)
    conf_hits = {c: 0 for c in CONF_GRID}
    conf_pred_boxes = {c: 0 for c in CONF_GRID}
    conf_total = 0
    near_miss = 0

    for stem, rows in gt.items():
        boxes, scores, cls = preds.get(stem, (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)))
        tile = stem.split("_")
        tile_name = "_".join(tile[:2]) if len(tile) >= 2 else stem
        records = match_image(rows, boxes, scores, cls, match_iou, 0.001, class_aware=class_aware)
        for (cls_id, matched, sbin, biou, _j) in records:
            name = CLASS_NAMES[cls_id]
            totals[name] += 1
            size_total[sbin] += 1
            class_size_total[name][sbin] += 1
            tile_total[tile_name] += 1
            class_tile_total[name][tile_name] += 1
            conf_total += 1
            if matched:
                hits[name] += 1
                size_hits[sbin] += 1
                class_size_hits[name][sbin] += 1
                tile_hits[tile_name] += 1
                class_tile_hits[name][tile_name] += 1
            elif biou >= 0.1:
                near_miss += 1

    for conf in CONF_GRID:
        cnt = 0
        for stem, rows in gt.items():
            boxes, scores, cls = preds.get(stem, (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)))
            conf_pred_boxes[conf] += int(np.count_nonzero(scores >= conf))
            recs = match_image(rows, boxes, scores, cls, match_iou, conf, class_aware=class_aware)
            cnt += sum(1 for r in recs if r[1])
        conf_hits[conf] = cnt

    def ratio(h: int, t: int) -> float:
        return round(h / t, 4) if t else 0.0

    per_class = {
        name: {
            "gt": totals[name],
            "tp": hits[name],
            "recall": ratio(hits[name], totals[name]),
            "by_size": {
                sbin: {
                    "gt": class_size_total[name][sbin],
                    "tp": class_size_hits[name][sbin],
                    "recall": ratio(class_size_hits[name][sbin], class_size_total[name][sbin]),
                }
                for sbin in SIZE_BINS
            },
        }
        for name in CLASS_NAMES
    }
    tiles = {
        tile: {
            "gt": tile_total[tile],
            "tp": tile_hits[tile],
            "recall": ratio(tile_hits[tile], tile_total[tile]),
        }
        for tile in sorted(tile_total)
    }
    tiles_by_recall = sorted(tiles.items(), key=lambda kv: (kv[1]["recall"], -kv[1]["gt"]))
    return {
        "overall": {
            "gt": conf_total,
            "tp": sum(hits.values()),
            "recall": ratio(sum(hits.values()), conf_total),
            "near_miss_unmatched_iou_ge_0.1": near_miss,
        },
        "recall_by_conf": {str(c): ratio(v, conf_total) for c, v in conf_hits.items()},
        "pred_boxes_by_conf": {str(c): conf_pred_boxes[c] for c in CONF_GRID},
        "precision_by_conf": {
            str(c): ratio(conf_hits[c], conf_pred_boxes[c]) for c in CONF_GRID
        },
        "false_positives_by_conf": {
            str(c): max(conf_pred_boxes[c] - conf_hits[c], 0) for c in CONF_GRID
        },
        "per_class": per_class,
        "by_size": {
            sbin: {
                "gt": size_total[sbin],
                "tp": size_hits[sbin],
                "recall": ratio(size_hits[sbin], size_total[sbin]),
            }
            for sbin in SIZE_BINS
        },
        "by_tile": tiles,
        "worst_tiles": [{"tile": t, **v} for t, v in tiles_by_recall[:15]],
        "best_tiles": [{"tile": t, **v} for t, v in tiles_by_recall[::-1][:10]],
    }


def diagnose_unmatched(
    gt: dict[str, list[tuple[int, float, float, float, float]]],
    preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    match_iou: float,
    det_conf: float,
    patch_px: int = 1024,
    edge_px: float = 8.0,
) -> dict:
    """Explain every GT box that has no matched detection at the operating threshold.

    Each record carries the raw evidence needed to separate the competing causes:
    a truly empty patch, a box of the wrong class only, a same-class box that is
    close but badly localized, a same-class box whose confidence is below the
    operating threshold, and a same-class box that is both confident and far away.
    """
    empty = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32))
    records: list[dict] = []

    for stem, rows in gt.items():
        if not rows:
            continue
        boxes, scores, cls = preds.get(stem, empty)
        rec001 = match_image(rows, boxes, scores, cls, match_iou, 0.001)
        recdet = match_image(rows, boxes, scores, cls, match_iou, det_conf)
        gt_arr = np.array([[g[1], g[2], g[3], g[4]] for g in rows], dtype=np.float32)
        tile = "_".join(stem.split("_")[:2])
        max_score_any = float(scores.max()) if scores.size else 0.0

        for i, g in enumerate(rows):
            if recdet[i][1]:
                continue
            cls_id = g[0]
            box = gt_arr[i]
            same = cls == cls_id if cls.size else np.zeros(0, bool)
            if same.any():
                sb, ss = boxes[same], scores[same]
                ious = iou_matrix(box[None, :], sb)[0]
                best_iou = float(ious.max())
                best_score_same = float(ss[int(ious.argmax())])
                max_score_same = float(ss.max())
            else:
                best_iou = 0.0
                best_score_same = 0.0
                max_score_same = 0.0

            if boxes.shape[0]:
                ious_any = iou_matrix(box[None, :], boxes)[0]
                j = int(ious_any.argmax())
                best_iou_any = float(ious_any[j])
                cls_at_best_iou = CLASS_NAMES[int(cls[j])]
                score_at_best_iou = float(scores[j])
            else:
                best_iou_any = 0.0
                cls_at_best_iou = None
                score_at_best_iou = 0.0

            if rec001[i][1]:
                cause = "below_det_conf"
            elif boxes.shape[0] == 0:
                cause = "patch_no_detection"
            elif best_iou_any >= match_iou and score_at_best_iou >= det_conf:
                cause = "class_confusion"
            elif max_score_same == 0.0:
                cause = "no_same_class_box"
            elif best_iou >= 0.1:
                cause = "near_miss_localization"
            elif max_score_same < det_conf:
                cause = "same_class_low_conf"
            else:
                cause = "same_class_far_box"

            records.append(
                {
                    "stem": stem,
                    "tile": tile,
                    "cls": CLASS_NAMES[cls_id],
                    "size_px": round(float(max(box[2] - box[0], box[3] - box[1])), 1),
                    "size_bin": recdet[i][2],
                    "edge_touch": bool(
                        box[0] <= edge_px
                        or box[1] <= edge_px
                        or box[2] >= patch_px - edge_px
                        or box[3] >= patch_px - edge_px
                    ),
                    "matched_at_conf_0.001": bool(rec001[i][1]),
                    "cause": cause,
                    "best_iou_same_class": round(best_iou, 4),
                    "best_score_same_class": round(best_score_same, 4),
                    "max_score_same_class": round(max_score_same, 4),
                    "max_score_any_class": round(max_score_any, 4),
                    "best_iou_any_class": round(best_iou_any, 4),
                    "cls_at_best_iou": cls_at_best_iou,
                    "score_at_best_iou": round(score_at_best_iou, 4),
                    "n_pred_in_patch": int(boxes.shape[0]),
                }
            )

    def group(key: str) -> dict:
        out: Counter = Counter()
        for r in records:
            out[r[key]] += 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    cross_class: dict[str, Counter] = defaultdict(Counter)
    cross_tile: dict[str, Counter] = defaultdict(Counter)
    cross_cause: dict[str, Counter] = defaultdict(Counter)
    cross_size: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        cross_class[r["cls"]][r["cause"]] += 1
        cross_tile[r["tile"]][r["cause"]] += 1
        cross_cause[r["cause"]][r["cls"]] += 1
        cross_size[r["size_bin"]][r["cause"]] += 1

    return {
        "det_conf": det_conf,
        "match_iou": match_iou,
        "n_unmatched": len(records),
        "n_unmatched_at_0.001": sum(1 for r in records if not r["matched_at_conf_0.001"]),
        "by_cause": group("cause"),
        "by_class": group("cls"),
        "by_tile": group("tile"),
        "by_size_bin": group("size_bin"),
        "edge_touch": sum(1 for r in records if r["edge_touch"]),
        "cause_by_class": {k: dict(v) for k, v in sorted(cross_class.items())},
        "cause_by_tile": {k: dict(v) for k, v in sorted(cross_tile.items())},
        "class_by_cause": {k: dict(v) for k, v in sorted(cross_cause.items())},
        "cause_by_size_bin": {k: dict(v) for k, v in sorted(cross_size.items())},
        "records": records,
    }


def class_agnostic_nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Flatten classes for NMS and return the kept indices.

    This mirrors eval_cascade.py's Stage1 candidate extraction, so a threshold
    swept here is directly comparable with the cascade's recorded numbers.
    """
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype=int)
    from torchvision.ops import nms  # pylint: disable=import-outside-toplevel

    keep = nms(
        torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(scores, dtype=np.float32)),
        iou,
    )
    return keep.numpy().astype(int)


def tau_sweep(
    gt: dict[str, list[tuple[int, float, float, float, float]]],
    preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    match_iou: float,
    nms_iou: float,
    taus: Iterable[float],
) -> list[dict]:
    """Recall / precision trade-off across the Stage1 operating threshold.

    Reports three quantities per threshold:
      * class-aware recall after the cascade's class-agnostic NMS (the pipeline
        as it actually runs);
      * the same matching without the class-agnostic NMS, which isolates how many
        ground-truth boxes that NMS suppresses by keeping a wrong-class box;
      * class-agnostic candidate recall, i.e. the ceiling a downstream
        re-classifier could reach on the surviving boxes.
    """
    empty = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32))
    n_gt = sum(len(rows) for rows in gt.values())
    n_patches = len(preds)
    rows: list[dict] = []

    def ratio(h: int, t: int) -> float:
        return round(h / t, 4) if t else 0.0

    for tau in taus:
        n_boxes = n_boxes_no_anms = 0
        tp = tp_no_anms = tp_pool = 0
        per_class: Counter = Counter()
        for stem, gt_rows in gt.items():
            boxes, scores, cls = preds.get(stem, empty)
            if boxes.shape[0] == 0:
                continue
            sel_no_anms = scores >= tau
            n_boxes_no_anms += int(sel_no_anms.sum())
            for rec in match_image(
                gt_rows, boxes[sel_no_anms], scores[sel_no_anms], cls[sel_no_anms],
                match_iou, tau,
            ):
                tp_no_anms += int(rec[1])
            keep = class_agnostic_nms(boxes, scores, nms_iou)
            sel = scores[keep] >= tau
            kb, ks, kc = boxes[keep][sel], scores[keep][sel], cls[keep][sel]
            n_boxes += int(kb.shape[0])
            for rec in match_image(gt_rows, kb, ks, kc, match_iou, tau):
                if rec[1]:
                    tp += 1
                    per_class[CLASS_NAMES[rec[0]]] += 1
            tp_pool += sum(
                1 for r in match_image(gt_rows, kb, ks, kc, match_iou, tau, class_aware=False) if r[1]
            )
        rows.append(
            {
                "tau": tau,
                "gt": n_gt,
                "n_boxes": n_boxes,
                "boxes_per_patch": round(n_boxes / n_patches, 2) if n_patches else 0.0,
                "tp_class_aware": tp,
                "recall": ratio(tp, n_gt),
                "fp_strict": n_boxes - tp,
                "precision_strict": ratio(tp, n_boxes),
                "tp_without_agnostic_nms": tp_no_anms,
                "recall_without_agnostic_nms": ratio(tp_no_anms, n_gt),
                "n_boxes_without_agnostic_nms": n_boxes_no_anms,
                "tp_candidate_pool": tp_pool,
                "recall_candidate_pool": ratio(tp_pool, n_gt),
                "per_class_tp": {c: per_class[c] for c in CLASS_NAMES},
            }
        )
    return rows


def diagnose_suppression(
    gt: dict[str, list[tuple[int, float, float, float, float]]],
    preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    match_iou: float,
    nms_iou: float,
    det_conf: float,
) -> dict:
    """List the ground-truth boxes lost purely to the class-agnostic NMS.

    A GT box counts as lost when it is matched by a box that survives the plain
    per-class NMS but not by any box that survives the additional class-agnostic
    NMS. For each one the suppressing box (the higher-scoring box that flattened
    it) is reported, so the loss can be attributed to a specific class pair.
    """
    empty = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32))
    records: list[dict] = []

    for stem, rows in gt.items():
        if not rows:
            continue
        boxes, scores, cls = preds.get(stem, empty)
        if boxes.shape[0] == 0:
            continue
        sel = scores >= det_conf
        sb, ss, sc = boxes[sel], scores[sel], cls[sel]
        rec_no = match_image(rows, sb, ss, sc, match_iou, 0.0)
        keep = class_agnostic_nms(boxes, scores, nms_iou)
        kb, ks, kc = boxes[keep], scores[keep], cls[keep]
        rec_anms = match_image(rows, kb, ks, kc, match_iou, det_conf)
        tile = "_".join(stem.split("_")[:2])
        ious_in = iou_matrix(sb, kb)

        for i, row in enumerate(rows):
            if not rec_no[i][1] or rec_anms[i][1]:
                continue
            j = rec_no[i][4]
            box = sb[j]
            dup = np.where(ious_in[j] > 0.0)[0]
            sup = int(dup[ious_in[j][dup].argmax()]) if dup.size else -1
            records.append(
                {
                    "stem": stem,
                    "tile": tile,
                    "cls": CLASS_NAMES[row[0]],
                    "size_px": round(float(max(box[2] - box[0], box[3] - box[1])), 1),
                    "size_bin": rec_no[i][2],
                    "matched_box_cls": CLASS_NAMES[int(sc[j])],
                    "matched_box_score": round(float(ss[j]), 4),
                    "matched_box_iou": round(float(rec_no[i][3]), 4),
                    "suppressor_cls": CLASS_NAMES[int(kc[sup])] if sup >= 0 else None,
                    "suppressor_score": round(float(ks[sup]), 4) if sup >= 0 else None,
                    "suppressor_iou_with_matched_box": (
                        round(float(ious_in[j][sup]), 4) if sup >= 0 else None
                    ),
                }
            )

    pairs: Counter = Counter()
    for r in records:
        pairs[f"{r['cls']}<-{r['suppressor_cls']}"] += 1

    def group(key: str) -> dict:
        out: Counter = Counter()
        for r in records:
            out[r[key]] += 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    return {
        "det_conf": det_conf,
        "nms_iou": nms_iou,
        "n_lost": len(records),
        "by_class": group("cls"),
        "by_tile": group("tile"),
        "by_size_bin": group("size_bin"),
        "gt_class_to_suppressor": dict(sorted(pairs.items(), key=lambda kv: -kv[1])),
        "records": records,
    }


def main() -> None:
    args = parse_args()
    root = resolve_dataset_root(args.data)
    images = load_images(root, args.split)
    if args.limit:
        images = images[: args.limit]
    gt = load_ground_truth(root, args.split, images)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    scales = [int(s) for s in args.scales.split(",") if s.strip()]

    report: dict = {
        "weights": str(args.weights),
        "split": args.split,
        "images": len(images),
        "match_iou": args.match_iou,
        "class_aware": not args.class_agnostic,
        "modes": {},
    }

    for mode in modes:
        if mode == "base":
            runs = [infer(args.weights, images, args.imgsz, args.conf, args.device, False)]
            preds = {p.stem: runs[0][p.stem] for p in images}
        elif mode == "tta":
            runs = [infer(args.weights, images, args.imgsz, args.conf, args.device, True)]
            preds = {p.stem: runs[0][p.stem] for p in images}
        elif mode == "multiscale":
            runs = [
                infer(args.weights, images, s, args.conf, args.device, False) for s in scales
            ]
            preds = {
                p.stem: merge_nms([r[p.stem] for r in runs], args.nms_iou) for p in images
            }
        else:
            raise ValueError(f"unknown mode {mode}")
        report["modes"][mode] = evaluate(
            gt, preds, args.match_iou, class_aware=not args.class_agnostic
        )
        overall = report["modes"][mode]["overall"]
        print(
            f"[{mode}] gt={overall['gt']} tp={overall['tp']} "
            f"recall={overall['recall']:.3f} "
            f"near_miss={overall['near_miss_unmatched_iou_ge_0.1']}"
        )
        for name in CLASS_NAMES:
            pc = report["modes"][mode]["per_class"][name]
            print(
                f"    {name}: gt={pc['gt']:4d} tp={pc['tp']:4d} recall={pc['recall']:.3f}"
            )
        if mode == "base" and args.dump_unmatched:
            diag = diagnose_unmatched(gt, preds, args.match_iou, args.det_conf)
            report["modes"][mode]["unmatched_diagnosis"] = {
                k: v for k, v in diag.items() if k != "records"
            }
            args.dump_unmatched.parent.mkdir(parents=True, exist_ok=True)
            args.dump_unmatched.write_text(
                json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"    unmatched@{args.det_conf}: {diag['n_unmatched']} "
                f"(no box even at 0.001: {diag['n_unmatched_at_0.001']}, "
                f"edge_touch: {diag['edge_touch']})"
            )
            for cause, cnt in diag["by_cause"].items():
                print(f"      {cause}: {cnt}")
            print(f"unmatched diagnosis -> {args.dump_unmatched}")
        if mode == "base" and args.dump_suppressed:
            lost = diagnose_suppression(gt, preds, args.match_iou, args.nms_iou, args.det_conf)
            report["modes"][mode]["suppressed_by_agnostic_nms"] = {
                k: v for k, v in lost.items() if k != "records"
            }
            args.dump_suppressed.parent.mkdir(parents=True, exist_ok=True)
            args.dump_suppressed.write_text(
                json.dumps(lost, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"    lost to agnostic NMS@{args.det_conf}: {lost['n_lost']} "
                f"| {lost['gt_class_to_suppressor']}"
            )
            print(f"suppression diagnosis -> {args.dump_suppressed}")
        if mode == "base" and args.tau_sweep:
            taus = [float(t) for t in args.taus.split(",") if t.strip()]
            sweep = tau_sweep(gt, preds, args.match_iou, args.nms_iou, taus)
            report["modes"][mode]["tau_sweep"] = sweep
            print(
                f"    {'tau':>6} {'boxes':>6} {'recall':>7} {'prec':>6} "
                f"{'pool':>6} {'no_anms':>8} {'SL/SP/FL/CL tp'}"
            )
            for row in sweep:
                pc = row["per_class_tp"]
                print(
                    f"    {row['tau']:>6} {row['n_boxes']:>6} {row['recall']:>7.4f} "
                    f"{row['precision_strict']:>6.4f} {row['recall_candidate_pool']:>6.4f} "
                    f"{row['recall_without_agnostic_nms']:>8.4f} "
                    f"{pc['SL']}/{pc['SP']}/{pc['FL']}/{pc['CL']}"
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
