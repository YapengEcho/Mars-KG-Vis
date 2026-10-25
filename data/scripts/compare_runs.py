# -*- coding: utf-8 -*-
"""Unified-protocol comparison across YOLO runs.

Evaluates several checkpoints under one fixed protocol (original val split, no
TTA, fixed imgsz/conf/IoU, one shared matcher) and pairs the numbers with each
run's training progress (epochs, optimizer steps, samples seen, wall-clock time,
best epoch and the metrics at that epoch). Per-epoch time is reported as a robust
median, and any suspended/stalled interval (e.g. machine sleep or a paused
process) is detected and excluded, so an interrupted run is not mistaken for a
slow one. Runs trained on different sampling can then be compared both per-epoch
and per-exposure instead of per-epoch only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import eval_recall_report as err

CLASS_NAMES = err.CLASS_NAMES
SIZE_BINS = err.SIZE_BINS
EMPTY = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32))


def parse_run_spec(spec: str) -> tuple[str, Path, int | None, int | None]:
    name, _, rest = spec.partition("=")
    weights, _, meta = rest.partition("|")
    train_images = batches = None
    if meta:
        parts = [p.strip() for p in meta.split(",")]
        if parts and parts[0]:
            train_images = int(parts[0])
        if len(parts) > 1 and parts[1]:
            batches = int(parts[1])
    return name.strip(), Path(weights.strip()), train_images, batches


def read_progress(run_dir: Path, train_images: int | None, batches: int | None) -> dict | None:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None
    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None

    def num(row: dict, key: str) -> float:
        try:
            return float(row.get(key, "nan"))
        except (TypeError, ValueError):
            return float("nan")

    epochs = len(rows)
    times = [num(r, "time") for r in rows]
    total_time = times[-1]
    per_epoch = [times[0]] + [times[i] - times[i - 1] for i in range(1, epochs)]
    median_epoch = float(np.median(per_epoch))
    stall_threshold = max(10.0 * median_epoch, 300.0)
    stalls = [d for d in per_epoch if d > stall_threshold]
    stall_total = float(sum(stalls))
    active_time = total_time - stall_total
    best50 = max(rows, key=lambda r: num(r, "metrics/mAP50(B)"))
    best95 = max(rows, key=lambda r: num(r, "metrics/mAP50-95(B)"))
    last = rows[-1]
    progress = {
        "epochs": epochs,
        "train_time_s": round(total_time, 1),
        "train_time_h": round(total_time / 3600.0, 3),
        "active_time_s": round(active_time, 1),
        "active_time_h": round(active_time / 3600.0, 3),
        "stall_count": len(stalls),
        "stall_total_s": round(stall_total, 1),
        "stall_threshold_s": round(stall_threshold, 1),
        "sec_per_epoch": round(median_epoch, 1),
        "sec_per_epoch_mean": round(total_time / epochs, 1),
        "sec_per_epoch_max": round(max(per_epoch), 1),
        "best_epoch_mAP50": int(float(best50["epoch"])),
        "best_mAP50": round(num(best50, "metrics/mAP50(B)"), 4),
        "best_epoch_mAP50_95": int(float(best95["epoch"])),
        "best_mAP50_95": round(num(best95, "metrics/mAP50-95(B)"), 4),
        "last_epoch": {
            "epoch": int(float(last["epoch"])),
            "precision": round(num(last, "metrics/precision(B)"), 4),
            "recall": round(num(last, "metrics/recall(B)"), 4),
            "mAP50": round(num(last, "metrics/mAP50(B)"), 4),
            "mAP50_95": round(num(last, "metrics/mAP50-95(B)"), 4),
        },
        "loss_last": {
            "train_box": round(num(last, "train/box_loss"), 4),
            "val_box": round(num(last, "val/box_loss"), 4),
            "val_minus_train_box": round(num(last, "val/box_loss") - num(last, "train/box_loss"), 4),
            "train_cls": round(num(last, "train/cls_loss"), 4),
            "val_cls": round(num(last, "val/cls_loss"), 4),
        },
    }
    if train_images:
        progress["train_images_per_epoch"] = train_images
        progress["samples_seen"] = train_images * epochs
    if batches:
        progress["batches_per_epoch"] = batches
        progress["optimizer_steps"] = batches * epochs
    return progress


def custom_recall(
    weights: Path,
    data_yaml: Path,
    split: str,
    imgsz: int,
    conf: float,
    match_iou: float,
    device: str,
    limit: int,
) -> dict:
    root = err.resolve_dataset_root(data_yaml)
    images = err.load_images(root, split)
    if limit:
        images = images[:limit]
    gt = err.load_ground_truth(root, split, images, patch_px=1024)
    preds = err.infer(weights, images, imgsz, conf, device, False)
    report = err.evaluate(gt, preds, match_iou)

    total_pred = 0
    for stem in gt:
        _b, scores, _c = preds.get(stem, EMPTY)
        total_pred += int((scores >= conf).sum())
    tp = report["overall"]["tp"]
    fp = max(total_pred - tp, 0)
    report["images"] = len(images)
    report["overall"]["pred_boxes"] = total_pred
    report["overall"]["false_positives"] = fp
    report["overall"]["fp_rate"] = round(fp / total_pred, 4) if total_pred else 0.0
    report["overall"]["precision_at_conf"] = round(tp / total_pred, 4) if total_pred else 0.0
    return report


def ultralytics_val(
    weights: Path,
    data_yaml: Path,
    split: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    out_dir: Path,
    name: str,
) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    res = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        augment=False,
        plots=False,
        verbose=False,
        device=device,
        project=str(out_dir),
        name=name,
        exist_ok=True,
    )
    names = {int(k): v for k, v in res.names.items()}
    positions = {int(c): i for i, c in enumerate(res.box.ap_class_index)}
    per_class = {}
    for cls_id in sorted(names):
        pos = positions.get(cls_id)
        if pos is None:
            per_class[names[cls_id]] = {
                "precision": None, "recall": None, "mAP50": 0.0, "mAP50_95": 0.0,
            }
        else:
            per_class[names[cls_id]] = {
                "precision": round(float(res.box.p[pos]), 4),
                "recall": round(float(res.box.r[pos]), 4),
                "mAP50": round(float(res.box.ap50[pos]), 4),
                "mAP50_95": round(float(res.box.ap[pos]), 4),
            }
    return {
        "overall": {
            "precision": round(float(res.box.mp), 4),
            "recall": round(float(res.box.mr), 4),
            "mAP50": round(float(res.box.map50), 4),
            "mAP50_95": round(float(res.box.map), 4),
        },
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="spec as name=path/to/best.pt|train_images,batches",
    )
    parser.add_argument("--data", type=Path, default=Path("data/processed/yolo/data.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--skip-custom", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/compare_runs_report.json"))
    args = parser.parse_args()

    report: dict = {
        "protocol": {
            "data": str(args.data),
            "split": args.split,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "nms_iou": args.iou,
            "match_iou": args.match_iou,
            "tta": False,
            "matcher": "eval_recall_report.greedy_class_aware",
            "limit": args.limit,
        },
        "runs": {},
    }

    for spec in args.runs:
        name, weights, train_images, batches = parse_run_spec(spec)
        run_dir = weights.parent.parent
        entry: dict = {"weights": str(weights)}
        entry["progress"] = read_progress(run_dir, train_images, batches)
        if not args.skip_custom:
            entry["custom_recall"] = custom_recall(
                weights, args.data, args.split, args.imgsz, args.conf,
                args.match_iou, args.device, args.limit,
            )
        if not args.skip_val:
            entry["ultralytics_val"] = ultralytics_val(
                weights, args.data, args.split, args.imgsz, args.conf,
                args.iou, args.device, args.out.parent / "compare_val", name,
            )
        report["runs"][name] = entry

        progress = entry.get("progress") or {}
        print(f"\n== {name} ==")
        if progress:
            print(
                f"  epochs={progress['epochs']} "
                f"steps={progress.get('optimizer_steps', '-')} "
                f"samples={progress.get('samples_seen', '-')} "
                f"time_h={progress['train_time_h']} "
                f"active_h={progress['active_time_h']} "
                f"best_mAP50=e{progress['best_epoch_mAP50']}:{progress['best_mAP50']} "
                f"best_mAP50-95=e{progress['best_epoch_mAP50_95']}:{progress['best_mAP50_95']}"
            )
            print(
                f"  sec/epoch: median={progress['sec_per_epoch']} "
                f"mean={progress['sec_per_epoch_mean']} "
                f"max={progress['sec_per_epoch_max']} "
                f"| stalls={progress['stall_count']} "
                f"({progress['stall_total_s']}s)"
            )
            loss = progress.get("loss_last") or {}
            if loss:
                print(
                    f"  loss_last: train_box={loss['train_box']} val_box={loss['val_box']} "
                    f"gap={loss['val_minus_train_box']}"
                )
        custom = entry.get("custom_recall")
        if custom:
            ov = custom["overall"]
            print(
                f"  custom: recall={ov['recall']:.3f} "
                f"precision@{args.conf}={ov['precision_at_conf']:.3f} "
                f"fp_rate={ov['fp_rate']:.3f}"
            )
            for cls_name in CLASS_NAMES:
                pc = custom["per_class"][cls_name]
                print(f"    {cls_name}: gt={pc['gt']:4d} tp={pc['tp']:4d} recall={pc['recall']:.3f}")
        val = entry.get("ultralytics_val")
        if val:
            ov = val["overall"]
            print(
                f"  ultralytics: P={ov['precision']:.3f} R={ov['recall']:.3f} "
                f"mAP50={ov['mAP50']:.3f} mAP50-95={ov['mAP50_95']:.3f}"
            )
            for cls_name in CLASS_NAMES:
                pc = val["per_class"].get(cls_name)
                if pc:
                    print(
                        f"    {cls_name}: P={pc['precision']} R={pc['recall']} "
                        f"mAP50={pc['mAP50']} mAP50-95={pc['mAP50_95']}"
                    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
