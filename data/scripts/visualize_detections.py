# -*- coding: utf-8 -*-
"""Visual acceptance check: overlay ground truth and predictions on val tiles.

Reuses eval_recall_report for dataset resolution, inference and the class-aware
matcher, then renders one annotated tile per selected image. A ground-truth box
is drawn green when a same-class prediction reaches match_iou and red when it is
missed, while predictions keep their class colour, so rare-class (SP/FL/CL)
recall can be eyeballed instead of taken on trust from the aggregate numbers.
Candidates are spread across the size range so both easy and hard instances are
shown, and one contact sheet per class is written for a single-glance review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import eval_recall_report as err

CLASS_COLORS = {
    "SL": (255, 82, 82),
    "SP": (255, 214, 0),
    "FL": (0, 229, 255),
    "CL": (213, 0, 249),
}
GT_HIT = (0, 230, 118)
GT_MISS = (255, 23, 68)
EMPTY = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32))
LABEL_BG = (0, 0, 0)
SHEET_BG = (18, 18, 18)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/detect/yolov8n_mars_rebal_w2/weights/best.pt"),
    )
    parser.add_argument("--data", type=Path, default=Path("data/processed/yolo/data.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="confidence used both to draw and to decide a hit (keep readable)",
    )
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--classes", default="SP,FL,CL", help="comma separated target classes")
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument(
        "--stems-file",
        type=Path,
        default=None,
        help="text file with one image stem per line; when given, only those images are rendered",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/vis_rebal_w2"))
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color, font) -> None:
    x, y = xy
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle([left - 2, top - 1, right + 2, bottom + 1], fill=LABEL_BG)
    draw.text((x, y), text, fill=color, font=font)


def annotate(
    image_path: Path,
    rows: list[tuple[int, float, float, float, float]],
    pred: tuple[np.ndarray, np.ndarray, np.ndarray],
    match_iou: float,
    conf: float,
    font: ImageFont.ImageFont,
) -> tuple[Image.Image, dict]:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    boxes, scores, cls = pred
    records = err.match_image(rows, boxes, scores, cls, match_iou, conf)

    per_class: dict[str, dict[str, int]] = {}
    for gt_box, record in zip(rows, records):
        cls_id, matched, _size_bin, best_iou, _j = record
        name = err.CLASS_NAMES[cls_id]
        color = GT_HIT if matched else GT_MISS
        x0, y0, x1, y1 = gt_box[1], gt_box[2], gt_box[3], gt_box[4]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        tag = f"GT {name}" if matched else f"GT {name} MISS {best_iou:.2f}"
        draw_label(draw, (int(x0) + 2, max(0, int(y0) - 18)), tag, color, font)
        entry = per_class.setdefault(name, {"gt": 0, "hit": 0})
        entry["gt"] += 1
        entry["hit"] += int(matched)

    keep = scores >= conf
    for box, score, cls_id in zip(boxes[keep], scores[keep], cls[keep]):
        name = err.CLASS_NAMES[int(cls_id)]
        color = CLASS_COLORS[name]
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        y = min(int(y1) + 2, img.height - 20)
        draw_label(draw, (int(x0) + 2, y), f"{name} {score:.2f}", color, font)
    return img, per_class


def spread(items: list, count: int) -> list:
    if count <= 0 or len(items) <= count:
        return items
    idx = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in dict.fromkeys(idx.tolist())]


def make_sheet(items: list[Image.Image], cols: int, title: str, font) -> Image.Image | None:
    if not items:
        return None
    width, height = items[0].size
    rows_n = (len(items) + cols - 1) // cols
    header = 36
    sheet = Image.new("RGB", (cols * width, header + rows_n * height), SHEET_BG)
    ImageDraw.Draw(sheet).text((10, 8), title, fill=(255, 255, 255), font=font)
    for i, img in enumerate(items):
        r, c = divmod(i, cols)
        sheet.paste(img, (c * width, header + r * height))
    return sheet


def main() -> None:
    args = parse_args()
    root = err.resolve_dataset_root(args.data)
    images = err.load_images(root, args.split)
    if args.stems_file is not None:
        want = {
            line.strip()
            for line in args.stems_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        images = [img for img in images if img.stem in want]
        missing = sorted(want - {img.stem for img in images})
        if missing:
            raise SystemExit(f"{len(missing)} stems not found in split: {missing[:5]}")
    gt = err.load_ground_truth(root, args.split, images)
    preds = err.infer(args.weights, images, args.imgsz, args.conf, args.device, False)
    font = load_font(18)
    title_font = load_font(26)
    args.out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    for name in [c.strip() for c in args.classes.split(",") if c.strip()]:
        cls_id = err.CLASS_NAMES.index(name)
        candidates = []
        for image in images:
            mine = [g for g in gt.get(image.stem, []) if g[0] == cls_id]
            if not mine:
                continue
            biggest = max((g[3] - g[1]) * (g[4] - g[2]) for g in mine)
            candidates.append((biggest, image))
        candidates.sort(key=lambda t: t[0])
        chosen = [image for _area, image in spread(candidates, args.per_class)]

        class_dir = args.out / name
        class_dir.mkdir(parents=True, exist_ok=True)
        tiles: list[Image.Image] = []
        agg = {"images": len(chosen), "gt": 0, "hit": 0}
        for image in chosen:
            pred = preds.get(image.stem, EMPTY)
            img, per_class = annotate(
                image, gt.get(image.stem, []), pred, args.match_iou, args.conf, font
            )
            img.save(class_dir / f"{image.stem}.jpg", quality=92)
            tiles.append(img)
            entry = per_class.get(name, {"gt": 0, "hit": 0})
            agg["gt"] += entry["gt"]
            agg["hit"] += entry["hit"]

        recall = round(agg["hit"] / agg["gt"], 4) if agg["gt"] else 0.0
        agg["recall"] = recall
        summary[name] = agg
        title = f"{name}  images={agg['images']}  gt={agg['gt']}  hit={agg['hit']}  recall={recall:.3f}"
        sheet = make_sheet(tiles, args.cols, title, title_font)
        if sheet is not None:
            sheet.save(args.out / f"sheet_{name}.jpg", quality=90)
        print(f"[{name}] images={agg['images']} gt={agg['gt']} hit={agg['hit']} recall={recall:.3f}")

    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"visuals -> {args.out}")


if __name__ == "__main__":
    main()
