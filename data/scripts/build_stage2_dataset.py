# -*- coding: utf-8 -*-
"""构建 Stage2 级联分类器的训练数据集（冻结骨干特征 + 框几何 + Stage1 置信度）。

设计要点（对应 spec 2026-09-25-cl-cascade-design.md §2/§3）：

1. 样本来源 = **已生成的 YOLO patch**（data/processed/yolo/images/{split}/*.png）。
   理由：patch 已经过与训练完全一致的拉伸/缩放，且其所在目录即权威 split；
   Stage2 的划分因此与检测器**逐 patch 一致**，不需要重新推导连通分量，天然无泄漏。
2. 每个 GT 框裁一块带边距的正方形 crop（默认 margin=0.25）并 resize 到 256。
3. 特征 = [冻结骨干 GAP 拼接 448 维] ⊕ [框几何 5 维] ⊕ [Stage1 置信度 1 维] = 454 维。
   框几何与置信度都是**推理期可得的量**；真形状量（elongation/圆度/凸性）**禁止入特征**
   （推理时算不出，见 spec §3）。
4. train split 额外做框抖动增强（位置/尺度），缓解"训练用真值框、推理用预测框"的域差。
   抖动样本共享同一个 group id，CV 时必须整组进同一折。

用法：
    python build_stage2_dataset.py --limit 20 --out runs/stage2/smoke.npz   # 冒烟
    python build_stage2_dataset.py                                          # 全量
"""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

from stage2_features import FEATURE_MODES, TerrainExtractor, compose_features, feature_names
import terrain_features

BASE = Path(__file__).resolve().parents[2]
DATA_YAML = BASE / "data" / "processed" / "yolo" / "data.yaml"
WEIGHTS = BASE / "runs" / "detect" / "yolov8n_mars_rebal_w2" / "weights" / "best.pt"
OUT = BASE / "runs" / "stage2" / "stage2_dataset.npz"
TERRAIN_DEM = BASE / "data" / "raw" / "terrain" / "Mars_MGS_MOLA_DEM_mosaic_global_463m_parallel.tif"

PATCH_PX = 1024

CLASS_NAMES = ("SL", "SP", "FL", "CL")
BACKBONE_LAYERS = (15, 18, 21)
BACKBONE_DIM = 448          # 64 + 128 + 256，GAP 后拼接
GEOM_DIM = 5
N_FEATURES = BACKBONE_DIM + GEOM_DIM + 1

MAX_ASPECT = 64.0           # 长宽比上限，防极端细长框把特征拉爆
JITTER_SHIFT = 0.08         # 位置抖动幅度（占框长边比例）
JITTER_SCALE = 0.10         # 尺度抖动幅度（±10%）
CROP_PX = 256               # Stage2 crop 输入边长（训练与推理必须一致）


def geom_features(box, patch_px: int = PATCH_PX):
    """框几何特征（5 维，全部推理期可得）。

    box = (cls, x0, y0, x1, y1) 像素坐标（patch 内）。
    返回 [w/patch, h/patch, 长宽比(<=MAX_ASPECT), log10(面积+1)/6, 相对面积]。
    """
    _cls, x0, y0, x1, y1 = box
    w = max(float(x1) - float(x0), 1e-6)
    h = max(float(y1) - float(y0), 1e-6)
    aspect = min(max(w, h) / max(min(w, h), 1e-6), MAX_ASPECT)
    return [
        w / patch_px,
        h / patch_px,
        aspect,
        math.log10(w * h + 1.0) / 6.0,
        (w * h) / (patch_px * patch_px),
    ]


def crop_with_margin(arr: np.ndarray, box, margin: float = 0.25,
                     out_size: int = CROP_PX) -> np.ndarray:
    """在 patch 灰度图上按框裁一个带边距的正方形，越界补 0，返回 (out,out,3) uint8。

    框在 crop 中恒居中，边长 = max(w,h)*(1+2*margin)；这保证训练与推理的裁切规则一致。
    """
    img = np.asarray(arr)
    height, width = img.shape[:2]
    _cls, x0, y0, x1, y1 = box
    w = max(float(x1) - float(x0), 1.0)
    h = max(float(y1) - float(y0), 1.0)
    cx, cy = (float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0
    side = max(max(w, h) * (1.0 + 2.0 * margin), 16.0)
    half = side / 2.0
    xa, ya = int(math.floor(cx - half)), int(math.floor(cy - half))
    xb, yb = xa + int(round(side)), ya + int(round(side))

    canvas = np.zeros((yb - ya, xb - xa), dtype=img.dtype)
    sx0, sy0 = max(0, xa), max(0, ya)
    sx1, sy1 = min(width, xb), min(height, yb)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - ya:sy1 - ya, sx0 - xa:sx1 - xa] = img[sy0:sy1, sx0:sx1]

    out = Image.fromarray(canvas).resize((out_size, out_size), Image.BILINEAR)
    rgb = np.asarray(out, dtype=np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    return rgb


def jitter_box(box, rng: np.random.RandomState):
    """框抖动：中心平移 ±JITTER_SHIFT*长边，尺度 ±JITTER_SCALE。用于缓解真值/预测框域差。"""
    cls, x0, y0, x1, y1 = box
    w, h = float(x1) - float(x0), float(y1) - float(y0)
    cx, cy = (float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0
    side = max(w, h)
    cx += rng.uniform(-JITTER_SHIFT, JITTER_SHIFT) * side
    cy += rng.uniform(-JITTER_SHIFT, JITTER_SHIFT) * side
    w *= 1.0 + rng.uniform(-JITTER_SCALE, JITTER_SCALE)
    h *= 1.0 + rng.uniform(-JITTER_SCALE, JITTER_SCALE)
    w, h = max(w, 4.0), max(h, 4.0)
    return (cls, cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def match_conf(gt_box, pred_boxes: np.ndarray, pred_scores: np.ndarray,
               pred_cls: np.ndarray, match_iou: float = 0.5) -> float:
    """GT 框对应 Stage1 置信度：取与之 IoU>=match_iou 的最高分预测的分数，无则 0.0。

    类无关匹配（只看"这里有没有框"），因为 Stage1 的类别本身不可信。
    """
    if pred_boxes is None or len(pred_boxes) == 0:
        return 0.0
    _cls, x0, y0, x1, y1 = gt_box
    gx0, gy0, gx1, gy1 = float(x0), float(y0), float(x1), float(y1)
    ix0 = np.maximum(pred_boxes[:, 0], gx0)
    iy0 = np.maximum(pred_boxes[:, 1], gy0)
    ix1 = np.minimum(pred_boxes[:, 2], gx1)
    iy1 = np.minimum(pred_boxes[:, 3], gy1)
    iw = np.clip(ix1 - ix0, 0.0, None)
    ih = np.clip(iy1 - iy0, 0.0, None)
    inter = iw * ih
    area_g = max(gx1 - gx0, 0.0) * max(gy1 - gy0, 0.0)
    area_p = np.clip(pred_boxes[:, 2] - pred_boxes[:, 0], 0.0, None) * \
        np.clip(pred_boxes[:, 3] - pred_boxes[:, 1], 0.0, None)
    union = area_g + area_p - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    keep = np.where(iou >= match_iou)[0]
    if keep.size == 0:
        return 0.0
    return float(pred_scores[keep].max())


class FrozenBackboneFeaturizer:
    """冻结 YOLOv8 骨干，取多层特征做 GAP 后拼接（默认 15/18/21 -> 448 维）。

    注意（实测踩坑）：
    - 必须调 `YOLO(w).model`（DetectionModel）整体前向，**不能**直接调 `model.model.model`
      （裸 Sequential 会破坏 ultralytics 的 list-of-tensor 契约，Concat 直接 TypeError）；
    - 必须 `eval()` + `no_grad()`；本类只用于提特征，不参与任何梯度；
    - 注册 hook 后每次前向会重新填充 `self._taps`，用完即清，避免张量滞留显存。
    """

    def __init__(self, weights=WEIGHTS, layers=BACKBONE_LAYERS, device: str = "0"):
        import torch  # 延迟导入：纯函数单测不需要付 torch 导入开销
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import select_device

        self._torch = torch
        # 设备串沿用项目惯例（ultralytics 风格："0" / "cuda:0" / "cpu"，见 eval_recall_report.py）。
        # 注意：不能直接 model.to("0")，torch 会报 Invalid device string: '0'，
        # 必须先经 select_device 归一成 torch.device。CUDA 不可用时退回 CPU。
        if torch.cuda.is_available() or device == "cpu":
            self.device = select_device(device)
        else:
            self.device = torch.device("cpu")
        model = YOLO(str(weights)).model
        model.eval()
        model.to(self.device)
        self.model = model
        self.layers = tuple(layers)
        self._taps = {}
        for idx in self.layers:
            model.model[idx].register_forward_hook(self._make_hook(idx))

    def _make_hook(self, idx: int):
        def hook(_module, _inputs, output):
            self._taps[idx] = output
        return hook

    def features(self, batch: np.ndarray) -> np.ndarray:
        """batch: (B,H,W,3) uint8 -> (B,448) float32。"""
        torch = self._torch
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(batch))
            x = x.permute(0, 3, 1, 2).float().div_(255.0).to(self.device)
            self.model(x)
            feats = []
            for idx in self.layers:
                t = self._taps.get(idx)
                if t is None:
                    raise RuntimeError(f"层 {idx} 未取到特征，检查 BACKBONE_LAYERS 是否与模型匹配")
                if t.dim() == 4:
                    t = t.mean(dim=(2, 3))
                feats.append(t)
            out = torch.cat(feats, dim=1)
            self._taps.clear()
        result = out.detach().cpu().numpy().astype(np.float32)
        if result.shape[1] != BACKBONE_DIM:
            raise RuntimeError(f"骨干维度 {result.shape[1]} != 预期 {BACKBONE_DIM}，BACKBONE_LAYERS 需同步更新")
        return result


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--data", default=str(DATA_YAML))
    ap.add_argument("--feature-mode", choices=FEATURE_MODES, default="ctx",
                    help="ctx=影像特征，terrain=地形特征，ctx_terrain=两者融合")
    ap.add_argument("--terrain-dem", default=str(TERRAIN_DEM))
    ap.add_argument("--terrain-padding", type=int, default=2,
                    help="地形窗口外扩的 MOLA 像元数")
    ap.add_argument("--splits", default="train,val", help="构建哪些 split（test 留作最终留出）")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.001, help="Stage1 预测置信度（统一协议值）")
    ap.add_argument("--device", default="0")
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--jitter", type=int, default=4, help="train 每框额外抖动变体数（0=关闭）")
    ap.add_argument("--batch", type=int, default=32, help="提特征批大小")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="每 split 只用前 N 张图（0=全部，冒烟用）")
    ap.add_argument("--out", default=str(OUT))
    return ap.parse_args()


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_recall_report import load_ground_truth, load_images, resolve_dataset_root

    args = parse_args()
    rng = np.random.RandomState(args.seed)
    root = resolve_dataset_root(Path(args.data))
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    recognizer = None  # 延迟到真正要提特征时再加载，冒烟/纯统计路径更轻
    need_ctx = args.feature_mode in ("ctx", "ctx_terrain")
    terrain_extractor = None
    if args.feature_mode in ("terrain", "ctx_terrain"):
        terrain_extractor = TerrainExtractor(Path(args.terrain_dem), padding_px=args.terrain_padding)

    crops, metas = [], []
    for split in splits:
        images = load_images(root, split)
        if args.limit:
            images = images[:args.limit]
        gt = load_ground_truth(root, split, images, patch_px=PATCH_PX)

        from eval_recall_report import infer
        preds = infer(Path(args.weights), images, args.imgsz, args.conf, args.device, False)

        for image in images:
            rows = gt.get(image.stem, [])
            if not rows:
                continue
            arr = np.asarray(Image.open(image).convert("L"))
            boxes, scores, classes = preds.get(
                image.stem,
                (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)),
            )
            for box_i, (cls, x0, y0, x1, y1) in enumerate(rows):
                group = len(metas)
                variants = [(cls, x0, y0, x1, y1)]
                if split == "train" and args.jitter > 0:
                    variants += [jitter_box((cls, x0, y0, x1, y1), rng)
                                 for _ in range(args.jitter)]
                for v in variants:
                    conf = match_conf(v, boxes, scores, classes, 0.5)
                    if need_ctx:
                        crops.append(crop_with_margin(arr, v, args.margin))
                    terrain_vector = np.zeros(len(terrain_features.TERRAIN_FEATURE_NAMES), dtype=np.float32)
                    if terrain_extractor is not None:
                        _cls, vx0, vy0, vx1, vy1 = v
                        terrain_box = (
                            _cls,
                            ((vx0 + vx1) / 2.0) / PATCH_PX,
                            ((vy0 + vy1) / 2.0) / PATCH_PX,
                            (vx1 - vx0) / PATCH_PX,
                            (vy1 - vy0) / PATCH_PX,
                        )
                        terrain_vector = terrain_extractor.for_patch(image.stem, terrain_box, patch_px=PATCH_PX)
                    metas.append({
                        "split": split, "cls": int(cls), "stem": image.stem,
                        "group": group, "conf": conf,
                        "geom": geom_features(v),
                        "terrain": terrain_vector,
                    })
        print(f"[{split}] 图 {len(images)} 张，累计样本 {len(metas)}")

    if not metas:
        print("未采集到任何样本，检查 --data/--splits/--limit")
        return

    if need_ctx:
        recognizer = FrozenBackboneFeaturizer(args.weights, device=args.device)
        feats = []
        for i in range(0, len(crops), args.batch):
            chunk = np.stack(crops[i:i + args.batch])
            feats.append(recognizer.features(chunk))
        backbone = np.concatenate(feats, axis=0)
    else:
        backbone = np.zeros((len(metas), 0), dtype=np.float32)

    geom = np.asarray([m["geom"] for m in metas], dtype=np.float32)
    conf = np.asarray([[m["conf"]] for m in metas], dtype=np.float32)
    terrain_values = np.asarray([m["terrain"] for m in metas], dtype=np.float32)
    X = compose_features(backbone, geom, conf, terrain_values, args.feature_mode)
    y = np.asarray([m["cls"] for m in metas], dtype=np.int64)
    split_arr = np.asarray([m["split"] for m in metas])
    group_arr = np.asarray([m["group"] for m in metas], dtype=np.int64)
    stem_arr = np.asarray([m["stem"] for m in metas])

    expected_features = len(feature_names(args.feature_mode, backbone_dim=backbone.shape[1]))
    assert X.shape[1] == expected_features, (X.shape, expected_features)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, X=X, y=y, split=split_arr, group=group_arr, stem=stem_arr, conf=conf,
        terrain=terrain_values, feature_mode=np.asarray(args.feature_mode),
        feature_names=np.asarray(feature_names(args.feature_mode, backbone_dim=backbone.shape[1])),
        terrain_feature_names=np.asarray(terrain_features.TERRAIN_FEATURE_NAMES),
        terrain_dem=np.asarray(str(Path(args.terrain_dem).resolve()) if terrain_extractor else "",
                               dtype="U"),
    )
    print(f"样本 {X.shape[0]}，特征 {X.shape[1]} -> {out}")

    from collections import Counter
    for split in splits:
        mask = split_arr == split
        cnt = Counter(CLASS_NAMES[c] for c in y[mask])
        print(f"  {split}: {int(mask.sum())} 样本，按类 {dict(cnt)}，"
              f"独立框组 {len(set(group_arr[mask].tolist()))}")
    print(f"  conf 特征: 均值 {conf.mean():.3f}，=0 的比例 {float((conf == 0).mean()):.3f}")


if __name__ == "__main__":
    main()
