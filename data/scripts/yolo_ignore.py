"""Detection-level ignored-box support for the official Ultralytics loader.

The experiment uses the ordinary five-column YOLO format.  A class value of
``-1`` is reserved for an ignored box.  The value is carried through the
existing geometric augmentations as part of ``cls`` and is split out only at
the final formatter.  Ignored boxes therefore contribute neither a positive
target nor a negative background target.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_CLASS = -1.0


def mark_ignored_rows(rows: Iterable[str], row_indexes: set[int]) -> list[str]:
    """Mark 1-based YOLO rows as ignored while keeping five columns."""
    marked = []
    for row_index, raw in enumerate(rows, start=1):
        fields = raw.strip().split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"YOLO detection row must have 5 columns: {raw!r}")
        values = [float(value) for value in fields]
        ignored = row_index in row_indexes
        class_text = "-1" if ignored else fields[0]
        marked.append(" ".join([class_text] + [f"{value:.6f}" for value in values[1:]]))
    return marked


def split_ignore_targets(cls: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split normal and ignored targets without changing box coordinates."""
    if cls.ndim == 1:
        cls = cls.view(-1, 1)
    valid = cls[:, 0] >= 0
    return cls[valid], boxes[valid], boxes[~valid]


def apply_ignore_bce_mask(
    pred_logits: torch.Tensor, target_scores: torch.Tensor, ignore_anchor: torch.Tensor | None
) -> torch.Tensor:
    """Return elementwise BCE with ignored-region background terms zeroed.

    Positive anchors remain trainable even when their center lies inside an
    ignored box.  This prevents a low-quality secondary annotation from
    suppressing a real primary target.
    """
    loss = F.binary_cross_entropy_with_logits(pred_logits.float(), target_scores.float(), reduction="none")
    if ignore_anchor is None:
        return loss
    if ignore_anchor.ndim == 2:
        ignore_anchor = ignore_anchor.unsqueeze(-1)
    background = target_scores.sum(dim=-1, keepdim=True).le(0)
    keep = ~(ignore_anchor.to(torch.bool) & background)
    return loss * keep.to(loss.dtype)


def load_ignore_index(manifest: str | Path) -> dict[tuple[str, str], dict[int, dict]]:
    """Load a sidecar keyed by split, patch stem and 1-based row index."""
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    index: dict[tuple[str, str], dict[int, dict]] = {}
    for entry in data.get("entries", []):
        if not entry.get("ignore", True):
            continue
        key = (str(entry["split"]), str(entry["patch"]))
        row_index = int(entry["row_index"])
        if row_index in index.setdefault(key, {}):
            raise ValueError(f"duplicate ignore sidecar key: {key} row {row_index}")
        index[key][row_index] = entry
    return index


def _split_from_image_path(path: str | Path) -> str | None:
    parts = Path(path).parts
    for split in ("train", "val", "test"):
        if split in parts:
            return split
    return None


def _mark_dataset_labels(
    labels: list[dict], ignore_index: dict[tuple[str, str], dict[int, dict]], *, allow_partial: bool = False
) -> list[dict]:
    """Mark sidecar rows in memory after stock label validation/cache loading."""
    matched_keys = set()
    for label in labels:
        split = _split_from_image_path(label["im_file"])
        key = (split, Path(label["im_file"]).stem) if split else None
        entries = ignore_index.get(key, {}) if key else {}
        if not entries:
            continue
        cls = np.asarray(label["cls"], dtype=np.float32).copy()
        boxes = np.asarray(label["bboxes"], dtype=np.float32)
        if any(row < 1 or row > len(cls) for row in entries):
            raise ValueError(f"ignore row out of range for {key}: {sorted(entries)} / {len(cls)}")
        for row, entry in entries.items():
            expected_class = int(entry.get("class_id", 0))
            expected_bbox = np.asarray(entry["bbox"], dtype=np.float32)
            if int(round(float(cls[row - 1, 0]))) != expected_class:
                raise ValueError(f"ignore class mismatch for {key} row {row}")
            if not np.allclose(boxes[row - 1], expected_bbox, atol=1e-6, rtol=0):
                raise ValueError(f"ignore bbox mismatch for {key} row {row}")
            cls[row - 1, 0] = IGNORE_CLASS
        label["cls"] = cls
        matched_keys.add(key)
    available_keys = set()
    for label in labels:
        split = _split_from_image_path(label["im_file"])
        if split:
            available_keys.add((split, Path(label["im_file"]).stem))
    loaded_splits = {split for split, _ in available_keys}
    relevant_index = {
        key: entries for key, entries in ignore_index.items() if key[0] in loaded_splits
    }
    missing_keys = set(relevant_index) - available_keys
    if missing_keys and not allow_partial:
        raise ValueError(f"ignore sidecar keys missing from dataset: {len(missing_keys)}")
    expected = sum(
        len(entries) for key, entries in relevant_index.items() if allow_partial and key not in missing_keys
    )
    if not allow_partial:
        expected = sum(len(entries) for entries in relevant_index.values())
    matched_unique = sum(len(relevant_index[key]) for key in matched_keys)
    if matched_unique != expected:
        raise ValueError(f"ignore sidecar matched {matched_unique} unique rows, expected {expected}")
    return labels


def _format_apply_instances(self, labels, params=None):
    """Formatter hook that separates ``-1`` boxes after all augmentations."""
    cls = np.asarray(params.get("cls", np.array([])), dtype=np.float32).reshape(-1, 1)
    instances = params.get("instances")
    if instances is None:
        raise AssertionError("instances are required for detection formatting")
    if not np.any(cls[:, 0] < 0):
        labels = _ORIG_FORMAT_APPLY_INSTANCES(self, labels, params)
        labels["ignore_bboxes"] = torch.zeros((0, 4), dtype=labels["bboxes"].dtype)
        labels["ignore_batch_idx"] = torch.zeros((0,), dtype=torch.long)
        return labels
    if getattr(self, "return_mask", False) or getattr(self, "return_keypoint", False) or getattr(self, "return_obb", False):
        raise ValueError("ignore sentinel is currently supported for detection training only")

    h, w = params.get("h", 0), params.get("w", 0)
    boxes = np.asarray(instances.bboxes, dtype=np.float32)
    valid = cls[:, 0] >= 0
    ignore = ~valid
    valid_boxes = torch.from_numpy(np.ascontiguousarray(boxes[valid]))
    ignore_boxes = torch.from_numpy(np.ascontiguousarray(boxes[ignore]))
    if getattr(self, "normalize", False):
        if w:
            valid_boxes[:, [0, 2]] /= w
            ignore_boxes[:, [0, 2]] /= w
        if h:
            valid_boxes[:, [1, 3]] /= h
            ignore_boxes[:, [1, 3]] /= h
    labels["cls"] = torch.from_numpy(np.ascontiguousarray(cls[valid]))
    labels["bboxes"] = valid_boxes
    labels["ignore_bboxes"] = ignore_boxes
    labels["ignore_batch_idx"] = torch.zeros((len(ignore_boxes),), dtype=torch.long)
    if getattr(self, "batch_idx", False):
        labels["batch_idx"] = torch.zeros((len(valid_boxes),))
    return labels


def _collate_with_ignore(batch: list[dict]) -> dict:
    """Use stock collation and add the two variable-length ignore tensors."""
    result = _ORIG_COLLATE_FN(batch)
    ignore_boxes = [item.get("ignore_bboxes", torch.zeros((0, 4))) for item in batch]
    ignore_indices = [item.get("ignore_batch_idx", torch.zeros((0,), dtype=torch.long)) for item in batch]
    result["ignore_bboxes"] = torch.cat(ignore_boxes, 0) if ignore_boxes else torch.zeros((0, 4))
    result["ignore_batch_idx"] = (
        torch.cat([idx.to(torch.long) + i for i, idx in enumerate(ignore_indices)], 0)
        if ignore_indices
        else torch.zeros((0,), dtype=torch.long)
    )
    return result


class _IgnoreAwareBCE(nn.Module):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def forward(self, pred_logits, target_scores):
        return apply_ignore_bce_mask(pred_logits, target_scores, getattr(self.owner, "_ignore_anchor_mask", None))


def _build_ignore_anchor_mask(owner, preds: dict, batch: dict) -> torch.Tensor | None:
    boxes = batch.get("ignore_bboxes")
    batch_idx = batch.get("ignore_batch_idx")
    if boxes is None or batch_idx is None or len(boxes) == 0:
        return None
    from ultralytics.utils.tal import make_anchors

    anchor_points, stride_tensor = make_anchors(preds["feats"], owner.stride, 0.5)
    feat_shape = preds["feats"][0].shape[2:]
    image_hw = torch.tensor(feat_shape, device=anchor_points.device, dtype=anchor_points.dtype) * owner.stride[0]
    centers = anchor_points * stride_tensor
    centers = centers / image_hw[[1, 0]]
    boxes = boxes.to(device=centers.device, dtype=centers.dtype)
    batch_idx = batch_idx.to(device=centers.device, dtype=torch.long)
    x1y1 = boxes[:, :2] - boxes[:, 2:] / 2
    x2y2 = boxes[:, :2] + boxes[:, 2:] / 2
    mask = torch.zeros((preds["scores"].shape[0], centers.shape[0]), dtype=torch.bool, device=centers.device)
    for image_index in batch_idx.unique().tolist():
        image_boxes = (batch_idx == image_index).nonzero(as_tuple=True)[0]
        if len(image_boxes):
            inside = (
                (centers[:, 0:1] >= x1y1[image_boxes, 0])
                & (centers[:, 0:1] <= x2y2[image_boxes, 0])
                & (centers[:, 1:2] >= x1y1[image_boxes, 1])
                & (centers[:, 1:2] <= x2y2[image_boxes, 1])
            ).any(dim=1)
            mask[int(image_index)] = inside
    return mask


def install_ignore_patch(ignore_manifest: str | Path | None = None) -> None:
    """Install the narrow runtime patch once, after importing Ultralytics."""
    global _ORIG_FORMAT_APPLY_INSTANCES, _ORIG_COLLATE_FN
    if globals().get("_PATCH_INSTALLED", False):
        return
    from ultralytics.data import augment as augment_module
    from ultralytics.data import dataset as dataset_module
    from ultralytics.utils import loss as loss_module

    _ORIG_FORMAT_APPLY_INSTANCES = augment_module.Format.apply_instances
    _ORIG_COLLATE_FN = dataset_module.YOLODataset.collate_fn
    augment_module.Format.apply_instances = _format_apply_instances
    dataset_module.YOLODataset.collate_fn = staticmethod(_collate_with_ignore)
    if ignore_manifest is not None:
        ignore_index = load_ignore_index(ignore_manifest)
        original_get_labels = dataset_module.YOLODataset.get_labels

        def patched_get_labels(self):
            labels = original_get_labels(self)
            return _mark_dataset_labels(labels, ignore_index)

        dataset_module.YOLODataset.get_labels = patched_get_labels

    original_init = loss_module.v8DetectionLoss.__init__
    original_get = loss_module.v8DetectionLoss.get_assigned_targets_and_loss

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.bce = _IgnoreAwareBCE(self)

    def patched_get(self, preds, batch):
        self._ignore_anchor_mask = _build_ignore_anchor_mask(self, preds, batch)
        try:
            return original_get(self, preds, batch)
        finally:
            self._ignore_anchor_mask = None

    loss_module.v8DetectionLoss.__init__ = patched_init
    loss_module.v8DetectionLoss.get_assigned_targets_and_loss = patched_get
    globals()["_PATCH_INSTALLED"] = True


_PATCH_INSTALLED = False
_ORIG_FORMAT_APPLY_INSTANCES = None
_ORIG_COLLATE_FN = None
