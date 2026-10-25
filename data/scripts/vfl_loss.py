# -*- coding: utf-8 -*-
"""Runtime VFL adapter for the Ultralytics detection criterion."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class VarifocalClassificationLoss(nn.Module):
    """Elementwise Varifocal loss compatible with the existing YOLO loss call.

    The current criterion calls ``self.bce(pred_scores, target_scores)`` and
    applies optional class weights before reducing.  Returning the unreduced
    map here keeps that reduction and the existing ``cls_pw`` behavior intact.
    ``target_scores > 0`` is the binary positive label because the
    TaskAlignedAssigner writes zero scores for background anchors and a
    positive IoU-quality score for assigned foreground anchors.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor) -> torch.Tensor:
        gt_score = gt_score.to(dtype=pred_score.dtype)
        label = gt_score.gt(0).to(dtype=pred_score.dtype)
        pred_float = pred_score.float()
        target_float = gt_score.float()
        weight = self.alpha * pred_float.sigmoid().pow(self.gamma) * (1.0 - label.float())
        weight = weight + target_float * label.float()
        return F.binary_cross_entropy_with_logits(
            pred_float, target_float, reduction="none"
        ) * weight


def install_varifocal_loss(
    ultralytics_loss_module: Any, *, alpha: float = 0.75, gamma: float = 2.0
) -> Callable[[], None]:
    """Install VFL on the imported Ultralytics detection criterion.

    This patches only the criterion constructor for the current process.  It
    does not edit the installed package and returns a restore callback for
    tests or callers that need to undo the patch.
    """

    criterion_cls = getattr(ultralytics_loss_module, "v8DetectionLoss")
    existing = getattr(criterion_cls, "_mars_vfl_patch", None)
    if existing is not None:
        return existing["restore"]

    original_init = criterion_cls.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.bce = VarifocalClassificationLoss(alpha=alpha, gamma=gamma)
        self._mars_cls_loss = "vfl"

    def restore() -> None:
        current = getattr(criterion_cls, "_mars_vfl_patch", None)
        if current is not None and current["patched_init"] is patched_init:
            criterion_cls.__init__ = current["original_init"]
            delattr(criterion_cls, "_mars_vfl_patch")

    criterion_cls.__init__ = patched_init
    criterion_cls._mars_vfl_patch = {
        "original_init": original_init,
        "patched_init": patched_init,
        "restore": restore,
    }
    return restore
