"""CPU prototypes for proposed experiments, not registered training models.

Selective context is inspired by LSKNet (https://github.com/zcablii/LSKNet),
but uses a reduced residual branch and softmax selection. It is not LSKNet.
LoRA follows https://arxiv.org/abs/2106.09685. Neither module is evidence of
improved Mars performance. Existing experiment code is intentionally untouched.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class SelectiveResidualContext(nn.Module):
    """Identity-initialized local/wide context; dilation is the controlled factor."""

    def __init__(self, channels: int, dilation: int = 3):
        super().__init__()
        if channels < 4 or dilation not in (1, 3):
            raise ValueError('channels>=4 and dilation in {1,3} required')
        hidden = channels // 4
        self.reduce = nn.Conv2d(channels, hidden, 1, bias=False)
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.context = nn.Conv2d(hidden, hidden, 7, padding=3*dilation,
                                 dilation=dilation, groups=hidden)
        self.selector = nn.Conv2d(4, 2, 7, padding=3)
        self.project = nn.Conv2d(hidden, channels, 1, bias=False)
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        a = self.local(self.reduce(x))
        b = self.context(a)
        descriptor = torch.cat((a.mean(1, keepdim=True), a.amax(1, keepdim=True),
                                b.mean(1, keepdim=True), b.amax(1, keepdim=True)), 1)
        weight = self.selector(descriptor).softmax(1)
        mixed = weight[:, :1] * a + weight[:, 1:] * b
        return x + self.scale.tanh() * self.project(mixed)


def attach_p5_context(model: nn.Module, dilation: int):
    """Preserve YOLOv8 routing indices; caller must validate its own trainer build.

    A default YOLO trainer rebuilds from YAML, which would discard this wrapper.
    Do not train this model until a custom trainer/reload audit is in place.
    """
    original = model.model[9]
    if type(original).__name__ != 'SPPF':
        raise ValueError('Expected unmodified YOLOv8 P5 SPPF at layer9')
    channels = original.cv2.conv.out_channels
    context = SelectiveResidualContext(channels, dilation)
    wrapped = nn.Sequential(original, context)
    for attribute in ('i', 'f', 'type'):
        setattr(wrapped, attribute, getattr(original, attribute))
    wrapped.np = sum(p.numel() for p in wrapped.parameters())
    wrapped.train(model.training)
    model.model[9] = wrapped
    return context


class LoRALinear(nn.Module):
    """Frozen linear map plus a zero-initialized rank-r update, no bias update."""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 4):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank <= 0 or alpha <= 0:
            raise ValueError('Linear base, positive rank/alpha required')
        self.base = base.requires_grad_(False)
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.multiplier = float(alpha) / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.multiplier


def attach_dinov2_lora(model: nn.Module, last_blocks: int = 2, rank: int = 4):
    """Adapt only query/value maps of the final HF DINOv2 blocks; fail on drift."""
    layers = model.encoder.layer
    if not 0 < last_blocks <= len(layers):
        raise ValueError('Invalid block count')
    targets = []
    for index in range(len(layers)-last_blocks, len(layers)):
        attention = layers[index].attention.attention
        for name in ('query', 'value'):
            base = getattr(attention, name)
            if type(base) is not nn.Linear:
                raise ValueError('DINOv2 query/value is not an unmodified Linear')
            targets.append((index, attention, name, base))
    model.requires_grad_(False)
    names = []
    for index, attention, name, base in targets:
        setattr(attention, name, LoRALinear(base, rank=rank, alpha=rank))
        names.append(f'encoder.layer.{index}.attention.attention.{name}')
    return names
