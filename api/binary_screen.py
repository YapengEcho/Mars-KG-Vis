"""Offline screening with the ConvNeXt/RBF fusion frozen on 2026-11-19."""

from copy import deepcopy
import json
import math
from pathlib import Path

from PIL import Image

from api.classifier import sha256


def _asset(manifest: Path, entry: dict) -> Path:
    path = (manifest.parent / entry["path"]).resolve()
    if not path.is_file() or sha256(path) != entry["sha256"]:
        raise ValueError(f"Binary screening artifact verification failed: {path.name}")
    return path


def _brightness_lut(raw, np):
    # Same nonzero 2nd/98th percentile interpolation as the archived CTX renderer.
    counts = np.bincount(raw.ravel(), minlength=256)
    counts[0] = 0
    cumulative = counts.cumsum()
    total = cumulative[-1]
    if total:
        positions = (total - 1) * np.array([0.02, 0.98])
        lower = np.floor(positions).astype(int)
        upper = np.ceil(positions).astype(int)
        a = np.searchsorted(cumulative, lower, side="right")
        b = np.searchsorted(cumulative, upper, side="right")
        lo, hi = a + (b - a) * (positions - lower)
    else:
        lo, hi = 0, 255
    if hi <= lo:
        hi = lo + 1
    lut = np.clip((np.arange(256, dtype=np.float64) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    lut[0] = 0
    return lut


def _views(image: Image.Image, np):
    raw = np.asarray(image.convert("L"))
    source = Image.fromarray(_brightness_lut(raw, np)[raw])
    width, height = source.size
    ratio = 224 / max(width, height)
    size = max(1, round(width * ratio)), max(1, round(height * ratio))
    for inset in (0, 0.125):
        box = (width * inset, height * inset, width * (1 - inset), height * (1 - inset))
        view = source.resize(size, Image.Resampling.BILINEAR, box=box)
        if size != (224, 224):
            square = Image.new("L", (224, 224))
            square.paste(view, ((224 - size[0]) // 2, (224 - size[1]) // 2))
            view = square
        yield view


def _load_tail(block, path, torch, device):
    nn = torch.nn

    class LowRankLinear(nn.Module):
        def __init__(self, layer, down, up):
            super().__init__()
            self.base = layer
            self.register_buffer("down", down)
            self.register_buffer("up", up)

        def forward(self, x):
            return self.base(x) + nn.functional.linear(nn.functional.linear(x, self.down), self.up)

    state = torch.load(path, map_location="cpu", weights_only=True)
    tail = deepcopy(block)
    for index in (3, 5):
        base = tail.block[index]
        down = state[f"tail.block.{index}.down.weight"]
        up = state[f"tail.block.{index}.up.weight"]
        if down.shape != (8, base.in_features) or up.shape != (base.out_features, 8):
            raise ValueError("Invalid binary screening adapter shapes")
        tail.block[index] = LowRankLinear(base, down, up)
    return tail.eval().requires_grad_(False).to(device=device, memory_format=torch.channels_last)


class BinaryScreen:
    """Screen the whole uploaded crop with the frozen CTX operating point.

    The continuous decision score is a signed model margin, not a probability.
    The training comparison used a strict cutoff. The public threshold is the
    next representable float to express the same comparison as score >= threshold.
    """

    def __init__(self, manifest_path, device="cpu"):
        import numpy as np
        import torch
        from torchvision.models import convnext_tiny

        path = Path(manifest_path).resolve()
        spec = json.loads(path.read_text(encoding="utf-8"))
        cutoff = float(spec["threshold"])
        if spec["schemaVersion"] != 1 or not math.isfinite(cutoff):
            raise ValueError("Invalid binary screening manifest")
        self.version = spec["modelVersion"]
        self.threshold = math.nextafter(cutoff, math.inf)
        self._means = np.asarray(spec["mean"], dtype=np.float64)
        self._scales = np.asarray(spec["std"], dtype=np.float64)
        self._weights = np.asarray(spec["weights"], dtype=np.float64)
        parameters = (self._means, self._scales, self._weights)
        if (
            not isinstance(self.version, str)
            or not self.version
            or any(a.shape != (3,) or not np.isfinite(a).all() for a in parameters)
            or np.any(self._scales <= 0)
            or np.any(self._weights < 0)
            or not math.isclose(float(self._weights.sum()), 1.0)
            or len(spec["adapters"]) != 2
            or len(spec["svms"]) != 3
        ):
            raise ValueError("Invalid binary screening fusion parameters")

        self._np = np
        self._torch = torch
        self._device = torch.device(device)
        checkpoint = _asset(path, spec["checkpoint"])
        adapters = [_asset(path, item) for item in spec["adapters"]]
        svms = [_asset(path, item) for item in spec["svms"]]
        net = convnext_tiny(weights=None)
        net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self._tails = [_load_tail(net.features[7][-1], p, torch, self._device) for p in adapters]
        self._net = net.eval().requires_grad_(False).to(
            device=self._device, dtype=torch.float32, memory_format=torch.channels_last,
        )
        self._prefix = torch.nn.Sequential(self._net.features[:7], self._net.features[7][:-1])
        self._mean = torch.tensor([.485, .456, .406], device=self._device)[None, :, None, None]
        self._std = torch.tensor([.229, .224, .225], device=self._device)[None, :, None, None]
        self._svms = []
        for svm in svms:
            with np.load(svm, allow_pickle=False) as stored:
                scales, vectors, coefficients, intercept, gamma = (
                    stored[key] for key in ("scales", "vectors", "coefficients", "intercept", "gamma")
                )
            if (
                scales.shape != (2,)
                or vectors.ndim != 2
                or vectors.shape[1] != 1536
                or coefficients.shape != (len(vectors),)
                or intercept.ndim != 0
                or gamma.ndim != 0
                or gamma <= 0
                or any(not np.isfinite(a).all() for a in (scales, vectors, coefficients, intercept, gamma))
            ):
                raise ValueError("Invalid binary screening SVM parameters")
            self._svms.append((scales, vectors, coefficients, float(intercept), float(gamma)))

    def _encode(self, images):
        tensor = self._torch.as_tensor(
            self._np.stack(images), device=self._device, dtype=self._torch.float32,
        )[:, None] / 255.0
        return ((tensor - self._mean) / self._std).contiguous(memory_format=self._torch.channels_last)

    def _embedding(self, features):
        return self._net.classifier[0](self._net.avgpool(features)).flatten(1).cpu().numpy()

    def _score_views(self, crops):
        scores = []
        with self._torch.inference_mode():
            maps = self._prefix(self._encode(crops))
            features = [self._embedding(tail(maps)) for tail in self._tails]
            larger = [v.resize((320, 320), Image.Resampling.BILINEAR) for v in crops]
            features.append(self._embedding(self._net.features(self._encode(larger))))
        for embedding, (scale, vectors, dual, bias, gamma) in zip(features, self._svms, strict=True):
            scaled = (embedding.astype(self._np.float64) * scale[:, None]).reshape(1536)
            delta = vectors - scaled
            squared = self._np.einsum("ij,ij->i", delta, delta)
            scores.append(self._np.exp(-gamma * squared) @ dual + bias)
        return float(((self._np.asarray(scores) - self._means) / self._scales) @ self._weights)

    def predict(self, image: Image.Image, roi=None) -> float:
        # ROI is solely the four-class pooling region. The binary screen needs the original context.
        return self._score_views(tuple(_views(image, self._np)))
