"""Offline inference for the frozen DINOv2-base + rank-4 LoRA service reference."""
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageOps

CLASSES = ('SL', 'SP', 'FL', 'CL')
PREPROCESSING_VERSION = 'gray-letterbox224-roiarea-v1'
REVIEW_POLICY = 'mandatory-review-v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class PreparedImage:
    pixels: bytes
    weights: list[float]
    metadata: dict
    preview: bytes


def prepare_image(image: Image.Image, roi=None) -> PreparedImage:
    """Preserve aspect ratio; pool exact overlap areas of 14x14 patch cells."""
    width, height = image.size
    source = 'provided_box' if roi is not None else 'whole_image_crop'
    roi = [0., 0., float(width), float(height)] if roi is None else roi
    if (not isinstance(roi, (list, tuple)) or len(roi) != 4
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in roi)):
        raise ValueError('ROI must contain four finite pixel coordinates.')
    x0, y0, x1, y1 = map(float, roi)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('ROI must be inside the EXIF-oriented image.')
    factor = 224 / max(width, height)
    rw, rh = max(1, round(width * factor)), max(1, round(height * factor))
    left, top = (224 - rw) // 2, (224 - rh) // 2
    gray = ImageOps.grayscale(image)
    if gray.size != (rw, rh):
        gray = gray.resize((rw, rh), Image.Resampling.BILINEAR)
    square = Image.new('L', (224, 224), 0)
    square.paste(gray, (left, top))
    box = [left + x0 * rw / width, top + y0 * rh / height,
           left + x1 * rw / width, top + y1 * rh / height]
    areas = [max(0., min((x+1)*14, box[2])-max(x*14, box[0])) *
             max(0., min((y+1)*14, box[3])-max(y*14, box[1]))
             for y in range(16) for x in range(16)]
    total = sum(areas)
    if total <= 0:
        raise ValueError('ROI has no usable area.')
    preview = image.copy()
    preview.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    preview.info.clear()  # The record preview does not retain EXIF metadata.
    output = BytesIO()
    preview.save(output, format='JPEG', quality=90)
    metadata = dict(width=width, height=height, roi=[x0,y0,x1,y1], roiSource=source,
                    modelSize=[224,224], resizedSize=[rw,rh], padding=[left,top],
                    modelRoi=box, preprocessingVersion=PREPROCESSING_VERSION)
    return PreparedImage(square.tobytes(), [a/total for a in areas], metadata, output.getvalue())


def load_manifest(path):
    path = Path(path).resolve()
    spec = json.loads(path.read_text(encoding='utf-8'))
    if (spec.get('schemaVersion') != 1 or spec.get('architecture') != 'dinov2-base-qv-last2-r4'
            or spec.get('classes') != list(CLASSES)
            or spec.get('preprocessingVersion') != PREPROCESSING_VERSION
            or not isinstance(spec.get('modelVersion'), str) or not spec['modelVersion'].strip()):
        raise ValueError('Classifier manifest schema, classes or preprocessing is incompatible.')
    resolved = {}
    for key in ('baseConfig', 'baseWeights', 'adapter', 'normalizer'):
        item = spec.get('files', {}).get(key, {})
        asset = Path(item.get('path', ''))
        asset = (asset if asset.is_absolute() else path.parent / asset).resolve()
        digest = item.get('sha256')
        if not isinstance(digest, str) or len(digest) != 64 or not asset.is_file() or sha256(asset) != digest:
            raise ValueError(f'Classifier asset verification failed: {key}')
        resolved[key] = asset
    if (resolved['baseConfig'].name != 'config.json' or resolved['baseWeights'].name != 'model.safetensors'
            or resolved['baseConfig'].parent != resolved['baseWeights'].parent):
        raise ValueError('Base config and weights must be in the same local model directory.')
    return spec, resolved


class DinoClassifier:
    """Load verified local assets only; no research-script imports or downloads."""
    def __init__(self, manifest_path, device='cpu', threads=2):
        spec, files = load_manifest(manifest_path)
        import numpy as np
        import torch
        from torch import nn
        from torch.nn import functional as F
        from transformers import AutoModel

        if device not in ('cpu', 'cuda') or not 1 <= threads <= 16:
            raise ValueError('Use cpu/cuda and 1..16 inference threads.')
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('The configured CUDA device is unavailable.')
        torch.set_num_threads(threads)
        self.torch, self.np, self.device = torch, np, device
        self.version = spec['modelVersion']
        self.asset_digest = sha256(manifest_path)
        self.encoder = AutoModel.from_pretrained(str(files['baseConfig'].parent),
            local_files_only=True, use_safetensors=True, trust_remote_code=False).float().eval()
        if self.encoder.config.hidden_size != 768 or len(self.encoder.encoder.layer) != 12:
            raise ValueError('Expected the 12-block DINOv2-base encoder.')

        class LoRALinear(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
                self.A = nn.Parameter(torch.zeros(4, base.in_features))
                self.B = nn.Parameter(torch.zeros(base.out_features, 4))

            def forward(self, x):
                return self.base(x) + F.linear(F.linear(x, self.A), self.B)

        for layer in self.encoder.encoder.layer[-2:]:
            attention = layer.attention.attention
            for name in ('query', 'value'):
                base = getattr(attention, name)
                if type(base) is not nn.Linear:
                    raise ValueError('Unexpected DINOv2 attention layout.')
                setattr(attention, name, LoRALinear(base))
        self.head = nn.Linear(768, 4)
        checkpoint = torch.load(files['adapter'], map_location='cpu', weights_only=True)
        parameters = checkpoint.get('parameters', {})
        targets = {f'encoder.{k}': v for k,v in self.encoder.named_parameters() if k.endswith(('.A','.B'))}
        targets.update({f'head.{k}': v for k,v in self.head.named_parameters()})
        if set(parameters) != set(targets):
            raise ValueError('Adapter parameter names do not match the service architecture.')
        with torch.no_grad():
            for name, destination in targets.items():
                value = parameters[name]
                if value.shape != destination.shape or not torch.isfinite(value).all():
                    raise ValueError('Invalid adapter parameter shape or values.')
                destination.copy_(value)
        with np.load(files['normalizer'], allow_pickle=False) as normalizer:
            if normalizer['class_names'].tolist() != list(CLASSES):
                raise ValueError('Normalizer class order differs from the classifier.')
            mean, scale = normalizer['mean'].copy(), normalizer['scale'].copy()
        if (mean.shape != (768,) or scale.shape != (768,) or not np.isfinite(mean).all()
                or not np.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError('Invalid feature normalization.')
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.scale = torch.tensor(scale, dtype=torch.float32, device=device)
        self.encoder.requires_grad_(False).to(device).eval()
        self.head.requires_grad_(False).to(device).eval()

    def predict(self, prepared: PreparedImage):
        torch, np = self.torch, self.np
        gray = np.frombuffer(prepared.pixels, dtype=np.uint8).reshape(224,224).copy()
        rgb = np.repeat(gray[...,None], 3, axis=-1)
        pixels = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float() / 255
        pixels = ((pixels-torch.tensor([.485,.456,.406]).view(1,3,1,1)) /
                  torch.tensor([.229,.224,.225]).view(1,3,1,1)).to(self.device)
        weights = torch.tensor([prepared.weights], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            tokens = self.encoder(pixel_values=pixels).last_hidden_state
            if tokens.shape != (1,257,768):
                raise ValueError('Unexpected patch token layout.')
            features = torch.einsum('bp,bpd->bd', weights, tokens[:,1:])
            logits = self.head((features-self.mean)/self.scale)
            values = torch.softmax(logits, dim=1)[0].cpu().tolist()
        return dict(zip(CLASSES, values))


def classification_result(scores, model):
    if set(scores) != set(CLASSES):
        raise ValueError('Classifier returned an unexpected class set.')
    values = [float(scores[k]) for k in CLASSES]
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values) or abs(sum(values)-1) > 1e-5:
        raise ValueError('Classifier returned invalid scores.')
    winner = max(range(4), key=lambda i: values[i])
    return dict(prediction=CLASSES[winner], probabilities=dict(zip(CLASSES, values)),
        confidence=values[winner], uncertain=True,
        reviewReasons=['MODEL_REQUIRES_HUMAN_REVIEW'], reviewPolicy=REVIEW_POLICY,
        scoresCalibrated=False, modelVersion=model.version, modelDigest=model.asset_digest,
        preprocessingVersion=PREPROCESSING_VERSION)
